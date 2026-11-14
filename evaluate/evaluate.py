#!/usr/bin/env python3
"""Evaluate experiment checkpoints with OPRD sampling and one versioned public math scorer.

Use evaluate/run_eval.sh to choose an experiment, models, datasets, and parameters.
Real evaluation requires ready checkpoints and free training GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

LMOPS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LMOPS_ROOT))
sys.path.insert(0, str(LMOPS_ROOT / "gad" / "tools"))
from config import load_config, nested_config, add_config_arguments, select_config
from pipeline import stage_step_budget

EVALUATE_DIR = Path(__file__).resolve().parent
METHODS = ("base", "seqkd", "warmup", "gad")


def load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return data


def resolve_from(path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampler_question_ids(oprd_root, path, max_prompts):
    """Use the pinned sampler's actual ID assignment, including its subset rule."""
    import importlib.util
    import importlib
    name='_lmops_oprd_sft_'+hashlib.sha256(str(oprd_root).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,oprd_root/'sft/__init__.py',submodule_search_locations=[str(oprd_root/'sft')])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    samples=importlib.import_module(name+'.sample_teacher').load_source_samples(path)
    if max_prompts:samples=samples[:max_prompts]
    ids=[sample['global_index'] for sample in samples]
    if len(set(map(str,ids)))!=len(ids):raise ValueError('sampler has duplicate question IDs')
    return ids


def model_content(path, *, recursive=True):
    return {str(p.relative_to(path)):sha256(p) for p in sorted(path.rglob('*') if recursive else path.iterdir()) if p.is_file() and
            p.suffix in ('.json','.safetensors','.bin','.pt','.jinja','.model','.txt','.tiktoken','.bpe')}


def select(value: str | None, defaults: list[str]) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()] if value else list(defaults)


def named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, sep, raw_path = value.partition("=")
        if not sep or not name or not raw_path or name in result:
            raise ValueError(f"expected unique NAME=PATH, got: {value}")
        result[name] = Path(raw_path).expanduser().resolve()
    return result


def hf_ready(model: Path) -> bool:
    if not (model / "config.json").is_file():
        return False
    index = next((model / name for name in
                  ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                  if (model / name).is_file()), None)
    if index:
        try:
            files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False
        return bool(files) and all((model / name).is_file() and (model / name).stat().st_size > 0
                                   for name in files)
    return any(path.is_file() and path.stat().st_size > 0
               for pattern in ("*.safetensors", "pytorch_model.bin")
               for path in model.glob(pattern))


def checkpoint_complete(stage_dir: Path, step: int) -> bool:
    checkpoint = stage_dir / f"global_step_{step}"
    actor = checkpoint / "actor"
    rank_zero = list(actor.glob("model_world_size_*_rank_0.pt"))
    if (len(rank_zero) != 1 or not (actor / "config.json").is_file()
            or not (checkpoint / "data.pt").is_file()
            or not any(actor.glob("optim_world_size_*_rank_*.pt"))
            or not any((actor / name).is_file()
                       for name in ("tokenizer.json", "tokenizer.model", "vocab.json"))):
        return False
    match = re.fullmatch(r"model_world_size_(\d+)_rank_0\.pt", rank_zero[0].name)
    if not match:
        return False
    world_size = int(match.group(1))
    return (world_size > 0 and all(
        (actor / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
        and (actor / f"model_world_size_{world_size}_rank_{rank}.pt").stat().st_size > 0
        for rank in range(world_size)
    ))


def named_steps(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        name, sep, raw_step = value.partition("=")
        if not sep or not name or name in result:
            raise ValueError(f"expected unique STAGE=STEP, got: {value}")
        try:
            step = int(raw_step)
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint step: {value}") from exc
        if step < 1:
            raise ValueError(f"checkpoint step must be positive: {value}")
        result[name] = step
    return result


def stage_completed(run_dir: Path, stage: str, step: int) -> bool:
    marker = run_dir / stage / "COMPLETED.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    verified = record.get("verified") or {}
    return bool(verified.get("ok") and int(verified.get("last_step", -1)) == step)


def latest_checkpoint_step(stage_dir: Path) -> int | None:
    marker = stage_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return None
    try:
        return int(marker.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def training_active(run_dir: Path) -> bool:
    # Only inspect process command lines; no signals or GPU queries are issued.
    run_dir = run_dir.resolve()
    proc = Path("/proc")
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            argv = (item / "cmdline").read_bytes().decode(errors="ignore").split("\x00")
        except (OSError, PermissionError):
            continue
        if not any(flag == "-m" and module == "verl.trainer.main_ppo"
                   for flag, module in zip(argv, argv[1:])):
            continue
        local_dir = next((arg.split("=", 1)[1] for arg in reversed(argv)
                          if arg.startswith("trainer.default_local_dir=")), None)
        if not local_dir:
            continue
        try:
            stage_dir = Path(local_dir)
            if not stage_dir.is_absolute():
                stage_dir = item / "cwd" / stage_dir
            # The launcher writes to RUN/{seqkd,warmup,gad}; model inputs may belong to other runs.
            if stage_dir.resolve().parent == run_dir:
                return True
        except OSError:
            continue  # The process may exit while its working directory is being resolved.
    return False


def build_plan(train_path: Path, *, methods: list[str], tasks: list[str],
               gpu_ids: str, max_prompts: int, oprd_root: Path, dataset_root: Path,
               profile: str, seed: int, top_k: int, repetition_penalty: float,
               enforce_eager: bool, model_overrides: dict[str, Path] | None = None,
               dataset_overrides: dict[str, Path] | None = None,
               checkpoint_steps: dict[str, int] | None = None,
               protocol_overrides: dict | None = None, allow_new_eval_scorer: bool = False,
               reuse_generation_from: str | None = None, generate_missing: bool = False,
               task_n: dict[str, int] | None = None, resolved_config=None) -> dict:
    task_n = dict(task_n or {})
    if any(type(v) is not int or v < 1 for v in task_n.values()):
        raise ValueError(f'--task-n needs positive sample counts: {task_n}')
    if set(task_n) - set(tasks):
        # Entry-point defaults may name tasks the caller did not select.
        print(f'[eval] --task-n ignored for unselected tasks: {sorted(set(task_n) - set(tasks))}', flush=True)
        task_n = {task: count for task, count in task_n.items() if task in tasks}
    if generate_missing and not allow_new_eval_scorer:
        raise ValueError('--generate-missing requires --allow-new-eval-scorer')
    if generate_missing and reuse_generation_from:
        raise ValueError('--generate-missing cannot be combined with --reuse-generation-from; '
                         'reuse is read-only rescoring, use a separate PROFILE for new generation')
    cfg = resolved_config or load_config(train_path)
    training_scorer = {key[13:]: value for key, value in cfg.data.items() if key.startswith('scoring.math.')}
    evaluation_scorer = None
    evaluation_scoring_config = None
    if cfg.get('scoring.math.version') == 'math_unified_v1':
        from scoring.math import VERSION as MATH_VERSION
        from scoring.math import identity
        evaluation_identity_info = identity(cfg.get('scoring.math.timeout_seconds'))
        evaluation_scorer = {'version': MATH_VERSION,
                             'fingerprint': evaluation_identity_info['fingerprint'],
                             'protocol_revision': evaluation_identity_info.get('protocol', {}).get('protocol_revision')}
        if cfg.get('scoring.math.fingerprint') != evaluation_identity_info['fingerprint']:
            if not allow_new_eval_scorer:
                raise ValueError('evaluation scorer differs from frozen run/dependency identity; '
                                 'use a new PROFILE with --allow-new-eval-scorer; '
                                 'add --generate-missing for first-time evaluation, or '
                                 '--reuse-generation-from PROFILE to rescore saved raw samples')
            print('[eval] using a NEW evaluation scorer for a model trained with the OLD scorer; '
                  'the frozen training identity is preserved; missing-answer generation is '
                  + ('enabled' if generate_missing else 'disabled'), flush=True)
            evaluation_scoring_config = {'version': MATH_VERSION,
                                         'entrypoint': str(LMOPS_ROOT / 'scoring/math.py'),
                                         'timeout_seconds': float(cfg.get('scoring.math.timeout_seconds', 2.0)),
                                         'fingerprint': evaluation_identity_info['fingerprint']}
    train_path = cfg.path
    train = nested_config(cfg.data)
    if train["rollout"].get("enable_thinking") is not False:
        raise ValueError("this evaluator requires rollout.enable_thinking=false in train.yaml")
    if train["rollout"].get("thinking_control") is not True:
        raise ValueError("training did not enable the chat-template thinking switch")

    run_cfg, data_cfg = train["run"], train["data"]
    run_dir = resolve_from(train_path, run_cfg["output_root"]) / run_cfg["run_id"]
    reuse_root = None
    if reuse_generation_from:
        if not allow_new_eval_scorer:
            raise ValueError('--reuse-generation-from requires --allow-new-eval-scorer')
        candidate_dir = Path(reuse_generation_from).expanduser()
        reuse_root = candidate_dir if candidate_dir.is_absolute() else run_dir / 'evaluation' / candidate_dir
        reuse_root = reuse_root.resolve()
        if not reuse_root.is_dir():
            raise ValueError(f'reuse generation directory not found: {reuse_root}')
    data_path = resolve_from(train_path, data_cfg["train_file"])
    rows = pq.ParquetFile(data_path).metadata.num_rows
    per_epoch = rows // int(data_cfg["train_batch_size"])
    if per_epoch < 1:
        raise ValueError("train dataset has no complete batch")
    targets = {stage: stage_step_budget(cfg.for_stage(stage), stage, per_epoch) for stage in ("seqkd", "warmup")}
    targets["gad"] = targets["warmup"] + stage_step_budget(cfg, "gad", per_epoch, handoff_step=targets["warmup"])
    if targets["gad"] <= targets["warmup"]:
        raise ValueError("GAD absolute target must exceed Warmup handoff step")

    snapshot_ptr = run_dir / "current_resolved_config.txt"
    snapshot_path = None
    snapshot_rollout_n = None
    if snapshot_ptr.is_file():
        snapshot_path = Path(snapshot_ptr.read_text(encoding="utf-8").strip())
        if snapshot_path.is_file():
            snapshot = load_config(snapshot_path).to_dict()
            if snapshot.get("rollout.enable_thinking") is not False:
                raise ValueError("the active run snapshot did not use non-thinking prompts")
            if str(snapshot.get("models.student")) != str(resolve_from(train_path, train["models"]["student"])):
                raise ValueError("the active run snapshot uses a different student model")
            expected_data = str(resolve_from(train_path, data_cfg["train_file"]))
            if str(snapshot.get("data.train_file")) != expected_data:
                raise ValueError("the active run snapshot uses a different training subset")
            snapshot_rollout_n = int(snapshot.get("rollout.n", -1))

    model_overrides = model_overrides or {}
    dataset_overrides = dataset_overrides or {}
    checkpoint_steps = checkpoint_steps or {}
    if set(checkpoint_steps) - set(METHODS[1:]) or set(checkpoint_steps) & set(model_overrides):
        raise ValueError("checkpoint-step overrides require a training stage and cannot use model-path")
    invalid_methods = set(methods) - set(METHODS) - set(model_overrides)
    if invalid_methods:
        raise ValueError(f"unknown methods without explicit model paths: {sorted(invalid_methods)}")
    if not methods or not tasks or len(methods) != len(set(methods)) or len(tasks) != len(set(tasks)):
        raise ValueError("select at least one unique model and dataset")
    for name in [*methods, *tasks]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"unsafe model/dataset name: {name}")
    model_plan = {}
    for method in methods:
        is_stage = method in METHODS[1:] and method not in model_overrides
        step = checkpoint_steps.get(method, targets.get(method)) if is_stage else None
        if method in model_overrides:
            path = model_overrides[method]
        elif is_stage:
            path = run_dir / method / f"global_step_{step}" / "actor" / "huggingface"
        else:
            path = resolve_from(train_path, train["models"]["student"])
        selection = "explicit_step" if method in checkpoint_steps else (
            "configured_final" if is_stage else "model_path")
        latest = latest_checkpoint_step(run_dir / method) if is_stage else None
        if is_stage:
            checkpoint_ready = (
                checkpoint_complete(run_dir / method, step)
                and (method in checkpoint_steps or (latest == step and stage_completed(run_dir, method, step)))
            )
        else:
            checkpoint_ready = None
        ready = hf_ready(path)
        model_plan[method] = {
            "path": str(path), "ready": ready,
            "target_step": step, "selection": selection,
            "checkpoint_ready": checkpoint_ready,
            "latest_checkpoint_step": latest,
            "needs_export": bool(checkpoint_ready and not ready),
            'content': model_content(run_dir/method/f'global_step_{step}'/'actor' if is_stage else path, recursive=not is_stage)
                if (run_dir/method/f'global_step_{step}'/'actor' if is_stage else path).is_dir() else None,
        }

    oprd_root = oprd_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    if not (oprd_root / "sft" / "sample_teacher.py").is_file():
        raise FileNotFoundError(f"OPRD sampler missing under {oprd_root}")
    data_plan = {}
    for task in tasks:
        path = dataset_overrides.get(task, dataset_root / task / "test.parquet")
        if not path.is_file():
            raise FileNotFoundError(f"test set missing: {path}")
        parquet = pq.ParquetFile(path)
        if not {"prompt", "reward_model", "extra_info"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"test set lacks evaluation columns: {path}")
        data_plan[task] = {"path": str(path), "rows": parquet.metadata.num_rows,
                           "sha256": sha256(path),
                           'expected_question_ids':sampler_question_ids(oprd_root,path,max_prompts)}
        from evaluate import answer_forms, choice, minerva
        if task in choice.TASKS:
            data_plan[task]['evaluation_scorer'] = choice.identity()
        if task in minerva.TASKS:
            data_plan[task]['evaluation_postprocess'] = minerva.identity()
        if task in answer_forms.TASKS:
            data_plan[task]['evaluation_answer_forms'] = answer_forms.identity()

    protocol = {**train["eval"], **(protocol_overrides or {})}
    if protocol.get("backend") != "vllm" or int(protocol.get("tensor_parallel_size", 0)) != 1:
        raise ValueError("OPRD reference sampler requires eval.backend=vllm and tensor_parallel_size=1")
    n = int(protocol["n"])
    math500_n = int(protocol["math500_n"])
    if n < 1 or math500_n < 1:
        raise ValueError("evaluation sample counts must be positive")
    if not 0 <= float(protocol["temperature"]) or not 0 < float(protocol["top_p"]) <= 1:
        raise ValueError("invalid temperature or top_p")
    max_tokens, max_model_len = int(protocol["max_tokens"]), int(protocol["max_model_len"])
    if max_tokens < 1 or max_model_len < 1 or not 0 < float(protocol["gpu_memory_utilization"]) <= 1:
        raise ValueError("invalid token limit or GPU memory utilization")
    if max_model_len < max_tokens + int(data_cfg["max_prompt_length"]):
        raise ValueError("eval.max_model_len is below the training prompt budget plus evaluation response cap")
    chosen_gpus = gpu_ids
    gpu_list = [s.strip() for s in chosen_gpus.split(",") if s.strip()]
    if not gpu_list or len(gpu_list) != len(set(gpu_list)) or not all(s.isdigit() for s in gpu_list):
        raise ValueError(f"invalid GPU IDs: {chosen_gpus}")
    if max_prompts < 0:
        raise ValueError("--max-prompts must be >= 0")
    profile = str(profile)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", profile):
        raise ValueError(f"unsafe output profile: {profile}")
    profile += f"_smoke{max_prompts}" if max_prompts else ""
    return {
        "experiment": train_path.parent.name, "train_config": str(train_path),
        "run_dir": str(run_dir), "training_snapshot": str(snapshot_path) if snapshot_path else None,
        "training_rollout_n": snapshot_rollout_n,
        "configured_rollout_n": int(train["rollout"]["n"]),
        "output_root": str(run_dir / "evaluation" / profile),
        "oprd_root": str(oprd_root), "python_bin": str(run_cfg["python_bin"]),
        "models": model_plan, "tasks": data_plan, "steps_per_epoch": per_epoch,
        "gpu_ids": ",".join(gpu_list), "max_prompts": max_prompts,
        "protocol": {"n": n, "math500_n": math500_n,
                     "temperature": float(protocol["temperature"]), "top_p": float(protocol["top_p"]),
                     "top_k": int(top_k),
                     "repetition_penalty": float(repetition_penalty),
                     "max_tokens": max_tokens, "max_model_len": max_model_len,
                     "gpu_memory_utilization": float(protocol["gpu_memory_utilization"]),
                     "enforce_eager": bool(enforce_eager),
                     "enable_thinking": False, "seed": int(seed),
                     # Only present when used, so existing profiles keep their identity.
                     **({"task_n": dict(sorted(task_n.items()))} if task_n else {})},
        "scoring_math": evaluation_scoring_config or (
            {key[13:]: value for key, value in cfg.data.items() if key.startswith('scoring.math.')}
            if cfg.get('scoring.math.version') == 'math_unified_v1' else __import__('scoring.math', fromlist=['DEFAULT_CONFIG']).DEFAULT_CONFIG),
        "training_scorer": training_scorer,
        "evaluation_scorer": evaluation_scorer,
        "allow_new_eval_scorer": bool(allow_new_eval_scorer),
        "generate_missing": bool(generate_missing),
        "reuse_generation_from": str(reuse_root) if reuse_root else None,
        "scorer_identity": __import__('scoring.math', fromlist=['identity']).identity(cfg.get('scoring.math.timeout_seconds')),
    }


def export_stage(plan: dict, stage: str, model: dict) -> None:
    stage_dir = Path(plan["run_dir"]) / stage
    manifest = stage_dir / f"eval_export_step_{model['target_step']}.json"
    command = [
        plan["python_bin"], str(LMOPS_ROOT / "gad" / "tools" / "prepare_gad_checkpoint.py"),
        "--mode", "export-final", "--gad-dir", str(stage_dir),
        "--step", str(model["target_step"]), "--python-bin", plan["python_bin"],
        "--merge-tool", str(LMOPS_ROOT / "gad" / "tools" / "merge_model2hf.py"),
        "--manifest", str(manifest),
    ]
    print(f"[eval] exporting {stage} global_step_{model['target_step']} -> {model['path']}", flush=True)
    subprocess.run(command, cwd=LMOPS_ROOT, check=True)
    result = json.loads(manifest.read_text(encoding="utf-8"))
    if (not result.get("ok") or int(result.get("step", -1)) != model["target_step"]
            or result.get("actor_hf") != model["path"] or not hf_ready(Path(model["path"]))):
        raise RuntimeError(f"{stage} export failed verification: {manifest}")
    model["ready"] = True
    model["needs_export"] = False
    model["export_manifest"] = str(manifest)


def evaluation_identity(plan):
    identity = {**{k:plan[k] for k in ('tasks','protocol','max_prompts','scoring_math','scorer_identity')},
                'models':{name:{k:model.get(k) for k in ('path','target_step','selection','content')}
                          for name,model in plan['models'].items()}}
    if plan.get('allow_new_eval_scorer'):
        # Record the model's training scorer separately from its evaluation scorer.
        # generate_missing controls execution, not the sampling/scoring identity.
        identity['training_scorer'] = plan.get('training_scorer')
        identity['reuse_generation_from'] = plan.get('reuse_generation_from')
    return identity


def raw_complete(path, expected, n):
    if not path.is_file():return False
    # JSONL records end at "\n" only; str.splitlines() would also split on U+2028 inside responses.
    try:rows=[json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]
    except (OSError,ValueError):return False
    keys=[(str(r.get('global_index')),r.get('rollout_index')) for r in rows]
    return (all(type(k[1]) is int for k in keys) and len(keys)==len(set(keys)) and
            set(keys)=={(str(q),r) for q in expected for r in range(n)})


def execute(plan: dict) -> None:
    run_dir = Path(plan["run_dir"])
    if training_active(run_dir):
        raise RuntimeError("training for this experiment is still active; evaluate after it frees its GPUs")
    unavailable = []
    for name, model in plan["models"].items():
        if model["checkpoint_ready"] is False:
            latest = model["latest_checkpoint_step"]
            detail = f"latest checkpoint {latest}" if latest is not None else "no checkpoint marker"
            if model["selection"] == "explicit_step":
                unavailable.append(
                    f"{name}: requested global_step_{model['target_step']} has no complete actor checkpoint "
                    f"({detail})"
                )
            else:
                unavailable.append(
                    f"{name}: stage not completed ({detail}, configured target {model['target_step']}); "
                    f"set --checkpoint-step {name}=STEP to evaluate a saved interim checkpoint"
                )
        elif not model["ready"] and not model["needs_export"]:
            unavailable.append(f"{name}: no loadable Hugging Face weights at {model['path']}")
    if unavailable:
        raise RuntimeError("selected models unavailable:\n  " + "\n  ".join(unavailable))
    out_root = Path(plan["output_root"])
    saved_plan = out_root / "plan.json"
    if saved_plan.is_file():
        previous = json.loads(saved_plan.read_text(encoding="utf-8"))
        if evaluation_identity(previous) != evaluation_identity(plan):
            raise RuntimeError(f"evaluation settings changed in {out_root}; choose a new PROFILE")
    for name, model in plan["models"].items():
        if model["needs_export"]:
            export_stage(plan, name, model)
        actual = model_content(Path(model['path']))
        if not actual:
            raise RuntimeError(f'no actual HF content after export: {model["path"]}')
        prior_model = previous['models'].get(name, {}) if saved_plan.is_file() else {}
        prior_hf = prior_model.get('hf_content')
        if saved_plan.is_file() and prior_hf is None:
            raise RuntimeError('old evaluation lacks actual HF identity; preserve it and use a new PROFILE')
        if prior_hf is not None and prior_hf != actual:
            raise RuntimeError(f'actual HF content changed for {name}; choose a new PROFILE')
        model['hf_content'] = actual
    out_root.mkdir(parents=True, exist_ok=True)
    from gad.extensions.common import atomic_json
    atomic_json(saved_plan,plan)
    if plan.get('allow_new_eval_scorer'):
        atomic_json(out_root / 'scoring_manifest.json', {
            'allow_new_eval_scorer': True,
            'generate_missing': plan.get('generate_missing', False),
            'training_scorer': plan.get('training_scorer'),
            'evaluation_scorer': plan.get('evaluation_scorer'),
            'scoring_math': plan.get('scoring_math'),
            'reuse_generation_from': plan.get('reuse_generation_from'),
            'aggregation_protocol': 'eval-fix.2 (fixed denominator, strict complete, lower/upper/coverage)',
        })
    protocol = plan["protocol"]
    python_bin = plan["python_bin"]
    oprd_root = Path(plan["oprd_root"])
    for method, model in plan["models"].items():
        for task, dataset in plan["tasks"].items():
            task_scorer = dataset.get('evaluation_scorer', plan.get('evaluation_scorer'))
            scoring_config = dict(plan['scoring_math'])
            if dataset.get('evaluation_scorer'):
                scoring_config['choice_fingerprint'] = task_scorer['fingerprint']
            if dataset.get('evaluation_postprocess'):
                scoring_config['minerva_fingerprint'] = dataset['evaluation_postprocess']['fingerprint']
            if dataset.get('evaluation_answer_forms'):
                scoring_config['answer_forms_fingerprint'] = dataset['evaluation_answer_forms']['fingerprint']
            task_dir = out_root / method / task
            generation = task_dir / "generation"
            scores = task_dir / "scores"
            samples = protocol.get("task_n", {}).get(task) or (protocol["math500_n"] if task == "MATH-500" else protocol["n"])
            expected=dataset['expected_question_ids']
            expected_path=task_dir/'expected_question_ids.json'
            atomic_json(expected_path,expected)
            raw=generation/'raw_samples.jsonl'
            reuse_raw = Path(plan['reuse_generation_from'])/method/task/'generation/raw_samples.jsonl' if plan.get('reuse_generation_from') else None
            if reuse_raw is not None and raw_complete(reuse_raw, expected, samples):
                raw = reuse_raw
            state_path=task_dir/'state.json'
            state=json.loads(state_path.read_text()) if state_path.is_file() else {}
            current_identity=hashlib.sha256(json.dumps({'evaluation':evaluation_identity(plan),'hf_content':model['hf_content'],'method':method,'task':task},sort_keys=True).encode()).hexdigest()
            metric_path=Path(state.get('metrics_path',scores/'metrics.json'))
            if (state.get('identity')==current_identity and state.get('status')=='complete' and raw.is_file() and
                    metric_path.is_file() and state.get('raw_sha256')==sha256(raw) and
                    state.get('metrics_sha256')==sha256(metric_path)):
                metrics=json.loads(metric_path.read_text())
                if metrics.get('evaluation_complete') and raw_complete(raw,expected,samples):
                    print(f'[eval] {method}/{task}: complete, skipped',flush=True);continue
            sampler = [python_bin, "-m", "sft.sample_teacher",
                       "--input-parquet", dataset["path"], "--model-path", model["path"],
                       "--output-dir", str(generation), "--gpu-ids", plan["gpu_ids"],
                       "--num-rollouts", str(samples), "--max-prompts", str(plan["max_prompts"]),
                       "--temperature", str(protocol["temperature"]), "--top-p", str(protocol["top_p"]),
                       "--top-k", str(protocol["top_k"]),
                       "--repetition-penalty", str(protocol["repetition_penalty"]),
                       "--max-new-tokens", str(protocol["max_tokens"]),
                       "--max-model-len", str(protocol["max_model_len"]),
                       "--gpu-memory-utilization", str(protocol["gpu_memory_utilization"]),
                       "--seed", str(protocol["seed"]), "--no-enable-thinking",
                       "--no-basic-rejection", "--no-regenerate-until-correct",
                       "--wandb-mode", "disabled"]
            if protocol["enforce_eager"]:
                sampler.append("--enforce-eager")
            print(f"[eval] {method}/{task}: {samples} samples/question", flush=True)
            if not raw_complete(raw,expected,samples):
                if plan.get('allow_new_eval_scorer') and not plan.get('generate_missing'):
                    raise RuntimeError(f'{method}/{task}: saved raw samples are incomplete; '
                                       'rescoring does not generate answers. For first-time evaluation '
                                       'add --generate-missing with a new PROFILE; for existing raw '
                                       'use --reuse-generation-from PROFILE')
                atomic_json(state_path,{'identity':current_identity,'status':'generating',
                                        'training_scorer':plan.get('training_scorer'),
                                        'evaluation_scorer':task_scorer})
                subprocess.run(sampler, cwd=oprd_root, check=True)
            if (scores/'metrics.json').exists():
                # Preserve old/incomplete/corrupt scores. Each replacement has its own directory.
                from datetime import datetime,timezone
                scores=task_dir/('scores_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            atomic_json(state_path,{'identity':current_identity,'status':'scoring','raw_sha256':sha256(raw),
                                    'raw_path':str(raw),
                                    'training_scorer':plan.get('training_scorer'),
                'evaluation_scorer':task_scorer})
            subprocess.run([python_bin, str(LMOPS_ROOT / 'evaluate/rescore.py'), '--raw-jsonl', str(raw),
                '--output-dir', str(scores), '--task', task, '--n', str(samples), '--expected-ids', str(expected_path),
                '--config-json', json.dumps(scoring_config)], cwd=LMOPS_ROOT, check=True)
            metrics=json.loads((scores/'metrics.json').read_text())
            atomic_json(state_path,{'identity':current_identity,'status':'complete' if metrics.get('evaluation_complete') else 'incomplete',
                'raw_sha256':sha256(raw),'raw_path':str(raw),'metrics_path':str(scores/'metrics.json'), 'metrics_sha256':sha256(scores/'metrics.json'),
                'counter_correct':metrics.get('confirmed_correct'),'counter_incorrect':metrics.get('confirmed_incorrect'),
                'unresolved_samples':metrics.get('unresolved_samples'),
                'sample_accuracy_lower':metrics.get('sample_accuracy_lower'),
                'sample_accuracy_upper':metrics.get('sample_accuracy_upper'),
                'coverage_ratio':metrics.get('coverage'),
                'training_scorer':plan.get('training_scorer'),
                'evaluation_scorer':task_scorer})
    print(f"[eval] results -> {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser, config_alias="--train-config")
    parser.add_argument("--oprd-root", type=Path, default=LMOPS_ROOT/"third_party/oprd")
    parser.add_argument("--dataset-root", type=Path, default=LMOPS_ROOT/"assets/datasets/math_eval")
    parser.add_argument("--methods", required=True, help="Comma-separated model names")
    parser.add_argument("--tasks", required=True, help="Comma-separated dataset names")
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU IDs")
    parser.add_argument("--model-path", action="append", default=[], metavar="METHOD=PATH",
                        help="Explicit local model path; may be repeated")
    parser.add_argument("--dataset-path", action="append", default=[], metavar="TASK=PATH",
                        help="Explicit Parquet path; may be repeated")
    parser.add_argument("--checkpoint-step", action="append", default=[], metavar="STAGE=STEP",
                        help="Evaluate an explicitly selected saved stage checkpoint")
    parser.add_argument("--profile", required=True, help="Output profile; change it when evaluation settings change")
    parser.add_argument("--allow-new-eval-scorer", action="store_true",
                        help="Use the current scorer with an old training identity (requires a NEW profile; rescore-only unless --generate-missing)")
    parser.add_argument("--generate-missing", action="store_true",
                        help="Allow first-time/missing-answer generation with --allow-new-eval-scorer; cannot combine with --reuse-generation-from")
    parser.add_argument("--reuse-generation-from", default=None, metavar="PROFILE",
                        help="Old evaluation PROFILE whose complete raw_samples.jsonl are scored read-only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--math500-n", type=int)
    parser.add_argument("--task-n", action="append", default=[], metavar="TASK=N",
                        help="Samples per question for one task (overrides --n/--math500-n); may be repeated")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-prompts", type=int, default=0, help="First N questions per task; 0=all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    protocol_keys = ("n", "math500_n", "temperature", "top_p", "max_tokens",
                     "max_model_len", "gpu_memory_utilization")
    protocol_overrides = {key: getattr(args, key) for key in protocol_keys
                          if getattr(args, key) is not None}
    cfg = select_config(args, existing_only=True)
    plan = build_plan(cfg.path,
                      methods=select(args.methods, []),
                      tasks=select(args.tasks, []),
                      gpu_ids=args.gpu_ids, max_prompts=args.max_prompts,
                      oprd_root=args.oprd_root, dataset_root=args.dataset_root,
                      profile=args.profile, seed=args.seed, top_k=args.top_k,
                      repetition_penalty=args.repetition_penalty,
                      enforce_eager=args.enforce_eager,
                      model_overrides=named_paths(args.model_path),
                      dataset_overrides=named_paths(args.dataset_path),
                      checkpoint_steps=named_steps(args.checkpoint_step),
                      protocol_overrides=protocol_overrides,
                      allow_new_eval_scorer=args.allow_new_eval_scorer,
                      reuse_generation_from=args.reuse_generation_from,
                      generate_missing=args.generate_missing,
                      # Later TASK=N wins, so entry-point defaults can be overridden.
                      task_n={name: int(count) for name, _, count in (v.partition('=') for v in args.task_n)},
                      resolved_config=cfg)
    if args.summary:
        subprocess.run([plan["python_bin"], str(EVALUATE_DIR / "summarize_eval.py"),
                        "--root", plan["output_root"]], check=True)
        return
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if not args.dry_run:
        execute(plan)
        subprocess.run([plan["python_bin"], str(EVALUATE_DIR / "summarize_eval.py"),
                        "--root", plan["output_root"]], check=True)


if __name__ == "__main__":
    main()
