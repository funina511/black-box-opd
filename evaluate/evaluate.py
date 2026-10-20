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
    proc = Path("/proc")
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            argv = (item / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "verl.trainer.main_ppo" in argv and str(run_dir) in argv:
            return True
    return False


def build_plan(train_path: Path, *, methods: list[str], tasks: list[str],
               gpu_ids: str, max_prompts: int, oprd_root: Path, dataset_root: Path,
               profile: str, seed: int, top_k: int, repetition_penalty: float,
               enforce_eager: bool, model_overrides: dict[str, Path] | None = None,
               dataset_overrides: dict[str, Path] | None = None,
               checkpoint_steps: dict[str, int] | None = None,
               protocol_overrides: dict | None = None, resolved_config=None) -> dict:
    cfg = resolved_config or load_config(train_path)
    if cfg.get('scoring.math.version') == 'math_unified_v1':
        from scoring.math import identity
        if cfg.get('scoring.math.fingerprint') != identity(cfg.get('scoring.math.timeout_seconds'))['fingerprint']:
            raise ValueError('evaluation scorer differs from frozen run/dependency identity')
    train_path = cfg.path
    train = nested_config(cfg.data)
    if train["rollout"].get("enable_thinking") is not False:
        raise ValueError("this evaluator requires rollout.enable_thinking=false in train.yaml")
    if train["rollout"].get("thinking_control") is not True:
        raise ValueError("training did not enable the chat-template thinking switch")

    run_cfg, data_cfg = train["run"], train["data"]
    run_dir = resolve_from(train_path, run_cfg["output_root"]) / run_cfg["run_id"]
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
        }

    oprd_root = oprd_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    if not (oprd_root / "sft" / "sample_teacher.py").is_file():
        raise FileNotFoundError(f"OPRD sampler missing under {oprd_root}")
    if not (oprd_root / "sft" / "score_math.py").is_file():
        raise FileNotFoundError(f"OPRD scorer missing under {oprd_root}")
    data_plan = {}
    for task in tasks:
        path = dataset_overrides.get(task, dataset_root / task / "test.parquet")
        if not path.is_file():
            raise FileNotFoundError(f"test set missing: {path}")
        parquet = pq.ParquetFile(path)
        if not {"prompt", "reward_model", "extra_info"}.issubset(parquet.schema_arrow.names):
            raise ValueError(f"test set lacks evaluation columns: {path}")
        data_plan[task] = {"path": str(path), "rows": parquet.metadata.num_rows,
                           "sha256": sha256(path)}

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
                     "enable_thinking": False, "seed": int(seed)},
        "scoring_math": {key[13:]: value for key, value in cfg.data.items() if key.startswith('scoring.math.')}
            if cfg.get('scoring.math.version') == 'math_unified_v1' else __import__('scoring.math', fromlist=['DEFAULT_CONFIG']).DEFAULT_CONFIG,
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
    for name, model in plan["models"].items():
        if model["needs_export"]:
            export_stage(plan, name, model)
    out_root = Path(plan["output_root"])
    saved_plan = out_root / "plan.json"
    if saved_plan.is_file():
        previous = json.loads(saved_plan.read_text(encoding="utf-8"))
        keys = ("models", "tasks", "protocol", "gpu_ids", "max_prompts", 'scoring_math', 'scorer_identity')
        if any(previous.get(key) != plan.get(key) for key in keys):
            raise RuntimeError(f"evaluation settings changed in {out_root}; choose a new PROFILE")
    out_root.mkdir(parents=True, exist_ok=True)
    saved_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol = plan["protocol"]
    python_bin = plan["python_bin"]
    oprd_root = Path(plan["oprd_root"])
    for method, model in plan["models"].items():
        for task, dataset in plan["tasks"].items():
            task_dir = out_root / method / task
            generation = task_dir / "generation"
            scores = task_dir / "scores"
            samples = protocol["math500_n"] if task == "MATH-500" else protocol["n"]
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
            subprocess.run(sampler, cwd=oprd_root, check=True)
            raw = generation / "raw_samples.jsonl"
            expected = min(dataset['rows'], plan['max_prompts']) if plan['max_prompts'] else dataset['rows']
            subprocess.run([python_bin, str(LMOPS_ROOT / 'evaluate/rescore.py'), '--raw-jsonl', str(raw),
                '--output-dir', str(scores), '--task', task, '--n', str(samples), '--expected-prompts', str(expected),
                '--config-json', json.dumps(plan['scoring_math'])], cwd=LMOPS_ROOT, check=True)
    print(f"[eval] results -> {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser, config_alias="--train-config")
    parser.add_argument("--oprd-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--math500-n", type=int)
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
                      protocol_overrides=protocol_overrides, resolved_config=cfg)
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
