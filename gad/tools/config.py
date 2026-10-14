"""Pipeline configuration: loading, defaults, validation, resolution.

The config is one YAML file (see ``configs/pipeline.example.yaml``) that covers
the three training stages, the model/data inputs, the per-stage budgets and the
external evaluation. Loading is strict on purpose:

* required keys must be present and non-empty - there is no silent fallback to a
  default model, dataset or output directory;
* every reported problem is collected before failing, so one run of
  ``preflight``/``--dry-run`` lists everything that is missing;
* relative paths resolve against the config file location, not the CWD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import PipelineError, TRAIN_STAGES, flatten_dict

GAD_DIR = Path(__file__).resolve().parents[1]

from extension_config import EXTENSION_DEFAULTS

DEFAULTS: dict[str, Any] = {
    "run.python_bin": None,  # resolved later: current interpreter
    "run.run_id": "auto",
    "run.log_backend": "console",
    "run.wandb_project": "",
    "run.eval_after_train": False,
    "run.skip_completed_stages": True,
    "run.allow_cpu_smoke": False,
    "run.val_before_train": True,
    "run.resume_worker_state": False,
    # PYTORCH_CUDA_ALLOC_CONF for the stage processes. Empty = leave it alone.
    # expandable_segments is NOT usable here: vLLM runs with enable_sleep_mode=True and its
    # CuMemAllocator asserts against it (vllm/device_allocator/cumem.py:146).
    "run.pytorch_cuda_alloc_conf": "",
    "backends.source_repo": str(GAD_DIR / "verl"),
    "backends.root": str(GAD_DIR / ".backends"),
    "backends.recreate": False,
    "models.student": None,
    "models.discriminator_init": None,
    "models.trust_remote_code": True,
    "models.use_remove_padding": True,
    "models.enable_gradient_checkpointing": True,
    "models.param_offload": False,
    "models.optimizer_offload": False,
    # The KL reference model is forward-only, so the original harness keeps its parameters
    # on CPU (``actor_rollout_ref.ref.fsdp_config.param_offload=True``): a ~3.4GB per-GPU
    # saving in Warmup/GAD on 24GB cards. Kept separate from models.param_offload, which
    # also covers the actor and the discriminator.
    "models.ref_param_offload": True,
    "data.train_file": None,
    "data.val_file": None,
    "data.prompt_key": "content",
    "data.max_prompt_length": 2048,
    "data.max_response_length": 8192,
    "data.truncation": "right",
    "data.train_batch_size": None,
    "data.val_batch_size": None,
    "data.teacher_index": None,
    "data.manifest": None,
    "data.expected_fingerprint_digest": None,
    "rollout.name": "vllm",
    # GAD needs a uid group to normalise advantages over (reference: ROLLOUT_N, guarded to
    # be >= 2). SeqKD/Warmup keep only the first candidate per uid
    # (``dp_actor.deduplicate_by_uid``). n=1 improves throughput but changes the
    # discriminator candidate distribution relative to n=8.
    "rollout.n": 8,
    "rollout.n_seqkd": 1,
    "rollout.n_warmup": 1,
    "rollout.temperature": 0.8,
    "rollout.top_p": 1.0,
    "rollout.tensor_model_parallel_size": 2,
    "rollout.gpu_memory_utilization": 0.7,
    "rollout.enforce_eager": False,
    "rollout.free_cache_engine": False,
    # vLLM engine limits. ``None`` means "derive from the length budget"
    # (``data.max_prompt_length + data.max_response_length``). vLLM's own default for
    # max_num_batched_tokens is 8192, which is smaller than that budget and makes
    # vLLM abort at engine construction while chunked prefill is enabled, so the
    # pipeline always passes an explicit value.
    "rollout.max_model_len": None,
    "rollout.max_num_batched_tokens": None,
    "rollout.max_num_seqs": 128,
    "rollout.val_n": 1,
    "rollout.val_do_sample": False,
    "rollout.val_temperature": 0.0,
    "rollout.enable_thinking": False,
    "rollout.thinking_control": True,
    "budget.seqkd_steps": 200,
    "budget.warmup_steps": 800,
    "budget.gad_steps": 200,
    # Original-recipe budget, per stage, in epochs (see tools/pipeline.py plan_stage):
    #   budget.stage_epochs.{seqkd,warmup} -> that many epochs of its own data;
    #   budget.stage_epochs.gad            -> ABSOLUTE global target in epochs, because the
    #                                         GAD stage continues the Warmup step counter.
    # When a stage has an epoch budget, the matching *_steps value is not used.
    "budget.stage_epochs": None,
    "budget.epochs_override": None,
    "budget.handoff_step": None,
    "optim.seqkd_actor_lr": 5e-6,
    "optim.warmup_actor_lr": 1e-6,
    "optim.warmup_critic_lr": 1e-6,
    "optim.gad_actor_lr": 1e-6,
    "optim.gad_critic_lr": 1e-6,
    "optim.grad_clip": 0.2,
    "optim.kl_loss_coef": 0.001,
    "optim.kl_loss_type": "low_var_kl",
    "optim.entropy_coeff": 0.0,
    "optim.critic_warmup": 10,
    # R5 of the reference harness (_dapo_common.sh): CRITIC_DYNAMIC_BSZ=False, CRITIC_MICRO=1.
    # The discriminator's batching is decoupled from the actor and static by default, so its
    # micro-batches come from ``batch.split(micro)`` and never from the token-budget
    # partitioner - which keeps it bit-identical to the pinned code and immune to the
    # "student 1024 + teacher 16384" overflow the actor path needs the patch for.
    "optim.critic_dynamic_bsz": False,
    "optim.critic_micro_batch_size_per_gpu": 1,
    "optim.actor_ppo_max_token_len_per_gpu": 12288,
    "optim.critic_ppo_max_token_len_per_gpu": 12288,
    "resources.n_gpus_per_node": 8,
    "resources.nnodes": 1,
    "resources.gpus": "",
    "resources.save_freq": 50,
    "resources.keep_last_checkpoints": None,
    "resources.test_freq": 50,
    "resources.max_actor_ckpt_to_keep": None,
    "resources.max_critic_ckpt_to_keep": None,
    "eval.enabled": False,
    "eval.parquet": None,
    "eval.benchmarks": "aime24,aime25,amc23,amc24,math500",
    "eval.n": 16,
    "eval.math500_n": 2,
    "eval.temperature": 1.0,
    "eval.top_p": 0.95,
    "eval.max_tokens": 32768,
    "eval.max_model_len": 40960,
    "eval.tensor_parallel_size": 2,
    "eval.gpu_memory_utilization": 0.85,
    "eval.backend": "vllm",
    "eval.dev_n": 1,
    "eval.dev_do_sample": False,
    "validation.scorer_path": None,
    "validation.require_math_metrics": True,
    "validation.quick_enabled": False,
    "validation.quick_size": 128,
    "validation.quick_seed": 42,
    "validation.quick_file": None,
    "validation.final_full": True,
}

REQUIRED = (
    "run.output_root",
    "models.student",
    "data.train_file",
    "data.val_file",
    "data.train_batch_size",
)

DEFAULTS.update(EXTENSION_DEFAULTS)

# Keys that name a filesystem path and must exist for a real (non-dry) run.
PATH_KEYS = (
    "models.student",
    "models.discriminator_init",
    "data.train_file",
    "data.val_file",
    "data.teacher_index",
    "data.manifest",
    "eval.parquet",
    "validation.scorer_path",
    "validation.quick_file",
    "rubric.spec",
    "rubric.prompt",
    "rubric.judge.tokenizer_path",
)

STAGE_LR_KEY = {
    "seqkd": "optim.seqkd_actor_lr",
    "warmup": "optim.warmup_actor_lr",
    "gad": "optim.gad_actor_lr",
}

STAGE_STEPS_KEY = {
    "seqkd": "budget.seqkd_steps",
    "warmup": "budget.warmup_steps",
    "gad": "budget.gad_steps",
}


class Config:
    """Resolved pipeline configuration with dotted-key access."""

    def __init__(self, data: dict, *, path: Path):
        self.data = data
        self.path = path
        self.identity_data = None
        self.selection = {}
        self.runtime_overrides = {}
        self.global_config = False
        self.existing_run = False
        self.legacy_selection = False
        self.stage_overrides = {}

    def for_stage(self, stage: str) -> "Config":
        overlay = self.stage_overrides.get(stage)
        if not overlay:
            return self
        data = {**self.data, **overlay}
        for key in self.runtime_overrides:
            data[key] = self.data[key]
        result = Config(data, path=self.path)
        result.identity_data = {**(self.identity_data or self.data), **overlay}
        result.runtime_overrides = dict(self.runtime_overrides)
        result.existing_run = self.existing_run
        return result

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        if key in DEFAULTS:
            return DEFAULTS[key]
        return default

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise PipelineError(f"config key '{key}' is required but empty ({self.path})")
        return value

    def stage_steps(self, stage: str) -> int:
        steps = int(self.require(STAGE_STEPS_KEY[stage]))
        if steps <= 0:
            raise PipelineError(f"config key '{STAGE_STEPS_KEY[stage]}' must be > 0, got {steps}")
        return steps

    def stage_epochs(self, stage: str) -> int | None:
        """Epoch budget for a stage, or ``None`` to derive it from the step budget.

        The original GAD scripts size every stage in epochs (``trainer.total_epochs``);
        the pipeline is step driven, so this converts the original recipe into steps
        using ``train_rows // train_batch_size``.
        """
        value = self.get(f"budget.stage_epochs.{stage}")
        if value is None:
            return None
        epochs = int(value)
        if epochs <= 0:
            raise PipelineError(f"config key 'budget.stage_epochs.{stage}' must be > 0, got {epochs}")
        return epochs

    def stage_lr(self, stage: str) -> float:
        return float(self.require(STAGE_LR_KEY[stage]))

    def as_flat(self) -> dict:
        return flatten_dict(self.data)

    def to_dict(self) -> dict:
        return dict(self.data)


def resolve_paths(data: dict, base_dir: Path) -> dict:
    def resolve(value: Any) -> Any:
        if isinstance(value, str) and value:
            if value.startswith("~"):
                return str(Path(value).expanduser())
            candidate = Path(value)
            if candidate.is_absolute():
                return str(candidate)
            return str((base_dir / candidate).resolve())
        return value

    resolved = dict(data)
    for key in PATH_KEYS:
        if key in resolved and resolved[key] is not None:
            resolved[key] = resolve(resolved[key])
    for extra in ("backends.source_repo", "backends.root", "run.output_root"):
        if extra in resolved and resolved[extra]:
            resolved[extra] = resolve(resolved[extra])
    return resolved


LMOPS_DIR = GAD_DIR.parent
GLOBAL_CONFIG = LMOPS_DIR / "train.yaml"
CONFIG_METADATA = {"selection", "catalog", "legacy_runs", "_stage_overrides"}
RUNTIME_KEYS = {
    "resources.gpus", "run.python_bin", "run.log_backend", "run.val_before_train",
    "resources.save_freq", "resources.test_freq", "resources.keep_last_checkpoints",
    "run.max_stage_steps",
}


def read_yaml(path: Path) -> dict:
    import yaml
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PipelineError(f"config must be a YAML mapping: {path}")
    return raw


def nested_config(data: dict) -> dict:
    """Normalize old dotted-key snapshots and new nested configs."""
    result: dict = {}
    for key, value in sorted(flatten_dict(data).items(), key=lambda item: (item[0].count("."), item[0])):
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return result


def legacy_runs() -> dict[str, Path]:
    if not GLOBAL_CONFIG.is_file():
        return {}
    return {name: (GLOBAL_CONFIG.parent / value).resolve()
            for name, value in read_yaml(GLOBAL_CONFIG).get("legacy_runs", {}).items()}


def resolve_run(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw, LMOPS_DIR / raw, LMOPS_DIR / "runs" / raw]
    registered = legacy_runs().get(str(value))
    if registered:
        candidates.append(registered)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    matches = [path for path in (LMOPS_DIR / "runs").glob("*/*")
               if path.is_dir() and path.name == str(value)]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise PipelineError(f"ambiguous run {value}; pass its full directory path")
    raise PipelineError(f"run directory not found: {value}")


def run_config_path(value: str | Path) -> Path:
    run_dir = resolve_run(value)
    primary = run_dir / "train.yaml"
    if primary.is_file():
        return primary
    pointer = run_dir / "current_resolved_config.txt"
    if pointer.is_file():
        snapshot = Path(pointer.read_text(encoding="utf-8").strip())
        if not snapshot.is_absolute():
            snapshot = run_dir / snapshot
        if snapshot.is_file():
            return snapshot
    old = run_dir / "resolved_config.yaml"
    if old.is_file():
        return old
    raise PipelineError(f"no saved training config under {run_dir}")


def config_file(path: str | Path) -> Path:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    # Preserve explicit old config paths after experiments/ has been archived.
    if (requested.name == "train.yaml" and requested.parent.parent == LMOPS_DIR / "experiments"
            and requested.parent.name in legacy_runs()):
        return run_config_path(legacy_runs()[requested.parent.name])
    raise PipelineError(f"pipeline config not found: {requested}")


def _resolved_config(data: dict, path: Path) -> Config:
    import sys
    resolved = resolve_paths({**DEFAULTS, **flatten_dict(data)}, path.parent)
    if not resolved.get("run.python_bin"):
        resolved["run.python_bin"] = sys.executable
    if not resolved.get("models.discriminator_init"):
        resolved["models.discriminator_init"] = resolved.get("models.student")
    if not resolved.get("validation.scorer_path"):
        candidate = GAD_DIR.parent / "sample" / "math_scorer.py"
        resolved["validation.scorer_path"] = str(candidate) if candidate.is_file() else None
    cfg = Config(resolved, path=path)
    from extension_config import configure_profile
    configure_profile(cfg)
    return cfg


def validate_override_keys(data: dict) -> None:
    known = set(DEFAULTS) | set(REQUIRED)
    unknown = [key for key in data if key not in known
               and not any(key == f"budget.{group}.{stage}" for group in ("stage_epochs", "epochs_override")
                           for stage in TRAIN_STAGES)]
    if unknown:
        raise PipelineError(f"unknown training override(s): {unknown}")


def compose_config(path: str | Path = GLOBAL_CONFIG, *, model: str | None = None,
                   dataset: str | None = None, overrides: list[str] | None = None,
                   settings: list[str] | None = None) -> Config:
    from common import parse_key_values
    path = config_file(path)
    raw = read_yaml(path)
    data = resolve_paths(flatten_dict({key: value for key, value in raw.items()
                                      if key not in CONFIG_METADATA}), path.parent)
    selection = raw.get("selection", {})
    chosen = {"model": model or selection.get("model"), "dataset": dataset or selection.get("dataset")}
    for kind, group in (("model", "models"), ("dataset", "datasets")):
        name = chosen[kind]
        if name:
            choices = raw.get("catalog", {}).get(group, {})
            if name not in choices:
                raise PipelineError(f"unknown {kind} {name!r}; available: {', '.join(choices)}")
            layer = resolve_paths(flatten_dict(choices[name]), path.parent)
            if kind == "model" and "models.discriminator_init" not in layer:
                layer["models.discriminator_init"] = layer.get("models.student")
            data.update(layer)
    for value in overrides or []:
        overlay_path = Path(value).expanduser().resolve()
        overlay = read_yaml(overlay_path)
        if CONFIG_METADATA & overlay.keys():
            raise PipelineError(f"override YAML must contain training settings only: {overlay_path}")
        overlay_data = flatten_dict(overlay)
        validate_override_keys(overlay_data)
        data.update(resolve_paths(overlay_data, overlay_path.parent))
    # CLI paths are relative to the invocation directory, never to runs/<id>/.
    cli_settings = parse_key_values(settings or [])
    validate_override_keys(cli_settings)
    data.update(resolve_paths(cli_settings, Path.cwd()))
    cfg = _resolved_config(data, path)
    cfg.selection = {key: value for key, value in chosen.items() if value}
    cfg.global_config = bool(raw.get("catalog"))
    cfg.stage_overrides = {stage: resolve_paths(flatten_dict(value), path.parent)
                           for stage, value in raw.get("_stage_overrides", {}).items()}
    return cfg


def load_config(path: str | Path) -> Config:
    path = config_file(path)
    raw = read_yaml(path)
    if raw.get("catalog"):
        return compose_config(path)
    cfg = _resolved_config({key: value for key, value in raw.items() if key not in CONFIG_METADATA}, path)
    cfg.stage_overrides = {stage: resolve_paths(flatten_dict(value), path.parent)
                           for stage, value in raw.get("_stage_overrides", {}).items()}
    return cfg


def load_run_config(value: str | Path) -> Config:
    run_dir = resolve_run(value)
    cfg = load_config(run_config_path(run_dir))
    recorded = Path(cfg.require("run.output_root")) / str(cfg.require("run.run_id"))
    if recorded.resolve() != run_dir:
        raise PipelineError(f"saved config points to {recorded}, not requested run {run_dir}")
    cfg.identity_data = dict(cfg.data)
    return cfg


def add_config_arguments(parser, *, config_alias: str | None = None) -> None:
    flags = ["--config"] + ([config_alias] if config_alias else [])
    parser.add_argument(*flags, dest="config", help="Global template or saved training YAML")
    parser.add_argument("--run", help="Existing run directory or registered historical name")
    parser.add_argument("--model", help="Model name in the global catalog")
    parser.add_argument("--dataset", help="Prepared dataset name in the global catalog")
    parser.add_argument("--overrides", action="append", default=[], help="Small training override YAML")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model-name", help="Legacy experiment selector; prefer --model for new runs")
    parser.add_argument("--experiment-name", help="Legacy experiment selector")
    parser.add_argument("--experiment", help="Registered historical experiment name")


def select_config(args, *, existing_only: bool = False) -> Config:
    from common import parse_key_values
    legacy = args.experiment
    if args.model_name and not args.dataset:
        legacy = legacy or (args.model_name + ("_" + args.experiment_name if args.experiment_name else ""))
    elif args.experiment_name and not args.model_name:
        raise PipelineError("--experiment-name requires --model-name; use --name for a new run label")
    if args.run and legacy:
        raise PipelineError("choose either --run or a legacy experiment selector")
    run = args.run or legacy
    if run:
        if args.config or args.model or args.dataset or args.overrides:
            raise PipelineError("--run selects a frozen config; do not combine it with model/data/template selection")
        cfg = load_run_config(run)
        updates = parse_key_values(args.set)
        forbidden = set(updates) - RUNTIME_KEYS
        if forbidden:
            raise PipelineError(f"resume cannot change training settings {sorted(forbidden)}; start a new run")
        cfg.data.update(resolve_paths(updates, Path.cwd()))
        cfg.runtime_overrides = updates
        cfg.existing_run = True
        cfg.legacy_selection = bool(legacy)
    else:
        if existing_only and not args.config:
            raise PipelineError("select an existing experiment with --run or --config")
        cfg = compose_config(args.config or GLOBAL_CONFIG, model=args.model or args.model_name,
                             dataset=args.dataset, overrides=args.overrides, settings=args.set)
        if existing_only and cfg.global_config:
            raise PipelineError("this operation needs an existing run; pass --run")
        saved_dir = Path(cfg.require("run.output_root")) / str(cfg.get("run.run_id"))
        if cfg.path == saved_dir / "train.yaml":
            cfg.identity_data = dict(cfg.data)
            if args.set or args.overrides:
                raise PipelineError("use --run for recorded runtime overrides on a saved config")
            cfg.existing_run = True
    return cfg


def new_run_id(cfg: Config, name: str = "") -> str:
    from datetime import datetime, timezone, timedelta
    import re
    parts = [cfg.selection.get("model", "model"), cfg.selection.get("dataset", "data")]
    if name:
        parts.append(name)
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", part) for part in parts):
        raise PipelineError("model, dataset and run label must use letters, digits, _, - or .")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f")
    return "_".join([*parts, stamp])


def save_run_config(cfg: Config, run_dir: Path) -> None:
    """Claim a new run directory; never overwrite an existing experiment."""
    import os
    import yaml
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PipelineError(f"run already exists: {run_dir}; choose a new name or --run ... --resume") from exc
    target = run_dir / "train.yaml"
    temporary = target.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(nested_config(cfg.data), sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(temporary, target)
    cfg.path = target


def validate_paths(config: Config, *, check_exists: bool) -> list[str]:
    problems: list[str] = []
    if not check_exists:
        return problems
    for key in PATH_KEYS:
        if key == "eval.parquet" and not config.get("eval.enabled"):
            # The evaluation parquet is only an input once evaluation is switched
            # on; keeping the path in the config for later must not fail a training
            # run. preflight owns the eval check and reports the missing file when
            # ``eval.enabled`` is true.
            continue
        value = config.get(key)
        if value and not Path(str(value)).exists():
            problems.append(f"{key}: path does not exist: {value}")
    return problems


def validate_config(config: Config) -> list[str]:
    """Collect every configuration problem (missing keys and bad values)."""
    from extension_config import validate as validate_extensions
    problems: list[str] = validate_extensions(config)
    for key in REQUIRED:
        value = config.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"missing required key: {key}")

    for stage in TRAIN_STAGES:
        value = config.get(f"budget.stage_epochs.{stage}")
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                problems.append(f"budget.stage_epochs.{stage} must be a positive integer")
            continue
        key = STAGE_STEPS_KEY[stage]
        try:
            steps = int(config.get(key) or 0)
        except (TypeError, ValueError):
            problems.append(f"{key} must be an integer")
            continue
        if steps <= 0:
            problems.append(f"{key} must be > 0 (got {config.get(key)!r})")

    warmup_epochs = config.get("budget.stage_epochs.warmup")
    gad_epochs = config.get("budget.stage_epochs.gad")
    if (config.get("budget.gad_new_steps") is None and isinstance(warmup_epochs, int) and isinstance(gad_epochs, int)
            and gad_epochs <= warmup_epochs):
        problems.append("GAD absolute epoch target must exceed the Warmup epoch budget")

    batch = config.get("data.train_batch_size")
    if batch is not None:
        try:
            batch = int(batch)
            if batch <= 0:
                problems.append("data.train_batch_size must be > 0")
        except (TypeError, ValueError):
            problems.append("data.train_batch_size must be an integer")

    n_gpus = int(config.get("resources.n_gpus_per_node") or 0)
    tp = int(config.get("rollout.tensor_model_parallel_size") or 0)
    if n_gpus <= 0:
        problems.append("resources.n_gpus_per_node must be > 0")
    if tp <= 0:
        problems.append("rollout.tensor_model_parallel_size must be > 0")
    elif n_gpus and tp > n_gpus:
        problems.append(f"rollout.tensor_model_parallel_size ({tp}) exceeds n_gpus_per_node ({n_gpus})")
    if n_gpus and tp and n_gpus % tp != 0:
        problems.append(f"n_gpus_per_node ({n_gpus}) must be divisible by tensor_model_parallel_size ({tp})")

    if batch and n_gpus:
        try:
            if int(batch) % n_gpus != 0:
                problems.append(
                    f"data.train_batch_size ({batch}) should be divisible by n_gpus_per_node ({n_gpus}) "
                    "for GRPO grouping"
                )
        except (TypeError, ValueError):
            pass

    keep_checkpoints = config.get("resources.keep_last_checkpoints")
    if keep_checkpoints is not None:
        if isinstance(keep_checkpoints, bool) or not isinstance(keep_checkpoints, int):
            problems.append("resources.keep_last_checkpoints must be an integer")
        elif keep_checkpoints < 1:
            problems.append("resources.keep_last_checkpoints must be >= 1")
    if config.get("resources.keep_last_epochs") is not None:
        problems.append("resources.keep_last_epochs was replaced by resources.keep_last_checkpoints")

    if config.get("validation.quick_enabled"):
        quick = config.get("validation.quick_file")
        if not quick or not Path(str(quick)).is_file():
            problems.append("validation.quick_enabled requires an existing validation.quick_file")
        if not config.get("validation.final_full"):
            problems.append("validation.quick_enabled requires validation.final_full=true")
        if int(config.get("validation.quick_size") or 0) < 1:
            problems.append("validation.quick_size must be >= 1")

    if float(config.get("optim.kl_loss_coef") or 0.0) < 0:
        problems.append("optim.kl_loss_coef must be >= 0")
    if int(config.get("optim.critic_warmup") or 0) < 0:
        problems.append("optim.critic_warmup must be >= 0")
    if float(config.get("rollout.temperature") or 0.0) <= 0:
        problems.append("rollout.temperature must be > 0 for on-policy sampling")
    # Reference _dapo_common.sh: STAGE_N_SEQKD/STAGE_N_WARMUP default to 1 (those stages keep
    # only the first candidate per uid anyway) and dapo_check_gad_rollout_n refuses GAD with
    # n<2, because GRPO normalises advantages inside a uid group.
    for key in ("rollout.n_seqkd", "rollout.n_warmup"):
        try:
            n = int(config.get(key) or 1)
        except (TypeError, ValueError):
            problems.append(f"{key} must be an integer")
            continue
        if n < 1:
            problems.append(f"{key} must be >= 1 (rollout.n for that stage)")
    try:
        gad_n = int(config.get("rollout.n") or 0)
    except (TypeError, ValueError):
        gad_n = 0
        problems.append("rollout.n must be an integer")
    if 0 < gad_n < 2:
        problems.append(
            f"rollout.n ({gad_n}) is the GAD group size and must be >= 2: GRPO normalises advantages "
            "inside a uid group, so n=1 leaves no advantage signal (the reference harness refuses it "
            "in dapo_check_gad_rollout_n). SeqKD/Warmup use rollout.n_seqkd/rollout.n_warmup instead."
        )
    if not 0 < float(config.get("rollout.gpu_memory_utilization") or 0) <= 1:
        problems.append("rollout.gpu_memory_utilization must be in (0, 1]")

    # R5: the discriminator batches statically by default (reference CRITIC_DYNAMIC_BSZ=False,
    # CRITIC_MICRO=1). A static micro size of 0 would produce empty micro-batches; the backend
    # additionally asserts that the normalized per-rank mini-batch is divisible by it
    # (fsdp_workers.py:882), so only the trivially-wrong value needs catching here.
    if not config.get("optim.critic_dynamic_bsz"):
        try:
            critic_micro = int(config.get("optim.critic_micro_batch_size_per_gpu") or 0)
        except (TypeError, ValueError):
            problems.append("optim.critic_micro_batch_size_per_gpu must be an integer")
        else:
            if critic_micro < 1:
                problems.append(
                    "optim.critic_micro_batch_size_per_gpu must be >= 1 when optim.critic_dynamic_bsz "
                    "is false: the static discriminator micro-batches would be empty. Use 1 to match "
                    "the reference harness (CRITIC_MICRO=1)."
                )
            elif batch and n_gpus > 0:
                for stage, rollout_n in (("warmup", config.get("rollout.n_warmup")),
                                         ("gad", config.get("rollout.n"))):
                    try:
                        sequences = int(batch) * int(rollout_n)
                    except (TypeError, ValueError):
                        continue  # rollout.n is checked separately above
                    if sequences % n_gpus:
                        problems.append(
                            f"{stage}: train_batch_size * rollout.n ({sequences}) must be divisible "
                            f"by n_gpus_per_node ({n_gpus}) for static critic batching"
                        )
                        continue
                    per_gpu = sequences // n_gpus
                    if per_gpu < critic_micro or per_gpu % critic_micro:
                        problems.append(
                            f"{stage}: normalized critic minibatch {per_gpu} per GPU must be "
                            f"a positive multiple of optim.critic_micro_batch_size_per_gpu={critic_micro}"
                        )

    # The backend asserts this pair (vllm_rollout_spmd.py: "disable CUDA graph
    # (enforce_eager = False) if free cache engine"), so catch it before GPU time.
    # Note: on the pinned vLLM (0.8.5) free_cache_engine itself is a no-op - the free
    # path is guarded by `vllm_version in ("0.5.4", "0.6.3")` - so all this flag really
    # buys is enforce_eager=True. It is not a memory lever; see PIPELINE.md 3.4/3.5.
    if config.get("rollout.free_cache_engine") and not config.get("rollout.enforce_eager"):
        problems.append(
            "rollout.free_cache_engine=True requires rollout.enforce_eager=True "
            "(the backend refuses CUDA graphs together with a freed KV cache). Note this flag is a "
            "no-op on vLLM >= 0.7 (vllm_rollout_spmd.py guards its free path on 0.5.4/0.6.3): set "
            "rollout.enforce_eager=True directly if dropping the CUDA-graph pool is the goal."
        )

    # The PyTorch OOM message suggests expandable_segments, but vLLM's CuMemAllocator
    # (active because the rollout engine uses enable_sleep_mode=True) asserts against it:
    #   assert "expandable_segments:True" not in conf  (vllm/device_allocator/cumem.py:146)
    # Reject it here so a copy-paste from the traceback fails in preflight, not on GPU.
    alloc_conf = str(config.get("run.pytorch_cuda_alloc_conf") or "")
    if "expandable_segments" in alloc_conf:
        problems.append(
            "run.pytorch_cuda_alloc_conf must not use expandable_segments: vLLM's "
            "CuMemAllocator refuses it (cumem.py:146). Use e.g. max_split_size_mb:128, or "
            "leave it empty."
        )

    try:
        length_budget = int(config.get("data.max_prompt_length") or 0) + int(
            config.get("data.max_response_length") or 0
        )
    except (TypeError, ValueError):
        length_budget = 0

    if length_budget:
        # rearrange_micro_batches asserts max_token_len >= the padded sequence length,
        # and that budget is what caps the per-micro-batch activation peak - the LM head
        # logits alone are tokens x vocab x dtype. Going below the length budget cannot
        # work, so fail here instead of inside the trainer. The discriminator's budget only
        # applies when it batches dynamically (R5: static by default).
        token_keys = ["optim.actor_ppo_max_token_len_per_gpu"]
        if config.get("optim.critic_dynamic_bsz"):
            token_keys.append("optim.critic_ppo_max_token_len_per_gpu")
        for key in token_keys:
            try:
                tokens = int(config.get(key) or 0)
            except (TypeError, ValueError):
                problems.append(f"{key} must be an integer")
                continue
            if tokens < length_budget:
                problems.append(
                    f"{key} ({tokens}) must be >= data.max_prompt_length + data.max_response_length "
                    f"({length_budget}): micro-batches are padded to that length and the backend "
                    "asserts max_token_len >= sequence length"
                )

        # vLLM refuses to start when chunked prefill is enabled and
        # max_num_batched_tokens < max_model_len, so catch it before the run starts.
        for key in ("rollout.max_model_len", "rollout.max_num_batched_tokens"):
            value = config.get(key)
            if value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                problems.append(f"{key} must be an integer")
                continue
            if number < length_budget:
                problems.append(
                    f"{key} ({number}) must be >= data.max_prompt_length + data.max_response_length "
                    f"({length_budget}), otherwise vLLM aborts with "
                    "'max_num_batched_tokens is smaller than max_model_len'"
                )
    try:
        if int(config.get("rollout.max_num_seqs") or 0) <= 0:
            problems.append("rollout.max_num_seqs must be > 0")
    except (TypeError, ValueError):
        problems.append("rollout.max_num_seqs must be an integer")

    # Nested mappings are flattened to dotted keys, so read the per-stage values directly.
    for stage in TRAIN_STAGES:
        for key in (f"budget.epochs_override.{stage}", f"budget.stage_epochs.{stage}"):
            value = config.get(key)
            if value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                problems.append(f"{key} must be an integer")
                continue
            if number <= 0:
                problems.append(f"{key} must be > 0")

    return problems
