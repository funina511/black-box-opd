#!/usr/bin/env python3
"""Three-stage GAD pipeline: SeqKD -> Warmup -> GAD, with real gates between them.

Design rules (plan sections 3.1 - 3.3 and 4):

* every stage runs as its own Python/Ray process with ``PYTHONPATH`` pinned to
  that stage's backend worktree, plus an import guard (``sitecustomize``) that
  aborts if ``verl`` is ever imported from anywhere else - in the driver **and**
  in the Ray workers that inherit the same environment;
* a stage is only marked complete after its artifacts have been verified; a
  marker is never written just because a directory exists;
* skipping an already-completed stage requires a matching config/input signature;
* GAD never loads the initial student: it consumes the merged Warmup actor and
  critic and resumes the Warmup global step, so ``gad_steps`` means *additional*
  steps on top of the Warmup step count;
* ``--dry-run`` prints the resolved config, the backends, the exact commands and
  the budgets without touching a GPU.

Examples::

    python tools/pipeline.py --config /abs/path/run.yaml --dry-run
    python tools/pipeline.py --config /abs/path/run.yaml --stages seqkd
    python tools/pipeline.py --config /abs/path/run.yaml --from-stage warmup --to-stage gad
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    TRAIN_STAGES,
    read_json,
    PipelineError,
    capture_command,
    ensure_dir,
    environment_summary,
    free_disk_gib,
    git_rev_parse,
    log,
    parse_key_values,
    run_command,
    run_id as make_run_id,
    sha256_file,
    sha256_text,
    utc_timestamp,
    write_json,
)
from config import (  # noqa: E402
    Config,
    load_config,
    validate_config,
    validate_paths,
    add_config_arguments, select_config, new_run_id, save_run_config,
)
from prune_checkpoints import RetentionMonitor  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
GUARD_DIR_NAME = "runtime/_guard"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def dataset_rows(path: str | Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(str(path)).metadata.num_rows


def steps_per_epoch(cfg: Config) -> int:
    rows = dataset_rows(cfg.require("data.train_file"))
    batch = int(cfg.require("data.train_batch_size"))
    return max(1, rows // batch)  # the train dataloader uses drop_last=True


def epochs_for(target_steps: int, steps_per_epoch_value: int, *, extra_epoch: int = 0) -> int:
    return max(1, math.ceil(target_steps / max(1, steps_per_epoch_value))) + extra_epoch


def stage_step_budget(cfg: Config, stage: str, steps_per_epoch_value: int,
                      *, handoff_step: int | None = None) -> int:
    """Steps a stage adds, honouring an epoch budget when the config sets one.

    The original scripts size every stage with ``trainer.total_epochs``, so
    ``budget.stage_epochs.<stage>`` is translated to steps with the local
    ``steps_per_epoch``. SeqKD/Warmup count from step 0; GAD continues Warmup's
    global step, so its epoch budget is an ABSOLUTE target (the original runs
    ``total_epochs=4`` on top of Warmup's checkpoint at step S).
    """
    if stage == "gad" and cfg.get("budget.gad_new_steps") is not None:
        return int(cfg.get("budget.gad_new_steps"))
    epoch_budget = cfg.stage_epochs(stage)
    if epoch_budget is None:
        return cfg.stage_steps(stage)
    target = epoch_budget * steps_per_epoch_value
    if stage != "gad":
        return target
    start_step = int(handoff_step or 0)
    added = target - start_step
    if added <= 0:
        raise PipelineError(
            f"budget.stage_epochs.gad={epoch_budget} is a {target}-step global target, but the Warmup "
            f"handoff is already at step {start_step}; the GAD epoch budget must exceed the Warmup one "
            "(the original uses 4 vs 1-2 epochs)"
        )
    return added



def backend_path(cfg: Config, stage: str) -> Path:
    return Path(cfg.require("backends.root")) / stage


def planned_backend_commit(backend: Path, stage: str) -> str:
    if backend.exists():
        return git_rev_parse(backend, "HEAD")
    from prepare_backends import PINNED_BACKENDS
    return PINNED_BACKENDS[stage]["commit"]


def stage_out_dir(cfg: Config, run_dir: Path, stage: str) -> Path:
    return run_dir / stage


def completion_marker(run_dir: Path, stage: str) -> Path:
    return run_dir / stage / "COMPLETED.json"


def write_guard(cfg: Config, run_dir: Path, stage: str, backend: Path) -> Path:
    """Install a PYTHONPATH guard that fails loudly on a stray ``verl`` import."""
    guard_dir = ensure_dir(run_dir / GUARD_DIR_NAME / stage)
    commit = git_rev_parse(backend, "HEAD")
    (guard_dir / "sitecustomize.py").write_text(
        '''"""Generated by tools/pipeline.py - refuse to import a foreign verl.

Runs at interpreter start-up for the driver process and for every Ray worker
that inherits PYTHONPATH, so a stage can never silently train with another
project's `verl` (for example an editable install pointing at a different
checkout).
"""
import importlib.machinery
import os
import sys

_EXPECTED = os.environ.get("GAD_EXPECT_VERL_PATH")
_COMMIT = os.environ.get("GAD_EXPECT_VERL_COMMIT", "unknown")
_STAGE = os.environ.get("GAD_BACKEND_STAGE", "unknown")


class _VerlBackendGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "verl":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin:
            return None
        origin = os.path.realpath(spec.origin)
        expected = os.path.realpath(_EXPECTED)
        if not origin.startswith(expected + os.sep):
            raise RuntimeError(
                f"[gad-guard] stage {_STAGE} must import verl from {expected} "
                f"(commit {_COMMIT}) but resolved {origin}"
            )
        sys.stderr.write(f"[gad-guard] stage={_STAGE} commit={_COMMIT} verl={origin}\\n")
        return None


if _EXPECTED:
    sys.meta_path.insert(0, _VerlBackendGuard())
''',
        encoding="utf-8",
    )
    return guard_dir


def stage_env(cfg: Config, run_dir: Path, stage: str, backend: Path, *, log_backend: str,
              install_guard: bool = True) -> dict:
    guard_dir = (write_guard(cfg, run_dir, stage, backend) if install_guard
                 else run_dir / GUARD_DIR_NAME / stage)
    pythonpath = os.pathsep.join([str(guard_dir), str(backend)])
    env = {
        "PYTHONPATH": pythonpath,
        "GAD_EXPECT_VERL_PATH": str(backend),
        "GAD_EXPECT_VERL_COMMIT": planned_backend_commit(backend, stage),
        "GAD_BACKEND_STAGE": stage,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "true",
        "HYDRA_FULL_ERROR": "1",
        "NCCL_TIMEOUT": "36000",
        "WANDB_INIT_TIMEOUT": "600",
    }
    # Optional allocator tuning. Empty by default: vLLM's CuMemAllocator (used because the
    # engine runs with enable_sleep_mode=True) refuses expandable_segments outright
    # (vllm/device_allocator/cumem.py:146), so that hint from a PyTorch OOM message must
    # NOT be applied here - validate_config rejects it with an explanation.
    alloc_conf = str(cfg.get("run.pytorch_cuda_alloc_conf") or "")
    if alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    scorer = cfg.get("validation.scorer_path")
    if scorer:
        # Consumed by the gad-math-validation backend patch: every branch then
        # reports dev-set math accuracy next to the original ROUGE-L metric.
        env["GAD_MATH_SCORER"] = str(scorer)
        if cfg.get('scoring.math.version') == 'math_unified_v1':
            external = os.environ.get('GAD_MATH_SCORER')
            if external and Path(external).resolve() != Path(scorer).resolve():
                raise PipelineError('GAD_MATH_SCORER conflicts with unified scoring.math')
            env['GAD_SCORING_MATH_CONFIG'] = json.dumps({key[13:]: value for key, value in cfg.data.items() if key.startswith('scoring.math.')})
    if cfg.get("validation.quick_enabled"):
        from hashlib import sha256
        quick = Path(str(cfg.require("validation.quick_file")))
        full = Path(str(cfg.require("data.val_file")))
        env["GAD_QUICK_DEV"] = str(quick)
        env["GAD_QUICK_SHA256"] = sha256(quick.read_bytes()).hexdigest()
        env["GAD_FULL_SHA256"] = sha256(full.read_bytes()).hexdigest()
    gpus = cfg.get("resources.gpus")
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    if "wandb" in str(log_backend):
        project = cfg.get("run.wandb_project")
        if project:
            env["WANDB_PROJECT"] = str(project)
    return env


def validation_metrics_expected(cfg: Config) -> tuple[bool, str]:
    """Whether a stage must produce the dev-set math metric artifact."""
    if cfg.get('run.disable_validation'):
        return False, 'run.disable_validation is true'
    if not cfg.get("validation.require_math_metrics"):
        return False, "validation.require_math_metrics is false"
    if not cfg.get("validation.scorer_path"):
        return False, "no math scorer is configured"
    if not cfg.get("run.val_before_train") and int(cfg.get("resources.test_freq") or 0) <= 0:
        return False, "validation never runs (val_before_train=false and test_freq<=0)"
    return True, "dev-set math accuracy must be reported by the stage"


# --------------------------------------------------------------------------- #
# stage command construction
# --------------------------------------------------------------------------- #
def stage_rollout_n(cfg: Config, stage: str) -> int:
    """``rollout.n`` for one stage, mirroring the reference's STAGE_N_* split.

    The reference harness runs SeqKD/Warmup with ``n=1`` and refuses GAD with ``n<2``
    (``_dapo_common.sh``: ``STAGE_N_SEQKD``/``STAGE_N_WARMUP`` default to 1, and
    ``dapo_check_gad_rollout_n`` errors out because "GRPO 需要组内多条候选才有优势信号").
    SeqKD/Warmup deduplicate by uid before the actor update
    (``dp_actor.deduplicate_by_uid`` keeps the first candidate), so extra candidates there
    are discarded by the actor, though the discriminator may still use them.
    """
    if stage == "seqkd":
        return int(cfg.get("rollout.n_seqkd") or 1)
    if stage == "warmup":
        return int(cfg.get("rollout.n_warmup") or 1)
    return int(cfg.require("rollout.n"))


def build_overrides(cfg: Config, stage: str, *, model_path: str, critic_path: str | None,
                    out_dir: Path, total_steps: int, epochs: int) -> list[str]:
    batch = int(cfg.require("data.train_batch_size"))
    val_batch = cfg.get("data.val_batch_size")
    prompt_len = int(cfg.get("data.max_prompt_length"))
    response_len = int(cfg.get("data.max_response_length"))
    length_budget = prompt_len + response_len
    # vLLM aborts at engine construction when chunked prefill is on and
    # max_num_batched_tokens < max_model_len (its defaults are 8192 and
    # prompt+response respectively), so both are always passed explicitly.
    max_model_len = int(cfg.get("rollout.max_model_len") or length_budget)
    max_num_batched_tokens = int(cfg.get("rollout.max_num_batched_tokens") or length_budget)
    keep_checkpoints = cfg.get("resources.keep_last_checkpoints")
    save_freq = int(cfg.get("resources.save_freq"))
    overrides = [
        "algorithm.adv_estimator=grpo",
        f"data.prompt_key={cfg.get('data.prompt_key')}",
        f"data.train_files={cfg.require('data.train_file')}",
        f"data.val_files={cfg.require('data.val_file')}",
        f"data.train_batch_size={batch}",
        f"data.max_prompt_length={int(cfg.get('data.max_prompt_length'))}",
        f"data.max_response_length={int(cfg.get('data.max_response_length'))}",
        f"data.truncation={cfg.get('data.truncation')}",
        f"actor_rollout_ref.model.path={model_path}",
        f"actor_rollout_ref.model.use_remove_padding={bool(cfg.get('models.use_remove_padding'))}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={bool(cfg.get('models.enable_gradient_checkpointing'))}",
        f"actor_rollout_ref.model.trust_remote_code={bool(cfg.get('models.trust_remote_code'))}",
        f"actor_rollout_ref.actor.optim.lr={cfg.stage_lr(stage)}",
        f"actor_rollout_ref.actor.grad_clip={cfg.get('optim.grad_clip')}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={batch}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={int(cfg.get('optim.actor_ppo_max_token_len_per_gpu'))}",
        f"actor_rollout_ref.actor.entropy_coeff={cfg.get('optim.entropy_coeff')}",
        f"actor_rollout_ref.actor.kl_loss_type={cfg.get('optim.kl_loss_type')}",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={bool(cfg.get('models.param_offload'))}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={bool(cfg.get('models.optimizer_offload'))}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={bool(cfg.get('models.ref_param_offload'))}",
        f"actor_rollout_ref.rollout.name={cfg.get('rollout.name')}",
        f"actor_rollout_ref.rollout.temperature={cfg.get('rollout.temperature')}",
        f"actor_rollout_ref.rollout.top_p={cfg.get('rollout.top_p')}",
        f"actor_rollout_ref.rollout.n={stage_rollout_n(cfg, stage)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={int(cfg.get('rollout.tensor_model_parallel_size'))}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg.get('rollout.gpu_memory_utilization')}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={int(cfg.get('rollout.max_num_seqs'))}",
        f"actor_rollout_ref.rollout.enforce_eager={bool(cfg.get('rollout.enforce_eager'))}",
        f"actor_rollout_ref.rollout.free_cache_engine={bool(cfg.get('rollout.free_cache_engine'))}",
        f"actor_rollout_ref.rollout.val_kwargs.n={int(cfg.get('rollout.val_n'))}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={bool(cfg.get('rollout.val_do_sample'))}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={cfg.get('rollout.val_temperature')}",
        f"trainer.critic_warmup={int(cfg.get('optim.critic_warmup')) if stage != 'gad' else 0}",
        f"trainer.val_before_train={bool(cfg.get('run.val_before_train'))}",
        f"trainer.validation_rouge_l_enabled={bool(cfg.get('validation.rouge_l_enabled'))}",
        f"trainer.logger=[{cfg.get('run.log_backend')}]",
        "trainer.default_hdfs_dir=null",
        f"trainer.n_gpus_per_node={int(cfg.get('resources.n_gpus_per_node'))}",
        f"trainer.nnodes={int(cfg.get('resources.nnodes'))}",
        f"trainer.save_freq={save_freq}",
        f"trainer.test_freq={int(cfg.get('resources.test_freq'))}",
        f"trainer.total_epochs={epochs}",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.default_local_dir={out_dir}",
        f"trainer.project_name={cfg.get('run.wandb_project') or 'gad'}",
        f"trainer.experiment_name={stage}",
        f"hydra.run.dir={out_dir.parent / 'runtime' / 'hydra' / stage}",
    ]
    backend = backend_path(cfg, stage)
    backend_config = backend / "verl/trainer/config/ppo_trainer.yaml"
    if (backend / ".gad_backend_used").exists() and "validation_rouge_l_enabled" not in backend_config.read_text():
        overrides = [a for a in overrides if not a.startswith("trainer.validation_rouge_l_enabled=")]
    # RetentionMonitor rotates whole checkpoints and can preserve exported models.
    # Trainer rotation removes only actor/critic subdirectories and ignores pins.
    if keep_checkpoints is None:
        if cfg.get("resources.max_actor_ckpt_to_keep"):
            overrides.append(f"trainer.max_actor_ckpt_to_keep={int(cfg.get('resources.max_actor_ckpt_to_keep'))}")
        if cfg.get("resources.max_critic_ckpt_to_keep"):
            overrides.append(f"trainer.max_critic_ckpt_to_keep={int(cfg.get('resources.max_critic_ckpt_to_keep'))}")
    if val_batch:
        overrides.append(f"data.val_batch_size={int(val_batch)}")

    # thinking switch: forwarded by the gad-chat-template-kwargs backend patch.
    if cfg.get("rollout.thinking_control"):
        overrides.append(
            f"+data.apply_chat_template_kwargs.enable_thinking="
            f"{'true' if cfg.get('rollout.enable_thinking') else 'false'}"
        )

    if stage == "seqkd":
        lr = cfg.stage_lr(stage)
        del lr
        overrides += [
            "actor_rollout_ref.actor.use_kl_loss=False",
            "actor_rollout_ref.actor.kl_loss_coef=0.0",
            "algorithm.kl_ctrl.kl_coef=0.0",
        ]
    else:
        overrides += [
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={cfg.get('optim.kl_loss_coef')}",
            f"algorithm.kl_ctrl.kl_coef={cfg.get('optim.kl_loss_coef')}",
        ]

    if stage in ("warmup", "gad"):
        if not critic_path:
            raise PipelineError(f"stage {stage} needs a discriminator (critic) initialisation path")
        overrides += [
            f"critic.model.path={critic_path}",
            f"critic.optim.lr={cfg.get('optim.warmup_critic_lr') if stage == 'warmup' else cfg.get('optim.gad_critic_lr')}",
            f"critic.model.use_remove_padding={bool(cfg.get('models.use_remove_padding'))}",
            f"critic.model.trust_remote_code={bool(cfg.get('models.trust_remote_code'))}",
            f"critic.grad_clip={cfg.get('optim.grad_clip')}",
            f"critic.model.fsdp_config.param_offload={bool(cfg.get('models.param_offload'))}",
            f"critic.model.fsdp_config.optimizer_offload={bool(cfg.get('models.optimizer_offload'))}",
        ]
        # R5 (reference _dapo_common.sh: CRITIC_DYNAMIC_BSZ=False, CRITIC_MICRO=1): the
        # discriminator's batching is decoupled from the actor and is static by default, so
        # its micro-batches come from ``batch.split(micro)`` and never from the token-budget
        # partitioner. That keeps the discriminator bit-identical to the pinned code.
        if bool(cfg.get("optim.critic_dynamic_bsz")):
            overrides += [
                "critic.use_dynamic_bsz=True",
                f"critic.ppo_max_token_len_per_gpu={int(cfg.get('optim.critic_ppo_max_token_len_per_gpu'))}",
            ]
        else:
            overrides += [
                "critic.use_dynamic_bsz=False",
                f"critic.ppo_micro_batch_size_per_gpu={int(cfg.get('optim.critic_micro_batch_size_per_gpu'))}",
            ]
        if stage == "gad":
            overrides.append("trainer.resume_mode=auto")
    from extension_config import active
    sampling_seed = cfg.get("rollout.seed")
    if sampling_seed is not None:
        prefix = "" if cfg.get("backends.profile") == "gad_extensions_v1" and stage != "seqkd" else "+"
        overrides.append(f"{prefix}actor_rollout_ref.rollout.seed={int(sampling_seed)}")
    if cfg.get('run.disable_validation'):
        overrides.append('trainer.val_before_train=False')
        overrides.append('trainer.test_freq=0')
    if cfg.get('scoring.math.version') == 'math_unified_v1' and stage in ('warmup', 'gad'):
        overrides += [f'++data.seed={int(cfg.get("run.seed"))}',
                      f'++trainer.seed={int(cfg.get("run.seed"))}',
                      f'++trainer.disable_validation={str(cfg.get("run.disable_validation")).lower()}']
    return overrides


def plan_stage(cfg: Config, run_dir: Path, stage: str, *, steps_per_epoch_value: int,
               handoff: dict | None = None, resume: dict | None = None, install_guard: bool = True) -> dict:
    backend = backend_path(cfg, stage)
    out_dir = stage_out_dir(cfg, run_dir, stage)
    python_bin = cfg.require("run.python_bin")

    if stage == "gad":
        if not handoff:
            raise PipelineError(
                "the GAD stage needs the Warmup handoff (merged actor+critic and the resume step). "
                "Run the warmup stage first or pass an explicit handoff."
            )
        model_path = handoff["actor_hf"]
        critic_path = handoff["critic_hf"]
        start_step = int(handoff["step"])
        added_steps = stage_step_budget(cfg, stage, steps_per_epoch_value, handoff_step=start_step)
        total_steps = start_step + added_steps
        epochs = epochs_for(total_steps, steps_per_epoch_value, extra_epoch=1)
    else:
        model_path = cfg.require("models.student")
        critic_path = None
        if stage == "warmup":
            critic_path = cfg.get("models.discriminator_init") or cfg.require("models.student")
        start_step = 0
        added_steps = stage_step_budget(cfg, stage, steps_per_epoch_value)
        total_steps = added_steps
        epochs = epochs_for(total_steps, steps_per_epoch_value)
    epoch_budget = cfg.stage_epochs(stage)

    override = cfg.get(f"budget.epochs_override.{stage}")
    if override:
        epochs = int(override)
        if epochs * steps_per_epoch_value < (total_steps - start_step):
            raise PipelineError(
                f"budget.epochs_override.{stage}={epochs} gives only {epochs * steps_per_epoch_value} steps, "
                f"which cannot reach the requested {total_steps - start_step} additional steps"
            )

    overrides = build_overrides(
        cfg, stage, model_path=model_path, critic_path=critic_path,
        out_dir=out_dir, total_steps=total_steps, epochs=epochs,
    )
    from extension_config import active, hydra_overrides
    if active(cfg, "rubric", stage):
        config_path = Path(critic_path) / "config.json"
        source_four = config_path.is_file() and json.loads(config_path.read_text()).get("rubric_head_layout")
        assumed_four = stage == "gad" and active(cfg, "rubric", "warmup") and not cfg.get("budget.init_from_warmup_run")
        if not source_four and not (not install_guard and assumed_four):
            prepared = run_dir / "initialization" / f"critic_{stage}"
            if install_guard:
                prepare_argv = [str(python_bin), str(PROJECT_DIR / "tools/prepare_rubric_critic.py"),
                                "--source", str(critic_path), "--target", str(prepared),
                                "--spec", str(cfg.require("rubric.spec")), "--seed", str(cfg.get("rubric.seed"))]
                run_command(prepare_argv, component="critic-initialization")
            critic_path = str(prepared)
            overrides = build_overrides(cfg, stage, model_path=model_path, critic_path=critic_path,
                                        out_dir=out_dir, total_steps=total_steps, epochs=epochs)
    overrides += hydra_overrides(cfg, stage, out_dir, start_step)
    if stage in ("warmup", "gad") and cfg.get("run.resume_worker_state"):
        # Same-stage full-state resume (requires the gad-resume-worker-state patch).
        overrides.append("trainer.resume_load_worker_state=True")
    argv = [str(python_bin), "-m", "verl.trainer.main_ppo", *overrides]
    env = stage_env(cfg, run_dir, stage, backend, log_backend=str(cfg.get("run.log_backend")),
                    install_guard=install_guard)
    return {
        "stage": stage,
        "backend": str(backend),
        "backend_commit": planned_backend_commit(backend, stage),
        "backend_profile": cfg.get("backends.profile"),
        "rubric_enabled": active(cfg, "rubric", stage),
        "replay_enabled": active(cfg, "replay", stage),
        "validation_diagnostics": {
            "configured_rouge_l_enabled": bool(cfg.get("validation.rouge_l_enabled")),
            "effective_rouge_l_enabled": (bool(cfg.get("validation.rouge_l_enabled"))
                if any(a.startswith("trainer.validation_rouge_l_enabled=") for a in overrides) else True),
            "block_training_on_incomplete": bool(cfg.get("validation.block_training_on_incomplete")),
        },
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "critic_path": str(critic_path) if critic_path else None,
        "start_step": start_step,
        "added_steps": added_steps,
        "total_steps": total_steps,
        "epochs": epochs,
        "epoch_budget": epoch_budget,
        "steps_per_epoch": steps_per_epoch_value,
        "resume": resume or {"requested": False, "mode": "fresh", "from_step": None},
        "argv": argv,
        "env": env,
        "log": str(run_dir / "logs" / f"{stage}.log"),
    }


# --------------------------------------------------------------------------- #
# artifact verification
# --------------------------------------------------------------------------- #
def checkpoint_dirs(out_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for child in out_dir.glob("global_step_*"):
        suffix = child.name.split("global_step_")[-1]
        if suffix.isdigit():
            found.append((int(suffix), child))
    return sorted(found)


def verify_checkpoint_dir(path: Path, *, roles: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    if not path.is_dir():
        return [f"checkpoint directory missing: {path}"]
    for role in roles:
        role_dir = path / role
        if not role_dir.is_dir():
            problems.append(f"{role_dir}: missing")
            continue
        shards = sorted(role_dir.glob("model_world_size_*_rank_*.pt"))
        if not shards:
            problems.append(f"{role_dir}: no model shards (model_world_size_*_rank_*.pt)")
        try:
            metadata = json.loads((role_dir / "config.json").read_text())
            if not isinstance(metadata, dict): raise ValueError("not a JSON object")
        except (OSError, ValueError) as exc:
            problems.append(f"{role_dir}: config.json missing/corrupt ({exc})")
        optimizers = sorted(role_dir.glob("optim_world_size_*_rank_*.pt"))
        if not optimizers:
            problems.append(f"{role_dir}: no optimizer shards (full-state resume would be incomplete)")
        for shard in shards + optimizers:
            if not shard.stat().st_size:
                problems.append(f"{shard}: empty/corrupt shard")
    if not (path / "data.pt").exists():
        problems.append(f"{path}/data.pt missing (dataloader state; step resume would restart the epoch)")
    from gad.extensions.checkpoint_state import checkpoint_problems
    problems.extend(checkpoint_problems(path))
    return problems


def verify_stage_artifacts(stage: str, out_dir: Path, *, expected_step: int | None,
                           math_metrics_expected: bool = False,
                           math_metrics_reason: str = "",
                           require_final_full: bool = False, full_count: int | None = None,
                           strict_evaluation: bool = False) -> dict:
    roles = ("actor",) if stage == "seqkd" else ("actor", "critic")
    steps = checkpoint_dirs(out_dir)
    last_step, last_dir = steps[-1] if steps else (None, None)
    checkpoint_problems = verify_checkpoint_dir(last_dir, roles=roles) if last_dir else [f"{out_dir}: no global_step_* checkpoint was written"]
    training_problems = list(checkpoint_problems)
    if expected_step is not None and last_step != expected_step:
        training_problems.append(f"last checkpoint is global_step_{last_step} but the budget requires global_step_{expected_step}")
    latest_file = out_dir / "latest_checkpointed_iteration.txt"
    try:
        recorded = latest_file.read_text(encoding="utf-8").strip()
        if recorded != str(last_step):
            training_problems.append(f"{latest_file}: says {recorded!r} but the newest checkpoint is {last_step}")
    except (OSError, UnicodeError) as exc:
        training_problems.append(f"{latest_file}: missing/unreadable ({type(exc).__name__})")
    evaluation_problems = []
    evaluation_status = "not_requested"
    math = {"expected": bool(math_metrics_expected), "reason": math_metrics_reason}
    if math_metrics_expected:
        artifact = out_dir / "validation_math_metrics.jsonl"
        records = []
        evaluation_status = "complete"
        try:
            for number, line in enumerate(artifact.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip(): continue
                row = json.loads(line)
                if not isinstance(row, dict): raise ValueError(f"line {number} is not an object")
                records.append(row)
        except FileNotFoundError:
            evaluation_status = "missing"
            evaluation_problems.append(f"{artifact}: missing; validation was not reported")
        except (OSError, UnicodeError, ValueError) as exc:
            evaluation_status = "failed"
            evaluation_problems.append(f"{artifact}: unparsable/unreadable ({exc})")
        if not records and not evaluation_problems:
            evaluation_status = "missing"
            evaluation_problems.append(f"{artifact}: empty; validation was not reported")
        if records:
            last = records[-1]
            math.update(records=len(records), steps=[r.get("global_step") for r in records], last=last)
            if last.get("evaluation_complete", True) is not True:
                evaluation_problems.append("last evaluation is incomplete (unscorable/missing answers)")
            for key in ("grader_error_rate", "unscorable_rate"):
                rate = last.get(key, 0)
                if not isinstance(rate, (int, float)) or not 0 <= rate <= 1 or rate > 0:
                    evaluation_problems.append(f"{key}={rate!r}; official accuracy is unavailable")
                    if key == "grader_error_rate": evaluation_status = "failed"
            acc = last.get("acc")
            if not isinstance(acc, (int, float)) or not 0 <= acc <= 1 or not last.get("count"):
                evaluation_problems.append("last record has no usable official accuracy (acc may be null)")
            if require_final_full and (last.get("validation_scope") != "full" or last.get("global_step") != last_step or last.get("count") != full_count):
                evaluation_problems.append(f"final validation must be full at global_step_{last_step} with {full_count} rows; got {last.get('validation_scope')}, step {last.get('global_step')}, count {last.get('count')}")
            if evaluation_problems and evaluation_status == "complete": evaluation_status = "incomplete"
    ready = not training_problems and (not strict_evaluation or not evaluation_problems)
    return {
        "schema_version": "gad.stage_verification.v2", "ok": ready, "handoff_ready": ready,
        "training_status": "artifacts_ready" if not training_problems else "incomplete",
        "checkpoint_status": "valid" if not checkpoint_problems else "invalid",
        "training_problems": training_problems, "evaluation_status": evaluation_status,
        "evaluation_problems": evaluation_problems, "strict_evaluation": strict_evaluation,
        "problems": training_problems + (evaluation_problems if strict_evaluation else []),
        "last_step": last_step, "math_metrics": math,
        "checkpoints": [{"step": step, "path": str(path)} for step, path in steps],
    }


def last_complete_checkpoint(out_dir: Path, *, stage: str) -> tuple[int, Path] | None:
    """Highest checkpoint of ``stage`` that is structurally complete."""
    roles = ("actor",) if stage == "seqkd" else ("actor", "critic")
    for step, path in reversed(checkpoint_dirs(out_dir)):
        if not verify_checkpoint_dir(path, roles=roles):
            return step, path
    return None


def recorded_handoff(run_dir: Path, cfg: Config) -> dict | None:
    """The Warmup -> GAD handoff a previous run of this run dir used."""
    manifest = read_json(run_dir / "handoff_manifest.json", default=None)
    if manifest and manifest.get("actor_hf") and manifest.get("critic_hf"):
        return {
            "actor_hf": manifest["actor_hf"],
            "critic_hf": manifest["critic_hf"],
            "step": int(manifest["step"]),
            "source": str(run_dir / "handoff_manifest.json"),
        }
    stage_json = read_json(run_dir / "stage_gad.json", default=None)
    plan = (stage_json or {}).get("plan") or {}
    if plan.get("model_path") and plan.get("critic_path"):
        return {
            "actor_hf": plan["model_path"],
            "critic_hf": plan["critic_path"],
            "step": int(plan["start_step"]),
            "source": str(run_dir / "stage_gad.json"),
        }
    return None


def resolve_gad_resume(cfg: Config, run_dir: Path, *, added_steps: int,
                       resume_requested: bool) -> dict:
    """Decide how the GAD stage starts, so a resume never re-runs the handoff.

    ``tools/prepare_gad_checkpoint.py`` refuses to write into a non-empty GAD
    directory, so a crashed run could not be resumed: the pipeline re-ran the
    Warmup -> GAD handoff before the trainer ever looked at the checkpoint. The
    three cases are now explicit:

    * ``fresh``            - no complete GAD checkpoint: run the normal handoff;
    * ``resume``           - a complete checkpoint below the target: reuse the
      recorded handoff for the model path and let the trainer resume in place;
    * ``resume-complete``  - the run already reached ``S + K``: verify the
      artifacts instead of training one step past the budget.
    """
    out_dir = stage_out_dir(cfg, run_dir, "gad")
    decision = {
        "requested": bool(resume_requested),
        "mode": "fresh",
        "from_step": None,
        "total_steps": None,
        "handoff": None,
        "reason": "no resume requested",
    }
    existing = last_complete_checkpoint(out_dir, stage="gad") if out_dir.is_dir() else None
    if not resume_requested:
        if existing is not None:
            # Do not let the handoff fail with "refusing to overwrite": say what
            # happened and what to do about it.
            step, path = existing
            decision["conflict"] = {"step": step, "path": str(path)}
            decision["reason"] = (
                f"{out_dir} already holds a complete checkpoint (global_step_{step}) but no resume was "
                "requested. Pass --resume-gad to continue that run in place, or start over in a fresh "
                "run directory (--run-id <new-id>)."
            )
        return decision

    if existing is None:
        decision["reason"] = (
            f"--resume-gad requested but {out_dir} has no complete checkpoint; "
            "running the normal Warmup -> GAD handoff"
        )
        log("WARNING: " + decision["reason"])
        return decision

    step, path = existing
    handoff = recorded_handoff(run_dir, cfg)
    if handoff is None:
        raise PipelineError(
            f"--resume-gad found global_step_{step} under {out_dir} but neither handoff_manifest.json nor "
            "stage_gad.json records the merged Warmup actor/critic paths. Those paths initialise the model "
            "architecture, so the stage cannot be resumed without them."
        )
    total_steps = int(handoff["step"]) + int(added_steps)
    decision.update(
        {
            "mode": "resume",
            "from_step": step,
            "total_steps": total_steps,
            "handoff": handoff,
            "resume_checkpoint": str(path),
            "reason": f"resuming from global_step_{step} (target global_step_{total_steps})",
        }
    )
    if step >= total_steps:
        decision["mode"] = "resume-complete"
        decision["reason"] = (
            f"the GAD run already reached global_step_{step} (target global_step_{total_steps}); "
            "the trainer is not started again"
        )
    return decision


# --------------------------------------------------------------------------- #
# signatures and manifests
# --------------------------------------------------------------------------- #
def _stage_signature(cfg: Config, stage: str, *, legacy: bool) -> dict:
    """Build a stage training identity; legacy reproduces pre-performance markers."""
    # Completion identity tracks training inputs. Validation cadence and relocatable
    # paths are recorded elsewhere, so they cannot trigger a second training run.
    cfg = cfg.for_stage(stage)
    excluded = {"run.output_root", "run.python_bin", "run.run_id", "run.val_before_train",
                "run.log_backend", "run.eval_after_train", "run.skip_completed_stages",
                "resources.test_freq", "resources.save_freq", "resources.keep_last_checkpoints",
                "resources.keep_last_epochs", "data.val_file",
                "rollout.val_n", "rollout.val_do_sample", "rollout.val_temperature",
                "run.resume_worker_state",
                "data.val_batch_size", "data.teacher_index", "data.manifest",
                "backends.source_repo", "backends.root", "validation.scorer_path"}
    excluded.add("run.max_stage_steps")
    from extension_config import EXTENSION_DEFAULTS, extension_enabled
    if not extension_enabled(cfg):
        excluded.update(key for key, default in EXTENSION_DEFAULTS.items() if cfg.get(key) == default)
        excluded.update(key for key in cfg.data if key.startswith(("rubric.", "replay.", "resources.judge_")))
    if stage == "seqkd" and not legacy:
        # SeqKD updates the actor on teacher responses; it does not generate student
        # rollouts or train a critic. These knobs only affect Warmup/GAD training.
        excluded.update({"rollout.gpu_memory_utilization", "optim.critic_dynamic_bsz",
                         "optim.critic_micro_batch_size_per_gpu"})
    training = {k: v for k, v in (cfg.identity_data or cfg.to_dict()).items()
                if k not in excluded and not k.startswith("validation.") and not k.startswith("eval.")}
    for key in ("data.train_file", "models.student", "models.discriminator_init"):
        if key in training and training[key]:
            training[key] = str(Path(str(training[key])).resolve())
    payload = {"stage": stage, "training_config": training,
               "backend_commit": git_rev_parse(backend_path(cfg, stage), "HEAD") if backend_path(cfg, stage).exists() else None}
    if cfg.get("backends.profile") == "gad_extensions_v1" and stage != "seqkd":
        from extension_config import deployed_code_identity
        deployed = deployed_code_identity(cfg,stage)
        payload["algorithm"] = {"extension_code_hash": deployed["extension_code_hash"],
                                "patch_generator_hash": deployed["patch_generator_hash"]}
        if cfg.get("rubric.enabled"):
            payload["algorithm"]["spec_hash"] = sha256_file(Path(cfg.require("rubric.spec")))
            payload["algorithm"]["prompt_hash"] = sha256_file(Path(cfg.require("rubric.prompt")))
            if cfg.get("rubric.mode") == "joint":
                from extension_config import service_identity
                from rubric.core.contracts import digest
                payload["algorithm"]["judge_fingerprint"] = digest(service_identity(cfg))
    if stage in ("warmup", "gad"):
        payload["discriminator_init"] = cfg.get("models.discriminator_init")
    if stage == "gad":
        payload["warmup_manifest"] = None  # filled in by the caller once the handoff exists
    return {"signature": sha256_text(json.dumps(payload, sort_keys=True, default=str)), "payload": payload}


def stage_signature(cfg: Config, stage: str) -> dict:
    """Inputs that must match before a completed stage may be skipped."""
    return _stage_signature(cfg, stage, legacy=False)


def legacy_stage_signature(cfg: Config, stage: str) -> dict:
    """Reproduce old signatures only when checking an existing completion marker."""
    return _stage_signature(cfg, stage, legacy=True)


def dependency_versions(cfg: Config) -> dict:
    """Record the versions the training interpreter will actually use."""
    code = (
        "import importlib, json;"
        "mods = ['torch','ray','vllm','transformers','datasets','pyarrow','hydra','omegaconf','rouge_score'];"
        "out = {};"
        "\nfor name in mods:\n"
        "    try:\n"
        "        module = importlib.import_module(name); out[name] = getattr(module, '__version__', 'unknown')\n"
        "    except Exception as exc:\n"
        "        out[name] = f'unavailable: {type(exc).__name__}'\n"
        "print('@@GAD_VERSIONS@@' + json.dumps(out))"
    )
    probe_env = {"VLLM_LOGGING_LEVEL": "ERROR", "TRANSFORMERS_VERBOSITY": "error"}
    code_out, out, err = capture_command([str(cfg.require("run.python_bin")), "-c", code], env=probe_env)
    if code_out != 0:
        return {"error": f"could not query dependency versions: {err[:200]}"}
    # Importing vLLM prints platform banners on stdout, so the payload is tagged.
    marker = "@@GAD_VERSIONS@@"
    for line in out.splitlines():
        if marker in line:
            try:
                return json.loads(line.split(marker, 1)[1])
            except json.JSONDecodeError:
                break
    return {"error": out[:200]}


def data_provenance(cfg: Config) -> dict:
    """Hashes and counts for every data artifact the run consumed."""
    provenance: dict = {}
    for key in ("data.train_file", "data.val_file", "data.teacher_index"):
        value = cfg.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.exists():
            provenance[key] = {"path": str(path), "exists": False}
            continue
        entry = {"path": str(path), "exists": True, "bytes": path.stat().st_size}
        if path.stat().st_size < 2 * 1024 ** 3:
            entry["sha256"] = sha256_file(path)
        provenance[key] = entry
    try:
        provenance["train_rows"] = dataset_rows(cfg.require("data.train_file"))
        provenance["val_rows"] = dataset_rows(cfg.require("data.val_file"))
    except Exception as exc:  # noqa: BLE001
        provenance["row_count_error"] = str(exc)
    manifest_path = cfg.get("data.manifest")
    if manifest_path and Path(str(manifest_path)).exists():
        provenance["sample_manifest"] = read_json(manifest_path)
    return provenance


def stage_inputs_digest(cfg: Config, stage: str) -> dict:
    cfg = cfg.for_stage(stage)
    digest = {
        "train_file_sha256": sha256_file(Path(str(cfg.require("data.train_file")))),
        "val_file_sha256": sha256_file(Path(str(cfg.require("data.val_file")))),
        "train_rows": dataset_rows(cfg.require("data.train_file")),
    }
    digest["config_signature"] = stage_signature(cfg, stage)["signature"]
    return digest


def _legacy_seqkd_marker_matches(cfg: Config, run_dir: Path, recorded_signature: str) -> bool:
    """Accept an old SeqKD marker only when a saved config proves equivalence."""
    import yaml

    snapshots = [run_dir / "resolved_config.yaml"]
    snapshots += sorted((run_dir / "reports/resolved_configs").glob("*.yaml"))
    current = stage_signature(cfg, "seqkd")["signature"]
    for path in snapshots:
        if not path.is_file():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            previous = Config(data, path=path)
            if (legacy_stage_signature(previous, "seqkd")["signature"] == recorded_signature
                    and stage_signature(previous, "seqkd")["signature"] == current):
                return True
        except (OSError, ValueError, PipelineError, yaml.YAMLError):
            continue
    return False


def recover_completed_stage(cfg: Config, run_dir: Path, stage: str) -> tuple[bool, str]:
    """Read-only recovery: never reinterpret checkpoint existence as exit=0."""
    import copy
    history = read_json(run_dir / f"stage_{stage}.json", default=None)
    if not history or history.get("returncode") != 0 or history.get("trainer_skipped"):
        return False, "no successful training exit record"
    previous = history.get("plan") or {}
    if not previous.get("argv"):
        return False, "historical launch argv is missing"
    frozen = copy.deepcopy(cfg)
    backend = backend_path(cfg, stage)
    if str(backend) != previous.get("backend"):
        return False, "historical backend identity differs"
    try:
        if previous.get("backend_commit") != git_rev_parse(backend, "HEAD"):
            return False, "historical backend commit differs"
        if cfg.get("backends.profile") == "gad_extensions_v1" and stage != "seqkd":
            from algorithm_patches import verify
            manifest = verify(backend)
            frozen._recovery_code_hash = manifest["extension_code_hash"]
        current = plan_stage(frozen, run_dir, stage, steps_per_epoch_value=steps_per_epoch(cfg),
            handoff=recorded_handoff(run_dir, cfg) if stage == "gad" else None, install_guard=False)
    except (PipelineError, RuntimeError, OSError, ValueError, KeyError) as exc:
        return False, f"historical backend/handoff cannot be verified: {exc}"
    def training_args(argv):
        diagnostic = ('trainer.resume_load_worker_state=', 'trainer.validation_rouge_l_enabled=',
            'trainer.val_before_train=', 'trainer.test_freq=', 'data.val_files=', 'data.val_batch_size=')
        return [a for a in argv if not a.startswith(diagnostic)]
    if training_args(previous["argv"]) != training_args(current["argv"]):
        return False, "historical launch training identity does not match current plan"
    # If the original run recorded immutable input hashes, require them too.
    historical_inputs = history.get("inputs")
    if historical_inputs and any(historical_inputs.get(k) != stage_inputs_digest(cfg, stage).get(k)
                                 for k in ("train_file_sha256", "train_rows")):
        return False, "historical training data digest differs"
    verification = verify_stage_artifacts(stage, stage_out_dir(cfg, run_dir, stage),
        expected_step=current["total_steps"], math_metrics_expected=validation_metrics_expected(cfg)[0],
        require_final_full=bool(cfg.get('validation.final_full')),
        full_count=dataset_rows(cfg.require('data.val_file')) * int(cfg.get('rollout.val_n') or 1),
        strict_evaluation=bool(cfg.get('validation.block_training_on_incomplete')))
    if not verification["handoff_ready"]:
        return False, "historical training/checkpoint is incomplete: " + str(verification["problems"])
    return True, (f"historical exit=0 and checkpoint {verification['last_step']} reverified; "
                  f"evaluation={verification['evaluation_status']}")


def check_skip(cfg: Config, run_dir: Path, stage: str) -> tuple[bool, str]:
    cfg = cfg.for_stage(stage)
    marker = completion_marker(run_dir, stage)
    if not marker.exists():
        return recover_completed_stage(cfg,run_dir,stage)
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    expected = stage_inputs_digest(cfg, stage)
    recorded_inputs = recorded.get("inputs")
    if recorded_inputs != expected:
        same_data = isinstance(recorded_inputs, dict) and all(
            recorded_inputs.get(key) == expected[key]
            for key in ("train_file_sha256", "val_file_sha256", "train_rows")
        )
        if not (stage == "seqkd" and same_data
                and _legacy_seqkd_marker_matches(
                    cfg, run_dir, recorded_inputs.get("config_signature", ""))):
            return False, "config or input signature changed since the stage completed"
    if recorded.get("returncode", 0) != 0:
        return False, "recorded training process did not exit successfully"
    if not recorded.get("verified", {}).get("ok"):
        return False, "the recorded completion was not verified"
    verified=verify_stage_artifacts(stage,stage_out_dir(cfg,run_dir,stage),
        expected_step=int(recorded.get('total_steps') or recorded.get('verified',{}).get('last_step') or stage_step_budget(cfg,stage,steps_per_epoch(cfg),handoff_step=recorded_handoff(run_dir,cfg)['step'] if stage=='gad' and recorded_handoff(run_dir,cfg) else 0)),
        math_metrics_expected=validation_metrics_expected(cfg)[0],
        require_final_full=bool(cfg.get('validation.final_full')),
        full_count=dataset_rows(cfg.require('data.val_file')) * int(cfg.get('rollout.val_n') or 1),
        strict_evaluation=bool(cfg.get('validation.block_training_on_incomplete')))
    if not verified['handoff_ready']:return False,'current checkpoint/evaluation policy no longer permits handoff: '+str(verified['problems'])
    return True, f"training already completed at step {verified['last_step']}; evaluation={verified['evaluation_status']}"


def write_completion(cfg: Config, run_dir: Path, stage: str, plan: dict, verification: dict,
                     *, started: str, finished: str, returncode: int) -> None:
    write_json(
        completion_marker(run_dir, stage),
        {
            "schema_version": "gad.training_completion.v2",
            "training_completed": True,
            "evaluation_status": verification.get("evaluation_status", "unknown"),
            "stage": stage,
            "backend": plan["backend"],
            "backend_commit": plan["backend_commit"],
            "out_dir": plan["out_dir"],
            "argv": plan["argv"],
            "start_step": plan["start_step"],
            "added_steps": plan["added_steps"],
            "total_steps": plan["total_steps"],
            "epochs": plan["epochs"],
            "steps_per_epoch": plan["steps_per_epoch"],
            "started_at": started,
            "finished_at": finished,
            "returncode": returncode,
            "verified": verification,
            "inputs": stage_inputs_digest(cfg, stage),
        },
    )


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def verify_backend_import(cfg: Config, run_dir: Path, stage: str, plan: dict) -> None:
    """Fail fast when the stage would import a foreign ``verl``."""
    backend = Path(plan["backend"])
    code = (
        "import verl, sys;"
        "from pathlib import Path;"
        f"expected = Path({str(backend)!r}).resolve();"
        "origin = Path(verl.__file__).resolve();"
        "inside = str(origin).startswith(str(expected) + '/');"
        "print(origin);"
        "sys.exit(0 if inside else 7)"
    )
    code_out, out, err = capture_command([cfg.require("run.python_bin"), "-c", code], env=plan["env"])
    if code_out != 0:
        raise PipelineError(
            f"[{stage}] backend import check failed: verl resolved to {out or err!r}, expected under {backend}"
        )
    log(f"{stage}: verl -> {out}")


def export_warmup_handoff(cfg: Config, run_dir: Path, *, dry_run: bool) -> dict:
    warmup_out = stage_out_dir(cfg, run_dir, "warmup")
    if cfg.get("budget.init_from_warmup_run"):
        from config import resolve_run, load_run_config
        source_run = resolve_run(cfg.get("budget.init_from_warmup_run"))
        source_cfg = load_run_config(source_run)
        for key in ("models.student", "data.train_file"):
            if Path(source_cfg.require(key)).resolve() != Path(cfg.require(key)).resolve():
                raise PipelineError(f"explicit Warmup initialization requires matching {key}; source {source_run}")
        warmup_out = source_run / "warmup"
    gad_dir = stage_out_dir(cfg, run_dir, "gad")
    argv = [
        str(cfg.require("run.python_bin")),
        str(PROJECT_DIR / "tools" / "prepare_gad_checkpoint.py"),
        "--warmup-dir", str(warmup_out),
        "--gad-dir", str(gad_dir),
        "--merge-tool", str(PROJECT_DIR / "tools" / "merge_model2hf.py"),
        "--manifest", str(run_dir / "handoff_manifest.json"),
    ]
    if cfg.get("budget.handoff_step"):
        argv += ["--step", str(int(cfg.get("budget.handoff_step")))]
    if cfg.get("budget.init_step"):
        argv += ["--step", str(int(cfg.get("budget.init_step")))]
    run_command(argv, dry_run=dry_run, component="handoff")
    manifest_path = run_dir / "handoff_manifest.json"
    if dry_run:
        return {"actor_hf": "<dry-run>", "critic_hf": "<dry-run>", "step": 0, "dry_run": True}
    if not manifest_path.exists():
        raise PipelineError(f"handoff manifest was not written: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("ok"):
        raise PipelineError(f"handoff failed: {manifest.get('problems')}")
    from extension_config import active
    if active(cfg, "rubric", "gad") or active(cfg, "replay", "gad"):
        initial = Path(manifest["gad_checkpoint"])
        write_json(initial / "handoff_initialization.json", {"source": manifest["source_checkpoint"],
                                                           "step": manifest["step"], "same_stage_resume": False})
    return manifest


def run_stage(cfg: Config, run_dir: Path, stage: str, *, dry_run: bool,
              resume_requested: bool = False) -> dict:
    cfg = cfg.for_stage(stage)
    steps_per_epoch_value = steps_per_epoch(cfg)
    handoff = None
    resume: dict | None = None
    if stage == "gad":
        recorded = recorded_handoff(run_dir, cfg)
        planned_warmup_step = int(cfg.get("budget.init_step") or stage_step_budget(cfg, "warmup", steps_per_epoch_value))
        gad_added_steps = stage_step_budget(
            cfg, "gad", steps_per_epoch_value,
            handoff_step=int(recorded["step"]) if recorded else planned_warmup_step,
        )
        decision = resolve_gad_resume(
            cfg, run_dir, added_steps=gad_added_steps, resume_requested=resume_requested
        )
        resume = decision
        if decision["mode"] == "fresh":
            if decision.get("conflict") and not dry_run:
                raise PipelineError(decision["reason"])
            if decision.get("conflict"):
                log("WARNING: " + decision["reason"])
            handoff = export_warmup_handoff(cfg, run_dir, dry_run=dry_run)
            if dry_run:
                # Mirror what prepare_gad_checkpoint.py will produce, so the printed
                # plan matches the real run instead of a placeholder. The last Warmup
                # step is saved by the trainer even when it is not a save_freq multiple.
                planned_step = int(cfg.get("budget.init_step") or cfg.get("budget.handoff_step") or planned_warmup_step)
                target = stage_out_dir(cfg, run_dir, "gad") / f"global_step_{planned_step}"
                handoff = {
                    "actor_hf": str(target / "actor" / "huggingface"),
                    "critic_hf": str(target / "critic" / "huggingface"),
                    "step": planned_step,
                }
        else:
            handoff = decision["handoff"]
            log(f"[gad] {decision['reason']}")
            log(f"[gad] reusing the recorded handoff ({decision['handoff']['source']}): "
                f"model={handoff['actor_hf']}")

    plan = plan_stage(cfg, run_dir, stage, steps_per_epoch_value=steps_per_epoch_value,
                      handoff=handoff, resume=resume, install_guard=not dry_run)

    if dry_run:
        return plan

    ensure_dir(Path(plan["out_dir"]))
    ensure_dir(run_dir / "logs")

    math_expected, math_reason = validation_metrics_expected(cfg)
    started = utc_timestamp()

    if stage == "gad" and (plan.get("resume") or {}).get("mode") == "resume-complete":
        # Training already reached the budget (e.g. the process died after the
        # final save but before the marker was written). Verify instead of
        # running one step past global_step_{S+K}.
        log(f"[gad] {plan['resume']['reason']}")
        recovered, reason = recover_completed_stage(cfg, run_dir, stage)
        if not recovered:
            raise PipelineError("resume-complete requires successful historical training provenance: " + reason)
        return finish_stage(
            cfg, run_dir, stage, plan, process=None, started=started,
            math_expected=math_expected, math_reason=math_reason, trainer_skipped=True,
        )

    verify_backend_import(cfg, run_dir, stage, plan)
    if cfg.get("backends.profile") == "gad_extensions_v1":
        (Path(plan["backend"]) / ".gad_backend_used").touch()

    log(f"===== stage {stage} =====")
    log(f"backend : {plan['backend']} @ {plan['backend_commit'][:10]}")
    log(f"out dir : {plan['out_dir']}")
    log(f"steps   : start={plan['start_step']} +{plan['added_steps']} -> {plan['total_steps']} "
        f"(epochs={plan['epochs']}, steps/epoch={plan['steps_per_epoch']})")
    if (plan.get("resume") or {}).get("mode") == "resume":
        log(f"resume  : from global_step_{plan['resume']['from_step']} (full-state resume enabled)")
    log(f"model   : {plan['model_path']}")
    log(f"discrim : {plan['critic_path']}")

    keep_checkpoints = cfg.get("resources.keep_last_checkpoints")
    retention = (
        RetentionMonitor(Path(plan["out_dir"]), keep_checkpoints=int(keep_checkpoints))
        if keep_checkpoints is not None else nullcontext()
    )
    with retention:
        process = run_command(
            plan["argv"], env=plan["env"], cwd=str(PROJECT_DIR), log_path=plan["log"],
            check=False, component=stage,
        )
    return finish_stage(
        cfg, run_dir, stage, plan, process=process, started=started,
        math_expected=math_expected, math_reason=math_reason, trainer_skipped=False,
    )


def preserve_failed_stage_log(plan: dict, *, finished: str, tail_lines: int = 60) -> dict:
    """Copy a failing stage's log aside and record its tail.

    The live ``logs/<stage>.log`` is truncated by the *next* attempt, so restarting used to
    destroy the only copy of the traceback (measured: a stage that died after 88 minutes left
    nothing to diagnose). Keep the file and the last lines inside ``stage_<stage>.json``.
    """
    log_path = Path(str(plan["log"]))
    info: dict = {"live_log": str(log_path)}
    tail: list[str] = []
    if log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            tail = [line.rstrip() for line in text.splitlines()[-tail_lines:]]
        except OSError:
            tail = []
        stamp = finished.replace(":", "").replace("-", "")
        preserved = log_path.with_name(f"{log_path.stem}.failed-{stamp}{log_path.suffix}")
        try:
            shutil.copy2(log_path, preserved)
            info["preserved"] = str(preserved)
        except OSError as exc:  # noqa: BLE001 - never hide the stage failure behind this
            info["preserved"] = None
            info["copy_error"] = str(exc)
    info["tail"] = tail
    # Ray keeps worker stdout/stderr outside the run dir; those survive a restart too.
    ray_root = Path("/tmp/ray")
    ray_dirs = sorted(ray_root.glob("session_*/logs")) if ray_root.is_dir() else []
    if ray_dirs:
        info["ray_log_dir"] = str(ray_dirs[-1])
    return info


def finish_stage(cfg: Config, run_dir: Path, stage: str, plan: dict, *, process, started: str,
                 math_expected: bool, math_reason: str, trainer_skipped: bool) -> dict:
    """Verify a stage's artifacts and only then write its completion marker."""
    finished = utc_timestamp()
    stage_record_path = run_dir / f"stage_{stage}.json"
    if trainer_skipped:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        stage_record_path = run_dir / "reports" / f"resume_complete_{stage}_{stamp}.json"
    verification = verify_stage_artifacts(
        stage, Path(plan["out_dir"]), expected_step=plan["total_steps"],
        math_metrics_expected=math_expected, math_metrics_reason=math_reason,
        require_final_full=bool(cfg.get("validation.final_full")),
        strict_evaluation=bool(cfg.get("validation.block_training_on_incomplete")),
        full_count=dataset_rows(cfg.require("data.val_file")) * int(cfg.get('rollout.val_n') or 1) if cfg.get("validation.final_full") else None,
    )
    returncode = 0 if trainer_skipped else process.returncode
    if not trainer_skipped and process.returncode != 0:
        verification["ok"] = verification["handoff_ready"] = False
        verification["training_status"] = "failed"
        failure = f"stage process exited with code {process.returncode} (log: {plan['log']})"
        verification["problems"].append(failure)
        verification["training_problems"].append(failure)
    if returncode == 0 and not verification["training_problems"]:
        verification["training_status"] = "completed"
    if trainer_skipped:
        verification["trainer_skipped"] = True

    stage_manifest = {
        'scorer_identity': {key[13:]: value for key, value in cfg.data.items() if key.startswith('scoring.math.')},
        'seeds': {'initial_critic': cfg.get('rubric.seed'), 'data_order': cfg.get('run.seed'),
                  'driver_torch_numpy_python': cfg.get('run.seed'), 'rollout_engine': cfg.get('rollout.seed'),
                  'sampling_params_request': cfg.get('rollout.seed'), 'request_seed_policy': 'same_explicit_seed_per_request',
                  'judge': cfg.get('rubric.seed') if plan.get('rubric_enabled') else None,
                  'judge_sampling_request': cfg.get('rubric.seed') if plan.get('rubric_enabled') else None,
                  'judge_engine': None,
                  'replay': cfg.get('replay.seed') if plan.get('replay_enabled') else None,
                  'worker_rng_seed': None},
        "stage": stage,
        "plan": {k: v for k, v in plan.items() if k != "env"},
        "env": {k: v for k, v in plan["env"].items() if k != "PYTHONPATH"},
        "pythonpath": plan["env"].get("PYTHONPATH"),
        "started_at": started,
        "finished_at": finished,
        "returncode": returncode,
        "trainer_skipped": trainer_skipped,
        "verification": verification,
    }
    intentional_limit = bool(cfg.get("run.max_stage_steps") and returncode == 0
                             and verification.get("problems")
                             and all(problem.startswith("last checkpoint is global_step_") for problem in verification["problems"]))
    if intentional_limit:
        stage_manifest["invocation_limit_reached"] = True
        plan["invocation_limit_reached"] = True
        write_json(stage_record_path, stage_manifest)
        log(f"[{stage}] engineering invocation limit reached; checkpoint saved, full stage remains incomplete")
        return plan
    if not verification["ok"]:
        # Preserve the evidence before the next attempt truncates the live log.
        stage_manifest["failure_log"] = preserve_failed_stage_log(plan, finished=finished)
    write_json(stage_record_path, stage_manifest)

    if not verification["ok"]:
        failure = stage_manifest.get("failure_log") or {}
        hint = f"; preserved log: {failure['preserved']}" if failure.get("preserved") else ""
        raise PipelineError(
            f"[{stage}] failed verification: {verification['problems']}{hint} "
            f"(tail + Ray log dir in {run_dir / f'stage_{stage}.json'})"
        )
    write_completion(cfg, run_dir, stage, plan, verification, started=started, finished=finished,
                     returncode=returncode)
    log(f"[{stage}] verified: last checkpoint global_step_{verification['last_step']}")
    math = verification.get("math_metrics") or {}
    if verification['evaluation_status'] != 'complete':
        log(f"[{stage}] training completed; evaluation={verification['evaluation_status']}: "
            f"{verification['evaluation_problems']}")
    if math.get('last'):
        last=math['last'];acc=last.get('acc')
        log(f"[{stage}] dev math accuracy: acc={format(acc,'.4f') if acc is not None else 'null'} "
            f"provisional_acc={last.get('provisional_acc')} status={verification['evaluation_status']} "
            f"(n={last.get('count')}, truncated={last.get('truncated_rate')})")
    return plan


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def print_plan(cfg: Config, run_dir: Path, plans: list[dict], *, steps_per_epoch_value: int) -> None:
    print("=" * 78)
    print("GAD pipeline plan (dry-run)")
    print("=" * 78)
    print(f"config        : {cfg.path}")
    print(f"run dir       : {run_dir}")
    print(f"python        : {cfg.get('run.python_bin')}")
    print(f"student       : {cfg.get('models.student')}")
    print(f"discriminator : {cfg.get('models.discriminator_init')}")
    print(f"train data    : {cfg.get('data.train_file')}")
    print(f"val data      : {cfg.get('data.val_file')}")
    print(f"train rows    : {dataset_rows(cfg.require('data.train_file'))} "
          f"(steps/epoch={steps_per_epoch_value}, batch={cfg.get('data.train_batch_size')})")
    print(f"lengths       : prompt={cfg.get('data.max_prompt_length')} "
          f"response={cfg.get('data.max_response_length')}")
    print(f"rollout       : n={cfg.get('rollout.n')} temperature={cfg.get('rollout.temperature')} "
          f"tp={cfg.get('rollout.tensor_model_parallel_size')} thinking={cfg.get('rollout.enable_thinking')} "
          f"(control={'on' if cfg.get('rollout.thinking_control') else 'off'})")
    print(f"ROUGE-L       : {cfg.get('validation.rouge_l_enabled')} (diagnostic)")
    print(f"eval blocks handoff: {cfg.get('validation.block_training_on_incomplete')}")
    print(f"gpus          : {cfg.get('resources.n_gpus_per_node')} x {cfg.get('resources.nnodes')} node(s)")
    keep_checkpoints = cfg.get("resources.keep_last_checkpoints")
    if keep_checkpoints is not None:
        print(f"checkpoints   : save every {cfg.get('resources.save_freq')} steps; "
              f"keep last {keep_checkpoints} saves per stage "
              "(plus handoff and exported models)")
    print()
    for plan in plans:
        print("-" * 78)
        print(f"stage {plan['stage']}")
        print(f"  extensions   : rubric={plan.get('rubric_enabled', False)} replay={plan.get('replay_enabled', False)}")
        print(f"  backend      : {plan['backend']} @ {plan['backend_commit'][:10]}")
        print(f"  output       : {plan['out_dir']}")
        print(f"  model path   : {plan['model_path']}")
        print(f"  discriminator: {plan['critic_path']}")
        print(f"  steps        : start={plan['start_step']} +{plan['added_steps']} -> {plan['total_steps']}"
              f"   epochs={plan['epochs']}")
        if plan.get("epoch_budget"):
            scope = "absolute target" if plan["stage"] == "gad" else "own data"
            print(f"  epoch budget : {plan['epoch_budget']} epoch(s) x {plan['steps_per_epoch']} steps/epoch"
                  f" ({scope})")
        resume = plan.get("resume") or {}
        if resume.get("requested"):
            print(f"  resume       : mode={resume.get('mode')} from_step={resume.get('from_step')} "
                  f"({resume.get('reason')})")
        print(f"  math metric  : {plan['env'].get('GAD_MATH_SCORER') or 'DISABLED'}")
        print(f"  log          : {plan['log']}")
        print(f"  PYTHONPATH   : {plan['env']['PYTHONPATH']}")
        print("  command:")
        print("    " + " \\\n      ".join(plan["argv"]))
    print("-" * 78)
    print(f"eval after train: {bool(cfg.get('run.eval_after_train'))}")
    print("nothing was executed (dry-run)")


def expected_gad_final_step(run_dir: Path) -> int | None:
    """The GAD step this run verified (completion marker wins over stage_gad.json)."""
    stage_json = read_json(run_dir / "stage_gad.json", default=None)
    expected = ((stage_json or {}).get("verification") or {}).get("last_step")
    marker = read_json(completion_marker(run_dir, "gad"), default=None)
    if marker:
        expected = (marker.get("verified") or {}).get("last_step", expected)
    return int(expected) if expected is not None else None


def verify_final_model_step(run_dir: Path, manifest: dict) -> int | None:
    """Refuse to evaluate anything but the step the GAD stage verified."""
    expected = expected_gad_final_step(run_dir)
    if expected is not None and int(manifest["step"]) != expected:
        raise PipelineError(
            f"final model export used global_step_{manifest['step']} but the GAD stage verified "
            f"global_step_{expected}; refusing to evaluate the wrong checkpoint"
        )
    return expected


def export_final_actor(cfg: Config, run_dir: Path, *, dry_run: bool) -> str:
    """Merge the trained GAD actor and prove it is the step the stage reported.

    ``handoff_manifest.actor_hf`` is the *Warmup* checkpoint GAD started from, so
    using it for the final evaluation would report before-training numbers.
    """
    gad_out = stage_out_dir(cfg, run_dir, "gad")
    manifest_path = run_dir / "final_model_manifest.json"
    argv = [
        str(cfg.require("run.python_bin")),
        str(PROJECT_DIR / "tools" / "prepare_gad_checkpoint.py"),
        "--mode", "export-final",
        "--gad-dir", str(gad_out),
        "--merge-tool", str(PROJECT_DIR / "tools" / "merge_model2hf.py"),
        "--manifest", str(manifest_path),
    ]
    run_command(argv, dry_run=dry_run, component="export-final")
    if dry_run:
        handoff = recorded_handoff(run_dir, cfg) or {}
        per_epoch = steps_per_epoch(cfg)
        start = int(handoff.get('step') or cfg.get('budget.init_step') or stage_step_budget(cfg,'warmup',per_epoch))
        target = start + stage_step_budget(cfg,'gad',per_epoch,handoff_step=start)
        return f"<dry-run: {gad_out}/global_step_{target}/actor/huggingface>"

    manifest = read_json(manifest_path, default=None)
    if not manifest:
        raise PipelineError(f"final model manifest was not written: {manifest_path}")
    if not manifest.get("ok"):
        raise PipelineError(f"final model export failed: {manifest.get('problems')}")
    actor_hf = manifest.get("actor_hf")
    if not actor_hf or not Path(actor_hf).exists():
        raise PipelineError(f"final model export did not produce a directory: {actor_hf!r}")

    # Guard against the original bug: the exported step must be the step the GAD
    # stage verified, not the Warmup init step.
    verify_final_model_step(run_dir, manifest)
    handoff = recorded_handoff(run_dir,cfg) or {}
    added = int(manifest['step']) - int(handoff['step']) if handoff.get('step') is not None else None
    log(f"final model: {actor_hf} (global_step_{manifest['step']}, "
        f"added steps from recorded Warmup handoff: {added if added is not None else 'unavailable'})")
    return str(actor_hf)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--name", default="", help="Optional label for a new run")
    parser.add_argument("--resume", action="store_true", help="Resume the saved config selected by --run")
    parser.add_argument("--gad-new-steps", type=int, help="Additional GAD steps K; overrides its absolute epoch budget")
    parser.add_argument("--init-from-warmup-run", help="Explicit read-only Warmup source for a new GAD experiment")
    parser.add_argument("--init-step", type=int, help="Exact complete Warmup initialization step")
    parser.add_argument("--warmup-smoke-steps", type=int, help="New engineering run: explicitly shorten Warmup")
    parser.add_argument("--max-stage-steps", type=int, help="Engineering-only invocation limit; does not change planned budgets")
    parser.add_argument("--stages", default=None, help="Comma-separated subset (default: seqkd,warmup,gad).")
    parser.add_argument("--from-stage", default=None, choices=TRAIN_STAGES)
    parser.add_argument("--to-stage", default=None, choices=TRAIN_STAGES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--force", action="store_true", help="Re-run stages even when a valid marker exists.")
    parser.add_argument("--resume-gad", action="store_true",
                        help="Enable full-state resume for the GAD stage (optimizer/scheduler/RNG).")
    parser.add_argument("--skip-data-check", action="store_true")
    args = parser.parse_args()

    cfg = select_config(args)
    updates = {"budget.gad_new_steps": args.gad_new_steps,
               "budget.init_from_warmup_run": args.init_from_warmup_run, "budget.init_step": args.init_step}
    if cfg.existing_run and any(value is not None for value in updates.values()):
        raise PipelineError("resume uses the saved budget/source; these initialization options require a new run")
    cfg.data.update({key: value for key, value in updates.items() if value is not None})
    if args.warmup_smoke_steps is not None:
        if cfg.existing_run or args.warmup_smoke_steps < 1:
            raise PipelineError("--warmup-smoke-steps requires positive steps and a new run")
        cfg.data.update({"budget.warmup_steps": args.warmup_smoke_steps,
                         "budget.stage_epochs.warmup": None, "run.engineering_smoke": True})
    if args.max_stage_steps is not None:
        cfg.data["run.max_stage_steps"] = args.max_stage_steps
        cfg.runtime_overrides["run.max_stage_steps"] = args.max_stage_steps
    from extension_config import configure_profile
    configure_profile(cfg)
    source_config = str(cfg.path)
    if args.resume and not cfg.existing_run:
        raise PipelineError("--resume requires --run (or a saved run train.yaml)")
    if cfg.existing_run and not (args.resume or args.resume_gad or cfg.legacy_selection):
        raise PipelineError("an existing run was selected; pass --resume to continue its saved config")
    if cfg.existing_run:
        args.resume_gad = True
    if args.set:
        log(f"config overrides: {parse_key_values(args.set)}")

    problems = validate_config(cfg)
    if not args.skip_data_check:
        problems += validate_paths(cfg, check_exists=True)
    if problems:
        for problem in problems:
            log(f"CONFIG ERROR: {problem}")
        return 2

    if args.resume_gad:
        cfg.data["run.resume_worker_state"] = True
        log("resume-gad: the GAD stage will restore optimizer/scheduler/RNG state")

    stages = list(TRAIN_STAGES)
    if args.stages:
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        if stages == ["all"]:
            stages = list(TRAIN_STAGES)
        unknown = [s for s in stages if s not in TRAIN_STAGES]
        if unknown:
            raise PipelineError(f"unknown stage(s): {unknown}; known: {TRAIN_STAGES}")
    if args.from_stage:
        stages = [s for s in stages if TRAIN_STAGES.index(s) >= TRAIN_STAGES.index(args.from_stage)]
    if args.to_stage:
        stages = [s for s in stages if TRAIN_STAGES.index(s) <= TRAIN_STAGES.index(args.to_stage)]
    if not stages:
        raise PipelineError("no stage selected")
    if cfg.get("budget.init_from_warmup_run") and stages != ["gad"]:
        raise PipelineError("explicit Warmup initialization is for --stages gad only")

    run_id = args.run_id or (None if cfg.get("run.run_id") == "auto" else str(cfg.get("run.run_id")))
    if cfg.existing_run and args.run_id and args.run_id != str(cfg.get("run.run_id")):
        raise PipelineError("--run-id cannot relocate an existing run; start a new experiment")
    run_id = run_id or (new_run_id(cfg, args.name) if cfg.global_config else make_run_id("gad"))
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise PipelineError("invalid run ID")
    cfg.data["run.run_id"] = run_id
    run_dir = Path(cfg.require("run.output_root")) / run_id
    if cfg.global_config and run_dir.exists():
        raise PipelineError(f"new run directory already exists: {run_dir}; use --run ... --resume")

    if not args.dry_run:
        from datetime import datetime, timezone
        if cfg.global_config:
            save_run_config(cfg, run_dir)
            (run_dir / "current_resolved_config.txt").write_text(str(cfg.path) + "\n", encoding="utf-8")
        else:
            ensure_dir(run_dir)
            if not cfg.existing_run:
                import yaml as _yaml
                versions = ensure_dir(run_dir / "reports" / "resolved_configs")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                snapshot = versions / f"resolved_config_{stamp}.yaml"
                snapshot.write_text(_yaml.safe_dump(cfg.to_dict(), sort_keys=True, allow_unicode=True), encoding="utf-8")
                (run_dir / "current_resolved_config.txt").write_text(str(snapshot) + "\n", encoding="utf-8")
                # Preflight must read the resolved snapshot, including CLI overrides.
                cfg.path = snapshot
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        write_json(run_dir / "reports" / f"launch_{stamp}.json", {
            "source_config": source_config, "config": str(cfg.path),
            "config_sha256": sha256_file(cfg.path), "selection": cfg.selection,
            "argv": sys.argv, "runtime_overrides": cfg.runtime_overrides,
            "resume": cfg.existing_run, "stages": stages,
        })
        if free_disk_gib(run_dir) < 5:
            log(f"WARNING: only {free_disk_gib(run_dir):.1f} GiB free under {run_dir}")

    # 1) backends
    prepare_argv = [
        str(cfg.require("run.python_bin")),
        str(PROJECT_DIR / "tools" / "prepare_backends.py"),
        "--repo", str(cfg.require("backends.source_repo")),
        "--backend-root", str(cfg.require("backends.root")),
        "--stages", ",".join(stages),
        "--profile", str(cfg.get("backends.profile")),
    ]
    if cfg.get("backends.recreate"):
        prepare_argv.append("--recreate")
    if not args.dry_run:
        run_command(prepare_argv, component="backends")
        if not args.skip_data_check:
            preflight_argv = [
                str(cfg.require("run.python_bin")), str(PROJECT_DIR / "tools" / "preflight.py"),
                "--stages", ",".join(stages), "--skip-packages",
                "--report", str(run_dir / "reports" / f"preflight_{stamp}.json"),
            ]
            if cfg.existing_run:
                preflight_argv += ["--run", str(run_dir)]
                for setting in args.set:
                    preflight_argv += ["--set", setting]
            else:
                preflight_argv += ["--config", str(cfg.path)]
            run_command(preflight_argv, component="preflight")

    # 2) plans
    steps_per_epoch_value = steps_per_epoch(cfg)
    plans = []
    for stage in stages:
        skip = False
        reason = ""
        if not args.force and cfg.get("run.skip_completed_stages") and (not args.dry_run or cfg.existing_run):
            skip, reason = check_skip(cfg, run_dir, stage)
        if skip:
            log(f"skipping stage {stage}: {reason}")
            if not args.dry_run and not completion_marker(run_dir,stage).exists() and not (run_dir/'reports'/f'training_recovery_{stage}.json').exists():
                write_json(run_dir/'reports'/f'training_recovery_{stage}.json',{
                    'schema_version':'gad.training_recovery.v1','stage':stage,'reason':reason,
                    'original_record':str(run_dir/f'stage_{stage}.json'),
                    'inputs':stage_inputs_digest(cfg,stage),'training_completed':True,
                    'evaluation_verified':False})
            continue
        plans.append(run_stage(cfg, run_dir, stage, dry_run=args.dry_run, resume_requested=args.resume_gad))
        if plans[-1].get("invocation_limit_reached"):
            break

    if args.dry_run:
        print_plan(cfg, run_dir, plans, steps_per_epoch_value=steps_per_epoch_value)
        return 0

    # 3) run summary
    summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config": str(cfg.path),
        "config_resolved": cfg.to_dict(),
        "stages": stages,
        "environment": environment_summary(),
        "dependency_versions": dependency_versions(cfg),
        "data_provenance": data_provenance(cfg),
        "created_at": utc_timestamp(),
        "stage_manifests": [str(run_dir / f"stage_{s}.json") for s in stages],
        "disk_free_gib": free_disk_gib(run_dir),
    }
    if (run_dir / "handoff_manifest.json").exists():
        summary["handoff_manifest"] = str(run_dir / "handoff_manifest.json")
    for stage in stages:
        stage_json = run_dir / f"stage_{stage}.json"
        if stage_json.exists():
            summary.setdefault("stage_results", {})[stage] = json.loads(stage_json.read_text(encoding="utf-8"))[
                "verification"
            ]
    write_json(run_dir / "run_summary.json", summary)
    log(f"run summary -> {run_dir / 'run_summary.json'}")

    if cfg.get("run.eval_after_train") and cfg.get("eval.enabled"):
        # Evaluate the model GAD produced. The handoff actor is the Warmup
        # checkpoint GAD started from, not the trained model.
        final_model = export_final_actor(cfg, run_dir, dry_run=False)
        summary.setdefault("final_model", {})
        summary["final_model"] = {
            "actor_hf": final_model,
            "manifest": str(run_dir / "final_model_manifest.json"),
        }
        write_json(run_dir / "run_summary.json", summary)
        eval_argv = [
            str(cfg.require("run.python_bin")),
            str(PROJECT_DIR / "tools" / "evaluate_math.py"),
            "--model", str(final_model),
            "--parquet", str(cfg.require("eval.parquet")),
            "--out-dir", str(run_dir / "eval" / "final"),
            "--benchmarks", str(cfg.get("eval.benchmarks")),
            "--n", str(int(cfg.get("eval.n"))),
            "--math500-n", str(int(cfg.get("eval.math500_n"))),
            "--temperature", str(cfg.get("eval.temperature")),
            "--top-p", str(cfg.get("eval.top_p")),
            "--max-tokens", str(int(cfg.get("eval.max_tokens"))),
            "--max-model-len", str(int(cfg.get("eval.max_model_len"))),
            "--tp", str(int(cfg.get("eval.tensor_parallel_size"))),
            "--gpu-memory-utilization", str(cfg.get("eval.gpu_memory_utilization")),
            "--enable-thinking", "true" if cfg.get("rollout.enable_thinking") else "false",
            '--scorer-path', str(cfg.require('validation.scorer_path')),
        ]
        if cfg.get('scoring.math.version') == 'math_unified_v1':
            eval_argv += ['--scoring-config-json',json.dumps({key[13:]:value for key,value in cfg.data.items() if key.startswith('scoring.math.')})]
        run_command(eval_argv, component="eval")

    log("pipeline finished")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        log(f"FATAL: {exc}")
        raise SystemExit(1)
