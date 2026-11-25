"""Train the H200 n=8 verified-reward GRPO control with the same GAD actor."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from common import run_command
from config import load_config
from pipeline import backend_path, build_overrides, epochs_for, stage_env, steps_per_epoch
from prepare_gad_checkpoint import has_tokenizer, inspect_safetensors
from rlvr_checkpoint import publish_eval_model


TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
DEFAULT_CONFIG = ROOT / "configs/h200_baseline_n8.yaml"


def training_budget(cfg, steps, smoke):
    if smoke:
        return 1, [1]
    total = steps or cfg.get("budget.rlvr_steps") or cfg.require("budget.gad_new_steps")
    configured = cfg.get("budget.rlvr_eval_steps") or []
    return total, sorted({step for step in configured if 0 < step <= total} | {total})


def prepare(cfg, run_dir, steps, smoke):
    """Freeze the trainer, reward code and effective command in this run only."""
    python = str(cfg.require("run.python_bin"))
    source = backend_path(cfg, "gad")
    subprocess.run([python, str(TOOLS / "prepare_backends.py"),
                    "--repo", str(cfg.require("backends.source_repo")),
                    "--backend-root", str(source.parent), "--stages", "gad",
                    "--profile", "gad_extensions_v1"], check=True)

    backend = run_dir / "runtime" / "rlvr_backend"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source / "verl", backend / "verl", ignore=ignore)
    shutil.copytree(ROOT / "scoring", backend / "scoring", ignore=ignore)
    shutil.copyfile(TOOLS / "rlvr_reward.py", backend / "verl/trainer/ppo/rlvr_reward.py")
    shutil.copyfile(TOOLS / "rlvr_checkpoint.py", backend / "verl/trainer/ppo/rlvr_checkpoint.py")
    subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i",
                    str(TOOLS.parent / "configs/rlvr_grpo.patch")], cwd=backend, check=True)

    env = stage_env(cfg, run_dir, "gad", source, log_backend=cfg.get("run.log_backend"))
    guard = env["PYTHONPATH"].split(os.pathsep)[0]
    env.update(PYTHONPATH=os.pathsep.join((guard, str(backend))), GAD_EXPECT_VERL_PATH=str(backend),
               GAD_BACKEND_STAGE="rlvr", GAD_MATH_SCORER=str(backend / "scoring/math.py"),
               WANDB_JOB_TYPE="rlvr", VLLM_USE_V1=os.environ.get("VLLM_USE_V1", "1"))
    scorer_config = json.loads(env["GAD_SCORING_MATH_CONFIG"])
    scorer_config["entrypoint"] = env["GAD_MATH_SCORER"]
    env["GAD_SCORING_MATH_CONFIG"] = json.dumps(scorer_config)

    steps, eval_steps = training_budget(cfg, steps, smoke)
    out = run_dir / "rlvr"
    # Start both policy and KL reference at the raw student, with the GAD optimizer
    # and sampling settings. build_overrides requires a critic path for the GAD
    # branch; all its critic options are removed before Hydra or Ray is invoked.
    model = cfg.require("models.student")
    options = build_overrides(cfg, "gad", model_path=model, critic_path=model,
                              out_dir=out, total_steps=steps,
                              epochs=epochs_for(steps, steps_per_epoch(cfg)))
    overrides = {key: value for key, value in (x.split("=", 1) for x in options)
                 if not key.startswith("critic.")}
    overrides.pop("trainer.checkpoint_steps", None)  # RLVR publishes its own eval_steps.
    # Native FSDP rotation deletes first and saves second. Keep the preceding
    # complete actor available if a later save is interrupted.
    retained = cfg.get("resources.keep_last_checkpoints")
    overrides.update({
        "trainer.experiment_name": f"{run_dir.name}_rlvr",
        "hydra.run.dir": str(run_dir / "runtime/hydra/rlvr"),
        "trainer.validation_rouge_l_enabled": "False",
        "trainer.resume_load_worker_state": "True",
        "+trainer.rlvr_eval_steps": str(eval_steps).replace(" ", ""),
        "trainer.max_actor_ckpt_to_keep": str(1 + int(retained)) if retained else "null",
        "actor_rollout_ref.actor.checkpoint.save_contents": "[model,optimizer,extra,hf_model]",
        "actor_rollout_ref.actor.checkpoint.load_contents": "[model,optimizer,extra]",
        "actor_rollout_ref.rollout.gad_extensions.enabled": "True",  # actual generation lengths
        "reward_model.enable": "False",
    })
    if smoke:
        overrides.update({"trainer.test_freq": "0", "trainer.val_before_train": "False",
                          "++trainer.disable_validation": "True", "trainer.logger": "[console]"})
    argv = [python, "-m", "verl.trainer.main_ppo", *(f"{k}={v}" for k, v in overrides.items())]
    plan = {"source_config": str(cfg.path), "steps": steps, "eval_steps": eval_steps, "argv": argv, "env": env,
            "backend": str(backend), "reward": "1 if the final answer is verified correct, else 0"}
    (run_dir / "rlvr_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="defaults to the two-epoch single-H200 recipe; use scripts/h200_8gpu.py rlvr for the eight-GPU series")
    parser.add_argument("--output", type=Path, help="new run directory, or existing run with --resume")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--steps", type=int, help="total optimizer steps (default: budget.rlvr_steps)")
    mode.add_argument("--smoke", action="store_true", help="one full-shape update and checkpoint, skip dev")
    mode.add_argument("--resume", action="store_true", help="use the original plan and full-state checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="compose the real Hydra job on CPU")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.resume and args.output is None:
        parser.error("--resume needs the original --output directory")
    cfg = None if args.resume else load_config(args.config)
    if cfg is not None and (cfg.get("rollout.n") != 8 or cfg.get("rubric.enabled")):
        parser.error("--config must be an n=8 Base recipe")

    def run(run_dir, dry_run):
        with (run_dir / ".rlvr.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = (json.loads((run_dir / "rlvr_plan.json").read_text()) if args.resume
                    else prepare(cfg, run_dir, args.steps, args.smoke))
            if dry_run:
                with (run_dir / "rlvr_hydra.yaml").open("w") as output:
                    subprocess.run([*plan["argv"], "--cfg", "job", "--resolve"], cwd=plan["backend"],
                                   env={**os.environ, **plan["env"]}, stdout=output, check=True)
                print(f"Hydra job composed: {plan['steps']} optimizer steps, "
                      f"evaluation steps {plan.get('eval_steps', [plan['steps']])}, {plan['reward']}")
                return
            stage_dir = run_dir / "rlvr"
            marker = stage_dir / "latest_checkpointed_iteration.txt"
            completed = int(marker.read_text()) if marker.exists() else 0
            eval_steps = plan.get("eval_steps", [])
            # A stop after the complete checkpoint and before publishing its HF
            # links must be repaired before FSDP can rotate that checkpoint.
            for step in eval_steps:
                if step <= completed:
                    publish_eval_model(stage_dir, step)
            if completed < plan["steps"]:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                run_command(plan["argv"], env=plan["env"], cwd=plan["backend"],
                            log_path=run_dir / "logs" / f"rlvr-{stamp}.log", component="rlvr")
            if not marker.is_file() or int(marker.read_text()) != plan["steps"]:
                raise RuntimeError("RLVR exited without saving its final training step")
            if eval_steps:
                models = [stage_dir / "eval_models" / f"step_{step}" for step in eval_steps]
            else:  # Frozen plans created before milestone exports.
                models = [stage_dir / f"global_step_{plan['steps']}" / "actor/huggingface"]
            for hf in models:
                if not (hf / "config.json").is_file() or not has_tokenizer(hf) or not inspect_safetensors(hf)["keys"]:
                    raise RuntimeError(f"RLVR checkpoint is missing the evaluation model: {hf}")
                print(f"Eval model: {hf}")

    if args.dry_run and not args.resume:
        with tempfile.TemporaryDirectory(prefix="rlvr-plan-") as tmp:
            run(Path(tmp), True)
        return
    if args.output:
        directory = args.output.expanduser().resolve()
    else:
        directory = Path(cfg.require("run.output_root")) / (
            "h200_rlvr_06b_17k_n8_smoke" if args.smoke else "h200_rlvr_06b_17k_n8")
    if not args.resume:
        directory.mkdir(parents=True, exist_ok=False)
    run(directory, args.dry_run)


if __name__ == "__main__":
    main()
