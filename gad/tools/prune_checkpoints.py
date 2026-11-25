#!/usr/bin/env python3
"""Keep the most recent N complete saved checkpoints for each stage.

This tool removes whole step directories across resumed runs, including
incomplete data.pt-only directories left by the trainer's optional shard rotation.
Explicitly exported Hugging Face models and completed stage results are pinned.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
from pathlib import Path

from config import add_config_arguments, select_config
from extension_config import active
from prepare_gad_checkpoint import TOKENIZER_HINTS, verify_checkpoint_dir


def checkpoint_dirs(stage_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for child in stage_dir.iterdir() if stage_dir.is_dir() else []:
        match = re.fullmatch(r"global_step_(\d+)", child.name)
        if match and child.is_dir() and not child.is_symlink():
            found.append((int(match.group(1)), child))
    return sorted(found)


def pinned_steps(stage_dir: Path, checkpoints: list[tuple[int, Path]] | None = None) -> set[int]:
    pins = set()
    if checkpoints is None:
        checkpoints = checkpoint_dirs(stage_dir)
    for step, checkpoint in checkpoints:
        # A merged model may be used by a downstream stage or a published eval.
        actor_hf = checkpoint / "actor" / "huggingface"
        critic_hf = checkpoint / "critic" / "huggingface"
        if any(actor_hf.glob("*.safetensors")) or any(critic_hf.glob("*.safetensors")):
            pins.add(step)
    marker = stage_dir / "COMPLETED.json"
    if marker.is_file():
        verified = json.loads(marker.read_text(encoding="utf-8")).get("verified") or {}
        if verified.get("ok") and verified.get("last_step") is not None:
            pins.add(int(verified["last_step"]))
    return pins


def checkpoint_requirements(cfg, stage):
    cfg = cfg.for_stage(stage)
    monitoring = stage == 'gad' and (cfg.get('actor_reward.fp32') or cfg.get('actor_reward.scorer_monitor'))
    return {
        "extensions_required": bool(active(cfg, 'rubric', stage) or active(cfg, 'replay', stage) or monitoring),
        "world_size": int(cfg.require('resources.n_gpus_per_node')) * int(cfg.require('resources.nnodes')),
    }


def plan_retention(stage_dir: Path, *, keep_checkpoints: int, extensions_required: bool = False,
                   world_size: int | None = None, checkpoint_steps=()) -> dict:
    if keep_checkpoints < 1:
        raise ValueError("keep_checkpoints must be positive")
    marker = stage_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return {"stage": str(stage_dir), "latest": None, "keep": [], "delete": [], "reason": "no latest marker"}
    latest = int(marker.read_text(encoding="utf-8").strip())
    found = checkpoint_dirs(stage_dir)
    pins = pinned_steps(stage_dir, found) | {step for step, _ in found if step in checkpoint_steps}
    older = [(step, path) for step, path in found if step < latest]
    if latest not in {step for step, _ in found}:
        return {"stage": str(stage_dir), "latest": latest, "keep": [], "delete": [],
                "reason": "latest checkpoint directory missing or linked"}
    beyond_limit = older[:max(0, len(older) + 1 - keep_checkpoints)]
    if all(step in pins for step, _ in beyond_limit):
        # The newest N directories and the pinned ones fit the retention policy.
        # A future rotation will validate which of the newest checkpoints can be kept.
        return {"stage": str(stage_dir), "latest": latest, "keep": [str(p) for _, p in found],
                "delete": [], "pinned": sorted(pins), "recent_steps": [], "reason": "no deletion candidates"}
    checkpoint = stage_dir / f"global_step_{latest}"
    roles = ("actor",) if stage_dir.name == "seqkd" else ("actor", "critic")

    def problems_for(path):
        # Retain the previous complete save if the new save lacks its tokenizer metadata.
        # Ordinary FSDP resume gets the tokenizer from the configured HF model.
        for role in roles:
            if not any((path / role / hint).is_file() for hint in TOKENIZER_HINTS + ("tokenizer.model", "spiece.model", "vocab.txt")):
                return [f"{path / role}: no tokenizer files"]
        return verify_checkpoint_dir(path, roles=roles, extensions_required=extensions_required, world_size=world_size)

    problems = problems_for(checkpoint)
    if problems:
        return {"stage": str(stage_dir), "latest": latest, "keep": [], "delete": [],
                "reason": "latest checkpoint incomplete: " + "; ".join(problems)}
    available = [(step, path) for step, path in found if step <= latest]
    recent_steps = [latest]
    for step, path in reversed(older):
        if len(recent_steps) == keep_checkpoints:
            break
        if not problems_for(path):
            recent_steps.append(step)
    recent_steps.reverse()
    keep_steps = set(recent_steps) | pins | {latest}
    keep = [str(path) for step, path in found if step in keep_steps or step > latest]
    delete = [str(path) for step, path in available if step not in keep_steps]
    return {"stage": str(stage_dir), "latest": latest, "keep": keep, "delete": delete,
            "pinned": sorted(pins), "recent_steps": recent_steps}


def apply_retention(stage_dir: Path, *, keep_checkpoints: int, extensions_required: bool = False,
                    world_size: int | None = None, checkpoint_steps=()) -> dict:
    plan = plan_retention(stage_dir, keep_checkpoints=keep_checkpoints,
                          extensions_required=extensions_required, world_size=world_size, checkpoint_steps=checkpoint_steps)
    removed = []
    for raw_path in plan["delete"]:
        path = Path(raw_path)
        marker = stage_dir / "latest_checkpointed_iteration.txt"
        if not marker.is_file() or marker.read_text().strip() != str(plan["latest"]):
            break
        if (path.parent.resolve() != stage_dir.resolve() or path.is_symlink()
                or not re.fullmatch(r"global_step_\d+", path.name)):
            raise RuntimeError(f"unsafe checkpoint path: {path}")
        step = int(path.name.rsplit("_", 1)[1])
        if step in pinned_steps(stage_dir, [(step, path)]):
            continue
        if path.exists():
            try:
                shutil.rmtree(path)
                removed.append(raw_path)
            except FileNotFoundError:
                pass  # the trainer may have rotated this directory already
    plan["delete"] = removed
    return plan


class RetentionMonitor:
    """Prune after the trainer advances its completed-checkpoint marker."""

    def __init__(self, stage_dir: Path, *, keep_checkpoints: int,
                 interval_seconds: int = 30, extensions_required: bool = False, world_size: int | None = None,
                 checkpoint_steps=()):
        self.stage_dir = stage_dir
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_steps = checkpoint_steps
        self.extensions_required = extensions_required
        self.world_size = world_size
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _once(self) -> None:
        try:
            plan = apply_retention(self.stage_dir, keep_checkpoints=self.keep_checkpoints,
                                   extensions_required=self.extensions_required, world_size=self.world_size,
                                   checkpoint_steps=self.checkpoint_steps)
            if plan["delete"]:
                print(f"[gad] checkpoint retention: removed {len(plan['delete'])} old "
                      f"checkpoint(s) under {self.stage_dir}", flush=True)
        except Exception as exc:
            print(f"[gad] WARNING: checkpoint retention failed for {self.stage_dir}: {exc}",
                  flush=True)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._once()

    def __enter__(self):
        self._once()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self._once()


def run_active(run_dir: Path) -> bool:
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            argv = (item / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if str(run_dir) in argv and any(
            marker in argv for marker in ("verl.trainer.main_ppo", "evaluate/evaluate.py",
                                          "sft.sample_teacher")
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--run-id", help="Override run.run_id")
    parser.add_argument("--stages", default="seqkd,warmup,gad")
    parser.add_argument("--keep-checkpoints", type=int, help="Override resources.keep_last_checkpoints")
    parser.add_argument("--apply", action="store_true", help="Delete planned old checkpoints")
    args = parser.parse_args()
    cfg = select_config(args, existing_only=True)
    if cfg.existing_run and args.run_id and args.run_id != str(cfg.get("run.run_id")):
        raise ValueError("--run-id cannot redirect an existing --run selection")
    run_id = args.run_id or str(cfg.get("run.run_id"))
    if not run_id or run_id == "auto":
        raise ValueError("run ID is auto; pass --run-id")
    keep_checkpoints = (args.keep_checkpoints if args.keep_checkpoints is not None
                        else cfg.get("resources.keep_last_checkpoints"))
    if keep_checkpoints is None:
        raise ValueError("set resources.keep_last_checkpoints or pass --keep-checkpoints")
    run_dir = Path(cfg.require("run.output_root")) / run_id
    if args.apply and run_active(run_dir):
        raise RuntimeError(f"training or evaluation is active for {run_dir}; retry after it finishes")
    for stage in [value.strip() for value in args.stages.split(",") if value.strip()]:
        if stage not in ("seqkd", "warmup", "gad"):
            raise ValueError(f"unknown stage: {stage}")
        stage_dir = run_dir / stage
        action = apply_retention if args.apply else plan_retention
        plan = action(stage_dir, keep_checkpoints=int(keep_checkpoints),
                      checkpoint_steps=cfg.for_stage(stage).checkpoint_steps(stage), **checkpoint_requirements(cfg, stage))
        print(f"[prune] {stage}: latest={plan['latest']}, keep={len(plan['keep'])}, "
              f"delete={len(plan['delete'])}", flush=True)
        if plan.get("reason"):
            print(f"[prune] {plan['reason']}", flush=True)
        for path in plan["delete"]:
            print(f"[prune] {'deleted' if args.apply else 'would delete'}: {path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"[prune] {exc}") from exc
