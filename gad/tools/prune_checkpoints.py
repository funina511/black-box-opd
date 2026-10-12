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

from config import load_config, add_config_arguments, select_config
from prepare_gad_checkpoint import check_checkpoint


def checkpoint_dirs(stage_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for child in stage_dir.iterdir() if stage_dir.is_dir() else []:
        match = re.fullmatch(r"global_step_(\d+)", child.name)
        if match and child.is_dir() and not child.is_symlink():
            found.append((int(match.group(1)), child))
    return sorted(found)


def pinned_steps(stage_dir: Path) -> set[int]:
    pins = set()
    for step, checkpoint in checkpoint_dirs(stage_dir):
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


def plan_retention(stage_dir: Path, *, keep_checkpoints: int) -> dict:
    if keep_checkpoints < 1:
        raise ValueError("keep_checkpoints must be positive")
    marker = stage_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return {"stage": str(stage_dir), "latest": None, "keep": [], "delete": [], "reason": "no latest marker"}
    latest = int(marker.read_text(encoding="utf-8").strip())
    checkpoint = stage_dir / f"global_step_{latest}"
    roles = ("actor",) if stage_dir.name == "seqkd" else ("actor", "critic")
    problems = check_checkpoint(checkpoint, roles=roles)
    if problems:
        return {"stage": str(stage_dir), "latest": latest, "keep": [], "delete": [],
                "reason": "latest checkpoint incomplete: " + "; ".join(problems)}
    found = checkpoint_dirs(stage_dir)
    available = [(step, path) for step, path in found if step <= latest]
    complete = [step for step, path in available if not check_checkpoint(path, roles=roles)]
    recent_steps = complete[-keep_checkpoints:]
    pins = pinned_steps(stage_dir)
    keep_steps = set(recent_steps) | pins | {latest}
    keep = [str(path) for step, path in found if step in keep_steps or step > latest]
    delete = [str(path) for step, path in available if step not in keep_steps]
    return {"stage": str(stage_dir), "latest": latest, "keep": keep, "delete": delete,
            "pinned": sorted(pins), "recent_steps": recent_steps}


def apply_retention(stage_dir: Path, *, keep_checkpoints: int) -> dict:
    plan = plan_retention(stage_dir, keep_checkpoints=keep_checkpoints)
    for raw_path in plan["delete"]:
        path = Path(raw_path)
        if (path.parent.resolve() != stage_dir.resolve() or path.is_symlink()
                or not re.fullmatch(r"global_step_\d+", path.name)):
            raise RuntimeError(f"unsafe checkpoint path: {path}")
        if path.exists():
            try:
                shutil.rmtree(path)
            except FileNotFoundError:
                pass  # the trainer may have rotated this directory already
    return plan


class RetentionMonitor:
    """Prune after the trainer advances its completed-checkpoint marker."""

    def __init__(self, stage_dir: Path, *, keep_checkpoints: int,
                 interval_seconds: int = 30):
        self.stage_dir = stage_dir
        self.keep_checkpoints = keep_checkpoints
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _once(self) -> None:
        try:
            plan = apply_retention(self.stage_dir, keep_checkpoints=self.keep_checkpoints)
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
        plan = plan_retention(stage_dir, keep_checkpoints=int(keep_checkpoints))
        print(f"[prune] {stage}: latest={plan['latest']}, keep={len(plan['keep'])}, "
              f"delete={len(plan['delete'])}", flush=True)
        if plan.get("reason"):
            print(f"[prune] {plan['reason']}", flush=True)
        for path in plan["delete"]:
            print(f"[prune] {'delete' if args.apply else 'would delete'}: {path}", flush=True)
        if args.apply:
            apply_retention(stage_dir, keep_checkpoints=int(keep_checkpoints))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"[prune] {exc}") from exc
