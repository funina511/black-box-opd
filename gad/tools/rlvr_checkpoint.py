"""Keep evaluation exports independently of FSDP's rolling resume checkpoints."""

import os
from pathlib import Path
import shutil


def saved_actor_paths(loaded_actor):
    """Recover FSDP's oldest-first rotation list from the loaded RLVR run."""
    checkpoint = Path(loaded_actor).parent
    latest = int(checkpoint.name.removeprefix("global_step_"))
    prior = []
    for path in checkpoint.parent.glob("global_step_*/actor"):
        number = path.parent.name.removeprefix("global_step_")
        if number.isdecimal() and int(number) <= latest and (path.parent / "data.pt").is_file():
            prior.append((int(number), str(path)))
    return [path for _, path in sorted(prior)]


def publish_eval_model(stage_dir, step):
    stage_dir = Path(stage_dir)
    models = stage_dir / "eval_models"
    target = models / f"step_{step}"
    if target.is_dir():
        return target
    models.mkdir(exist_ok=True)
    source = stage_dir / f"global_step_{step}" / "actor/huggingface"
    partial = models / f".step_{step}.partial"
    # The launcher holds an exclusive run lock. An interrupted export can
    # leave this unpublished directory behind, even when the checkpoint is good.
    if partial.exists():
        shutil.rmtree(partial)
    shutil.copytree(source, partial, copy_function=os.link)
    partial.rename(target)
    return target


def commit_checkpoint(stage_dir, step, eval_steps):
    stage_dir = Path(stage_dir)
    latest = stage_dir / "latest_checkpointed_iteration.txt"
    writing = latest.with_suffix(".tmp")
    writing.write_text(str(step))
    writing.replace(latest)
    if step in eval_steps:
        publish_eval_model(stage_dir, step)
