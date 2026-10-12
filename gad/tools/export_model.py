#!/usr/bin/env python3
"""Export the latest saved SeqKD/Warmup/GAD actor as a verified HF model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from config import Config, load_config, add_config_arguments, select_config
from prepare_gad_checkpoint import check_checkpoint


def training_active(run_dir: Path) -> bool:
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            argv = (item / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "verl.trainer.main_ppo" in argv and str(run_dir) in argv:
            return True
    return False


def export(config_path: Path, stage: str, *, step_arg: str, run_id: str | None,
           dry_run: bool, resolved_config: Config | None = None) -> Path:
    cfg = resolved_config or load_config(config_path)
    if cfg.existing_run and run_id and run_id != str(cfg.get("run.run_id")):
        raise ValueError("--run-id cannot redirect an existing --run selection")
    selected_run_id = run_id or str(cfg.get("run.run_id"))
    if not selected_run_id or selected_run_id == "auto":
        raise ValueError("run ID is auto; pass --run-id explicitly")
    run_dir = Path(cfg.require("run.output_root")) / selected_run_id
    stage_dir = run_dir / stage
    marker = stage_dir / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"latest checkpoint marker missing: {marker}")
    latest = int(marker.read_text(encoding="utf-8").strip())
    step = latest if step_arg == "latest" else int(step_arg)
    if step < 1 or step > latest:
        raise ValueError(f"requested step {step} is outside saved range 1..{latest}")
    checkpoint = stage_dir / f"global_step_{step}"
    problems = check_checkpoint(checkpoint, roles=("actor",))
    if problems:
        raise RuntimeError("checkpoint is incomplete: " + "; ".join(problems))
    actor_hf = checkpoint / "actor" / "huggingface"
    manifest = stage_dir / f"eval_export_step_{step}.json"
    print(f"[export] {stage} global_step_{step}: {checkpoint}")
    print(f"[export] output: {actor_hf}")
    if dry_run:
        print("[export] dry-run: no files written")
        return actor_hf
    if training_active(run_dir):
        raise RuntimeError(f"training is active for {run_dir}; export after it finishes")

    for existing_manifest in (manifest, stage_dir / "eval_export_manifest.json"):
        if not existing_manifest.is_file():
            continue
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        weight_files = list(actor_hf.glob("*.safetensors"))
        if (previous.get("ok") and previous.get("step") == step
                and previous.get("actor_hf") == str(actor_hf)
                and weight_files and all(path.stat().st_size > 0 for path in weight_files)):
            if existing_manifest != manifest:
                manifest.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
            print(f"[export] already verified: {manifest}")
            return actor_hf

    python_bin = str(cfg.require("run.python_bin"))
    command = [
        python_bin, str(Path(__file__).with_name("prepare_gad_checkpoint.py")),
        "--mode", "export-final", "--gad-dir", str(stage_dir), "--step", str(step),
        "--python-bin", python_bin,
        "--merge-tool", str(Path(__file__).with_name("merge_model2hf.py")),
        "--manifest", str(manifest),
    ]
    subprocess.run(command, check=True)
    result = json.loads(manifest.read_text(encoding="utf-8"))
    if (not result.get("ok") or result.get("step") != step
            or result.get("actor_hf") != str(actor_hf)):
        raise RuntimeError(f"export verification failed: {manifest}")
    print(f"[export] verified: {actor_hf}")
    return actor_hf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--stage", choices=("seqkd", "warmup", "gad"), required=True)
    parser.add_argument("--step", default="latest", help="Latest saved step (default), or its explicit number")
    parser.add_argument("--run-id", help="Override run.run_id when it is auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = select_config(args, existing_only=True)
    export(cfg.path, args.stage, step_arg=args.step, run_id=args.run_id,
           dry_run=args.dry_run, resolved_config=cfg)


if __name__ == "__main__":
    main()
