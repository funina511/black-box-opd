#!/usr/bin/env python3
"""One-time, resumable migration of the Qwen3 experiment's local artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RLRM = ROOT.parent
RUN_ID = "qwen3-1.7b-4x3090"
RUN = ROOT / "runs/gad_qwen3_1.7b" / RUN_ID
DATA = ROOT / "data/dapo_math_17k_gad"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def move_with_link(source: Path, target: Path) -> None:
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise RuntimeError(f"unexpected link: {source}")
        return
    if not source.exists():
        if not target.exists():
            raise RuntimeError(f"neither source nor target exists: {source} -> {target}")
        source.symlink_to(target, target_is_directory=target.is_dir())
        return
    if target.exists():
        raise RuntimeError(f"target exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError(f"cross-filesystem move refused: {source}")
    source.rename(target)
    source.symlink_to(target, target_is_directory=target.is_dir())


def inventory(directory: Path) -> list[dict]:
    return [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and not p.is_symlink()]


def main() -> None:
    old_run = RLRM / "runs/gad_qwen3_1.7b" / RUN_ID
    old_data = RLRM / "dataset/dapo_math_17k_gad"
    marker = old_run / "seqkd/latest_checkpointed_iteration.txt"
    if marker.read_text().strip() != "600":
        raise RuntimeError(f"unexpected latest checkpoint: {marker.read_text()!r}")
    checkpoint = old_run / "seqkd/global_step_600"
    required = [checkpoint / "data.pt"] + [checkpoint / "actor" / f"{kind}_world_size_4_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state") for rank in range(4)]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"incomplete checkpoint: {missing}")
    if os.system("pgrep -f '[g]ad/tools/pipeline.py|[r]un_all.sh|[r]ay::.*Trainer' >/dev/null") == 0:
        raise RuntimeError("training process appears active")

    data_names = ("gad_train.parquet", "gad_dev.parquet", "teacher_index.jsonl", "manifest.json", "split_manifest.json")
    before = {name: {"bytes": (old_data / name).stat().st_size, "sha256": sha256(old_data / name)} for name in data_names}
    run_files = inventory(old_run)
    move_with_link(old_data, DATA)
    move_with_link(old_run, RUN)
    after = {name: {"bytes": (DATA / name).stat().st_size, "sha256": sha256(DATA / name)} for name in data_names}
    if before != after or inventory(RUN) != run_files:
        raise RuntimeError("migration verification failed")
    (RUN / "reports").mkdir(exist_ok=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
              "data": {"old": str(old_data), "new": str(DATA), "files": before},
              "run": {"old": str(old_run), "new": str(RUN), "files": run_files},
              "checkpoint": {"latest_step": 600, "required_files": [str(p.relative_to(old_run)) for p in required]}}
    (RUN / "reports/migration_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"migrated {len(run_files)} run files and verified {len(before)} data hashes")


if __name__ == "__main__":
    main()
