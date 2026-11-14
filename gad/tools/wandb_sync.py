#!/usr/bin/env python3
"""Upload offline W&B runs, or rebuild them from the available console logs.

Training nodes have no internet, so stages log with WANDB_MODE=offline into
<run_dir>/wandb/offline-run-*. Run this afterwards on a machine with network access.

  wandb_sync.py runs/<id> [...]                  # list offline runs, upload nothing
  wandb_sync.py runs/<id> --upload               # upload unsynced offline runs
  wandb_sync.py runs/<id> --from-logs            # console-only run: rebuild offline runs locally
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PipelineError, read_json, sha256_file, write_json  # noqa: E402
from config import resolve_run  # noqa: E402

STAGES = ("seqkd", "warmup", "gad")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"(?:^|\s)step:(\d+) - (.*)$")
BACKFILL_TAG = "console-backfill"
BACKFILL_VERSION = 2  # Adds a stage-relative axis and tracks both source files.


def offline_runs(run_dir: Path) -> list[Path]:
    # Keep prior snapshots on disk, but upload only the latest backfill per stage.
    superseded = set()
    for marker in (run_dir / "wandb").glob("backfill_*.json"):
        superseded.update(read_json(marker).get("superseded_runs", []))
    return sorted(p for p in (run_dir / "wandb").glob("offline-run-*")
                  if p.is_dir() and p.name not in superseded)


def offline_stage(offline: Path) -> str | None:
    """Read recorded job_type, for both native runs and console backfills."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    for path in offline.glob("*.wandb"):
        store = DataStore()
        store.open_for_scan(str(path))
        try:
            while (data := store.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.HasField("run"):
                    return record.run.job_type or None
        finally:
            store.close()
    return None


def is_synced(offline: Path) -> bool:
    return any(offline.glob("*.wandb.synced"))


def parse_console_log(log: Path) -> dict[int, dict[str, float]]:
    """verl console lines `step:N - key:value - ...`; a resumed log keeps the last record per step."""
    records: dict[int, dict[str, float]] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        matched = STEP.search(ANSI.sub("", line).replace("\r", ""))
        if not matched:
            continue
        fields = {}
        for item in matched.group(2).split(" - "):
            key, _, value = item.partition(":")
            try:
                fields[key.strip()] = float(value)
            except ValueError:
                continue
        records[int(matched.group(1))] = fields
    return records


def validation_records(stage_dir: Path) -> dict[int, dict[str, float]]:
    """Numeric fields of validation_math_metrics.jsonl under val_math/<scope>/."""
    records: dict[int, dict[str, float]] = {}
    path = stage_dir / "validation_math_metrics.jsonl"
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else None
        if not row or "global_step" not in row:
            continue
        scope = row.get("validation_scope", "val")
        records.setdefault(int(row["global_step"]), {}).update(
            {f"val_math/{scope}/{k}": float(v) for k, v in row.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "global_step"})
    return records


def stage_start_step(run_dir: Path, stage: str) -> int:
    """Use the original handoff, not the first surviving/resumed log entry."""
    if stage != "gad":
        return 0
    handoff = read_json(run_dir / "handoff_manifest.json", default={})
    if "step" in handoff:
        return int(handoff["step"])
    plan = read_json(run_dir / "stage_gad.json", default={}).get("plan", {})
    if "start_step" in plan:
        return int(plan["start_step"])
    raise PipelineError("cannot determine GAD stage_step: need handoff_manifest.json or stage_gad.json")


def backfill_stage(run_dir: Path, stage: str, project: str, *, force: bool) -> str:
    import wandb

    log = run_dir / "logs" / f"{stage}.log"
    record_path = run_dir / "wandb" / f"backfill_{stage}.json"
    if not log.is_file():
        return f"{stage}: no console log, skipped"
    validation = run_dir / stage / "validation_math_metrics.jsonl"
    start = stage_start_step(run_dir, stage)
    source = {"backfill_version": BACKFILL_VERSION, "source_sha256": sha256_file(log),
              "validation_sha256": sha256_file(validation) if validation.is_file() else None,
              "start_step": start, "project": project}
    previous = read_json(record_path, default={})
    previous_offline = record_path.parent / Path(previous.get("offline_run", "missing")).name
    if not force and previous_offline.is_dir() and all(previous.get(k) == v for k, v in source.items()):
        return f"{stage}: sources unchanged, already rebuilt ({record_path.name})"
    steps = parse_console_log(log)
    for step, fields in validation_records(run_dir / stage).items():
        steps.setdefault(step, {}).update(fields)
    if not steps:
        return f"{stage}: no step records in {log.name}, skipped"
    summary = json.loads((run_dir / "run_summary.json").read_text()) if (run_dir / "run_summary.json").is_file() else {}
    run = wandb.init(
        mode="offline", dir=str(run_dir), project=project, group=run_dir.name, job_type=stage,
        name=f"{run_dir.name}_{stage}", tags=[BACKFILL_TAG],
        config={"gad_pipeline": summary.get("config_resolved", {}),
                "backfill": {"source": str(log), **source}},
        settings=wandb.Settings(console="off", x_disable_stats=True, silent=True))
    run.define_metric("training/stage_step")
    run.define_metric("*", step_metric="training/stage_step")
    for step in sorted(steps):
        metrics = {k: v for k, v in steps[step].items() if math.isfinite(v)}
        metrics["training/stage_step"] = step - start
        metrics["training/absolute_global_step"] = step
        run.log(metrics, step=step)
    offline = Path(run.dir).parent
    run.finish()
    superseded = previous.get("superseded_runs", [])
    if previous.get("offline_run"):
        superseded = sorted(set(superseded + [previous_offline.name]))
    write_json(record_path, {"stage": stage, "offline_run": str(offline), "source": str(log),
                             **source, "superseded_runs": superseded, "steps": len(steps),
                             "first_step": min(steps), "last_step": max(steps),
                             "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    note = " (previous snapshot was uploaded; its cloud run is retained)" if is_synced(previous_offline) else ""
    return f"{stage}: {len(steps)} steps ({min(steps)}..{max(steps)}) -> {offline.name}{note}"


def frozen_log_backend(run_dir: Path) -> str:
    import yaml
    raw = yaml.safe_load((run_dir / "train.yaml").read_text()) if (run_dir / "train.yaml").is_file() else {}
    return str(((raw or {}).get("run") or {}).get("log_backend", "console"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run directories or run ids")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--upload", action="store_true", help="Run `wandb sync` on unsynced offline runs")
    action.add_argument("--from-logs", action="store_true",
                        help="Rebuild offline runs from logs/<stage>.log (local only, nothing is uploaded)")
    parser.add_argument("--stages", default=",".join(STAGES),
                        help="Comma-separated stages to list, rebuild or upload")
    parser.add_argument("--project", help="W&B project (upload: overrides the recorded one; from-logs: default lmops-gad)")
    parser.add_argument("--entity")
    parser.add_argument("--force", action="store_true", help="Re-upload synced runs / rebuild existing backfills")
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages or set(stages) - set(STAGES):
        raise PipelineError(f"unknown stage(s): {sorted(set(stages) - set(STAGES))}")
    wandb_cli = Path(sys.executable).with_name("wandb")
    wandb_cli = str(wandb_cli) if wandb_cli.exists() else shutil.which("wandb")
    failed = False
    for value in args.runs:
        run_dir = resolve_run(value)
        print(f"== {run_dir.name}")
        if args.from_logs:
            if "wandb" in frozen_log_backend(run_dir) and not args.force:
                print("  run already logged to W&B natively; pass --force to also rebuild from console logs")
                continue
            for stage in stages:
                print("  " + backfill_stage(run_dir, stage, args.project or "lmops-gad", force=args.force))
            continue
        runs = []
        for offline in offline_runs(run_dir):
            stage = offline_stage(offline)
            if stage in stages:
                runs.append(offline)
            elif stage not in STAGES:
                print(f"  skipped {offline.name}: unrecognized stage {stage!r}")
        if not runs:
            print("  no offline W&B runs for the selected stages under wandb/")
            continue
        for offline in runs:
            print(f"  {'synced  ' if is_synced(offline) else 'unsynced'} {offline.name}")
        if not args.upload:
            continue
        pending = [p for p in runs if args.force or not is_synced(p)]
        if not pending:
            continue
        if not wandb_cli:
            raise PipelineError("wandb CLI not found next to this interpreter or on PATH")
        command = [wandb_cli, "sync", "--include-offline", "--mark-synced"]
        command += ["--include-synced"] if args.force else []
        command += ["--project", args.project] if args.project else []
        command += ["--entity", args.entity] if args.entity else []
        failed |= subprocess.run(command + [str(p) for p in pending]).returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
