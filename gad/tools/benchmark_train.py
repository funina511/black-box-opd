#!/usr/bin/env python3
"""Isolated full-state Warmup short runs using the existing pipeline planner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PipelineError, ensure_dir, sha256_file, write_json  # noqa: E402
from config import load_config, load_run_config, resolve_run, validate_config, validate_paths  # noqa: E402
from pipeline import (plan_stage, steps_per_epoch, verify_checkpoint_dir,  # noqa: E402
                      dataset_rows)
from analyze_training_perf import parse_steps, summarize_steps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = None
DEFAULT_CANDIDATES = ROOT / "gad/configs/benchmark_candidates.yaml"
DEFAULT_SOURCE = ROOT / "runs/gad_qwen3_1.7b/qwen3-1.7b-4x3090"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def require_complete_checkpoint(source: Path, step: int) -> Path:
    checkpoint = source / "warmup" / f"global_step_{step}"
    problems = verify_checkpoint_dir(checkpoint, roles=("actor", "critic"))
    for role in ("actor", "critic"):
        for kind in ("model", "optim", "extra_state"):
            for rank in range(4):
                name = f"{kind}_world_size_4_rank_{rank}.pt"
                if not (checkpoint / role / name).is_file():
                    problems.append(f"missing {role}/{name}")
    if problems:
        raise PipelineError("source checkpoint is incomplete: " + "; ".join(problems))
    return checkpoint


def assert_no_training() -> None:
    current = os.getpid()
    output = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
    for line in output.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == current:
            continue
        command = fields[1]
        if ("verl.trainer.main_ppo" in command or "gad/tools/pipeline.py" in command) and "ps -eo" not in command:
            raise PipelineError(f"another training process is active: {line.strip()}")


def copy_checkpoint(source: Path, target: Path) -> None:
    if target.exists():
        raise PipelineError(f"benchmark checkpoint target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(target)], check=True)
    source_files = {str(p.relative_to(source)): p.stat().st_size for p in source.rglob("*") if p.is_file()}
    target_files = {str(p.relative_to(target)): p.stat().st_size for p in target.rglob("*") if p.is_file()}
    if source_files != target_files:
        raise PipelineError("benchmark checkpoint copy differs in file names or sizes")
    for rel in ("data.pt", "actor/extra_state_world_size_4_rank_0.pt",
                "critic/extra_state_world_size_4_rank_0.pt"):
        if sha256_file(source / rel) != sha256_file(target / rel):
            raise PipelineError(f"benchmark checkpoint copy hash mismatch: {rel}")


def compose_hydra(plan: dict, run_dir: Path) -> None:
    env = os.environ.copy()
    env.update(plan["env"])
    command = plan["argv"] + ["--cfg", "job", "--resolve"]
    result = subprocess.run(command, cwd=ROOT / "gad", env=env,
                            capture_output=True, text=True, timeout=180)
    reports = ensure_dir(run_dir / "reports")
    (reports / "hydra_composed.yaml").write_text(result.stdout, encoding="utf-8")
    (reports / "hydra_composition.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise PipelineError(f"Hydra composition failed ({result.returncode}): {result.stderr[-1200:]}")


def sample_gpu(stop: threading.Event, target: Path) -> None:
    query = ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]
    with target.open("w", encoding="utf-8") as stream:
        while not stop.is_set():
            try:
                output = subprocess.check_output(query, text=True, timeout=15)
                rows = []
                for line in output.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) == 3:
                        rows.append({"gpu": int(parts[0]), "memory_used_mib": int(parts[1]),
                                     "utilization_pct": int(parts[2])})
                stream.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                         "gpus": rows}) + "\n")
                stream.flush()
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                stream.write(json.dumps({"error": str(exc)}) + "\n")
                stream.flush()
            stop.wait(3)


def run_benchmark(plan: dict, run_dir: Path) -> int:
    log = run_dir / "logs/warmup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    gpu_file = run_dir / "reports/gpu_samples.jsonl"
    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, gpu_file), daemon=True)
    env = os.environ.copy()
    env.update(plan["env"])
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(plan["argv"], cwd=ROOT / "gad", env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        monitor.start()
        try:
            return process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            return process.wait()
        finally:
            stop.set()
            monitor.join(timeout=10)


def backend_fingerprints(backend: Path) -> dict[str, str]:
    paths = (
        "verl/trainer/ppo/core_algos.py",
        "verl/trainer/ppo/ray_trainer.py",
        "verl/workers/fsdp_workers.py",
        "verl/workers/critic/dp_critic.py",
        "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py",
    )
    return {name: sha256_file(backend / name) for name in paths}


def peak_memory(path: Path, gpu_ids: set[int]) -> dict:
    peaks: dict[int, int] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            for row in json.loads(line).get("gpus", []):
                gpu = row["gpu"]
                if gpu in gpu_ids:
                    peaks[gpu] = max(peaks.get(gpu, 0), row["memory_used_mib"])
    return {str(gpu): value for gpu, value in sorted(peaks.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--source-run", "--run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-step", type=int)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true", help="Copy checkpoint and run the GPU benchmark")
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.warmup_steps < args.steps:
        parser.error("steps must be positive and warmup-steps must be smaller than steps")
    import yaml
    candidates = yaml.safe_load(args.candidates.read_text()) or {}
    if args.candidate not in candidates:
        parser.error(f"unknown candidate {args.candidate}; choose from {', '.join(candidates)}")
    source_run = resolve_run(args.source_run)
    marker = source_run / "warmup/latest_checkpointed_iteration.txt"
    if not marker.is_file():
        parser.error(f"source marker missing: {marker}")
    source_step = int(args.source_step or marker.read_text().strip())
    source_checkpoint = require_complete_checkpoint(source_run, source_step)
    run_id = args.run_id or f"warmup_{args.candidate}_from{source_step}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if not SAFE_ID.fullmatch(run_id):
        parser.error("run-id must contain only letters, numbers, dot, underscore or dash")
    if args.execute:
        assert_no_training()
    cfg = load_config(args.config) if args.config else load_run_config(source_run).for_stage("warmup")
    cfg.identity_data = None
    cfg.data.update(candidates[args.candidate] or {})
    cfg.data.update({"run.output_root": str(ROOT / "runs/performance_benchmarks"),
                     "run.run_id": run_id,
                     "run.val_before_train": False,
                     "run.eval_after_train": False,
                     "validation.quick_enabled": False,
                     "validation.require_math_metrics": False,
                     "resources.test_freq": 0,
                     "budget.stage_epochs.warmup": None,
                     "budget.warmup_steps": source_step + args.steps})
    problems = validate_config(cfg) + validate_paths(cfg, check_exists=True)
    if problems:
        raise PipelineError("benchmark config invalid: " + "; ".join(problems))
    run_dir = Path(cfg.require("run.output_root")) / run_id
    if run_dir.exists():
        raise PipelineError(f"benchmark run_id already exists: {run_dir}")
    plan = plan_stage(cfg, run_dir, "warmup", steps_per_epoch_value=steps_per_epoch(cfg),
                      install_guard=args.execute)
    plan["argv"].append("trainer.resume_mode=auto")
    print(f"candidate={args.candidate} source=warmup/global_step_{source_step} target={source_step + args.steps}")
    print(f"run_dir={run_dir}")
    print(f"gpu_memory_utilization={cfg.get('rollout.gpu_memory_utilization')} "
          f"critic_micro={cfg.get('optim.critic_micro_batch_size_per_gpu')}")
    if not args.execute:
        from tempfile import TemporaryDirectory
        with TemporaryDirectory(prefix="lmops-hydra-") as temporary:
            compose_hydra(plan, Path(temporary))
        print("dry-run: Hydra composed; no run directory, checkpoint copy or GPU training")
        return 0
    reports = ensure_dir(run_dir / "reports")
    config_path = reports / "resolved_benchmark_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=True, allow_unicode=True))
    (run_dir / "current_resolved_config.txt").write_text(str(config_path) + "\n")
    compose_hydra(plan, run_dir)
    copy_checkpoint(source_checkpoint, run_dir / "warmup" / source_checkpoint.name)
    (run_dir / "warmup/latest_checkpointed_iteration.txt").write_text(str(source_step))
    started = datetime.now(timezone.utc).isoformat()
    code = run_benchmark(plan, run_dir)
    log = run_dir / "logs/warmup.log"
    records = [r for r in parse_steps(log) if r["step"] > source_step]
    latest = run_dir / "warmup/latest_checkpointed_iteration.txt"
    latest_step = int(latest.read_text().strip()) if latest.is_file() else None
    expected = source_step + args.steps
    if latest_step == expected:
        try:
            require_complete_checkpoint(run_dir, expected)
            full_state = []
        except PipelineError as exc:
            full_state = [str(exc)]
    else:
        full_state = ["target checkpoint missing"]
    success = code == 0 and len(records) == args.steps and latest_step == expected and not full_state
    gpu_ids = {int(value) for value in str(cfg.get("resources.gpus")).split(",") if value.strip().isdigit()}
    report = {"kind": "performance_benchmark", "candidate": args.candidate,
              "source_run": str(source_run), "source_checkpoint": str(source_checkpoint),
              "source_step": source_step, "target_step": expected, "warmup_steps": args.warmup_steps,
              "completed_steps": [r["step"] for r in records], "latest_checkpoint_step": latest_step,
              "returncode": code, "ok": success, "checkpoint_problems": full_state,
              "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
              "config": str(config_path), "argv": plan["argv"],
              "backend_commit": plan["backend_commit"],
              "backend_file_sha256": backend_fingerprints(Path(plan["backend"])),
              "gpu_peak_memory_mib_per_card": peak_memory(run_dir / "reports/gpu_samples.jsonl", gpu_ids),
              "training": summarize_steps(records, stage="warmup", skip_first=args.warmup_steps,
                                          prompt_count=int(cfg.get("data.train_batch_size")),
                                          candidate_count=int(cfg.get("rollout.n_warmup")))}
    write_json(reports / "benchmark_result.json", report)
    if not success:
        print(f"benchmark failed; evidence retained at {run_dir}", file=sys.stderr)
        return 1
    official_reports = ensure_dir(source_run / "reports/performance")
    result_copy = official_reports / f"{run_id}.json"
    write_json(result_copy, report)
    print(f"benchmark passed: {result_copy}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PipelineError, subprocess.SubprocessError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(2)
