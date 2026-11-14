from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    init_wandb,
    log_wandb_artifact,
    percentile,
    read_jsonl,
    sanitize_name,
    sha256_file,
)
from .quality import grade_output, validate_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score raw vLLM math rollouts with the verl math grader.")
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", default="math")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "OPRD-High-Entropy"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.getenv("WANDB_MODE", "disabled"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.raw_jsonl.is_file():
        raise FileNotFoundError(args.raw_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    for record in read_jsonl(args.raw_jsonl):
        key = (int(record["global_index"]), int(record.get("rollout_index", 0)))
        deduplicated.setdefault(key, record)
    if not deduplicated:
        raise ValueError(f"No evaluation samples found in {args.raw_jsonl}")

    scored: list[dict[str, Any]] = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(deduplicated):
        record = deduplicated[key]
        response = str(record.get("response", ""))
        quality = validate_output(response)
        correct = grade_output(response, record.get("ground_truth"))
        result = {
            **record,
            "is_valid": quality.valid,
            "validation_reason": quality.reason,
            "has_boxed_answer": quality.has_boxed_answer,
            "correct": correct,
        }
        scored.append(result)
        grouped[int(record["global_index"])].append(result)

    known_samples = [row for row in scored if isinstance(row.get("correct"), bool)]
    prompt_groups = [sorted(rows, key=lambda row: int(row.get("rollout_index", 0))) for rows in grouped.values()]
    known_prompt_groups = [rows for rows in prompt_groups if any(isinstance(row.get("correct"), bool) for row in rows)]
    first_correct = [
        bool(rows[0]["correct"])
        for rows in prompt_groups
        if rows and isinstance(rows[0].get("correct"), bool)
    ]
    best_correct = [any(row.get("correct") is True for row in rows) for rows in known_prompt_groups]
    lengths = [int(row.get("output_tokens", 0)) for row in scored]
    metrics = {
        "task": args.task_name,
        "num_prompts": len(grouped),
        "num_scored_prompts": len(known_prompt_groups),
        "num_samples": len(scored),
        "rollouts_per_prompt_mean": (
            statistics.fmean(len(rows) for rows in prompt_groups) if prompt_groups else 0.0
        ),
        "sample_accuracy_avg_at_n": (
            sum(bool(row["correct"]) for row in known_samples) / len(known_samples) if known_samples else None
        ),
        "first_rollout_accuracy": sum(first_correct) / len(first_correct) if first_correct else None,
        "best_of_n_accuracy": sum(best_correct) / len(best_correct) if best_correct else None,
        "valid_rate": sum(bool(row.get("is_valid")) for row in scored) / len(scored) if scored else 0.0,
        "boxed_rate": sum(bool(row.get("has_boxed_answer")) for row in scored) / len(scored) if scored else 0.0,
        "truncated_rate": (
            sum(row.get("finish_reason") == "length" for row in scored) / len(scored) if scored else 0.0
        ),
        "output_tokens_mean": statistics.fmean(lengths) if lengths else 0.0,
        "output_tokens_p50": percentile(lengths, 0.50),
        "output_tokens_p95": percentile(lengths, 0.95),
    }

    scored_path = args.output_dir / "scored_samples.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    atomic_write_jsonl(scored_path, scored)
    atomic_write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    run_name = args.wandb_run_name or f"eval-{args.task_name}"
    run = init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        job_type="eval",
        config={
            "task_name": args.task_name,
            "raw_jsonl": str(args.raw_jsonl.resolve()),
            "raw_jsonl_sha256": sha256_file(args.raw_jsonl),
        },
    )
    if run is not None:
        numeric_metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        run.log({f"eval/{key}": value for key, value in numeric_metrics.items()})
        run.summary.update({f"eval/{key}": value for key, value in numeric_metrics.items()})
        log_wandb_artifact(
            run,
            name=f"{sanitize_name(run_name)}-results",
            artifact_type="evaluation",
            files=[scored_path, metrics_path],
            metadata=numeric_metrics,
        )
        run.finish(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
