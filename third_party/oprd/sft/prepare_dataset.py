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
    git_commit,
    init_wandb,
    log_wandb_artifact,
    percentile,
    read_jsonl,
    sanitize_name,
    sha256_file,
)
from .quality import grade_output, validate_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw teacher rollouts into a registered LlamaFactory messages-format SFT dataset."
    )
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="dapo5k_teacher_sft")
    parser.add_argument("--selection", choices=["all", "valid", "correct"], default="all")
    parser.add_argument(
        "--max-per-prompt",
        type=int,
        default=1,
        help="Maximum selected responses per source prompt; 0 keeps every matching response.",
    )
    parser.add_argument(
        "--grade-correctness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Grade all candidates against DAPO ground truth for diagnostics.",
    )
    parser.add_argument("--keep-empty", action="store_true")
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


def normalize_prompt_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or "content" not in item:
            return None
        role = str(item.get("role", "user"))
        if role not in {"system", "user", "assistant", "tool", "function"}:
            return None
        messages.append({"role": role, "content": str(item["content"])})

    conversational = [message for message in messages if message["role"] != "system"]
    if not conversational or conversational[-1]["role"] != "user":
        return None
    return messages


def enrich_records(records: list[dict[str, Any]], grade_correctness: bool) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        response = str(record.get("response", ""))
        quality = validate_output(response)
        correctness = (
            grade_output(response, record.get("ground_truth")) if grade_correctness else record.get("teacher_correct")
        )
        enriched.append(
            {
                **record,
                "is_valid": quality.valid,
                "validation_reason": quality.reason,
                "has_boxed_answer": quality.has_boxed_answer,
                "teacher_correct": correctness,
            }
        )
    return enriched


def choose_records(
    records: list[dict[str, Any]],
    *,
    selection: str,
    max_per_prompt: int,
    keep_empty: bool,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["global_index"])].append(record)

    selected: list[dict[str, Any]] = []
    for global_index in sorted(grouped):
        candidates = sorted(grouped[global_index], key=lambda row: int(row.get("rollout_index", 0)))
        matching: list[dict[str, Any]] = []
        for candidate in candidates:
            response = str(candidate.get("response", ""))
            if not keep_empty and not response.strip():
                continue
            if selection == "valid" and not candidate.get("is_valid", False):
                continue
            if selection == "correct" and candidate.get("teacher_correct") is not True:
                continue
            matching.append(candidate)

        if max_per_prompt > 0:
            matching = matching[:max_per_prompt]
        selected.extend(matching)
    return selected


def to_sft_row(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = normalize_prompt_messages(record.get("raw_prompt"))
    if prompt is None:
        return None
    response = str(record.get("response", ""))
    return {
        "messages": prompt + [{"role": "assistant", "content": response}],
        "source_index": int(record["global_index"]),
        "rollout_index": int(record.get("rollout_index", 0)),
        "ground_truth": record.get("ground_truth"),
        "teacher_correct": record.get("teacher_correct"),
        "is_valid": bool(record.get("is_valid", False)),
        "validation_reason": record.get("validation_reason"),
        "teacher_output_tokens": int(record.get("output_tokens", 0)),
        "teacher_finish_reason": record.get("finish_reason"),
        "data_source": record.get("data_source"),
    }


def dataset_info(dataset_name: str, file_name: str) -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": file_name,
            # This repository's customized OpenAI converter injects an extra
            # `detailed thinking off` system message. The ShareGPT converter with
            # explicit OpenAI-style tags preserves the sampled prompt exactly.
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
                "observation_tag": "tool",
                "function_tag": "function",
            },
        }
    }


def main() -> int:
    args = build_parser().parse_args()
    if not args.raw_jsonl.is_file():
        raise FileNotFoundError(args.raw_jsonl)
    if args.max_per_prompt < 0:
        raise ValueError("--max-per-prompt must be non-negative")
    if args.selection == "correct" and not args.grade_correctness:
        raise ValueError("--selection=correct requires --grade-correctness")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = read_jsonl(args.raw_jsonl)
    if not raw_records:
        raise ValueError(f"No teacher samples found in {args.raw_jsonl}")
    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    for record in raw_records:
        key = (int(record["global_index"]), int(record.get("rollout_index", 0)))
        deduplicated.setdefault(key, record)
    records = [deduplicated[key] for key in sorted(deduplicated)]
    enriched = enrich_records(records, args.grade_correctness)
    selected = choose_records(
        enriched,
        selection=args.selection,
        max_per_prompt=args.max_per_prompt,
        keep_empty=args.keep_empty,
    )

    sft_rows: list[dict[str, Any]] = []
    malformed_prompts = 0
    for record in selected:
        row = to_sft_row(record)
        if row is None:
            malformed_prompts += 1
            continue
        sft_rows.append(row)
    if not sft_rows:
        raise ValueError(
            f"Selection `{args.selection}` produced no trainable rows from {len(enriched)} candidates."
        )

    output_path = args.output_dir / "teacher_sft.jsonl"
    info_path = args.output_dir / "dataset_info.json"
    stats_path = args.output_dir / "stats.json"
    manifest_path = args.output_dir / "manifest.json"
    atomic_write_jsonl(output_path, sft_rows)
    atomic_write_json(info_path, dataset_info(args.dataset_name, output_path.name))

    prompt_ids = {int(record["global_index"]) for record in enriched}
    selected_prompt_ids = {int(row["source_index"]) for row in sft_rows}
    correctness_values = [record.get("teacher_correct") for record in enriched]
    correctness_known = [value for value in correctness_values if isinstance(value, bool)]
    selected_correctness = [row.get("teacher_correct") for row in sft_rows]
    selected_known = [value for value in selected_correctness if isinstance(value, bool)]
    output_lengths = [int(row.get("teacher_output_tokens", 0)) for row in sft_rows]
    stats = {
        "selection": args.selection,
        "max_per_prompt": args.max_per_prompt,
        "raw_records": len(raw_records),
        "deduplicated_candidates": len(enriched),
        "source_prompts": len(prompt_ids),
        "selected_samples": len(sft_rows),
        "selected_prompts": len(selected_prompt_ids),
        "prompt_coverage": len(selected_prompt_ids) / len(prompt_ids) if prompt_ids else 0.0,
        "raw_valid_rate": (
            sum(bool(record.get("is_valid")) for record in enriched) / len(enriched) if enriched else 0.0
        ),
        "raw_correct_rate": (
            sum(bool(value) for value in correctness_known) / len(correctness_known) if correctness_known else None
        ),
        "selected_correct_rate": (
            sum(bool(value) for value in selected_known) / len(selected_known) if selected_known else None
        ),
        "malformed_prompts_dropped": malformed_prompts,
        "output_tokens_mean": statistics.fmean(output_lengths) if output_lengths else 0.0,
        "output_tokens_p50": percentile(output_lengths, 0.50),
        "output_tokens_p95": percentile(output_lengths, 0.95),
    }
    atomic_write_json(stats_path, stats)
    manifest = {
        "kind": "llamafactory_teacher_sft_dataset",
        "dataset_name": args.dataset_name,
        "raw_jsonl": str(args.raw_jsonl.resolve()),
        "raw_jsonl_sha256": sha256_file(args.raw_jsonl),
        "teacher_sft_sha256": sha256_file(output_path),
        "preparer_sha256": sha256_file(Path(__file__)),
        "quality_sha256": sha256_file(Path(__file__).with_name("quality.py")),
        "git_commit": git_commit(),
        "config": {
            "selection": args.selection,
            "max_per_prompt": args.max_per_prompt,
            "grade_correctness": args.grade_correctness,
            "keep_empty": args.keep_empty,
        },
        "stats": stats,
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    run_name = args.wandb_run_name or f"prepare-{args.dataset_name}-{args.selection}"
    run = init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        job_type="dataset_prepare",
        config=manifest["config"] | {"dataset_name": args.dataset_name, "raw_jsonl": str(args.raw_jsonl)},
    )
    if run is not None:
        numeric_stats = {key: value for key, value in stats.items() if isinstance(value, (int, float))}
        run.log({f"dataset/{key}": value for key, value in numeric_stats.items()})
        run.summary.update({f"dataset/{key}": value for key, value in numeric_stats.items()})
        try:
            import wandb

            table = wandb.Table(
                columns=["source_index", "ground_truth", "teacher_correct", "response"]
            )
            for row in sft_rows[:64]:
                table.add_data(
                    row["source_index"],
                    row.get("ground_truth"),
                    row.get("teacher_correct"),
                    row["messages"][-1]["content"][:4000],
                )
            run.log({"dataset/examples": table})
        except Exception as error:
            print(f"W&B dataset table skipped: {error}", flush=True)
        log_wandb_artifact(
            run,
            name=f"{sanitize_name(args.dataset_name)}-{args.selection}",
            artifact_type="teacher_sft_dataset",
            files=[output_path, info_path, stats_path, manifest_path],
            metadata=stats,
        )
        run.finish(exit_code=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
