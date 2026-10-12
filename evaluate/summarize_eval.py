#!/usr/bin/env python3
"""Compare OPRD benchmark grades with training-aligned CARD grades."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def collect(root: Path) -> list[dict]:
    rows = []
    for method_dir in sorted(root.iterdir() if root.is_dir() else []):
        if not method_dir.is_dir():
            continue
        for task_dir in sorted(method_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            oprd = load(task_dir / "scores" / "metrics.json")
            card = load(task_dir / "scores" / "card_metrics.json")
            if oprd is None and card is None:
                continue
            card_task = (card or {}).get("benchmarks", {}).get(task_dir.name, {})
            rows.append({
                "method": method_dir.name, "task": task_dir.name,
                "questions": (oprd or {}).get("num_prompts", card_task.get("questions")),
                "samples_per_question": (oprd or {}).get("rollouts_per_prompt_mean", card_task.get("samples_per_question")),
                "oprd_sample_accuracy": (oprd or {}).get("sample_accuracy_avg_at_n"),
                "oprd_first_accuracy": (oprd or {}).get("first_rollout_accuracy"),
                "oprd_best_of_n": (oprd or {}).get("best_of_n_accuracy"),
                "card_pass_at_1": card_task.get("pass_at_1"),
                "card_any_success": card_task.get("pass_at_n_any_success"),
                "truncated_rate": (oprd or {}).get("truncated_rate", card_task.get("truncation_rate")),
            })
    return rows


def percent(value) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Evaluation profile output directory")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    rows = collect(args.root)
    if not rows:
        raise SystemExit(f"no evaluation metrics under {args.root}")
    print("method       task          q    n   OPRD sample  OPRD first  OPRD best   CARD pass@1  CARD any  trunc")
    for row in rows:
        print(f"{row['method']:<12} {row['task']:<12} {row['questions'] or 0:>3} "
              f"{(row['samples_per_question'] or 0):>4.1f}   "
              f"{percent(row['oprd_sample_accuracy']):>10}  "
              f"{percent(row['oprd_first_accuracy']):>10}  "
              f"{percent(row['oprd_best_of_n']):>9}  "
              f"{percent(row['card_pass_at_1']):>11}  "
              f"{percent(row['card_any_success']):>8}  "
              f"{percent(row['truncated_rate']):>5}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
