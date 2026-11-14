#!/usr/bin/env python3
"""Compare OPRD benchmark grades with training-aligned CARD grades."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def collect_legacy(root: Path) -> list[dict]:
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


def interval_text(row: dict) -> str:
    """Display a single accuracy only when it is certified; otherwise show the interval."""
    empty = percent(None)
    if row.get("evaluation_complete"):
        return percent(row.get("sample_accuracy", row.get("pass_at_1")))
    lower, upper = row.get("sample_accuracy_lower"), row.get("sample_accuracy_upper")
    if lower is None or upper is None:
        return empty
    text = f"{percent(lower)}-{percent(upper)}"
    coverage = row.get("coverage")
    if coverage is not None:
        text += f" cov {percent(coverage)}"
    return text


def collect(root: Path, legacy=False):
    if legacy:return collect_legacy(root)
    plan = load(root / 'plan.json') or {}
    rows = []
    tasks={p.parent.parent for p in root.glob('*/*/scores/metrics.json')}
    tasks.update(p.parent for p in root.glob('*/*/state.json'))
    for task_dir in sorted(tasks):
        state=load(task_dir/'state.json') or {}
        path=Path(state.get('metrics_path',task_dir/'scores/metrics.json'))
        metrics = load(path)
        if not metrics:continue
        if metrics.get('scorer_version') not in ('math_unified_v1', 'choice_exact_v1', 'choice_exact_v2'):continue
        sampling = load(task_dir / 'generation/sampling_config.json') or {}
        protocol = plan.get('protocol', {})
        task = task_dir.name
        rows.append({'method':task_dir.parent.name,'task':task,'questions':metrics.get('num_prompts'),
            'n':metrics.get('samples_per_question', sampling.get('num_rollouts')),
            'max_new_tokens':sampling.get('max_new_tokens', protocol.get('max_tokens')),
            'seed':sampling.get('seed', protocol.get('seed')),
            'pass_at_1':metrics.get('pass_at_1'),'sample_accuracy':metrics.get('sample_accuracy'),
            'sample_accuracy_lower':metrics.get('sample_accuracy_lower'),
            'sample_accuracy_upper':metrics.get('sample_accuracy_upper'),
            'coverage':metrics.get('coverage'),
            'confirmed_correct':metrics.get('confirmed_correct'),
            'confirmed_incorrect':metrics.get('confirmed_incorrect'),
            'unresolved_samples':metrics.get('unresolved_samples'),
            'sample_index_complete':metrics.get('sample_index_complete'),
            'evaluation_complete':metrics.get('evaluation_complete'),
            'first_rollout_accuracy':metrics.get('first_rollout_accuracy'),
            'any_success_at_n':metrics.get('any_success_at_n'),
            'parse_failure_rate':metrics.get('parse_failure_rate'),'no_answer_rate':metrics.get('no_answer_rate'),
            'grader_error_rate':metrics.get('grader_error_rate'),
            'truncated_rate':metrics.get('truncated_rate'),
            'output_tokens_mean':metrics.get('output_tokens_mean'),
            'scorer_version':metrics.get('scorer_version'),'scorer_fingerprint':metrics.get('scorer_fingerprint'),
            'scoring_fingerprint':(metrics.get('scorer_fingerprint') or '')[:12],
            'post_process':'+'.join(metrics[k]['version'] for k in ('eval_postprocess','answer_forms')
                                    if isinstance(metrics.get(k), dict) and metrics[k].get('version')),
            'aggregation_protocol':json.dumps(metrics.get('aggregation_protocol') or {}, sort_keys=True),
            'training_scorer_fingerprint':(state.get('training_scorer') or {}).get('fingerprint'),
            'evaluation_scorer_fingerprint':(state.get('evaluation_scorer') or {}).get('fingerprint')})
    return rows

def percent(value) -> str:
    return "-" if value is None else f"{100 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Evaluation profile output directory")
    parser.add_argument("--csv", type=Path)
    parser.add_argument('--legacy', action='store_true', help='Explicit historical dual-scoring table')
    args = parser.parse_args()
    rows = collect(args.root, args.legacy)
    if not rows:
        raise SystemExit(f"no evaluation metrics under {args.root}")
    if args.legacy:
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
    else:
        print('method task q n max_tokens seed accuracy first any_success trunc mean_tokens parse_fail grader_error complete scorer')
        for row in rows:
            print(f"{row['method']} {row['task']} {row['questions']} {row['n']} {row['max_new_tokens']} {row['seed']} "
                  f"{interval_text(row)} {percent(row['first_rollout_accuracy'])} {percent(row['any_success_at_n'])} "
                  f"{percent(row['truncated_rate'])} {row['output_tokens_mean']} "
                  f"{percent(row['parse_failure_rate'])} {percent(row['grader_error_rate'])} {row['evaluation_complete']} "
                  f"{(row['scorer_version'] or '?')}:{(row['scorer_fingerprint'] or '')[:12]}"
                  + (f"+{row['post_process']}" if row['post_process'] else ''))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list({key for row in rows for key in row})
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
