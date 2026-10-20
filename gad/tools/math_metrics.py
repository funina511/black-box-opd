#!/usr/bin/env python3
"""Version-aware math metrics for dev monitoring and external evaluation.

The default is math_unified_v1 (explicit answers/boxes and exact mathematical
normalization). An explicitly selected sample/math_scorer.py keeps historical
CARD extraction/string comparison. Self-tests use fixtures for the loaded
version. Grader failures and malformed/ambiguous answers make evaluation
incomplete; they never become official incorrect labels or complete accuracy.
No metric here is fed into actor reward or advantages.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

GAD_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCORER_PATHS = (
    Path(os.environ.get("GAD_MATH_SCORER", "")) if os.environ.get("GAD_MATH_SCORER") else None,
    GAD_DIR.parent / "scoring" / "math.py",
    GAD_DIR.parent / "sample" / "math_scorer.py",
    GAD_DIR / "sample" / "math_scorer.py",
)


class MathScorerNotFound(RuntimeError):
    pass


def load_scorer(scorer_path: str | Path | None = None):
    """Import an explicitly selected scorer, or the unified default."""
    inherited = json.loads(os.environ.get('GAD_SCORING_MATH_CONFIG','{}'))
    legacy = [GAD_DIR.parent/'sample/math_scorer.py'] if not scorer_path and inherited.get('version') == 'legacy_card_v1' else []
    candidates = [Path(scorer_path)] if scorer_path else legacy + [p for p in DEFAULT_SCORER_PATHS if p]
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            spec = importlib.util.spec_from_file_location("gad_math_scorer", path)
            if spec is None or spec.loader is None:  # pragma: no cover
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.__gad_source__ = str(path)
            return module
    raise MathScorerNotFound(
        "could not find the vendored CARD math scorer. Expected LMOps/sample/math_scorer.py; "
        f"tried: {[str(p) for p in candidates]}. Pass --scorer-path explicitly."
    )


SCORER = None


def scorer():
    global SCORER
    inherited = json.loads(os.environ.get('GAD_SCORING_MATH_CONFIG','{}'))
    if SCORER is None or inherited.get('version') == 'legacy_card_v1' and getattr(SCORER,'VERSION',None) == 'math_unified_v1':
        SCORER = load_scorer()
    return SCORER


def score_response(response: str, ground_truth: str, *, scorer_module=None) -> dict:
    module = scorer_module or scorer()
    try:
        result = module.compute_score(response or "", ground_truth if ground_truth is not None else "")
    except Exception as exc:
        return {'acc':None, 'score':None, 'pred':None, 'answer_parsed':False,
                'evaluation_valid':False, 'grade_status':'error', 'parse_status':'unknown',
                'error_reason':f'{type(exc).__name__}: {exc}'}
    prediction = result["pred"]
    if result.get("scorer_version") == "math_unified_v1":
        return {**result, "acc": bool(result["acc"]) if result['acc'] is not None else None,
                "answer_parsed": result["parse_status"] == "ok"}
    return {
        "score": float(result["score"]),
        "acc": bool(result["acc"]),
        "pred": prediction,
        "answer_parsed": prediction != "[INVALID]",
    }


def _quantiles(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def q(frac: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, int(round(frac * (len(ordered) - 1)))))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": q(0.50),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def aggregate(records: Iterable[dict], *, answer_token_limit: int | None = None) -> dict:
    """Aggregate scored records.

    Each record needs ``acc`` (bool) and optionally ``response``, ``finish_reason``,
    ``truncated``, ``num_tokens``/``response_tokens``, ``ground_truth``.
    """
    records = list(records)
    total = len(records)
    if total == 0:
        return {"count": 0}

    correct = sum(1 for r in records if r.get("acc"))
    parsed = sum(1 for r in records if r.get("answer_parsed"))
    invalid = sum(r.get('evaluation_valid') is False or r.get('acc') is None for r in records)
    scores = [float(r.get("score", 1.0 if r.get("acc") else -1.0)) for r in records if r.get('acc') is not None]
    no_answer = sum(r.get('parse_status') == 'no_answer' or r.get('pred') == '[INVALID]' for r in records)
    parse_failed = sum(r.get('parse_status') in ('malformed','ambiguous') for r in records)
    errors = sum(r.get('grade_status') in ('error','timeout') for r in records)
    lengths = [int(r["num_tokens"]) for r in records if r.get("num_tokens") is not None]
    truncated = sum(1 for r in records if r.get("truncated") or r.get("finish_reason") == "length")

    report = {
        "count": total,
        'evaluation_complete':invalid == 0,
        'scorable_count':total-invalid, 'unscorable_count':invalid,
        'grader_error_rate':errors/total, 'no_answer_rate':no_answer/total,
        "accuracy": correct / total,
        "correct": correct,
        "incorrect": total - invalid - correct,
        # Kept for comparison only - NOT an accuracy.
        "score_mean_pm1": statistics.fmean(scores) if scores else None,
        "answer_parsed": parsed,
        "answer_parse_failure_rate": parse_failed / total,
        "truncated": truncated,
        "truncation_rate": truncated / total,
    }
    if invalid:
        report['provisional_metrics'] = {'accuracy_lower_bound':report.pop('accuracy')}
        report['scorable_score_mean_pm1'] = report.pop('score_mean_pm1')
    if lengths:
        report["response_tokens"] = _quantiles(lengths)
        if answer_token_limit:
            over = sum(1 for value in lengths if value >= answer_token_limit)
            report["response_tokens"]["at_or_over_limit"] = over
            report["response_tokens"]["at_or_over_limit_rate"] = over / len(lengths)
            report["response_tokens"]["limit"] = answer_token_limit
    return report


def group_by_benchmark(records: Iterable[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("benchmark") or record.get("data_source") or "unknown"), []).append(record)
    return {name: aggregate(items) for name, items in sorted(grouped.items())}


def pass_at_1(per_question: Iterable[list[bool]]) -> float:
    """Mean per-question success rate - the pass@1 estimate used for reporting.

    Sampling ``n`` answers per question and averaging the per-question success
    rate estimates pass@1. Do **not** report this as pass@n.
    """
    rates = [sum(1 for flag in flags if flag) / len(flags) for flags in per_question if flags]
    return statistics.fmean(rates) if rates else 0.0


def pass_at_n(per_question: Iterable[list[bool]]) -> float:
    """Fraction of questions solved at least once (pass@n; reported separately)."""
    flags = [any(group) for group in per_question if group]
    return sum(1 for flag in flags if flag) / len(flags) if flags else 0.0


def summarize_sampling(per_question: dict[str, list[dict]], *, answer_token_limit: int | None = None) -> dict:
    """Per-benchmark report for multi-sample evaluation.

    ``per_question`` maps a benchmark name to a flat list of scored records that
    still carry their ``uid``; grouping happens here.
    """
    result: dict[str, Any] = {"benchmarks": {}, "overall": {}}
    all_groups: list[list[bool]] = []
    all_records: list[dict] = []
    for benchmark, records in sorted(per_question.items()):
        by_uid: dict[str, list[dict]] = {}
        for record in records:
            by_uid.setdefault(str(record.get("uid")), []).append(record)
        groups = [[bool(r["acc"]) for r in items] for items in by_uid.values()]
        all_groups.extend(groups)
        all_records.extend(records)
        entry = aggregate(records, answer_token_limit=answer_token_limit)
        entry["questions"] = len(by_uid)
        entry["samples_per_question"] = (len(records) / len(by_uid)) if by_uid else 0
        entry["pass_at_1"] = pass_at_1(groups)
        entry["pass_at_n_any_success"] = pass_at_n(groups)
        if not entry.get('evaluation_complete',False):
            entry.setdefault('provisional_metrics',{}).update({k:entry.pop(k) for k in ('pass_at_1','pass_at_n_any_success')})
        result["benchmarks"][benchmark] = entry
    if all_records:
        overall = aggregate(all_records, answer_token_limit=answer_token_limit)
        overall["questions"] = len(all_groups)
        overall["pass_at_1"] = pass_at_1(all_groups)
        overall["pass_at_n_any_success"] = pass_at_n(all_groups)
        if not overall.get('evaluation_complete',False):
            overall.setdefault('provisional_metrics',{}).update({k:overall.pop(k) for k in ('pass_at_1','pass_at_n_any_success')})
        overall["aggregation"] = "question-weighted mean over benchmarks present in the run"
        result["overall"] = overall
    return result


# --------------------------------------------------------------------------- #
# parity fixtures / CLI
# --------------------------------------------------------------------------- #
PARITY_CASES = [
    # (response, ground_truth, expected_acc, note)
    ("work\nAnswer: 42", "42", True, "plain Answer line"),
    ("work\nanswer: 42", "42", True, "the Answer pattern is case-insensitive"),
    ("work\nAnswer: $42$", "42", True, "$ wrappers are stripped"),
    ("work\nAnswer: 4,200", "4200", True, "thousands separators inside plain numbers are stripped"),
    ("work\nAnswer: 42", "42.0", False, "no numeric equivalence between 42 and 42.0"),
    ("work\nAnswer: 142.0", "142", False, "decimal form of an integer answer is NOT accepted by the CARD scorer"),
    ("work\nAnswer: 0.5", "\\frac{1}{2}", False, "no symbolic/numeric equivalence"),
    ("work \\boxed{42}", "42", False, "boxed-only answers are not parsed without an Answer line"),
    ("work\nAnswer: 5", "42", False, "wrong answer"),
    ("x" * 400 + "\nAnswer: 42", "42", True, "only the last 300 characters are inspected"),
    ("work\nAnswer: 42" + "y" * 400, "42", False, "an answer buried far from the end is not found"),
]


def self_test(*, verbose: bool = True) -> tuple[bool, list[str]]:
    module = scorer()
    problems: list[str] = []
    cases = list(PARITY_CASES)
    if getattr(module,'VERSION',None) == 'math_unified_v1':
        notes={4:'unified numeric equivalence',5:'unified integral decimal equivalence',
               6:'unified exact fraction equivalence',7:'unified boxed extraction',9:'unified full-response marker scan'}
        cases = [(response,gt,True if index in (4,5,6,7) else expected,notes.get(index,note))
                 for index,(response,gt,expected,note) in enumerate(cases)]
        cases += [('- Answer: **42**','42',True,'Markdown list/value'),
                  ('Answer: 1 2','12',False,'digit groups must not concatenate')]
    for response, ground_truth, expected, note in cases:
        result = score_response(response, ground_truth, scorer_module=module)
        if bool(result["acc"]) != expected:
            problems.append(
                f"parity failure ({note}): response={response[-30:]!r} gt={ground_truth!r} "
                f"expected acc={expected} got acc={result['acc']} pred={result['pred']!r}"
            )
        elif verbose:
            print(f"  ok: {note} -> pred={result['pred']!r} acc={result['acc']}")
    # acc/score separation
    results = [score_response("Answer: 1", "1"), score_response("Answer: 2", "1")]
    accuracy = sum(1 for r in results if r["acc"]) / len(results)
    score_mean = sum(r["score"] for r in results) / len(results)
    if not (abs(accuracy - 0.5) < 1e-9 and abs(score_mean - 0.0) < 1e-9):
        problems.append(f"acc/score separation broken: accuracy={accuracy} score_mean={score_mean}")
    elif verbose:
        print(f"  ok: accuracy={accuracy} score_mean_pm1={score_mean} (they must not be conflated)")
    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="JSONL with scored/per-question records.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--scorer-path", default=None)
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--ground-truth-field", default="ground_truth")
    parser.add_argument("--answer-token-limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true", help="Run fixtures for the loaded scorer version and exit.")
    args = parser.parse_args()

    global SCORER
    SCORER = load_scorer(args.scorer_path)
    print(f"[math_metrics] scorer: {getattr(SCORER, '__gad_source__', 'unknown')}", file=sys.stderr)

    if args.self_test:
        print(f"[math_metrics] {getattr(SCORER, 'VERSION', 'legacy_card_v1')} self-test")
        ok, problems = self_test()
        for problem in problems:
            print(f"  FAIL: {problem}")
        print("[math_metrics] self-test", "PASSED" if ok else "FAILED")
        return 0 if ok else 1

    if not args.input:
        parser.error("--input is required unless --self-test is used")

    records = []
    with open(args.input, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            scored = score_response(
                str(raw.get(args.response_field, "")), str(raw.get(args.ground_truth_field, ""))
            )
            records.append({**raw, **scored})

    report = summarize_sampling(
        _benchmark_groups(records), answer_token_limit=args.answer_token_limit
    )
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _benchmark_groups(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("benchmark") or record.get("data_source") or "unknown"), []).append(record)
    return grouped


if __name__ == "__main__":
    raise SystemExit(main())
