#!/usr/bin/env python3
"""CARD-aligned math metrics for dev monitoring and external evaluation.

The scoring rule itself is **not** re-implemented here: it is the vendored copy
of CARD's ``math_dapo.compute_score`` that lives in ``LMOps/sample/math_scorer.py``
(searched in this order: ``$GAD_MATH_SCORER``, ``--scorer-path``, the sibling
``LMOps/sample`` directory). This module only adds aggregation, diagnostics and
a parity self-test, so the numbers stay comparable with CARD.

What the underlying scorer does (verified by ``--self-test``):

* it looks at the **last 300 characters** of the completion and takes the last
  ``Answer: ...`` match, case-insensitively;
* ``\\boxed{...}`` alone, without an ``Answer:`` line, is **not** parsed;
* normalization strips ``$``, spaces, commas inside plain numbers and some LaTeX
  wrappers, but it does **not** do numeric equivalence: ``142.0`` does not match
  ``142`` and ``0.5`` does not match ``\\frac{1}{2}``;
* ``score`` is ±1 while ``acc`` is 0/1 - an accuracy is the mean of ``acc`` (or
  ``score > 0``), never the mean of ``score``.

Nothing in this module ever writes back into training: these numbers are
diagnostics only and never become a reward, an advantage or a discriminator
label.
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
    GAD_DIR.parent / "sample" / "math_scorer.py",
    GAD_DIR / "sample" / "math_scorer.py",
)


class MathScorerNotFound(RuntimeError):
    pass


def load_scorer(scorer_path: str | Path | None = None):
    """Import the vendored CARD scorer as a standalone module."""
    candidates = ([Path(scorer_path)] if scorer_path else []) + [p for p in DEFAULT_SCORER_PATHS if p]
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
    if SCORER is None:
        SCORER = load_scorer()
    return SCORER


def score_response(response: str, ground_truth: str, *, scorer_module=None) -> dict:
    module = scorer_module or scorer()
    result = module.compute_score(response or "", ground_truth or "")
    prediction = result["pred"]
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
    scores = [float(r.get("score", 1.0 if r.get("acc") else -1.0)) for r in records]
    lengths = [int(r["num_tokens"]) for r in records if r.get("num_tokens") is not None]
    truncated = sum(1 for r in records if r.get("truncated") or r.get("finish_reason") == "length")

    report = {
        "count": total,
        "accuracy": correct / total,
        "correct": correct,
        "incorrect": total - correct,
        # Kept for comparison only - NOT an accuracy.
        "score_mean_pm1": statistics.fmean(scores),
        "answer_parsed": parsed,
        "answer_parse_failure_rate": (total - parsed) / total,
        "truncated": truncated,
        "truncation_rate": truncated / total,
    }
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
        result["benchmarks"][benchmark] = entry
    if all_records:
        overall = aggregate(all_records, answer_token_limit=answer_token_limit)
        overall["questions"] = len(all_groups)
        overall["pass_at_1"] = pass_at_1(all_groups)
        overall["pass_at_n_any_success"] = pass_at_n(all_groups)
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
    for response, ground_truth, expected, note in PARITY_CASES:
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
    parser.add_argument("--self-test", action="store_true", help="Run the CARD parity fixtures and exit.")
    args = parser.parse_args()

    global SCORER
    SCORER = load_scorer(args.scorer_path)
    print(f"[math_metrics] scorer: {getattr(SCORER, '__gad_source__', 'unknown')}", file=sys.stderr)

    if args.self_test:
        print("[math_metrics] CARD parity self-test")
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
