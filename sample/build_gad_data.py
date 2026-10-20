#!/usr/bin/env python3
"""Build the GAD train/val parquet: prompts + the offline teacher response.

Contract produced by this script (read by all three branches as
``data.prompt_key=content``)::

    content           list[{role, content}]   chat prompt, unchanged from input
    teacher_response  str                     teacher text for this prompt
    data_source       str
    ability           str
    reward_model      {ground_truth, style}   used by scoring/statistics only
    extra_info        {index}                 stable uid

Selection policy (plan section 3.4 / 5.3):

* default ``--select all`` (alias ``first``) takes the first valid teacher answer --
  fully deterministic, no correctness filtering, so all three stages see identical
  teacher text. It matches the source pool's ``selection=all`` and is the ablation
  pool of the original pipeline;
* ``--select correct`` keeps only prompts whose teacher answer the CARD-aligned
  scorer marks correct and drops the rest. It matches the source pool's
  ``selection=correct-nontruncated`` (the original's default), and it also removes
  truncated answers, which have no final-answer line and therefore score 0. It is
  opt-in: ``run_sample.sh`` selects it through ``gad_data.select``;
* ground truth is stored in ``reward_model`` only. It is never concatenated into
  the prompt or used as a teacher substitute.

The script fails when prompts are missing from the index or have an empty
teacher answer, unless ``--allow-missing`` is given explicitly; in that case the
dropped uids are written to ``dropped.json`` and counted in ``stats.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_index import TeacherIndex, TeacherIndexError  # noqa: E402
from task_template import extract_final_answer, normalize_raw_prompt, sha256_file  # noqa: E402
from math_scorer import compute_score  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gad.extensions.teacher_provenance import bind_selected
from scoring.math import compute_score

GAD_OUTPUT_SCHEMA = pa.schema(
    [
        ("content", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("teacher_response", pa.string()),
        ("data_source", pa.string()),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.string()), ('selected_teacher_index', pa.int64()),
            ('selected_teacher_text_sha256', pa.string()), ('source_finish_reason', pa.string()),
            ('source_truncated', pa.bool_()), ('teacher_metadata_version', pa.string())])),
    ]
)


def log(message: str) -> None:
    print(f"[build_gad_data] {message}", flush=True)


SELECT_CHOICES = ("all", "first", "correct", "nontruncated", "correct-nontruncated")
# Reasons that mean "the teacher never produced an answer for this prompt" (a data
# problem) as opposed to "this answer was filtered on purpose" (a selection decision).
MISSING_REASONS = ("not_in_teacher_index", "no_teacher_answer")


def pick_teacher_candidate(answers: list[str], *, select: str, ground_truth: str,
                        truncated: list[bool] | None = None) -> tuple[str | None, str]:
    """Return (answer, reason). ``answer is None`` means the prompt must be dropped.

    ``all``/``first`` mirror the source pool's ``selection=all``: every prompt that has a
    teacher answer is kept, with no correctness filter (the ablation setting).
    ``correct`` mirrors ``selection=correct-nontruncated`` in the original pipeline: the
    prompt is dropped unless the teacher answer matches the ground truth. In practice a
    truncated answer has no final-answer line and therefore scores 0, so ``correct``
    usually excludes truncated answers as a side effect - but only usually.
    ``nontruncated`` / ``correct-nontruncated`` make that explicit by reading the
    sampler's per-answer ``truncated`` flags instead of inferring them from the score.
    """
    if not answers:
        return None, "no_teacher_answer", None
    flags = list(truncated or [])
    flags += [False] * (len(answers) - len(flags))

    def first_ok(require_correct: bool, require_complete: bool) -> str | None:
        for i, (answer, was_truncated) in enumerate(zip(answers, flags)):
            if require_complete and was_truncated:
                continue
            if require_correct:
                graded = compute_score(answer, ground_truth)
                if graded.get('grade_status') in ('error', 'timeout'):
                    raise ValueError('teacher correctness filtering has unresolved grader error: ' + str(graded.get('error_reason')))
                if float(graded['score']) <= .5:continue
            return (answer, i)
        return None

    if select in ("all", "first"):
        return answers[0], select, 0
    if select == "nontruncated":
        answer = first_ok(require_correct=False, require_complete=True)
        return (answer[0], 'first_nontruncated', answer[1]) if answer is not None else (None, 'all_answers_truncated', None)
    answer = first_ok(require_correct=True, require_complete=select == "correct-nontruncated")
    if answer is not None:
        return answer[0], "first_correct_nontruncated" if select == "correct-nontruncated" else "first_correct", answer[1]
    if select == "correct-nontruncated":
        # Distinguish "wrong" from "right but cut off" so the report is actionable.
        if first_ok(require_correct=True, require_complete=False) is not None:
            return None, "correct_but_truncated", None
        return None, "no_correct_answer", None
    return None, "no_correct_answer", None


def pick_teacher_answer(*args, **kwargs):
    answer, reason, _ = pick_teacher_candidate(*args, **kwargs)
    return answer, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-parquet", required=True, help="Prompt parquet (train or dev).")
    parser.add_argument("--index", required=True, help="Teacher index JSONL.")
    parser.add_argument("--out", required=True, help="Output GAD parquet.")
    parser.add_argument("--sidecar", default=None, help="Per-prompt selection report JSON (default: <out>.selection.json)")
    parser.add_argument("--select", choices=list(SELECT_CHOICES), default="all",
                        help="Teacher-pool filter. all/first: keep every prompt with a teacher "
                             "answer (source pool selection=all, the default). correct: keep only "
                             "prompts whose teacher answer matches the ground truth. nontruncated: "
                             "drop answers the sampler marked truncated. correct-nontruncated: both, "
                             "i.e. exactly the source pool selection=correct-nontruncated.")
    parser.add_argument("--prompt-key", default=None, help="Input prompt column (default: auto-detect).")
    parser.add_argument("--expect-fingerprint-digest", default=None,
                        help="Fail unless the index was produced with this sampling fingerprint.")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Drop prompts without a usable teacher answer instead of failing.")
    parser.add_argument("--max-teacher-tokens", type=int, default=0,
                        help="With --student-tokenizer: count answers at/over this limit (0 = skip).")
    parser.add_argument("--student-tokenizer", default=None, help="Optional tokenizer for length statistics.")
    parser.add_argument("--tokenizer-limit", type=int, default=0, help="Only measure the first N rows (0 = all).")
    parser.add_argument("--stats-out", default=None)
    args = parser.parse_args()

    prompt_path = Path(args.prompt_parquet).expanduser().resolve()
    if not prompt_path.exists():
        raise SystemExit(f"[build_gad_data] ERROR: prompt parquet not found: {prompt_path}")

    table = pq.read_table(prompt_path)
    columns = set(table.column_names)
    prompt_key = args.prompt_key or ("prompt" if "prompt" in columns else ("content" if "content" in columns else None))
    if prompt_key is None:
        raise SystemExit(f"[build_gad_data] ERROR: no prompt column in {prompt_path} (columns: {sorted(columns)})")

    try:
        index = TeacherIndex.load(args.index, strict=True)
    except TeacherIndexError as exc:
        raise SystemExit(f"[build_gad_data] ERROR: {exc}") from exc
    log(f"teacher index: {len(index)} records from {args.index}")

    digest = None
    if index.header is not None:
        digest = index.header.get("teacher_fingerprint_digest")
    if args.expect_fingerprint_digest and digest != args.expect_fingerprint_digest:
        raise SystemExit(
            f"[build_gad_data] ERROR: teacher index fingerprint digest {digest!r} does not match the expected "
            f"{args.expect_fingerprint_digest!r}. Re-sample or point at the right index."
        )

    rows = table.select([prompt_key, "extra_info", "reward_model"] + (["data_source"] if "data_source" in columns else []) + (["ability"] if "ability" in columns else [])).to_pylist()

    out_rows: list[dict] = []
    dropped: list[dict] = []
    selection_report: list[dict] = []
    reasons: dict[str, int] = {}
    selected_answers: list[str] = []
    truncated_selected = 0

    for row in rows:
        uid = str(row["extra_info"]["index"])
        prompt = normalize_raw_prompt(row[prompt_key])
        ground_truth = str(row["reward_model"]["ground_truth"])
        record = index.get(uid=uid, raw_prompt=prompt)
        if record is None:
            dropped.append({"uid": uid, "reason": "not_in_teacher_index"})
            reasons["not_in_teacher_index"] = reasons.get("not_in_teacher_index", 0) + 1
            continue

        truncation_flags = list((record.sampling or {}).get("truncated") or [])
        answer, reason, selected_index = pick_teacher_candidate(
            record.teacher_answers,
            select=args.select,
            ground_truth=ground_truth,
            truncated=truncation_flags,
        )
        reasons[reason] = reasons.get(reason, 0) + 1
        if answer is None:
            dropped.append({"uid": uid, "reason": reason, "num_answers": len(record.teacher_answers)})
            continue
        provenance = bind_selected(record.teacher_answers, record.sampling or {}, answer, selected_index)
        if provenance['source_truncated']:
            truncated_selected += 1

        selection_report.append(
            {
                "uid": uid,
                "selected_index": selected_index,
                "num_answers": len(record.teacher_answers),
                "reason": reason,
                "teacher_answer_chars": len(answer),
            }
        )
        selected_answers.append(answer)
        out_rows.append(
            {
                "content": [{"role": m["role"], "content": m["content"]} for m in prompt],
                "teacher_response": answer,
                "data_source": row.get("data_source", "math_dapo"),
                "ability": row.get("ability", "MATH"),
                "reward_model": {
                    "ground_truth": ground_truth,
                    "style": row["reward_model"].get("style", "rule-lighteval/MATH_v2"),
                },
                "extra_info": {**row['extra_info'], "index": uid, **provenance},
            }
        )

    missing = [row for row in dropped if row.get("reason") in MISSING_REASONS]
    if missing and not args.allow_missing:
        # A deliberate filter (wrong answer / truncated / no correct answer) is a
        # selection decision and must not look like missing data. Only rows the
        # teacher never answered, or prompts absent from the index, are "missing".
        raise SystemExit(
            f"[build_gad_data] ERROR: {len(missing)} of {len(rows)} prompts have no usable teacher "
            f"answer at all (reasons: {sorted({row['reason'] for row in missing})}). Re-run sampling, "
            "or pass --allow-missing to drop them explicitly. Selection drops "
            f"({len(dropped) - len(missing)}) are reported, not errors."
        )

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows, schema=GAD_OUTPUT_SCHEMA), out_path)
    log(f"wrote {len(out_rows)} rows -> {out_path} (dropped {len(dropped)}: {reasons})")

    sidecar_path = Path(args.sidecar).expanduser() if args.sidecar else Path(str(out_path) + ".selection.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "select": args.select,
                "teacher_fingerprint_digest": digest,
                "num_rows": len(out_rows),
                "num_dropped": len(dropped),
                "num_missing": len(missing),
                "reasons": reasons,
                "teacher_truncated_selected": truncated_selected,
                "teacher_truncated_rate": (
                    truncated_selected / len(out_rows) if out_rows else 0.0
                ),
                "selections": selection_report,
                "dropped": dropped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    stats: dict = {
        "prompt_parquet": str(prompt_path),
        "prompt_parquet_sha256": sha256_file(str(prompt_path)),
        "index": str(Path(args.index).expanduser().resolve()),
        "select": args.select,
        "teacher_fingerprint_digest": digest,
        "rows_in": len(rows),
        "rows_out": len(out_rows),
        "rows_dropped": len(dropped),
        "drop_reasons": reasons,
        "out_parquet": str(out_path),
        "out_parquet_sha256": sha256_file(str(out_path)),
        "unique_uids": len({row["extra_info"]["index"] for row in out_rows}),
        "teacher_chars": {
            "mean": (sum(len(a) for a in selected_answers) / len(selected_answers)) if selected_answers else 0,
            "max": max((len(a) for a in selected_answers), default=0),
        },
    }
    if args.student_tokenizer and selected_answers:
        stats["teacher_token_length"] = token_length_stats(
            selected_answers, args.student_tokenizer, args.tokenizer_limit, args.max_teacher_tokens
        )
    stats_path = Path(args.stats_out).expanduser() if args.stats_out else Path(str(out_path) + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"stats -> {stats_path}")
    return 0


def token_length_stats(texts: list[str], tokenizer_path: str, limit: int, max_tokens: int) -> dict:
    """Teacher length distribution plus the truncation diagnostics the plan asks for.

    ``truncated_before_final_answer`` counts answers that would be cut off by the
    response budget *and* whose final ``Answer:`` line is therefore missing - a
    direct measure of "the final answer was truncated", not just "the text is long".
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths = []
    over = 0
    truncated_before_answer = 0
    no_answer_line = 0
    for text in (texts[:limit] if limit else texts):
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n_tokens)
        has_answer = extract_final_answer(text) is not None
        if not has_answer:
            no_answer_line += 1
        if max_tokens and n_tokens >= max_tokens:
            over += 1
            if not has_answer:
                truncated_before_answer += 1
    lengths.sort()

    def quantile(q: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, max(0, int(round(q * (len(lengths) - 1)))))]

    return {
        "count": len(lengths),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": lengths[-1] if lengths else 0,
        "max_tokens": max_tokens,
        "at_or_over_limit": over,
        "at_or_over_limit_rate": (over / len(lengths)) if lengths else 0.0,
        "without_final_answer_line": no_answer_line,
        "without_final_answer_line_rate": (no_answer_line / len(lengths)) if lengths else 0.0,
        "truncated_before_final_answer": truncated_before_answer,
        "truncated_before_final_answer_rate": (truncated_before_answer / len(lengths)) if lengths else 0.0,
        "tokenizer_path": tokenizer_path,
    }


if __name__ == "__main__":
    raise SystemExit(main())
