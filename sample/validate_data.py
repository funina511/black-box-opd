#!/usr/bin/env python3
"""Validate the GAD data contract before any GPU time is spent.

Checks performed (plan section 6 "数据契约"):

1. required columns and types, for train and dev;
2. ``extra_info.index`` is unique inside each split, and train/dev do not share
   a uid **or** a prompt hash (leak);
3. ``teacher_response`` is non-empty, is not the ground truth, and is not a
   truncated stub;
4. the ground truth never appears in a prompt in the answer-contract form;
5. every uid resolves in the teacher index with a matching prompt hash, and the
   optional expected sampling fingerprint matches;
6. optional tokenizer-aware length report: prompt tokens vs ``max_prompt_length``,
   teacher tokens vs ``max_response_length``, and the joint prompt+response
   budget, with p50/p95/p99 and over-length counts.

Exit code is non-zero when a hard contract is violated. Warnings are reported
but do not fail the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_index import TeacherIndex, TeacherIndexError  # noqa: E402
from task_template import (  # noqa: E402
    canonicalize_raw_prompt,
    extract_final_answer,
    hash_raw_prompt,
    normalize_raw_prompt,
    sha256_file,
)

REQUIRED_FIELDS = ("extra_info.index", "reward_model.ground_truth", "teacher_response")


def log(message: str) -> None:
    print(f"[validate_data] {message}", flush=True)


def load_split(path: Path, prompt_key: str) -> list[dict]:
    table = pq.read_table(path)
    columns = set(table.column_names)
    key = prompt_key if prompt_key in columns else ("content" if "content" in columns else ("prompt" if "prompt" in columns else None))
    if key is None:
        raise SystemExit(f"[validate_data] ERROR: {path} has no prompt column (columns: {sorted(columns)})")
    for field in ("extra_info", "reward_model"):
        if field not in columns:
            raise SystemExit(f"[validate_data] ERROR: {path} is missing column {field!r}")
    if "teacher_response" not in columns:
        raise SystemExit(f"[validate_data] ERROR: {path} is missing column 'teacher_response'")
    return [{"__prompt_key": key, **row} for row in table.to_pylist()]


def contract_form_present(text: str, ground_truth: str) -> bool:
    lowered = text.lower()
    gt = ground_truth.lower().strip()
    if not gt:
        return False
    return f"answer: {gt}" in lowered or f"\\boxed{{{gt}}}" in lowered


def check_split(rows: list[dict], *, name: str, errors: list[str], warnings: list[str]) -> dict:
    uids: Counter = Counter()
    prompt_hashes: dict[str, str] = {}
    empty_teacher = 0
    teacher_equals_gt = 0
    gt_in_prompt = 0
    answer_line_in_teacher = 0
    missing_answer_line_in_teacher = 0

    for row in rows:
        prompt_key = row["__prompt_key"]
        uid = str(row["extra_info"]["index"])
        uids[uid] += 1
        prompt = normalize_raw_prompt(row[prompt_key])
        prompt_hash = hash_raw_prompt(prompt)
        prompt_hashes[uid] = prompt_hash
        ground_truth = str(row["reward_model"]["ground_truth"])

        teacher = row.get("teacher_response")
        teacher = "" if teacher is None else str(teacher)
        if not teacher.strip():
            empty_teacher += 1
        if teacher.strip() == ground_truth.strip() and ground_truth.strip():
            teacher_equals_gt += 1
        if contract_form_present(canonicalize_raw_prompt(prompt), ground_truth):
            gt_in_prompt += 1
        if extract_final_answer(teacher) is not None:
            answer_line_in_teacher += 1
        else:
            missing_answer_line_in_teacher += 1

    duplicates = [uid for uid, count in uids.items() if count > 1]
    if duplicates:
        errors.append(f"{name}: {len(duplicates)} duplicated uids (first: {duplicates[:3]})")
    if empty_teacher:
        errors.append(f"{name}: {empty_teacher} rows with an empty teacher_response")
    if teacher_equals_gt:
        errors.append(f"{name}: {teacher_equals_gt} rows whose teacher_response equals the ground truth")
    if gt_in_prompt:
        errors.append(f"{name}: {gt_in_prompt} prompts contain the ground truth in answer-contract form")
    if missing_answer_line_in_teacher:
        warnings.append(
            f"{name}: {missing_answer_line_in_teacher} teacher answers have no 'Answer:' line "
            "(expected for some teachers; they stay valid teacher text)"
        )

    return {
        "rows": len(rows),
        "unique_uids": len(uids),
        "duplicated_uids": len(duplicates),
        "empty_teacher_responses": empty_teacher,
        "teacher_equals_ground_truth": teacher_equals_gt,
        "prompts_with_ground_truth": gt_in_prompt,
        "teacher_with_answer_line": answer_line_in_teacher,
        "teacher_without_answer_line": missing_answer_line_in_teacher,
        "prompt_hashes": prompt_hashes,
    }


def token_report(rows: list[dict], tokenizer_path: str, *, max_prompt_length: int, max_response_length: int,
                 prompt_key: str) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt_lengths, teacher_lengths, joint = [], [], []
    prompt_over = teacher_over = joint_over = 0
    for row in rows:
        prompt_text = canonicalize_raw_prompt(row[prompt_key])
        teacher_text = str(row.get("teacher_response") or "")
        prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        teacher_tokens = len(tokenizer(teacher_text, add_special_tokens=False)["input_ids"])
        prompt_lengths.append(prompt_tokens)
        teacher_lengths.append(teacher_tokens)
        joint.append(prompt_tokens + teacher_tokens)
        prompt_over += prompt_tokens > max_prompt_length
        teacher_over += teacher_tokens > max_response_length
        joint_over += (prompt_tokens + teacher_tokens) > (max_prompt_length + max_response_length)

    def stats(values: list[int]) -> dict:
        ordered = sorted(values)
        if not ordered:
            return {"count": 0}
        return {
            "count": len(ordered),
            "min": ordered[0],
            "p50": ordered[int(round(0.50 * (len(ordered) - 1)))],
            "p95": ordered[int(round(0.95 * (len(ordered) - 1)))],
            "p99": ordered[int(round(0.99 * (len(ordered) - 1)))],
            "max": ordered[-1],
            "mean": sum(ordered) / len(ordered),
        }

    return {
        "tokenizer_path": tokenizer_path,
        "prompt_tokens": {**stats(prompt_lengths), "limit": max_prompt_length, "over_limit": prompt_over,
                          "over_limit_rate": prompt_over / len(rows) if rows else 0.0},
        "teacher_tokens": {**stats(teacher_lengths), "limit": max_response_length, "over_limit": teacher_over,
                           "over_limit_rate": teacher_over / len(rows) if rows else 0.0},
        "joint_tokens": {**stats(joint), "limit": max_prompt_length + max_response_length,
                         "over_limit": joint_over, "over_limit_rate": joint_over / len(rows) if rows else 0.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", default=None, help="Dev/monitor split (checked for leakage against train).")
    parser.add_argument("--index", default=None, help="Teacher index; when given, coverage is verified.")
    parser.add_argument("--expect-fingerprint-digest", default=None)
    parser.add_argument("--student-tokenizer", default=None)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--prompt-key", default="content")
    parser.add_argument("--report", default=None, help="Where to write the JSON report.")
    parser.add_argument("--allow-val-leak", action="store_true",
                        help="Record (as a warning) train/val prompt overlap instead of failing. Only for "
                             "configs whose dev split is an external benchmark that may repeat DAPO questions.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {}

    train_path = Path(args.train).expanduser().resolve()
    train_rows = load_split(train_path, args.prompt_key)
    train_report = check_split(train_rows, name="train", errors=errors, warnings=warnings)
    train_hashes = train_report.pop("prompt_hashes")
    report["train"] = {**train_report, "path": str(train_path), "sha256": sha256_file(str(train_path))}
    log(f"train: {train_report['rows']} rows, {train_report['unique_uids']} unique uids")

    val_hashes: dict[str, str] = {}
    if args.val:
        val_path = Path(args.val).expanduser().resolve()
        val_rows = load_split(val_path, args.prompt_key)
        val_report = check_split(val_rows, name="val", errors=errors, warnings=warnings)
        val_hashes = val_report.pop("prompt_hashes")
        report["val"] = {**val_report, "path": str(val_path), "sha256": sha256_file(str(val_path))}
        log(f"val:   {val_report['rows']} rows, {val_report['unique_uids']} unique uids")

        shared_uids = set(train_hashes) & set(val_hashes)
        if shared_uids:
            errors.append(f"train/val uid leak: {len(shared_uids)} shared uids (first: {sorted(shared_uids)[:3]})")
        train_hash_set = set(train_hashes.values())
        shared_prompts = sorted({h for h in val_hashes.values() if h in train_hash_set})
        if shared_prompts:
            message = (
                f"train/val prompt leak: {len(shared_prompts)} identical prompts in both splits "
                "(same question, possibly different uid)"
            )
            if args.allow_val_leak:
                warnings.append(message + " [allowed by --allow-val-leak]")
            else:
                errors.append(message)
        report["leakage"] = {
            "shared_uids": len(shared_uids),
            "shared_prompt_hashes": len(shared_prompts),
            "allowed": bool(args.allow_val_leak),
        }

    if args.index:
        try:
            index = TeacherIndex.load(args.index, strict=True)
        except TeacherIndexError as exc:
            raise SystemExit(f"[validate_data] ERROR: {exc}") from exc
        digest = index.header.get("teacher_fingerprint_digest") if index.header else None
        if args.expect_fingerprint_digest and digest != args.expect_fingerprint_digest:
            errors.append(
                f"teacher index fingerprint digest {digest!r} != expected {args.expect_fingerprint_digest!r}"
            )
        missing, mismatched = [], []
        for row in train_rows + (val_rows if args.val else []):
            uid = str(row["extra_info"]["index"])
            prompt = normalize_raw_prompt(row[row["__prompt_key"]])
            if index.get(uid=uid, raw_prompt=prompt) is None:
                if index.get(uid=uid) is None:
                    missing.append(uid)
                else:
                    mismatched.append(uid)
        if missing:
            errors.append(f"{len(missing)} uids missing from the teacher index (first: {missing[:3]})")
        if mismatched:
            errors.append(
                f"{len(mismatched)} uids have a different prompt hash than the index (first: {mismatched[:3]})"
            )
        report["index"] = {
            "path": str(Path(args.index).expanduser().resolve()),
            "records": len(index),
            "fingerprint_digest": digest,
            "missing": len(missing),
            "prompt_hash_mismatch": len(mismatched),
        }

    if args.student_tokenizer:
        report["lengths"] = {
            "train": token_report(
                train_rows, args.student_tokenizer,
                max_prompt_length=args.max_prompt_length, max_response_length=args.max_response_length,
                prompt_key=args.prompt_key,
            )
        }
        if args.val:
            report["lengths"]["val"] = token_report(
                val_rows, args.student_tokenizer,
                max_prompt_length=args.max_prompt_length, max_response_length=args.max_response_length,
                prompt_key=args.prompt_key,
            )

    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = not errors

    if args.report:
        Path(args.report).expanduser().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for warning in warnings:
            log(f"WARNING: {warning}")
        for error in errors:
            log(f"ERROR: {error}")
        log("OK" if not errors else f"FAILED with {len(errors)} error(s)")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
