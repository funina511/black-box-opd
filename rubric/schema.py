"""Hashes, strict annotation validation, and anonymous judge inputs."""
from __future__ import annotations
import json
import re
from rubric.common import DIMENSIONS, digest
from task_template import hash_raw_prompt

TRAJECTORY_SCHEMA = "rubric.trajectory.v1"
ANNOTATION_SCHEMA = "rubric.annotation.v1"


def response_id(prompt, response):
    return digest({"prompt_hash": hash_raw_prompt(prompt), "response": response})


def normalized_prompt(prompt):
    return digest([{"role": p["role"], "content": re.sub(r"\s+", " ", p["content"]).strip()} for p in prompt])


def validate_trajectory(row):
    if row.get("schema_version") != TRAJECTORY_SCHEMA:
        raise ValueError("trajectory schema mismatch")
    if row["prompt_hash"] != hash_raw_prompt(row["prompt"]):
        raise ValueError("prompt hash mismatch")
    if row["response_hash"] != digest(row["response"]) or row["response_id"] != response_id(row["prompt"], row["response"]):
        raise ValueError("response hash/ID mismatch")
    if row["source_role"] not in ("teacher", "student") or not row["response"].strip():
        raise ValueError("invalid role or empty response")


def label(value=None, valid=False, source="unknown"):
    return {"value": value, "valid": valid, "source": source}


def validate_label(item, dimension):
    value, valid = item.get("value"), item.get("valid")
    if type(valid) is not bool:
        raise ValueError("label valid must be boolean")
    if not valid and value is not None:
        raise ValueError("unknown label must be null")
    allowed = (0, 1) if dimension == DIMENSIONS[0] else (0, 0.5, 1)
    if valid and (isinstance(value, bool) or value not in allowed):
        raise ValueError(f"invalid {dimension} label: {value}")


def annotation_key(row, rubric_hash, judge_fingerprint, prompt_hash):
    return digest({"prompt": row["prompt_hash"], "response": row["response_hash"],
                   "ground_truth": digest(row["ground_truth"]), "rubric": rubric_hash,
                   "judge": judge_fingerprint, "judge_prompt": prompt_hash})


def validate_annotation(row, trajectory, rubric_hash):
    if row.get("schema_version") != ANNOTATION_SCHEMA:
        raise ValueError("annotation schema mismatch")
    if row["response_id"] != trajectory["response_id"] or row["rubric_hash"] != rubric_hash:
        raise ValueError("annotation response/rubric mismatch")
    if row.get("cache_key") != annotation_key(trajectory, rubric_hash, row["judge_fingerprint"], row["judge_prompt_hash"]):
        raise ValueError("annotation cache hash mismatch")
    if row.get("rubric_version") != "math_v1" or set(row["labels"]) != set(DIMENSIONS):
        raise ValueError("rubric version/dimension mismatch")
    for dimension in DIMENSIONS:
        validate_label(row["labels"][dimension], dimension)
        if dimension != DIMENSIONS[0] and row["labels"][dimension]["valid"] and not row["evidence"].get(dimension):
            raise ValueError("valid process label needs evidence")


def judge_messages(row, spec, instruction):
    # No UID, role, model name, critic score or teacher reference solution.
    return [{"role": "system", "content": instruction + "\n固定 rubric:\n" + spec},
            {"role": "user", "content": "待评材料 JSON:\n" +
             json.dumps({"problem": row["prompt"], "response": row["response"],
                         "reference_answer": row["ground_truth"]}, ensure_ascii=False)}]


def parse_judge(text):
    cleaned = text.strip()
    fence = chr(96) * 3
    if cleaned.startswith(fence + "json\n") and cleaned.endswith("\n" + fence):
        cleaned = cleaned[8:-4]
    row = json.loads(cleaned)
    if set(row) != set(DIMENSIONS[1:]):
        raise ValueError("judge must return exactly two process dimensions")
    for dimension, item in row.items():
        validate_label(item, dimension)
        if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
            raise ValueError("judge needs evidence or an unknown reason")
    return row
