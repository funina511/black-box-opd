"""Human review is explicit; automatic checks never satisfy the human gate."""
from collections import Counter
from copy import deepcopy
from rubric.common import *
from rubric.schema import validate_annotation, validate_label


def audit(cfg):
    from rubric.data import load_pool
    run, dataset = claim_run(cfg)
    pool = load_pool(dataset)
    raw = read_jsonl(dataset / "annotations.raw.jsonl")
    if len(raw) != len(pool) or len({r["response_id"] for r in raw}) != len(raw):
        raise ValueError("complete unique annotations required before audit")
    spec_hash = file_hash(cfg["rubric"]["spec"])
    for row in raw:
        validate_annotation(row, pool[row["response_id"]], spec_hash)
    directory = dataset / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    # Round robin strata: role, correctness, length and truncation.
    buckets = {}
    for row in raw:
        t = pool[row["response_id"]]
        length = t["original_response_tokens"]
        stratum = (t["source_role"], row["labels"][DIMENSIONS[0]]["value"],
                   "long" if length >= 4096 else "medium" if length >= 1024 else "short", t["truncated"])
        buckets.setdefault(str(stratum), []).append(row)
    chosen = []
    ordered = sorted(buckets)
    while any(buckets.values()):
        for key in ordered:
            if buckets[key]:
                chosen.append(buckets[key].pop(0))
    needed = min(cfg["audit"]["min_reviews"], len(pool))
    template = directory / "review_requests.jsonl"
    if not template.exists():
        write_jsonl(template, [{"response_id": row["response_id"], "cache_key": row["cache_key"],
                               "trajectory": pool[row["response_id"]], "annotation": row,
                               "reviewer": None, "accepted": None, "severe_reverse": None,
                               "reason": None, "corrections": {}}
                              for row in chosen[:needed]])
    review_path = cfg["audit"]["reviews_path"] or directory / "reviews.jsonl"
    reviews = read_jsonl(review_path)
    by_id = {r["response_id"]: r for r in raw}
    reviewed = deepcopy(by_id)
    seen, accepted, severe = set(), 0, 0
    for review in reviews:
        rid = review["response_id"]
        if rid not in by_id or rid in seen or review["cache_key"] != by_id[rid]["cache_key"]:
            raise ValueError("review ID/cache mismatch or duplicate")
        if not review.get("reviewer") or type(review.get("accepted")) is not bool or type(review.get("severe_reverse")) is not bool:
            raise ValueError("each imported review requires reviewer and explicit boolean decisions")
        if not isinstance(review.get("reason"), str) or not review["reason"].strip():
            raise ValueError("human review needs a reason")
        seen.add(rid)
        accepted += int(review["accepted"])
        severe += int(review["severe_reverse"])
        target = reviewed[rid]
        target["review_status"] = "human_reviewed"
        target["human_review"] = review
        target["original_labels"] = deepcopy(target["labels"])
        for dim, correction in review.get("corrections", {}).items():
            if dim not in DIMENSIONS:
                raise ValueError("unknown correction dimension")
            validate_label(correction, dim)
            target["labels"][dim] = {**correction, "source": "human_correction"}
        if not review["accepted"] and not review.get("corrections"):
            # Rejecting a process label without a replacement makes it unknown.
            for dim in DIMENSIONS[1:]:
                target["labels"][dim] = {"value": None, "valid": False, "source": "human_rejected"}
    write_jsonl(dataset / "annotations.reviewed.jsonl", reviewed.values())
    attempted = [row for row in raw if not pool[row["response_id"]]["truncated"]]
    valid_rate = sum(bool(r.get("judge_json_valid", True)) for r in attempted) / len(attempted) if attempted else 0
    repeat_rows = read_jsonl(dataset / "judge_repeats.jsonl")
    repeat_stats = {}
    for dim in DIMENSIONS[1:]:
        differences = [abs(by_id[r["response_id"]]["labels"][dim]["value"] - r["labels"][dim]["value"])
                       for r in repeat_rows if r["response_id"] in by_id
                       and by_id[r["response_id"]]["cache_key"] == r["cache_key"]
                       and by_id[r["response_id"]]["labels"][dim]["valid"] and r["labels"][dim]["valid"]]
        repeat_stats[dim] = {"both_valid": len(differences), "requested": len(repeat_rows),
                             "exact": sum(d == 0 for d in differences),
                             "one_bin": sum(d == .5 for d in differences),
                             "two_bins": sum(d == 1 for d in differences),
                             "coverage": len(differences) / len(repeat_rows) if repeat_rows else None,
                             "exact_rate": sum(d == 0 for d in differences) / len(differences) if differences else None,
                             "one_bin_rate": sum(d == .5 for d in differences) / len(differences) if differences else None,
                             "two_bins_rate": sum(d == 1 for d in differences) / len(differences) if differences else None}
    acceptance = accepted / len(seen) if seen else None
    reverse = severe / len(seen) if seen else None
    requested_ids = {row["response_id"] for row in read_jsonl(template)}
    gate = (len(seen) >= needed and requested_ids.issubset(seen) and valid_rate >= cfg["audit"]["min_json_valid_rate"]
            and acceptance >= cfg["audit"]["min_human_acceptance"]
            and reverse <= cfg["audit"]["max_severe_reverse_rate"])
    report = {"quality_gate_passed": bool(gate), "status": "passed" if gate else "pending_or_failed",
              "label_quality_human_verified": bool(gate), "required_reviews": needed,
              "human_review_count": len(seen), "human_acceptance": acceptance,
              "severe_reverse_rate": reverse, "judge_json_valid_rate": valid_rate,
              "repeat_consistency": repeat_stats,
              "dimensions": {dim: {"valid": sum(r["labels"][dim]["valid"] for r in raw),
                                   "unknown": sum(not r["labels"][dim]["valid"] for r in raw),
                                   "distribution": dict(Counter(str(r["labels"][dim]["value"]) for r in raw))}
                             for dim in DIMENSIONS},
              "review_template": str(template), "review_ids": sorted(seen),
              "annotation_hash": file_hash(dataset / "annotations.raw.jsonl")}
    report["answer_parse_status"] = dict(Counter(r.get("parse_status","imported") for r in raw))
    report["raw_correctness_by_role"] = {
        role: {"responses":sum(t["source_role"] == role for t in pool.values()),
               "correct":sum(int(bool(r["raw_scorer_result"]["acc"])) for r in raw
                             if pool[r["response_id"]]["source_role"] == role)}
        for role in ("teacher","student")}
    outputs = read_jsonl(dataset / "judge_outputs.jsonl")
    actual_outputs = [r for r in outputs if r.get("raw_output") is not None]
    report["judge_output_attempts"] = len(actual_outputs)
    report["judge_json_valid_rate_per_attempt"] = (
        sum(not r.get("error") for r in actual_outputs) / len(actual_outputs) if actual_outputs else None)
    report["truncated_responses"] = sum(r["truncated"] for r in pool.values())
    report["quarantined_responses"] = sum(not r["input_eligible"] for r in pool.values())
    write_json(directory / "summary.json", report)
    return report
