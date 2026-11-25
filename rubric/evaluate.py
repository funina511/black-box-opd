"""Fixed-pool quality metrics, grouped bootstrap and one-time test evaluation."""
from __future__ import annotations
import itertools
import math
import random
from rubric.common import *


def mean(values):
    return sum(values) / len(values) if values else None


def auroc(scores, targets):
    positives = [s for s, t in zip(scores, targets) if t == 1]
    negatives = [s for s, t in zip(scores, targets) if t == 0]
    if not positives or not negatives:
        return None
    # Average ranks with exact tie handling, O(n log n).
    ordered = sorted(zip(scores, targets))
    positive_rank_sum, offset = 0.0, 0
    while offset < len(ordered):
        end = offset + 1
        while end < len(ordered) and ordered[end][0] == ordered[offset][0]:
            end += 1
        rank = (offset + 1 + end) / 2
        positive_rank_sum += rank * sum(t == 1 for _, t in ordered[offset:end])
        offset = end
    return (positive_rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))


def direction(a, b):
    return float(a > b) if a != b else .5


def metrics(predictions, bootstrap=0, seed=42):
    pairs, quality, conflict, first, random_choice, oracle, selected = [], [], [], [], [], [], []
    all_rows, process_ranks = [], {dim: [] for dim in DIMENSIONS[1:]}
    for group in predictions:
        records = group["records"]
        teacher = records[group["teacher_id"]]
        for rid in group["student_ids"]:
            student = records[rid]
            gap = teacher["adv"] - student["adv"]
            loss = max(-gap, 0) + math.log1p(math.exp(-abs(gap)))
            pairs.append((direction(teacher["adv"], student["adv"]), loss))
            if teacher["acc"] == 0 and student["acc"] == 1:
                conflict.append(direction(student["adv"], teacher["adv"]))
        for a, b in itertools.combinations(records.values(), 2):
            if a["acc"] != b["acc"]:
                quality.append(direction(a["adv"], b["adv"]) if a["acc"] > b["acc"]
                               else direction(b["adv"], a["adv"]))
            else:
                for dim in DIMENSIONS[1:]:
                    left, right = a["labels"][dim], b["labels"][dim]
                    if left["valid"] and right["valid"] and left["value"] != right["value"]:
                        process_ranks[dim].append(direction(a["adv"], b["adv"]) if left["value"] > right["value"]
                                                  else direction(b["adv"], a["adv"]))
        candidates = [records[rid] for rid in group["student_ids"]]
        first.append(candidates[0]["acc"])
        random_choice.append(mean([c["acc"] for c in candidates]))
        oracle.append(max(c["acc"] for c in candidates))
        # Stable first candidate on ties.
        selected.append(max(candidates, key=lambda c: c["adv"])["acc"])
        all_rows.extend(records.values())
    result = {"prompts": len(predictions), "unique_responses": len(all_rows), "pairs": len(pairs),
              "d_acc": mean([p[0] for p in pairs]), "adv_loss": mean([p[1] for p in pairs]),
              "quality_ordering": mean(quality), "quality_comparisons": len(quality),
              "adv_correctness_auroc": auroc([r["adv"] for r in all_rows], [r["acc"] for r in all_rows]),
              "teacher_wrong_student_right": {"comparisons": len(conflict), "quality_ordering": mean(conflict)},
              "best_of_2": mean(selected), "first": mean(first), "random_expected": mean(random_choice),
              "oracle": mean(oracle), "auxiliary": {},
              "same_correctness_process_ordering": {dim: {"comparisons": len(v), "adv_ordering": mean(v)}
                                                    for dim, v in process_ranks.items()}}
    for dim in DIMENSIONS:
        enabled = [r for r in all_rows if dim in r.get("aux", {}) and r["labels"][dim]["valid"]]
        if enabled:
            values = [r["aux"][dim] for r in enabled]
            targets = [r["labels"][dim]["value"] for r in enabled]
            result["auxiliary"][dim] = {
                "valid_responses": len(enabled), "mae": mean([abs(p-t) for p,t in zip(values, targets)]),
                "brier": mean([(p-t)**2 for p,t in zip(values, targets)]),
                "ordinal_agreement": mean([float(min((0,.5,1), key=lambda x: abs(x-p)) == t)
                                          for p,t in zip(values, targets)]),
                "auroc": auroc(values, targets) if dim == DIMENSIONS[0] else None}
    if bootstrap and predictions:
        rng = random.Random(seed)
        keys = ("d_acc", "adv_loss", "quality_ordering", "best_of_2", "first", "random_expected", "oracle")
        sampled = {key: [] for key in keys}
        for _ in range(bootstrap):
            sample = metrics([predictions[rng.randrange(len(predictions))] for _ in predictions])
            for key in keys:
                if sample[key] is not None:
                    sampled[key].append(sample[key])
        def percentile(values, proportion):
            if not values:
                return None
            values = sorted(values)
            at = (len(values)-1)*proportion
            lo, hi = math.floor(at), math.ceil(at)
            return values[lo] + (values[hi]-values[lo])*(at-lo)
        result["ci95_by_prompt_bootstrap"] = {
            key: [percentile(v,.025), percentile(v,.975)] for key,v in sampled.items()}
    return result


def paired_deltas(candidate, reference, bootstrap=1000, seed=42):
    left = {row["uid"]:row for row in candidate}
    right = {row["uid"]:row for row in reference}
    if set(left) != set(right):
        raise ValueError("paired comparison requires identical fixed question sets")
    uids = sorted(left)
    keys = ("d_acc","quality_ordering","best_of_2")
    a,b = metrics(candidate),metrics(reference)
    draws = {key:[] for key in keys}
    rng = random.Random(seed)
    for _ in range(bootstrap):
        sampled = [uids[rng.randrange(len(uids))] for _ in uids]
        x,y = metrics([left[uid] for uid in sampled]),metrics([right[uid] for uid in sampled])
        for key in keys:
            if x[key] is not None and y[key] is not None:
                draws[key].append(x[key]-y[key])
    result = {}
    for key in keys:
        values = sorted(draws[key])
        result[key] = {"delta": a[key]-b[key] if a[key] is not None and b[key] is not None else None,
                       "ci95": [values[int((len(values)-1)*.025)],values[int((len(values)-1)*.975)]] if values else None,
                       "valid_bootstrap_draws":len(values),"prompts":len(uids)}
    return result


def predict_groups(model, tokenizer, groups, generation, device, variant):
    import torch
    from rubric.model import encode_response, score_response
    predictions = []
    model.eval()
    with torch.inference_mode():
        for group in groups:
            records = {}
            for rid, record in group["records"].items():
                row, annotation = record["trajectory"], record["annotation"]
                logits = score_response(model, encode_response(tokenizer, row, generation), device)
                if not torch.isfinite(logits).all():
                    raise ValueError("non-finite critic score")
                auxiliary = {}
                if variant in ("correctness_aux", "joint"):
                    dims = DIMENSIONS[:1] if variant == "correctness_aux" else DIMENSIONS
                    auxiliary = {dim: float(logits[1 + DIMENSIONS.index(dim)].sigmoid()) for dim in dims}
                records[rid] = {"adv": float(logits[0]), "aux": auxiliary,
                                "acc": int(bool(annotation["raw_scorer_result"]["acc"])),
                                "labels": annotation["labels"], "source_role": row["source_role"],
                                "tokens": row["original_response_tokens"], "truncated": row["truncated"],
                                "review_status": annotation["review_status"]}
            predictions.append({key: group[key] for key in ("uid","split","teacher_id","student_ids")} | {"records": records})
    return predictions


def evaluate(cfg, variant, split="test"):
    import torch
    from transformers import AutoTokenizer
    from rubric.model import export_single_head, load_classifier
    from rubric.data import load_groups
    run, dataset = claim_run(cfg)
    manifest = read_json(dataset / "manifest.json")
    directory = run / "variants" / variant
    if variant == "baseline":
        path = export_single_head(cfg["source"]["critic_checkpoint"], run / "exports/critic")
        model = load_classifier(path, torch.float32)
    else:
        selection = read_json(directory / "selection.json")
        if split == "test" and not selection["training_complete"]:
            raise ValueError("test requires completed dev selection")
        path = Path(selection["checkpoint"])
        model = load_classifier(path, torch.float32, file_hash(cfg["rubric"]["spec"]))
    signature = digest({"model": model_fingerprint(path), "split": manifest["packed"]["parquet_hashes"][split],
                        "variant": variant, "evaluation_code": file_hash(__file__),
                        "bootstrap": cfg["training"]["bootstrap_samples"]})
    report_path = directory / f"{split}.metrics.json"
    if report_path.exists():
        cached = read_json(report_path)
        if cached["signature"] != signature:
            raise ValueError("evaluation is locked to selected model/data/code; start a new run for changed selection")
        return cached
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    predictions = predict_groups(model, tokenizer, load_groups(dataset, split),
                                 manifest["source"]["generation"], "cuda", variant)
    summary = metrics(predictions, cfg["training"]["bootstrap_samples"], cfg["run"]["seed"])
    # Response-stratified auxiliary diagnostics; pair metrics remain grouped.
    strata = {}
    for criterion, values in (("source_role", ("teacher","student")), ("truncated", (False,True)),
                              ("acc", (0,1)), ("length", ("short","medium","long"))):
        for value in values:
            selected = []
            for group in predictions:
                rows = {rid:r for rid,r in group["records"].items()
                        if (("long" if r["tokens"] >= 4096 else "medium" if r["tokens"] >= 1024 else "short")
                            if criterion == "length" else r[criterion]) == value}
                selected.extend(rows.values())
            strata[f"{criterion}={value}"] = {
                "responses": len(selected), "mean_adv": mean([r["adv"] for r in selected]),
                "accuracy": mean([r["acc"] for r in selected]),
                "auxiliary_mae": {dim: mean([abs(r["aux"][dim]-r["labels"][dim]["value"]) for r in selected
                                            if dim in r["aux"] and r["labels"][dim]["valid"]]) for dim in DIMENSIONS}}
    humans = [r for g in predictions for r in g["records"].values() if r["review_status"] == "human_reviewed"]
    report = {"signature": signature, "variant": variant, "split": split, "selected_checkpoint": str(path),
              "metrics": summary, "strata": strata, "human_reviewed_responses": len(humans),
              "human_reviewed_auxiliary_mae": {dim: mean([abs(r["aux"][dim]-r["labels"][dim]["value"])
                                                        for r in humans if dim in r["aux"] and r["labels"][dim]["valid"]])
                                              for dim in DIMENSIONS},
              "exploratory": manifest["packed"]["exploratory"],
              "precision": "FP32 master parameters, BF16 autocast; identical to dev selection",
              "interpretation": "fixed-pool critic evaluation; no actor was trained; 24 questions cannot establish efficacy"}
    write_jsonl(directory / f"{split}.predictions.jsonl", predictions)
    write_json(report_path, report)
    return report
