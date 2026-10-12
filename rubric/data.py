"""Grouped sampling, provenance, and locked offline training tables."""
from __future__ import annotations
import random
from pathlib import Path

from rubric.common import *
from rubric.schema import normalized_prompt, validate_trajectory, validate_annotation


def prepare(cfg):
    import pyarrow.parquet as pq
    from teacher_index import TeacherIndex
    from task_template import hash_raw_prompt
    run, dataset = claim_run(cfg)
    source = source_config(cfg)
    train_path = Path(source.require("data.train_file"))
    index_path = Path(source.require("data.teacher_index"))
    generation = generation_settings(cfg)
    actor, critic = map(Path, (cfg["source"]["actor_checkpoint"], cfg["source"]["critic_checkpoint"]))
    expected_run = Path(source.require("run.output_root")) / source.require("run.run_id")
    if actor.parent != critic.parent or actor.parent.parent != expected_run / "warmup":
        raise ValueError("actor/critic must be from the same explicit Warmup checkpoint of source run")
    from rubric.model import inspect_head
    head = inspect_head(critic)
    fingerprints = {
        "run_config": file_hash(cfg["source"]["run_config"]),
        "source_parquet": file_hash(train_path), "teacher_index": file_hash(index_path),
        "dataset_manifest": file_hash(cfg["source"]["dataset_manifest"]),
        "actor": model_fingerprint(actor), "critic": model_fingerprint(critic),
        "rubric": file_hash(cfg["rubric"]["spec"]), "generation": generation, "head": head,
    }
    manifest_path = dataset / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["source"] != fingerprints or manifest["config_hash"] != digest(cfg):
            raise ValueError("source/config changed; cached dataset cannot be reused")
        for name, expected in manifest["prepared_inputs"].items():
            if file_hash(dataset / name) != expected:
                raise ValueError("prepared prompt/split files changed")
        return run, dataset
    forbidden = set()
    exclusions = []
    files = {Path(source.require("data.val_file"))}
    if source.get("validation.quick_file"):
        files.add(Path(source.get("validation.quick_file")))
    files.update((ROOT.parent / "OPRD/datasets/test_data_dapo").glob("*/test.parquet"))
    for path in sorted(files):
        parquet = pq.ParquetFile(path)
        key = "content" if "content" in parquet.schema_arrow.names else "prompt"
        for prompt in parquet.read(columns=[key]).column(0).to_pylist():
            forbidden.add(normalized_prompt(prompt))
        exclusions.append({"path": str(path), "sha256": file_hash(path)})
    rows = pq.read_table(train_path).to_pylist()
    groups = {}
    seen_uids = {}
    for row in rows:
        uid = str(row["extra_info"]["index"])
        key = normalized_prompt(row[source.get("data.prompt_key")])
        if uid in seen_uids and seen_uids[uid] != key:
            raise ValueError("same uid refers to different questions")
        seen_uids[uid] = key
        if key not in forbidden:
            groups.setdefault(key, []).append(row)
    keys = sorted(groups)
    random.Random(cfg["run"]["seed"]).shuffle(keys)
    keys = keys[:cfg["sampling"]["prompt_count"]]
    if len(keys) < cfg["sampling"]["prompt_count"]:
        raise ValueError("not enough unique non-evaluation prompts")
    index = TeacherIndex.load(index_path)
    selected, splits = [], {part: [] for part in ("train", "dev", "test")}
    boundaries = [("train", cfg["split"]["train"]), ("dev", cfg["split"]["dev"]), ("test", cfg["split"]["test"])]
    offset = 0
    for split, count in boundaries:
        for key in keys[offset:offset + count]:
            row = groups[key][0]
            uid = str(row["extra_info"]["index"])
            prompt = row[source.get("data.prompt_key")]
            record = index.require(uid=uid, raw_prompt=prompt)
            text = row["teacher_response"]
            if text not in record.teacher_answers:
                raise ValueError(f"teacher parquet/index answer mismatch: {uid}")
            slot = record.teacher_answers.index(text)
            if record.teacher_fingerprint_digest != source.get("data.expected_fingerprint_digest"):
                raise ValueError("teacher fingerprint mismatch")
            metadata = record.sampling
            selected.append({"uid": uid, "prompt": prompt, "prompt_hash": hash_raw_prompt(prompt),
                             "normalized_prompt_hash": key, "split": split, "teacher_response": text,
                             "ground_truth": row["reward_model"]["ground_truth"],
                             "equivalent_uids": [str(r["extra_info"]["index"]) for r in groups[key]],
                             "teacher_finish_reason": (metadata.get("finish_reasons") or [None] * len(record.teacher_answers))[slot],
                             "teacher_truncated": (metadata.get("truncated") or [False] * len(record.teacher_answers))[slot]})
            splits[split].append(uid)
        offset += count
    dataset.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset / "prompts.jsonl", selected)
    write_json(dataset / "splits.json", {"seed": cfg["run"]["seed"], "unit": "normalized_prompt", **splits})
    write_json(manifest_path, {"schema_version": "rubric.dataset.v1", "config_hash": digest(cfg),
                              "source": fingerprints, "excluded_evaluation_files": exclusions,
                              "excluded_prompt_count": len(forbidden), "source_rows": len(rows),
                              "selected_prompts": len(selected), "seed": cfg["run"]["seed"],
                              "split_sizes": {key: len(value) for key, value in splits.items()},
                              "prepared_inputs": {name: file_hash(dataset / name) for name in
                                                  ("prompts.jsonl", "splits.json")},
                              "near_duplicate_policy": "whitespace-normalized message text; semantic duplicates not guaranteed",
                              "test_exposure": "unseen by auxiliary training; Warmup already trained on full DAPO train"})
    return run, dataset


def load_pool(dataset):
    trajectories = read_jsonl(Path(dataset) / "trajectories.jsonl")
    if not trajectories:
        raise ValueError("trajectory pool is empty or collection has not completed")
    manifest = read_json(Path(dataset) / "manifest.json")
    if file_hash(Path(dataset) / "trajectories.jsonl") != manifest["collection"]["trajectory_hash"]:
        raise ValueError("fixed trajectory pool changed")
    ids = {}
    split_by_hash = {}
    for row in trajectories:
        validate_trajectory(row)
        if row["response_id"] in ids:
            raise ValueError("duplicate unique response")
        ids[row["response_id"]] = row
        key = normalized_prompt(row["prompt"])
        if key in split_by_hash and split_by_hash[key] != row["split"]:
            raise ValueError("same question crosses splits")
        split_by_hash[key] = row["split"]
    return ids


def pack(cfg, exploratory=False):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    run, dataset = paths(cfg)
    audit = read_json(dataset / "audit/summary.json")
    if not audit["quality_gate_passed"] and not exploratory:
        raise ValueError("human label-quality gate is pending; use --exploratory for an explicitly limited run")
    rows = load_pool(dataset)
    annotations = {r["response_id"]: r for r in read_jsonl(dataset / "annotations.reviewed.jsonl")}
    if len(annotations) != len(read_jsonl(dataset / "annotations.reviewed.jsonl")):
        raise ValueError("duplicate annotation IDs")
    rubric_hash = file_hash(cfg["rubric"]["spec"])
    for rid, row in rows.items():
        if rid not in annotations:
            raise ValueError("missing annotation (unknown must be explicit)")
        validate_annotation(annotations[rid], row, rubric_hash)
    groups = read_jsonl(dataset / "groups.jsonl")
    manifest = read_json(dataset / "manifest.json")
    if file_hash(dataset / "groups.jsonl") != manifest["collection"]["group_hash"]:
        raise ValueError("fixed candidate groups changed")
    splits = read_json(dataset / "splits.json")
    expected_split = {uid: part for part in ("train","dev","test") for uid in splits[part]}
    if len(groups) != len(expected_split) or len({group["uid"] for group in groups}) != len(groups):
        raise ValueError("each prepared question must have exactly one complete group")
    for group in groups:
        if expected_split.get(group["uid"]) != group["split"]:
            raise ValueError("candidate group disagrees with locked question split")
        for rid in [group["teacher_id"], *group["student_ids"]]:
            if rows[rid]["uid"] != group["uid"] or rows[rid]["split"] != group["split"]:
                raise ValueError("cross-question/cross-split candidate or annotation leakage")
    inputs = {name: file_hash(dataset / name) for name in ("trajectories.jsonl", "groups.jsonl",
                                                         "annotations.reviewed.jsonl", "splits.json")}
    signature = digest(inputs)
    if manifest.get("packed"):
        if manifest["packed"]["data_hash"] != signature:
            raise ValueError("packed labels/splits are locked; create a new dataset version")
        return
    packed, quarantined = {part: [] for part in ("train", "dev", "test")}, []
    for group in groups:
        ids = list(dict.fromkeys([group["teacher_id"], *group["student_ids"]]))
        if any(not rows[rid].get("input_eligible") for rid in ids) or len(group["student_ids"]) != 2:
            quarantined.append({"uid": group["uid"], "reason": "incomplete or over-budget/empty input group"})
            continue
        payload = dict(group)
        payload["records_json"] = json.dumps({rid: {"trajectory": rows[rid], "annotation": annotations[rid]}
                                             for rid in ids}, ensure_ascii=False)
        packed[group["split"]].append(payload)
    for part, values in packed.items():
        schema = pa.schema([("uid", pa.string()), ("split", pa.string()), ("teacher_id", pa.string()),
                            ("student_ids", pa.list_(pa.string())), ("records_json", pa.string())])
        pq.write_table(pa.Table.from_pylist(values, schema=schema), dataset / f"{part}.parquet")
    pairs = [{"uid": group["uid"], "teacher_id": group["teacher_id"], "student_id": rid,
              "weight": 1.0, "split": group["split"]} for group in groups for rid in group["student_ids"]]
    write_jsonl(dataset / "pairs.jsonl", pairs)
    write_jsonl(dataset / "audit/quarantined_groups.jsonl", quarantined)
    manifest["packed"] = {"data_hash": signature, "inputs": inputs, "counts": {p: len(v) for p, v in packed.items()},
                          "exploratory": not audit["quality_gate_passed"],
                          "quarantined": len(quarantined),
                          "parquet_hashes": {p: file_hash(dataset / f"{p}.parquet") for p in packed}}
    write_json(dataset / "manifest.json", manifest)


def load_groups(dataset, split):
    import json
    import pyarrow.parquet as pq
    manifest = read_json(Path(dataset) / "manifest.json")
    path = Path(dataset) / f"{split}.parquet"
    if file_hash(path) != manifest["packed"]["parquet_hashes"][split]:
        raise ValueError("packed data changed")
    groups = pq.read_table(path).to_pylist()
    for group in groups:
        group["records"] = json.loads(group.pop("records_json"))
    return groups
