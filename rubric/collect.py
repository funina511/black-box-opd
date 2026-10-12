"""Freeze teacher and two student candidates, preserving source settings."""
from pathlib import Path
from rubric.common import *
from rubric.schema import TRAJECTORY_SCHEMA, response_id, validate_trajectory


def collect(cfg):
    from transformers import AutoTokenizer
    from rubric.data import prepare
    from rubric.model import encode_response
    from rubric.inference import Generator
    run, dataset = prepare(cfg)
    manifest = read_json(dataset / "manifest.json")
    generation = manifest["source"]["generation"]
    actor = Path(cfg["source"]["actor_checkpoint"]) / "huggingface"
    if not (actor / "model.safetensors").exists():
        raise ValueError("explicit actor HF export missing; export source actor first")
    tokenizer = AutoTokenizer.from_pretrained(actor, local_files_only=True)
    cache_path = dataset / "sampling.cache.jsonl"
    cache = {row["uid"]: row for row in read_jsonl(cache_path)}
    prompts = read_jsonl(dataset / "prompts.jsonl")
    signature = digest({"actor": manifest["source"]["actor"], "generation": generation,
                        "tokenizer": model_fingerprint(actor)["digest"]})
    for row in cache.values():
        if row["signature"] != signature:
            raise ValueError("sampling cache provenance changed")
    pending = [row for row in prompts if row["uid"] not in cache]
    if pending:
        engine = Generator(actor, tp=cfg["sampling"]["tensor_parallel_size"],
                           context=generation["max_prompt_length"] + generation["max_response_length"],
                           utilization=cfg["sampling"]["gpu_memory_utilization"],
                           seed=cfg["run"]["seed"],
                           thinking=generation["enable_thinking"] if generation["thinking_control"] else None)
        batch_size = cfg["sampling"]["batch_size"]
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            # Per-question seeds make restarts independent of batch boundaries.
            seeds = [(cfg["run"]["seed"] + int(p["prompt_hash"][:8], 16)) % (2 ** 31) for p in batch]
            generated = engine.generate([p["prompt"] for p in batch], n=2, temperature=generation["temperature"],
                                        top_p=generation["top_p"], max_tokens=generation["max_response_length"],
                                        seed=seeds, max_prompt_tokens=generation["max_prompt_length"])
            for prompt, sample_seed, result in zip(batch, seeds, generated):
                row = {"uid": prompt["uid"], "prompt_hash": prompt["prompt_hash"],
                       "signature": signature, "seed": sample_seed, **result}
                append_jsonl(cache_path, row)
                cache[prompt["uid"]] = row
                print(f"[rubric] collected {len(cache)}/{len(prompts)} prompts", flush=True)
        engine.close()
    unique, groups = {}, []
    tokenizer_hashes = {p.name: file_hash(p) for p in actor.iterdir()
                        if p.name.startswith("tokenizer") or p.name in ("chat_template.jinja", "special_tokens_map.json")}
    for prompt in prompts:
        sample = cache[prompt["uid"]]
        if sample["prompt_hash"] != prompt["prompt_hash"] or len(sample["outputs"]) != 2:
            raise ValueError("incomplete or mismatched sample")
        candidates = [("teacher", {"text": prompt["teacher_response"],
                                   "finish_reason": prompt["teacher_finish_reason"],
                                   "truncated": prompt["teacher_truncated"]})]
        candidates += [("student", output) for output in sample["outputs"]]
        ids = []
        for slot, (role, output) in enumerate(candidates):
            text = output["text"]
            rid = response_id(prompt["prompt"], text)
            occurrence = {"role": role, "slot": slot}
            ids.append(rid)
            if rid in unique:
                unique[rid]["occurrences"].append(occurrence)
                continue
            row = {"schema_version": TRAJECTORY_SCHEMA, "uid": prompt["uid"],
                   "split": prompt["split"], "prompt": prompt["prompt"],
                   "prompt_hash": prompt["prompt_hash"],
                   "normalized_prompt_hash": prompt["normalized_prompt_hash"],
                   "response": text, "response_hash": digest(text), "response_id": rid,
                   "ground_truth": prompt["ground_truth"], "source_role": role,
                   "occurrences": [occurrence], "finish_reason": output["finish_reason"],
                   "truncated": bool(output["truncated"]),
                   "source_fingerprint": manifest["source"]["actor"] if role == "student"
                   else {"teacher_index": manifest["source"]["teacher_index"],
                         "teacher_fingerprint_digest": source_config(cfg).get("data.expected_fingerprint_digest")},
                   "source_parquet_hash": manifest["source"]["source_parquet"],
                   "teacher_index_hash": manifest["source"]["teacher_index"],
                   "tokenizer_hashes": tokenizer_hashes, "sampling": generation,
                   "sampling_seed": sample["seed"] if role == "student" else None,
                   "generated_token_ids": output.get("token_ids"),
                   "original_response_tokens": len(output["token_ids"]) if "token_ids" in output
                   else len(tokenizer.encode(text, add_special_tokens=False)),
                   "retokenized_response_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                   "original_token_count_source": "vllm_token_ids" if "token_ids" in output else "actor_tokenizer_estimate",
                   "visible_response_tokens": None, "input_eligible": False,
                   "quarantine_reason": None}
            try:
                encoded = encode_response(tokenizer, row, generation)
                row["visible_response_tokens"] = encoded["visible_response_tokens"]
                row["input_eligible"] = True
                validate_trajectory(row)
            except ValueError as exc:
                row["quarantine_reason"] = str(exc)
            unique[rid] = row
        groups.append({"uid": prompt["uid"], "split": prompt["split"],
                       "teacher_id": ids[0], "student_ids": ids[1:]})
    # Empty responses remain in quarantine, not valid training trajectories.
    invalid = [row for row in unique.values() if not row["response"].strip()]
    if invalid:
        write_jsonl(dataset / "empty_responses.jsonl", invalid)
        raise ValueError("empty generated responses; inspect empty_responses.jsonl")
    write_jsonl(dataset / "trajectories.jsonl", unique.values())
    write_jsonl(dataset / "groups.jsonl", groups)
    manifest["collection"] = {"unique_responses": len(unique), "candidate_slots": len(groups) * 3,
                              "quarantined_responses": sum(not r["input_eligible"] for r in unique.values()),
                              "trajectory_hash": file_hash(dataset / "trajectories.jsonl"),
                              "group_hash": file_hash(dataset / "groups.jsonl"), "signature": signature}
    write_json(dataset / "manifest.json", manifest)
    return manifest["collection"]
