"""Rule correctness and externally grounded, anonymous process annotations."""
from __future__ import annotations
import json
import os
import time
import urllib.request
from rubric.common import *
from rubric.schema import *


def correctness(row):
    from math_scorer import compute_score
    raw = compute_score(row["response"], row["ground_truth"])
    pred = raw.get("pred")
    if pred in (None, "", "INVALID", "[INVALID]"):
        # Keep the scorer's original result; mask uncertain extraction for auxiliary supervision.
        status = "extraction_failed" if ("boxed" in row["response"] or "Answer:" in row["response"]) else "no_final_answer"
    else:
        status = "parsed_match" if raw["acc"] else "parsed_mismatch"
    valid = status != "extraction_failed"
    return label(int(bool(raw["acc"])) if valid else None, valid, "math_scorer"), {
        "raw_scorer_result": raw, "pred": pred, "parse_status": status,
        "scorer_version": file_hash(ROOT / "sample/math_scorer.py")}


def api_call(cfg, messages):
    options = cfg["judge"]
    if not options["api_base_url"] or not options["api_model"] or not options["api_version"]:
        raise ValueError("API requires explicit URL, model and immutable api_version")
    key = os.environ.get(options["api_key_env"])
    if not key:
        raise ValueError("configured API key environment variable is unset")
    data = json.dumps({"model": options["api_model"], "messages": messages,
                       "temperature": options["temperature"], "max_tokens": options["max_tokens"],
                       "seed": options["seed"]}).encode()
    request = urllib.request.Request(options["api_base_url"].rstrip("/") + "/chat/completions",
                                    data=data, headers={"Content-Type": "application/json",
                                                        "Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    choice = payload["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("judge output truncated")
    return choice["message"]["content"]


def annotate(cfg):
    from rubric.data import load_pool
    run, dataset = claim_run(cfg)
    pool = load_pool(dataset)
    spec_hash = file_hash(cfg["rubric"]["spec"])
    prompt_hash = file_hash(cfg["judge"]["prompt"])
    options = cfg["judge"]
    if options["backend"] == "import":
        if not options["annotations_path"]:
            raise ValueError("judge.annotations_path required for import")
        rows = read_jsonl(options["annotations_path"])
        if len(rows) != len(pool) or len({r["response_id"] for r in rows}) != len(rows):
            raise ValueError("import must label every unique response exactly once")
        for row in rows:
            validate_annotation(row, pool[row["response_id"]], spec_hash)
            expected, scorer = correctness(pool[row["response_id"]])
            if row["labels"][DIMENSIONS[0]] != expected:
                raise ValueError("imported rule correctness differs; use audited corrections")
            row.update(scorer)
            row.setdefault("review_status","pending")
            row.setdefault("judge_json_valid",True)
        write_jsonl(dataset / "annotations.raw.jsonl", rows)
        return {"imported": len(rows)}
    if options["backend"] == "vllm":
        judge_fp = {"model": model_fingerprint(options["model"], run / "judge_fingerprint.json"),
                    "generation": {key: options[key] for key in
                                   ("temperature", "seed", "max_tokens", "max_model_len", "tensor_parallel_size")}}
    elif options["backend"] == "api":
        from transformers import AutoTokenizer
        if not options["model"]:
            raise ValueError("API mode needs an explicit matching local tokenizer at judge.model for context validation")
        api_tokenizer = AutoTokenizer.from_pretrained(options["model"], local_files_only=True)
        judge_fp = {key: options[key] for key in ("api_base_url", "api_model", "api_version",
                                                  "temperature", "seed", "max_tokens")}
        judge_fp["context_tokenizer"] = {p.name: file_hash(p) for p in Path(options["model"]).iterdir()
                                         if p.name.startswith("tokenizer") or p.suffix == ".jinja"}
    else:
        raise ValueError("judge.backend must be import, vllm or api")
    existing = {row["cache_key"]: row for row in read_jsonl(dataset / "annotation.cache.jsonl")}
    instruction = Path(options["prompt"]).read_text()
    spec = Path(cfg["rubric"]["spec"]).read_text()
    engine = None
    api_tokenizer = locals().get("api_tokenizer")
    results, repeats = [], []
    repeat_ids = set(sorted(pool, key=lambda rid: digest({"rid": rid, "seed": cfg["run"]["seed"]}))
                     [:max(1, round(len(pool) * options["repeat_fraction"]))])
    repeat_cache = {r["cache_key"]: r for r in read_jsonl(dataset / "judge_repeats.jsonl")}
    jobs = []
    for rid, trajectory in pool.items():
        if trajectory["truncated"]:
            continue
        key = annotation_key(trajectory, spec_hash, judge_fp, prompt_hash)
        if key not in existing:
            jobs.append((key, "primary", options["seed"], judge_messages(trajectory, spec, instruction)))
        if rid in repeat_ids and key not in repeat_cache:
            jobs.append((key, "repeat", options["seed"] + 1, judge_messages(trajectory, spec, instruction)))
    prefetched, job_offset = {}, 0
    for rid, trajectory in pool.items():
        key = annotation_key(trajectory, spec_hash, judge_fp, prompt_hash)
        if key in existing:
            row = existing[key]
            validate_annotation(row, trajectory, spec_hash)
        else:
            answer_label, scorer = correctness(trajectory)
            row = {"schema_version": ANNOTATION_SCHEMA, "response_id": rid, "rubric_version": "math_v1",
                   "rubric_hash": spec_hash, "cache_key": key, "judge_fingerprint": judge_fp,
                   "judge_prompt_hash": prompt_hash, "review_status": "pending",
                   "labels": {DIMENSIONS[0]: answer_label}, "evidence": {}, **scorer}
        requests = []
        if key not in existing:
            requests.append(("primary", options["seed"]))
        if rid in repeat_ids and key not in repeat_cache and not trajectory["truncated"]:
            requests.append(("repeat", options["seed"] + 1))
        for kind, seed in requests:
            parsed, failures = None, []
            if trajectory["truncated"]:
                failures.append("truncated candidate: full-process judgment masked")
            else:
                messages = judge_messages(trajectory, spec, instruction)
                if options["backend"] == "vllm" and engine is None:
                    from rubric.inference import Generator
                    engine = Generator(options["model"], tp=options["tensor_parallel_size"],
                                       context=options["max_model_len"],
                                       utilization=options["gpu_memory_utilization"],
                                       seed=options["seed"], thinking=False)
                for attempt in range(options["max_retries"] + 1):
                    started, raw = time.monotonic(), None
                    try:
                        if engine:
                            if attempt == 0:
                                if (key, kind) not in prefetched:
                                    batch = jobs[job_offset:job_offset + options["batch_size"]]
                                    job_offset += len(batch)
                                    valid_jobs = []
                                    for job in batch:
                                        rendered = engine.render(job[3])
                                        token_count = len(engine.tokenizer.encode(rendered, add_special_tokens=False))
                                        if token_count + options["max_tokens"] > options["max_model_len"]:
                                            prefetched[(job[0],job[1])] = {"error": "full judge input exceeds context"}
                                        else:
                                            valid_jobs.append(job)
                                    if valid_jobs:
                                        outputs = engine.generate([job[3] for job in valid_jobs], n=1,
                                                                  temperature=options["temperature"], top_p=1,
                                                                  max_tokens=options["max_tokens"],
                                                                  seed=[job[2] for job in valid_jobs])
                                        for job, output in zip(valid_jobs, outputs):
                                            prefetched[(job[0],job[1])] = {
                                                **output["outputs"][0], "error": None,
                                                "seconds_batch": output["seconds_batch"], "batch_size": len(valid_jobs)}
                                generated = prefetched[(key,kind)]
                                if generated["error"]:
                                    raise ValueError(generated["error"])
                            else:
                                generated = engine.generate([messages], n=1, temperature=options["temperature"],
                                                            top_p=1, max_tokens=options["max_tokens"], seed=seed)[0]["outputs"][0]
                            raw = generated["text"]
                            if generated["truncated"]:
                                raise ValueError("judge output truncated")
                        else:
                            input_ids = api_tokenizer.apply_chat_template(messages, tokenize=True,
                                                                          add_generation_prompt=True, enable_thinking=False)
                            if len(input_ids) + options["max_tokens"] > options["max_model_len"]:
                                raise ValueError("API full prompt exceeds configured context; not sent")
                            raw = api_call(cfg, messages)
                        parsed = parse_judge(raw)
                        error = None
                    except (ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        failures.append(error)
                    append_jsonl(dataset / "judge_outputs.jsonl",
                                 {"response_id": rid, "cache_key": key, "kind": kind, "attempt": attempt,
                                  "raw_output": raw, "error": error, "seconds": time.monotonic() - started,
                                  "batch_size": prefetched.get((key,kind),{}).get("batch_size",1) if attempt == 0 else 1,
                                  "batch_generation_seconds": prefetched.get((key,kind),{}).get("seconds_batch") if attempt == 0 else None})
                    if parsed is not None:
                        break
            process = {dimension: (label(parsed[dimension]["value"], parsed[dimension]["valid"], "judge")
                                    if parsed else label(source="judge_failure"))
                       for dimension in DIMENSIONS[1:]}
            evidence = {dimension: parsed[dimension]["evidence"] if parsed else "; ".join(failures)
                        for dimension in DIMENSIONS[1:]}
            if kind == "repeat":
                repeated = {"response_id": rid, "cache_key": key, "labels": process,
                            "evidence": evidence, "json_valid": parsed is not None}
                append_jsonl(dataset / "judge_repeats.jsonl", repeated)
                repeat_cache[key] = repeated
            else:
                row.update(labels={DIMENSIONS[0]: row["labels"][DIMENSIONS[0]], **process},
                           evidence=evidence, judge_json_valid=parsed is not None,
                           judge_attempt_errors=failures)
                validate_annotation(row, trajectory, spec_hash)
                append_jsonl(dataset / "annotation.cache.jsonl", row)
                existing[key] = row
        results.append(row)
        print(f"[rubric] annotated {len(results)}/{len(pool)}", flush=True)
    if engine is not None:
        engine.close()
    write_jsonl(dataset / "annotations.raw.jsonl", results)
    return {"annotations": len(results), "repeat_reviews": len(repeat_cache)}
