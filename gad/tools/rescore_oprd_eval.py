#!/usr/bin/env python3
"""Rescore OPRD raw_samples.jsonl with the CARD scorer used by GAD validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from math_metrics import load_scorer, score_response, summarize_sampling


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rescore(raw_path: Path, *, task: str, max_tokens: int, scorer_path: Path) -> dict:
    scorer = load_scorer(scorer_path)
    records = []
    seen = set()
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            index = int(raw["global_index"])
            sample_index = int(raw["rollout_index"])
            key = (index, sample_index)
            if key in seen:
                raise ValueError(f"duplicate prompt/rollout pair: {key}")
            seen.add(key)
            info = raw.get("extra_info") or {}
            uid = str(info.get("index", index))
            scored = score_response(str(raw.get("response") or ""),
                                    str(raw.get("ground_truth") or ""), scorer_module=scorer)
            records.append({"uid": uid, "benchmark": task,
                            "sample_index": sample_index,
                            "response": raw.get("response"),
                            "ground_truth": raw.get("ground_truth"),
                            "finish_reason": raw.get("finish_reason"),
                            "num_tokens": raw.get("output_tokens"),
                            **scored})
    if not records:
        raise ValueError(f"no samples in {raw_path}")
    grouped = defaultdict(list)
    for record in records:
        grouped[task].append(record)
    report = summarize_sampling(grouped, answer_token_limit=max_tokens)
    report["scoring_protocol"] = "CARD math_dapo scorer used by LMOps GAD validation"
    report["scorer_path"] = str(scorer_path.resolve())
    report["scorer_sha256"] = sha256(scorer_path)
    report["raw_samples"] = str(raw_path.resolve())
    report["raw_samples_sha256"] = sha256(raw_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-jsonl", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = rescore(args.raw_jsonl, task=args.task, max_tokens=args.max_tokens,
                     scorer_path=args.scorer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CARD pass@1: {result['benchmarks'][args.task]['pass_at_1']:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
