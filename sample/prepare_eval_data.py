#!/usr/bin/env python3
"""Build the external math evaluation set (AIME24/25, AMC23/24, MATH500).

Every path is parameterised: nothing is hard-coded to a particular cluster.
Sources are described by a JSON config (see ``configs/eval_sources.example.json``)
so a new benchmark is a config change, not a code change.

Guarantees:

* a benchmark whose data cannot be found is **listed explicitly** in the report;
  with ``--strict`` the run fails instead of quietly producing a smaller suite
  that would later be reported as "all five benchmarks";
* prompt duplicates between the eval sets and the training/dev prompts are
  detected and reported (``--train-parquet``/``--val-parquet``), so leakage can
  be excluded from the final numbers;
* answers are normalised only for cosmetic cases known from CARD
  (``142.0`` -> ``142``); the scoring itself is done by the CARD-aligned scorer.

Output parquet mirrors the verl prompt schema so the same loader can read it::

    data_source  "eval_<bench>"
    prompt       [{role, content}] using the shared math instruction
    ability      "MATH"
    reward_model {ground_truth, style}
    extra_info   {index: "<bench>-<id>"}
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_template import build_content_messages, canonicalize_raw_prompt, hash_raw_prompt, sha256_file  # noqa: E402

DEFAULT_SOURCES = [
    {"name": "aime24", "kind": "parquet_dir", "path": "", "split": None},
    {"name": "aime25", "kind": "load_from_disk", "path": "", "split": "train"},
    {"name": "amc23", "kind": "parquet_dir", "path": "", "split": None},
    {"name": "amc24", "kind": "jsonl", "path": "", "split": None},
    {"name": "math500", "kind": "load_from_disk", "path": "", "split": "test"},
]

PROBLEM_KEYS = ("problem", "question", "Problem", "Question", "prompt", "problem_statement")
ANSWER_KEYS = ("answer", "Answer", "final_answer", "gt", "label", "solution_answer")
ID_KEYS = ("id", "ID", "problem_id", "index", "unique_id")

OUTPUT_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.string())])),
    ]
)


def log(message: str) -> None:
    print(f"[prepare_eval_data] {message}", flush=True)


def pick(row: dict, keys, what: str, bench: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise KeyError(f"{bench}: cannot find {what} in row keys {sorted(row)}")


def load_rows(src: dict) -> list[dict]:
    kind = src["kind"]
    path = src.get("path") or ""
    if kind == "parquet_dir":
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet under {path}")
        return pq.read_table(files[0]).to_pylist()
    if kind == "parquet_file":
        return pq.read_table(path).to_pylist()
    if kind == "jsonl":
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if kind == "load_from_disk":
        from datasets import load_from_disk

        dataset = load_from_disk(path)
        split = src.get("split")
        if split and hasattr(dataset, "keys") and split in dataset:
            dataset = dataset[split]
        return [dict(row) for row in dataset]
    if kind == "hf":
        from datasets import load_dataset

        dataset = load_dataset(path, src.get("config"), split=src.get("split") or "test")
        return [dict(row) for row in dataset]
    raise ValueError(f"unknown source kind: {kind}")


def normalize_answer(answer: str) -> str:
    text = str(answer).strip()
    if text.endswith(".0") and text[:-2].replace("-", "").isdigit():
        return text[:-2]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="Directory for test.parquet and the manifest.")
    parser.add_argument("--sources", default=None, help="JSON file with the benchmark source list.")
    parser.add_argument("--benchmarks", default=None,
                        help="Comma-separated subset to build (default: every configured benchmark).")
    parser.add_argument("--train-parquet", default=None, help="Training prompts, for leakage detection.")
    parser.add_argument("--val-parquet", default=None, help="Dev prompts, for leakage detection.")
    parser.add_argument("--strict", action="store_true", help="Fail when a configured benchmark is unavailable.")
    parser.add_argument("--limit", type=int, default=0, help="Only keep the first N rows per benchmark.")
    args = parser.parse_args()

    if args.sources:
        sources = json.loads(Path(args.sources).expanduser().read_text(encoding="utf-8"))
        if isinstance(sources, dict):
            sources = sources.get("sources", [])
    else:
        sources = DEFAULT_SOURCES

    wanted = {b.strip() for b in args.benchmarks.split(",")} if args.benchmarks else None

    train_hashes: dict[str, str] = {}
    for label, path in (("train", args.train_parquet), ("val", args.val_parquet)):
        if not path:
            continue
        table = pq.read_table(Path(path).expanduser())
        key = "prompt" if "prompt" in table.column_names else "content"
        for row in table.select([key, "extra_info"]).to_pylist():
            train_hashes[hash_raw_prompt(row[key])] = f"{label}:{row['extra_info']['index']}"

    rows_out: list[dict] = []
    counts: dict[str, int] = {}
    skipped: list[dict] = []
    duplicates_within: dict[str, list[str]] = {}
    leakage: list[dict] = []
    seen_prompts: dict[str, str] = {}

    for src in sources:
        name = src["name"]
        if wanted and name not in wanted:
            continue
        try:
            raw_rows = load_rows(src)
        except Exception as exc:  # noqa: BLE001 - unavailable benchmark must be reported, not hidden
            skipped.append({"benchmark": name, "reason": f"{type(exc).__name__}: {exc}", "source": src})
            log(f"[skip] {name}: {type(exc).__name__}: {exc}")
            continue

        kept = 0
        for position, row in enumerate(raw_rows):
            if args.limit and kept >= args.limit:
                break
            try:
                problem = pick(row, PROBLEM_KEYS, "problem statement", name)
                answer = normalize_answer(pick(row, ANSWER_KEYS, "answer", name))
            except KeyError as exc:
                if position == 0:
                    skipped.append({"benchmark": name, "reason": str(exc), "source": src})
                    log(f"[skip] {name}: {exc}")
                break

            try:
                row_id = pick(row, ID_KEYS, "id", name)
            except KeyError:
                row_id = str(position)

            messages = build_content_messages(problem)
            prompt_hash = hash_raw_prompt(messages)
            uid = f"{name}-{row_id}"

            if prompt_hash in seen_prompts:
                duplicates_within.setdefault(prompt_hash, [seen_prompts[prompt_hash]]).append(uid)
            seen_prompts[prompt_hash] = uid
            if prompt_hash in train_hashes:
                leakage.append({"uid": uid, "benchmark": name, "matches": train_hashes[prompt_hash]})

            rows_out.append(
                {
                    "data_source": f"eval_{name}",
                    "prompt": messages,
                    "ability": "MATH",
                    "reward_model": {"ground_truth": answer, "style": "rule-lighteval/MATH_v2"},
                    "extra_info": {"index": uid},
                }
            )
            kept += 1
        counts[name] = kept
        log(f"[ok]   {name}: {kept} rows")

    unavailable = [item["benchmark"] for item in skipped]
    if args.strict and unavailable:
        raise SystemExit(
            f"[prepare_eval_data] ERROR: configured benchmarks unavailable: {unavailable}. "
            "Fix the paths or drop --strict and report only the benchmarks that were actually built."
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.parquet"
    pq.write_table(pa.Table.from_pylist(rows_out, schema=OUTPUT_SCHEMA), out_path)

    manifest = {
        "stage": "prepare_eval_data",
        "benchmarks_requested": sorted(wanted) if wanted else [s["name"] for s in sources],
        "benchmarks_built": {name: count for name, count in counts.items() if count},
        "benchmarks_unavailable": unavailable,
        "skipped": skipped,
        "rows": len(rows_out),
        "duplicate_prompts_within_eval": {
            prompt_hash: uids for prompt_hash, uids in duplicates_within.items()
        },
        "leakage_against_training_data": leakage,
        "sources": sources,
        "out_parquet": str(out_path),
        "out_sha256": sha256_file(str(out_path)) if rows_out else None,
    }
    (out_dir / "eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {len(rows_out)} rows -> {out_path}")
    if unavailable:
        log(f"WARNING: {len(unavailable)} benchmark(s) unavailable: {unavailable}")
    if leakage:
        log(f"WARNING: {len(leakage)} eval prompts also appear in train/dev; exclude them from final numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
