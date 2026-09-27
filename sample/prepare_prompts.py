#!/usr/bin/env python3
"""Prepare the DAPO-Math-17k prompt set for all three GAD stages.

What it does (see ``LMOps/GAD修改计划.md`` section 3.4 "数据源与去重"):

1. Reads a local DAPO parquet, or downloads
   ``BytedTsinghua-SIA/DAPO-Math-17k`` ``data/dapo-math-17k.parquet`` with a
   **pinned revision**, and records the file SHA-256.
2. De-duplicates by ``extra_info.index`` (the DAPO file repeats every unique
   problem ~100x). The first row per index wins, but *every* row of a group is
   checked: if the same index carries different prompt text or a different
   ground truth, the group is recorded in ``gt_conflicts.json`` and the run
   fails unless ``--on-conflict`` says otherwise. It never silently truncates
   the file to N rows.
3. Detects *distinct* ids that share the same prompt text and saves the id
   mapping (``id_map.json``) so duplicate questions can be audited later.
4. Splits the unique problems into train/dev with a fixed seed (default 42),
   holding out ``--dev-size`` (default 512) problems for the dev monitor set,
   and writes both the split manifest and the uid lists.
5. Writes verl-format parquet files that all three branches can read directly
   (``prompt`` messages are passed through untouched -- DAPO already carries its
   own instruction, so no math prompt is wrapped twice).

Outputs under ``--out-dir``::

    train.parquet        train split (prompt + reward_model + extra_info)
    dev.parquet          dev monitor split
    manifest.json        source, revision, hashes, counts, schema fingerprint
    split_manifest.json  seed, sizes, sorted uid lists
    id_map.json          index -> duplicate count / prompt hash / ground truth
    gt_conflicts.json    same-id inconsistencies (empty list when clean)
    stats.json           length statistics (with --tokenizer) and drop counts
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_template import (  # noqa: E402
    PROMPT_TEMPLATE_VERSION,
    canonicalize_raw_prompt,
    hash_raw_prompt,
    sha256_file,
    sha256_text,
)

DEFAULT_REPO_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_FILENAME = "data/dapo-math-17k.parquet"
# Revision observed when this pipeline was written; pin it for reproducibility.
DEFAULT_REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"

PROMPT_OUTPUT_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])) ,
        ("extra_info", pa.struct([("index", pa.string())])),
    ]
)


def log(message: str) -> None:
    print(f"[prepare_prompts] {message}", flush=True)


def resolve_source(args: argparse.Namespace) -> tuple[Path, dict]:
    """Return (parquet path, provenance dict). Downloads only when needed."""
    if args.input:
        path = Path(args.input).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"[prepare_prompts] ERROR: input parquet not found: {path}")
        return path, {"kind": "local", "path": str(path)}

    from huggingface_hub import hf_hub_download

    revision = args.revision
    path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            revision=revision,
        )
    )
    return path, {
        "kind": "huggingface",
        "repo_id": args.repo_id,
        "filename": args.filename,
        "revision": revision,
        "path": str(path),
    }


def iter_rows(path: Path, *, batch_size: int, limit_raw: int = 0):
    parquet = pq.ParquetFile(path)
    seen = 0
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield row
            seen += 1
            if limit_raw and seen >= limit_raw:
                return


def deduplicate(path: Path, *, batch_size: int, limit_raw: int, on_conflict: str):
    """Stream the raw file, keeping the first row of every extra_info.index."""
    unique: dict[str, dict] = {}
    raw_rows = 0
    duplicate_rows = 0
    conflicts: list[dict] = []
    prompt_to_indices: dict[str, list[str]] = defaultdict(list)
    occurrences: dict[str, int] = defaultdict(int)

    for row in iter_rows(path, batch_size=batch_size, limit_raw=limit_raw):
        raw_rows += 1
        index = row.get("extra_info", {}).get("index")
        if index is None or str(index).strip() == "":
            raise SystemExit("[prepare_prompts] ERROR: row without extra_info.index")
        index = str(index)
        occurrences[index] += 1
        prompt = row.get("prompt")
        ground_truth = str(row.get("reward_model", {}).get("ground_truth", ""))
        prompt_hash = hash_raw_prompt(prompt)

        existing = unique.get(index)
        if existing is None:
            unique[index] = {
                "index": index,
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "ground_truth": ground_truth,
                "data_source": row.get("data_source", "math_dapo"),
                "ability": row.get("ability", "MATH"),
                "reward_style": row.get("reward_model", {}).get("style", "rule-lighteval/MATH_v2"),
            }
            prompt_to_indices[prompt_hash].append(index)
            continue

        duplicate_rows += 1
        same_prompt = existing["prompt_hash"] == prompt_hash
        same_gt = existing["ground_truth"] == ground_truth
        if not (same_prompt and same_gt):
            conflicts.append(
                {
                    "index": index,
                    "prompt_hash_kept": existing["prompt_hash"],
                    "prompt_hash_seen": prompt_hash,
                    "ground_truth_kept": existing["ground_truth"],
                    "ground_truth_seen": ground_truth,
                    "same_prompt": same_prompt,
                    "same_ground_truth": same_gt,
                }
            )

    if conflicts:
        log(f"WARNING: {len(conflicts)} conflicting duplicate groups for the same extra_info.index")
        for conflict in conflicts[:5]:
            log(f"  index={conflict['index']} same_prompt={conflict['same_prompt']} same_gt={conflict['same_ground_truth']}")
        if on_conflict == "error":
            raise SystemExit(
                "[prepare_prompts] ERROR: same extra_info.index maps to inconsistent prompt/ground_truth. "
                "Re-run with --on-conflict keep-first (recorded in gt_conflicts.json) or drop to continue."
            )
        if on_conflict == "drop":
            for conflict in conflicts:
                unique.pop(conflict["index"], None)
            log(f"dropped {len(conflicts)} conflicting indices")

    same_prompt_groups = {
        prompt_hash: indices for prompt_hash, indices in prompt_to_indices.items() if len(indices) > 1
    }
    same_prompt_groups = {
        prompt_hash: indices
        for prompt_hash, indices in same_prompt_groups.items()
        if all(index in unique for index in indices)
    }

    return unique, raw_rows, duplicate_rows, conflicts, same_prompt_groups, occurrences


def prompt_groups(unique: dict[str, dict]) -> dict[str, list[str]]:
    """Group every surviving uid by its prompt hash (singletons included)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for index in sorted(unique):
        groups[unique[index]["prompt_hash"]].append(index)
    return dict(groups)


def split(unique: dict[str, dict], *, seed: int, dev_size: int):
    """Split the unique problems into train/dev **by prompt**, never by id.

    DAPO-Math-17k contains the same question under several ``extra_info.index``
    values (495 such groups in the pinned revision). Splitting the uids directly
    puts those twins on opposite sides of the boundary, which leaks the dev
    questions into training and makes the downstream leakage check fail. So the
    unit of the split is the prompt-hash group: a group is drawn for dev as a
    whole, and the dev size is the requested size plus the remainder of the last
    group (reported, never hidden).
    """
    groups = prompt_groups(unique)
    if dev_size < 0:
        raise SystemExit("[prepare_prompts] ERROR: --dev-size must be >= 0")
    if dev_size >= len(unique):
        raise SystemExit(
            f"[prepare_prompts] ERROR: --dev-size {dev_size} >= unique problems {len(unique)}; "
            "no training data would remain"
        )

    rng = random.Random(seed)
    order = sorted(groups)  # deterministic before shuffling
    rng.shuffle(order)

    dev_keys: list[str] = []
    dev_count = 0
    for key in order:
        if dev_count >= dev_size:
            break
        dev_keys.append(key)
        dev_count += len(groups[key])

    dev_key_set = set(dev_keys)
    dev_uids = sorted(index for key in dev_keys for index in groups[key])
    train_uids = sorted(index for key in order if key not in dev_key_set for index in groups[key])

    leaked = sorted({unique[index]["prompt_hash"] for index in train_uids}
                    & {unique[index]["prompt_hash"] for index in dev_uids})
    if leaked:
        raise SystemExit(
            f"[prepare_prompts] ERROR: internal split bug - {len(leaked)} prompt(s) appear in both splits"
        )

    info = {
        "unit": "prompt_hash_group",
        "prompt_groups_total": len(groups),
        "same_prompt_groups_total": sum(1 for indices in groups.values() if len(indices) > 1),
        "dev_size_requested": dev_size,
        "dev_size_actual": len(dev_uids),
        "dev_size_overshoot": len(dev_uids) - dev_size,
        "dev_prompt_groups": len(dev_keys),
        "multi_id_groups_in_dev": sum(1 for key in dev_keys if len(groups[key]) > 1),
        "leaked_prompt_hashes": leaked,
    }
    return train_uids, dev_uids, info


def build_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "data_source": row["data_source"],
                "prompt": [{"role": m["role"], "content": m["content"]} for m in row["prompt"]],
                "ability": row["ability"],
                "reward_model": {"ground_truth": row["ground_truth"], "style": row["reward_style"]},
                "extra_info": {"index": row["index"]},
            }
            for row in rows
        ],
        schema=PROMPT_OUTPUT_SCHEMA,
    )


def token_length_stats(texts: list[str], tokenizer_path: str, max_prompt_length: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths = []
    over = 0
    for text in texts:
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n_tokens)
        if n_tokens > max_prompt_length:
            over += 1
    lengths.sort()

    def quantile(q: float) -> int:
        if not lengths:
            return 0
        position = min(len(lengths) - 1, max(0, int(round(q * (len(lengths) - 1)))))
        return lengths[position]

    return {
        "count": len(lengths),
        "min": lengths[0] if lengths else 0,
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": lengths[-1] if lengths else 0,
        "max_prompt_length": max_prompt_length,
        "over_length_count": over,
        "over_length_rate": (over / len(lengths)) if lengths else 0.0,
        "tokenizer_path": tokenizer_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="Directory for train/dev parquet and manifests.")
    parser.add_argument("--input", default=None, help="Local DAPO parquet. When omitted it is downloaded.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="Pinned HF revision (recorded in the manifest).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--limit-raw", type=int, default=0, help="Only read the first N raw rows (smoke tests).")
    parser.add_argument("--on-conflict", choices=["error", "keep-first", "drop"], default="error")
    parser.add_argument("--expected-unique", type=int, default=0,
                        help="Fail when the unique count differs (0 = no check). DAPO-Math-17k is expected to give 17917.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path for prompt length stats.")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--card-full-train", action="store_true",
                        help="Emit the full unique set as train (no dev hold-out); dev comes from --dev-file.")
    parser.add_argument("--dev-file", default=None,
                        help="With --card-full-train: an explicit independent dev parquet (e.g. aime24).")
    parser.add_argument("--allow-dev-leak", action="store_true",
                        help="With --card-full-train: record (instead of refusing) dev questions that also "
                             "occur in train.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path, provenance = resolve_source(args)
    log(f"source: {source_path}")
    file_hash = sha256_file(str(source_path))
    log(f"sha256: {file_hash}")

    unique, raw_rows, duplicate_rows, conflicts, same_prompt_groups, occurrences = deduplicate(
        source_path, batch_size=args.batch_size, limit_raw=args.limit_raw, on_conflict=args.on_conflict
    )
    log(
        f"raw rows={raw_rows} unique={len(unique)} duplicate rows dropped={duplicate_rows} "
        f"same-id conflicts={len(conflicts)} same-prompt-different-id groups={len(same_prompt_groups)}"
    )
    if args.expected_unique and len(unique) != args.expected_unique:
        raise SystemExit(
            f"[prepare_prompts] ERROR: expected {args.expected_unique} unique problems, got {len(unique)}. "
            "Inspect the source file before training."
        )

    groups = prompt_groups(unique)
    if args.card_full_train:
        train_uids = sorted(unique)
        dev_uids: list[str] = []
        split_info = {
            "unit": "external_dev_file",
            "prompt_groups_total": len(groups),
            "same_prompt_groups_total": sum(1 for indices in groups.values() if len(indices) > 1),
            "dev_size_requested": None,
            "dev_size_actual": None,
            "dev_size_overshoot": None,
            "dev_prompt_groups": None,
            "multi_id_groups_in_dev": None,
            "leaked_prompt_hashes": [],
        }
        log("card_full_train: all unique problems go to train; the dev set must be supplied via --dev-file")
    else:
        train_uids, dev_uids, split_info = split(unique, seed=args.seed, dev_size=args.dev_size)
    log(
        f"train={len(train_uids)} dev={len(dev_uids)} "
        f"(split unit={split_info['unit']}, dev groups={split_info['dev_prompt_groups']}, "
        f"overshoot={split_info['dev_size_overshoot']})"
    )

    train_table = build_table([unique[uid] for uid in train_uids])
    pq.write_table(train_table, out_dir / "train.parquet")
    if dev_uids:
        dev_table = build_table([unique[uid] for uid in dev_uids])
        pq.write_table(dev_table, out_dir / "dev.parquet")
    else:
        dev_table = None
        if args.dev_file:
            dev_source = Path(args.dev_file).expanduser().resolve()
            if not dev_source.exists():
                raise SystemExit(f"[prepare_prompts] ERROR: --dev-file not found: {dev_source}")
            pq.write_table(pq.read_table(dev_source), out_dir / "dev.parquet")
            log(f"dev.parquet copied from {dev_source}")

    overlap = set(train_uids) & set(dev_uids)
    if overlap:
        raise SystemExit(f"[prepare_prompts] ERROR: train/dev overlap of {len(overlap)} uids")

    # Independent dev splits (documents from outside DAPO) are checked too: a
    # shared question means the dev accuracy is measured on trained material.
    train_hashes = {unique[uid]["prompt_hash"] for uid in train_uids}
    dev_hashes = {unique[uid]["prompt_hash"] for uid in dev_uids}
    if not dev_uids and (out_dir / "dev.parquet").exists():
        external = pq.read_table(out_dir / "dev.parquet").to_pylist()
        dev_hashes = {hash_raw_prompt(row["prompt"]) for row in external}
        leaked = sorted(dev_hashes & train_hashes)
        split_info["external_dev_rows"] = len(external)
        split_info["leaked_prompt_hashes"] = leaked
        if leaked:
            message = (
                f"[prepare_prompts] train/dev prompt leak: {len(leaked)} of {len(dev_hashes)} dev prompts "
                "also appear in train (the external dev split is not independent of DAPO)"
            )
            if not args.allow_dev_leak:
                raise SystemExit(message + "; pass --allow-dev-leak to record it and continue")
            log("WARNING: " + message)
        else:
            log(f"external dev split is disjoint from train ({len(dev_hashes)} prompts)")
    split_info["leaked_prompt_hashes_total"] = len(split_info["leaked_prompt_hashes"])

    id_map = {
        index: {
            "duplicate_count": occurrences.get(index, 0),
            "prompt_hash": row["prompt_hash"],
            "ground_truth": row["ground_truth"],
            "split": "train" if index in set(train_uids) else ("dev" if index in set(dev_uids) else "unused"),
        }
        for index, row in unique.items()
    }

    (out_dir / "id_map.json").write_text(
        json.dumps(
            {
                "same_id_duplicate_counts_included": True,
                "same_prompt_different_id_groups": {
                    prompt_hash: sorted(indices) for prompt_hash, indices in sorted(same_prompt_groups.items())
                },
                "indices": id_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "gt_conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "dev_size_requested": args.dev_size,
                "train_size": len(train_uids),
                "dev_size": len(dev_uids),
                "card_full_train": bool(args.card_full_train),
                "split": split_info,
                "train_uids": train_uids,
                "dev_uids": dev_uids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    stats: dict = {
        "raw_rows": raw_rows,
        "unique_problems": len(unique),
        "duplicate_rows_dropped": duplicate_rows,
        "same_id_conflicts": len(conflicts),
        "same_prompt_different_id_groups": len(same_prompt_groups),
        "train_size": len(train_uids),
        "dev_size": len(dev_uids),
        "split": split_info,
        "train_dev_prompt_leak": split_info["leaked_prompt_hashes_total"],
        "drop_last_note": "verl train dataloader uses drop_last=True; the dropped sample count is reported by the trainer",
    }
    if args.tokenizer:
        texts = [canonicalize_raw_prompt(unique[uid]["prompt"]) for uid in train_uids]
        stats["prompt_length"] = token_length_stats(texts, args.tokenizer, args.max_prompt_length)

    manifest = {
        "stage": "prepare_prompts",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "source": provenance,
        "source_sha256": file_hash,
        "source_size_bytes": source_path.stat().st_size,
        "on_conflict": args.on_conflict,
        "expected_unique": args.expected_unique,
        "raw_rows": raw_rows,
        "unique_problems": len(unique),
        "train_size": len(train_uids),
        "dev_size": len(dev_uids),
        "split": split_info,
        "train_dev_prompt_leak": split_info["leaked_prompt_hashes_total"],
        "outputs": {
            "train_parquet": str(out_dir / "train.parquet"),
            "train_sha256": sha256_file(str(out_dir / "train.parquet")),
            "dev_parquet": str(out_dir / "dev.parquet") if (out_dir / "dev.parquet").exists() else None,
            "dev_sha256": sha256_file(str(out_dir / "dev.parquet")) if (out_dir / "dev.parquet").exists() else None,
        },
        "train_manifest_digest": sha256_text(json.dumps(train_uids)),
        "dev_manifest_digest": sha256_text(json.dumps(dev_uids)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"wrote train.parquet ({train_table.num_rows} rows)" + (f", dev.parquet ({dev_table.num_rows} rows)" if dev_table is not None else ""))
    log(f"manifests -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
