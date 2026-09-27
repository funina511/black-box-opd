#!/usr/bin/env python3
"""Make a reproducible UID-level subset of an existing GAD training split.

The teacher responses are copied from the prepared GAD parquet; this does not
resample the teacher or alter the held-out dev split. Raw and GAD parquet rows
are selected in the same seeded order so their UIDs stay aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def uids(table: pa.Table) -> list[str]:
    if "extra_info" not in table.column_names:
        raise ValueError("parquet is missing extra_info")
    values = [str(row["index"]) for row in table["extra_info"].to_pylist()]
    if len(values) != len(set(values)):
        raise ValueError("source parquet has duplicate UIDs")
    return values


def make_subset(raw_path: Path, gad_path: Path, dev_path: Path,
                out_dir: Path, *, size: int, seed: int) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out_dir}")
    raw = pq.read_table(raw_path)
    gad = pq.read_table(gad_path)
    dev = pq.read_table(dev_path, columns=["extra_info"])
    raw_uids, gad_uids, dev_uids = uids(raw), uids(gad), set(uids(dev))
    if set(raw_uids) != set(gad_uids):
        raise ValueError("raw and GAD train parquets contain different UIDs")
    if set(gad_uids) & dev_uids:
        raise ValueError("train and dev parquets overlap by UID")
    if not 0 < size <= len(gad_uids):
        raise ValueError(f"sample size must be within 1..{len(gad_uids)}")
    if "teacher_response" not in gad.column_names:
        raise ValueError("GAD parquet is missing teacher_response")

    selected = random.Random(seed).sample(sorted(gad_uids), size)
    raw_index = {uid: i for i, uid in enumerate(raw_uids)}
    gad_index = {uid: i for i, uid in enumerate(gad_uids)}
    raw_subset = raw.take(pa.array([raw_index[uid] for uid in selected], type=pa.int64()))
    gad_subset = gad.take(pa.array([gad_index[uid] for uid in selected], type=pa.int64()))
    if any(not str(answer or "").strip() for answer in gad_subset["teacher_response"].to_pylist()):
        raise ValueError("selected GAD rows contain an empty teacher_response")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / "train.parquet"
    gad_out = out_dir / "gad_train.parquet"
    uid_out = out_dir / "selected_uids.txt"
    pq.write_table(raw_subset, raw_out, compression="zstd")
    pq.write_table(gad_subset, gad_out, compression="zstd")
    uid_out.write_text("\n".join(selected) + "\n", encoding="utf-8")
    manifest = {
        "stage": "make_train_subset",
        "selection": {"unit": "extra_info.index", "method": "random_sample_without_replacement",
                      "seed": seed, "size": size, "source_train_size": len(gad_uids)},
        "sources": {"train_parquet": str(raw_path.resolve()), "train_sha256": sha256_file(raw_path),
                    "gad_train_parquet": str(gad_path.resolve()), "gad_train_sha256": sha256_file(gad_path),
                    "dev_parquet": str(dev_path.resolve()), "dev_sha256": sha256_file(dev_path)},
        "outputs": {"train_parquet": str(raw_out.resolve()), "train_sha256": sha256_file(raw_out),
                    "gad_train_parquet": str(gad_out.resolve()), "gad_train_sha256": sha256_file(gad_out),
                    "selected_uids": str(uid_out.resolve()), "selected_uids_sha256": sha256_file(uid_out)},
        "dev_size": len(dev_uids),
        "train_dev_uid_overlap": 0,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--gad-train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = make_subset(args.raw_train, args.gad_train, args.dev,
                           args.out_dir, size=args.size, seed=args.seed)
    print(json.dumps({"out_dir": str(args.out_dir), "selection": manifest["selection"],
                      "train_dev_uid_overlap": manifest["train_dev_uid_overlap"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
