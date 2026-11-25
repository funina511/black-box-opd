#!/usr/bin/env python3
"""Build a stable UID subset from the existing full dev parquet."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    full = pq.read_table(args.full)
    rows = full.to_pylist()
    uids = [str(row['extra_info']['index']) for row in rows]
    if len(set(uids)) != len(uids) or args.size < 1 or args.size > len(uids):
        raise ValueError('dev UIDs must be unique and size must be within dev row count')
    selected = set(random.Random(args.seed).sample(uids, args.size))
    indices = [i for i, uid in enumerate(uids) if uid in selected]
    quick = full.take(pa.array(indices))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        existing = pq.read_table(args.out)
        if existing != quick:
            raise RuntimeError(f'existing subset differs: {args.out}')
    else:
        pq.write_table(quick, args.out)
    manifest = {
        'full_file': str(args.full.resolve()), 'full_sha256': digest(args.full),
        'quick_file': str(args.out.resolve()), 'quick_sha256': digest(args.out),
        'method': 'Python random.Random(seed).sample(sorted by source row order); retain source order',
        'seed': args.seed, 'size': args.size, 'full_rows': len(rows),
        'uids': [uids[i] for i in indices],
    }
    path = args.out.with_suffix('.manifest.json')
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'uids'}, indent=2))

if __name__ == '__main__':
    main()
