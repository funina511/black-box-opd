#!/usr/bin/env python3
"""Create a derived parquet with selected-candidate metadata; keep text/UID/GT fixed."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from gad.extensions.teacher_provenance import bind_selected
from scoring.math import validate_ground_truth


def migrate(source, index, output, sidecar=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():raise ValueError('metadata migration must write a derived dataset')
    records = {}
    for line in Path(index).open():
        row = json.loads(line)
        if not row.get('_header'):
            uid = str(row.get('uid', row.get('index', '')))
            if uid in records:raise ValueError('duplicate teacher index UID: ' + uid)
            records[uid] = row
    selections = {}
    if sidecar:
        data = json.loads(Path(sidecar).read_text())
        if isinstance(data, dict):data = data.get('selections', data.get('selection', data.get('selected', data.get('records', []))))
        selections = {str(row['uid']): int(row['selected_index']) for row in data}
    rows = pq.read_table(source).to_pylist()
    original = deepcopy(rows)
    uids = [str(row['extra_info']['index']) for row in rows]
    if len(uids) != len(set(uids)):raise ValueError('duplicate dataset UID')
    for row in rows:
        validate_ground_truth(row['reward_model']['ground_truth'])
        uid = str(row['extra_info']['index']);record = records[uid]
        metadata = bind_selected(record['teacher_answers'], record.get('sampling', {}), row['teacher_response'],
            row['extra_info'].get('selected_teacher_index', selections.get(uid)))
        row['extra_info'].update(metadata)
    # Compare actual fields, not a tool's self-reported success flag.
    for before, after in zip(original, rows):
        if any(after[key] != value for key, value in before.items() if key != 'extra_info') or any(
                after['extra_info'].get(key) != value for key, value in before['extra_info'].items()):
            raise ValueError('migration changed original training content')
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    if output.exists():
        if pq.read_table(output).to_pylist() != rows:raise ValueError('existing derived data changed')
    else:pq.write_table(table, output)
    manifest = {'source': str(source.resolve()), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'derived_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'rows': len(rows), 'uid_text_gt_preserved': True,
        'teacher_metadata_version': 'teacher.selection.v2', 'teacher_index': str(Path(index).resolve())}
    manifest['teacher_index_sha256'] = hashlib.sha256(Path(index).read_bytes()).hexdigest()
    manifest['uid_order_sha256'] = hashlib.sha256(json.dumps(uids).encode()).hexdigest()
    stats_path = Path(str(source) + '.stats.json')
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if stats:
        if stats.get('out_parquet_sha256') != manifest['source_sha256'] or stats.get('rows_out') != len(rows):
            raise ValueError('source stats are not bound to the input parquet')
        stats = {**stats, 'length_statistics_source': {'path': str(stats_path.resolve()),
            'sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest(), 'retokenized': False}}
    stats.update(rows_in=len(rows), rows_out=len(rows), rows_dropped=0,
        out_parquet=str(output.resolve()), out_parquet_sha256=manifest['derived_sha256'],
        source_truncated=sum(row['extra_info']['source_truncated'] for row in rows))
    if not stats.get('teacher_token_length'):
        stats['teacher_token_length'] = None  # Unknown is not zero; caller must supply verified source stats.
    Path(str(output) + '.stats.json').write_text(json.dumps(stats, indent=2))
    (output.parent / (output.stem + '.metadata.json')).write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--source', required=True);p.add_argument('--index', required=True)
    p.add_argument('--output', required=True);p.add_argument('--sidecar');args = p.parse_args()
    print(json.dumps(migrate(args.source, args.index, args.output, args.sidecar), indent=2))
