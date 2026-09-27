#!/usr/bin/env python3
"""Audit historical machine-readable run paths after the directory move."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/gad_qwen3_1.7b/qwen3-1.7b-4x3090'
OLD = ROOT.parent
FILES = ['stage_seqkd.json', 'stage_warmup.json', 'stage_gad.json',
         'handoff_manifest.json', 'final_model_manifest.json',
         'seqkd/COMPLETED.json', 'warmup/COMPLETED.json', 'gad/COMPLETED.json']


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def main():
    report = {'run_id': RUN.name, 'old_run_link': str(OLD / 'runs/gad_qwen3_1.7b' / RUN.name),
              'old_data_link': str(OLD / 'dataset/dapo_math_17k_gad'), 'manifests': {}}
    for name in FILES:
        path = RUN / name
        if not path.exists():
            continue
        values = list(strings(json.loads(path.read_text())))
        old_paths = sorted({part for value in values for part in value.split(":")
                            if part.startswith(str(OLD / 'runs/gad_qwen3_1.7b'))
                            or part.startswith(str(OLD / 'dataset/dapo_math_17k_gad'))})
        report['manifests'][name] = {'old_path_references': old_paths,
                                     'missing_old_paths': [v for v in old_paths if not Path(v).exists()]}
    for key in ('old_run_link', 'old_data_link'):
        report[key + '_valid'] = Path(report[key]).is_symlink() and Path(report[key]).exists()
    target = RUN / 'reports/path_audit.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    missing = sum(len(item['missing_old_paths']) for item in report['manifests'].values())
    if missing or not report['old_run_link_valid'] or not report['old_data_link_valid']:
        raise SystemExit(f'path audit failed: {missing} missing old references')
    print(f'path audit passed: {len(report["manifests"])} manifests, {missing} missing paths')

if __name__ == '__main__':
    main()
