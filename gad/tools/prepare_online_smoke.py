#!/usr/bin/env python3
"""Derive a bounded engineering dataset without overwriting formal inputs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import compose_config
from extension_config import ROOT
from rubric.core.contracts import file_hash


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'train.yaml'))
    p.add_argument('--output',default=str(ROOT/'data/gad_online_smoke_v1'))
    p.add_argument('--count',type=int,default=16)
    args=p.parse_args();cfg=compose_config(args.config);out=Path(args.output)
    if out.exists():raise ValueError('smoke data directory already exists; reuse it or choose a new output')
    tokenizer=AutoTokenizer.from_pretrained(cfg.require('models.student'),local_files_only=True)
    table=pq.read_table(cfg.require('data.train_file'))
    selected=[]
    # Engineering calibration includes complete teachers and unfiltered current students.
    for row in table.to_pylist():
        if len(tokenizer.encode(row['teacher_response'],add_special_tokens=False)) < 400 and len(tokenizer.apply_chat_template(
                row[cfg.get('data.prompt_key')],tokenize=True,add_generation_prompt=True,enable_thinking=False)) < 480:
            selected.append(row)
            if len(selected)==args.count:break
    if len(selected)!=args.count:raise ValueError('not enough bounded teacher/prompt examples')
    out.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(selected,schema=table.schema),out/'train.parquet')
    uids={str(row['extra_info']['index']) for row in selected}
    records=[]
    with Path(cfg.require('data.teacher_index')).open() as source:
        for line in source:
            record=json.loads(line)
            if record.get('_header') or str(record.get('uid')) in uids:records.append(record)
    (out/'teacher_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    manifest={'schema_version':'gad.online_smoke.v1','purpose':'engineering only; not an efficacy dataset',
              'parent_train':cfg.require('data.train_file'),'parent_sha256':file_hash(cfg.require('data.train_file')),
              'uids':sorted(uids),'rows':len(selected),'teacher_fingerprint_digest':cfg.get('data.expected_fingerprint_digest'),
              'train_sha256':file_hash(out/'train.parquet'),'teacher_index_sha256':file_hash(out/'teacher_index.jsonl')}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(out)


if __name__=='__main__':main()
