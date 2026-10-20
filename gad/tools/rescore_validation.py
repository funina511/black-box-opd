#!/usr/bin/env python3
"""Rescore saved training validation responses in a NEW directory; no generation."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scoring.math import score, identity, evaluation_value
from rubric.core.contracts import file_hash


def run(rows_path,output,expected_count,step,scope):
    source=Path(rows_path);output=Path(output)
    if not source.is_file():raise ValueError('saved responses missing: regeneration is required')
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if expected_count<1:raise ValueError('expected-count must be positive')
    scored=[]
    for i,row in enumerate(rows):
        if 'output' not in row or 'ground_truth' not in row:raise ValueError(f'row {i+1}: missing saved output/reference; regeneration or explicit source recovery is required')
        scored.append({**row,**score(row['output'],row['ground_truth']), 'original_source_file':str(source.resolve()),'original_source_line':i+1})
    ids=[r.get('response_id',str(i)) for i,r in enumerate(scored)]
    duplicate=len(ids)-len(set(ids));valid_count=sum(evaluation_value(r) is not None for r in scored)
    complete=len(scored)==expected_count and not duplicate and valid_count==len(scored)
    unique={};unexpected=0
    for i,row in enumerate(scored):
        index=row.get('row_index',i)
        if not isinstance(index,int) or not 0<=index<expected_count:
            unexpected+=1;continue
        unique.setdefault(row.get('response_id',str(index)),row)
    complete=complete and not unexpected and len(unique)==expected_count
    provisional=sum(evaluation_value(r) or 0 for r in unique.values())/expected_count
    metrics={'schema_version':'gad.validation_rescore.v1','global_step':step,'validation_scope':scope,
        'expected_count':expected_count,'actual_count':len(scored),'duplicate_count':duplicate,'unexpected_count':unexpected,
        'evaluation_complete':complete,'acc':provisional if complete else None,'provisional_acc':provisional,
        'provisional_denominator':expected_count,'unknown_contribution':'zero_in_provisional_only',
        'source':str(source.resolve()),'source_sha256':file_hash(source),'scorer_identity':identity(),
        'generation_reused':True,'failures':[{'source_line':i+1,'response_id':r.get('response_id'),
            'parse_status':r['parse_status'],'grade_status':r['grade_status'],'error_reason':r.get('error_reason')}
            for i,r in enumerate(scored) if evaluation_value(r) is None]}
    # Validate every input before creating the output; never overwrite any old pass.
    output.mkdir(parents=True,exist_ok=False)
    (output/'scored_rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--expected-count',required=True,type=int);parser.add_argument('--step',required=True,type=int)
    parser.add_argument('--scope',choices=['quick','full'],required=True)
    args=parser.parse_args();print(json.dumps(run(args.rows,args.output,args.expected_count,args.step,args.scope),ensure_ascii=False,indent=2))
