#!/usr/bin/env python3
"""Compare frozen formal contracts; rubric-only differences are explicit."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from config import compose_config
from rubric.core.contracts import file_hash


@lru_cache(maxsize=8)
def directory_identity(path):
    path=Path(path)
    return {p.name:file_hash(p) for p in sorted(path.iterdir()) if p.is_file() and
        (p.suffix in ('.json','.safetensors','.jinja') or p.name in ('tokenizer.model','merges.txt','vocab.txt'))}


def compare(a,b):
    allowed=lambda key:key.startswith('rubric.') or key in ('run.run_id','run.output_root')
    differences={key:{'baseline':a.get(key),'rubric':b.get(key)} for key in sorted(set(a.data)|set(b.data)) if a.get(key)!=b.get(key)}
    unexpected={key:value for key,value in differences.items() if not allowed(key)}
    for key in ('models.student','models.discriminator_init'):
        if directory_identity(a.require(key))!=directory_identity(b.require(key)):unexpected[key]='model/tokenizer bytes differ'
    for key in ('data.train_file','data.teacher_index','data.val_file'):
        if file_hash(a.require(key))!=file_hash(b.require(key)):unexpected[key]='data bytes differ'
    return {'comparable_contract':not unexpected,'allowed_differences':differences,'unexpected_differences':unexpected,
        'student_identity':directory_identity(a.require('models.student')),
        'critic_initialization_identity':directory_identity(a.require('models.discriminator_init')),
        'data_identity':{key:file_hash(a.require(key)) for key in ('data.train_file','data.val_file','data.teacher_index')},
        'scorer_fingerprint':a.get('scoring.math.fingerprint'),'rollout_seed':a.get('rollout.seed')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',default='qwen3_0.6b');p.add_argument('--dataset',default='dapo_math_5k')
    p.add_argument('--baseline',default='configs/overrides/rubric_v2_baseline.yaml');p.add_argument('--joint',default='configs/overrides/rubric_v2_joint.yaml');p.add_argument('--output',required=True)
    args=p.parse_args();a=compose_config(model=args.model,dataset=args.dataset,overrides=[args.baseline]);b=compose_config(model=args.model,dataset=args.dataset,overrides=[args.joint]);result=compare(a,b)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(result['comparable_contract']);raise SystemExit(0 if result['comparable_contract'] else 1)
