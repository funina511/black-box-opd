#!/usr/bin/env python3
"""Appendix-D judge protocol validation, with one shared finite GPU budget."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'gad/tools'))
from config import compose_config
from extension_config import service_identity
from rubric.core.contracts import digest,judge_messages,file_hash
from gad.extensions.judge_client import JudgeClient


def fixtures():
    equation='Solve 2x+3=7 over real numbers.'
    examples=[('complete',equation,'Subtract 3: 2x=4.\nDivide by 2: x=2.\nAnswer: 2',False),
        ('prefix',equation,'Subtract 3: 2x=4.',True),
        ('explicit_error',equation,'Subtract 3: 2x=4.\nDividing by 2 gives x=3.\nAnswer: 3',False),
        ('claimed_repair',equation,'Subtract 3: 2x=4.\nDividing gives x=3.\nThe error is now corrected.',False),
        ('real_repair',equation,'Subtract 3: 2x=4.\nI first wrote x=3; that division was wrong.\nCorrecting the division: x=2. Then 2*2+3=7.\nAnswer: 2',False),
        ('restatement',equation,'We need to solve the equation 2x+3=7 over real numbers.',False),
        ('repeat',equation,'Subtract 3: 2x=4.\nSubtract 3: 2x=4.\nSubtract 3: 2x=4.',True),
        ('partial_formula',equation,'Subtract 3: 2x=4.\nx=4/',True),
        ('alternative',equation,'For a linear equation with nonzero coefficient 2 the solution is unique. Substituting x=2 gives 2*2+3=7, so this is that unique solution.\nAnswer: 2',False),
        ('illegal_division','Solve x(x-1)=0 over the reals.','Divide by x to get x-1=0. Thus all solutions are x=1.\nAnswer: 1',False),
        ('pending_branch','Solve x(x-1)=0 over the reals.','For the x!=0 branch, division gives x-1=0, hence x=1 in this branch.',True),
        ('no_math',equation,'I will think about it.',False)]
    return [{'id':name,'prompt':prompt,'response':response,'ground_truth':'{0,1}' if 'x(x-1)' in prompt else '2',
        'truncated':truncated,'human_review':'pending','model_assisted_review':'pending'} for name,prompt,response,truncated in examples]


def main():
    import argparse
    import math
    parser=argparse.ArgumentParser(description='Appendix D: one 24-case judge check, at most one affected retry, 30 minutes total; no training launcher')
    default=ROOT/'reports/rubric_v2/judge_protocol_v21_20261015'
    parser.add_argument('--output',default=str(default))
    parser.add_argument('--config',default=str(default/'candidate.yaml'))
    parser.add_argument('--inputs',default=str(default/'selected24.jsonl'))
    parser.add_argument('--retry-ids',help='Comma-separated IDs from the first cohort; requires a concrete fix note')
    parser.add_argument('--fix-note')
    parser.add_argument('--output-tokens',type=int,choices=(1536,2048),default=1536)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    ledger_path=output/'phase_budget.json'
    rows=[json.loads(line) for line in Path(args.inputs).read_text().splitlines()]
    if len(rows)!=24 or sum(r['kind']=='real' for r in rows)!=16:raise ValueError('first cohort must be exactly 16 real + 8 boundary')
    from rubric.core.contracts import file_hash
    if ledger_path.exists():
        ledger=json.loads(ledger_path.read_text())
        if not args.retry_ids or not args.fix_note or len(ledger['rounds'])!=1:
            raise ValueError('existing phase: only one explicitly justified affected retry is allowed')
        if ledger['frozen_inputs_sha256']!=file_hash(args.inputs):raise ValueError('frozen cohort changed')
        ids=set(args.retry_ids.split(','));selected=[r for r in rows if r['id'] in ids]
        if not selected or len(selected)!=len(ids):raise ValueError('retry IDs must belong to the first cohort')
        if args.output_tokens==2048:
            first=[json.loads(line) for line in (output/'round1/samples.jsonl').read_text().splitlines()]
            truncated={r['id'] for r in first if r['result'].get('error')=='judge_output_truncated'}
            if not ids<=truncated:raise ValueError('2048 retry is limited to demonstrated output-truncated rows')
        label='round2_affected'
    else:
        if args.retry_ids or args.output_tokens!=1536:raise ValueError('first check uses the single 1536-token candidate')
        selected=rows;label='round1'
        ledger={'phase':'appendix_D_v2.1','started_epoch':time.time(),'wall_clock_limit_seconds':1800,
                'maximum_applications':48,'reserved_applications':0,'actual_applications':0,'actual_http_attempts':0,
                'actual_training_steps':0,'training_smoke':'not_started','rounds':[],
                'frozen_inputs_sha256':file_hash(args.inputs),'prior_reports_preserved':True}
    remaining=math.floor(1800-(time.time()-ledger['started_epoch']))
    if remaining<=30:raise TimeoutError('shared 30-minute judge/GPU budget exhausted; no model launched')
    if ledger['reserved_applications']+len(selected)>48:raise ValueError('48-application budget exceeded')
    directory=output/label
    if directory.exists():raise ValueError('round output already exists')
    directory.mkdir()
    from config import compose_config
    import yaml
    cfg=compose_config(args.config,settings=[f'rubric.judge.max_output_tokens={args.output_tokens}', 'rubric.judge.concurrency=1'])
    (directory/'candidate.yaml').write_text(yaml.safe_dump(cfg.to_dict(),allow_unicode=True,sort_keys=True))
    (directory/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    record={'name':label,'reserved_applications':len(selected),'fix_note':args.fix_note,'output_tokens':args.output_tokens,
            'gpu_ids':cfg.get('resources.judge_gpus'),'config_sha256':file_hash(directory/'candidate.yaml')}
    ledger['reserved_applications']+=len(selected);ledger['rounds'].append(record)
    def save_ledger():
        ledger['elapsed_seconds']=time.time()-ledger['started_epoch'];ledger_path.write_text(json.dumps(ledger,indent=2))
    save_ledger()
    argv=[cfg.require('run.python_bin'),str(ROOT/'gad/tools/calibrate_rubric_judge.py'),
          '--config',str(directory/'candidate.yaml'),'--samples',str(directory/'inputs.jsonl'),
          '--output',str(directory),'--wall-seconds',str(remaining)]
    before=time.monotonic()
    try:
        with (directory/'run.log').open('w') as stream:
            process=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                     env={**os.environ,'OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'})
            try:code=process.wait(timeout=remaining+10)
            except BaseException:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                raise
        record['exit_code']=code
        if code:raise RuntimeError('bounded judge round failed; inspect '+str(directory/'run.log'))
    except BaseException as exc:
        record['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Also cover a child interrupted before its own finally block ran.
        stopped=subprocess.run(['bash','scripts/rubric_judge.sh','--config',str(directory/'candidate.yaml'),'--action','stop'],cwd=ROOT,capture_output=True,text=True,timeout=30)
        record['cleanup']={'exit_code':stopped.returncode,'stdout':stopped.stdout.strip()}
        record['elapsed_seconds']=time.monotonic()-before
        if (directory/'budget.json').exists():
            budget=json.loads((directory/'budget.json').read_text());record['budget']=budget
        ledger['actual_applications']=sum(r.get('budget',{}).get('actual_applications',0) for r in ledger['rounds'])
        ledger['actual_http_attempts']=sum(r.get('budget',{}).get('actual_http_attempts',0) for r in ledger['rounds'])
        ledger['status']='judge_round_failed' if 'error' in record else 'judge_round_completed_training_not_started'
        save_ledger()
        print(json.dumps(ledger,indent=2))


if __name__=='__main__':main()
