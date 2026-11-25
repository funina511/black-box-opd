#!/usr/bin/env python3
"""CPU-only old/new token audit and a preselected Appendix-D 24-case cohort."""
import argparse
from collections import defaultdict, Counter
import importlib
import json
from pathlib import Path
import random
import sys
import types

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools')]
from config import compose_config
from rubric.core.contracts import judge_messages, file_hash, digest
from rubric.core.process_v2 import compact_response, protocol_identity, response_from_messages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--frozen',default=str(ROOT/'reports/rubric_v2/qwen3_14b_128_20261015'))
    args=parser.parse_args();out=Path(args.output).resolve();source=Path(args.frozen).resolve()
    if (out/'selected24.jsonl').exists() and not args.audit_only:raise ValueError('selection already frozen; do not reselect after outputs')
    cfg=compose_config(args.config)
    before=out/'before'
    package=types.ModuleType('frozen_protocol_before');package.__path__=[str(before/'rubric/core')]
    sys.modules['frozen_protocol_before']=package
    old=importlib.import_module('frozen_protocol_before.contracts')
    spec=Path(cfg.require('rubric.spec')).read_text();instruction=Path(cfg.require('rubric.prompt')).read_text()
    old_spec=(before/'rubric/specs/math_process_v2.yaml').read_text();old_instruction=(before/'rubric/prompts/judge_math_process_v2.txt').read_text()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(cfg.require('rubric.judge.tokenizer_path'),local_files_only=True)
    def count(text):return len(tokenizer.encode(text,add_special_tokens=False))
    def chat(messages):return len(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False))
    rows=[json.loads(line) for line in (source/'inputs.jsonl').read_text().splitlines()]
    original_lengths={row['id']:row for row in (json.loads(line) for line in (source/'input_lengths.jsonl').read_text().splitlines())}
    audits=[];new_requests={}
    for row in rows:
        visibility={'response_source_truncated':row['truncated'],'training_token_truncated':False,'prompt_truncated':False,'judge_context_overflow':False}
        old_messages=old.judge_messages(row['prompt'],row['response'],row['ground_truth'],old_spec,old_instruction,'math_process_v2',visibility)
        messages=judge_messages(row['prompt'],row['response'],row['ground_truth'],spec,instruction,'math_process_v2',visibility)
        assert response_from_messages(messages)==row['response']
        a,b=chat(old_messages),chat(messages)
        assert a==original_lengths[row['id']]['input_tokens'],row['id']
        audits.append({'id':row['id'],'kind':row['kind'],'source_role':row.get('source_role'),'truncated':row['truncated'],
                       'old_full_chat_tokens':a,'new_full_chat_tokens':b,'raw_response_tokens':count(row['response']),
                       'problem_reference_tokens':count(json.dumps({'problem':row['prompt'],'reference_final_answer':row['ground_truth']},ensure_ascii=False)),
                       'old_fixed_instruction_tokens':count(old_messages[0]['content']),'new_fixed_instruction_tokens':count(messages[0]['content']),
                       'old_response_wrapper_tokens':count(json.dumps(json.loads(old_messages[1]['content'])['visible_response'],ensure_ascii=False)),
                       'new_response_wrapper_tokens':count(compact_response(row['response']))})
        new_requests[row['id']]=messages
    import numpy as np
    def statistics(selected):
        result={}
        for version in ('old','new'):
            numbers=[a[version+'_full_chat_tokens'] for a in selected]
            result[version]={**{name:float(np.percentile(numbers,p)) for name,p in (('p50',50),('p90',90),('p95',95))},'max':max(numbers),'mean':sum(numbers)/len(numbers)}
        result['old_exceeded_16k_output768']=sum(a['old_full_chat_tokens']+768>16384 for a in selected)
        result['new_exceeded_16k_output1536']=sum(a['new_full_chat_tokens']+1536>16384 for a in selected)
        result['new_exceeded_32k_output1536']=sum(a['new_full_chat_tokens']+1536>32768 for a in selected)
        return result
    summary={kind:statistics([a for a in audits if kind=='all' or a['kind']==kind]) for kind in ('all','real','boundary')}
    summary.update(protocol_identity=protocol_identity(spec,instruction),component_counts_are_not_additive=True,
                   config_sha256=file_hash(args.config),frozen_inputs_sha256=file_hash(source/'inputs.jsonl'))
    (out/'token_audit.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in audits))
    (out/'token_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if args.audit_only:
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return
    # Select without using new labels. Include role extremes and both teacher
    # truncations; student complete subgroup covers both answer outcomes.
    by_id={a['id']:a for a in audits};rng=random.Random(42);chosen=[]
    for role,truncated,number in (('teacher',False,6),('teacher',True,2),('student',False,4),('student',True,4)):
        pool=sorted([r for r in rows if r['kind']=='real' and r['source_role']==role and r['truncated']==truncated],key=lambda r:by_id[r['id']]['old_full_chat_tokens'])
        if len(pool)<number:raise ValueError('insufficient frozen stratum')
        selected=[pool[0],pool[-1]] if number>1 else [pool[-1]]
        if role=='student' and not truncated:
            for correct in (True,False):
                candidates=[r for r in pool if r['scoring']['correct'] is correct and r not in selected]
                if not any(r['scoring']['correct'] is correct for r in selected) and candidates:selected.append(candidates[0])
        selected+=rng.sample([r for r in pool if r not in selected],number-len(selected))
        chosen+=selected
    # Longest first is a real capacity probe at initial concurrency=1.
    chosen.sort(key=lambda r:by_id[r['id']]['new_full_chat_tokens'],reverse=True)
    assert len(chosen)==16 and len({r['id'] for r in chosen})==16
    assert max(by_id[r['id']]['old_full_chat_tokens'] for r in chosen)==max(a['old_full_chat_tokens'] for a in audits)
    for category in ('valid_prefix','real_repair','false_repair','pending_branch','partial_formula','no_math'):
        chosen.append(next(r for r in rows if r['kind']=='boundary' and r['category']==category and r['id'].endswith('_0')))
    prior=ROOT/'reports/rubric_v2/qwen3_14b_20261015/samples.jsonl'
    existing={r['id']:r for r in (json.loads(line) for line in prior.read_text().splitlines())}
    for name,allowed in (('illegal_division',{'verified_progress':[0],'unresolved_error_severity':[1]}),('alternative',{'verified_progress':[1],'unresolved_error_severity':[0]})):
        row=existing[name]
        chosen.append({k:row[k] for k in ('prompt','response','ground_truth','truncated')} | {'id':'frozen12_'+name,'kind':'boundary','category':name,'expected_allowed':allowed,'source_frozen':str(prior),'expectation_status':'model_assisted_anchor_human_pending'})
    assert len(chosen)==24
    median=float(np.median([a['new_full_chat_tokens'] for a in audits if a['kind']=='real']))
    for row in chosen:
        if row['kind']=='real':
            row['judge_input_tokens']=by_id[row['id']]['new_full_chat_tokens']
            row['length_group']='short' if row['judge_input_tokens'] <= median else 'long'
    (out/'selected24.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in chosen))
    (out/'selection.json').write_text(json.dumps({'seed':42,'count':24,'real':16,'boundary':8,'ids':[r['id'] for r in chosen],
        'source_128':str(source),'source_12':str(prior),'includes_longest_old_input':True,'selection_precedes_new_outputs':True,
        'real_groups':dict(Counter(str((r.get('source_role'),r['truncated'])) for r in chosen if r['kind']=='real'))},indent=2))
    (out/'request_examples.json').write_text(json.dumps({chosen[0]['id']:new_requests[chosen[0]['id']],chosen[16]['id']:new_requests[chosen[16]['id']]},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
