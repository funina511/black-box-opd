#!/usr/bin/env python3
"""Bounded gad_dev pair diagnostic for opt-in Actor rewards. Never trains a model."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'gad/tools'), str(ROOT/'sample')]
from config import load_config
from rubric.core.contracts import file_hash


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def prepare(cfg, out):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source = Path(cfg.require('budget.init_from_warmup_run'))
    step = cfg.require('budget.init_step')
    handoff = json.loads((source/'handoff_manifest.json').read_text())
    assert handoff['ok'] and handoff['step'] == step
    actor, critic = [Path(handoff[role+'_hf']) for role in ('actor','critic')]
    tokenizer = AutoTokenizer.from_pretrained(actor, local_files_only=True)
    rows = pq.read_table(cfg.require('data.val_file')).to_pylist()
    order = list(range(len(rows))); random.Random(42).shuffle(order)
    records=[]
    for ordinal, i in enumerate(order):
        row=rows[i]
        ids=tokenizer.apply_chat_template(row[cfg.require('data.prompt_key')],tokenize=True,
                add_generation_prompt=True,enable_thinking=False)
        records.append({'ordinal':ordinal,'uid':str(row['extra_info']['index']),'dev_index':i,
            'prompt':row[cfg.require('data.prompt_key')],'ground_truth':row['reward_model']['ground_truth'],
            'prompt_token_ids':ids[:cfg.require('data.max_prompt_length')],
            'prompt_truncated':len(ids)>cfg.require('data.max_prompt_length')})
    path=out/'questions.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
    if path.exists(): assert path.read_text()==text,'fixed question order changed'
    else:path.write_text(text)
    identity={'source_warmup':str(source/'warmup'/f'global_step_{step}'),
        'actor':str(actor),'critic':str(critic),'handoff_manifest_sha256':file_hash(source/'handoff_manifest.json'),
        'data':str(cfg.require('data.val_file')),'data_sha256':file_hash(cfg.require('data.val_file')),
        'train_data_sha256':file_hash(cfg.require('data.train_file')),
        'rubric_spec_sha256':file_hash(cfg.require('rubric.spec')),
        'scorer_fingerprint':cfg.require('scoring.math.fingerprint'),
        'tokenizer_files':{p.name:file_hash(p) for p in actor.iterdir() if p.suffix in ('.json','.jinja') and p.name!='config.json'},
        'sampling':{'seed':42,'temperature':.8,'top_p':1.,'top_k':-1,'min_p':0.,'n':2,
            'max_tokens':8192,'enable_thinking':False,'max_prompt_length':2048,'max_model_len':10240},
        'available_dev_questions':len(records),
        'reuse_check':'Historical calibration has <=32 mixed Teacher/Student texts; dev validation is n=1 greedy. No paired gad_dev samples from this Warmup/settings were found. Old test-set ranking responses are excluded.',
        'token_replay':'Generation preserves the actual vLLM prompt and completion token IDs. Inference uses these IDs without text re-encoding.'}
    if (out/'input_manifest.json').exists():assert json.loads((out/'input_manifest.json').read_text())==identity
    else:write(out/'input_manifest.json',identity)
    print('Prepared fixed order',len(records),'dev questions',flush=True)


def generate(cfg,out,count,shard=0,shards=1):
    from teacher_engines import VllmOfflineEngine
    m=json.loads((out/'input_manifest.json').read_text())
    all_questions=read(out/'questions.jsonl')[:count]
    path=out/'responses.jsonl' if shards==1 else out/f'responses.part_{shard}of{shards}.jsonl'
    existing=read(path)+(read(out/'responses.jsonl') if shards!=1 else [])
    done=Counter(r['uid'] for r in existing)
    assert all(n==2 for n in done.values()),'partial pair: inspect before resuming'
    questions=[q for q in all_questions if q['uid'] not in done and q['ordinal']%shards==shard]
    if not questions:print('All requested pairs already exist');return
    engine=VllmOfflineEngine(model_path=m['actor'],m=2,temperature=.8,top_p=1.,max_tokens=8192,
        tensor_parallel_size=1,max_model_len=10240,gpu_memory_utilization=.55,enforce_eager=False,
        seed=42,chat_template_kwargs={'enable_thinking':False},
        engine_kwargs={'max_num_seqs':128,'max_num_batched_tokens':12288,'enable_chunked_prefill':False,'enable_prefix_caching':True})
    engine._build()
    engine._sampling_params.detokenize=False
    from scoring.math import score,correctness_label
    started=time.monotonic()
    with path.open('a') as stream:
        for start in range(0,len(questions),4):
            chunk=questions[start:start+4]
            outputs=engine._llm.generate([{'prompt_token_ids':q['prompt_token_ids']} for q in chunk],engine._sampling_params,use_tqdm=False)
            assert len(outputs)==len(chunk)
            for q,output in zip(chunk,outputs):
                assert list(output.prompt_token_ids)==q['prompt_token_ids']
                assert len(output.outputs)==2
                for sample in output.outputs:
                    ids=list(sample.token_ids)
                    text=engine._tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                    result=score(text,q['ground_truth'],{'timeout_seconds':cfg.get('scoring.math.timeout_seconds')})
                    label=correctness_label(result,q['prompt_truncated'])
                    rid=hashlib.sha256(json.dumps([q['uid'],ids],separators=(',',':')).encode()).hexdigest()
                    record={**q,'rollout_index':sample.index,'response_id':rid,'response':text,
                        'token_ids':ids,'finish_reason':sample.finish_reason,'tokens':len(ids),
                        'correctness':label,'scorer':result,'sampling':m['sampling'],'scorer_fingerprint':m['scorer_fingerprint']}
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
            print('Completed',min(start+len(chunk),len(questions)),'/',len(questions),'new questions',flush=True)
    write(out/f'generation_{count}_shard{shard}of{shards}.json',{'new_questions':len(questions),'requested_total_questions':count,
          'elapsed_seconds':time.monotonic()-started,'engine':engine.describe()})


def merge(out,count,shards):
    path=out/'responses.jsonl';original=read(path)
    known={(r['uid'],r['rollout_index']):r for r in original}
    new=[]
    for shard in range(shards):
        for row in read(out/f'responses.part_{shard}of{shards}.jsonl'):
            key=(row['uid'],row['rollout_index'])
            if key in known:assert row==known[key],'completed answer differs'
            else:known[key]=row;new.append(row)
    assert len(known)==count*2,'shards incomplete'
    with path.open('a') as stream:
        for row in sorted(new,key=lambda r:(r['ordinal'],r['rollout_index'])):
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print('Merged',len(known),'responses without replacing completed answers')


def infer(cfg,out):
    import torch
    from transformers import AutoModelForTokenClassification,AutoTokenizer
    from gad.extensions.worker import forward_channels
    from gad.extensions.core.heads import last_response_logits
    from gad.extensions.actor_reward import combine_last
    from types import SimpleNamespace
    manifest=json.loads((out/'input_manifest.json').read_text())
    # Reuse the deployed model padding implementation; reward extraction is current main code.
    source=Path(cfg.require('budget.init_from_warmup_run'))
    backend=json.loads((source/'stage_gad.json').read_text())['plan']['backend']
    sys.path.insert(0,str(backend))
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    model=AutoModelForTokenClassification.from_pretrained(manifest['critic'],torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',local_files_only=True).cuda().eval()
    apply_monkey_patch(model,ulysses_sp_size=1,use_remove_padding=True,use_fused_kernels=False)
    critic=SimpleNamespace(device_name='cuda',use_remove_padding=True,ulysses_sequence_parallel_size=1,critic_module=model)
    existing={r['response_id'] for r in read(out/'components.jsonl')}
    rows=read(out/'responses.jsonl');started=time.monotonic();n=0
    with (out/'components.jsonl').open('a') as stream,torch.inference_mode():
        for row in rows:
            if row['response_id'] in existing:continue
            prompt,response=row['prompt_token_ids'],row['token_ids']
            ids=torch.tensor([prompt+response],device='cuda')
            data={'input_ids':ids,'attention_mask':torch.ones_like(ids),'position_ids':torch.arange(ids.shape[1],device='cuda')[None,:],
                'responses':torch.tensor([response],device='cuda')}
            logits,mask=forward_channels(critic,data,False)
            last=last_response_logits(logits,mask)
            _,parts=combine_last(last,{'actor_reward_coef':0.})
            if not torch.isfinite(parts).all():raise ValueError('nonfinite Critic predictions')
            stream.write(json.dumps({'response_id':row['response_id'],'logits':last.float().cpu()[0].tolist(),
                'parts':parts.cpu()[0].tolist()})+'\n');stream.flush();existing.add(row['response_id']);n+=1
            if n%32==0:print('Scored',n,'new responses',flush=True)
    write(out/f'inference_{len(rows)}.json',{'new_unique_responses':n,'elapsed_seconds':time.monotonic()-started,
        'critic':manifest['critic'],'monkey_patch_backend':backend,'reward_worker_sha256':file_hash(ROOT/'gad/extensions/worker.py'),
        'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30})


def analyze(cfg,out):
    import torch
    from gad.extensions.actor_reward import category,summarize,rewards
    components={r['response_id']:r for r in read(out/'components.jsonl')}
    samples=read(out/'responses.jsonl');rows=[];pairs=defaultdict(list)
    for r in samples:
        lab=r['correctness'];y=[lab['value'] if lab['valid'] else 0.,0.,0.];mask=[float(lab['valid']),0.,0.]
        row={'step':271,'uid':r['uid'],'response_id':r['response_id'],'parts':components[r['response_id']]['parts'],
            'target':y,'mask':mask,'finish_reason':r['finish_reason'],'tokens':r['tokens'],
            'category':category(y[0],mask[0],r['finish_reason'])}
        rows.append(row);pairs[r['uid']].append(row)
    assert all(len(rs)==2 for rs in pairs.values())
    result={'questions':len(pairs),'responses':len(rows),'unique_responses':len(components),
        'category_counts':dict(Counter(r['category'] for r in rows)),
        'pair_categories':dict(Counter(''.join(sorted(r['category'] for r in rs)) for rs in pairs.values())),
        'truncated_correctness':dict(Counter(str(r['target'][0]) if r['mask'][0] else 'unknown' for r in rows if r['category']=='T')),
        'candidates':{}}
    detailed=[]
    for beta in [0.,.05,.1,.3]:
        options={'actor_reward_coef':beta,'actor_reward_weights':{'answer_correctness':1.,'verified_progress':1.,'unresolved_error_severity':1.}}
        summary=summarize(rows,options,prefix='')
        result['candidates'][str(beta)]=summary
        for uid,rs in pairs.items():
            p=torch.tensor([r['parts'] for r in rs]);a,c,f=rewards(p,options)
            detailed.append({'uid':uid,'beta':beta,'response_ids':[r['response_id'] for r in rs],
                'categories':[r['category'] for r in rs],'delta_adv':float(a[0]-a[1]),
                'delta_c':float(p[0,1]-p[1,1]),'delta_p':float(p[0,2]-p[1,2]),'delta_e':float(p[0,3]-p[1,3]),
                'delta_c_reward':float(c[0]-c[1]),'delta_full':float(f[0]-f[1])})
    for i,name in enumerate(['adv','c','p','e']):
        x=torch.tensor([r['parts'][i] for r in rows]);result.setdefault('heads',{})[name]={
            'min':float(x.min()),'max':float(x.max()),'std':float(x.std(unbiased=False)),
            'fraction_below_0.01':float((x<.01).float().mean()),'fraction_above_0.99':float((x>.99).float().mean())}
    n=len(pairs);write(out/f'analysis_{n}.json',result)
    (out/f'pair_deltas_{n}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detailed))
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
    for beta,s in result['candidates'].items():
        print(beta,{k:v for k,v in s.items() if k.startswith('pairs/') and ('all/' in k or '/CW/' in k or '/TW/' in k) and k.endswith(('count','strict_flips','corrected','damaged','toward_T','toward_W'))})


def judge(cfg,out):
    """At most 32 targeted references, with unchanged direct_v1 protocol."""
    import threading
    from transformers import AutoTokenizer
    from extension_config import service_identity
    from rubric.core.contracts import digest,judge_messages
    from gad.extensions.judge_client import JudgeClient,RequestBudget
    samples=read(out/'responses.jsonl');count=len(samples)//2
    deltas=read(out/f'pair_deltas_{count}.jsonl')
    selected=[];seen=set();reasons={}
    # Prioritize small-beta changes, TW flips, and both members of each pair.
    changed=[r for r in deltas if r['beta']>0 and
             (r['delta_adv']*r['delta_full']<0 or r['delta_c_reward']*r['delta_full']<0)]
    changed.sort(key=lambda r:(r['beta'],set(r['categories'])!={'T','W'},abs(r['delta_full'])))
    for row in changed:
        if row['uid'] not in seen and len(selected)<12:
            selected.append(row);seen.add(row['uid']);reasons[row['uid']]='targeted_flip'
    control=[r for r in deltas if r['beta']==.05 and r['uid'] not in seen and r['delta_adv']*r['delta_full']>0]
    control.sort(key=lambda r:(set(r['categories'])!={'T','W'},r['uid']))
    for row in control[:4]:
        selected.append(row);seen.add(row['uid']);reasons[row['uid']]='nonflip_control'
    ids={rid for r in selected for rid in r['response_ids']}
    assert len(ids)<=32
    examples=[r for r in samples if r['response_id'] in ids]
    unique={r['response_id']:r for r in examples}
    frozen=out/'judge_selection.json'
    if frozen.exists():assert json.loads(frozen.read_text())==selected
    else:write(frozen,selected)
    options={k[13:]:v for k,v in cfg.data.items() if k.startswith('rubric.judge.')}
    options.update(rubric_version='math_process_v2',process_protocol='direct_v1',seed=42,
                   request_budget=32,budget_state_path=None)
    budget=RequestBudget(32,out/'judge_reference_budget.json',7200)
    lock=threading.Lock()
    class RecordingClient(JudgeClient):
        def annotate_one(self,*args,**kwargs):
            rid,result=super().annotate_one(*args,**kwargs)
            with lock:
                with (out/'judge_references.jsonl').open('a') as stream:
                    stream.write(json.dumps({'response_id':rid,'result':result},ensure_ascii=False)+'\n')
            return rid,result
    client=RecordingClient(options,digest(service_identity(cfg)),request_budget=budget)
    done={r['response_id'] for r in read(out/'judge_references.jsonl')}
    actor=json.loads((out/'input_manifest.json').read_text())['actor']
    tokenizer=AutoTokenizer.from_pretrained(actor,local_files_only=True)
    spec=Path(cfg.require('rubric.spec')).read_text();prompt=Path(cfg.require('rubric.prompt')).read_text()
    requests={rid:judge_messages(tokenizer.decode(r['prompt_token_ids'],skip_special_tokens=False,clean_up_tokenization_spaces=False),
        r['response'],r['ground_truth'],spec,prompt,'math_process_v2',
        {'prompt_truncated':r['prompt_truncated'],'source_truncated':r['finish_reason']=='length',
         'training_token_truncated':False,'judge_context_overflow':False},process_protocol='direct_v1')
        for rid,r in unique.items() if rid not in done}
    write(out/'judge_reference_requests.json',requests)
    started=time.monotonic()
    try:
        client.annotate(requests)
    finally:
        write(out/'judge_reference_ledger.json',client.api_ledger)
        results=read(out/'judge_references.jsonl')
        write(out/'judge_reference_summary.json',{'selected_unique_responses':len(ids),'selected_pairs':len(selected),
            'reference_results':len(results),'parsed':sum('parsed' in r['result'] for r in results),
            'selection_reasons':reasons,'elapsed_seconds':time.monotonic()-started,
            'prompt_tokens':sum(r['result'].get('usage',{}).get('prompt_tokens',0) for r in results),
            'completion_tokens':sum(r['result'].get('usage',{}).get('completion_tokens',0) for r in results),
            'human_verified':False,'independent_process_truth':False,
            'scope':'Targeted flips and controls only. Does not estimate population accuracy; no prompt/model tuning.'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','generate','merge','infer','analyze','judge'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--questions',type=int,choices=[128,256,512],default=128)
    p.add_argument('--shard-index',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    a=p.parse_args();cfg=load_config(a.config);a.output.mkdir(parents=True,exist_ok=True)
    if a.action=='prepare':prepare(cfg,a.output)
    elif a.action=='generate':generate(cfg,a.output,a.questions,a.shard_index,a.shards)
    elif a.action=='merge':merge(a.output,a.questions,a.shards)
    elif a.action=='infer':infer(cfg,a.output)
    elif a.action=='judge':judge(cfg,a.output)
    else:analyze(cfg,a.output)


if __name__=='__main__':main()
