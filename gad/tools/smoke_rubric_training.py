#!/usr/bin/env python3
"""Bounded Qwen worker probe on saved trajectories (no rollout, no Judge calls).

One Teacher/Student pair; production joint critic + PPO Actor; each stage updates,
checks save/restore, then updates again. This does not test Ray/vLLM scheduling.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',required=True);parser.add_argument('--samples',required=True)
    parser.add_argument('--paired-samples',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--wall-seconds',type=int,default=1800)
    args=parser.parse_args()
    if not 0<args.wall_seconds<=1800:raise ValueError('smoke is bounded to 30 minutes')
    import signal
    def alarm(*_):raise TimeoutError('real training 30-minute window exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.wall_seconds)
    started=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(args.backend).resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer,AutoModelForTokenClassification,AutoModelForCausalLM
    from verl import DataProto
    from verl.experimental.gad_extensions.worker import update_joint,reward_forward
    from verl.experimental.gad_extensions.critic_batch import auxiliary_weights
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from scoring.math import score,correctness_label
    from rubric.core.contracts import file_hash
    from rubric.core.heads import expand_head
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    rows=[json.loads(l) for l in Path(args.samples).read_text().splitlines()]
    dev=[json.loads(l) for l in Path(args.paired_samples).read_text().splitlines()]
    # Never teach a confirmed wrong full-progress assertion from the reflection row.
    candidates=[r for r in rows if r['source_role']=='student' and r['id']!='holdout_0544c0284caefe14fadb'
        and r.get('result',{}).get('parsed',{}).get('verified_progress',{}).get('valid')]
    if not candidates:raise ValueError('no reviewed, usable Student process target in fixed cohort; training admission unmet')
    student=candidates[0]
    teacher=next(r for r in dev if r['source_role']=='teacher' and r['prompt_hash']==student['prompt_hash'])
    tok=AutoTokenizer.from_pretrained(ROOT.parent/'model/Qwen/Qwen3-0.6B',local_files_only=True)
    prompt=tok.apply_chat_template(student['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if len(prompt)>2048:raise ValueError('saved prompt exceeds formal limit')
    tensors={};role_rows={'student':student,'teacher':teacher}
    for role,row in role_rows.items():
        ids=tok.encode(row['response'],add_special_tokens=False)
        if len(ids)>8192:raise ValueError('saved response exceeds formal limit; no silent truncation')
        # One original saved response per role, tokenized losslessly.
        prefix='teacher_' if role=='teacher' else ''
        tensors['teacher_response' if role=='teacher' else 'responses']=torch.tensor([ids])
        tensors[prefix+'input_ids']=torch.tensor([prompt+ids]);tensors[prefix+'attention_mask']=torch.ones(1,len(prompt)+len(ids),dtype=torch.long)
        tensors[prefix+'position_ids']=torch.arange(len(prompt)+len(ids))[None,:]
        label=correctness_label(score(row['response'],row['ground_truth']))
        process=student['result']['parsed'] if role=='student' else {}
        targets=[label['value'] if label['valid'] else 0];mask=[int(label['valid'])]
        for dim in ('verified_progress','unresolved_error_severity'):
            item=process.get(dim,{});targets.append(item.get('value') if item.get('valid') else 0);mask.append(int(item.get('valid',False)))
        tensors[role+'_rubric_targets']=torch.tensor([targets],dtype=torch.float32)
        tensors[role+'_rubric_mask']=torch.tensor([mask],dtype=torch.float32)
    pairs=[{r:{'response_id':role_rows[r]['response_id']} for r in role_rows}]
    weights,counts=auxiliary_weights(pairs,{r:tensors[r+'_rubric_targets'] for r in role_rows},
        {r:tensors[r+'_rubric_mask'] for r in role_rows},1,1)
    for role in role_rows:tensors[role+'_aux_weight']=weights[role]
    tensors['rubric_global_counts']=counts
    proto=DataProto.from_dict(tensors=tensors)
    provenance={'student_id':student['id'],'teacher_id':teacher['id'],'same_prompt_hash':student['prompt_hash'],
        'student_parsed':student['result']['parsed'],'teacher_process_policy':'masked: no new-protocol review for paired Teacher',
        'human_verified':False,'source_samples_sha256':file_hash(args.samples),'source_pair_sha256':file_hash(args.paired_samples),
        'token_lengths':{r:len(tensors['teacher_response' if r=='teacher' else 'responses'][0]) for r in role_rows},
        'targets':{r:tensors[r+'_rubric_targets'].tolist() for r in role_rows},'masks':{r:tensors[r+'_rubric_mask'].tolist() for r in role_rows},'global_counts':counts.tolist()}
    (out/'label_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    torch.cuda.set_device(0)
    dist.init_process_group('nccl',init_method='file://'+str((out/'rendezvous').resolve()),world_size=1,rank=0)
    report={'status':'running','scope':'real Qwen0.6B; FSDP single-rank Critic, original PPO Actor on second GPU; fixed saved trajectories, no live rollout/Ray/pipeline test',
        'maximum_pairs':1,'maximum_critic_updates':4,'maximum_actor_updates':2,'judge_calls':0,'updates':[]}
    try:
        critic_model=AutoModelForTokenClassification.from_pretrained(ROOT/'runs/_initialization/qwen3_0.6b_math_v2_adv_seed42',torch_dtype=torch.float32,attn_implementation='flash_attention_2',local_files_only=True)
        expand_head(critic_model,42,file_hash(ROOT/'rubric/specs/math_process_v27.yaml'),'math_process_v2')
        critic_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        critic_model=FSDP(critic_model.to('cuda:0'),device_id=0,use_orig_params=False,mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16))
        copt=torch.optim.AdamW(critic_model.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01)
        ccfg=OmegaConf.create({'use_dynamic_bsz':False,'ppo_epochs':1,'ppo_mini_batch_size':1,'ppo_micro_batch_size_per_gpu':1,
            'rubric':{'loss_coef':.1,'loss_mode':'fixed_dimension_gated','count_gate_tau':4,'version':'math_process_v2',
                'head_layout':['adv','answer_correctness','verified_progress','unresolved_error_severity'],
                'gradient_diagnostics_interval':100,'gradient_diagnostics_max_pairs':1,'gradient_diagnostics_dir':str(out/'gradient_diagnostics')}})
        critic=SimpleNamespace(config=ccfg,critic_module=critic_model,critic_optimizer=copt,ulysses_sequence_parallel_size=1,use_remove_padding=False,device_name='cuda')
        def cstep():
            norm=critic_model.clip_grad_norm_(.2)
            if torch.isfinite(norm):copt.step()
            return norm
        critic._optimizer_step=cstep
        torch.cuda.set_device(1)
        actor_model=AutoModelForCausalLM.from_pretrained(ROOT.parent/'model/Qwen/Qwen3-0.6B',torch_dtype=torch.float32,attn_implementation='flash_attention_2',local_files_only=True).to('cuda:1')
        actor_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        acfg=OmegaConf.load(Path(args.backend)/'verl/trainer/config/ppo_trainer.yaml').actor_rollout_ref.actor
        OmegaConf.set_struct(acfg,False)
        for k,v in {'use_remove_padding':False,'use_fused_kernels':False,'use_torch_compile':False,'ppo_mini_batch_size':1,'ppo_micro_batch_size_per_gpu':1,
            'ppo_epochs':1,'use_dynamic_bsz':False,'entropy_coeff':0.,'use_kl_loss':True,'kl_loss_coef':.001,'kl_loss_type':'low_var_kl',
            'grad_clip':.2,'ulysses_sequence_parallel_size':1,'entropy_from_logits_with_chunking':False}.items():acfg[k]=v
        aopt=torch.optim.AdamW(actor_model.parameters(),lr=1e-6,betas=(.9,.999),weight_decay=.01)
        actor=DataParallelPPOActor(acfg,actor_model,aopt)
        initial_actor=actor_model.model.layers[0].self_attn.q_proj.weight.detach().cpu().clone()
        reference=None
        for stage in ('warmup','gad'):
            for update in (1,2):
                mark=time.monotonic();torch.cuda.set_device(0)
                with torch.no_grad():reward=reward_forward(critic,{k:v.to('cuda:0') for k,v in tensors.items()},False).float().cpu()
                with FSDP.summon_full_params(critic_model,writeback=False):
                    before=critic_model.module.model.layers[0].self_attn.q_proj.weight.detach().cpu().clone()
                cm=update_joint(critic,proto)
                with FSDP.summon_full_params(critic_model,writeback=False):
                    delta=(critic_model.module.model.layers[0].self_attn.q_proj.weight.detach().cpu()-before).abs().max().item()
                if not delta:raise AssertionError('shared critic backbone did not update')
                item={'stage':stage,'logical_update':update,'critic_metrics':cm,'shared_backbone_max_delta':delta,'adv_reward':reward.sum().item(),'actor_skipped':stage=='warmup'}
                if stage=='gad':
                    torch.cuda.set_device(1)
                    policy=DataProto.from_dict(tensors={k:v for k,v in tensors.items() if k in ('input_ids','responses','attention_mask','position_ids')},meta_info={'temperature':.8,'micro_batch_size':1,'use_dynamic_bsz':False})
                    policy.to('cuda:1')
                    old,_=actor.compute_log_prob(policy)
                    if reference is None:reference=old.detach().clone()
                    advantages,_=compute_grpo_outcome_advantage(reward,torch.ones_like(reward),np.array(['fixed_pair']),norm_adv_by_std_in_grpo=True)
                    policy.batch['old_log_probs']=old.detach().cpu();policy.batch['ref_log_prob']=reference.cpu();policy.batch['advantages']=advantages.cpu()
                    before_actor=actor_model.model.layers[0].self_attn.q_proj.weight.detach().cpu().clone()
                    am=actor.update_policy(policy)
                    actor_delta=(actor_model.model.layers[0].self_attn.q_proj.weight.detach().cpu()-before_actor).abs().max().item()
                    if not actor_delta:raise AssertionError('Actor did not update from original adv reward')
                    item.update(actor_metrics=am,actor_backbone_max_delta=actor_delta,advantage_abs_sum=advantages.abs().sum().item(),reference_policy='frozen initial Actor logits on this saved Student response')
                report['updates'].append(item);item['seconds']=time.monotonic()-mark
                if update==1:
                    torch.cuda.set_device(0);target=out/f'{stage}_resume.pt'
                    saved={'critic':critic_model.state_dict(),'critic_optimizer':copt.state_dict(),'actor':actor_model.state_dict(),'actor_optimizer':aopt.state_dict(),
                        'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'stage':stage,'step':update}
                    torch.save(saved,target);del saved
                    checkpoint=torch.load(target,map_location='cpu',weights_only=False)
                    critic_model.load_state_dict(checkpoint['critic']);copt.load_state_dict(checkpoint['critic_optimizer'])
                    actor_model.load_state_dict(checkpoint['actor']);aopt.load_state_dict(checkpoint['actor_optimizer'])
                    for state in aopt.state.values():
                        for key,value in state.items():
                            if torch.is_tensor(value) and key!='step':state[key]=value.to('cuda:1')
                    torch.set_rng_state(checkpoint['cpu_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng']);np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
                    item['resume_verified']={'critic_optimizer_step':max(float(v['step']) for v in copt.state.values()),'actor_optimizer_step':max((float(v['step']) for v in aopt.state.values()),default=0),'rng':True,'checkpoint':str(target),'sha256':file_hash(target)}
                    del checkpoint;gc.collect();torch.cuda.empty_cache()
                (out/'report.json').write_text(json.dumps(report,indent=2))
        diagnostic=report['updates'][0]['critic_metrics']
        if diagnostic['rubric/diagnostic/verified_progress/aux_norm'][0]<=0:raise AssertionError('Student Progress has zero shared-backbone gradient')
        report.update(status='passed',critic_updates=4,actor_updates=2,warmup_actor_updates=0,student_progress_shared_gradient_nonzero=True)
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,peak_gpu_mib={str(i):torch.cuda.max_memory_allocated(i)/1024**2 for i in range(torch.cuda.device_count())})
        (out/'report.json').write_text(json.dumps(report,indent=2));dist.destroy_process_group();signal.alarm(0)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
