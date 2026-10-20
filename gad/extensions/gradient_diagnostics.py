"""Opt-in periodic gradients on a frozen, bounded batch; never optimizer updates.

Separate backward passes support FSDP(use_orig_params=False). Flat-parameter
metadata is checked explicitly so score-head gradients cannot masquerade as
shared-backbone gradients. The pinned Torch implementation must pass GPU smoke.
"""
from pathlib import Path
from collections import Counter
import torch
from .core.heads import last_response_logits
from .core.losses import pair_loss, weighted_contribution, dimension_weights
from .common import digest, atomic_json


def shared_gradients(model):
    chunks=[]
    for name,p in model.named_parameters():
        if p.grad is None:continue
        if hasattr(p,'_fqns'):
            infos=getattr(p,'_shard_param_infos',None)
            if infos is None or len(infos)!=len(p._fqns):raise ValueError('unsupported FSDP flat-parameter diagnostic metadata')
            grad=p.grad.detach().reshape(-1)
            for fqn,info in zip(p._fqns,infos):
                if 'score' in fqn.split('.') or not info.in_shard:continue
                chunks.append(grad[info.offset_in_shard:info.offset_in_shard+info.numel_in_shard].float().cpu().clone())
        elif 'score' not in name.split('.'):
            chunks.append(p.grad.detach().float().reshape(-1).cpu().clone())
    return chunks


def geometry(a,b,device,distributed=False):
    if len(a)!=len(b) or any(x.shape!=y.shape for x,y in zip(a,b)):raise ValueError('diagnostic gradient structure changed')
    # Float64 CPU reductions; only three scalars are copied to the collective.
    sums=torch.tensor([sum((x.double()*x.double()).sum().item() for x in a),sum((y.double()*y.double()).sum().item() for y in b),sum((x.double()*y.double()).sum().item() for x,y in zip(a,b))],dtype=torch.float64,device=device)
    if distributed:torch.distributed.all_reduce(sums)
    an,bn=sums[:2].sqrt();denom=an*bn
    return {'bt_norm':an.item(),'aux_norm':bn.item(),'aux_bt_norm_ratio':(bn/an).item() if an>0 else 0.,'cosine':(sums[2]/denom).item() if denom>0 else 0.,'cosine_defined':int(denom>0)}


def freeze(critic, mini, directory, max_pairs, device):
    distributed=torch.distributed.is_initialized()
    rank=torch.distributed.get_rank() if distributed else 0
    dp=torch.distributed.get_world_size() if distributed else 1
    path=Path(directory)/f'rank_{rank}.pt'
    if distributed:
        present=torch.tensor(int(path.exists()),device=device)
        torch.distributed.all_reduce(present)
        if present.item() not in (0,dp):raise ValueError('partial fixed diagnostic batch across ranks; restore complete checkpoint snapshot')
    if path.exists():data=torch.load(path,map_location='cpu',weights_only=True)
    else:
        data={k:v[:max_pairs].detach().cpu().clone() for k,v in mini.items()}
        # Recompute inverse multiplicities over the diagnostic subset, both
        # roles and ranks; full-training-minibatch counts must not be reused.
        ids=[]
        for i in range(len(data['responses'])):
            for role in ('teacher','student'):
                prefix='teacher_' if role=='teacher' else ''
                ids.append(digest({'tokens':data[prefix+'input_ids'][i][data[prefix+'attention_mask'][i].bool()].tolist()}))
        gathered=[None]*dp
        if distributed:torch.distributed.all_gather_object(gathered,ids)
        else:gathered=[ids]
        multiplicity=Counter(x for rows in gathered for x in rows)
        for j,role in enumerate(('teacher','student')):
            data[role+'_aux_weight']=torch.tensor([1/multiplicity[ids[2*i+j]] for i in range(len(data['responses']))])
        counts=sum((data[role+'_aux_weight'][:,None]*data[role+'_rubric_mask']).sum(0) for role in ('teacher','student')).to(device)
        if distributed:torch.distributed.all_reduce(counts)
        data['rubric_global_counts']=counts.cpu().expand(len(data['responses']),-1).clone()
        path.parent.mkdir(parents=True,exist_ok=True)
        # One frozen file per rank and stage, retained for same-stage resume.
        torch.save(data,path)
        atomic_json(path.with_suffix('.json'),{'rank':rank,'world_size':dp,'pairs':len(data['responses']),'response_ids':ids,'source':'first selected labeled critic minibatch; frozen, not a representative evaluation set'})
    return data


def diagnose(critic, mini, forward, device):
    cfg=critic.config.rubric
    directory=cfg.get('gradient_diagnostics_dir')
    if not directory:raise ValueError('gradient diagnostics require an isolated persistent directory')
    cached=getattr(critic,'_rubric_fixed_diagnostic',None)
    if cached is None:
        cached=freeze(critic,mini,directory,cfg.get('gradient_diagnostics_max_pairs',4),device)
        critic._rubric_fixed_diagnostic=cached
    if len(cached['responses'])>cfg.get('gradient_diagnostics_max_pairs',4):raise ValueError('fixed diagnostic batch exceeds configured pair budget')
    data={k:v.to(device) for k,v in cached.items()}
    counts=data['rubric_global_counts'][0]
    dp=torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    mode,tau=cfg.get('loss_mode','legacy_active'),cfg.get('count_gate_tau',4)
    training=critic.critic_module.training
    # Keep training mode so HF gradient checkpointing remains enabled. Replay
    # identical RNG states for every diagnostic objective and restore the caller
    # state; diagnostics must neither disable checkpointing nor advance dropout.
    cpu_rng=torch.get_rng_state()
    model_device=next(critic.critic_module.parameters()).device
    cuda_rng=torch.cuda.get_rng_state(model_device) if model_device.type=='cuda' else None
    gradients={}
    try:
        for objective in ('bt','aux','answer_correctness','verified_progress','unresolved_error_severity'):
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,model_device)
            critic.critic_optimizer.zero_grad()
            # Keep the original microbatch memory bound during diagnostics.
            total=len(data['responses'])
            for start in range(0,total,critic.config.ppo_micro_batch_size_per_gpu):
                stop=min(total,start+critic.config.ppo_micro_batch_size_per_gpu)
                chunk={k:v[start:stop] for k,v in data.items()}
                s,sm=forward(critic,chunk,False);t,tm=forward(critic,chunk,True)
                student,teacher=last_response_logits(s,sm).float(),last_response_logits(t,tm).float()
                if objective=='bt':loss=pair_loss(teacher[:,0],student[:,0]).mean()*(stop-start)/total
                else:
                    numerator=student.sum(0)[1:]*0
                    for role,logits in (('student',student),('teacher',teacher)):
                        _,local=weighted_contribution(logits[:,1:],chunk[role+'_rubric_targets'],chunk[role+'_rubric_mask'],chunk[role+'_aux_weight'],counts,dp,mode,tau)
                        numerator=numerator+local*dp
                    coeff=dimension_weights(counts,mode,tau)[2]
                    if objective!='aux':coeff=coeff*torch.tensor([objective==d for d in ('answer_correctness','verified_progress','unresolved_error_severity')],device=device)
                    loss=(numerator*coeff).sum()*cfg.loss_coef
                loss.backward()
            gradients[objective]=shared_gradients(critic.critic_module)
        metrics={}
        for key in ('aux','answer_correctness','verified_progress','unresolved_error_severity'):
            values=geometry(gradients['bt'],gradients[key],device,torch.distributed.is_initialized())
            metrics.update({f'rubric/diagnostic/{key}/{k}':v for k,v in values.items()})
        return metrics
    finally:
        critic.critic_optimizer.zero_grad()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,model_device)
        critic.critic_module.train(training)
