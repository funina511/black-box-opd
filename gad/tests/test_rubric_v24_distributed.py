"""Two real CPU Gloo ranks exercise the production joint update and DP scaling."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=torch.nn.Linear(7,3)
        self.score=torch.nn.Linear(3,4)
    def forward(self,x):return self.score(self.backbone(x).tanh())


def run_rank(rank, rendezvous, output):
    from omegaconf import OmegaConf
    from tensordict import TensorDict
    from gad.extensions.critic_batch import auxiliary_weights
    from gad.extensions.worker import update_joint
    import gad.extensions.worker as worker
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(43)
        names=['T0','T1','S0','S1','S2','S3']
        features={name:torch.randn(7) for name in names}
        pairs=[{'teacher':{'response_id':'T'+str(i%2)},'student':{'response_id':'T0' if i%4==0 else 'S'+str(i//2)}} for i in range(8)]
        target_by_id={name:torch.tensor([float(i%2),.5,float(i%2)]) for i,name in enumerate(names)}
        mask_by_id={name:torch.tensor([1.,float(name=='T0'),float(name in ('T1','S3'))]) for name in names}
        x={r:torch.stack([features[p[r]['response_id']] for p in pairs])[:,None,:] for r in ('teacher','student')}
        targets={r:torch.stack([target_by_id[p[r]['response_id']] for p in pairs]) for r in x}
        initial=Net().state_dict()
        model=torch.nn.parallel.DistributedDataParallel(Net())
        results=[]
        for mode in ('legacy_active','fixed_dimension','fixed_dimension_gated'):
          for micro in (1,2):
           for mask_case in ('normal','correctness_only','zero'):
            masks={r:torch.stack([mask_by_id[p[r]['response_id']] for p in pairs]) for r in x}
            if mask_case=='correctness_only':
                for v in masks.values():v[:,1:]=0
            if mask_case=='zero':
                for v in masks.values():v.zero_()
            weights,counts=auxiliary_weights(pairs,targets,masks,8,2)
            local=slice(rank*4,(rank+1)*4)
            tensors={'input_ids':x['student'][local],'teacher_input_ids':x['teacher'][local],
              'responses':torch.ones(4,1,dtype=torch.long),'teacher_response':torch.ones(4,1,dtype=torch.long),
              'attention_mask':torch.ones(4,1),'teacher_attention_mask':torch.ones(4,1),
              'position_ids':torch.zeros(4,1),'teacher_position_ids':torch.zeros(4,1),
              'rubric_global_counts':counts[local]}
            for role in x:
                tensors[role+'_rubric_targets']=targets[role][local]
                tensors[role+'_rubric_mask']=masks[role][local]
                tensors[role+'_aux_weight']=weights[role][local]
            batch=TensorDict(tensors,batch_size=[4])
            proto=SimpleNamespace(select=lambda **kwargs:SimpleNamespace(batch=batch))
            model.module.load_state_dict(initial)
            optimizer=torch.optim.SGD(model.parameters(),lr=.02)
            config=OmegaConf.create({'use_dynamic_bsz':False,'ppo_epochs':1,'ppo_mini_batch_size':4,
                'ppo_micro_batch_size_per_gpu':micro,'rubric':{'loss_coef':.1,'loss_mode':mode,'count_gate_tau':4,
                'head_layout':['adv','answer_correctness','verified_progress','unresolved_error_severity'],
                'gradient_diagnostics_interval':0}})
            critic=SimpleNamespace(config=config,critic_module=model,critic_optimizer=optimizer,ulysses_sequence_parallel_size=1)
            def step():
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1e9)
                optimizer.step()
                return norm
            critic._optimizer_step=step
            def forward(c,data,teacher):
                return c.critic_module(data['teacher_input_ids' if teacher else 'input_ids']),data['teacher_attention_mask' if teacher else 'attention_mask']
            with patch.dict(sys.modules,{'verl.utils.device':SimpleNamespace(get_device_id=lambda:torch.device('cpu'))}),patch.object(worker,'forward_channels',forward):
                metrics=update_joint(critic,proto)
            # Independent single-batch formula, unique responses across both
            # roles. Do not call the production auxiliary loss for the reference.
            reference=Net();reference.load_state_dict(initial)
            teacher,student=reference(x['teacher']).squeeze(1),reference(x['student']).squeeze(1)
            bt=-F.logsigmoid(teacher[:,0]-student[:,0]).mean()
            ids={p[role]['response_id']:(role,i) for i,p in enumerate(pairs) for role in x}
            unique_logits=torch.stack([(teacher if role=='teacher' else student)[i,1:] for role,i in ids.values()])
            y=torch.stack([target_by_id[rid] for rid in ids])
            m=torch.stack([masks[role][i] for role,i in ids.values()])
            n=m.sum(0);active=n>0
            numerator=(F.binary_cross_entropy_with_logits(unique_logits,y,reduction='none')*m).sum(0)
            gate=(n/4).clamp(max=1) if mode=='fixed_dimension_gated' else active
            denominator=active.sum().clamp_min(1) if mode=='legacy_active' else 3
            loss=bt+.1*(numerator/n.clamp_min(1)*gate).sum()/denominator
            loss.backward()
            max_error=0.
            for (name,actual),(ref_name,param) in zip(model.module.named_parameters(),reference.named_parameters()):
                assert name==ref_name
                expected=param.detach()-.02*param.grad
                torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
                max_error=max(max_error,(actual-expected).abs().max().item())
            total=torch.tensor(metrics['critic/loss_total'][0]);dist.all_reduce(total);total/=2
            torch.testing.assert_close(total,loss.detach(),atol=1e-6,rtol=1e-6)
            results.append({'mode':mode,'microbatch':micro,'mask_case':mask_case,'global_unique_counts':counts[0].tolist(),
                            'parameter_max_abs_error':max_error,'loss_abs_error':abs(total.item()-loss.item())})
        gathered=[None]*2;dist.all_gather_object(gathered,results)
        if rank==0:Path(output).write_text(json.dumps({'world_size':2,'backend':'gloo','cases_per_rank':18,'rank_results':gathered,'scope':'production update_joint, CPU DDP small neural net; not Qwen/FSDP GPU training'},indent=2))
    finally:dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_joint_update_on_two_real_ranks(self):
        with tempfile.TemporaryDirectory(prefix='rubric-v24-ddp-') as d:
            output=str(Path(d)/'results.json')
            mp.spawn(run_rank,args=(str(Path(d)/'store'),output),nprocs=2,join=True)
            result=json.loads(Path(output).read_text())
            self.assertEqual(18,result['cases_per_rank'])
            if os.environ.get('V24_DDP_REPORT'):Path(os.environ['V24_DDP_REPORT']).write_text(json.dumps(result,indent=2))

if __name__=='__main__':unittest.main()
