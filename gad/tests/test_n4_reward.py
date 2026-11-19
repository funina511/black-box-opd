"""n=4 driver grouping, precision controls, monitoring and selection contracts."""
import ast
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'gad/tools'), str(ROOT/'gad/tests')]
from gad.extensions.actor_reward import (DIMS, reward_scores, groupnorm_scores, group_advantages,
    driver_reward, summarize, constraint_statistics, monitor_records, student_scorer_labels)
from gad.extensions import worker
from config import validate_config
from extension_config import hydra_overrides, N4_DEFAULTS
from select_actor_reward import analyze, question_half, read_shadow
from cpu_fixtures import compose_cpu_config, load_cpu_config as load_config
import pipeline

OPTIONS = {'enabled': True, 'reward_source': 'adv_plus_rubric_groupnorm',
           'actor_reward_group_weight': 1., 'actor_reward_rubric_std_floor': .05,
           'actor_reward_weights': dict(zip(DIMS, (1., 1., 1.)))}


def rows_for(parts, cats=None, step=272):
    cats = cats or ['C','W','T','U']
    return [{'step': step, 'uid': 'u'+str(i//4), 'question_id': 'q'+str(i//4), 'parts': p,
             'target': [1. if cats[i%4]=='C' else 0.,0.,0.],
             'mask': [0. if cats[i%4]=='U' else 1.,0.,0.],
             'category': cats[i%4], 'finish_reason': 'length' if cats[i%4]=='T' else 'stop', 'tokens': 3}
            for i,p in enumerate(parts)]


class N4Tests(unittest.TestCase):
    def test_formula_n2_n3_n4(self):
        for n in (2,3,4):
            adv = torch.arange(n).float()*.1
            rub = torch.arange(n).float().flip(0)*.2
            expected = (adv-adv.mean())/(adv.std()+1e-6) + .5*(rub-rub.mean())/rub.std().clamp_min(.05)
            actual = groupnorm_scores(adv,rub,['a']*n,.5,.05)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_passthrough_and_zero_floor(self):
        adv = torch.tensor([0.,1e-8,2e-8,3e-8,1.,1.,1.,1.])
        uid = ['a']*4+['b']*4
        for w,r in [(0.,torch.arange(8).float()),(1.,torch.ones(8))]:
            actual = groupnorm_scores(adv,r,uid,w,0.)
            self.assertTrue(torch.equal(adv,actual))
            self.assertTrue(torch.equal(group_advantages(adv,uid),group_advantages(actual,uid)))
            self.assertTrue(torch.isfinite(actual).all())
        # Passthrough is per group, not per batch.
        actual=groupnorm_scores(adv,torch.tensor([.5]*4+[.1,.2,.3,.4]),uid,1.,0.)
        self.assertTrue(torch.equal(actual[:4],adv[:4])); self.assertFalse(torch.equal(actual[4:],adv[4:]))

    def test_floor_and_weak_adv_boundary(self):
        adv=torch.tensor([0.,1.,2.,3.]);rub=torch.tensor([.500,.501,.502,.503]);uid=['a']*4
        s=groupnorm_scores(adv,rub,uid,1.,.05)
        torch.testing.assert_close(s-group_advantages(adv,uid),(rub-rub.mean())/.05)
        rub=torch.tensor([.500000,.500001,.500002,.500003])
        s=groupnorm_scores(torch.ones(4),rub,uid,1.,.05)
        a=group_advantages(s,uid)
        torch.testing.assert_close(a,torch.tensor([-1.12,-.37,.37,1.12]),atol=.025,rtol=0)
        rows=rows_for([[1.,float(r),0.,0.] for r in rub])
        stats=summarize(rows,OPTIONS)
        self.assertEqual(1,stats['actor_reward/groups/adv_weak_count'])
        self.assertGreater(stats['actor_reward/groups/adv_weak_advantage_change_mean'],.5)

    def test_na_and_group_diagnostics(self):
        rows=rows_for([[1.,.5,.5,.5]]*4,cats=['T']*4)
        result=constraint_statistics(rows,OPTIONS)
        self.assertIsNone(result['correction_precision']);self.assertIsNone(result['tw_toward_T_share'])
        self.assertIsNone(result['cw_gap_increase']);self.assertFalse(result['eligible'])
        stats=summarize(rows,OPTIONS)
        self.assertEqual(1.,stats['actor_reward/groups/all_truncated_rate'])
        self.assertEqual(1.,stats['actor_reward/groups/zero_advantage_rate'])
        self.assertEqual(1,stats['actor_reward/groups/passthrough_count'])
        rows=rows_for([[1.,.5,.5,.5]]*4)
        stats=summarize(rows,OPTIONS)
        self.assertEqual(1,stats['actor_reward/correctness_unknown'])
        self.assertEqual(1.,stats['actor_reward/groups/mixed_correctness_rate'])
        self.assertEqual(1,stats['actor_reward/groups/cw_gap_defined'])

    def test_three_configs_and_historical_defaults(self):
        from hydra.core.override_parser.overrides_parser import OverridesParser
        parser=OverridesParser.create()
        for name in ('baseline','rubric_naive','rubric_actor'):
            cfg=load_config(ROOT/f'configs/launch_{name}_n4_06b_17k.yaml')
            self.assertEqual([],validate_config(cfg));self.assertEqual(4,cfg.get('rollout.n'))
            opts={p.key_or_group:p.value() for p in parser.parse_overrides(hydra_overrides(cfg,'gad',Path('/tmp/n4'),271))}
            self.assertTrue(opts['trainer.gad_extensions']['actor_reward_fp32'])
            self.assertTrue(opts['trainer.gad_extensions']['enabled'])
            if name=='baseline':self.assertEqual({'enabled':False},opts['critic.rubric'])
            else:self.assertEqual(32,opts['trainer.rubric']['process_sampling']['max_unique_per_global_minibatch'])
            self.assertFalse(cfg.get('models.actor_param_offload'))
            self.assertTrue(cfg.get('models.critic_gradient_checkpointing'))
            self.assertEqual(2,cfg.get('optim.critic_micro_batch_size_per_gpu'))
        # Newly added no-op keys must be absent from both old adv and additive identity/config.
        for filename in ('configs/launch_rubric_actor_06b_17k.yaml',):
            cfg=load_config(ROOT/filename)
            opts={p.key_or_group:p.value() for p in parser.parse_overrides(hydra_overrides(cfg,'gad',Path('/tmp/n4'),271))}
            for key in N4_DEFAULTS:
                self.assertNotIn(key[7:] if key.startswith('rubric.') else key,opts['trainer.rubric'])
            self.assertNotIn('actor_reward_fp32',opts['trainer.gad_extensions'])
        bad=load_config(ROOT/'configs/launch_rubric_actor_n4_06b_17k.yaml');bad.data['actor_reward.fp32']=False
        self.assertTrue(validate_config(bad))

    def test_unconfirmed_formal_actor_launch_is_blocked_before_writes(self):
        cfg=load_config(ROOT/'configs/launch_rubric_actor_n4_06b_17k.yaml')
        with tempfile.TemporaryDirectory() as d:
            cfg.data['run.output_root']=d
            with patch.object(pipeline,'select_config',return_value=cfg),patch.object(sys,'argv',
                    ['pipeline.py','--stages','gad','--skip-data-check','--run-id','must_not_start']):
                with self.assertRaisesRegex(Exception,'await offline selection'):
                    pipeline.main()
            self.assertEqual([],list(Path(d).iterdir()))

    def test_metadata_migration_checks_content_order_and_cursor(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        cfg=compose_cpu_config();source=copy.deepcopy(cfg)
        rows=[{'content':[{'role':'user','content':'q'+str(i)}],'teacher_response':'Answer: 2',
               'reward_model':{'ground_truth':'2','style':'rule'},'data_source':'math','ability':'MATH',
               'extra_info':{'index':str(i)}} for i in range(2)]
        migrated=copy.deepcopy(rows)
        for r in migrated:r['extra_info']['new_metadata']='checked'
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.parquet',Path(d)/'b.parquet'
            pq.write_table(pa.Table.from_pylist(rows),a);pq.write_table(pa.Table.from_pylist(migrated),b)
            source.data['data.train_file']=str(a);cfg.data['data.train_file']=str(b)
            result=pipeline.warmup_source_compatibility(cfg,source)
            self.assertTrue(result['metadata_only_migration']);self.assertEqual(2,result['rows'])
            pq.write_table(pa.Table.from_pylist(migrated[::-1]),b)
            with self.assertRaisesRegex(Exception,'ordered training content'):
                pipeline.warmup_source_compatibility(cfg,source)
            source.data['data.train_batch_size']=32
            for target in (a, b):
                cfg.data['data.train_file'] = str(target)
                with self.subTest(target=target), self.assertRaisesRegex(Exception,'train_batch_size'):
                    pipeline.warmup_source_compatibility(cfg,source)

    def test_baseline_scorer_is_record_only(self):
        from scoring.math import identity,score,correctness_label
        opts={'version':'math_unified_v1','timeout_seconds':2.,'fingerprint':identity(2.)['fingerprint']}
        fresh=[{'student':{'response':text},'ground_truth':'2','prompt_truncated':False,'prompt':'1+1?'}
               for text in ('Answer: 2','Answer: 3','unfinished')]
        before=copy.deepcopy(fresh);labels=student_scorer_labels(fresh,opts)
        self.assertEqual(before,fresh)
        self.assertEqual([correctness_label(score(p['student']['response'],'2',opts)) for p in fresh],labels)
        values=torch.tensor([[0.,1.],[0.,2.]],dtype=torch.bfloat16)
        batch=SimpleNamespace(batch={},non_tensor_batch={})
        for enabled in (False,True):
            opts={'gad_extensions':{'stage':'gad','actor_reward_fp32':True,'student_scorer_monitor':enabled},'rubric':{'enabled':False}}
            self.assertTrue(torch.equal(values.float(),driver_reward(batch,values,opts,{})))

    @unittest.skipUnless(os.environ.get('GAD_TEST_EXTENSION_ROOT'),'needs generated fixture')
    def test_baseline_scorer_in_isolated_backend_import_environment(self):
        from scoring.math import identity
        backend=Path(os.environ['GAD_TEST_EXTENSION_ROOT'])/'gad'
        options={'version':'math_unified_v1','entrypoint':str(ROOT/'scoring/math.py'),
                 'timeout_seconds':2.,'fingerprint':identity(2.)['fingerprint']}
        code='''import sys,json,importlib.util
sys.path.insert(0,sys.argv[1])
from verl.experimental.gad_extensions.actor_reward import student_scorer_labels
assert importlib.util.find_spec('scoring') is None, 'test must start without LMOps on sys.path'
fresh=[{'student':{'response':'Answer: 2'},'ground_truth':'2','prompt':'1+1?','prompt_truncated':False}]
labels=student_scorer_labels(fresh,json.loads(sys.argv[2]))
assert labels[0]['valid'] and labels[0]['value']==1, labels
print('isolated_scorer_ok')
'''
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([sys.executable,'-I','-c',code,str(backend),json.dumps(options)],
                cwd=d,capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=60)
        self.assertEqual(0,result.returncode,result.stderr)
        self.assertIn('isolated_scorer_ok',result.stdout)

    @unittest.skipUnless(os.environ.get('GAD_TEST_EXTENSION_ROOT'),'needs generated fixture')
    def test_actual_driver_scattered_microbatches_and_grpo(self):
        backend=Path(os.environ['GAD_TEST_EXTENSION_ROOT'])/'gad';sys.path.insert(0,str(backend))
        from verl import DataProto
        from verl.experimental.gad_extensions import worker as deployed
        tree=ast.parse((backend/'verl/workers/critic/dp_critic.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compute_values')
        method.decorator_list=[];scope={'torch':torch,'DataProto':DataProto}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<critic>','exec'),scope)
        # Each of the four uid groups is spread across all four DP ranks and micros.
        order=[12,1,6,11,0,5,10,15,4,9,14,3,8,13,2,7]
        ids=torch.tensor(order);n=len(ids)
        data=DataProto.from_dict(tensors={'responses':torch.ones(n,2,dtype=torch.long),
            'input_ids':ids[:,None].expand(-1,4),'attention_mask':torch.ones(n,4,dtype=torch.long),
            'position_ids':torch.arange(4)[None,:].expand(n,-1)},
            non_tensors={'uid':np.array(['u'+str(i%4) for i in order],dtype=object)},
            meta_info={'use_dynamic_bsz':False,'micro_batch_size':2})
        critic=SimpleNamespace(config=OmegaConf.create({'rubric':OPTIONS}),ulysses_sequence_parallel_size=1,
                               critic_module=SimpleNamespace(eval=lambda:None))
        def forward(critic,micro,teacher):
            x=micro['input_ids'][:,0].float()
            logits=torch.stack((x*.1,x*.03,-x*.01,x*.02),-1).to(torch.bfloat16)[:,None,:].expand(-1,2,-1)
            return logits,micro['attention_mask'][:,-2:]
        with patch.object(deployed,'forward_channels',side_effect=forward):
            payload=DataProto.concat([DataProto.from_dict(tensors=scope['compute_values'](critic,d)) for d in data.chunk(4)])
        data=data.union(payload);data.batch['response_mask']=data.batch['attention_mask'][:,-2:]
        trainer={'gad_extensions':{'stage':'gad','actor_reward_fp32':True},'rubric':OPTIONS}
        values=driver_reward(data,data.batch['values'],trainer,{'use_kl_in_reward':False})
        parts=data.batch['rubric_reward_components'];uid=data.non_tensor_batch['uid']
        expected=reward_scores(parts,OPTIONS,uid)[2]
        torch.testing.assert_close(values.sum(-1),expected,rtol=0,atol=0)
        for group in set(uid):
            select=np.flatnonzero(uid==group)
            direct=reward_scores(parts[select],OPTIONS,['a']*4)[2]
            torch.testing.assert_close(expected[select],direct,rtol=0,atol=0)
        core=ast.parse((backend/'verl/trainer/ppo/core_algos.py').read_text())
        grpo=next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='compute_grpo_outcome_advantage')
        grpo.decorator_list=[];scope={'torch':torch,'np':np,'defaultdict':defaultdict}
        exec(compile(ast.Module(body=[grpo],type_ignores=[]),'<grpo>','exec'),scope)
        actual,_=scope[grpo.name](values,data.batch['response_mask'],uid)
        torch.testing.assert_close(actual[:,0],group_advantages(expected,uid),rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'use_kl_in_reward'):
            driver_reward(data,values,trainer,{'use_kl_in_reward':True})
        # The emitted trainer actually calls the shared driver function before GRPO.
        trainer_source=(backend/'verl/trainer/ppo/ray_trainer.py').read_text()
        self.assertIn('reward_tensor = driver_reward(batch, reward_tensor, self.config.trainer, self.config.algorithm)',trainer_source)
        # A baseline monitor must return the very same Critic batch, not rebuild it.
        from gad.extensions.runtime import Runtime
        data.batch.pop('rubric_reward_components')
        data.batch['advantages']=group_advantages(data.batch['values'].float().sum(-1),uid)[:,None].expand(-1,2)
        data.batch['token_level_rewards']=data.batch['values'].float()
        fresh=[]
        for i in range(n):
            fresh.append({'source_index':str(i),'source_step':272,'prompt_ids':[1], 'pair_id':str(i),
                'prompt_hash':'stable-'+str(i%4),'prompt':'1+1?','ground_truth':'2','prompt_truncated':False,
                'student':{'token_ids':[1,2],'response':'Answer: 2','response_id':'s'+str(i),'finish_reason':'stop'},
                'teacher':{'token_ids':[1,2],'response_id':'t'+str(i),'truncated':False}})
        runtime=Runtime.__new__(Runtime)
        runtime.trainer=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        from scoring.math import identity
        runtime.options={'token_identity':'template','student_scorer_monitor':True,
                         'scorer_identity':{'version':'math_unified_v1','timeout_seconds':2.,'fingerprint':identity(2.)['fingerprint']}}
        runtime.stage='gad';runtime.stage_step=0;runtime.teacher_provenance={};runtime.replay=None;runtime.rubric=None
        runtime.pending_reward_monitor=None
        runtime.execution={};runtime.prepared=None;runtime.annotation_future=None;runtime.teacher_future=None
        original={k:v.clone() for k,v in data.batch.items()}
        with patch('gad.extensions.runtime.extract_pairs',return_value=fresh),patch('gad.extensions.runtime.pack_pairs',side_effect=AssertionError('baseline must not rebuild')):
            passed,_=runtime.prepare(data,272)
        self.assertIs(data,passed)
        for key,value in original.items():self.assertTrue(torch.equal(value,data.batch[key]),key)
        self.assertTrue(all(r['question_id'].startswith('stable-') for r in runtime.pending_reward_monitor))
        self.assertTrue(all(r['parts'][1:]==[None,None,None] for r in runtime.pending_reward_monitor))

    def test_shadow_values_exact_and_no_uid_math_on_worker(self):
        # Shadow cannot alter either BF16 values or FP32-controlled advantages.
        from gad.extensions.core.heads import last_response_logits
        logits=torch.tensor([[[1.,2.,3.,4.],[2.,3.,4.,5.]],[[3.,4.,5.,6.],[4.,5.,6.,7.]]],dtype=torch.bfloat16)
        mask=torch.tensor([[1,0],[1,1]])
        class Batch:
            def split(self,n):return [{'responses':torch.ones(2,2)}]
        proto=SimpleNamespace(meta_info={'use_dynamic_bsz':False,'micro_batch_size':2},
                              select=lambda **kw:SimpleNamespace(batch=Batch()))
        critic=SimpleNamespace(config=OmegaConf.create({'rubric':{'enabled':True,'reward_source':'adv_only','actor_reward_shadow':True}}),
            ulysses_sequence_parallel_size=1,critic_module=SimpleNamespace(eval=lambda:None))
        with patch.object(worker,'forward_channels',return_value=(logits,mask)):
            ordinary=worker.reward_forward(critic,{},False)
            shadow=worker.compute_mixed_values(critic,proto)['values']
        self.assertEqual(ordinary.dtype,shadow.dtype);self.assertTrue(torch.equal(ordinary,shadow))
        for scores in (ordinary.sum(-1),ordinary.float().sum(-1)):
            self.assertTrue(torch.equal(group_advantages(scores,['a','a']),group_advantages(shadow.to(scores.dtype).sum(-1),['a','a'])))

    def test_offline_split_floors_and_no_implicit_freeze(self):
        qids={half:next('q'+str(i) for i in range(100) if question_half('q'+str(i))==half) for half in (0,1)}
        rows=[]
        for step in (272,450):
            for half,qid in qids.items():
                batch=rows_for([[0.,.9,.6,.1],[.1,.1,.2,.9],[.2,.3,.4,.5],[.3,.5,.8,.3]],step=step)
                for r in batch:r.update(question_id=qid,uid='u'+str(half))
                rows+=batch
        result=analyze(rows)
        self.assertEqual(8,len(result['candidates']));self.assertEqual(16,len(result['sensitivity_only']))
        self.assertTrue(all(r['tau']==result['floors_frozen_before_weight_results'][r['components']]['q25'] for r in result['candidates']))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl';p.write_text('\n'.join(json.dumps({'step':s,'actor_reward_monitor':[r for r in rows if r['step']==s]}) for s in (272,450)))
            self.assertEqual(rows,read_shadow(p))
        # Every path uses the same reward function (not separate formula copies).
        import gad.extensions.actor_reward as ar
        with patch.object(ar,'groupnorm_scores',wraps=ar.groupnorm_scores) as f:
            constraint_statistics(rows,OPTIONS);summarize(rows,OPTIONS)
            self.assertEqual(4,f.call_count)


if __name__=='__main__':unittest.main()
