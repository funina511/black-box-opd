"""Reward dtype, actual worker route, GRPO and label/identity regressions (CPU)."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'gad/tools'), str(ROOT/'gad/tests')]
from gad.extensions import worker
from gad.extensions.actor_reward import DIMS, rewards, group_advantages, monitor_records, summarize, load_window
from cpu_fixtures import compose_cpu_config
from config import validate_config
from extension_config import hydra_overrides
import pipeline

OPTIONS = {'reward_source': 'adv_plus_rubric', 'actor_reward_coef': .1,
           'actor_reward_weights': dict(zip(DIMS, [1., 1., 1.]))}


def config():
    return compose_cpu_config(overrides=[str(ROOT/'configs/methods/rubric_joint.yaml'),
        str(ROOT/'configs/overrides/rubric_direct_v1.yaml')], settings=['rubric.reward_source=adv_plus_rubric', 'rubric.actor_reward_coef=0.1'])


class RewardTests(unittest.TestCase):
    def test_old_dtype_exact_zero_and_last_valid_token(self):
        logits = torch.tensor([[[2., -3., 4., 5.], [1., 2., 3., 4.], [9., 9., 9., 9.]],
                               [[3., 0., 1., 2.], [4., 3., 2., 1.], [5., 3., 2., 1.]]], dtype=torch.bfloat16)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])  # Second response has no EOS.
        critic = SimpleNamespace(config=OmegaConf.create({'rubric': OPTIONS}))
        expected = torch.tensor([[0., 1., 0.], [0., 0., 5.]], dtype=torch.bfloat16)
        with patch.object(worker, 'forward_channels', return_value=(logits, mask)) as forward:
            for mode, coef in [('adv_only', 100.), ('adv_plus_rubric', 0.)]:
                critic.config.rubric.reward_source = mode
                critic.config.rubric.actor_reward_coef = coef
                actual = worker.reward_forward(critic, {}, False)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(torch.equal(expected, actual))
            critic.config.rubric = OmegaConf.create(OPTIONS)
            mixed = worker.reward_forward(critic, {}, False)
            self.assertEqual(torch.float32, mixed.dtype)
            self.assertEqual(2, int((mixed != 0).sum()))
            last = logits[torch.arange(2), torch.tensor([1, 2])].float()
            score = last[:, 0] + .1*(last[:, 1].sigmoid()+last[:, 2].sigmoid()-last[:, 3].sigmoid())
            torch.testing.assert_close(mixed.sum(-1), score)
            self.assertEqual(3, forward.call_count)

    def test_sign_and_no_gradient(self):
        from gad.extensions.actor_reward import combine_last
        base = torch.zeros(4, 4)
        for i in range(3): base[i+1, i+1] = 2
        r, parts = combine_last(base, OPTIONS)
        self.assertGreater(float(r[1]), float(r[0]))
        self.assertGreater(float(r[2]), float(r[0]))
        self.assertLess(float(r[3]), float(r[0]))
        self.assertTrue(torch.equal(parts[:, 0], base[:, 0]))

    def test_worker_hydra_and_resume_identity(self):
        from hydra.core.override_parser.overrides_parser import OverridesParser
        c = config()
        self.assertEqual([], validate_config(c))
        parser = OverridesParser.create()
        def decode(stage):
            args = hydra_overrides(c, stage, Path('/tmp/actor-reward-test'), 271)
            return {p.key_or_group: p.value() for p in parser.parse_overrides(args)}
        gad = decode('gad')
        self.assertEqual(OPTIONS['actor_reward_weights'], gad['critic.rubric']['actor_reward_weights'])
        self.assertEqual('adv_plus_rubric', gad['critic.rubric']['reward_source'])
        warm = decode('warmup')
        self.assertEqual('adv_only', warm['critic.rubric']['reward_source'])
        self.assertEqual(0, warm['critic.rubric']['actor_reward_coef'])
        self.assertEqual([], hydra_overrides(c, 'seqkd', Path('/tmp'), 0))
        identity = gad['trainer.gad_extensions']['identity']
        c.data['rubric.actor_reward_weights.verified_progress'] = 0
        self.assertNotEqual(identity, decode('gad')['trainer.gad_extensions']['identity'])
        c = config(); c.data['rubric.reward_source'] = 'adv_only'; c.data['rubric.actor_reward_coef'] = 0.
        old = decode('gad')
        self.assertNotIn('actor_reward_coef', old['critic.rubric'])
        self.assertNotIn('actor_reward_weights', old['trainer.rubric'])
        for key, value in [('rubric.actor_reward_coef', float('nan')), ('rubric.actor_reward_weights.answer_correctness', -1),
                           ('rubric.process_protocol', 'events_v27'), ('rubric.mode', 'correctness_only')]:
            c = config(); c.data[key] = value
            self.assertTrue(validate_config(c), key)

    def test_grpo_flips_ties_and_small_gaps(self):
        scores = torch.tensor([1., 2., 2., 1., 1., 1., 1e-8, 0.])
        uid = ['a','a','b','b','c','c','d','d']
        a = group_advantages(scores, uid)
        self.assertLess(float(a[0]), 0)
        self.assertGreater(float(a[2]), 0)
        self.assertEqual(0, float(a[4]))
        self.assertLess(float(a[6]), .02)
        torch.testing.assert_close(a[:2], group_advantages(scores[:2]+.2, uid[:2]))
        parts = [[0.,.5,1.,0.],[.01,.5,0.,0.],[0.,.9,0.,0.],[0.,.1,0.,0.],[.05,.5,0.,0.],[0.,1.,0.,0.]]
        rows = [{'step':272,'uid':str(i//2),'parts':p,'target':[0.,0.,0.], 'mask':[1.,0.,0.],
                 'finish_reason':'length' if i==0 else 'stop','category':'T' if i==0 else 'W','tokens':1}
                for i,p in enumerate(parts)]
        stats = summarize(rows, OPTIONS)
        stem = 'actor_reward/pairs/adv_to_full/all/'
        self.assertEqual(1, stats[stem+'strict_flips'])
        self.assertEqual(1, stats[stem+'ties_resolved'])
        self.assertEqual(1, stats[stem+'ties_created'])
        self.assertAlmostEqual(1/3, stats[stem+'before_tie_rate'])
        self.assertEqual(1, stats['actor_reward/pairs/adv_to_full/TW/toward_T'])
        self.assertGreater(stats['actor_reward/pairs/gaps/TW/T_minus_W_total_mean'], 0)

    @unittest.skipUnless(os.environ.get('GAD_TEST_EXTENSION_ROOT'), 'needs generated CPU fixture backend')
    def test_actual_generated_critic_and_dp_payload(self):
        backend = Path(os.environ['GAD_TEST_EXTENSION_ROOT'])/'gad'
        sys.path.insert(0, str(backend))
        from verl import DataProto
        import numpy as np
        n = 8
        mask = torch.ones(n, 4, dtype=torch.long)
        data = DataProto.from_dict(tensors={'responses': torch.ones(n, 2, dtype=torch.long),
            'input_ids': torch.arange(n)[:,None].expand(n,4), 'attention_mask': mask,
            'position_ids': torch.arange(4)[None,:].expand(n,4)},
            meta_info={'micro_batch_size': 2, 'use_dynamic_bsz': False})
        tree = ast.parse((backend/'verl/workers/critic/dp_critic.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataParallelPPOCritic')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_values')
        method.decorator_list = []
        ns = {'torch':torch, 'DataProto':DataProto}
        exec(compile(ast.Module(body=[method],type_ignores=[]), '<actual_compute_values>', 'exec'), ns)
        from verl.experimental.gad_extensions import worker as deployed
        def forward(critic, micro, teacher):
            self.assertFalse(torch.is_grad_enabled())
            ids = micro['input_ids'][:,0].float()
            logits = torch.stack((ids,ids,-ids,ids*.1),dim=-1)[:,None,:].expand(-1,2,-1).to(torch.bfloat16)
            return logits, micro['attention_mask'][:,-2:]
        critic = SimpleNamespace(config=OmegaConf.create({'rubric':{'enabled':True, **OPTIONS}}), ulysses_sequence_parallel_size=1,
                                 critic_module=SimpleNamespace(eval=lambda: None))
        outputs=[]
        with patch.object(deployed, 'forward_channels', side_effect=forward) as f:
            # Actual static microbatch route, then actual DP concat ordering.
            for rank_data in data.chunk(4):
                payload = ns['compute_values'](critic, rank_data)
                outputs.append(DataProto.from_dict(tensors=payload))
            result = DataProto.concat(outputs)
            self.assertEqual(4, f.call_count)
        self.assertEqual(torch.float32, result.batch['values'].dtype)
        self.assertFalse(result.batch['values'].requires_grad)
        self.assertFalse(result.batch['rubric_reward_components'].requires_grad)
        self.assertEqual([8,4], list(result.batch['rubric_reward_components'].shape))
        self.assertEqual(list(range(8)), result.batch['rubric_reward_components'][:,0].tolist())
        torch.testing.assert_close(result.batch['values'].sum(-1), rewards(result.batch['rubric_reward_components'],OPTIONS)[2])
        from collections import defaultdict
        core = ast.parse((backend/'verl/trainer/ppo/core_algos.py').read_text())
        grpo = next(n for n in core.body if isinstance(n,ast.FunctionDef) and n.name=='compute_grpo_outcome_advantage')
        grpo.decorator_list=[]
        scope={'torch':torch,'np':np,'defaultdict':defaultdict}
        exec(compile(ast.Module(body=[grpo],type_ignores=[]),'<actual_grpo>','exec'),scope)
        uids=np.array(['a','a','b','b','c','c','d','d'])
        vals=result.batch['values'].clone()
        actual,_=scope[grpo.name](vals,mask[:,-2:],uids)
        expected=group_advantages(vals.sum(-1),uids)
        torch.testing.assert_close(actual[:,0],expected)
        # Exercise actual FSDP payload expression for both enabled and old callers.
        fsdp = ast.parse((backend/'verl/workers/fsdp_workers.py').read_text())
        assignment = next(n for n in ast.walk(fsdp) if isinstance(n,ast.Assign) and 'tensors=values if isinstance(values, dict)' in ast.unparse(n))
        for values in (result.batch['values'], {k:v for k,v in result.batch.items()}):
            scope={'DataProto':DataProto,'values':values}
            exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<actual_fsdp_payload>','exec'),scope)
            self.assertEqual(isinstance(values,dict), 'rubric_reward_components' in scope['output'].batch)
        # Labels permuted across Critic rows; duplicate response IDs remain distinct occurrences.
        actor = data
        actor.batch['rubric_reward_components'] = result.batch['rubric_reward_components']
        uid = ['a','b','a','b','c','d','c','d']
        actor.non_tensor_batch['uid'] = np.array(uid,dtype=object)
        actor.batch['advantages'] = group_advantages(result.batch['values'].sum(-1),uid)[:,None].expand(-1,2)
        fresh=[{'pair_id':str(i),'student':{'response_id':str(i//2),'finish_reason':'length' if i%2 else 'stop'}} for i in range(n)]
        permutation=[7,0,5,2,6,1,4,3]
        labels=DataProto.from_dict(tensors={'student_rubric_targets':torch.tensor([[i%2,.5,1.] for i in permutation]),
            'student_rubric_mask':torch.tensor([[1.,float(i%2),0.] for i in permutation])},
            non_tensors={'pair_id':np.array([str(i) for i in permutation],dtype=object),
                         'student_response_id':np.array([str(i//2) for i in permutation],dtype=object)})
        rows=monitor_records(actor,labels,fresh,272)
        self.assertEqual([i%2 for i in range(n)], [r['target'][0] for r in rows])
        s=summarize(rows,OPTIONS)
        self.assertEqual(4,s['actor_reward/calibration/p/count'])
        self.assertEqual(0,s['actor_reward/calibration/e/count'])
        self.assertEqual(4,s['actor_reward/pairs/adv_to_full/all/count'])
        self.assertLess(s['actor_reward/advantages_actual_max_error'],1e-6)

    def test_external_warmup_dry_run_preserves_four_heads(self):
        from rubric.core.contracts import file_hash, layout_for
        c=config()
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); source=d/'source'; metadata=source/'warmup/global_step_271/critic/config.json'
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps({'num_labels':4,'rubric_head_layout':list(layout_for('math_process_v2')),
                'rubric_hash':file_hash(c.require('rubric.spec')),'rubric_version':'math_process_v2'}))
            c.data.update({'budget.init_from_warmup_run':str(source),'budget.init_step':271,'budget.gad_new_steps':271})
            target=d/'new/gad/global_step_271'
            handoff={'step':271,'actor_hf':str(target/'actor/huggingface'),'critic_hf':str(target/'critic/huggingface')}
            plan=pipeline.plan_stage(c,d/'new','gad',steps_per_epoch_value=271,handoff=handoff,install_guard=False)
            self.assertEqual(handoff['critic_hf'],plan['critic_path'])
            self.assertEqual((271,271,542),(plan['start_step'],plan['added_steps'],plan['total_steps']))
            self.assertNotIn('trainer.resume_load_worker_state=True',plan['argv'])
            resumed=pipeline.plan_stage(c,d/'new','gad',steps_per_epoch_value=271,handoff=handoff,
                resume={'mode':'resume'},install_guard=False)
            self.assertIn('trainer.resume_load_worker_state=True',resumed['argv'])

    def test_window_filters_uncommitted_steps_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'extension_metrics.jsonl'
            path.write_text('\n'.join(json.dumps({'step':s,'actor_reward_monitor':[{'row':i}]}) for i,s in enumerate([272,273,273,274]))+'\n')
            self.assertEqual([[{'row':0}],[{'row':2}]],load_window(d,273))


if __name__ == '__main__': unittest.main()
