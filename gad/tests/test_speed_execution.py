"""Exact question order, speculative annotation parity and optimizer barriers."""
from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools'), str(ROOT / 'gad/tests')]
import torch
from torch.utils.data import RandomSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from gad.extensions.sampler_state import ExactRandomSampler, prepare_state, main_sampler_state, VERSION
from gad.extensions.prefetch import prefetch, selected_teachers, teacher_pairs
from gad.extensions.process_sampling import select_process
from gad.extensions.critic_batch import extract_pairs, supervision_identity
from gad.extensions.runtime import Runtime
import test_direct_judge as direct_tests
from test_process_sampling_direct import pairs


def stream(loader, epochs=10):
    for _ in range(epochs):
        for batch in loader:
            yield batch.tolist()


def loader(cls=ExactRandomSampler, workers=0, size=37, batch=4, seed=42):
    data = list(range(size))
    return StatefulDataLoader(data, batch_size=batch, drop_last=True, num_workers=workers,
        sampler=cls(data, generator=torch.Generator().manual_seed(seed)))


class SamplerTests(unittest.TestCase):
    def test_uninterrupted_order_is_original_torch_order_including_epoch_boundaries(self):
        for workers in [0, 2]:
            a = stream(loader(RandomSampler, workers), 3)
            b = stream(loader(ExactRandomSampler, workers), 3)
            self.assertEqual(list(a), list(b))

    def test_resume_matches_question_ids_across_multiple_epochs_and_prefetch(self):
        for workers in [0, 2, 8]:
            for cut in [2, 9, 10]:
                a = loader(workers=workers)
                uninterrupted = stream(a)
                for _ in range(cut):
                    next(uninterrupted)
                state = deepcopy(a.state_dict())
                b = loader(workers=workers, seed=999)
                b.load_state_dict(prepare_state(b, state, '/nonexistent'))
                resumed = stream(b)
                # Values stand for question identities; cursor equality alone is insufficient.
                self.assertEqual([next(uninterrupted) for _ in range(20)], [next(resumed) for _ in range(20)])

    def test_legacy_resume_fails_and_explicit_handoff_preserves_original_import_order(self):
        with tempfile.TemporaryDirectory() as d:
            a = loader(RandomSampler)
            old_stream = stream(a)
            for _ in range(9):
                next(old_stream)
            state = deepcopy(a.state_dict())
            exact = loader()
            with self.assertRaisesRegex(ValueError, 'legacy checkpoint'):
                prepare_state(exact, state, d)
            marker = Path(d) / 'handoff_initialization.json'
            marker.write_text(json.dumps({'same_stage_resume': False}))
            migrated = prepare_state(exact, state, d)
            self.assertNotIn('sampler_iter_state', main_sampler_state(state))
            self.assertEqual(VERSION, main_sampler_state(migrated)['sampler_iter_state']['version'])
            old = loader(RandomSampler)
            old.load_state_dict(state)
            exact.load_state_dict(migrated)
            self.assertEqual(list(stream(old, 3)), list(stream(exact, 3)))

    def test_changed_dataset_rejected(self):
        a = loader()
        next(iter(a))
        state = a.state_dict()
        b = loader(size=39)
        b.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'dataset/options changed'):
            next(iter(b))


class PrefetchTests(unittest.TestCase):
    def owner(self, directory):
        owner = direct_tests.DirectTests().owner(Path(directory) / 'gad')
        owner.options['process_sampling'] = {'max_unique_per_global_minibatch': 32, 'seed': 42}
        return owner

    def test_teacher_quota_independent_of_student_output_and_global_rng(self):
        ps = pairs(256, 4)
        options = {'mode': 'joint', 'process_sampling': {'max_unique_per_global_minibatch': 32, 'seed': 42}}
        rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
        pre = selected_teachers(ps, 256, 4, options, 272, 'gad')
        selection, groups = select_process(ps, 256, 4, options, 272, 'gad')
        self.assertEqual(11, len(pre))
        self.assertEqual({supervision_identity(p, 'teacher') for p in pre},
            {supervision_identity(p, 'teacher') for i, p in enumerate(ps) if selection['teacher'][i]})
        for i, p in enumerate(ps):
            p['student']['response_id'] = 'changed' + str(i)
        self.assertEqual(pre, selected_teachers(ps, 256, 4, options, 272, 'gad'))
        self.assertEqual(rng, random.getstate())
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))

    def test_prefetch_and_synchronous_labels_masks_weights_losses_gradients_match(self):
        from rubric.core.losses import weighted_contribution
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            original, fast = self.owner(a), self.owner(b)
            ps1, ps2 = pairs(256, 4), pairs(256, 4)
            batch1, batch2 = SimpleNamespace(batch={}), SimpleNamespace(batch={})
            original.annotate(ps1, batch1, 256, 4, 272)
            chosen = selected_teachers(ps2, 256, 4, fast.options, 272, 'gad')
            result = prefetch(fast, chosen, 272)
            self.assertEqual((11, 11), (len(result['candidate_keys']), len(result['cached_keys'])))
            fast.annotate(ps2, batch2, 256, 4, 272)
            self.assertEqual((32, 32), (original.judge.calls, fast.judge.calls))
            for key in batch1.batch:
                self.assertTrue(torch.equal(batch1.batch[key], batch2.batch[key]), key)
            def loss_and_grad(batch):
                theta = torch.tensor([.1, -.2, .3], requires_grad=True)
                loss = theta.sum() * 0
                for role in ['teacher', 'student']:
                    logits = torch.arange(256).float()[:, None] / 256 * theta[None, :]
                    term, _ = weighted_contribution(logits, batch.batch[role + '_rubric_targets'],
                        batch.batch[role + '_rubric_mask'], batch.batch[role + '_aux_weight'],
                        batch.batch['rubric_global_counts'][0], 4, 'fixed_dimension_gated', 4)
                    loss = loss + term
                loss.backward()
                return loss.detach(), theta.grad
            for left, right in zip(loss_and_grad(batch1), loss_and_grad(batch2)):
                self.assertTrue(torch.equal(left, right))
            attempts = [json.loads(s) for s in (Path(b) / 'gad/judge_attempts.jsonl').read_text().splitlines()]
            self.assertEqual(32, len(attempts))
            self.assertEqual(32, len({r['request_id'] for r in attempts}))
            self.assertTrue(all(r['step'] == 272 for r in attempts))

    def test_failed_prefetch_retries_with_same_bounded_journal(self):
        with tempfile.TemporaryDirectory() as d:
            owner = self.owner(d)
            ps = pairs(256, 4)
            chosen = selected_teachers(ps, 256, 4, owner.options, 272, 'gad')
            target = owner.cache_key(chosen[0], 'teacher')
            once = [False]
            original = owner.judge.annotate_one
            def fail_once(rid, messages, deadline):
                if rid == target and not once[0]:
                    once[0] = True
                    owner.judge.calls += 1
                    return rid, {'request_id': rid, 'error': 'judge_reasoning_budget_exhausted',
                        'attempts': 1, 'usage': {'completion_tokens': 8192}, 'cacheable': False}
                return original(rid, messages, deadline)
            owner.judge.annotate_one = fail_once
            result = prefetch(owner, chosen, 272)
            self.assertEqual(10, len(result['cached_keys']))
            batch = SimpleNamespace(batch={})
            owner.annotate(ps, batch, 256, 4, 272)
            self.assertEqual(33, owner.judge.calls)
            self.assertIsNotNone(owner.cache.get(target))
            journals = [json.loads(p.read_text()) for p in (Path(d) / 'gad/annotation_state').rglob('*.json')]
            record = next(r for r in journals if r['request_id'] == target)
            self.assertEqual(2, record['annotation_attempt'])
            self.assertEqual(2, record['requests_used'])


class RuntimeTests(unittest.TestCase):
    def fixture(self, directory, early=False, replay=False):
        import numpy as np
        from omegaconf import OmegaConf
        from verl import DataProto
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 9
            def decode(self, ids, **kw):
                return 'Answer: 2' if kw.get('skip_special_tokens') else 'prompt ' + ','.join(map(str, ids))
        prompt = torch.tensor([[0, i + 1] for i in range(8)])
        teacher = torch.tensor([[3, 9, 9] for _ in range(8)])
        student = torch.tensor([[5, 6, 9] for _ in range(8)])
        pm = torch.tensor([[0, 1] for _ in range(8)])
        tm = torch.tensor([[1, 1, 0] for _ in range(8)])
        sm = torch.ones(8, 3, dtype=torch.long)
        non = {'extra_info': np.array([{'index': str(i)} for i in range(8)], dtype=object),
               'reward_model': np.array([{'ground_truth': '2'}] * 8, dtype=object)}
        tensors = {'prompts': prompt, 'responses': student, 'teacher_response': teacher,
            'input_ids': torch.cat([prompt, student], -1), 'teacher_input_ids': torch.cat([prompt, teacher], -1),
            'attention_mask': torch.cat([pm, sm], -1), 'teacher_attention_mask': torch.cat([pm, tm], -1),
            'position_ids': torch.arange(5)[None, :].repeat(8, 1),
            'teacher_position_ids': torch.arange(5)[None, :].repeat(8, 1)}
        batch = DataProto.from_dict(tensors=tensors, non_tensors=non).repeat(4, interleave=True)
        gen = DataProto.from_dict(tensors={'input_ids': prompt, 'teacher_response': teacher, 'attention_mask': pm},
                                  meta_info={'eos_token_id': 9})
        raw = DataProto.from_dict(tensors={'dummy': torch.zeros(8, 1)}, non_tensors=non)
        model = Path(directory) / 'model'
        model.mkdir(exist_ok=True)
        (model / 'generation_config.json').write_text(json.dumps({'eos_token_id': 9, 'pad_token_id': 0}))
        cfg = OmegaConf.create({'trainer': {'n_gpus_per_node': 4, 'nnodes': 1, 'default_local_dir': str(Path(directory) / 'gad'),
            'gad_extensions': {'identity': 'test', 'stage': 'gad', 'start_step': 271, 'token_identity': 'tok',
                'sampling_seed': 42, 'execution': {'early_annotation': early, 'teacher_prefetch': True}},
            'rubric': {'enabled': False}, 'critic_replay': {'enabled': False}},
            'critic': {'ppo_mini_batch_size': 8}, 'actor_rollout_ref': {'rollout': {'n': 4},
                'model': {'path': str(model), 'trust_remote_code': False}}})
        rt = Runtime(SimpleNamespace(config=cfg, tokenizer=Tokenizer()))
        rt.rubric = direct_tests.DirectTests().owner(Path(directory) / 'gad')
        rt.rubric.options['process_sampling'] = {'max_unique_per_global_minibatch': 8, 'seed': 42}
        if replay:
            from gad.extensions.replay_buffer import ReplayBuffer
            rt.replay = ReplayBuffer({'seed': 43, 'mix_ratio': .25, 'max_age_steps': 8,
                'capacity_pairs': 256, 'max_bytes': 10000000})
            previous = rt._fresh_pairs(batch, 272)
            rt.replay.commit(previous, 1)
            rt.stage_step = 1
        return rt, batch, gen, raw

    def test_teacher_extraction_matches_main_cache_keys_including_eos_and_visibility(self):
        with tempfile.TemporaryDirectory() as d:
            rt, batch, gen, raw = self.fixture(d)
            pre = teacher_pairs(gen, raw, rt.trainer.tokenizer, 'tok', 272, 'gad', 4)
            fresh = rt._fresh_pairs(batch, 272)
            rt._attach_teacher_source(pre)
            self.assertEqual([rt.rubric.cache_key(p, 'teacher') for p in fresh],
                             [rt.rubric.cache_key(p, 'teacher') for p in pre])
            rt.close()

    def test_full_prefetch_early_submit_wait_commit_lifecycle(self):
        with tempfile.TemporaryDirectory() as d:
            rt, batch, gen, raw = self.fixture(d, early=True)
            original_scorer = rt.rubric.scorer
            def main_only(*args):
                self.assertIs(threading.current_thread(), threading.main_thread())
                return original_scorer(*args)
            rt.rubric.scorer = main_only
            rt.prefetch_teachers(gen, raw, 272)
            rt.begin_prepare(batch, 272)
            critic, stats = rt.prepare(batch, 272)
            self.assertEqual(3, stats['execution/teacher_prefetch_candidates'])
            self.assertEqual(3, stats['execution/teacher_prefetch_hits'])
            self.assertEqual(8, stats['rubric/judge_requests'])
            self.assertEqual(8, stats['rubric/json_valid_count'])
            self.assertEqual(8, rt.rubric.judge.calls)
            self.assertNotIn('teacher_rubric_targets', batch.batch.keys())
            self.assertIn('teacher_rubric_targets', critic.batch.keys())
            rt.commit(True)
            self.assertEqual(272, rt.committed_step)
            self.assertIsNone(rt.teacher_future)
            attempts = [json.loads(s) for s in (Path(d) / 'gad/judge_attempts.jsonl').read_text().splitlines()]
            self.assertEqual(8, len(attempts))
            self.assertTrue(all(r['step'] == 272 for r in attempts))
            rt.close()

    def test_prefetch_uses_worker_generation_eos_list(self):
        with tempfile.TemporaryDirectory() as d:
            rt, batch, gen, raw = self.fixture(d, early=True)
            model = Path(rt.trainer.config.actor_rollout_ref.model.path)
            (model / 'generation_config.json').write_text(json.dumps({'eos_token_id': [9, 8], 'pad_token_id': 0}))
            # Token 8 stops the real worker before tokenizer's token 9. The
            # driver's unprocessed meta may still contain only token 9.
            batch.batch['teacher_response'][:] = torch.tensor([3, 8, 9])
            batch.batch['teacher_input_ids'][:, -3:] = torch.tensor([3, 8, 9])
            gen.batch['teacher_response'][:] = torch.tensor([3, 8, 9])
            rt.prefetch_teachers(gen, raw, 272)
            rt.begin_prepare(batch, 272)
            _, stats = rt.prepare(batch, 272)
            self.assertEqual(3, stats['execution/teacher_prefetch_hits'])
            self.assertEqual(8, rt.rubric.judge.calls)
            self.assertEqual([9, 8], rt._prefetch_eos)
            rt.close()

    def test_early_submit_keeps_policy_tensors_and_selects_replay_once(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            sync, batch1, _, _ = self.fixture(a, replay=True)
            fast, batch2, _, _ = self.fixture(b, early=True, replay=True)
            started, release = threading.Event(), threading.Event()
            original = fast.rubric.annotate
            def delayed(*args, **kwargs):
                started.set()
                if not release.wait(10):
                    raise TimeoutError('test gate')
                return original(*args, **kwargs)
            fast.rubric.annotate = delayed
            originals = {k: v.clone() for k, v in batch2.batch.items()}
            with patch.object(fast.replay, 'select', wraps=fast.replay.select) as choose:
                fast.begin_prepare(batch2, 273)
                self.assertTrue(started.wait(5))
                self.assertEqual(1, choose.call_count)
                for key, value in originals.items():
                    self.assertTrue(torch.equal(value, batch2.batch[key]))
                release.set()
                actual, stats = fast.prepare(batch2, 273)
                self.assertEqual(1, choose.call_count)
            expected, _ = sync.prepare(batch1, 273)
            for key in expected.batch.keys():
                self.assertTrue(torch.equal(expected.batch[key], actual.batch[key]), key)
            self.assertEqual(sync.replay.rng.getstate(), fast.replay.rng.getstate())
            fast.commit(True)
            sync.commit(True)
            self.assertEqual(sync.replay.state_dict(), fast.replay.state_dict())
            fast.close()
            sync.close()

    def test_annotation_failure_propagates_before_optimizers_or_commit(self):
        with tempfile.TemporaryDirectory() as d:
            rt, batch, _, _ = self.fixture(d, early=True)
            with patch.object(rt.rubric, 'annotate', side_effect=RuntimeError('Judge unavailable')):
                rt.begin_prepare(batch, 272)
                with self.assertRaisesRegex(RuntimeError, 'Judge unavailable'):
                    rt.prepare(batch, 272)
            self.assertIsNone(rt.pending)
            self.assertEqual(271, rt.committed_step)
            rt.close()


class BackendTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('GAD_TEST_EXTENSION_ROOT'), 'needs freshly generated backend')
    def test_real_trainer_barriers_and_sampler_creation(self):
        import subprocess
        backend = Path(os.environ['GAD_TEST_EXTENSION_ROOT']) / 'gad'
        source = (backend / 'verl/trainer/ppo/ray_trainer.py').read_text()
        fit = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        lines = {}
        for node in ast.walk(fit):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                lines.setdefault(node.func.attr, []).append(node.lineno)
        phases = ['prefetch_teachers', 'generate_sequences', 'begin_prepare', 'compute_log_prob',
                  'prepare', 'update_critic', 'update_actor']
        order = [min(lines[name]) for name in phases]
        self.assertEqual(sorted(order), order)
        code = '''import ast,sys,torch
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from omegaconf import OmegaConf
from verl.experimental.gad_extensions.sampler_state import ExactRandomSampler
p=Path(sys.argv[1])/'verl/trainer/main_ppo.py'
node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='create_rl_sampler')
scope={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),scope)
sampler=scope['create_rl_sampler'](OmegaConf.create({'shuffle':True,'seed':42,'gad_extensions':{'enabled':True}}),list(range(37)))
assert isinstance(sampler,ExactRandomSampler)
expected=torch.utils.data.RandomSampler(list(range(37)),generator=torch.Generator().manual_seed(42))
assert list(sampler)==list(expected)
'''
        result = subprocess.run([sys.executable, '-I', '-c', code, str(backend)], capture_output=True,
            text=True, timeout=60, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
