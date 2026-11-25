import json
import random
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gad.extensions.replay_buffer import ReplayBuffer
from gad.extensions.checkpoint_state import checkpoint_problems, latest_complete, save, restore
from gad.extensions.label_cache import LabelCache
from test_online_rubric import pair


class ReplayTests(unittest.TestCase):
    def options(self, **changes):
        return {"seed": 43, "mix_ratio": .25, "capacity_pairs": 8, "max_age_steps": 2,
                "max_bytes": 100000, **changes}

    def test_empty_pool_fixed_budget_and_late_insert(self):
        buffer = ReplayBuffer(self.options())
        fresh = [pair(i) for i in range(8)]
        chosen, n = buffer.select(fresh, 1)
        self.assertEqual((8, 0), (len(chosen), n))
        self.assertEqual(0, len(buffer.entries))
        buffer.commit(fresh, 1)
        chosen, n = buffer.select([pair(i + 10, step=2) for i in range(8)], 2)
        self.assertEqual((8, 2), (len(chosen), n))
        self.assertEqual(8, len(set(p['pair_id'] for p in chosen)))
        for p in chosen: self.assertEqual('p', p['prompt_id'])
        self.assertFalse(any('advantages' in p or 'old_log_probs' in p for p in buffer.entries))

    def test_eviction_age_capacity_bytes(self):
        buffer = ReplayBuffer(self.options(capacity_pairs=2))
        buffer.commit([pair(i) for i in range(5)], 1)
        self.assertEqual(['3', '4'], [p['pair_id'] for p in buffer.entries])
        buffer.evict(4)
        self.assertFalse(buffer.entries)
        self.assertEqual(2, buffer.evicted_age)
        tiny = ReplayBuffer(self.options(max_bytes=1))
        tiny.commit([pair(1)], 1)
        self.assertEqual(1, tiny.skipped_oversize)
        self.assertFalse(tiny.entries)

    def test_rng_restore_and_global_rng_isolation(self):
        random.seed(9); expected = random.getstate()
        buffer = ReplayBuffer(self.options())
        buffer.commit([pair(i) for i in range(8)], 1)
        state = buffer.state_dict()
        restored = ReplayBuffer(self.options());restored.load_state_dict(state)
        fresh = [pair(i + 10, step=2) for i in range(8)]
        self.assertEqual(buffer.select(fresh, 2), restored.select(fresh, 2))
        self.assertEqual(expected, random.getstate())
        state['entries'][0]['student']['token_ids'].append(88)
        with self.assertRaises(ValueError): restored.load_state_dict(state)

    def test_ratio_zero_and_undersized_pool(self):
        buffer = ReplayBuffer(self.options(mix_ratio=0))
        buffer.commit([pair(0)], 1)
        fresh = [pair(i + 10) for i in range(8)]
        self.assertEqual(0, buffer.select(fresh, 2)[1])
        buffer.options['mix_ratio'] = .75
        self.assertEqual((8, 1), (len(buffer.select(fresh, 2)[0]), buffer.select(fresh, 2)[1]))


class CheckpointTests(unittest.TestCase):
    def runtime(self, directory):
        from types import SimpleNamespace
        return SimpleNamespace(identity='identity', world_size=4, stage='gad', stage_start=2,
                               stage_step=1, committed_step=3, replay=ReplayBuffer(ReplayTests().options()),
                               rubric=SimpleNamespace(cache=LabelCache(Path(directory) / 'live'), empty_student_steps=2))

    def test_atomic_restore_cache_and_incomplete_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); runtime = self.runtime(root)
            runtime.replay.commit([pair(0)], 1)
            runtime.rubric.cache.put('a', {'value': 1})
            complete = root / 'global_step_3';save(runtime, complete)
            self.assertEqual([], checkpoint_problems(complete, required=True))
            incomplete = root / 'global_step_4';incomplete.mkdir()
            self.assertEqual(str(complete), latest_complete(root))
            target = self.runtime(root / 'new')
            restore(target, complete)
            self.assertEqual(runtime.replay.state_dict(), target.replay.state_dict())
            self.assertEqual(1, target.rubric.cache.get('a')['value'])
            (complete / 'training_extensions/replay_state.pt').unlink()
            self.assertTrue(checkpoint_problems(complete, required=True))
            self.assertIsNone(latest_complete(root))

    def test_wrong_world_or_algorithm_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);runtime = self.runtime(root);save(runtime, root/'global_step_3')
            runtime.world_size=2
            with self.assertRaises(ValueError):restore(runtime, root/'global_step_3')

    def test_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual([],checkpoint_problems(directory))
            self.assertTrue(checkpoint_problems(directory,required=True))

    def test_missing_or_changed_dataloader_and_role_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);runtime = self.runtime(root)
            runtime.trainer = object()
            complete = root / 'global_step_3'
            complete.mkdir()
            (complete / 'data.pt').write_bytes(b'dataloader fixture')
            for role in ('actor', 'critic'):
                folder = complete / role;folder.mkdir()
                for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
                    (folder / name).write_text('{}')
                for rank in range(4):
                    for prefix in ('model', 'optim', 'extra_state'):
                        (folder / f'{prefix}_world_size_4_rank_{rank}.pt').write_bytes(b'fixture')
            save(runtime, complete)
            self.assertFalse(checkpoint_problems(complete, required=True))
            for name in ('data.pt', 'critic/config.json', 'actor/tokenizer.json'):
                target = complete / name;before = target.read_bytes();target.unlink()
                self.assertTrue(checkpoint_problems(complete, required=True), name)
                self.assertIsNone(latest_complete(root))
                with self.assertRaises(ValueError):
                    save(runtime, complete)
                self.assertFalse((complete / 'checkpoint_complete.json').exists())
                target.write_bytes(before);save(runtime, complete)
                target.write_bytes(before + b'changed')
                self.assertTrue(checkpoint_problems(complete, required=True), name)
                target.write_bytes(before);save(runtime, complete)


if __name__ == '__main__':unittest.main()
