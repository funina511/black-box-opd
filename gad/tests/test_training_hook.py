"""Execute the deployed trainer update block, including failure and Warmup gate."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TrainingHookTests(unittest.TestCase):
    def block(self, stage):
        path = ROOT / 'gad/.backends_gad_extensions_v1' / stage / 'verl/trainer/ppo/ray_trainer.py'
        if not path.exists():
            self.skipTest('deploy the isolated engineering backend to execute its trainer hook')
        tree = ast.parse(path.read_text())
        fit = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        locations = {}
        for call in ast.walk(fit):
            if isinstance(call, ast.Call):
                name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, 'id', '')
                if name in ('compute_values', 'compute_advantage', 'prepare', 'update_critic', 'update_actor', 'commit'):
                    locations[name] = call.lineno
        order = [locations[n] for n in ('compute_values', 'compute_advantage', 'prepare', 'update_critic', 'update_actor', 'commit')]
        self.assertEqual(sorted(order), order)
        for node in ast.walk(fit):
            for _, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for i, child in enumerate(value):
                        if isinstance(child, ast.If) and ast.unparse(child.test) == 'self.use_critic':
                            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'update_critic'
                                   for n in ast.walk(child)):
                                return compile(ast.fix_missing_locations(ast.Module(body=value[i:i + 3], type_ignores=[])), str(path), 'exec')
        self.fail('actual update hook not found')

    def execute(self, block, step=2, fail=None):
        events = []
        actor_batch = SimpleNamespace(meta_info={})
        critic_batch = object()
        def prepare(batch, current_step):
            self.assertIs(batch, actor_batch)
            events.append('annotate')
            if fail == 'judge':
                raise RuntimeError('service down')
            return critic_batch, {}
        def critic(batch):
            self.assertIs(batch, critic_batch)
            events.append('critic')
            return SimpleNamespace(meta_info={'metrics': {}})
        def actor(batch):
            self.assertIs(batch, actor_batch)
            events.append('actor')
            if fail == 'actor':
                raise RuntimeError('actor update failed')
            return SimpleNamespace(meta_info={'metrics': {}})
        def commit(updated):
            events.append(('commit', updated))
            return {}
        trainer = SimpleNamespace(use_critic=True, global_steps=step,
            gad_extensions=SimpleNamespace(prepare=prepare, commit=commit),
            critic_wg=SimpleNamespace(update_critic=critic), actor_rollout_wg=SimpleNamespace(update_actor=actor),
            config=SimpleNamespace(trainer=SimpleNamespace(critic_warmup=2),
                actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(multi_turn=SimpleNamespace(enable=False)))))
        namespace = {'self': trainer, 'batch': actor_batch, 'metrics': {}, 'timing_raw': {},
                     'marked_timer': lambda *a, **k: nullcontext(), 'reduce_metrics': lambda value: value}
        if fail:
            with self.assertRaises(RuntimeError):
                exec(block, namespace)
        else:
            exec(block, namespace)
        return events

    def test_warmup_gate_and_gad_update_order(self):
        self.assertEqual(['annotate', 'critic', ('commit', False)], self.execute(self.block('warmup'), step=1))
        for stage in ('warmup', 'gad'):
            self.assertEqual(['annotate', 'critic', 'actor', ('commit', True)], self.execute(self.block(stage)))

    def test_failure_prevents_optimizer_or_replay_commit(self):
        for stage in ('warmup', 'gad'):
            block = self.block(stage)
            self.assertEqual(['annotate'], self.execute(block, fail='judge'))
            self.assertEqual(['annotate', 'critic', 'actor'], self.execute(block, fail='actor'))


if __name__ == '__main__':
    unittest.main()
