"""Execute the deployed trainer update block, including failure and Warmup gate."""
import ast
import os
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TrainingHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured=os.environ.get('GAD_TEST_EXTENSION_ROOT')
        if configured:
            cls.backend_root=Path(configured)
        else:
            sys.path.insert(0,str(ROOT/'gad/tests'))
            from backend_fixture import prepare,cleanup
            temporary=tempfile.TemporaryDirectory(prefix='gad-hook-cpu-')
            cls.addClassCleanup(temporary.cleanup)
            cls.backend_root=Path(temporary.name)/'extensions'
            cls.addClassCleanup(cleanup,cls.backend_root)
            prepare(cls.backend_root,'gad_extensions_v1','warmup,gad')

    def block(self, stage):
        import sys
        sys.path.insert(0, str(ROOT / 'gad/tools'))
        path = self.backend_root / stage / 'verl/trainer/ppo/ray_trainer.py'
        self.assertTrue(path.exists(),'current isolated backend must be prepared for the optimizer barrier test')
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

    def test_teacher_truncation_is_observed_before_length_cap(self):
        import torch
        source=self.backend_root/'gad/verl/utils/dataset/rl_dataset.py'
        tree=ast.parse(source.read_text())
        code=None
        for parent in ast.walk(tree):
            for _,body in ast.iter_fields(parent):
                if not isinstance(body,list):continue
                for i,node in enumerate(body[:-1]):
                    if (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='teacher_response' for t in node.targets)
                            and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='self.tokenizer'):
                        self.assertIsInstance(body[i+1],ast.If)
                        self.assertIn('gad_teacher_token_truncated',ast.unparse(body[i+1]))
                        code=compile(ast.fix_missing_locations(ast.Module(body=body[i:i+2],type_ignores=[])),str(source),'exec')
        self.assertIsNotNone(code)
        for length in (2,5):
            namespace={'self':SimpleNamespace(gad_extensions_enabled=True,max_response_length=3,
                tokenizer=lambda *a,**k:{'input_ids':torch.ones((1,length),dtype=torch.long)}),
                'teacher_response':'a teacher answer','row_dict':{}}
            exec(code,namespace)
            namespace['teacher_response']['input_ids']=namespace['teacher_response']['input_ids'][:,:3]
            self.assertEqual(length>3,namespace['row_dict']['gad_teacher_token_truncated'])

    def test_unset_extension_seed_matches_upstream_default(self):
        import yaml
        cfg=yaml.safe_load((self.backend_root/'gad/verl/trainer/config/ppo_trainer.yaml').read_text())
        self.assertEqual(0,cfg['actor_rollout_ref']['rollout']['seed'])


if __name__ == '__main__':
    unittest.main()
