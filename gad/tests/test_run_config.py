"""CPU checks for config composition and saved-run isolation."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from common import PipelineError
from config import (add_config_arguments, compose_config, load_config, load_run_config,
                    nested_config, new_run_id, save_run_config, select_config, validate_config)
from pipeline import stage_signature
from prune_checkpoints import apply_retention, plan_retention


class RunConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.template = self.root / 'train.yaml'
        self.raw = {
            'selection': {'model': 'small', 'dataset': 'full'},
            'catalog': {
                'models': {'small': {'models': {'student': 'models/small'}},
                           'large': {'models': {'student': 'models/large'}}},
                'datasets': {name: {'data': {'train_file': f'data/{name}/train.parquet',
                                            'val_file': f'data/{name}/dev.parquet',
                                            'manifest': f'data/{name}/manifest.json',
                                            'teacher_index': f'data/{name}/index.jsonl',
                                            'expected_fingerprint_digest': name},
                                    'validation': {'quick_file': f'data/{name}/quick.parquet'}}
                             for name in ('full', 'subset')},
            },
            'run': {'output_root': 'runs', 'run_id': 'auto'},
            'backends': {'root': 'backends'},
            'data': {'train_batch_size': 64},
            'budget': {'stage_epochs': {'seqkd': 2, 'warmup': 1, 'gad': 2}},
            'rollout': {'n': 2},
            'optim': {'gad_actor_lr': 1e-6},
        }
        self.write_template()

    def write_template(self):
        self.template.write_text(yaml.safe_dump(self.raw))

    def arguments(self, *values):
        parser = argparse.ArgumentParser()
        add_config_arguments(parser)
        return parser.parse_args(values)

    def saved_run(self):
        cfg = compose_config(self.template)
        cfg.data['run.run_id'] = 'example'
        run = self.root / 'runs/example'
        save_run_config(cfg, run)
        return cfg, run

    def test_model_and_dataset_switch_together(self):
        cfg = compose_config(self.template, model='large', dataset='subset')
        self.assertEqual(cfg.get('models.student'), str(self.root / 'models/large'))
        self.assertEqual(cfg.get('models.discriminator_init'), cfg.get('models.student'))
        for key, filename in [('train_file', 'train.parquet'), ('manifest', 'manifest.json'),
                              ('teacher_index', 'index.jsonl')]:
            self.assertEqual(cfg.get('data.' + key), str(self.root / 'data/subset' / filename))
        self.assertEqual(cfg.get('data.expected_fingerprint_digest'), 'subset')
        self.assertEqual(cfg.get('validation.quick_file'), str(self.root / 'data/subset/quick.parquet'))
        self.assertEqual(cfg.get('run.output_root'), str(self.root / 'runs'))
        self.assertFalse(any(key.startswith('catalog.') for key in cfg.data))

    def test_override_priority_and_relative_paths(self):
        folder = self.root / 'choices'
        folder.mkdir()
        override = folder / 'small.yaml'
        override.write_text('rollout:\n  n: 4\ndata:\n  train_file: ./local.parquet\n')
        cfg = compose_config(self.template, overrides=[str(override)], settings=['rollout.n=6'])
        self.assertEqual(cfg.get('rollout.n'), 6)
        self.assertEqual(cfg.get('data.train_file'), str(folder / 'local.parquet'))

    def test_unknown_model_and_mistyped_override_fail(self):
        with self.assertRaises(PipelineError):
            compose_config(self.template, model='missing')
        with self.assertRaises(PipelineError):
            compose_config(self.template, settings=['rollout.nn=4'])

    def test_flat_and_nested_configs_round_trip(self):
        cfg = compose_config(self.template)
        flat = self.root / 'flat.yaml'
        nested = self.root / 'nested.yaml'
        flat.write_text(yaml.safe_dump(cfg.data))
        nested.write_text(yaml.safe_dump(nested_config(cfg.data)))
        self.assertEqual(load_config(flat).data, load_config(nested).data)

    def test_saved_run_ignores_future_global_changes(self):
        cfg, run = self.saved_run()
        self.raw['optim']['gad_actor_lr'] = 3e-5
        self.write_template()
        self.assertEqual(load_run_config(run).get('optim.gad_actor_lr'), 1e-6)
        self.assertEqual(compose_config(self.template).get('optim.gad_actor_lr'), 3e-5)
        with self.assertRaises(PipelineError):
            save_run_config(compose_config(self.template), run)
        self.assertEqual(load_run_config(run).get('optim.gad_actor_lr'), 1e-6)

    def test_resume_rejects_training_changes_and_preserves_identity_for_gpu_change(self):
        cfg, run = self.saved_run()
        before = stage_signature(load_run_config(run), 'seqkd')['signature']
        with self.assertRaises(PipelineError):
            select_config(self.arguments('--run', str(run), '--set', 'rollout.n=8'))
        changed = select_config(self.arguments('--run', str(run), '--set', 'resources.gpus=6,7'))
        self.assertEqual(changed.get('resources.gpus'), '6,7')
        self.assertEqual(stage_signature(changed, 'seqkd')['signature'], before)
        self.assertEqual(load_run_config(run).get('resources.gpus'), '')

    def test_resume_conflicting_selectors_fail(self):
        _, run = self.saved_run()
        with self.assertRaises(PipelineError):
            select_config(self.arguments('--run', str(run), '--dataset', 'subset'))

    def test_historical_stage_configs_remain_distinct(self):
        cfg, run = self.saved_run()
        raw = yaml.safe_load((run / 'train.yaml').read_text())
        raw['_stage_overrides'] = {'seqkd': {'rollout': {'n': 8},
                                           'budget': {'stage_epochs': {'gad': 4}}}}
        (run / 'train.yaml').write_text(yaml.safe_dump(raw))
        restored = load_run_config(run)
        self.assertEqual(restored.get('rollout.n'), 2)
        self.assertEqual(restored.for_stage('seqkd').get('rollout.n'), 8)
        self.assertEqual(restored.for_stage('seqkd').stage_epochs('gad'), 4)
        self.assertNotIn('_stage_overrides', restored.data)

    def test_run_names_are_unique_and_invalid_label_fails(self):
        cfg = compose_config(self.template)
        self.assertNotEqual(new_run_id(cfg, 'trial'), new_run_id(cfg, 'trial'))
        with self.assertRaises(PipelineError):
            new_run_id(cfg, '../escape')

    def test_invalid_epoch_budgets_fail_validation(self):
        cfg = compose_config(self.template, settings=['budget.stage_epochs.gad=1'])
        self.assertTrue(any('GAD absolute' in message for message in validate_config(cfg)))
        cfg = compose_config(self.template, settings=['budget.stage_epochs.seqkd=2.5'])
        self.assertTrue(any('positive integer' in message for message in validate_config(cfg)))

    def test_retention_keeps_five_saves_exports_and_in_progress_checkpoint(self):
        stage = self.root / 'seqkd'
        stage.mkdir()
        for step in range(50, 351, 50):
            checkpoint = stage / f'global_step_{step}'
            actor = checkpoint / 'actor'
            actor.mkdir(parents=True)
            for name in ('model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt',
                         'config.json', 'tokenizer.json'):
                (actor / name).touch()
            (checkpoint / 'data.pt').touch()
        exported = stage / 'global_step_50/actor/huggingface'
        exported.mkdir()
        (exported / 'model.safetensors').touch()
        (stage / 'global_step_400').mkdir()
        marker = stage / 'latest_checkpointed_iteration.txt'
        marker.write_text('350')
        plan = plan_retention(stage, keep_checkpoints=5)
        self.assertEqual(plan['recent_steps'], [150, 200, 250, 300, 350])
        self.assertEqual([Path(path).name for path in plan['delete']], ['global_step_100'])
        apply_retention(stage, keep_checkpoints=5)
        self.assertTrue(exported.exists())
        self.assertTrue((stage / 'global_step_400').exists())
        self.assertFalse((stage / 'global_step_100').exists())
        (stage / 'global_step_350/data.pt').unlink()
        self.assertEqual(plan_retention(stage, keep_checkpoints=1)['delete'], [])


if __name__ == '__main__':
    unittest.main()
