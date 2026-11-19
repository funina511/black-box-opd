"""Baseline planning and preflight must work without any Judge assets or service."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools'), str(ROOT / 'gad/tests')]
from config import validate_config, validate_paths
from cpu_fixtures import load_cpu_config
from extension_config import hydra_overrides
from gad.extensions.judge_client import JudgeClient
from pipeline import plan_stage, stage_signature
from preflight import Report, check_backends, check_models, check_online_judge


class BaselineWithoutJudgeTests(unittest.TestCase):
    def setUp(self):
        scorer = patch('scoring.math.identity', return_value={'fingerprint': 'cpu-baseline-scorer'})
        scorer.start()
        self.addCleanup(scorer.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.cfg = load_cpu_config(ROOT / 'configs/h200_baseline_n4.yaml')
        model = self.folder / 'cpu-model'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        (model / 'tokenizer.json').write_text('{}')
        (model / 'model.safetensors').write_bytes(b'cpu-path-fixture-not-model-weights')
        index = self.folder / 'teacher_index.jsonl'
        index.write_text('')
        manifest = self.folder / 'manifest.json'
        manifest.write_text('{}')
        self.cfg.data.update({
            'models.student': str(model), 'models.discriminator_init': str(model),
            'data.teacher_index': str(index), 'data.manifest': str(manifest),
            'backends.root': str(self.folder / 'not-yet-built-backends'),
            'rubric.judge.tokenizer_path': str(self.folder / 'missing-Qwen3-8B'),
            'rubric.spec': str(self.folder / 'missing-rubric-spec.yaml'),
            'rubric.prompt': str(self.folder / 'missing-rubric-prompt.txt'),
            'rubric.judge.base_url': 'http://127.0.0.1:1/v1',
        })

    def test_disabled_rubric_paths_do_not_block_baseline_config(self):
        self.assertFalse(self.cfg.get('rubric.enabled'))
        self.assertFalse(Path(self.cfg.get('rubric.judge.tokenizer_path')).exists())
        self.assertEqual([], validate_config(self.cfg))
        self.assertEqual([], validate_paths(self.cfg, check_exists=True))

    def test_preflight_checks_only_student_and_single_head_critic(self):
        report = Report()
        with patch('extension_config.service_identity', side_effect=AssertionError('Judge identity requested')), \
                patch.object(JudgeClient, 'health', side_effect=AssertionError('Judge service requested')) as health, \
                redirect_stdout(io.StringIO()):
            check_models(self.cfg, report)
            check_online_judge(self.cfg, report, ['seqkd', 'warmup', 'gad'])
        health.assert_not_called()
        self.assertEqual([], report.failures)
        self.assertEqual({'models.student', 'models.discriminator_init'}, set(report.sections['models']))
        self.assertNotIn('online_judge', report.sections)

    def test_frozen_preflight_verifies_algorithm_once_and_detects_changed_source(self):
        import algorithm_patches
        from rubric.core.contracts import digest, file_hash

        backend = Path(self.cfg.require('backends.root')) / 'warmup'
        (backend / 'verl').mkdir(parents=True)
        (backend / '.gad_backend_used').touch()
        source = backend / 'verl/worker.py'
        source.write_text('value = 1\n')
        (backend / '.gad_algorithm_manifest.json').write_text(json.dumps({
            'patched_files': {'verl/worker.py': file_hash(source)}, 'extension_code_hash': digest({})}))
        with patch('preflight.git_rev_parse', return_value='dummy'), redirect_stdout(io.StringIO()):
            for corrupt in (False, True):
                if corrupt:
                    source.write_text('value = 2\n')
                report = Report()
                with patch.object(algorithm_patches, 'verify', wraps=algorithm_patches.verify) as verify:
                    entry = check_backends(self.cfg, report, ['warmup'])['warmup']
                verify.assert_called_once_with(backend)
                self.assertEqual(corrupt, bool(report.failures))
                self.assertEqual(not corrupt, 'algorithm' in entry)
            with patch('backend_patches.verify_frozen_backend', return_value={'stage': 'warmup'}):
                report = Report()
                check_backends(self.cfg, report, ['warmup'])
                self.assertIn('no verified algorithm manifest', report.failures[0])

    def test_all_stage_plans_and_signatures_exclude_judge_execution(self):
        with patch('extension_config.service_identity', side_effect=AssertionError('Judge identity requested')):
            for stage in ('seqkd', 'warmup', 'gad'):
                with self.subTest(stage=stage):
                    handoff = {'actor_hf': str(self.folder / 'handoff-actor'),
                               'critic_hf': str(self.folder / 'handoff-critic'), 'step': 271}
                    plan = plan_stage(self.cfg, self.folder / 'run', stage,
                                      steps_per_epoch_value=271,
                                      handoff=handoff if stage == 'gad' else None,
                                      install_guard=False)
                    self.assertFalse(plan['rubric_enabled'])
                    self.assertEqual('adv_only', plan['actor_reward']['source'])
                    args = dict(arg.split('=', 1) for arg in plan['argv'][3:] if '=' in arg)
                    if stage != 'seqkd':
                        self.assertEqual('{enabled:false}', args['trainer.rubric'])
                        self.assertEqual('{enabled:false}', args['critic.rubric'])
                    self.assertNotIn('missing-Qwen3-8B', ' '.join(plan['argv']))
                    signature = stage_signature(self.cfg, stage)
                    self.assertNotIn('judge_fingerprint', signature['payload'].get('algorithm', {}))

    def test_hydra_resolves_one_code_hash_per_plan(self):
        first, second = 'first-code-hash', 'second-code-hash'
        with patch('extension_config.deployed_code_identity', side_effect=[
            {'extension_code_hash': first}, {'extension_code_hash': second},
        ]) as identify:
            args = hydra_overrides(self.cfg, 'gad', self.folder, 0)
            identify.assert_called_once_with(self.cfg, 'gad')
            self.assertIn(f'critic.extensions_code_hash={first}', args)
            next_args = hydra_overrides(self.cfg, 'gad', self.folder, 0)
            self.assertIn(f'critic.extensions_code_hash={second}', next_args)
            initial_state = next(v for v in args if v.startswith('trainer.gad_extensions='))
            self.assertNotIn(initial_state, next_args)
            self.cfg._recovery_code_hash = first
            self.assertEqual(args, hydra_overrides(self.cfg, 'gad', self.folder, 0))
            self.assertEqual(2, identify.call_count)


if __name__ == '__main__':
    unittest.main()
