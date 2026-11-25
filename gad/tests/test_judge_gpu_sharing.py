"""CPU contracts for opt-in Judge colocation; no model loading or GPU claims."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools'), str(ROOT / 'gad/tests')]
from config import validate_config
from cpu_fixtures import load_cpu_config
from gad.extensions.judge_client import JudgeClient
from pipeline import stage_signature
from preflight import Report, check_online_judge
from rubric.core.contracts import digest
import rubric_judge


FLAG = 'resources.allow_judge_gpu_sharing'


class JudgeGpuSharingTests(unittest.TestCase):
    def setUp(self):
        # These tests exercise allocation and health guards, not installed model/scorer
        # identities. Explicit fixtures keep the tests runnable before GPU dependencies.
        scorer = patch('scoring.math.identity', return_value={'fingerprint': 'cpu-test-scorer'})
        scorer.start()
        self.addCleanup(scorer.stop)
        self.cfg = load_cpu_config(ROOT / 'configs/launch_rubric_naive_n4_06b_17k.yaml')
        self.cfg.data.update({'resources.gpus': '0', 'resources.judge_gpus': '0',
                             'resources.n_gpus_per_node': 1,
                             'rubric.judge.tensor_parallel_size': 1})
        self.identity = {'model': self.cfg.get('rubric.judge.model'), 'protocol': 'direct_v1',
                         'fixture': 'cpu-service-identity'}

    def test_config_requires_explicit_boolean_sharing(self):
        self.cfg.data.pop(FLAG)
        self.assertIs(self.cfg.get(FLAG), False)
        self.assertIn('judge GPUs and training GPUs overlap', validate_config(self.cfg))
        self.cfg.data[FLAG] = True
        self.assertEqual([], validate_config(self.cfg))
        for value in ('true', 'false', 1, 0, None):
            with self.subTest(value=value):
                self.cfg.data[FLAG] = value
                problems = validate_config(self.cfg)
                self.assertIn(FLAG + ' must be boolean', problems)
                self.assertIn('judge GPUs and training GPUs overlap', problems)
        self.cfg.data[FLAG] = False
        self.cfg.data['resources.judge_gpus'] = '1'
        self.assertEqual([], validate_config(self.cfg))

    def launch_plan(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output), \
                patch.object(rubric_judge, 'ROOT', Path(directory)), \
                patch.object(rubric_judge, 'select_config', return_value=self.cfg), \
                patch.object(rubric_judge, 'service_identity', return_value=self.identity), \
                patch.object(sys, 'argv', ['rubric_judge.py', '--action', 'start', '--dry-run']), \
                patch.object(rubric_judge.subprocess, 'run') as run:
            rubric_judge.main()
            run.assert_not_called()
        return json.loads(output.getvalue())

    def test_launcher_requires_explicit_sharing_and_preserves_judge_mode(self):
        for value in (False, 'true', 1):
            with self.subTest(value=value):
                self.cfg.data[FLAG] = value
                with self.assertRaisesRegex(RuntimeError, 'overlap training'):
                    self.launch_plan()
        self.cfg.data[FLAG] = True
        plan = self.launch_plan()
        self.assertEqual('0', plan['gpu_ids'])
        self.assertEqual({'VLLM_USE_V1': '0'}, plan['environment'])
        self.assertIn('--enforce-eager', plan['argv'])
        self.assertEqual('1', plan['argv'][plan['argv'].index('--tensor-parallel-size') + 1])

    def test_restart_waits_for_old_gpu_pid_and_leaves_foreign_clients_running(self):
        owner = {'pid': 41, 'birth': 'starttime'}
        owner_file = Mock()
        with patch.object(rubric_judge, 'live_group', side_effect=[{41, 42}, set(), set()]), \
                patch.object(rubric_judge, 'process_birth', return_value='starttime'), \
                patch.object(rubric_judge.os, 'getpgid', return_value=41), \
                patch.object(Path, 'read_text', return_value='rubric_judge_server.py'), \
                patch.object(rubric_judge, 'gpu_pids', side_effect=[{42}, {42}, set()]) as gpu, \
                patch.object(rubric_judge.os, 'killpg') as kill, patch.object(rubric_judge.time, 'sleep') as sleep:
            rubric_judge.stop_owned(owner, owner_file, restart_gpus='0')
            kill.assert_called_once_with(41, rubric_judge.signal.SIGTERM)
            self.assertEqual([call('0')] * 3, gpu.call_args_list)
            sleep.assert_called_once()
            owner_file.unlink.assert_called_once()

        owner_file.reset_mock()
        with patch.object(rubric_judge, 'live_group', return_value={41, 42}), \
                patch.object(rubric_judge, 'process_birth', return_value='starttime'), \
                patch.object(rubric_judge.os, 'getpgid', return_value=41), \
                patch.object(Path, 'read_text', return_value='rubric_judge_server.py'), \
                patch.object(rubric_judge, 'gpu_pids', return_value={42, 100}), \
                patch.object(rubric_judge.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'Other GPU processes'):
                rubric_judge.stop_owned(owner, owner_file, restart_gpus='0')
            kill.assert_not_called()
            owner_file.unlink.assert_not_called()

    def test_restart_waits_for_verified_service(self):
        from gad.extensions.judge_client import ServiceUnavailable
        child = Mock(pid=41)
        child.poll.return_value = None
        options = {'model': 'judge'}
        with patch('gad.extensions.judge_client.JudgeClient') as make_client, \
                patch.object(rubric_judge.time, 'sleep') as sleep:
            health = make_client.return_value.health
            health.side_effect = [ServiceUnavailable('loading'), {'gpu_ids': '0'}]
            rubric_judge.wait_ready(child, options, 'expected', '0', Path('server.log'))
            self.assertEqual(('expected',), make_client.call_args.args[1:])
            self.assertEqual(2, health.call_count)
            sleep.assert_called_once()
            health.side_effect = [{'gpu_ids': '1'}]
            with self.assertRaisesRegex(RuntimeError, 'allocation differs'):
                rubric_judge.wait_ready(child, options, 'expected', '0', Path('server.log'))

    def preflight(self, gpu_ids='0', model=None, fingerprint=None):
        report = Report()
        responses = {
            '/models': {'data': [{'id': model or self.cfg.get('rubric.judge.model')}]},
            '/gad_fingerprint': {'identity_hash': fingerprint or digest(self.identity),
                                 'gpu_ids': gpu_ids},
        }
        with patch('extension_config.service_identity', return_value=self.identity), \
                patch.object(JudgeClient, 'request', side_effect=lambda suffix: responses[suffix]), \
                redirect_stdout(io.StringIO()):
            check_online_judge(self.cfg, report, ['warmup', 'gad'])
        return report

    def test_preflight_allows_only_declared_sharing_and_keeps_identity_checks(self):
        self.assertIn('running judge uses training GPUs', self.preflight().failures[0])
        self.cfg.data[FLAG] = True
        report = self.preflight()
        self.assertEqual([], report.failures)
        self.assertEqual('0', report.sections['online_judge']['gpu_ids'])
        for kwargs, error in (
                ({'gpu_ids': '1'}, 'GPU allocation differs'),
                ({'gpu_ids': '0,1'}, 'GPU allocation differs'),
                ({'gpu_ids': ''}, 'GPU allocation differs'),
                ({'model': 'different-model'}, 'served model mismatch'),
                ({'fingerprint': 'different-protocol-fingerprint'}, 'fingerprint mismatch')):
            with self.subTest(kwargs=kwargs):
                report = self.preflight(**kwargs)
                self.assertEqual(1, len(report.failures))
                self.assertIn(error, report.failures[0])
                self.assertNotIn('online_judge', report.sections)

    def test_default_sharing_does_not_enter_historical_stage_identity(self):
        # SeqKD has no Judge identity dependency; the resource flag still uses the
        # same no-op-default exclusion as the other additions to frozen configs.
        self.cfg.data.pop(FLAG)
        before = stage_signature(self.cfg, 'seqkd')
        self.cfg.data[FLAG] = False
        after = stage_signature(self.cfg, 'seqkd')
        self.assertEqual(before, after)
        self.assertNotIn(FLAG, after['payload']['training_config'])
        self.cfg.data[FLAG] = True
        self.assertTrue(stage_signature(self.cfg, 'seqkd')['payload']['training_config'][FLAG])


if __name__ == '__main__':
    unittest.main()
