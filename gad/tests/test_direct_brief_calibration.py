"""Calibrate both non-thinking contracts through their live request path, on CPU."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools')]
import calibrate_rubric_judge as calibration
from cpu_fixtures import compose_cpu_config
import score_rubric_judge_reviews as scoring


class PairedCalibrationTests(unittest.TestCase):
    def test_matched_student_deltas_weight_questions_and_report_common_denominator(self):
        p, e = scoring.DIMENSIONS
        def sample(uid, text, correct, progress, error):
            return {'uid': uid, 'kind': 'real', 'source_role': 'student', 'truncated': False, 'response': text,
                    'scoring': {'correct': correct, 'grade_status': 'correct' if correct else 'incorrect',
                                'parse_status': 'ok'},
                    'result': {'parsed': {p: {'value': progress, 'valid': progress is not None},
                                          e: {'value': error, 'valid': error is not None}}}}
        rows = {
            'a1': sample('a', 'first', True, 1, 0), 'a2': sample('a', 'second', True, .5, 0),
            'a3': sample('a', 'wrong', False, .5, 1), 'a4': sample('a', 'first', True, 0, 1),
            'b1': sample('b', 'correct', True, 0, 1), 'b2': sample('b', 'wrong', False, .5, 0),
            'c1': sample('c', 'correct', True, None, .5), 'c2': sample('c', 'wrong', False, 1, .5),
            'old': sample(None, 'correct', True, 1, 0),
            'skip1': sample('d', 'correct', True, 1, 0), 'skip2': sample('d', 'wrong', False, 0, 1),
        }
        del rows['skip2']['scoring']['parse_status']
        other = {**rows, 'b1': {**rows['b1'], 'result': {'error': 'timeout'}}}
        report = scoring.within_question_student(rows, other)
        self.assertEqual(3, report['eligible_questions'])
        self.assertEqual(['a1', 'a2'], report['response_ids_by_question']['a']['correct'])
        progress = report['dimensions'][p]
        self.assertEqual({'questions': 3, 'valid': 2, 'abstained': 1, 'failed': 0,
                          'positive': 1, 'tie': 0, 'negative': 1, 'mean_delta': -.125}, progress['benchmark'])
        self.assertEqual(1, progress['comparison']['failed'])
        self.assertEqual(1, progress['both_arms_known']['questions'])
        self.assertEqual(.25, progress['both_arms_known']['benchmark']['mean_delta'])
        error = report['dimensions'][e]
        self.assertEqual(1, error['benchmark']['negative'])
        self.assertEqual(1, error['benchmark']['tie'])
        self.assertEqual(2, error['both_arms_known']['questions'])
        self.assertEqual(.5, error['both_arms_known']['comparison']['mean_delta'])
        self.assertIsNone(scoring.within_question_student({'old': rows['old']})['dimensions'][p]['benchmark']['mean_delta'])

    def test_brief_and_direct_use_the_training_contract_and_same_budget(self):
        progress, error = scoring.DIMENSIONS
        data = [
            dict(id=str(i), kind='real', prompt='Question', response=str(i), ground_truth='1', category='blind-stratum',
                 source_role=role, truncated=truncated, scoring={'correct': correct})
            for i, (role, truncated, correct) in enumerate((
                ('teacher', False, True), ('student', False, False), ('student', True, None)))
        ]
        calls = []

        class FakeServiceClient(calibration.JudgeClient):
            def __init__(self, options, identity):
                tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1])
                super().__init__(options, identity, tokenizer)

            def request(self, suffix, payload=None, timeout=None):
                if suffix == '/models':
                    return {'data': [{'id': self.options['model']}]}
                if suffix == '/gad_fingerprint':
                    return {'identity_hash': self.expected}
                calls.append(payload)
                self_id = json.loads(payload['messages'][-1]['content'])['response']
                fields = list(payload['guided_json']['properties'])
                row = {}
                if fields[0] == 'brief_check':
                    row['brief_check'] = '' if self_id == '2' else 'The second equation reverses the sign.'
                row.update({progress: .5, error: 1 if self_id == '0' else .5 if self_id == '1' else None})
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(row)}}]}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def subprocess_run(cmd, **kwargs):
                if cmd[0] == 'nvidia-smi':
                    return SimpleNamespace(stdout='')
                owner = root / 'runs/_rubric_judge/0/owner.json'
                owner.parent.mkdir(parents=True, exist_ok=True)
                owner.write_text(json.dumps({'pid': os.getpid(), 'birth': 'mock-service'}))
                return SimpleNamespace(returncode=0)

            samples = root / 'pool.jsonl'
            samples.write_text(''.join(json.dumps(row) + '\n' for row in data))
            for protocol, template, expected in (
                ('direct_v1', 'direct_v1_instruct', [progress, error]),
                ('direct_brief_v1', 'direct_brief_v1', ['brief_check', progress, error]),
            ):
                cfg = compose_cpu_config(overrides=[ROOT / 'configs/methods/rubric_joint.yaml'])
                cfg.data.update({
                    'rubric.process_protocol': protocol, 'rubric.semantic_check': False,
                    'rubric.prompt': str(ROOT / f'rubric/prompts/judge_math_process_{template}.txt'),
                    'rubric.spec': str(ROOT / f'rubric/specs/math_process_{protocol}.yaml'),
                    'rubric.judge.base_url': 'http://127.0.0.1:0/v1',
                    'rubric.judge.model': 'Qwen3-30B-A3B-Instruct-2507',
                    'rubric.judge.max_output_tokens': 128, 'rubric.judge.enable_thinking': False,
                    'rubric.judge.reasoning_parser': None,
                })
                config = root / (protocol + '.yaml')
                config.write_text(yaml.safe_dump(cfg.to_dict()))
                out = root / protocol
                argv = ['calibrate', '--config', str(config), '--output', str(out), '--samples', str(samples),
                        '--max-requests', '3', '--max-output-tokens', '768', '--wall-seconds', '7200', '--finish-cohort']
                calls.clear()
                with patch.object(sys, 'argv', argv), patch.object(calibration, 'ROOT', root), \
                     patch.object(calibration, 'JudgeClient', FakeServiceClient), \
                     patch.object(calibration.subprocess, 'run', side_effect=subprocess_run), \
                     patch.object(calibration, 'stop_owned_service', return_value={'stopped': True}), \
                     redirect_stdout(io.StringIO()):
                    calibration.main()
                (root / 'runs/_rubric_judge/0/owner.json').unlink()
                self.assertEqual(len(data), len(calls))
                for payload in calls:
                    self.assertEqual(expected, list(payload['guided_json']['properties']))
                    self.assertEqual(768, payload['max_tokens'])
                    self.assertFalse(payload['chat_template_kwargs']['enable_thinking'])
                    visible = json.loads(payload['messages'][-1]['content'])
                    self.assertEqual({'problem', 'response', 'reference_final_answer', 'visibility'}, set(visible))
                    self.assertNotIn('blind-stratum', json.dumps(payload['messages']))
                metadata = json.loads((out / 'protocol.json').read_text())
                self.assertEqual(protocol, metadata['process_protocol'])
                self.assertEqual(768, metadata['request_output_cap_override'])
                self.assertEqual(128, metadata['service_identity']['request_defaults']['max_output_tokens'])

            direct, brief = (scoring.load_samples(root / protocol) for protocol in ('direct_v1', 'direct_brief_v1'))
            self.assertEqual('direct_brief_v1', brief['1']['result']['parsed']['_process_protocol'])
            self.assertIn('second equation', brief['1']['result']['parsed']['brief_check'])
            self.assertEqual('technical', brief['2']['result']['failure_kind'])
            self.assertEqual(2, scoring.compare_arms(direct, brief)[error]['mutually_known'])
            summary = json.loads((root / 'direct_brief_v1/summary.json').read_text())
            self.assertEqual(2, summary['overall']['parsed'])
            self.assertEqual({'1': 1, '0.5': 1}, summary['overall']['raw_labels'][error])
            self.assertEqual({'correct': ['0'], 'incorrect': ['1'], 'undetermined': ['2']},
                             scoring.real_strata(brief)['verified_answer'])
            self.assertEqual(['0', '1', '2'], scoring.real_strata(brief)['category']['blind-stratum'])
            reviews = [({progress: {'valid': True, 'value': .5}, error: {'valid': True, 'value': target}}, brief[rid])
                       for rid, target in (('0', 0), ('1', 1), ('2', 1))]
            measured = scoring.human_metrics(reviews)
            for mode in ('raw', 'effective'):
                self.assertEqual(1., measured[mode][error]['false_severe_rate_on_human_nonsevere'])
                self.assertEqual(0., measured[mode][error]['recall_on_human_severe'])
                self.assertEqual(2, measured[mode][error]['severe_error_counts']['reference_severe'])
            self.assertEqual(1, measured['effective'][error]['unusable_responses_on_human_known'])


if __name__ == '__main__':
    unittest.main()
