"""Exercise the brief wire format and its Teacher/Student training path on CPU."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools')]
from test_direct_judge import cfg
from test_process_sampling_direct import pairs
from gad.extensions.online_rubric import OnlineRubric
from gad.extensions.process_client import JudgeClient, guidance_schema
from rubric.core import process_brief
from rubric.core.contracts import file_hash
from rubric.core.protocols import parse_judge
from extension_config import service_identity

ROW = {'brief_check': 'The visible substitution gives x=2.', 'verified_progress': .5, 'unresolved_error_severity': None}


class BriefJudge(unittest.TestCase):
    def test_parser_and_deployed_xgrammar_field_order(self):
        from xgrammar import Grammar
        from xgrammar.testing import _is_grammar_accept_string
        schema = guidance_schema('math_process_v2', process_protocol=process_brief.PROTOCOL)
        grammar = Grammar.from_json_schema(json.dumps(schema), any_whitespace=True)
        reordered = {'verified_progress': .5, **ROW}
        self.assertTrue(_is_grammar_accept_string(grammar, json.dumps(ROW)))
        self.assertFalse(_is_grammar_accept_string(grammar, json.dumps(reordered)))
        for raw in (json.dumps(reordered), json.dumps(dict(ROW, brief_check='  ')), json.dumps(dict(ROW, verified_progress=True)),
                    json.dumps(dict(ROW, extra=2)), json.dumps(dict(ROW, brief_check=[])), 'null',
                    '{"brief_check":"a","brief_check":"b","verified_progress":0,"unresolved_error_severity":1}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_judge(raw, 'math_process_v2', process_protocol=process_brief.PROTOCOL)

    def test_guided_request_cache_journal_and_training_targets(self):
        c = cfg()
        legacy_identity = service_identity(c)
        c.data.update({'rubric.process_protocol': process_brief.PROTOCOL,
                       'rubric.spec': str(ROOT / 'rubric/specs/math_process_direct_brief_v1.yaml'),
                       'rubric.prompt': str(ROOT / 'rubric/prompts/judge_math_process_direct_brief_v1.txt'),
                       'rubric.judge.enable_thinking': False, 'rubric.judge.reasoning_parser': None})
        self.assertNotEqual(legacy_identity, service_identity(c))
        options = {k[7:]: v for k,v in c.data.items() if k.startswith('rubric.')}
        for group in ('judge', 'process_sampling'):
            prefix = group + '.'
            options[group] = {k[len(prefix):]: options.pop(k) for k in list(options) if k.startswith(prefix)}
        options.update(spec_hash=file_hash(options['spec']), scorer_path=c.require('validation.scorer_path'),
                       judge_fingerprint='brief-test', scoring_math={k[13:]: v for k,v in c.data.items() if k.startswith('scoring.math.')})
        calls = []
        class LocalClient(JudgeClient):
            def __init__(self, options, expected):
                super().__init__(options, expected, tokenizer=SimpleNamespace(apply_chat_template=lambda *a, **k: [42]))

            def health(self, **kwargs):
                return {}

            def request(self, suffix, payload, timeout):
                calls.append(payload)
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(ROW)}}]}

        with tempfile.TemporaryDirectory() as tmp, patch('gad.extensions.online_rubric.JudgeClient', LocalClient):
            owner = OnlineRubric(options, Path(tmp) / 'warmup')
            trajectories = pairs(32)
            batch = SimpleNamespace(batch={})
            owner.annotate(trajectories, batch, 32, 4, 1)
            self.assertEqual(len(calls), 16)
            self.assertEqual(calls[0]['guided_json']['required'], list(process_brief.FIELDS))
            self.assertFalse(calls[0]['chat_template_kwargs']['enable_thinking'])
            for role in ('teacher', 'student'):
                labels = [t['labels'][role] for t in trajectories if t['labels'][role].get('brief_check')]
                self.assertTrue(labels)
                for label in labels:
                    self.assertEqual(label['schema_version'], process_brief.SCHEMA_VERSION)
                    self.assertEqual(label['brief_check'], ROW['brief_check'])
                    self.assertEqual(set(label['raw_labels']), set(process_brief.FIELDS[1:]))
                targets, mask = batch.batch[role + '_rubric_targets'], batch.batch[role + '_rubric_mask']
                self.assertEqual(targets.shape, (32, 3))
                self.assertEqual(mask[:, 0].sum(), 32)
                self.assertTrue((targets[mask[:, 1].bool(), 1] == .5).all())
                self.assertEqual(mask[:, 2].sum(), 0)
            owner.annotate(trajectories, SimpleNamespace(batch={}), 32, 4, 1)
            self.assertEqual(len(calls), 16)


if __name__ == '__main__':
    unittest.main()
