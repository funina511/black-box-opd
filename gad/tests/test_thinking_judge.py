"""Independent judge mode, output separation, actual v0 reasoner and finite budgets."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools'), str(ROOT / 'scoring/tests')]
from config import compose_config, validate_config
from extension_config import service_identity
from rubric.core.contracts import digest, judge_messages, file_hash
from test_unified import event_record
from gad.extensions.judge_client import JudgeClient, guidance_schema, guidance_backend
from validate_thinking_judge import remaining_budget, completion_partition, merge_completion


class ThinkingTests(unittest.TestCase):
    def frozen_completion(self):
        source = ROOT / 'reports/rubric_v2/thinking_judge_20261015'
        canonical = [json.loads(s) for s in (source / 'selected24.jsonl').read_text().splitlines()]
        previous = [json.loads(s) for s in (source / '14b_compat_retry/samples.jsonl').read_text().splitlines()]
        return canonical, previous

    def test_completion_reuses_all_six_even_masked_and_preserves_order(self):
        canonical, previous = self.frozen_completion()
        reused, pending = completion_partition(canonical, previous)
        self.assertEqual((6, 18), (len(reused), len(pending)))
        self.assertTrue(any(not row['result']['parsed']['verified_progress']['valid'] for row in reused))
        new = [{**row, 'result': {'error': 'judge_output_truncated'}} for row in pending]
        combined = merge_completion(canonical, reused, new)
        self.assertEqual([r['id'] for r in canonical], [r['id'] for r in combined])
        by_id = {r['id']: r for r in combined}
        for row in reused:self.assertEqual(row['result'], by_id[row['id']]['result'])
        self.assertEqual(6, sum(r['result_origin'] == 'previous_thinking' for r in combined))

    def test_completion_rejects_changed_inputs_duplicate_results_and_unchanged_retries(self):
        canonical, previous = self.frozen_completion()
        wrong = deepcopy(previous);wrong[0]['response'] += 'changed'
        with self.assertRaises(ValueError):completion_partition(canonical, wrong)
        reused, pending = completion_partition(canonical, previous)
        with self.assertRaises(ValueError):merge_completion(canonical, reused, [reused[0]])
        wrong = deepcopy(previous)
        next(r for r in wrong if 'parsed' not in r['result'])['result'] = {'error': 'judge_transport_error'}
        with self.assertRaises(ValueError):completion_partition(canonical, wrong)

    def client(self, thinking=True, context=100):
        class Tokenizer:
            def apply_chat_template(self, *a, **kw):
                self.kwargs = kw
                return [0] * (20 if kw['enable_thinking'] else 10)
            def encode(self, text, **kw):return list(text)
        options = {'model': 'judge', 'enable_thinking': thinking,
            'reasoning_parser': 'deepseek_r1' if thinking else None,
            'temperature': .6, 'top_p': .95, 'top_k': 20, 'min_p': 0,
            'max_output_tokens': 30, 'max_context_tokens': context, 'max_retries': 1,
            'request_timeout_seconds': 3, 'rubric_version': 'math_process_v2'}
        return JudgeClient(options, 'fp', Tokenizer())

    def run_one(self, client, content=None, reasoning='audit, not trajectory evidence', finish='stop', usage=None):
        calls = []
        def request(suffix, payload, timeout):
            calls.append(payload)
            return {'choices': [{'finish_reason': finish, 'message': {'reasoning_content': reasoning,
                'content': json.dumps(event_record()) if content is None and finish == 'stop' else content}}],
                'usage': usage or {'completion_tokens': 25}}
        client.request = request
        messages = judge_messages('p', '2x=4', '2', 'spec', 'prompt', 'math_process_v2')
        result = client.annotate_one('r', messages, time.monotonic() + 3)[1]
        return result, calls

    def test_request_template_and_final_only_parser(self):
        client = self.client()
        result, calls = self.run_one(client, reasoning='{"fake":"not valid evidence"}')
        self.assertIn('parsed', result)
        self.assertEqual({'enable_thinking': True}, calls[0]['chat_template_kwargs'])
        for key, expected in [('temperature', .6), ('top_p', .95), ('top_k', 20), ('min_p', 0)]:
            self.assertEqual(expected, calls[0][key])
        self.assertTrue(client.tokenizer.kwargs['enable_thinking'])
        self.assertNotIn('fake', json.dumps(result['parsed']))
        self.assertIn('fake', result['reasoning_output'])
        self.assertEqual('local_retokenization_estimate', result['output_lengths']['reasoning_token_source'])

    def test_budget_exhaustion_and_absent_reasoning_do_not_silently_pass(self):
        result, calls = self.run_one(self.client(), content=None, finish='length')
        self.assertEqual('judge_reasoning_budget_exhausted', result['error'])
        self.assertEqual(1, len(calls));self.assertNotIn('parsed', result)
        self.assertEqual(25, result['usage']['completion_tokens'])
        result, calls = self.run_one(self.client(), reasoning=None)
        self.assertEqual('judge_thinking_not_observed', result['error'])
        self.assertEqual(1, len(calls))
        result, _ = self.run_one(self.client(), content='{', finish='length')
        self.assertEqual('judge_output_truncated', result['error'])

    def test_exact_capacity_uses_judge_mode(self):
        result, calls = self.run_one(self.client(context=49))
        self.assertEqual('judge_context_exceeded', result['error']);self.assertEqual([], calls)
        result, _ = self.run_one(self.client(context=50))
        self.assertIn('parsed', result)
        result, _ = self.run_one(self.client(thinking=False, context=40), reasoning=None)
        self.assertIn('parsed', result)

    def test_provider_total_and_optional_split_usage(self):
        result, _ = self.run_one(self.client(), usage={'completion_tokens': 25,
            'completion_tokens_details': {'reasoning_tokens': 17}})
        lengths = result['output_lengths']
        self.assertEqual(25, lengths['total_tokens']);self.assertEqual(17, lengths['reasoning_tokens'])
        self.assertEqual('api.completion_tokens_details.reasoning_tokens', lengths['reasoning_token_source'])
        self.assertTrue(lengths['split_counts_are_not_additive'])

    def candidate(self):
        return compose_config(model='qwen3_0.6b', dataset='dapo_math_5k', overrides=[
            str(ROOT / 'configs/overrides/rubric_v2_joint.yaml'),
            str(ROOT / 'configs/overrides/rubric_thinking_14b_candidate.yaml')])

    def test_config_identity_and_actor_independence(self):
        cfg = self.candidate()
        self.assertFalse(cfg.get('rollout.enable_thinking'));self.assertTrue(cfg.get('rubric.judge.enable_thinking'))
        self.assertEqual([], validate_config(cfg))
        baseline = digest(service_identity(cfg))
        for key, value in [('enable_thinking', False), ('reasoning_parser', None), ('temperature', .7),
                           ('top_p', .8), ('top_k', 30), ('min_p', .1), ('max_output_tokens', 8191)]:
            other = deepcopy(cfg);other.data['rubric.judge.' + key] = value
            self.assertNotEqual(baseline, digest(service_identity(other)), key)
        for key, value in [('enable_thinking', 'true'), ('reasoning_parser', 'qwen3'), ('temperature', 0),
                           ('top_p', 0), ('top_k', 0), ('min_p', -1)]:
            other = deepcopy(cfg);other.data['rubric.judge.' + key] = value
            self.assertTrue(validate_config(other), key)

    def test_shared_budget_does_not_reset_and_caps_applications(self):
        ledger = {'started_epoch': 1000, 'actual_applications': 24}
        self.assertEqual(600, remaining_budget(ledger, 24, now=2200))
        with self.assertRaises(TimeoutError):remaining_budget(ledger, 24, now=2780)
        with self.assertRaises(ValueError):remaining_budget(ledger, 25, now=2200)

    def test_thinking_models_have_separate_cache_directories(self):
        from gad.extensions.online_rubric import OnlineRubric
        cfg = self.candidate()
        options = {k[7:]: v for k, v in cfg.data.items() if k.startswith('rubric.')}
        options['judge'] = {k[6:]: options.pop(k) for k in list(options) if k.startswith('judge.')}
        options.update(spec_hash=file_hash(options['spec']), judge_fingerprint='14b-fingerprint',
            scorer_path=cfg.require('validation.scorer_path'),
            scoring_math={k[13:]: v for k, v in cfg.data.items() if k.startswith('scoring.math.')})
        with tempfile.TemporaryDirectory() as d, patch('gad.extensions.online_rubric.JudgeClient',
                return_value=SimpleNamespace(health=lambda: None)):
            one = OnlineRubric(options, Path(d) / 'warmup')
            other = deepcopy(options);other['judge_fingerprint'] = '8b-fingerprint'
            two = OnlineRubric(other, Path(d) / 'gad')
            self.assertNotEqual(one.cache.directory, two.cache.directory)
            self.assertIn('_thinking_', one.cache.directory.name)

    def test_unstarted_boundary_is_not_a_mathematical_mismatch(self):
        from calibrate_rubric_judge import benchmark_summary
        sample = {'id': 'not_started', 'kind': 'boundary', 'category': 'false_repair',
            'expected_allowed': {'verified_progress': [.5], 'unresolved_error_severity': [1]},
            'result': {'error': 'not_run_interrupted'}}
        summary = benchmark_summary([sample])
        self.assertEqual(1, summary['boundary_not_evaluable_count'])
        self.assertEqual(0, summary['boundary_raw_anchor_mismatch_count'])
        self.assertEqual(0, summary['overall']['attempted'])

    def test_failed_startup_preserves_partial_artifacts(self):
        import subprocess
        import yaml
        import calibrate_rubric_judge
        cfg = self.candidate()
        cfg.data['rubric.judge.base_url'] = 'http://127.0.0.1:0/v1'
        def fail_start(cmd, **kwargs):
            if cmd[0] == 'nvidia-smi':return SimpleNamespace(stdout='')
            raise subprocess.TimeoutExpired(cmd, .01)
        with tempfile.TemporaryDirectory() as d:
            config = Path(d) / 'config.yaml';config.write_text(yaml.safe_dump(cfg.to_dict()))
            output = Path(d) / 'output'
            argv = ['calibrate_rubric_judge.py', '--config', str(config), '--output', str(output),
                '--samples', str(ROOT / 'reports/rubric_v2/thinking_judge_20261015/selected24.jsonl'), '--wall-seconds', '30']
            with patch.object(sys, 'argv', argv), patch('calibrate_rubric_judge.subprocess.run', side_effect=fail_start):
                with self.assertRaises(subprocess.TimeoutExpired):calibrate_rubric_judge.main()
            summary = json.loads((output / 'summary.partial.json').read_text())
            self.assertEqual(0, summary['overall']['attempted'])
            self.assertEqual(8, summary['boundary_not_evaluable_count'])
            self.assertEqual('failed', json.loads((output / 'budget.json').read_text())['status'])

    def test_compat_guidance_keeps_full_parser_and_compiles(self):
        import xgrammar as xgr
        from transformers import AutoConfig, AutoTokenizer
        from rubric.core.process_v2 import JUDGE_SCHEMA_V2, parse_events, ProtocolError
        from vllm.model_executor.guided_decoding.utils import has_xgrammar_unsupported_json_features
        full = deepcopy(JUDGE_SCHEMA_V2)
        hint = guidance_schema('math_process_v2')
        self.assertEqual(full, JUDGE_SCHEMA_V2)
        self.assertFalse(has_xgrammar_unsupported_json_features(hint))
        self.assertEqual('xgrammar:no-fallback', guidance_backend('math_process_v2'))
        row = event_record();row['events']['e1']['supporting_span_ranges'] = {'r1': {'start': '', 'end': ''}}
        self.assertFalse(parse_events(json.dumps(row), '2x=4')['verified_progress']['valid'])
        row = event_record();row['verified_progress']['reason'] = ''
        with self.assertRaises(ProtocolError):parse_events(json.dumps(row), '2x=4')
        path = ROOT.parent / 'model/Qwen/Qwen3-14B'
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=AutoConfig.from_pretrained(path).vocab_size)
        compiled = xgr.GrammarCompiler(info, max_threads=2).compile_json_schema(json.dumps(hint), any_whitespace=True)
        self.assertIsNotNone(compiled)
        from rubric.core.semantic_v24 import SCHEMA
        targeted=guidance_schema('math_process_v2',SCHEMA)
        self.assertFalse(has_xgrammar_unsupported_json_features(targeted))
        self.assertIsNotNone(xgr.GrammarCompiler(info,max_threads=2).compile_json_schema(json.dumps(targeted),any_whitespace=True))

    def test_actual_installed_parser_and_grammar_gate(self):
        # CPU only: exercise the installed v0 parser/logits gate, not a copied implementation.
        import torch
        from transformers import AutoTokenizer
        from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
        from vllm.model_executor.guided_decoding.xgrammar_decoding import XGrammarLogitsProcessor
        tokenizer = AutoTokenizer.from_pretrained(ROOT.parent / 'model/Qwen/Qwen3-14B', local_files_only=True)
        parser = DeepSeekR1ReasoningParser(tokenizer)
        reasoning, final = parser.extract_reasoning_content('check original constraints</think>{"ok":true}', None)
        self.assertEqual('check original constraints', reasoning);self.assertEqual('{"ok":true}', final)
        processor = XGrammarLogitsProcessor.__new__(XGrammarLogitsProcessor)
        processor.reasoner, processor.ctx = parser, None
        scores = torch.zeros(8)
        self.assertIs(scores, processor([42], scores))
        with patch.object(processor, '_ensure_ctx', side_effect=RuntimeError('final grammar activated')):
            with self.assertRaisesRegex(RuntimeError, 'final grammar activated'):
                processor([42, parser.end_token_id], scores)
        # v0 routes this unchanged schema to Outlines because it contains minLength.
        # Check the actual fallback and its independent reasoning gate as well.
        import sqlite3
        try:
            from vllm.model_executor.guided_decoding import maybe_backend_fallback
            from vllm.model_executor.guided_decoding.outlines_logits_processors import BaseLogitsProcessor
        except sqlite3.OperationalError as exc:
            if 'readonly' in str(exc).lower() or 'unable to open' in str(exc).lower():
                self.skipTest('optional Outlines fallback cache is not writable: '+str(exc))
            raise
        from vllm.sampling_params import GuidedDecodingParams
        from rubric.core.process_v2 import JUDGE_SCHEMA_V2
        selected = maybe_backend_fallback(GuidedDecodingParams(json=JUDGE_SCHEMA_V2, backend='xgrammar'))
        self.assertEqual('outlines', selected.backend)
        def final_instruction(**kwargs):raise RuntimeError('outlines final grammar activated')
        processor = BaseLogitsProcessor(SimpleNamespace(get_next_instruction=final_instruction), parser)
        self.assertIs(scores, processor([42], scores))
        with self.assertRaisesRegex(RuntimeError, 'outlines final grammar activated'):
            processor([42, parser.end_token_id], scores)


if __name__ == '__main__':unittest.main()
