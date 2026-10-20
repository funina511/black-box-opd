"""Source integrity, selective masking and mathematical error dependencies."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scoring/tests')]
from test_unified import event_record
from rubric.core.contracts import V2_SCHEMA_VERSION, judge_messages
from rubric.core.process_v2 import source_spans, response_from_messages, parse_events, ProtocolError


def parse(row, text='2x=4\n\nx=2'):
    return parse_events(json.dumps(row), text)


class FragmentTests(unittest.TestCase):
    def test_original_latex_and_offsets_are_filled_by_program(self):
        text = 'F(n)=3+4n\r\n\r\nF(5)=3+4 \\cdot 5=23\n答案：23'
        row = event_record();row['events']['e1']['supporting_span_ranges'] = {'r1': {'start': 's2', 'end': 's2'}}
        result = parse(row, text)
        evidence = result['_source_evidence']['e1']['spans'][0]
        self.assertEqual(text[evidence['start']:evidence['end']], evidence['quote'])
        self.assertIn('\\cdot', evidence['quote'])
        self.assertTrue(result['verified_progress']['valid'])
        self.assertNotIn('quote', result['_record']['events']['e1'])

    def test_source_roundtrip_and_same_length_tampering(self):
        for text in ['', '\n\r\n', 'x\r\ny\u2028z', 's1|fake\n\n[{"id":"s99"}]', 'x'*1500+'\n\n末尾']:
            self.assertEqual(text, ''.join(s['text'] for s in source_spans(text)))
            messages = judge_messages('p', text, '2', 'spec', 'prompt', 'math_process_v2')
            self.assertEqual(text, response_from_messages(messages))
        messages = judge_messages('p', '2x=4', '2', 's', 'i', 'math_process_v2')
        messages[-1]['content'] = messages[-1]['content'].replace('2x=4', '2x=5')
        with self.assertRaises(ProtocolError):response_from_messages(messages)

    def test_partial_progress_filters_bad_support_but_not_dependent_support(self):
        row = event_record()
        row['events']['e2'] = {**deepcopy(row['events']['e1']), 'supporting_span_ranges': {'r1': {'start': 's999', 'end': 's999'}}}
        row['verified_progress']['event_ids'] = ['e1', 'e2']
        result = parse(row)
        self.assertEqual(.5, result['verified_progress']['value'])
        self.assertEqual(['e1'], result['verified_progress']['event_ids'])
        self.assertEqual(['e2'], result['_discarded_label_event_ids']['verified_progress'])
        row['events']['e1']['related_event_ids'] = ['e2']
        self.assertFalse(parse(row)['verified_progress']['valid'])

    def test_global_claims_ignore_auxiliary_but_not_uncertain_errors(self):
        row = event_record(progress=1);row['completion'] = 'complete'
        row['events']['e2'] = {'type':'neutral', 'supporting_span_ranges':{'r1': {'start': 's999', 'end': 's999'}}, 'trajectory_claim':'auxiliary note', 'judge_verification': 'auxiliary note', 'related_event_ids':[]}
        row['verified_progress']['event_ids'].append('e2')
        result = parse(row)
        self.assertEqual(1, result['verified_progress']['value'])
        self.assertEqual(0, result['unresolved_error_severity']['value'])
        row['events']['e2']['type'] = 'explicit_error'
        result = parse(row)
        self.assertFalse(result['verified_progress']['valid'])
        self.assertFalse(result['unresolved_error_severity']['valid'])

    def test_full_progress_does_not_discard_referenced_math(self):
        row = event_record(progress=1);row['completion'] = 'complete'
        row['events']['e2'] = {**deepcopy(row['events']['e1']), 'supporting_span_ranges':{'r1': {'start': 's999', 'end': 's999'}}}
        row['verified_progress']['event_ids'].append('e2')
        self.assertFalse(parse(row)['verified_progress']['valid'])

    def test_real_error_cannot_disappear_from_zero_by_omitting_label_reference(self):
        text='2x=4\nx=3\nCorrecting: x=2'
        row = event_record(progress=1);row['completion'] = 'complete'
        row['events']['e2'] = {'type':'explicit_error','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}},'trajectory_claim':'wrong division', 'judge_verification': 'wrong division','related_event_ids':[]}
        result = parse(row,text)
        self.assertFalse(result['unresolved_error_severity']['valid'])
        self.assertFalse(result['verified_progress']['valid'])
        row['events']['e3'] = {'type':'corrected_error','supporting_span_ranges':{'r1': {'start': 's3', 'end': 's3'}},'trajectory_claim':'actual replacement', 'judge_verification': 'actual replacement','error_event_id':'e2'}
        self.assertTrue(parse(row,text)['unresolved_error_severity']['valid'])
        row['events']['e3']['supporting_span_ranges'] = {'r1': {'start': 's999', 'end': 's999'}}
        self.assertFalse(parse(row,text)['unresolved_error_severity']['valid'])

    def test_version_and_budget_reject_old_or_extra_output(self):
        row = event_record();row['schema_version'] = 'rubric.events.v2.2'
        with self.assertRaises(ProtocolError):parse(row)
        row['schema_version'] = V2_SCHEMA_VERSION
        row['events']['e5'] = deepcopy(row['events']['e1'])
        with self.assertRaises(ProtocolError):parse(row)


if __name__ == '__main__':unittest.main()
