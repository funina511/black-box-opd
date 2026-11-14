"""Confirmed protocol failures and preserved mathematical-unknown semantics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'gad/tools'), str(ROOT/'scoring/tests'), str(ROOT/'gad/tests')]
from test_unified import event_record
from test_rubric_v24 import review, event_decision
from rubric.core.process_v2 import parse_events, ProtocolError
from rubric.core.semantic_v24 import apply_review
from config import compose_config, validate_config
from rubric.core.contracts import file_hash


def parse(row):
    return parse_events(json.dumps(row), '2x=4\nx=2')

from cpu_fixtures import compose_cpu_config as compose_config


class ProtocolRegressionTests(unittest.TestCase):
    def test_invalid_unjudgeable_still_blocks_global_labels(self):
        for evidence in ({}, {'r1': {'start':'s999', 'end':'s999'}}):
            row=event_record(progress=1);row['completion']='complete'
            row['events']['e2']={**deepcopy(row['events']['e1']), 'type':'unjudgeable', 'supporting_span_ranges':evidence}
            result=parse(row)
            self.assertFalse(result['verified_progress']['valid'])
            self.assertFalse(result['unresolved_error_severity']['valid'])
            self.assertIn('e2', result['_unknown_math_events'])
        row['events']['e2']['type']='neutral'
        self.assertTrue(parse(row)['verified_progress']['valid'])

    def test_review_can_change_label_reference_to_another_reviewed_event(self):
        row=event_record();row['events']['e2']=deepcopy(row['events']['e1'])
        row['events']['e1']['supporting_span_ranges']={'r1':{'start':'s999','end':'s999'}}
        r=review();r['events']={'e2':event_decision()}
        r['labels']['verified_progress']['event_ids']=['e2']
        result=apply_review(parse(row),'2x=4\nx=2',r)
        self.assertTrue(result['verified_progress']['valid'])
        self.assertEqual(['e2'], result['verified_progress']['event_ids'])
        self.assertEqual(['e1'], result['_initial_record']['verified_progress']['event_ids'])

    def test_math_unknown_uses_empty_streak_instead_of_immediate_contract_stop(self):
        from gad.extensions.online_rubric import OnlineRubric
        from gad.extensions.online_v2 import StudentProcessSupervisionError
        cfg=compose_config();options={k[7:]:v for k,v in cfg.data.items() if k.startswith('rubric.')}
        options['judge']={k[6:]:options.pop(k) for k in list(options) if k.startswith('judge.')}
        options.update(spec_hash=file_hash(options['spec']),judge_fingerprint='test',
            scorer_path=cfg.require('validation.scorer_path'),scoring_math={k[13:]:v for k,v in cfg.data.items() if k.startswith('scoring.math.')})
        row=event_record(progress=0);row['analysis_complete']=False
        row['unresolved_error_severity'].update(value=None,valid=False)
        class Judge:
            def __init__(self,*args):self.options=options['judge'];self.api_peak=0;self.api_ledger=[]
            def health(self,**kwargs):pass
            def annotate_one(self,rid,messages,deadline):return rid,{'parsed':parse(row),'attempts':1}
        from test_online_rubric import pair
        with tempfile.TemporaryDirectory() as d, patch('gad.extensions.online_rubric.JudgeClient',Judge):
            owner=OnlineRubric(options,Path(d)/'warmup')
            for step in (1,2):
                stats=owner.annotate([pair(step)],SimpleNamespace(batch={}),1,1,step)
                self.assertEqual(0,stats['rubric/process_student_all_contract_rejected'])
            with self.assertRaisesRegex(StudentProcessSupervisionError,'empty_streak'):
                owner.annotate([pair(3)],SimpleNamespace(batch={}),1,1,3)

    def test_progress_and_error_remain_independent_without_unknown_math(self):
        for error in (0,.5,1,None):
            row=event_record(progress=1);row['completion']='complete'
            row['unresolved_error_severity'].update(value=error,valid=error is not None)
            if error in (.5,1):
                row['events']['e2']={**deepcopy(row['events']['e1']),'type':'explicit_error'}
                row['unresolved_error_severity']['event_ids']=['e2']
            self.assertEqual(error in (0,None),parse(row)['verified_progress']['valid'])
        row=event_record(progress=.5);row['unresolved_error_severity'].update(value=None,valid=False)
        self.assertTrue(parse(row)['verified_progress']['valid'])

    def test_references_reject_illegal_duplicate_and_overlong_ids(self):
        for refs in (['e9'],['e1','e1'],['e1']*5):
            row=event_record();row['verified_progress']['event_ids']=refs
            with self.assertRaises(ProtocolError):parse(row)
            row=event_record();row['events']['e1']['related_event_ids']=refs
            self.assertFalse(parse(row)['_event_validity']['e1']['valid'])
        self.assertTrue(parse(event_record())['verified_progress']['valid'])

    def test_events_require_evidence_but_empty_label_refs_are_allowed(self):
        for kind in ('neutral','unjudgeable','verified_progress','explicit_error'):
            row=event_record();row['events']['e1'].update(type=kind,supporting_span_ranges={})
            self.assertFalse(parse(row)['_event_validity']['e1']['valid'])
        self.assertTrue(parse(event_record())['unresolved_error_severity']['valid'])

    def test_protocol_versions_fail_before_any_judge_request(self):
        import yaml
        cfg=compose_config(settings=['rubric.enabled=true','rubric.mode=correctness_only'])
        self.assertEqual([],validate_config(cfg))
        with tempfile.TemporaryDirectory() as d:
            spec=Path(d)/'spec.yaml'; prompt=Path(d)/'prompt.txt'
            current=yaml.safe_load(Path(cfg.require('rubric.spec')).read_text())
            for field in ('version','schema_version','input_protocol'):
                changed={**current,field:'obsolete'};spec.write_text(yaml.safe_dump(changed))
                cfg.data['rubric.spec']=str(spec)
                self.assertTrue(any('protocol' in e for e in validate_config(cfg)))
            cfg.data['rubric.spec']=str(ROOT/'rubric/specs/math_process_v27.yaml')
            prompt.write_text('rubric.events.v2.6 rubric.input.v2.4')
            cfg.data['rubric.prompt']=str(prompt)
            self.assertTrue(any('protocol' in e for e in validate_config(cfg)))
            spec.write_text('invalid: [');cfg.data['rubric.spec']=str(spec)
            self.assertTrue(validate_config(cfg))


if __name__=='__main__':unittest.main()
