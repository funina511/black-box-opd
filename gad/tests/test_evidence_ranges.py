"""Appendix A: lossless migration against the frozen legacy interpreter."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from rubric.core.evidence_ranges import encode,expand,EvidenceError
from rubric.core.process_v2 import source_spans,parse_events
from rubric.core.process_v25 import parse_events as parse_legacy
from rubric.core.contracts import V2_SCHEMA_VERSION


def migrate(row,response):
    row=deepcopy(row);row['schema_version']=V2_SCHEMA_VERSION
    for event in row['events'].values():
        event['supporting_span_ranges']=encode(event.pop('supporting_span_ids'),source_spans(response))
    return row


class RangeTests(unittest.TestCase):
    def test_source_order_overlap_dedup_offsets(self):
        text='\r\n'.join(str(i) for i in range(20));spans=source_spans(text)
        units={'r1':{'start':'s10','end':'s12'},'r2':{'start':'s2','end':'s10'}}
        self.assertEqual([f's{i}' for i in range(2,13)],expand(units,spans))
        ids=['s20','s3','s2','s3','s10'];encoded=encode(ids,spans)
        self.assertEqual(['s2','s3','s10','s20'],expand(encoded,spans))
        self.assertEqual(3,len(encoded))
        for s in spans:self.assertEqual(s['text'],text[s['start']:s['end']])

    def test_bad_endpoints_and_capacity_are_explicit(self):
        spans=source_spans('\n'.join('x' for _ in range(20)))
        for units in ([{'start':'s10','end':'s2'}],[{'start':'s0','end':'s2'}],[{'start':None,'end':'s2'}]):
            with self.assertRaises(EvidenceError):expand({f'r{i}':u for i,u in enumerate(units,1)},spans)
        with self.assertRaisesRegex(EvidenceError,'evidence_capacity_conflict'):encode([f's{i}' for i in range(1,20,2)],spans)
        with self.assertRaisesRegex(EvidenceError,'evidence_capacity_conflict'):expand({f'r{i}':{'start':'s1','end':'s20'} for i in range(1,10)},spans)

    def test_saved_eight_primary_records_lossless_masks_and_dependencies(self):
        rows=[json.loads(l) for l in (ROOT/'reports/rubric_v24_appendix_a_20261016/development8.jsonl').read_text().splitlines()]
        checked=0;conflicts=[]
        for r in rows:
            result=r['result'].get('initial_parsed',r['result'].get('parsed',{}))
            if not result:continue
            original=result['_record'];legacy=parse_legacy(json.dumps(original),r['response'])
            try: migrated=migrate(original,r['response'])
            except EvidenceError as exc:
                self.assertEqual('evidence_capacity_conflict',exc.code);conflicts.append(r['id']);continue
            new=parse_events(json.dumps(migrated),r['response'])
            for d in ('verified_progress','unresolved_error_severity'):
                self.assertEqual(legacy[d],new[d]);self.assertEqual(legacy['_mask_reasons'][d],new['_mask_reasons'][d])
            self.assertEqual(legacy['_event_validity'],new['_event_validity'])
            for e,old in legacy['_source_evidence'].items():
                expected=sorted(set(old['supporting_span_ids']),key=lambda x:int(x[1:]))
                self.assertEqual(expected,new['_source_evidence'][e]['supporting_span_ids'])
                lookup={s['span_id']:s for s in old['spans']}
                self.assertEqual([lookup[s] for s in expected],new['_source_evidence'][e]['spans'])
            checked+=1
        self.assertGreaterEqual(checked,5)
        print('Historical lossless rows:',checked,'explicit capacity conflicts:',conflicts)

    def test_frozen_array_input_protocol_restores_saved_requests(self):
        from rubric.core.process_v26 import response_from_messages
        requests=json.loads((ROOT/'reports/rubric_v24_appendix_a_20261016/corrected_gpu/requests.json').read_text())
        rows=[json.loads(l) for l in (ROOT/'reports/rubric_v24_appendix_a_20261016/fixed4.jsonl').read_text().splitlines()]
        for row in rows:
            self.assertEqual(row['response'],response_from_messages(requests[row['id']]))

    def test_actual_repair_and_independent_progress_masks_unchanged(self):
        row={'schema_version':'rubric.events.v2.5','rubric_version':'math_process_v2','completion':'prefix','analysis_complete':True,'events':{},
             'verified_progress':{'value':.5,'valid':True,'reason':'independent','event_ids':['e1','e2']},
             'unresolved_error_severity':{'value':0,'valid':True,'reason':'checked','event_ids':[]}}
        def e(kind,ids):return dict(type=kind,supporting_span_ids=ids,trajectory_claim='original',judge_verification='checked',related_event_ids=[])
        row['events']={'e1':e('verified_progress',['s1']),'e2':e('explicit_error',['s2','s3'])}
        row['events']['e3']=e('corrected_error',['s3']);row['events']['e3'].pop('related_event_ids');row['events']['e3']['error_event_id']='e2'
        for ids in (['s3'],['s2','s3','s4']):
            row['events']['e3']['supporting_span_ids']=ids
            a=parse_legacy(json.dumps(row),'valid\nerror\ncontext\nactual replacement')
            b=parse_events(json.dumps(migrate(row,'valid\nerror\ncontext\nactual replacement')),'valid\nerror\ncontext\nactual replacement')
            self.assertEqual(a['_event_validity'],b['_event_validity']);self.assertEqual(a['_mask_reasons'],b['_mask_reasons'])
            self.assertEqual(.5,b['verified_progress']['value'])

if __name__=='__main__':unittest.main()
