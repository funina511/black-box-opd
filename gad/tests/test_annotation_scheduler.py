"""Production scheduler safety under controlled API ordering, budgets and failures."""
import json
import tempfile
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'scoring/tests'),str(ROOT/'gad/tests')]
from test_unified import event_record
import test_online_rubric
from rubric.core.contracts import judge_messages
from rubric.core.semantic_v24 import VERSION,apply_review
from rubric.core.process_v2 import parse_events
from gad.extensions.judge_client import JudgeClient,RequestBudget,ServiceUnavailable
from gad.extensions.annotation_scheduler import annotate_chains


def decision(events=None, verdict='supported',ranges=None):
    return dict(verdict=verdict,event_ids=events or [],supporting_span_ranges=ranges or {},refuting_span_ranges={},reason='checked assertion independently')


def review_record():
    return {'version':VERSION,'labels':{d:decision(['e1']) for d in ('verified_progress','unresolved_error_severity')},
            'events':{'e1':{k:v for k,v in decision(ranges={'r1':{'start':'s1','end':'s1'}}).items() if k!='event_ids'}}}


class FakeClient(JudgeClient):
    def __init__(self,concurrency=2,budget=20,shared=None,delays=None,fail=None):
        opts=test_online_rubric.ClientTests().options();opts.update(concurrency=concurrency,request_budget=budget,rubric_version='math_process_v2',max_retries=0,enable_thinking=False)
        super().__init__(opts,'f',SimpleNamespace(apply_chat_template=lambda *a,**k:[1]),request_budget=shared)
        self.delays=delays or {};self.fail=fail or set();self.trace=[];self.trace_lock=threading.Lock()
    def request(self,suffix,payload=None,timeout=None):
        targeted='version' in payload['guided_json']['properties']
        if targeted:name=json.loads(payload['messages'][-1]['content'])['problem']
        else:name=json.loads(payload['messages'][-1]['content'].split('\n\nVISIBLE_RESPONSE\n')[0])['problem']
        stage='review' if targeted else 'primary'
        with self.trace_lock:self.trace.append(('start',stage,name,time.monotonic()))
        time.sleep(self.delays.get((stage,name),.005))
        if (stage,name) in self.fail:raise ServiceUnavailable('controlled transport failure')
        content=json.dumps(review_record() if targeted else event_record())
        with self.trace_lock:self.trace.append(('end',stage,name,time.monotonic()))
        return {'choices':[{'message':{'content':content},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}


def run(c,n=4,deadline=None,**kwargs):
    requests={str(i):judge_messages(str(i),'2x=4','2','s','p','math_process_v2') for i in range(n)}
    return annotate_chains(c,requests,{str(i):(str(i),'2x=4') for i in range(n)},deadline or time.monotonic()+5,**kwargs)


class SchedulerTests(unittest.TestCase):
    def test_concurrency_one_and_two_exactly_one_review(self):
        for concurrency in (1,2):
            c=FakeClient(concurrency);rs=run(c)
            self.assertEqual(['0','1','2','3'],list(rs))
            self.assertEqual(8,c.request_count);self.assertLessEqual(c.api_peak,concurrency)
            self.assertTrue(all(r['parsed']['verified_progress']['valid'] for r in rs.values()))
            self.assertEqual(4,len(c.targeted_ids))
            for i in range(4):
                primary_end=next(t[3] for t in c.trace if t[:3]==('end','primary',str(i)))
                review_start=next(t[3] for t in c.trace if t[:3]==('start','review',str(i)))
                self.assertGreaterEqual(review_start,primary_end)

    def test_review_starts_before_other_primary_finishes(self):
        c=FakeClient(2,delays={('primary','1'):.12});run(c,3)
        start=next(t[3] for t in c.trace if t[:3]==('start','review','0'))
        end=next(t[3] for t in c.trace if t[:3]==('end','primary','1'))
        self.assertLess(start,end)

    def test_atomic_budget_shared_across_arms_and_not_queue_count(self):
        shared=RequestBudget(5);a=FakeClient(shared=shared);b=FakeClient(shared=shared)
        first=run(a,2);second=run(b,2)
        self.assertEqual(5,shared.count);self.assertEqual(5,len(a.trace[::2])+len(b.trace[::2]))
        self.assertEqual(1,b.request_count)
        self.assertTrue(any('judge_request_budget_exhausted' in json.dumps(r) for r in second.values()))
        self.assertFalse(any(r.get('parsed',{}).get('verified_progress',{}).get('valid') for r in second.values()))

    def test_timeout_preserves_finished_results_and_real_inflight_state(self):
        c=FakeClient(2,delays={('primary','1'):.15});rs=run(c,2,deadline=time.monotonic()+.05)
        self.assertTrue(rs['0']['parsed']['verified_progress']['valid'])
        self.assertEqual('judge_deadline_exceeded',rs['1']['error'])
        self.assertGreater(c.api_active,0)
        self.assertFalse(c.last_schedule['unfinished'][0]['server_cancellation_confirmed'])
        time.sleep(.16);self.assertEqual(0,c.api_active)

    def test_partial_failure_and_recovery_does_not_repeat_completed_nodes(self):
        c=FakeClient(2,fail={('primary','1')});rs=run(c,2)
        self.assertTrue(rs['0']['parsed']['verified_progress']['valid'])
        self.assertEqual('judge_transport_error',rs['1']['error'])
        count=c.request_count;again=run(c,2,recovered=rs)
        self.assertEqual(count,c.request_count);self.assertEqual(rs,again)
        initial=rs['0']['initial_parsed']
        retry=run(c,1,recovered={'0':{'parsed':initial,'request_id':'0','attempts':1}})
        self.assertEqual(count,c.request_count)
        self.assertFalse(retry['0']['parsed']['verified_progress']['valid'])
        self.assertEqual('targeted_request_already_attempted',retry['0']['targeted_result']['error'])

    def test_persistent_recovery_never_resends_uncertain_review(self):
        from gad.extensions.annotation_journal import AnnotationJournal
        with tempfile.TemporaryDirectory() as folder:
            journal=AnnotationJournal(Path(folder),{'version':'test'})
            initial={'request_id':'0','parsed':parse_events(json.dumps(event_record()),'2x=4'),'attempts':1}
            journal.save('0','review_submitted',initial)
            recovered=journal.recover({'0':('0','2x=4')})
            c=FakeClient();result=run(c,1,recovered=recovered)
            self.assertEqual(0,c.request_count)
            self.assertFalse(result['0']['parsed']['verified_progress']['valid'])
            self.assertEqual('unknown_not_zero',result['0']['targeted_result']['prior_send_cost'])

    def test_label_reuse_requires_own_decision_and_valid_dependencies(self):
        parsed=parse_events(json.dumps(event_record()),'2x=4')
        r=review_record();self.assertTrue(apply_review(parsed,'2x=4',r)['verified_progress']['valid'])
        r['labels']['verified_progress']['verdict']='unresolved'
        self.assertFalse(apply_review(parsed,'2x=4',r)['verified_progress']['valid'])
        r=review_record();r['events']['e1']['verdict']='refuted'
        self.assertFalse(apply_review(parsed,'2x=4',r)['verified_progress']['valid'])
        r=review_record();r['events']['e4']=r['events'].pop('e1');r['labels']['verified_progress']['event_ids']=['e4']
        self.assertFalse(apply_review(parsed,'2x=4',r)['verified_progress']['valid'])
        r=review_record();r['labels']['verified_progress']['event_ids']=[]
        self.assertFalse(apply_review(parsed,'2x=4',r)['verified_progress']['valid'])

if __name__=='__main__':unittest.main()
