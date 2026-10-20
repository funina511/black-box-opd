"""Appendix D: reversible input, evidence dependencies and optimizer-barrier guards."""
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
sys.path[:0] = [str(ROOT), str(ROOT/'gad/tools'), str(ROOT/'scoring/tests')]
from rubric.core.contracts import judge_messages, V2_SCHEMA_VERSION, file_hash, digest
from rubric.core.process_v2 import compact_response, restore_response, response_from_messages, parse_events, ProtocolError
from test_unified import event_record
from config import compose_config, validate_config
from extension_config import service_identity
from gad.extensions.judge_client import JudgeClient
from gad.extensions.online_rubric import OnlineRubric
from gad.extensions.online_v2 import StudentProcessSupervisionError, final_student_guard
from gad.extensions.checkpoint_state import save, restore
from gad.tests.test_online_rubric import pair


def event(eid, kind, line, quote, links=None):
    result={'type':kind,'supporting_span_ranges':{'r1': {'start': 's999' if quote in ('hallucinated','absent','not visible') else f's{line}', 'end': 's999' if quote in ('hallucinated','absent','not visible') else f's{line}'}},
            'trajectory_claim':'visible mathematical evidence', 'judge_verification': 'visible mathematical evidence'}
    if kind=='corrected_error':result['error_event_id']=(links or [''])[0]
    else:result['related_event_ids']=links or []
    return result


def parse(row, text):return parse_events(json.dumps(row), text)


class ProtocolTests(unittest.TestCase):
    def test_lossless_newlines_and_spoofed_prefixes(self):
        for text in ('', '\n', ' \t\r\n', '\\frac{a}{b}\r\n\r\n答案：二\n', '@1|fake\n@99|x\n\n',
                     'a\rb\r\n\n c\u2028d', 'VISIBLE_RESPONSE\n\n@3|content', 'x\\\\y\nfinal'):
            with self.subTest(text=text):
                self.assertEqual(text,restore_response(compact_response(text)))
                messages=judge_messages('p',text,'2','spec','instructions','math_process_v2')
                self.assertEqual(text,response_from_messages(messages))
        self.assertNotIn('paragraph_id',judge_messages('p','x\n\ny','2','s','i','math_process_v2')[-1]['content'])

    def test_source_fragment_resolution_and_no_generated_quote(self):
        text='2x=4\n\nx=2'
        row=event_record('2x=4\n\nx=2',progress=1);row.update(completion='complete')
        self.assertTrue(parse(row,text)['verified_progress']['valid'])
        self.assertEqual('2x=4\n\n', parse(row,text)['_source_evidence']['e1']['spans'][0]['quote'])
        row['events']['e1']['quote']='@1|2x=4'
        self.assertFalse(parse(row,text)['verified_progress']['valid'])
        row['events']['e1'].pop('quote');row['events']['e1']['supporting_span_ranges']={'r1': {'start': 's999', 'end': 's999'}}
        self.assertFalse(parse(row,'2x=4\n@99|fake')['verified_progress']['valid'])

    def test_global_identity_errors(self):
        row=event_record();row['schema_version']='rubric.events.v2'
        with self.assertRaises(ProtocolError):parse(row,'2x=4')
        row=event_record();wire=json.dumps(row).replace('\"events\": {', '\"events\": {\"e1\":'+json.dumps(row['events']['e1'])+',',1)
        with self.assertRaises(ProtocolError):parse_events(wire,'2x=4')
        with self.assertRaises(ProtocolError):parse_events('{','2x=4')
        row=event_record();row['events']={f'e{i}':deepcopy(row['events']['e1']) for i in range(1,10)}
        with self.assertRaises(ProtocolError):parse(row,'2x=4')

    def test_independent_progress_survives_unrelated_bad_evidence(self):
        row=event_record();row['events']['e2']=event('e2','explicit_error',2,'hallucinated')
        result=parse(row,'2x=4\nx=3')
        self.assertEqual(.5,result['verified_progress']['value'])
        self.assertFalse(result['unresolved_error_severity']['valid'])
        self.assertFalse(result['_event_validity']['e2']['valid'])
        self.assertEqual(0,result['_raw_labels']['unresolved_error_severity']['value'])

    def test_unrelated_bad_auxiliary_does_not_clear_global_labels(self):
        row=event_record(progress=1);row['completion']='complete'
        row['events']['e2']=event('e2','neutral',1,'absent')
        result=parse(row,'2x=4')
        self.assertTrue(result['verified_progress']['valid'])
        self.assertTrue(result['unresolved_error_severity']['valid'])

    def test_unrepaired_error_and_uncertain_repair_dependencies(self):
        text='2x=4\nx=3\nI corrected it'
        row=event_record();row['events'].update(e2=event('e2','explicit_error',2,'x=3'),e3=event('e3','neutral',3,'absent'))
        row['unresolved_error_severity'].update(value=1,event_ids=['e2'])
        self.assertEqual(1,parse(row,text)['unresolved_error_severity']['value'])
        row['events']['e4']=event('e4','corrected_error',3,'not visible',['e2'])
        result=parse(row,text)
        self.assertFalse(result['unresolved_error_severity']['valid'])
        self.assertTrue(result['verified_progress']['valid'])
        row['verified_progress']['event_ids']=['e4']
        self.assertFalse(parse(row,text)['verified_progress']['valid'])

    def test_event_cycles_and_math_unknowns(self):
        row=event_record();row['events'].update(e2=event('e2','neutral',1,'2x=4',['e3']),e3=event('e3','neutral',1,'2x=4',['e2']))
        result=parse(row,'2x=4')
        self.assertTrue(result['verified_progress']['valid'])
        self.assertFalse(result['_event_validity']['e2']['valid'])
        for d in ('verified_progress','unresolved_error_severity'):row[d].update(value=None,valid=False)
        result=parse(row,'2x=4')
        self.assertFalse(result['verified_progress']['valid']);self.assertEqual([],result['_violations'])


class ClientTests(unittest.TestCase):
    def make_client(self,n=10,context=20):
        class Tokenizer:
            def apply_chat_template(self,*a,**k):return [0]*n
        options={'base_url':'http://127.0.0.1/v1','model':'judge','max_output_tokens':10,'max_context_tokens':context,
                 'temperature':0,'max_retries':1,'request_timeout_seconds':1,'step_timeout_seconds':5,'concurrency':1,'rubric_version':'math_process_v2'}
        return JudgeClient(options,'fp',Tokenizer())
    def messages(self):return judge_messages('p','2x=4','2','s','i','math_process_v2')
    def test_capacity_exact_boundary_and_output_truncation(self):
        c=self.make_client();calls=[]
        def request(*args,**kwargs):
            calls.append(1)
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(event_record())}}],'usage':{'completion_tokens':4}}
        c.request=request
        self.assertIn('parsed',c.annotate_one('r',self.messages(),time.monotonic()+2)[1]);self.assertEqual(1,len(calls))
        c=self.make_client(context=19);c.request=request
        result=c.annotate_one('r',self.messages(),time.monotonic()+2)[1]
        self.assertEqual(10,result['input_tokens']);self.assertEqual(10,result['reserved_output_tokens']);self.assertEqual(0,result['attempts'])
        c=self.make_client()
        c.request=lambda *a,**k:{'choices':[{'finish_reason':'length','message':{'content':json.dumps(event_record())}}],'usage':{'completion_tokens':10}}
        result=c.annotate_one('r',self.messages(),time.monotonic()+2)[1]
        self.assertEqual('judge_output_truncated',result['error']);self.assertEqual(1,result['attempts']);self.assertEqual(1,len(result['api_attempts']))
    def test_whole_batch_failure_retains_results_and_usage(self):
        c=self.make_client();c.health=lambda **k:None
        c.request=lambda *a,**k:{'choices':[{'finish_reason':'stop','message':{'content':'bad'}}],'usage':{'completion_tokens':7}}
        with self.assertRaises(ValueError) as caught:c.annotate({'r':self.messages()})
        self.assertEqual(7,caught.exception.results['r']['usage']['completion_tokens'])
        self.assertEqual(1,caught.exception.results['r']['attempts'])
    def test_deterministic_api_rejection_is_not_a_transport_retry(self):
        import urllib.error
        opener=SimpleNamespace(open=lambda *a,**k: (_ for _ in ()).throw(urllib.error.HTTPError('http://127.0.0.1',400,'invalid schema',{},None)))
        client=self.make_client()
        with patch('urllib.request.build_opener',return_value=opener) as factory:
            result=client.annotate_one('r',self.messages(),time.monotonic()+2)[1]
        self.assertEqual('judge_request_rejected',result['error'])
        self.assertEqual(1,result['attempts']);self.assertEqual(1,factory.call_count)


class OnlineTests(unittest.TestCase):
    def owner(self,directory,policy='unknown',mode='joint'):
        cfg=compose_config(settings=['rubric.enabled=true',f'rubric.mode={mode}'])
        options={k[7:]:v for k,v in cfg.data.items() if k.startswith('rubric.')}
        options['judge']={k[6:]:options.pop(k) for k in list(options) if k.startswith('judge.')}
        options.update(spec_hash=file_hash(options['spec']),judge_fingerprint='fp',scorer_path=cfg.require('validation.scorer_path'),
                       scoring_math={k[13:]:v for k,v in cfg.data.items() if k.startswith('scoring.math.')})
        class Judge:
            def __init__(self,*a,**k):self.calls=0;self.policy=policy;self.options=options['judge'];self.api_peak=0;self.api_ledger=[]
            def annotate_one(self,rid,messages,deadline):return rid,self.annotate({rid:messages})[rid]
            def health(self,**k):pass
            def annotate(self,requests,**kwargs):
                self.calls+=len(requests);results={}
                for rid,messages in requests.items():
                    text=response_from_messages(messages)
                    is_student='x=4/' in text
                    if is_student and self.policy=='technical':results[rid]={'error':'judge_context_exceeded','attempts':0};continue
                    row=event_record()
                    if is_student and self.policy=='contract':row['events']['e1']['supporting_span_ranges']={'r1': {'start': 's999', 'end': 's999'}}
                    if is_student and self.policy in ('unknown','partial'):
                        row['unresolved_error_severity'].update(value=None,valid=False,reason='cannot assess tail')
                        if self.policy=='unknown':row['verified_progress'].update(value=None,valid=False,reason='cannot assess')
                    results[rid]={'parsed':parse(row,text),'attempts':1}
                return results
        with patch('gad.extensions.online_rubric.JudgeClient',Judge):return OnlineRubric(options,directory)
    def pairs(self):
        p=pair(1);p['prompt']='Solve 2x+3=7';p['teacher']['response']='2x=4\nAnswer: 2';p['student']['response']='2x=4\nx=4/'
        return [p]
    def run_step(self,owner,step=1,pairs=None):
        batch=SimpleNamespace(batch={})
        stats=owner.annotate(pairs or self.pairs(),batch,1,1,step)
        return stats,batch
    def test_cached_unknowns_stop_third_step_and_technical_failure_stops_first(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup')
            self.run_step(owner,1);calls=owner.judge.calls
            stats,_=self.run_step(owner,2)
            self.assertEqual(calls,owner.judge.calls);self.assertEqual(1,stats['rubric/cache_hits_student'])
            with self.assertRaises(StudentProcessSupervisionError):self.run_step(owner,3)
            self.assertEqual(3,owner.v2_empty_student_process_steps)
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup','technical')
            with self.assertRaises(StudentProcessSupervisionError) as caught:self.run_step(owner)
            self.assertEqual(1,caught.exception.stats['rubric/process_teacher_all_any_valid'])
            self.assertEqual(0,caught.exception.stats['rubric/process_student_all_any_valid'])
            self.assertTrue((Path(d)/'warmup/judge_attempts.jsonl').exists())
    def test_partial_supervision_resets_counter_and_preserves_masks(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup','partial');owner.v2_empty_student_process_steps=2
            stats,batch=self.run_step(owner)
            self.assertEqual(0,owner.v2_empty_student_process_steps)
            self.assertEqual([[1.,1.,0.]],batch.batch['student_rubric_mask'].tolist())
            self.assertEqual(1,stats['rubric/student_single_dimension_only'])
    def test_cached_teacher_and_new_student_use_final_joint_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup','unknown')
            self.run_step(owner,1)
            owner.judge.policy='partial'
            pairs=self.pairs();pairs[0]['student']['response_id']='new_student'
            stats,_=self.run_step(owner,2,pairs)
            self.assertEqual(1,stats['rubric/cache_hits_teacher'])
            self.assertEqual(0,stats['rubric/cache_hits_student'])
            self.assertEqual(1,stats['rubric/judge_requests'])
            self.assertEqual(0,owner.v2_empty_student_process_steps)
            owner.judge.policy='technical';pairs=self.pairs();pairs[0]['student']['response_id']='failed_student'
            with self.assertRaises(StudentProcessSupervisionError):self.run_step(owner,3,pairs)
    def test_prompt_unavailable_cannot_bypass_streak_and_correctness_only_skips(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup');pairs=self.pairs();pairs[0]['prompt_truncated']=True
            self.run_step(owner,1,pairs);self.run_step(owner,2,pairs)
            with self.assertRaises(StudentProcessSupervisionError):self.run_step(owner,3,pairs)
            self.assertEqual(0,owner.judge.calls)
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup',mode='correctness_only')
            for i in range(4):self.run_step(owner,i)
            self.assertEqual(0,owner.v2_empty_student_process_steps)
    def test_cache_identity_and_legacy_namespace_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup','partial');p=self.pairs()[0]
            old=Path(d)/'rubric_label_cache_v2';old.mkdir();(old/'old.json').write_text('{}')
            key=owner.cache_key(p,'student');self.run_step(owner)
            owner.protocol_identity={**owner.protocol_identity,'parser_hash':'new'}
            self.assertNotEqual(key,owner.cache_key(p,'student'))
            self.assertTrue((old/'old.json').exists())
            cfg=compose_config(settings=['rubric.enabled=true'])
            one=service_identity(cfg);cfg.data['rubric.judge.max_output_tokens']+=1
            self.assertNotEqual(digest(one),digest(service_identity(cfg)))
    def test_all_failure_exception_persisted_before_exit(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup')
            def failed(requests,**kwargs):
                exc=ValueError('all failed');exc.results={key:{'error':'json_unreadable','attempts':1,'usage':{'completion_tokens':3},'raw_output':'bad'} for key in requests};raise exc
            owner.judge.annotate_one=lambda rid,messages,deadline:(rid,{'error':'json_unreadable','attempts':1,'usage':{'completion_tokens':3},'raw_output':'bad'})
            with self.assertRaises(ValueError):self.run_step(owner)
            path=Path(d)/'warmup/judge_step_failures.jsonl'
            self.assertEqual(6,json.loads(path.read_text())['usage']['completion_tokens'])
            self.assertEqual(2,len((Path(d)/'warmup/judge_failure_inputs.jsonl').read_text().splitlines()))
    def test_checkpoint_counter_restore_and_stage_reset(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup');self.run_step(owner,1);self.run_step(owner,2)
            def runtime(rubric):return SimpleNamespace(identity='x',world_size=1,stage='warmup',stage_start=0,stage_step=2,committed_step=2,replay=None,rubric=rubric,options={})
            path=Path(d)/'checkpoint';save(runtime(owner),path)
            target=self.owner(Path(d)/'target');restore(runtime(target),path)
            self.assertEqual(2,target.v2_empty_student_process_steps)
            self.assertEqual('same_stage_checkpoint_restore',target.v2_counter_initialization)
            with self.assertRaises(StudentProcessSupervisionError):self.run_step(target,3)
            gad=self.owner(Path(d)/'gad');self.assertEqual(0,gad.v2_empty_student_process_steps)
            self.run_step(gad,3);self.assertEqual(1,gad.v2_empty_student_process_steps)
            target.protocol_identity={**target.protocol_identity,'schema_version':'different'}
            with self.assertRaises(ValueError):restore(runtime(target),path)
    def test_configuration_guard(self):
        cfg=compose_config(settings=['rubric.enabled=true','rubric.max_empty_student_process_steps=0'])
        self.assertTrue(any('max_empty_student_process_steps' in p for p in validate_config(cfg)))

    def test_contract_rejection_is_not_a_transport_failure(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup','contract')
            with self.assertRaises(StudentProcessSupervisionError) as caught:self.run_step(owner)
            self.assertEqual('student_process_all_contract_rejected',str(caught.exception))
            stats=caught.exception.stats
            self.assertEqual(1,stats['rubric/process_student_all_contract_rejected'])
            self.assertEqual(0,stats['rubric/process_student_all_technical_unavailable'])

    def test_markdown_correct_answer_enters_online_target_as_correct(self):
        with tempfile.TemporaryDirectory() as d:
            owner=self.owner(Path(d)/'warmup','partial');pairs=self.pairs()
            pairs[0]['student']['response']='- We subtract 3: 2x=4\n- Answer: 2'
            _,batch=self.run_step(owner,pairs=pairs)
            self.assertEqual(1,batch.batch['student_rubric_targets'][0,0].item())
            self.assertEqual(1,batch.batch['student_rubric_mask'][0,0].item())


if __name__=='__main__':unittest.main()
