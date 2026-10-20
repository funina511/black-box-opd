"""Exercise the full two-arm harness with an owned fake service, never GPU work."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'scoring/tests')]
import calibrate_rubric_judge as calibration
from config import compose_config
from gad.extensions.judge_client import JudgeClient
from test_unified import event_record
from rubric.core.semantic_v24 import VERSION


def review():
    def d(ids=None,ranges=None):return {'verdict':'supported','event_ids':ids or [],'supporting_span_ranges':ranges or {},'refuting_span_ranges':{},'reason':'checked'}
    return {'version':VERSION,'labels':{k:d(['e1']) for k in ('verified_progress','unresolved_error_severity')},
            'events':{'e1':{k:v for k,v in d(ranges={'r1':{'start':'s1','end':'s1'}}).items() if k!='event_ids'}}}


class LocalClient(JudgeClient):
    def __init__(self,options,expected,tokenizer=None,request_budget=None):
        super().__init__(options,expected,SimpleNamespace(apply_chat_template=lambda *a,**k:[1]),request_budget=request_budget)
    def health(self,**kwargs):return {}
    def request(self,suffix,payload=None,timeout=None):
        row=review() if 'version' in payload['guided_json']['properties'] else event_record()
        return {'choices':[{'message':{'content':json.dumps(row),'reasoning_content':'checked in thinking'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}


class BenchmarkHarnessTests(unittest.TestCase):
    def test_one_start_two_fresh_arms_shared_budget_and_all_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cfg=compose_config(ROOT/'reports/rubric_v24_holdout_window1_20261016/config.yaml')
            cfg.data['rubric.judge.base_url']='http://127.0.0.1:0/v1'
            config=root/'config.yaml';config.write_text(yaml.safe_dump(cfg.to_dict()))
            output=root/'out';starts=[]
            def subprocess_run(cmd,**kwargs):
                if cmd[0]=='nvidia-smi':return SimpleNamespace(stdout='6, 1, 0\n7, 1, 0')
                if cmd[-1]=='start':
                    starts.append(cmd);directory=root/'runs/_rubric_judge/0';directory.mkdir(parents=True)
                    (directory/'owner.json').write_text(json.dumps({'pid':os.getpid(),'birth':'test-only'}))
                return SimpleNamespace(returncode=0)
            argv=['calibrate','--config',str(config),'--output',str(output),'--samples',str(ROOT/'reports/rubric_v24_appendix_a_20261016/fixed4.jsonl'),
                  '--concurrency-groups','1,2','--finish-cohort','--max-requests','12','--wall-seconds','30']
            with patch.object(sys,'argv',argv),patch.object(calibration,'ROOT',root),patch.object(calibration,'JudgeClient',LocalClient),patch.object(calibration.subprocess,'run',side_effect=subprocess_run),patch.object(calibration,'stop_owned_service',return_value={'stopped':True}):
                calibration.main()
            ledger=json.loads((output/'budget.json').read_text())
            self.assertEqual(1,len(starts));self.assertEqual(12,ledger['actual_http_attempts'])
            self.assertEqual(12,ledger['shared_reserved_attempts'])
            self.assertEqual(8,ledger['final_arms']['B1']['actual_requests'])
            self.assertEqual(4,ledger['final_arms']['B2']['actual_requests'])
            self.assertEqual(4,ledger['final_arms']['B1']['targeted_dedup_size'])
            self.assertGreater(ledger['final_arms']['B2']['targeted_dedup_size'],0)
            for arm in ('B1','B2'):
                rows=(output/arm/'samples.jsonl').read_text().splitlines()
                self.assertEqual(4,len(rows));self.assertTrue(all('result' in json.loads(r) for r in rows))

if __name__=='__main__':unittest.main()
