"""Bounded protocol generation and independent training/evaluation state checks."""
import ast
import collections
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'gad/tests')]
import pipeline
import test_pipeline_cpu as pipeline_test
from test_pipeline_cpu import base_config, write_yaml
from config import load_config

class StateTests(unittest.TestCase):
    def test_exit_zero_incomplete_missing_corrupt_metrics_default_vs_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'warmup';pipeline_test.StageArtifactTests()._complete_checkpoint(out,78)
            metric=out/'validation_math_metrics.jsonl'
            cases=[(None,'missing'),('{bad','failed'),('null\n','failed'),('[]\n','failed'),
                (json.dumps(dict(global_step=78,count=512,acc=None,provisional_acc=52/512,
                    evaluation_complete=False,unscorable_rate=2/512,grader_error_rate=0,validation_scope='full')),'incomplete')]
            for content,status in cases:
                if content is not None:metric.write_text(content)
                report=pipeline.verify_stage_artifacts('warmup',out,expected_step=78,math_metrics_expected=True,require_final_full=True,full_count=512)
                self.assertTrue(report['handoff_ready'],report)
                self.assertEqual(status,report['evaluation_status']);self.assertEqual('valid',report['checkpoint_status'])
                strict=pipeline.verify_stage_artifacts('warmup',out,expected_step=78,math_metrics_expected=True,strict_evaluation=True)
                self.assertFalse(strict['handoff_ready'])
            self.assertIsNone(report['math_metrics']['last']['acc'])
            self.assertEqual(52/512,report['math_metrics']['last']['provisional_acc'])
            self.assertFalse(pipeline.verify_stage_artifacts('warmup',out,expected_step=79)['handoff_ready'])
            (out/'global_step_78/critic/optim_world_size_1_rank_0.pt').unlink()
            self.assertFalse(pipeline.verify_stage_artifacts('warmup',out,expected_step=78)['handoff_ready'])
            self.assertEqual('invalid',pipeline.verify_stage_artifacts('warmup',Path(tmp)/'missing',expected_step=78)['checkpoint_status'])

    def test_recovery_identity_exit_and_repeated_read_only_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'config.yaml';write_yaml(path,base_config(root));cfg=load_config(path)
            run=root/'run';run.mkdir();out=pipeline.stage_out_dir(cfg,run,'warmup')
            pipeline_test.StageArtifactTests()._complete_checkpoint(out,10)
            plan=pipeline.plan_stage(cfg,run,'warmup',steps_per_epoch_value=pipeline.steps_per_epoch(cfg),install_guard=False)
            record={'returncode':0,'plan':{k:v for k,v in plan.items() if k!='env'}}
            history=run/'stage_warmup.json';history.write_text(json.dumps(record));before=history.read_bytes()
            for _ in range(2):
                okay,reason=pipeline.check_skip(cfg,run,'warmup');self.assertTrue(okay,reason)
            self.assertEqual(before,history.read_bytes());self.assertFalse(pipeline.completion_marker(run,'warmup').exists())
            record['returncode']=1;history.write_text(json.dumps(record));self.assertFalse(pipeline.check_skip(cfg,run,'warmup')[0])
            record['returncode']=0;record['plan']['argv'].append('critic.optim.lr=9');history.write_text(json.dumps(record))
            self.assertFalse(pipeline.check_skip(cfg,run,'warmup')[0])

    def test_rescore_saved_responses_new_directory_only(self):
        from rescore_validation import run
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'saved.jsonl'
            source.write_text(''.join(json.dumps({'response_id':str(i),'output':text,'ground_truth':'2'})+'\n' for i,text in enumerate(['Answer: 2','Answer: \\boxed{'])))
            before=source.read_bytes();target=Path(tmp)/'rescored'
            result=run(source,target,2,78,'full')
            self.assertFalse(result['evaluation_complete']);self.assertIsNone(result['acc']);self.assertEqual(.5,result['provisional_acc'])
            self.assertEqual(2,result['failures'][0]['source_line']);self.assertEqual(before,source.read_bytes())
            with self.assertRaises(FileExistsError):run(source,target,2,78,'full')
            with self.assertRaisesRegex(ValueError,'regeneration'):run(Path(tmp)/'missing',Path(tmp)/'new',2,78,'full')

    def test_used_backend_preparation_never_mutates_sources(self):
        import subprocess
        backend=Path(os.environ['GAD_TEST_EXTENSION_ROOT'])/'warmup'
        from algorithm_patches import verify
        manifest=verify(backend)
        marker=backend/'.gad_backend_used';existed=marker.exists()
        before={str(p.relative_to(backend)):p.read_bytes() for p in backend.rglob('*.py')}
        try:
            marker.touch()
            result=subprocess.run([sys.executable,str(ROOT/'gad/tools/prepare_backends.py'),'--repo',str(ROOT/'gad/verl'),
                '--backend-root',str(backend.parent),'--profile','gad_extensions_v1','--stages','warmup'],capture_output=True,text=True)
            self.assertEqual(0,result.returncode,result.stdout+result.stderr)
            self.assertIn('verified frozen deployed backend',result.stdout)
            self.assertEqual(before,{str(p.relative_to(backend)):p.read_bytes() for p in backend.rglob('*.py')})
            self.assertEqual(manifest,verify(backend))
        finally:
            if not existed:marker.unlink()


    def test_finish_null_accuracy_nonzero_exit_and_resume_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'config.yaml';write_yaml(path,base_config(root));cfg=load_config(path)
            run=root/'run';out=pipeline.stage_out_dir(cfg,run,'warmup')
            pipeline_test.StageArtifactTests()._complete_checkpoint(out,10)
            (out/'validation_math_metrics.jsonl').write_text(json.dumps(dict(global_step=10,count=8,acc=None,provisional_acc=.25,evaluation_complete=False,unscorable_rate=.25)))
            plan=pipeline.plan_stage(cfg,run,'warmup',steps_per_epoch_value=pipeline.steps_per_epoch(cfg),install_guard=False)
            with patch('pipeline.preserve_failed_stage_log',return_value={}):
                pipeline.finish_stage(cfg,run,'warmup',plan,process=SimpleNamespace(returncode=0),started='test',math_expected=True,math_reason='test',trainer_skipped=False)
                marker=pipeline.completion_marker(run,'warmup')
                self.assertTrue(json.loads(marker.read_text())['training_completed'])
                self.assertEqual('incomplete',json.loads(marker.read_text())['evaluation_status'])
                original=(run/'stage_warmup.json').read_bytes()
                pipeline.finish_stage(cfg,run,'warmup',plan,process=None,started='test',math_expected=True,math_reason='test',trainer_skipped=True)
                self.assertEqual(original,(run/'stage_warmup.json').read_bytes())
                self.assertEqual(1,len(list((run/'reports').glob('resume_complete_warmup_*.json'))))
                marker.unlink()
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.finish_stage(cfg,run,'warmup',plan,process=SimpleNamespace(returncode=1),started='test',math_expected=True,math_reason='test',trainer_skipped=False)
                self.assertFalse(marker.exists())
                self.assertEqual('failed',json.loads((run/'stage_warmup.json').read_text())['verification']['training_status'])


class GrammarTests(unittest.TestCase):
    def test_installed_qwen8b_grammar_cannot_generate_ninth_or_duplicate_slot(self):
        import xgrammar as xgr
        from transformers import AutoTokenizer,AutoConfig
        from rubric.core.evidence_ranges import RANGES,expand,EvidenceError
        from rubric.core.process_v2 import JUDGE_SCHEMA_V2,source_spans
        from rubric.core.semantic_v24 import SCHEMA
        from gad.extensions.judge_client import guidance_schema
        path=ROOT.parent/'model/Qwen/Qwen3-8B'
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        info=xgr.TokenizerInfo.from_huggingface(tokenizer,vocab_size=AutoConfig.from_pretrained(path).vocab_size)
        compiler=xgr.GrammarCompiler(info,max_threads=2)
        for schema in (JUDGE_SCHEMA_V2,SCHEMA):
            self.assertIsNotNone(compiler.compile_json_schema(json.dumps(guidance_schema('math_process_v2',schema)),any_whitespace=True))
        compiled=compiler.compile_json_schema(json.dumps(RANGES),any_whitespace=True)
        def accepts(value):
            matcher=xgr.GrammarMatcher(compiled)
            return all(matcher.accept_token(i) for i in tokenizer.encode(value,add_special_tokens=False))
        self.assertTrue(accepts('{}'))
        eight={f'r{i}':{'start':f's{i}','end':f's{i}'} for i in range(1,9)}
        self.assertTrue(accepts(json.dumps(eight)))
        self.assertFalse(accepts(json.dumps({**eight,'r9':{'start':'s9','end':'s9'}})))
        self.assertFalse(accepts('{"r1":{"start":"s1","end":"s1"},"r1":{"start":"s2","end":"s2"}}'))
        self.assertEqual(['s1','s2'],expand({'r2':{'start':'s2','end':'s2'},'r1':{'start':'s1','end':'s1'}},source_spans('a\nb')))
        with self.assertRaises(EvidenceError):expand([{'start':'s1','end':'s1'}],source_spans('a'))

class ActualValidationTests(unittest.TestCase):
    """Execute the actual generated _validate method, retaining all its branches.

    Only transport/tokenization is faked; math scorer and optional ROUGE are real.
    """
    def validate(self,backend,scope,enabled):
        source=(backend/'verl/trainer/ppo/ray_trainer.py').read_text();tree=ast.parse(source)
        namespace={'os':os,'json':json,'time':time,'np':np,'defaultdict':collections.defaultdict}
        functions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign)) and
            (isinstance(n,ast.FunctionDef) and n.name.startswith('_gad_') or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_GAD_MATH') for t in n.targets))]
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(backend),'exec'),namespace)
        method=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
        method=next(n for n in method.body if isinstance(n,ast.FunctionDef) and n.name=='_validate')
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual_validate>','exec'),namespace)
        class Batch:
            def __init__(self):
                self.batch={'input_ids':torch.tensor([[1],[2]]),'responses':torch.tensor([[3],[4]]),'teacher_response':torch.tensor([[5],[6]]),'attention_mask':torch.ones(2,1)}
                self.non_tensor_batch={'reward_model':[{'ground_truth':'2'},{'ground_truth':'2'}],'index':['q0','q1']}
                self.meta_info={}
            @staticmethod
            def from_single_dict(x):return Batch()
            def repeat(self,**kw):return self
            def pop(self,**kw):return self
            def union(self,other):return self
        namespace.update(DataProto=Batch,pad_dataproto_to_divisor=lambda x,y:(x,0),unpad_dataproto=lambda x,**kw:x,
            process_validation_metrics=lambda *args:{})
        decoded=[]
        def decode(ids,**kw):
            token=int(ids[0]);decoded.append(token)
            return {1:'p1',2:'p2',3:'Answer: 2',4:'Answer: \\boxed{',5:'Answer: 2',6:'Answer: 1'}[token]
        with tempfile.TemporaryDirectory() as tmp:
            cfg=OmegaConf.create({'trainer':{'default_local_dir':tmp,'validation_rouge_l_enabled':enabled},
                'data':{'max_response_length':8,'gad_math_scorer_path':str(ROOT/'scoring/math.py')},
                'actor_rollout_ref':{'rollout':{'response_length':8,'val_kwargs':{'n':1,'do_sample':False,'temperature':0}}},'reward_model':{'enable':False}})
            class Loader:
                dataset=[1,2]
                def __iter__(self):return iter([None])
            obj=SimpleNamespace(config=cfg,global_steps=7,_gad_validation_scope=scope,
                quick_val_dataloader=Loader(),val_dataloader=Loader(),tokenizer=SimpleNamespace(decode=decode,eos_token_id=0,pad_token_id=0),
                actor_rollout_wg=SimpleNamespace(world_size=1,generate_sequences=lambda _:Batch()),async_rollout_mode=False,
                _maybe_log_val_generations=lambda **kw:None)
            import rouge_score.rouge_scorer as rouge
            real=rouge.RougeScorer
            with patch.object(rouge,'RougeScorer',wraps=real) as ctor:
                metrics=namespace['_validate'](obj)
            record=json.loads((Path(tmp)/'validation_math_metrics.jsonl').read_text())
            rows=[json.loads(l) for l in Path(record['rows_file']).read_text().splitlines()]
            self.assertEqual(2,record['expected_count']);self.assertEqual(2,record['actual_count'])
            self.assertFalse(record['evaluation_complete']);self.assertIsNone(record['acc'])
            self.assertEqual('malformed',rows[1]['parse_status']);self.assertEqual(2,record['failures'][0]['source_line'])
            self.assertIn('ground_truth',rows[1]);self.assertIn('scorer_fingerprint',rows[1])
            self.assertEqual(enabled,bool(ctor.call_count));self.assertEqual(enabled,5 in decoded)
            self.assertEqual(enabled,f'{scope}/val/rouge-L/mean' in metrics)
            if not enabled:self.assertEqual(0,record['timing_seconds']['rouge_l']);self.assertEqual('disabled',record['rouge_l_status'])
            evidence=os.environ.get('GAD_BC_EVIDENCE')
            if evidence:
                with open(evidence,'a') as handle:
                    handle.write(json.dumps({'backend':str(backend),'scope':scope,'rouge_l_enabled':enabled,
                        'rouge_constructor_calls':ctor.call_count,'teacher_only_decodes':sum(t in (5,6) for t in decoded),
                        'rouge_l_value':metrics.get(f'{scope}/val/rouge-L/mean'),'timing_seconds':record['timing_seconds'],
                        'acc':record['acc'],'provisional_acc':record['provisional_acc'],'evaluation_complete':record['evaluation_complete'],
                        'failures':record['failures'],'scorer_fingerprint':record['scorer_fingerprint'],
                        'transport':'CPU saved responses; no real generation'})+'\n')
            return {k:v for k,v in record.items() if k in ('acc','provisional_acc','count','scorer_fingerprint','unscorable_rate','grader_error_rate')}

    def test_actual_seqkd_warmup_gad_quick_full_rouge_on_off(self):
        root=Path(os.environ['GAD_TEST_BACKEND_ROOT'])
        ext=Path(os.environ['GAD_TEST_EXTENSION_ROOT'])
        for stage in ('seqkd','warmup','gad'):
            backend=root/stage if stage=='seqkd' else ext/stage
            for scope in ('quick','full'):
                with self.subTest(stage=stage,scope=scope):
                    self.assertEqual(self.validate(backend,scope,False),self.validate(backend,scope,True))

if __name__=='__main__':unittest.main(verbosity=2)
