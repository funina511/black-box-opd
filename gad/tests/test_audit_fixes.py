"""Regression cases from docs/核查问题修复计划.md; no generation or training."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'scoring/tests')]
from scoring.math import score,correctness_label,evaluation_value,_grade,_clean,compute_score
from evaluate.rescore import summarize
from rubric.core.process_v2 import parse_events
from test_unified import event_record
from config import Config,compose_config,validate_config
import pipeline
import math_metrics
from backend_patches import MathValidationPatch


class AnswerFixTests(unittest.TestCase):
    def test_markdown_prefixes_and_bold_values(self):
        for prefix in ['- ','* ','+ ','> ','1. ','1) ','> > ','> - ','### ']:
            for value in ['42','**42**','__42__']:
                with self.subTest(prefix=prefix,value=value):
                    result=score(prefix+'**Answer:** '+value,'42')
                    self.assertEqual('42',result['pred']);self.assertIs(result['correct'],True)
        self.assertEqual('42',_clean('**42'));self.assertEqual('42',_clean('42**'))

    def test_numeric_whitespace_is_not_concatenation(self):
        for value in ['1 2','1 2 3','1 .2']:
            self.assertFalse(_grade(value,value.replace(' ','')))
        self.assertTrue(_grade('5 cm','5cm'));self.assertTrue(_grade('2 x','2x'))

    def test_template_does_not_replace_answer_and_malformed_does_not_become_wrong(self):
        for suffix in ['\n\nFormat note:\nAnswer: <the number>','\nThus ... Answer: $Answer where ...']:
            self.assertTrue(score('Answer: 42'+suffix,'42')['correct'])
        for text in ['Answer: <the number>','Answer: the number should be written here','Answer: **']:
            result=score(text,'42');self.assertEqual('malformed',result['parse_status'])
            self.assertFalse(correctness_label(result)['valid']);self.assertIsNone(evaluation_value(result))

    def test_bad_box_does_not_swallow_later_box(self):
        self.assertTrue(score('We tried \\boxed{ here.\nFinal Answer: \\boxed{42 cm}','42 cm')['correct'])
        result=score('\\boxed{ oops\nAnswer: \\boxed{1} and \\boxed{2}','1')
        self.assertEqual('ambiguous',result['parse_status'])

    def test_units_angles_and_exponential_grouping(self):
        self.assertTrue(_grade(r'42\text{ cm}','42 cm'))
        self.assertTrue(_grade(r'90^\circ',r'90^{\circ}'))
        self.assertTrue(_grade(r'e^{i\pi}','-1'))
        self.assertFalse(_grade('42 cm','42 m'))

    def test_invalid_scores_propagate_without_official_accuracy(self):
        for text in ['Answer: <number>','Answer: \\boxed{','no final answer']:
            result=score(text,'42');label=correctness_label(result)
            self.assertEqual(label['valid'],evaluation_value(result) is not None)
        result=compute_score('Answer: <number>','42')
        self.assertIsNone(result['acc']);self.assertIsNone(result['score'])
        result.update(global_index=0,rollout_index=0)
        metrics=summarize([result],expected_questions=[0],n=1)
        self.assertFalse(metrics['evaluation_complete']);self.assertNotIn('sample_accuracy',metrics)
        self.assertEqual(1,metrics['unscorable_sample_count'])


class EvidenceFixTests(unittest.TestCase):
    def parse(self,row,text):return parse_events(json.dumps(row),text)
    def test_same_error_span_cannot_be_its_own_repair(self):
        text='2x=4\n\nx=3'
        row=event_record();row['events']['e2']={'type':'explicit_error','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}},'trajectory_claim':'bad division', 'judge_verification': 'bad division','related_event_ids':[]}
        row['events']['e3']={'type':'corrected_error','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}},'trajectory_claim':'judge fixed it mentally', 'judge_verification': 'judge fixed it mentally','error_event_id':'e2'}
        result=self.parse(row,text)
        self.assertFalse(result['unresolved_error_severity']['valid'])
        self.assertIn('correction_reuses_error_span',result['_event_validity']['e3']['reasons'])
        row['events']['e3']['supporting_span_ranges']={'r1': {'start': 's3', 'end': 's3'}}
        result=self.parse(row,text+'\nCorrecting: x=2; 2*2=4.')
        self.assertEqual(0,result['unresolved_error_severity']['value'])

    def test_unknown_math_blocks_global_but_neutral_does_not(self):
        row=event_record(progress=1);row['completion']='complete'
        row['events']['e2']={'type':'unjudgeable','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}},'trajectory_claim':'cannot check the next math step', 'judge_verification': 'cannot check the next math step','related_event_ids':[]}
        result=self.parse(row,'2x=4\nx=12/')
        for d in ['verified_progress','unresolved_error_severity']:self.assertFalse(result[d]['valid'])
        row['events']['e2']['type']='neutral'
        result=self.parse(row,'2x=4\nAn unrelated note')
        self.assertTrue(result['verified_progress']['valid']);self.assertTrue(result['unresolved_error_severity']['valid'])

    def test_first_mask_reason_is_stable_and_not_rechecked(self):
        row=event_record(progress=1);row['events']['e1']['supporting_span_ranges']={'r1': {'start': 's999', 'end': 's999'}}
        result=self.parse(row,'2x=4')
        self.assertEqual(['invalid_label_event_dependency'],result['_mask_reasons']['verified_progress'])
        self.assertEqual('invalid_label_event_dependency',result['verified_progress']['contract_violation'])

    def test_repair_grammar_cannot_generate_an_empty_link(self):
        from rubric.core.process_v2 import JUDGE_SCHEMA_V2
        branch=JUDGE_SCHEMA_V2['properties']['events']['properties']['e1']['anyOf'][1]
        self.assertIn('error_event_id',branch['required'])
        self.assertEqual(['e1','e2','e3','e4'],branch['properties']['error_event_id']['enum'])

    def test_parser_exposes_its_semantic_limit(self):
        row=event_record(progress=1);row['completion']='complete'
        result=self.parse(row,'We compute 1+2*3+4*5+6.\nAnswer: 721')
        self.assertTrue(result['verified_progress']['valid'])
        self.assertEqual('not_checked',result['_validation_scope']['semantic_entailment'])

    def test_epoch_final_export_uses_resolved_epoch_budget(self):
        from test_pipeline_cpu import base_config,write_yaml
        from config import load_config
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg_path=root/'train.yaml'
            data=base_config(root);data['budget']['stage_epochs']={'seqkd':1,'warmup':1,'gad':3}
            write_yaml(cfg_path,data);cfg=load_config(cfg_path)
            # 32 prompts / batch8: Warmup4, absolute GAD target12 (8 added),
            # independent of the legacy warmup_steps10/gad_steps6 defaults.
            target=pipeline.export_final_actor(cfg,root/'run',dry_run=True)
            self.assertIn('global_step_12/',target)


class PipelineFixTests(unittest.TestCase):
    def test_numeric_strings_and_bad_types_do_not_crash_extensions(self):
        base=compose_config(settings=['rubric.enabled=true','rubric.mode=correctness_only'])
        cfg=Config({**base.data,'rollout.n':'2','budget.stage_epochs.gad':'2'},path=base.path)
        self.assertEqual(2,cfg.get('rollout.n'));self.assertFalse(validate_config(cfg))
        for value in ['bad',True,2.5]:
            bad=Config({**base.data,'rollout.n':value},path=base.path)
            self.assertTrue(any('rollout.n' in x for x in validate_config(bad)))
        cfg=Config({**base.data,'budget.stage_epochs.gad':'1'},path=base.path)
        self.assertTrue(any('exceed' in x for x in validate_config(cfg)))

    def test_stage_local_n_wins_and_specific_stage_key_wins_inside_overlay(self):
        cfg=compose_config();cfg.stage_overrides={'warmup':{'rollout.n':'8'},'seqkd':{'rollout.n':'8'}}
        for stage in ['warmup','seqkd']:self.assertEqual(8,pipeline.stage_rollout_n(cfg.for_stage(stage),stage))
        cfg.stage_overrides['warmup']['rollout.n_warmup']='4'
        self.assertEqual(4,pipeline.stage_rollout_n(cfg.for_stage('warmup'),'warmup'))

    def test_disable_validation_requires_no_metric_artifact(self):
        for version in ['legacy_card_v1','math_unified_v1']:
            cfg=compose_config(settings=['run.disable_validation=true','run.engineering_smoke=true','scoring.math.version='+version])
            self.assertFalse(pipeline.validation_metrics_expected(cfg)[0])
            args=pipeline.build_overrides(cfg,'warmup',model_path='model',critic_path='critic',out_dir=Path('/tmp/audit'),total_steps=1,epochs=1)
            self.assertIn('trainer.val_before_train=False',args);self.assertIn('trainer.test_freq=0',args)

    def test_incomplete_artifact_and_grader_errors_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);ckpt=out/'global_step_10';ckpt.mkdir()
            (out/'latest_checkpointed_iteration.txt').write_text('10')
            for changes in [{'evaluation_complete':False},{'grader_error_rate':1.0}]:
                row={'global_step':10,'acc':0.0,'count':64,**changes}
                (out/'validation_math_metrics.jsonl').write_text(json.dumps(row)+'\n')
                with patch('pipeline.verify_checkpoint_dir',return_value=[]):
                    result=pipeline.verify_stage_artifacts('seqkd',out,expected_step=10,math_metrics_expected=True,strict_evaluation=True)
                self.assertFalse(result['ok']);self.assertTrue(result['evaluation_problems'])

    def test_full_validation_counts_every_requested_rollout(self):
        from test_pipeline_cpu import base_config,write_yaml
        from config import load_config
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg_path=root/'train.yaml';data=base_config(root)
            data['validation']={'quick_enabled':True,'final_full':True};data['rollout']['val_n']=3
            write_yaml(cfg_path,data);cfg=load_config(cfg_path)
            with patch('pipeline.verify_stage_artifacts',side_effect=RuntimeError('captured verification')) as verify:
                with self.assertRaisesRegex(RuntimeError,'captured verification'):
                    pipeline.finish_stage(cfg,root/'run','gad',{'out_dir':str(root/'out'),'total_steps':10},
                        process=None,started='test',math_expected=True,math_reason='test',trainer_skipped=True)
            self.assertEqual(24,verify.call_args.kwargs['full_count'])

    def test_scorer_self_tests_and_legacy_environment(self):
        for path in [ROOT/'scoring/math.py',ROOT/'sample/math_scorer.py']:
            with patch.object(math_metrics,'SCORER',math_metrics.load_scorer(path)):
                ok,problems=math_metrics.self_test(verbose=False);self.assertTrue(ok,problems)
        with patch.dict(os.environ,{'GAD_SCORING_MATH_CONFIG':'{"version":"legacy_card_v1"}'}),patch.object(math_metrics,'SCORER',None):
            self.assertTrue(math_metrics.score_response('Answer: 42','42')['acc'])

    def test_legacy_grader_failure_cannot_be_complete_accuracy(self):
        namespace={'os':os,'json':json}
        exec(MathValidationPatch.HELPERS.replace('@@MARKER@@','test'),namespace)
        class Exploding:
            def compute_score(self,*args):raise RuntimeError('grader unavailable')
        row=namespace['_gad_score_math_row'](Exploding(),'Answer: 1','1')
        metrics=namespace['_gad_math_metrics']([{**row,'length':1,'truncated':False}])
        self.assertFalse(metrics['evaluation_complete']);self.assertEqual(1,metrics['grader_error_rate']);self.assertIsNone(metrics['acc'])


if __name__=='__main__':unittest.main()
