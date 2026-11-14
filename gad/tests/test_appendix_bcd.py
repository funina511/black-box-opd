"""Targeted startup/resume regressions use temporary files, never GPUs."""
import json,sys,tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'gad/tests')]
import pipeline
from config import compose_config,validate_config,load_config
from test_pipeline_cpu import base_config,write_yaml,make_model_dir
import test_pipeline_cpu as fixture
from gad.extensions.checkpoint_state import save
from cpu_fixtures import compose_cpu_config as compose_config


class HandoffTests(unittest.TestCase):
    def test_initial_marker_and_existing_extension_are_distinct(self):
        for kind in ('missing','bad_source','bad_step','bad_mode','initial','extension','stale_marker','broken_extension','baseline'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);path=root/'config.yaml';write_yaml(path,base_config(root));cfg=load_config(path)
                cfg.data['rubric.enabled']=kind!='baseline';cfg.data['rubric.mode']='correctness_only'
                run=root/'run';out=run/'gad';fixture.StageArtifactTests()._complete_checkpoint(out,10);cp=out/'global_step_10'
                for role in ('actor','critic'):make_model_dir(cp/role/'huggingface')
                manifest={'step':10,'source_checkpoint':str(run/'warmup/global_step_10'),**{role+'_hf':str(cp/role/'huggingface') for role in ('actor','critic')}}
                (run/'handoff_manifest.json').write_text(json.dumps(manifest));(run/'runtime').mkdir()
                (run/'runtime/gad_handoff_identity.json').write_text(json.dumps(pipeline.handoff_identity(cfg,run)))
                marker={'source':manifest['source_checkpoint'],'step':10,'same_stage_resume':False}
                if kind=='bad_source':marker['source']='another-source'
                if kind=='bad_step':marker['step']=11
                if kind=='bad_mode':marker['same_stage_resume']=True
                if kind!='missing':(cp/'handoff_initialization.json').write_text(json.dumps(marker))
                if kind in ('extension','stale_marker','broken_extension'):
                    save(SimpleNamespace(identity='fixture',world_size=1,stage='gad',stage_start=10,stage_step=0,committed_step=10,replay=None,rubric=None),cp)
                if kind=='broken_extension':(cp/'training_extensions/sampling_rng.pt').write_bytes(b'corrupt')
                if kind in ('missing','bad_source','bad_step','bad_mode','broken_extension'):
                    with self.assertRaises(pipeline.PipelineError):pipeline.resolve_gad_resume(cfg,run,added_steps=6,resume_requested=True)
                else:
                    result=pipeline.resolve_gad_resume(cfg,run,added_steps=6,resume_requested=True)
                    self.assertEqual('resume' if kind in ('extension','stale_marker') else 'handoff',result['mode'])
                    self.assertEqual(10,result['from_step'])

class VersionTests(unittest.TestCase):
    def test_spec_versions_and_malformed_specs_fail_before_judge(self):
        cfg=compose_config(overrides=[str(ROOT/'configs/methods/correctness_only.yaml')])
        self.assertEqual([],validate_config(cfg))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'spec.yaml'
            import yaml
            spec=yaml.safe_load(Path(cfg.require('rubric.spec')).read_text())
            contents=['[]',': bad yaml: [',yaml.safe_dump({**spec,'version':'math_v1'}),yaml.safe_dump({**spec,'input_protocol':'old'}),yaml.safe_dump({**spec,'schema_version':'rubric.events.v2.6'})]
            for content in contents:
                path.write_text(content);other=deepcopy(cfg);other.data['rubric.spec']=str(path)
                self.assertTrue(validate_config(other))
            other=deepcopy(cfg);other.data.update({'rubric.version':'math_v1','scoring.math.version':None,'rubric.semantic_check':False})
            self.assertTrue(any('spec.version' in x for x in validate_config(other)))
            other.data['rubric.spec']=str(ROOT/'rubric/specs/math_v1.yaml')
            from rubric.core.contracts import LAYOUT
            other.data.update({'rubric.head_layout':list(LAYOUT),'rubric.truncated_process_policy':'mask',
                               'rubric.prompt':str(ROOT/'rubric/prompts/judge_math_v1.txt'),
                               'scoring.math.version':'legacy_card_v1',
                               'validation.scorer_path':str(ROOT/'sample/math_scorer.py')})
            self.assertEqual([],validate_config(other))
            other=deepcopy(cfg);other.data['rubric.spec']=str(Path(d)/'missing.yaml')
            self.assertTrue(any('unreadable' in x for x in validate_config(other)))
            prompt=Path(d)/'historical.txt';prompt.write_text('Historical prompt without protocol declaration.')
            other=deepcopy(cfg);other.data['rubric.prompt']=str(prompt)
            self.assertEqual([],validate_config(other))

class SelectionTests(unittest.TestCase):
    def test_unresolved_nontruncated_is_not_dropped_for_truncated_correct(self):
        from unittest.mock import patch
        from sample.build_gad_data import pick_teacher_candidate,UnresolvedCandidates
        def graded(text,gt):return {'score':None,'grade_status':'timeout'} if text=='unknown' else {'score':int(text=='correct'),'grade_status':'ok'}
        with patch('sample.build_gad_data.compute_score',side_effect=graded):
            for answers,flags in [(['unknown','correct'],[False,True]),(['correct','unknown'],[True,False])]:
                with self.assertRaises(UnresolvedCandidates):pick_teacher_candidate(answers,select='correct-nontruncated',ground_truth='2',truncated=flags)
                self.assertEqual(answers[0],pick_teacher_candidate(answers,select='all',ground_truth='2',truncated=flags)[0])
            self.assertEqual((None,'correct_but_truncated',None),pick_teacher_candidate(['wrong','correct'],select='correct-nontruncated',ground_truth='2',truncated=[False,True]))
            self.assertEqual(('correct','first_correct_nontruncated',1),pick_teacher_candidate(['unknown','correct'],select='correct-nontruncated',ground_truth='2',truncated=[False,False]))

if __name__=='__main__':unittest.main()
