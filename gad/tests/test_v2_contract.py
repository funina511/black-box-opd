"""V2 caller, metadata, seed, cache and loss controls without extra trainers."""
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'gad/tools'))
from config import compose_config,validate_config
from extension_config import hydra_overrides
from pipeline import build_overrides
from gad.extensions.teacher_provenance import bind_selected
from scoring.math import score,identity
sys.path.insert(0,str(ROOT/'scoring/tests'))
from test_unified import event_record


class V2ContractTests(unittest.TestCase):
    def test_selected_teacher_duplicates(self):
        answers=['same','same'];sampling={'truncated':[True,False],'finish_reasons':['length','stop']}
        self.assertFalse(bind_selected(answers,sampling,'same',1)['source_truncated'])
        self.assertTrue(bind_selected(answers,sampling,'same',0)['source_truncated'])
        with self.assertRaises(ValueError):bind_selected(answers,sampling,'same')
        self.assertEqual(1,bind_selected(['cut','full'],sampling,'full')['selected_teacher_index'])
        from sample.build_gad_data import pick_teacher_candidate
        self.assertEqual(('same','first_nontruncated',1),pick_teacher_candidate(answers,select='nontruncated',ground_truth='2',truncated=[True,False]))

    def test_explicit_seed_all_groups_and_version(self):
        for rubric,replay in ((False,False),(True,False),(False,True),(True,True)):
            cfg=compose_config(settings=[f'rubric.enabled={str(rubric).lower()}',f'replay.enabled={str(replay).lower()}'])
            for stage in ('warmup','gad'):
                overrides=build_overrides(cfg,stage,model_path=cfg.require('models.student'),critic_path=cfg.require('models.discriminator_init'),out_dir=Path('/tmp/test'),total_steps=2,epochs=1)
                self.assertIn('actor_rollout_ref.rollout.seed=42',overrides)
                self.assertIn('++data.seed=42',overrides)
            cfg.data['rollout.seed']=None
            self.assertTrue(any('explicit rollout.seed' in x for x in validate_config(cfg)))

    def test_v2_prefix_cache_and_three_consumers(self):
        import torch
        from gad.extensions.online_rubric import OnlineRubric
        from rubric.core.contracts import file_hash
        from rubric.core.process_v2 import parse_events
        cfg=compose_config();options={key[7:]:value for key,value in cfg.data.items() if key.startswith('rubric.')}
        options['judge']={key[6:]:options.pop(key) for key in list(options) if key.startswith('judge.')}
        options.update(spec_hash=file_hash(options['spec']),judge_fingerprint='test',scorer_path=cfg.require('validation.scorer_path'),
            scoring_math={key[13:]:value for key,value in cfg.data.items() if key.startswith('scoring.math.')})
        class Judge:
            def __init__(self,*args):self.options=options['judge'];self.api_peak=0;self.api_ledger=[]
            def annotate_one(self,rid,messages,deadline):return rid,self.annotate({rid:messages})[rid]
            def health(self,**kwargs):pass
            def annotate(self,requests,**kwargs):
                return {key:{'parsed':parse_events(json.dumps(event_record()),'2x=4'), 'attempts':1} for key in requests}
        pair={'pair_id':'p','prompt_hash':'prompt','ground_truth':'2','prompt':'solve 2x+3=7','prompt_truncated':False,
            'teacher':{'response_id':'T','response':'2x=4','token_ids':[1],'truncated':True,'source_truncated':True},
            'student':{'response_id':'S','response':'2x=4','token_ids':[2],'truncated':True,'source_truncated':True}}
        with tempfile.TemporaryDirectory() as d,patch('gad.extensions.online_rubric.JudgeClient',Judge):
            owner=OnlineRubric(options,Path(d)/'warmup')
            for invocation in range(2):
                batch=SimpleNamespace(batch={});stats=owner.annotate([deepcopy(pair)],batch,1,1,invocation)
                self.assertEqual([[1.,1.,1.]],batch.batch['student_rubric_mask'].tolist())
                self.assertEqual([[0.,.5,0.]],batch.batch['student_rubric_targets'].tolist())
                self.assertEqual(1,stats['rubric/truncated_student']);self.assertLessEqual(stats['rubric/coverage_verified_progress_student'],1)
                result=owner.scorer('2x=4','2');self.assertEqual(score('2x=4','2'),result)
                from scoring.math import compute_score
                for key,value in result.items():self.assertEqual(value,compute_score('2x=4','2')[key])
            self.assertEqual(0,stats['rubric/judge_requests']);self.assertEqual(1,stats['rubric/cache_hits_student'])
            altered=deepcopy(pair);altered['teacher']['source_truncated']=False
            self.assertNotEqual(owner.cache_key(pair,'teacher'),owner.cache_key(altered,'teacher'))

    def test_lambda_zero_and_auxiliary_sign(self):
        import torch
        from rubric.core.losses import pair_loss,auxiliary_loss
        teacher=torch.tensor([[.7,.2,-.3,.4]],requires_grad=True);student=torch.tensor([[.1,-.2,.3,-.4]],requires_grad=True)
        bt=pair_loss(teacher[:,0],student[:,0]).mean();targets=torch.tensor([[1.,.5,1.]])
        aux=auxiliary_loss(student[:,1:],targets,torch.ones(1,3))
        old=torch.autograd.grad(bt,(teacher,student),retain_graph=True)
        controlled=torch.autograd.grad(bt+0*aux,(teacher,student),retain_graph=True)
        for a,b in zip(old,controlled):torch.testing.assert_close(a,b)
        grad=torch.autograd.grad(aux,student)[0];self.assertLess(grad[0,3],0) # error=1 trains severity upwards, no negative loss.


if __name__=='__main__':unittest.main()
