"""Sparse process masks, paired policies, and durable send accounting."""
import json
import random
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'scoring/tests')]
import torch
from gad.extensions.process_sampling import select_process,logical_minibatches
from gad.extensions.critic_batch import supervision_identity,auxiliary_weights
from gad.extensions.online_v2 import process_failure_kind,request_belongs_to,StudentProcessSupervisionError
from gad.extensions.judge_client import RequestBudget
from gad.extensions.annotation_journal import AnnotationJournal
from rubric.core.losses import weighted_contribution,dimension_weights
from rubric.core.process_v2 import parse_events
from test_unified import event_record
import test_judge_protocol_v21 as protocol_tests


def pairs(n=64,nrollout=1):
    base=protocol_tests.OnlineTests().pairs()[0]
    result=[]
    for i in range(n):
        p=deepcopy(base)
        p['teacher']['response_id']='teacher_'+str(i//nrollout)
        p['student']['response_id']='student_'+str(i)
        p['producer_stage']='gad' if nrollout>1 else 'warmup'
        result.append(p)
    return result


def options(budget):
    return {'mode':'joint','process_sampling':{'max_unique_per_global_minibatch':budget,'seed':42}}


class SamplingTests(unittest.TestCase):
    def test_role_quotas_and_driver_budget(self):
        for budget,stage,n,nrollout,teacher,student in [
            (16,'warmup',64,1,8,8),(8,'warmup',64,1,4,4),
            (16,'gad',128,2,5,11),(8,'gad',128,2,3,5)]:
            ps=pairs(n,nrollout)
            selection,groups=select_process(ps,n,4,options(budget),12,stage)
            self.assertEqual(1,len(groups));g=groups[0]
            self.assertEqual(budget,g['selected_unique'])
            self.assertEqual((teacher,student),(g['teacher_selected_unique'],g['student_selected_unique']))
            self.assertEqual(student,sum(selection['student']))
            self.assertLessEqual(sum(selection['teacher']),teacher*nrollout)

    def test_groups_follow_rank_dispatch_not_contiguous_driver_chunks(self):
        ps=pairs(128,2)
        selection,groups=select_process(ps,64,4,options(16),20,'gad')
        self.assertEqual([list(range(0,16))+list(range(32,48))+list(range(64,80))+list(range(96,112)),
                          list(range(16,32))+list(range(48,64))+list(range(80,96))+list(range(112,128))],
                         [g['rows'] for g in groups])
        self.assertEqual([16,16],[g['selected_unique'] for g in groups])
        self.assertEqual(32,sum(g['selected_unique'] for g in groups))

    def test_recovery_is_deterministic_and_does_not_consume_rollout_rng(self):
        ps=pairs()
        random.seed(3);torch.manual_seed(4)
        a,b=random.getstate(),torch.get_rng_state().clone()
        first=select_process(ps,64,4,options(16),10,'warmup')
        self.assertEqual(a,random.getstate());self.assertTrue(torch.equal(b,torch.get_rng_state()))
        random.seed(999);torch.manual_seed(1000)
        self.assertEqual(first,select_process(ps,64,4,options(16),10,'warmup'))
        self.assertNotEqual(first,select_process(ps,64,4,options(16),11,'warmup'))

    def test_cross_role_duplicates_share_selection_and_one_budget_slot(self):
        ps=pairs(8)
        ps[1]['student']=deepcopy(ps[0]['teacher'])
        selection,groups=select_process(ps,8,2,options(8),2,'warmup')
        seen={}
        for r in selection:
            for i,p in enumerate(ps):
                rid=supervision_identity(p,r)
                if rid in seen:self.assertEqual(seen[rid],selection[r][i])
                seen[rid]=selection[r][i]
        self.assertEqual(8,sum(seen.values()))

    def test_null_and_correctness_modes_keep_full_existing_behavior(self):
        for opts in [options(None),{**options(8),'mode':'correctness_only'}]:
            selection,_=select_process(pairs(),64,4,opts,1,'warmup')
            self.assertTrue(all(all(v) for v in selection.values()))
        for bad in [0,1,-1,True]:
            with self.assertRaises(ValueError):select_process(pairs(),64,4,options(bad),1,'warmup')

    def test_sampled_auxiliary_gradient_matches_unique_global_reference(self):
        ps=pairs(32,2);ps[3]['student']=deepcopy(ps[0]['teacher'])
        selection,_=select_process(ps,16,4,options(8),7,'gad')
        identities=sorted({supervision_identity(p,r) for p in ps for r in selection})
        code={rid:i+1 for i,rid in enumerate(identities)}
        target={};mask={};features={}
        for role in selection:
            nums=[code[supervision_identity(p,role)] for p in ps]
            target[role]=torch.tensor([[i%2,(i%3)/2,1-(i%3)/2] for i in nums])
            mask[role]=torch.tensor([[1,float(selection[role][i]),float(selection[role][i])] for i in range(len(ps))])
            features[role]=torch.tensor(nums,dtype=torch.float32)/len(code)
        weights,counts=auxiliary_weights(ps,target,mask,16,4)
        for micro in [1,2,4]:
            theta=torch.tensor([.1,.2,.3],requires_grad=True)
            total=theta.sum()*0
            for rows in logical_minibatches(len(ps),16,4):
                for offset in range(0,len(rows),micro):
                    ids=rows[offset:offset+micro]
                    for role in selection:
                        logits=features[role][ids,None]*theta[None,:]
                        contribution,_=weighted_contribution(logits,target[role][ids],mask[role][ids],
                            weights[role][ids],counts[ids[0]],4,'fixed_dimension_gated',4)
                        total=total+contribution/4
            total.backward();actual=theta.grad.clone()
            theta=torch.tensor([.1,.2,.3],requires_grad=True);reference=theta.sum()*0
            for rows in logical_minibatches(len(ps),16,4):
                unique={}
                for i in rows:
                    for role in selection:unique[supervision_identity(ps[i],role)]=(features[role][i],target[role][i],mask[role][i])
                valid=sum((v[2] for v in unique.values()),torch.zeros(3))
                coefficient=dimension_weights(valid,'fixed_dimension_gated',4)[2]
                for x,y,m in unique.values():reference=reference+(torch.nn.functional.binary_cross_entropy_with_logits(x*theta,y,reduction='none')*m*coefficient).sum()
            reference.backward()
            torch.testing.assert_close(actual,theta.grad,atol=1e-6,rtol=1e-6)

    def test_gate_saturates_at_four_without_scaling_by_sampling_fraction(self):
        for n in [0,1,3,4,8]:
            before,gate,after=dimension_weights(torch.tensor([n,n,n],dtype=torch.float32),'fixed_dimension_gated',4)
            self.assertAlmostEqual(min(n/4,1),gate[0].item())
            self.assertAlmostEqual(0 if n==0 else min(n/4,1)/n/3,after[0].item())


class OnlineSamplingTests(unittest.TestCase):
    def test_cache_never_expands_training_selection_or_poisons_unselected(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=protocol_tests.OnlineTests().owner(Path(directory)/'warmup','valid')
            owner.options.update(process_sampling=options(8)['process_sampling'],semantic_check=False)
            ps=pairs(32)
            saved=[]
            for step in [1,2]:
                batch=SimpleNamespace(batch={})
                stats=owner.annotate(ps,batch,32,4,step)
                total=sum(batch.batch[r+'_rubric_mask'][:,1].sum().item() for r in ['teacher','student'])
                self.assertEqual(8,total)
                self.assertEqual(4,stats['rubric/process_student_all_selected'])
                self.assertEqual(4,stats['rubric/process_student_all_any_valid'])
                self.assertEqual(8,stats['rubric/process_sampling_selected_unique_sum'])
                saved.append({r:batch.batch[r+'_rubric_targets'][:,0].clone() for r in ['teacher','student']})
            for r in saved[0]:torch.testing.assert_close(saved[0][r],saved[1][r])
            self.assertGreater(owner.judge.calls,8)
            self.assertLess(owner.judge.calls,32)
            selection,_=select_process(ps,32,4,owner.options,1,'warmup')
            unused=next((p,r) for i,p in enumerate(ps) for r in ['teacher','student']
                        if not selection[r][i] and owner.cache.get(owner.cache_key(p,r)) is None)
            self.assertIsNone(owner.cache.get(owner.cache_key(*unused)))

    def test_selected_student_zero_coverage_uses_streak_not_teacher_success(self):
        with tempfile.TemporaryDirectory() as directory:
            owner=protocol_tests.OnlineTests().owner(Path(directory)/'warmup','unknown')
            owner.options.update(process_sampling=options(8)['process_sampling'],semantic_check=False)
            ps=pairs(16)
            for step in [1,2]:owner.annotate(ps,SimpleNamespace(batch={}),16,4,step)
            with self.assertRaises(StudentProcessSupervisionError) as caught:
                owner.annotate(ps,SimpleNamespace(batch={}),16,4,3)
            self.assertEqual('student_process_empty_streak',str(caught.exception))
            self.assertEqual(4,caught.exception.stats['rubric/process_student_all_selected'])

    def test_invalid_unjudgeable_is_unknown_and_does_not_restore_global_labels(self):
        row=event_record(progress=1);row['completion']='complete'
        row['events']['e2']={**deepcopy(row['events']['e1']),'type':'unjudgeable',
            'supporting_span_ranges':{'r1':{'start':'s999','end':'s999'}}}
        result=parse_events(json.dumps(row),'2x=4')
        self.assertFalse(result['verified_progress']['valid'])
        self.assertFalse(result['unresolved_error_severity']['valid'])
        self.assertEqual('mathematical_unknown',process_failure_kind(result))
        row=event_record();row['events']['e1']['supporting_span_ranges']={'r1':{'start':'s999','end':'s999'}}
        result=parse_events(json.dumps(row),'2x=4')
        result['unresolved_error_severity'].update(valid=False,value=None)
        self.assertEqual('contract_rejected',process_failure_kind(result))



if __name__=="__main__":unittest.main()
