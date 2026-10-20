"""Research V2.4: bounded semantic review and fixed-label numerical controls."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'scoring/tests'),str(ROOT/'gad/tests')]
from test_unified import event_record
from test_online_rubric import pair
from rubric.core.process_v2 import parse_events,ProtocolError
from rubric.core.semantic_v24 import apply_review,parse_review,SCHEMA,VERSION
from rubric.core.losses import weighted_contribution,dimension_weights,LOSS_MODES,auxiliary_loss
from gad.extensions.critic_batch import auxiliary_weights
from gad.extensions.judge_client import JudgeClient,ServiceUnavailable,guidance_schema
from gad.extensions.semantic_annotation import review_one
from gad.tools.quarantine_rubric_cache import quarantine


def range_units(ids):
    return {f'r{i}':({'start':x,'end':x} if isinstance(x,str) else x) for i,x in enumerate(ids,1)}


def parsed(row=None,response='2x=4\nx=2'):
    return parse_events(json.dumps(row or event_record()),response)


def decision(verdict='supported',spans=None):
    return {'verdict':verdict,'supporting_span_ranges':range_units(spans or ['s1']),'refuting_span_ranges':{},'event_ids':[], 'reason':'specific original assertion checked'}


def event_decision(**kwargs):
    row=decision(**kwargs);row.pop('event_ids');return row


def review():
    return {'version':VERSION,'labels':{d:decision() for d in ('verified_progress','unresolved_error_severity')},'events':{'e1':event_decision()}}


class ProtocolTests(unittest.TestCase):
    def test_multiple_source_spans_and_provenance(self):
        row=event_record();row['events']['e1']['supporting_span_ranges']={'r1': {'start': 's1', 'end': 's1'}, 'r2': {'start': 's2', 'end': 's2'}}
        result=parsed(row)
        self.assertEqual(['2x=4\n','x=2'],[r['quote'] for r in result['_source_evidence']['e1']['spans']])
        self.assertEqual('passed',result['_structural_validation_status'])
        self.assertEqual('pending',result['_semantic_review']['status'])
        self.assertFalse(result['_semantic_review']['human_verified'])
        self.assertTrue(result['_training_eligible']['verified_progress'])
        self.assertEqual('not_checked',result['_validation_scope']['semantic_entailment'])

    def test_bad_span_types_duplicates_and_legacy_wire(self):
        for ids in ([], ['s999'], [None], [{}]):
            row=event_record();row['events']['e1']['supporting_span_ranges']=range_units(ids)
            self.assertFalse(parsed(row)['verified_progress']['valid'])
        row=event_record();row['schema_version']='rubric.events.v2.4'
        with self.assertRaises(ProtocolError):parsed(row)

    def test_repair_can_include_context_but_must_have_replacement_location(self):
        row=event_record(progress=1);row['completion']='complete'
        base=deepcopy(row['events']['e1'])
        row['events']['e2']={**base,'type':'explicit_error','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}}}
        row['events']['e3']={**base,'type':'corrected_error','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}, 'r2': {'start': 's3', 'end': 's3'}},'error_event_id':'e2'}
        row['events']['e3'].pop('related_event_ids')
        self.assertTrue(parsed(row,'2x=4\nx=3\nCorrection: x=2')['unresolved_error_severity']['valid'])
        row['events']['e3']['supporting_span_ranges']={'r1': {'start': 's2', 'end': 's2'}}
        self.assertFalse(parsed(row,'2x=4\nx=3\nCorrection: x=2')['unresolved_error_severity']['valid'])
        row['events']['e2']['supporting_span_ranges']={'r1': {}}
        self.assertFalse(parsed(row,'2x=4\nx=3\nCorrection: x=2')['unresolved_error_severity']['valid'])

    def test_relocation_once_and_partial_masks(self):
        row=event_record();row['events']['e1']['supporting_span_ranges']={'r1': {'start': 's999', 'end': 's999'}}
        initial=parsed(row)
        self.assertFalse(initial['verified_progress']['valid'])
        result=apply_review(initial,'2x=4\nx=2',parse_review(json.dumps(review())))
        self.assertTrue(result['verified_progress']['valid'])
        self.assertEqual({'r1':{'start':'s999','end':'s999'}},result['_initial_record']['events']['e1']['supporting_span_ranges'])
        self.assertEqual('model_supported',result['_semantic_review']['dimensions']['verified_progress'])
        self.assertFalse(result['_semantic_review']['human_verified'])
        row=event_record(error=1);row['events']['e2']={**deepcopy(row['events']['e1']),'type':'explicit_error','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}}};row['unresolved_error_severity']['event_ids']=['e2']
        r=review();r['labels']['verified_progress']=decision('refuted');r['events']['e2']=event_decision(spans=['s2'])
        result=apply_review(parsed(row),'2x=4\nx=3',r)
        self.assertFalse(result['verified_progress']['valid']);self.assertTrue(result['unresolved_error_severity']['valid'])

    def test_missing_budget_disagreement_or_refutation_never_defaults_to_pass(self):
        for r in (None, {**review(),'events':{}}, {**review(),'labels':{d:decision('unresolved') for d in ('verified_progress','unresolved_error_severity')}}):
            result=apply_review(parsed(),'2x=4\nx=2',r,failure='budget_exhausted' if r is None else None)
            self.assertFalse(result['verified_progress']['valid'])
        r=review();r['events']['e1']['refuting_span_ranges']={'r1': {'start': 's2', 'end': 's2'}}
        self.assertFalse(apply_review(parsed(),'2x=4\nx=2',r)['verified_progress']['valid'])
        with self.assertRaises(ProtocolError):parse_review('{"version":"x"}')

    def test_review_event_and_label_shapes_are_distinct_and_old_version_rejected(self):
        r=review();parse_review(json.dumps(r))
        self.assertNotIn('event_ids',r['events']['e1'])
        r['events']['e1']['event_ids']=[]
        with self.assertRaises(ProtocolError):parse_review(json.dumps(r))
        r=review();r['version']='rubric.targeted_review.v2'
        with self.assertRaises(ProtocolError):parse_review(json.dumps(r))
        prompt=(ROOT/'rubric/prompts/judge_math_process_v2.txt').read_text()
        self.assertNotIn('s408',prompt);self.assertNotIn('s24',prompt)

    def test_parser_never_calls_a_model(self):
        with patch.object(JudgeClient,'request',side_effect=AssertionError('hidden call')):
            self.assertTrue(parsed()['verified_progress']['valid'])


class BudgetTests(unittest.TestCase):
    def client(self,budget=2):
        from test_online_rubric import ClientTests
        opts=ClientTests().options();opts.update(rubric_version='math_process_v2',enable_thinking=False,request_budget=budget)
        return JudgeClient(opts,'f',SimpleNamespace(apply_chat_template=lambda *a,**k:[1]))

    def test_first_pass_and_transport_retry_share_one_budget(self):
        c=self.client(1)
        with patch.object(c,'request',side_effect=ServiceUnavailable('transport')) as call:
            _,r=c.annotate_one('x',[],time.monotonic()+10)
        self.assertEqual(1,call.call_count);self.assertEqual(1,r['attempts']);self.assertEqual('judge_request_budget_exhausted',r['error'])

    def test_targeted_is_one_attempt_without_retry(self):
        c=self.client()
        with patch.object(c,'request',side_effect=ServiceUnavailable('transport')) as call:
            r=c.targeted('x',[],time.monotonic()+10)
            again=c.targeted('x',[],time.monotonic()+10)
        self.assertEqual(1,call.call_count);self.assertEqual(1,r['attempts']);self.assertEqual(0,again['attempts'])
        self.assertEqual(1,c.request_count)

    def test_client_futures_timeout_preserves_partial_results(self):
        import threading
        release=threading.Event();client=self.client()
        def annotate_one(rid,messages,deadline):
            if rid=='slow':release.wait(.2)
            return rid,{'parsed':parsed(),'attempts':1}
        try:
            with patch.object(client,'annotate_one',side_effect=annotate_one):
                with self.assertRaisesRegex(ServiceUnavailable,'whole-step judge deadline') as caught:
                    client.annotate({'fast':[],'slow':[]},deadline=time.monotonic()+.05,health_checked=True)
            self.assertIn('fast',caught.exception.results)
        finally:release.set()

    def test_context_overflow_does_not_spend_a_generation_call(self):
        c=self.client();c.options['max_context_tokens']=1
        r=c.targeted('x',[],time.monotonic()+10)
        self.assertEqual('judge_context_exceeded',r['error']);self.assertEqual(0,c.request_count)

    def test_concurrent_reservations_respect_total_budget(self):
        from concurrent.futures import ThreadPoolExecutor
        c=self.client(24)
        with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(lambda _:c.reserve_request(),range(100)))
        self.assertEqual(24,sum(results));self.assertEqual(24,c.request_count)

    def test_targeted_technical_failures_are_not_frozen_into_success_cache(self):
        from test_judge_protocol_v21 import OnlineTests
        from gad.extensions.online_v2 import StudentProcessSupervisionError
        helper=OnlineTests()
        with tempfile.TemporaryDirectory() as d:
            owner=helper.owner(Path(d)/'warmup','valid');owner.options['semantic_check']=True
            owner.judge.targeted=lambda *args:{'error':'judge_request_budget_exhausted','attempts':0}
            with self.assertRaises(StudentProcessSupervisionError) as caught:helper.run_step(owner)
            self.assertEqual('student_process_all_technical_failure',str(caught.exception))
            self.assertEqual([],list(owner.cache.directory.glob('*.json')))

    def test_double_annotation_does_not_trigger_second_reviewer(self):
        c=SimpleNamespace(targeted=lambda *a:{'error':'budget','attempts':0})
        r=review_one(c,'p','2x=4\nx=2',{'request_id':'x','parsed':parsed()},time.monotonic()+10)
        with patch.object(c,'targeted',side_effect=AssertionError('second request')):
            self.assertEqual(r,review_one(c,'p','2x=4\nx=2',r,time.monotonic()+10))


class CleanupTests(unittest.TestCase):
    def test_graceful_timeout_still_forces_only_owned_group(self):
        import subprocess
        import calibrate_rubric_judge as calibration
        with patch('calibrate_rubric_judge.subprocess.run',side_effect=subprocess.TimeoutExpired('stop',1)), patch('calibrate_rubric_judge.time.monotonic',return_value=100.), patch('calibrate_rubric_judge.os.killpg') as kill, patch('calibrate_rubric_judge.Path.exists',return_value=False):
            record=calibration.stop_owned_service(['service'],99999999,'birth',100.)
        self.assertEqual('graceful_stop_timeout',record['error'])
        kill.assert_called_once_with(99999999,calibration.signal.SIGKILL)

    def test_python310_futures_timeout_is_a_bounded_stop(self):
        from concurrent.futures import ThreadPoolExecutor
        import threading
        import calibrate_rubric_judge as calibration
        release=threading.Event()
        executor=ThreadPoolExecutor(max_workers=1)
        future=executor.submit(release.wait)
        try:self.assertFalse(calibration.wait_for_futures([future],.01))
        finally:release.set();executor.shutdown(wait=True)

    def test_reused_pid_is_never_stopped(self):
        import calibrate_rubric_judge as calibration
        with patch('calibrate_rubric_judge.Path.exists',return_value=True),patch('rubric_judge.process_birth',return_value='new'),patch('calibrate_rubric_judge.subprocess.run') as stop,patch('calibrate_rubric_judge.os.killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'birth changed'):
                calibration.stop_owned_service(['service'],99999999,'old',100.)
        stop.assert_not_called();kill.assert_not_called()


class LossTests(unittest.TestCase):
    def test_dp_accumulation_against_independent_unique_response_reference(self):
        torch.manual_seed(4)
        features={r:torch.randn(8,5) for r in ('teacher','student')}
        pairs=[pair(i,teacher='T'+str(i//2),student='S'+str(i//2)) for i in range(8)]
        # Repeated responses must have the same features as well as labels.
        for role in features:features[role][1::2]=features[role][::2]
        targets={r:torch.tensor([[1.,.5,0.]]*8) for r in features}
        masks={r:torch.tensor([[1.,1.,0.]]*4+[[1.,0.,1.]]*4) for r in features}
        initial=torch.randn(5,3)
        for mode in LOSS_MODES:
          for dp in (1,2,4):
           for micro in (1,2):
            for mask_case in ('normal','zero','single'):
             m=deepcopy(masks)
             if mask_case=='zero':m={r:torch.zeros_like(v) for r,v in m.items()}
             if mask_case=='single':
                m={r:torch.zeros_like(v) for r,v in m.items()};m['student'][:2,1]=1
             weights,counts=auxiliary_weights(pairs,targets,m,8,dp)
             grads=[];losses=[]
             for rank in range(dp):
              w=initial.clone().requires_grad_();total=0.
              for start in range(rank*8//dp,(rank+1)*8//dp,micro):
               stop=min(start+micro,(rank+1)*8//dp)
               for role in targets:
                loss,_=weighted_contribution(features[role][start:stop]@w,targets[role][start:stop],m[role][start:stop],weights[role][start:stop],counts[start],dp,mode,4)
                loss.backward();total+=loss.detach()
              grads.append(w.grad);losses.append(total)
             unique={p[role]['response_id']:(features[role][i],targets[role][i],m[role][i]) for i,p in enumerate(pairs) for role in targets}
             x,y,mask=map(torch.stack,zip(*unique.values()));w=initial.clone().requires_grad_()
             n=mask.sum(0);active=n>0
             per_dim=(F.binary_cross_entropy_with_logits(x@w,y,reduction='none')*mask).sum(0)/n.clamp_min(1)
             gate=(n/4).clamp(max=1) if mode=='fixed_dimension_gated' else active
             denom=active.sum().clamp_min(1) if mode=='legacy_active' else 3
             ref=(per_dim*gate).sum()/denom;ref.backward()
             torch.testing.assert_close(torch.stack(grads).mean(0),w.grad,atol=1e-6,rtol=1e-6)
             torch.testing.assert_close(torch.stack(losses).mean(),ref.detach())

    def test_correctness_coefficient_identical_in_b_c_and_fixed_is_one_third(self):
        for mode in ('fixed_dimension','fixed_dimension_gated'):
            b=dimension_weights(torch.tensor([5.,0.,0.]),mode,4)[2]
            c=dimension_weights(torch.tensor([5.,2.,1.]),mode,4)[2]
            self.assertEqual(b[0].item(),c[0].item())
        x=torch.tensor([[.2,0.,0.]],requires_grad=True);y=torch.ones_like(x);m=torch.tensor([[1.,0.,0.]])
        torch.testing.assert_close(auxiliary_loss(x,y,m,'fixed_dimension'),auxiliary_loss(x,y,m,'legacy_active')/3)
        torch.testing.assert_close(auxiliary_loss(x,y,m,'fixed_dimension_gated',4),auxiliary_loss(x,y,m,'legacy_active')/12)

    def test_modes_are_validated_and_transferred_to_worker(self):
        from config import compose_config,validate_config
        from extension_config import hydra_overrides
        for mode in LOSS_MODES:
            cfg=compose_config(overrides=[str(ROOT/'configs/overrides/rubric_v2_correctness.yaml')],settings=['rubric.loss_mode='+mode])
            self.assertEqual([],validate_config(cfg))
            args=hydra_overrides(cfg,'warmup',Path('/tmp/v24'),0)
            self.assertIn('"'+mode+'"',next(a for a in args if a.startswith('critic.rubric=')))
        cfg.data['rubric.loss_mode']='oops';self.assertTrue(any('loss_mode' in e for e in validate_config(cfg)))


class IsolationTests(unittest.TestCase):
    def test_quarantine_requires_human_confirmation_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'source';src.mkdir();(src/'abc.json').write_text('{"cache_key":"abc"}')
            r={'cache_key':'abc','review_source':'model_assisted','review_status':'failed','reason':'attribution'}
            with self.assertRaises(ValueError):quarantine(src,root/'new',[r])
            r['review_source']='human';result=quarantine(src,root/'new',[r])
            self.assertFalse(result['source_modified']);self.assertTrue((src/'abc.json').exists())
            self.assertFalse((root/'new/abc.json').exists());self.assertTrue((root/'new/quarantine/abc.json').exists())

    def test_budget_monitor_and_fixed_diagnostic_snapshot_restore(self):
        from omegaconf import OmegaConf
        from test_replay import CheckpointTests
        from gad.extensions.checkpoint_state import save,restore,checkpoint_problems
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'fixed';source.mkdir();torch.save({'x':torch.ones(1)},source/'rank_0.pt')
            runtime=CheckpointTests().runtime(root);runtime.world_size=1
            runtime.rubric.judge=SimpleNamespace(request_count=7)
            runtime.rubric.quality_monitor_state={'coverage':.5}
            runtime.trainer=SimpleNamespace(config=OmegaConf.create({'critic':{'rubric':{'gradient_diagnostics_dir':str(source)}}}))
            checkpoint=root/'global_step_3';checkpoint.mkdir();(checkpoint/'data.pt').write_bytes(b'data')
            for role in ('actor','critic'):
                folder=checkpoint/role;folder.mkdir()
                for name in ('config.json','tokenizer_config.json','tokenizer.json'):(folder/name).write_text('{}')
                for prefix in ('model','optim','extra_state'):(folder/f'{prefix}_world_size_1_rank_0.pt').write_bytes(b'fixture')
            save(runtime,checkpoint);self.assertEqual([],checkpoint_problems(checkpoint,required=True))
            other=CheckpointTests().runtime(root/'new');other.world_size=1;other.rubric.judge=SimpleNamespace(request_count=0)
            destination=root/'restored_fixed'
            other.trainer=SimpleNamespace(config=OmegaConf.create({'critic':{'rubric':{'gradient_diagnostics_dir':str(destination)}}}))
            restore(other,checkpoint)
            self.assertEqual(7,other.rubric.judge.request_count);self.assertEqual({'coverage':.5},other.rubric.quality_monitor_state)
            self.assertEqual((source/'rank_0.pt').read_bytes(),(destination/'rank_0.pt').read_bytes())
            (destination/'rank_0.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'fixed gradient diagnostic'):restore(other,checkpoint)

    def test_audit_never_counts_model_agreement_as_human_reliability(self):
        from summarize_rubric_v24 import summary
        r={'id':'x','dimension':'verified_progress','final_score':1,'raw_score':1,'review_source':'targeted_model','review_status':'completed','score_valid':True,'evidence_valid':True}
        group=summary([r])['groups'][0]
        self.assertEqual(0,group['human_reliable_retained_dimensions']);self.assertEqual(0,group['reviewed_kept_score_correct']['denominator'])
        r.update(review_source='human',reviewer='reviewer',evidence_valid=False)
        group=summary([r])['groups'][0]
        self.assertEqual(1,group['accidentally_right_score_unsupported_evidence']);self.assertEqual(0,group['human_reliable_retained_dimensions'])

    def test_diagnostics_preserve_checkpointing_mode_rng_and_parameters(self):
        from omegaconf import OmegaConf
        from gad.extensions.gradient_diagnostics import diagnose
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.Dropout(.5));self.score=torch.nn.Linear(3,4)
            def forward(self,x):return self.score(self.backbone(x))
        with tempfile.TemporaryDirectory() as d:
            net=Net().train();before={k:v.clone() for k,v in net.state_dict().items()}
            cfg=OmegaConf.create({'ppo_micro_batch_size_per_gpu':1,'rubric':{'gradient_diagnostics_dir':d,'gradient_diagnostics_max_pairs':2,'loss_coef':.1,'loss_mode':'fixed_dimension_gated','count_gate_tau':4}})
            critic=SimpleNamespace(config=cfg,critic_module=net,critic_optimizer=torch.optim.SGD(net.parameters(),lr=.1))
            data={'responses':torch.ones(2,1,dtype=torch.long),'input_ids':torch.randn(2,1,3),'teacher_input_ids':torch.randn(2,1,3),'attention_mask':torch.ones(2,1),'teacher_attention_mask':torch.ones(2,1)}
            for role in ('teacher','student'):
                data[role+'_rubric_targets']=torch.ones(2,3);data[role+'_rubric_mask']=torch.ones(2,3)
            def forward(c,chunk,teacher):
                self.assertTrue(c.critic_module.training,'HF checkpointing requires training mode')
                return c.critic_module(chunk['teacher_input_ids' if teacher else 'input_ids']),chunk['teacher_attention_mask' if teacher else 'attention_mask']
            rng=torch.get_rng_state().clone()
            one=diagnose(critic,data,forward,'cpu');two=diagnose(critic,data,forward,'cpu')
            self.assertEqual(one,two);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for k,v in net.state_dict().items():torch.testing.assert_close(v,before[k])
            self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_shared_backbone_geometry_excludes_score_head(self):
        from gad.extensions.gradient_diagnostics import shared_gradients,geometry
        net=torch.nn.Module();net.backbone=torch.nn.Linear(2,2,bias=False);net.score=torch.nn.Linear(2,4,bias=False)
        net.backbone.weight.grad=torch.ones_like(net.backbone.weight);net.score.weight.grad=torch.full_like(net.score.weight,10000)
        g=shared_gradients(net);self.assertEqual(4,sum(x.numel() for x in g))
        metrics=geometry(g,[2*x for x in g],'cpu')
        self.assertEqual(2.,metrics['aux_bt_norm_ratio']);self.assertEqual(1.,metrics['cosine'])


class WorkerTests(unittest.TestCase):
    def test_real_update_function_matches_full_batch_and_diagnostics_do_not_update(self):
        from omegaconf import OmegaConf
        from tensordict import TensorDict
        from gad.extensions.worker import update_joint
        import gad.extensions.worker as worker
        torch.manual_seed(8)
        xs=torch.randn(4,1,5);xt=torch.randn(4,1,5)
        pairs=[pair(i,teacher='T'+str(i),student='S'+str(i)) for i in range(4)]
        targets={r:torch.tensor([[1.,.5,0.]]*4) for r in ('teacher','student')}
        masks={r:torch.tensor([[1.,1.,0.]]*4) for r in targets}
        weights,counts=auxiliary_weights(pairs,targets,masks,4,1)
        tensors={'input_ids':xs,'teacher_input_ids':xt,'responses':torch.ones(4,1,dtype=torch.long),'teacher_response':torch.ones(4,1,dtype=torch.long),'attention_mask':torch.ones(4,1),'teacher_attention_mask':torch.ones(4,1),'position_ids':torch.zeros(4,1),'teacher_position_ids':torch.zeros(4,1),'rubric_global_counts':counts}
        for role in targets:
            tensors[role+'_rubric_targets']=targets[role];tensors[role+'_rubric_mask']=masks[role];tensors[role+'_aux_weight']=weights[role]
        batch=TensorDict(tensors,batch_size=[4]);proto=SimpleNamespace(select=lambda **kw:SimpleNamespace(batch=batch))
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=torch.nn.Linear(5,3);self.score=torch.nn.Linear(3,4)
            def forward(self,x):return self.score(self.backbone(x).tanh())
        initial=Net().state_dict()
        def forward(c,d,teacher):
            key='teacher_input_ids' if teacher else 'input_ids'
            return c.critic_module(d[key]),d['teacher_attention_mask' if teacher else 'attention_mask']
        for mode in LOSS_MODES:
            ref=Net();ref.load_state_dict(initial)
            s,t=ref(xs).squeeze(1),ref(xt).squeeze(1)
            loss=-F.logsigmoid(t[:,0]-s[:,0]).mean()
            for role,logits in (('student',s),('teacher',t)):
                loss=loss+.1*weighted_contribution(logits[:,1:],targets[role],masks[role],weights[role],counts[0],mode=mode)[0]
            loss.backward()
            expected={n:p.detach()-.02*p.grad for n,p in ref.named_parameters()}
            for micro,interval in ((1,0),(2,1)):
                net=Net();net.load_state_dict(initial);opt=torch.optim.SGD(net.parameters(),lr=.02)
                with tempfile.TemporaryDirectory() as d:
                    config=OmegaConf.create({'use_dynamic_bsz':False,'ppo_epochs':1,'ppo_mini_batch_size':4,'ppo_micro_batch_size_per_gpu':micro,'rubric':{'loss_coef':.1,'loss_mode':mode,'count_gate_tau':4,'head_layout':['adv','answer_correctness','verified_progress','unresolved_error_severity'],'gradient_diagnostics_interval':interval,'gradient_diagnostics_max_pairs':2,'gradient_diagnostics_dir':d}})
                    critic=SimpleNamespace(config=config,critic_module=net,critic_optimizer=opt,ulysses_sequence_parallel_size=1)
                    def step():
                        norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1e9);opt.step();return norm
                    critic._optimizer_step=step
                    with patch.dict(sys.modules,{'verl.utils.device':SimpleNamespace(get_device_id=lambda:torch.device('cpu'))}),patch.object(worker,'forward_channels',forward):
                        metrics=update_joint(critic,proto)
                    for n,p in net.named_parameters():torch.testing.assert_close(p,expected[n],atol=1e-6,rtol=1e-6)
                    self.assertAlmostEqual(loss.item(),metrics['critic/loss_total'][0],places=6)
                    if interval:
                        self.assertIn('rubric/diagnostic/aux/cosine',metrics)
                        self.assertTrue((Path(d)/'rank_0.pt').exists())


if __name__=='__main__':unittest.main()
