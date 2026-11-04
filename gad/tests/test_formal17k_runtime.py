"""Local restore, completion and supervision boundaries; no model service."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'gad/tests'),str(ROOT/'scoring/tests')]
import torch
from omegaconf import OmegaConf
import pipeline
from config import load_config,compose_config
from test_pipeline_cpu import base_config,write_yaml,make_model_dir
import test_pipeline_cpu as fixture
from gad.extensions.checkpoint_state import save,checkpoint_problems
from gad.extensions.annotation_journal import AnnotationJournal
from gad.extensions.annotation_scheduler import annotate_chains
from test_annotation_scheduler import FakeClient,run as run_chains

from cpu_fixtures import compose_cpu_config as compose_config


class CompletionTests(unittest.TestCase):
    def test_extension_state_required_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture.StageArtifactTests()._complete_checkpoint(root,16);checkpoint=root/'global_step_16'
            self.assertEqual([],pipeline.verify_checkpoint_dir(checkpoint,roles=('actor','critic')))
            self.assertTrue(pipeline.verify_checkpoint_dir(checkpoint,roles=('actor','critic'),extensions_required=True))
            runtime=SimpleNamespace(identity='fixture',world_size=1,stage='gad',stage_start=10,stage_step=6,committed_step=16,replay=None,rubric=None)
            save(runtime,checkpoint)
            self.assertEqual([],pipeline.verify_checkpoint_dir(checkpoint,roles=('actor','critic'),extensions_required=True))
            (checkpoint/'training_extensions/sampling_rng.pt').write_bytes(b'corrupt')
            self.assertTrue(checkpoint_problems(checkpoint,required=True))
            (checkpoint/'checkpoint_complete.json').unlink()
            self.assertTrue(pipeline.verify_checkpoint_dir(checkpoint,roles=('actor','critic'),extensions_required=True))

    def test_gad_marker_binds_handoff_and_current_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config=root/'config.yaml';write_yaml(config,base_config(root));cfg=load_config(config)
            run=root/'run';run.mkdir();fixture.StageArtifactTests()._complete_checkpoint(run/'gad',16)
            for role in ('actor','critic'):make_model_dir(run/'init'/role)
            manifest={'step':10,'source_checkpoint':'global_step_10',**{role+'_hf':str(run/'init'/role) for role in ('actor','critic')}}
            path=run/'handoff_manifest.json';path.write_text(json.dumps(manifest))
            plan=pipeline.plan_stage(cfg,run,'gad',steps_per_epoch_value=4,handoff=manifest,install_guard=False)
            verification=pipeline.verify_stage_artifacts('gad',run/'gad',expected_step=16)
            cfg.data['validation.require_math_metrics']=False
            pipeline.write_completion(cfg,run,'gad',plan,verification,started='test',finished='test',returncode=0)
            marker=(run/'gad/COMPLETED.json').read_bytes()
            self.assertTrue(pipeline.check_skip(cfg,run,'gad')[0])
            for change in ('weights','step','legacy'):
                if change=='weights':(run/'init/actor/model.safetensors').write_bytes(b'new')
                elif change=='step':manifest['step']=9;path.write_text(json.dumps(manifest))
                else:
                    row=json.loads(marker);row['inputs'].pop('warmup_handoff');(run/'gad/COMPLETED.json').write_text(json.dumps(row))
                with self.assertRaisesRegex(pipeline.PipelineError,'handoff'):pipeline.check_skip(cfg,run,'gad')
                (run/'init/actor/model.safetensors').write_bytes(b'\0');manifest['step']=10;path.write_text(json.dumps(manifest))
                (run/'gad/COMPLETED.json').write_bytes(marker)
            cfg.data['budget.gad_steps']=7
            self.assertFalse(pipeline.check_skip(cfg,run,'gad')[0])
            self.assertEqual(marker,(run/'gad/COMPLETED.json').read_bytes())


class WorkerBoundaryTests(unittest.TestCase):
    def test_actual_load_method_restores_once_per_branch(self):
        for stage in ('seqkd','warmup','gad'):
            backend=Path(os.environ['GAD_TEST_BACKEND_ROOT'])/stage
            tree=ast.parse((backend/'verl/trainer/ppo/ray_trainer.py').read_text())
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RayPPOTrainer')
            method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_load_checkpoint')
            for enabled in (False,True):
                with tempfile.TemporaryDirectory() as d:
                    checkpoint=Path(d)/'global_step_10';checkpoint.mkdir()
                    for role in ('actor','critic'):(checkpoint/role).mkdir()
                    torch.save({'position':17},checkpoint/'data.pt')
                    calls=[];loaded=[]
                    worker=lambda role:SimpleNamespace(load_checkpoint=lambda *a,**k:calls.append(role))
                    trainer=SimpleNamespace(global_steps=0,use_critic=stage!='seqkd',actor_rollout_wg=worker('actor'),critic_wg=worker('critic'),
                        train_dataloader=SimpleNamespace(load_state_dict=loaded.append),config=OmegaConf.create({'trainer':{
                            'resume_mode':'auto','default_hdfs_dir':None,'default_local_dir':d,'resume_load_worker_state':enabled,'del_local_ckpt_after_load':False}}))
                    namespace={'os':os,'torch':torch,'find_latest_ckpt_path':lambda _:str(checkpoint)}
                    exec(compile(ast.Module(body=[method],type_ignores=[]),'<real_load>','exec'),namespace)
                    namespace['_load_checkpoint'](trainer)
                    expected=['actor']+(['critic'] if stage!='seqkd' else [])
                    self.assertEqual(expected if stage!='gad' or enabled else [],calls)
                    self.assertEqual([{'position':17}],loaded)

    def test_reference_fsdp_objects_and_role_overrides(self):
        from torch.distributed.fsdp import CPUOffload
        backend=Path(os.environ['GAD_TEST_BACKEND_ROOT'])/'gad'
        source=(backend/'verl/workers/fsdp_workers.py').read_text();tree=ast.parse(source)
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cpu_offload' for t in n.targets)
                     and isinstance(n.value,ast.IfExp) and 'param_offload' in ast.unparse(n)]
        self.assertEqual(2,len(assignments))
        for enabled in (False,True):
            for role in ('actor','ref'):
                ns={'role':role,'fsdp_config':{'param_offload':enabled},'CPUOffload':CPUOffload,'CPUOffloadPolicy':lambda **kw:('fsdp2',kw)}
                for node in assignments:
                    exec(compile(ast.Module(body=[node],type_ignores=[]),'<real_fsdp_policy>','exec'),ns)
                    self.assertEqual(enabled and role=='ref',ns['cpu_offload'] is not None)
        cfg=compose_config(settings=['models.param_offload=true','models.actor_param_offload=false','models.critic_param_offload=true'])
        for stage in ('seqkd','warmup','gad'):
            overrides=pipeline.build_overrides(cfg,stage,model_path=cfg.require('models.student'),critic_path=cfg.require('models.discriminator_init'),out_dir=Path('/tmp/fixture'),total_steps=10,epochs=1)
            self.assertIn('actor_rollout_ref.actor.fsdp_config.param_offload=False',overrides)
            if stage!='seqkd':self.assertIn('critic.model.fsdp_config.param_offload=True',overrides)

    def test_visibility_deduplication_preserves_equal_supervision_across_roles(self):
        from test_online_rubric import pair
        from gad.extensions.critic_batch import auxiliary_weights
        pairs=[pair(0,teacher='same',student='same')]
        targets={role:torch.ones(1,3) for role in ('teacher','student')};masks=deepcopy(targets)
        weights,counts=auxiliary_weights(pairs,targets,masks,1,1)
        self.assertEqual(.5,weights['student'][0]);self.assertEqual([[1,1,1]],counts.tolist())
        pairs[0]['teacher']['source_truncated']=True;targets['teacher'][0,1]=.5
        weights,counts=auxiliary_weights(pairs,targets,masks,1,1)
        self.assertEqual(1,weights['student'][0]);self.assertEqual([[2,2,2]],counts.tolist())
        pairs[0]['teacher']['source_truncated']=False
        with self.assertRaisesRegex(ValueError,'inconsistent'):auxiliary_weights(pairs,targets,masks,1,1)


class JournalTests(unittest.TestCase):
    def test_confirmed_failure_retries_once_across_processes_and_success_reuses(self):
        for failing_stage in ('primary','review'):
            with tempfile.TemporaryDirectory() as d:
                contexts={'0':('0','2x=4')};first=FakeClient(fail={(failing_stage,'0')})
                journal=AnnotationJournal(Path(d),{'version':'fixture'})
                run_chains(first,1,on_state=journal.save)
                recovered=AnnotationJournal(Path(d),{'version':'fixture'}).recover(contexts)
                second=FakeClient();result=run_chains(second,1,recovered=recovered,on_state=journal.save)
                self.assertTrue(result['0']['parsed']['verified_progress']['valid'])
                self.assertEqual(2,result['0']['annotation_attempt'])
                self.assertEqual(2 if failing_stage=='primary' else 1,second.request_count)
                recovered=journal.recover(contexts);third=FakeClient();run_chains(third,1,recovered=recovered)
                self.assertEqual(0,third.request_count)

    def test_attempt_budget_is_per_training_step_across_restarts(self):
        with tempfile.TemporaryDirectory() as d:
            contexts={'0':('0','2x=4')}
            journal=AnnotationJournal(Path(d),{'version':'fixture'},step=7)
            run_chains(FakeClient(fail={('primary','0')}),1,on_state=journal.save)
            recovered=AnnotationJournal(Path(d),{'version':'fixture'},step=7).recover(contexts)
            run_chains(FakeClient(fail={('primary','0')}),1,recovered=recovered,on_state=journal.save)
            with self.assertRaisesRegex(RuntimeError,'attempt limit'):
                AnnotationJournal(Path(d),{'version':'fixture'},step=7).recover(contexts)
            self.assertEqual({},AnnotationJournal(Path(d),{'version':'fixture'},step=8).recover(contexts))

    def test_attempt_bound_and_unknown_submission_are_persistent(self):
        with tempfile.TemporaryDirectory() as d:
            journal=AnnotationJournal(Path(d),{'v':1});contexts={'0':('0','2x=4')}
            run_chains(FakeClient(fail={('primary','0')}),1,on_state=journal.save)
            recovered=journal.recover(contexts)
            run_chains(FakeClient(fail={('primary','0')}),1,recovered=recovered,on_state=journal.save)
            with self.assertRaisesRegex(RuntimeError,'attempt limit'):journal.recover(contexts)
        with tempfile.TemporaryDirectory() as d:
            journal=AnnotationJournal(Path(d),{'v':1});journal.save('0','primary_submitted',{'request_id':'0','attempts':0,'usage':{}})
            recovered=journal.recover({'0':('0','2x=4')});c=FakeClient();run_chains(c,1,recovered=recovered)
            self.assertEqual(0,c.request_count);self.assertEqual(1,journal.recovered_request_count)
            explicit=AnnotationJournal(Path(d),{'v':1},unknown_policy='retry').recover({'0':('0','2x=4')})
            self.assertTrue(explicit['0']['possible_duplicate_remote_cost'])


if __name__=='__main__':unittest.main()
