"""Actual SeqKD config/SFT and CPU pipeline boundary regression checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools')]
import pipeline
from config import compose_config
from backend_patches import verify_frozen_backend
BACKENDS=Path(os.environ.get('GAD_TEST_EXTENSION_ROOT','/tmp/gad-seqkd-baseline-audit-20261017'))

CHILD=r'''
import json,sys,tempfile
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2],str(Path(sys.argv[2])/'gad/tools')]
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from config import compose_config
from pipeline import plan_stage,steps_per_epoch
root=Path(sys.argv[2]);backend=Path(sys.argv[1]);stage=backend.name
cfg=compose_config(model='qwen3_0.6b',dataset='dapo_math_17k',overrides=[str(root/'configs/overrides/rubric_v2_baseline.yaml')])
cfg.data['backends.root']=str(backend.parent)
plan=plan_stage(cfg,Path('/tmp/seqkd-config-fixture'),stage,steps_per_epoch_value=steps_per_epoch(cfg),
    handoff={'actor_hf':'/tmp/actor','critic_hf':'/tmp/critic','step':271} if stage=='gad' else None,install_guard=False)
with initialize_config_dir(config_dir=str(backend/'verl/trainer/config'),version_base=None):
    actual=compose(config_name='ppo_trainer',overrides=plan['argv'][3:])
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
trainer=RayPPOTrainer.__new__(RayPPOTrainer)
trainer.config=actual;trainer.use_critic=stage!='seqkd';trainer.use_reference_policy=stage!='seqkd'
RayPPOTrainer._validate_config(trainer)
result={'stage':stage,'config_composed':True,'trainer_config_validated':True,'steps':actual.trainer.total_training_steps,
    'rollout_n':actual.actor_rollout_ref.rollout.n,'critic_warmup':actual.trainer.critic_warmup,
    'rouge':actual.trainer.validation_rouge_l_enabled,'judge':actual.trainer.get('rubric',{}).get('enabled',False)}
if stage=='seqkd':
    import torch
    import torch.distributed as dist
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from verl import DataProto
    import verl.workers.actor.dp_actor as worker
    from flash_attn.bert_padding import index_first_axis,pad_input,unpad_input,rearrange
    # These padding helpers are ordinary Torch operations; the CPU-only import
    # does not otherwise install CUDA branch globals in the production module.
    worker.index_first_axis,worker.pad_input,worker.unpad_input,worker.rearrange=index_first_axis,pad_input,unpad_input,rearrange
    from unittest.mock import patch
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group('gloo',init_method='file://'+str(Path(tmp)/'gloo'),rank=0,world_size=1)
        try:
            torch.manual_seed(42)
            model=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=16,pad_token_id=0,attn_implementation='eager'))
            config=OmegaConf.create(OmegaConf.to_container(actual.actor_rollout_ref.actor,resolve=True))
            config.update(use_remove_padding=bool(actual.actor_rollout_ref.model.use_remove_padding),use_torch_compile=False,use_dynamic_bsz=False,ppo_epochs=1,ppo_mini_batch_size=1,ppo_micro_batch_size_per_gpu=1)
            optim=torch.optim.AdamW(model.parameters(),lr=5e-6)
            actor=worker.DataParallelPPOActor(config,model,optim);actor.device_name='cpu'
            batch=DataProto.from_dict(tensors={'teacher_input_ids':torch.tensor([[1,2,3,4,5]]),
                'teacher_response':torch.tensor([[3,4,5]]),'teacher_attention_mask':torch.ones(1,5,dtype=torch.long),
                'teacher_position_ids':torch.arange(5)[None,:]},non_tensors={'uid':__import__('numpy').array(['same_question'],dtype=object)},meta_info={'temperature':.8})
            before=model.model.layers[0].self_attn.q_proj.weight.detach().clone()
            with patch.object(worker,'get_device_id',return_value=torch.device('cpu')), patch('verl.utils.debug.performance._get_current_mem_info',return_value=(0,0,0,0)), patch('verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE',False):
                metrics=actor.update_policy(batch)
            delta=(model.model.layers[0].self_attn.q_proj.weight.detach()-before).abs().max().item()
            assert delta>0 and all(torch.isfinite(p).all() for p in model.parameters())
            result.update(actual_qwen_sft_cpu_update=True,parameter_max_delta=delta,metrics=metrics,
                scope='tiny Qwen3 real production SeqKD SFT worker on CPU; no full 0.6B/GPU rollout')
        finally:dist.destroy_process_group()
print('@@RESULT@@'+json.dumps(result))
'''


def checkpoint(out,step,stage):
    import torch
    directory=out/f'global_step_{step}'
    for role in ('actor',) if stage=='seqkd' else ('actor','critic'):
        path=directory/role;path.mkdir(parents=True)
        (path/'config.json').write_text(json.dumps({'architectures':['Qwen3ForCausalLM' if role=='actor' else 'Qwen3ForTokenClassification']}))
        for name in ('model','optim','extra_state'):
            torch.save({'fixture':torch.tensor([1]),'step':step},path/f'{name}_world_size_1_rank_0.pt')
    torch.save({'step':step},directory/'data.pt')
    (out/'latest_checkpointed_iteration.txt').write_text(str(step))


class SeqKDBaselineTests(unittest.TestCase):
    def test_actual_each_stage_config_and_seqkd_sft_worker(self):
        for stage in ('seqkd','warmup','gad'):
            with self.subTest(stage=stage):
                result=subprocess.run([sys.executable,'-c',CHILD,str(BACKENDS/stage),str(ROOT)],
                    cwd=ROOT,capture_output=True,text=True,timeout=90,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
                self.assertEqual(0,result.returncode,result.stdout+result.stderr)
                record=json.loads(next(l[len('@@RESULT@@'):] for l in result.stdout.splitlines() if l.startswith('@@RESULT@@')))
                self.assertFalse(record['judge']);self.assertFalse(record['rouge'])
                self.assertEqual({'seqkd':542,'warmup':271,'gad':542}[stage],record['steps'])
                if stage=='seqkd':
                    self.assertTrue(record['actual_qwen_sft_cpu_update']);self.assertGreater(record['parameter_max_delta'],0)
                    # Historical reproduction uses this gate in the pinned script.
                    self.assertEqual(10,record['critic_warmup'])
                    self.assertNotIn('critic/',str(record['metrics']))
                print('actual backend check:',json.dumps(record))

    def exercise_pipeline(self,mode='incomplete',resume=False):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);run=output/'audit';calls=[]
            argv=['pipeline.py','--model','qwen3_0.6b','--dataset','dapo_math_17k','--run-id','audit',
                '--stages','seqkd,warmup','--overrides',str(ROOT/'configs/overrides/rubric_v2_baseline.yaml'),
                '--set',f'backends.root={BACKENDS}','--set',f'run.output_root={output}']
            if mode=='strict':argv+=['--set','validation.block_training_on_incomplete=true']
            def command(command,**kwargs):
                stage=kwargs.get('component')
                if stage not in ('seqkd','warmup'):return SimpleNamespace(returncode=0)
                calls.append(stage)
                values=dict(a.split('=',1) for a in command[3:] if '=' in a)
                out=Path(values['trainer.default_local_dir']);step=int(values['trainer.total_training_steps'])
                if mode=='short' and stage=='seqkd':step-=1
                checkpoint(out,step,stage)
                if mode=='missing_optimizer' and stage=='seqkd':next((out/f'global_step_{step}'/'actor').glob('optim*.pt')).unlink()
                target=out/'validation_math_metrics.jsonl'
                payload={'global_step':step,'validation_scope':'full','count':512,'evaluation_complete':False,
                    'acc':None,'provisional_acc':52/512,'unscorable_rate':2/512,'grader_error_rate':0}
                if stage=='warmup':payload.update(evaluation_complete=True,acc=.1,unscorable_rate=0)
                if not(mode=='missing_metrics' and stage=='seqkd'):
                    target.write_text('{bad json' if mode=='bad_metrics' and stage=='seqkd' else json.dumps(payload)+'\n')
                return SimpleNamespace(returncode=1 if mode=='nonzero' and stage=='seqkd' else 0)
            context=[patch.object(sys,'argv',argv),patch.object(pipeline,'run_command',side_effect=command),
                patch.object(pipeline,'verify_backend_import'),patch.object(pipeline,'dependency_versions',return_value={}),
                patch.object(pipeline,'data_provenance',return_value={})]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for c in context:stack.enter_context(c)
                if mode in ('nonzero','short','missing_optimizer','strict'):
                    with self.assertRaises(pipeline.PipelineError):pipeline.main()
                    self.assertEqual(['seqkd'],calls)
                    self.assertFalse((run/'warmup/COMPLETED.json').exists())
                else:
                    self.assertEqual(0,pipeline.main());self.assertEqual(['seqkd','warmup'],calls)
                    record=json.loads((run/'seqkd/COMPLETED.json').read_text())
                    self.assertTrue(record['training_completed']);self.assertEqual('incomplete' if mode=='incomplete' else 'missing' if mode=='missing_metrics' else 'failed',record['evaluation_status'])
                    self.assertFalse((run/'seqkd/global_step_542/critic').exists())
                    warmup=json.loads((run/'stage_warmup.json').read_text())['plan']
                    self.assertEqual(str(ROOT.parent/'model/Qwen/Qwen3-0.6B'),warmup['model_path'])
                    self.assertEqual(0,warmup['start_step'])
                    if resume:
                        calls.clear();sys.argv=['pipeline.py','--run',str(run),'--resume','--stages','seqkd,warmup']
                        self.assertEqual(0,pipeline.main());self.assertEqual([],calls)

    def test_incomplete_seqkd_allows_warmup_and_resume_skips_both(self):self.exercise_pipeline(resume=True)
    def test_missing_metrics_allows_warmup(self):self.exercise_pipeline('missing_metrics')
    def test_bad_metrics_allows_warmup(self):self.exercise_pipeline('bad_metrics')
    def test_nonzero_seqkd_blocks_warmup(self):self.exercise_pipeline('nonzero')
    def test_short_seqkd_blocks_warmup(self):self.exercise_pipeline('short')
    def test_missing_optimizer_blocks_warmup(self):self.exercise_pipeline('missing_optimizer')
    def test_explicit_strict_evaluation_blocks_warmup(self):self.exercise_pipeline('strict')

    def test_frozen_seqkd_self_contained_manifest_and_tamper_rejection(self):
        backend=BACKENDS/'seqkd'
        manifest=backend.parent/'backend_manifest.json';manifest_bytes=manifest.read_bytes()
        used=backend/'.gad_backend_used';existed=used.exists()
        try:
            payload=json.loads(manifest_bytes);payload['backends'].pop('seqkd',None)
            manifest.write_text(json.dumps(payload));missing_entry=manifest.read_bytes();used.touch()
            record=verify_frozen_backend(backend)
            result=subprocess.run([sys.executable,str(ROOT/'gad/tools/prepare_backends.py'),'--repo',str(ROOT/'gad/verl'),
                '--backend-root',str(backend.parent),'--profile','gad_extensions_v1','--stages','seqkd'],capture_output=True,text=True)
            self.assertEqual(0,result.returncode,result.stdout+result.stderr)
            self.assertEqual(missing_entry,manifest.read_bytes())
        finally:
            manifest.write_bytes(manifest_bytes)
            if not existed:used.unlink()
        self.assertEqual('gad.engineering_backend.v1',record['schema_version'])
        target=backend/'verl/trainer/config/ppo_trainer.yaml';original=target.read_bytes()
        try:
            target.write_bytes(original+b'\n# tamper fixture\n')
            with self.assertRaises(RuntimeError):verify_frozen_backend(backend)
        finally:target.write_bytes(original)

if __name__=='__main__':unittest.main(verbosity=2)
