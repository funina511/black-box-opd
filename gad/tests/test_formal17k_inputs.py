"""Data, startup identity and response boundaries; all assets are temporary."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'gad/tools'),str(ROOT/'sample'),str(ROOT/'gad/tests')]
from config import compose_config
from extension_config import hydra_overrides
from rubric.core.contracts import file_hash

from cpu_fixtures import compose_cpu_config as compose_config


class InputTests(unittest.TestCase):
    def test_method_overrides_compose_with_actual_backend_schema(self):
        from hydra import compose, initialize_config_dir
        from pipeline import build_overrides
        for method in ('baseline', 'correctness_only', 'rubric_joint'):
            cfg = compose_config(model='qwen3_0.6b', dataset='dapo_math_17k_v2',
                overrides=[str(ROOT/f'configs/methods/{method}.yaml')])
            cfg.data['rubric.judge.tokenizer_path'] = str(ROOT/'gad/tests/fixtures/identity_model')
            cfg.data['backends.profile'] = 'gad_extensions_v1'
            cfg.data['backends.root'] = os.environ['GAD_TEST_EXTENSION_ROOT']
            for stage in ('warmup', 'gad'):
                backend = Path(cfg.require('backends.root'))/stage
                cfg.data['models.student']=str(ROOT/'gad/tests/fixtures/identity_model')
                overrides = build_overrides(cfg, stage, model_path=cfg.require('models.student'),
                    critic_path=cfg.require('models.discriminator_init'), out_dir=Path('/tmp/schema-probe'),
                    total_steps=542, epochs=2) + hydra_overrides(cfg, stage, Path('/tmp/schema-probe'), 0)
                with initialize_config_dir(config_dir=str(backend/'verl/trainer/config'), version_base=None):
                    actual = compose(config_name='ppo_trainer', overrides=overrides)
                self.assertEqual(method != 'baseline', actual.trainer.rubric.enabled)
                if method != 'baseline':
                    self.assertEqual(file_hash(cfg.require('rubric.prompt')), actual.trainer.rubric.prompt_hash)
                    self.assertEqual('rubric.events.v2.7', actual.trainer.rubric.protocol_identity.schema_version)

    def test_complete_methods_have_explicit_compatible_settings(self):
        from config import validate_config
        for method in ('baseline','correctness_only','rubric_joint'):
            cfg=compose_config(model='qwen3_0.6b',dataset='dapo_math_17k_v2',overrides=[str(ROOT/f'configs/methods/{method}.yaml'),str(ROOT/'configs/machines/3090_4gpu.yaml')])
            self.assertEqual([],validate_config(cfg))
            self.assertEqual('fixed_dimension_gated',cfg.get('rubric.loss_mode'))
            self.assertEqual(2,cfg.get('rubric.judge.concurrency'))
            self.assertEqual(2,cfg.get('budget.stage_epochs.gad'))

    def test_prompt_spec_and_semantic_policy_are_pinned_before_judge(self):
        from gad.extensions.online_rubric import OnlineRubric
        from hydra.core.override_parser.overrides_parser import OverridesParser
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg=compose_config(settings=['rubric.enabled=true','rubric.mode=correctness_only'])
            for field in ('spec','prompt'):
                p=root/field;p.write_text(Path(cfg.require('rubric.'+field)).read_text());cfg.data['rubric.'+field]=str(p)
            cfg.data['models.student']=str(ROOT/'gad/tests/fixtures/identity_model')
            options=OverridesParser.create().parse_overrides(hydra_overrides(cfg,'warmup',root/'stage',0))[0].value()
            with patch('gad.extensions.online_rubric.JudgeClient',side_effect=AssertionError('unexpected Judge')):
                OnlineRubric(deepcopy(options),root/'ok')
                for field in ('prompt','spec'):
                    p=Path(options[field]);before=p.read_text();p.write_text(before+'\nchanged')
                    with self.assertRaisesRegex(ValueError,field+' changed'):OnlineRubric(deepcopy(options),root/'bad')
                    p.write_text(before)
                changed=deepcopy(options);changed['semantic_check']=True
                with self.assertRaisesRegex(ValueError,'semantic policy'):OnlineRubric(changed,root/'bad')

    def test_four_head_critic_must_match_spec_before_launch(self):
        import pipeline
        from test_pipeline_cpu import make_model_dir
        cfg=compose_config(settings=['rubric.enabled=true','rubric.mode=correctness_only'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);make_model_dir(root/'critic')
            config=root/'critic/config.json';row=json.loads(config.read_text());row.update(num_labels=4,
                rubric_head_layout=cfg.get('rubric.head_layout'),rubric_version='math_process_v2',rubric_hash='old')
            config.write_text(json.dumps(row));cfg.data['models.discriminator_init']=str(config.parent)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                cfg.data['models.student']=str(ROOT/'gad/tests/fixtures/identity_model')
                pipeline.plan_stage(cfg,root/'run','warmup',steps_per_epoch_value=1,install_guard=False)
            row['rubric_hash']=file_hash(cfg.require('rubric.spec'));config.write_text(json.dumps(row))
            cfg.data['models.student']=str(ROOT/'gad/tests/fixtures/identity_model')
            plan=pipeline.plan_stage(cfg,root/'run','warmup',steps_per_epoch_value=1,install_guard=False)
            self.assertEqual(str(config.parent),plan['critic_path'])

    def test_real_token_masks_distinguish_empty_text_from_zero_tokens(self):
        import torch
        from gad.extensions.critic_batch import extract_pairs
        from rubric.core.heads import last_response_logits
        class Batch:
            def __len__(self):return 1
        batch=Batch();batch.batch={'prompts':torch.tensor([[1]]),'responses':torch.tensor([[2]]),'teacher_response':torch.tensor([[2]]),
            'attention_mask':torch.tensor([[1,1]]),'teacher_attention_mask':torch.tensor([[1,1]])}
        batch.non_tensor_batch={'extra_info':[{'index':'sample'}],'reward_model':[{'ground_truth':'2'}]}
        tokenizer=SimpleNamespace(decode=lambda ids,**k:'' if k.get('skip_special_tokens') else 'question')
        rows=extract_pairs(batch,tokenizer,'fixture',11,'gad')
        self.assertEqual('',rows[0]['student']['response']);self.assertEqual([2],rows[0]['student']['token_ids'])
        self.assertIsNotNone(last_response_logits(torch.ones(1,1,4),torch.tensor([[1]])))
        for role,key in [('student','attention_mask'),('teacher','teacher_attention_mask')]:
            batch.batch[key][0,-1]=0
            with self.assertRaisesRegex(ValueError,'stage=gad step=11 role='+role):extract_pairs(batch,tokenizer,'fixture',11,'gad')
            batch.batch[key][0,-1]=1

    def test_unscorable_candidates_do_not_hide_a_later_correct_one(self):
        from sample.build_gad_data import pick_teacher_candidate,UnresolvedCandidates
        from scoring.math import score,correctness_label
        self.assertEqual(('Answer: 2','first_correct',1),pick_teacher_candidate(['Answer: \\boxed{','Answer: 2'],select='correct',ground_truth='2'))
        with self.assertRaises(UnresolvedCandidates):pick_teacher_candidate(['Answer: \\boxed{'],select='correct',ground_truth='2')
        self.assertEqual((None,'no_correct_answer',None),pick_teacher_candidate(['unfinished'],select='correct',ground_truth='2'))
        self.assertEqual(0,correctness_label(score('unfinished','2'))['value'])
        self.assertTrue(correctness_label(score('unfinished','2'))['valid'])

    def test_sampling_identity_exists_before_gpu_and_tracks_same_path_content(self):
        from teacher_engines import VllmOfflineEngine
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):return 'prompt:'+str(kwargs.get('enable_thinking'))
        with tempfile.TemporaryDirectory() as d,patch('transformers.AutoTokenizer.from_pretrained',return_value=Tokenizer()),patch('teacher_engines.importlib.metadata.version',return_value='0.8.5'):
            root=Path(d);(root/'model.safetensors').write_bytes(b'weights');(root/'tokenizer.json').write_text('{}')
            def engine():return VllmOfflineEngine(model_path=d,m=1,temperature=.7,top_p=.95,max_tokens=8,seed=42,chat_template_kwargs={'enable_thinking':False})
            a=engine();first=a.fingerprint_fields();self.assertIsNone(a._llm);self.assertEqual('0.8.5',first['vllm_version'])
            self.assertEqual(first,engine().fingerprint_fields())
            for name in ('model.safetensors','tokenizer.json'):
                p=root/name;p.write_bytes(p.read_bytes()+b'changed')
                self.assertNotEqual(first,engine().fingerprint_fields())


if __name__=='__main__':unittest.main()
