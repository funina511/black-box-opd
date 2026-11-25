"""Export real tiny Qwen NO_SHARD states and reject ambiguous ordinary shards."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, Qwen3Config

ROOT = Path(__file__).resolve().parents[2]
MERGE = ROOT / 'gad/tools/merge_model2hf.py'


class SingleRankMergeTests(unittest.TestCase):
    def run_merge(self, directory, output_dir=None):
        command = [sys.executable, str(MERGE), '--local_dir', str(directory)]
        if output_dir is not None:
            command.extend(['--output-dir', str(output_dir)])
        return subprocess.run(command,
                              env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONFAULTHANDLER': '1'},
                              capture_output=True, text=True, timeout=60)

    def test_actor_and_critic_plain_tensor_exports_reload_exact_bfloat16_weights(self):
        for role, factory in (('actor', AutoModelForCausalLM), ('critic', AutoModelForTokenClassification)):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / (role + '-archive-huggingface')
                directory.mkdir()
                output_dir = Path(temporary) / 'actor-export' if role == 'actor' else None
                hf = output_dir or directory / 'huggingface'
                hf.mkdir()
                config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=2, head_dim=8,
                                     tie_word_embeddings=True, num_labels=1)
                torch.manual_seed(42)
                model = factory.from_config(config)
                with torch.no_grad():
                    model.model.embed_tokens.weight[0, 0] = .75
                    if role == 'critic':
                        model.score.weight.fill_(.125)
                        if model.score.bias is not None:
                            model.score.bias.fill_(-.5)
                model.config.architectures = [type(model).__name__]
                model.config.save_pretrained(hf)
                source = {key: value.detach().clone() for key, value in model.state_dict().items()}
                torch.save(source, directory / 'model_world_size_1_rank_0.pt')
                result = self.run_merge(directory, output_dir)
                self.assertEqual(0, result.returncode, result.stdout + result.stderr)
                self.assertIn('FSDP NO_SHARD', result.stdout)
                loaded = factory.from_pretrained(hf, local_files_only=True, torch_dtype=torch.bfloat16)
                actual = loaded.state_dict()
                self.assertEqual(set(source), set(actual))
                for key in source:
                    self.assertTrue(torch.equal(source[key].bfloat16(), actual[key]), key)
                if role == 'critic':
                    self.assertEqual((1, 16), tuple(actual['score.weight'].shape))
                else:
                    self.assertEqual(loaded.lm_head.weight.data_ptr(), loaded.model.embed_tokens.weight.data_ptr())

    def test_output_alias_to_source_shards_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'actor'
            directory.mkdir()
            shard = directory / 'model_world_size_1_rank_0.pt'
            torch.save({'weight': torch.ones(2)}, shard)
            original = shard.read_bytes()
            alias = Path(temporary) / 'export'
            alias.symlink_to(directory, target_is_directory=True)
            result = self.run_merge(directory, alias)
            self.assertNotEqual(0, result.returncode)
            self.assertIn('output directory must differ', result.stderr)
            self.assertEqual([shard], list(directory.iterdir()))
            self.assertEqual(original, shard.read_bytes())

    def test_hf_directory_without_source_shards_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'huggingface'
            directory.mkdir()
            (directory / 'model.safetensors').touch()
            result = self.run_merge(directory)
            self.assertNotEqual(0, result.returncode)
            self.assertIn('No rank-zero model shard', result.stderr)
            self.assertFalse((directory / 'huggingface').exists())

    def test_plain_multirank_or_extra_rank_is_rejected_before_export(self):
        for filenames in (('model_world_size_2_rank_0.pt',),
                          ('model_world_size_1_rank_0.pt', 'model_world_size_1_rank_1.pt')):
            with self.subTest(filenames=filenames), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for filename in filenames:
                    torch.save({'weight': torch.ones(2)}, directory / filename)
                result = self.run_merge(directory)
                self.assertNotEqual(0, result.returncode)
                self.assertIn('requires exactly one world_size=1 rank_0 model file', result.stderr)
                self.assertFalse((directory / 'huggingface').exists())

    def test_mixed_tensor_and_non_tensor_state_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            torch.save({'a_weight': torch.ones(2), 'z_metadata': 123},
                       directory / 'model_world_size_1_rank_0.pt')
            result = self.run_merge(directory)
            self.assertNotEqual(0, result.returncode)
            self.assertIn('must contain only ordinary Tensors', result.stderr)
            self.assertFalse((directory / 'huggingface').exists())


if __name__ == '__main__':
    unittest.main()
