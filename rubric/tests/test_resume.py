from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForTokenClassification
from rubric.model import expand_head,load_classifier
from rubric.train_offline import save_checkpoint,restore_training_state
from rubric.train_offline import prune_checkpoints
from rubric.common import write_json


class Resume(unittest.TestCase):
    def test_retention_keeps_recent_five_and_best(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for step in range(1,8):
                path=root/f"step_{step:06d}"
                write_json(path/"offline_manifest.json",{"identity":"fixture"})
            best=root/"step_000001"
            prune_checkpoints(root,{"identity":"fixture","best_checkpoint":str(best)})
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             ["step_000001"]+[f"step_{i:06d}" for i in range(3,8)])

    def test_optimizer_scheduler_rng_and_order_resume_matches_uninterrupted(self):
        class Tokenizer:
            def save_pretrained(self,path):
                (Path(path)/"fixture_tokenizer.txt").write_text("fixture")
        model=Qwen3ForTokenClassification(Qwen3Config(
            vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,num_labels=1))
        expand_head(model,42,"fixture")
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.9**step)
        def update(model,optimizer,scheduler):
            model.train()
            optimizer.zero_grad()
            inputs=torch.randint(0,32,(1,4))
            scale=random.random()+np.random.random()+float(torch.rand(1))
            model(inputs).logits.square().mean().mul(scale).backward()
            optimizer.step()
            scheduler.step()
        update(model,optimizer,scheduler)
        with TemporaryDirectory() as directory:
            path=Path(directory)/"step_1"
            with patch("torch.cuda.get_rng_state_all",return_value=[]):
                save_checkpoint(path,model,Tokenizer(),optimizer,scheduler,
                                {"epoch":0,"step":1,"offset":1,"order":[2,0,1],"identity":"fixture"})
            update(model,optimizer,scheduler)
            expected={key:value.clone() for key,value in model.state_dict().items()}
            restored=torch.load(path/"training_state.pt",map_location="cpu",weights_only=False)
            resumed=load_classifier(path,rubric_hash="fixture")
            resumed_optimizer=torch.optim.AdamW(resumed.parameters(),lr=1e-4)
            resumed_scheduler=torch.optim.lr_scheduler.LambdaLR(resumed_optimizer,lambda step:.9**step)
            restore_training_state(resumed_optimizer,resumed_scheduler,restored)
            self.assertEqual(restored["order"],[2,0,1])
            self.assertEqual(restored["offset"],1)
            self.assertTrue(resumed_optimizer.state)
            update(resumed,resumed_optimizer,resumed_scheduler)
            for key,value in resumed.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            self.assertEqual(resumed_scheduler.state_dict(),scheduler.state_dict())


if __name__=="__main__":
    unittest.main()
