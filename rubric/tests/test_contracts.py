import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from rubric.common import DIMENSIONS, ROOT, digest, file_hash, load_config, model_fingerprint
from rubric.schema import *
from rubric.losses import *
from rubric.model import expand_head, last_response_logits, encode_response, load_classifier, score_response
from rubric.evaluate import metrics, auroc, paired_deltas
from rubric.train_offline import backward_batch


class Contracts(unittest.TestCase):
    def test_unknown_is_not_partial_and_cache_covers_ground_truth(self):
        row = {"prompt": [{"role":"user","content":"x?"}], "response":"Answer: 2", "ground_truth":"2"}
        row.update(prompt_hash=hash_raw_prompt(row["prompt"]), response_hash=digest(row["response"]))
        key = annotation_key(row, "rubric", "judge", "prompt")
        changed = dict(row, ground_truth="3")
        self.assertNotEqual(key, annotation_key(changed,"rubric","judge","prompt"))
        with self.assertRaises(ValueError):
            validate_label({"valid":False,"value":.5}, "reasoning_validity")
        self.assertEqual(parse_judge(json.dumps({d:{"value":None,"valid":False,"evidence":"unreadable"}
                                                for d in DIMENSIONS[1:]}))[DIMENSIONS[1]]["value"], None)

    def test_judge_input_anonymous_and_scoring_input_no_metadata(self):
        row = {"prompt":[{"role":"user","content":"x?"}], "response":"Answer: 2", "ground_truth":"2",
               "source_role":"teacher", "model":"secret-model", "judge_score":1}
        rendered = json.dumps(judge_messages(row,"spec","instructions"))
        self.assertNotIn("secret-model", rendered)
        self.assertNotIn("source_role", rendered)
        self.assertNotIn("judge_score", rendered)

    def test_last_valid_response_token_padding_empty(self):
        logits = torch.arange(2*5*4).reshape(2,5,4)
        masks = torch.tensor([[0,1,1,0,0], [0,0,0,1,1]])
        torch.testing.assert_close(last_response_logits(logits,masks), torch.stack([logits[0,2],logits[1,4]]))
        with self.assertRaises(ValueError):
            last_response_logits(logits,torch.zeros(2,5))

    def test_eos_and_no_silent_truncation(self):
        class Tokenizer:
            eos_token_id=9
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return [1,2]
            def encode(self, text, **kwargs):
                return [3,4] if text else []
        tokenizer = Tokenizer()
        generation = {"thinking_control":True,"enable_thinking":False,
                      "max_prompt_length":2,"max_response_length":3}
        row = {"prompt":[{"role":"user","content":"x"}],"response":"text","truncated":False,
               "ground_truth":"secret", "source_role":"teacher"}
        encoded = encode_response(tokenizer,row,generation)
        self.assertEqual(encoded["input_ids"], [1,2,3,4,9])
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertEqual(encode_response(tokenizer, dict(row,truncated=True),generation)["input_ids"],[1,2,3,4])
        with self.assertRaises(ValueError):
            encode_response(tokenizer,row,dict(generation,max_response_length=2))

    def test_generated_tokens_preserved_instead_of_retokenizing(self):
        class Tokenizer:
            eos_token_id=9
            def apply_chat_template(self,messages,**kwargs):
                return [1]
            def encode(self,text,**kwargs):
                return [5]*5
            def decode(self,tokens,**kwargs):
                return "response" if tokens==[3,4] else "wrong"
        row={"prompt":[],"response":"response","generated_token_ids":[3,4],"truncated":True}
        settings={"thinking_control":False,"max_prompt_length":1,"max_response_length":2}
        self.assertEqual(encode_response(Tokenizer(),row,settings)["input_ids"],[1,3,4])
        with self.assertRaises(ValueError):
            encode_response(Tokenizer(),dict(row,generated_token_ids=[3]),settings)

    def test_expand_head_strict_reload_and_scores(self):
        from transformers import Qwen3Config, Qwen3ForTokenClassification
        config=Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=8, num_labels=1)
        model=Qwen3ForTokenClassification(config).eval()
        inputs=torch.tensor([[1,2,3]])
        encoded={"input_ids":[1,2,3],"attention_mask":[1,1,1],"response_mask":[0,1,1]}
        torch.testing.assert_close(score_response(model,encoded,"cpu"),model(inputs).logits[0,-1])
        old_weight=model.score.weight.detach().clone()
        old_bias=model.score.bias.detach().clone()
        before=model(inputs).logits.detach()
        expand_head(model,42,"rubric")
        torch.testing.assert_close(model.score.weight[0],old_weight[0],rtol=0,atol=0)
        torch.testing.assert_close(model.score.bias[0],old_bias[0],rtol=0,atol=0)
        torch.testing.assert_close(model(inputs).logits[:,:,:1],before)
        with TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored=load_classifier(path,rubric_hash="rubric").eval()
            torch.testing.assert_close(restored(inputs).logits,model(inputs).logits,rtol=0,atol=0)
            with self.assertRaises(ValueError):
                load_classifier(path)
            with self.assertRaises(ValueError):
                load_classifier(path,rubric_hash="wrong")

    def test_auxiliary_streaming_matches_whole_batch_gradient(self):
        logits=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
        targets=torch.tensor([[1,.5,0],[0,0,1],[1,1,0]],dtype=torch.float64)
        masks=torch.tensor([[1,1,0],[1,0,1],[1,1,0]],dtype=torch.float64)
        whole=auxiliary_loss(logits,targets,masks)
        streamed=sum(auxiliary_contribution(logits[i],targets[i],masks[i],masks.sum(0)) for i in range(3))
        torch.testing.assert_close(whole,streamed)
        a=torch.autograd.grad(whole,logits,retain_graph=True)[0]
        b=torch.autograd.grad(streamed,logits)[0]
        torch.testing.assert_close(a,b)
        self.assertEqual(float(auxiliary_loss(logits,targets,torch.zeros_like(masks))),0)

    def test_teacher_auxiliary_dedup_and_original_pairs_full_gradient(self):
        # Different masks per response make a per-microbatch mean fail this test.
        records={}
        for i,rid in enumerate(("teacher","s1","s2")):
            records[rid]={"annotation":{"labels":{dim:label((i%2 if dim==DIMENSIONS[0] else .5),
                                                           valid=not(i==1 and dim==DIMENSIONS[1]))
                                                        for dim in DIMENSIONS}}}
        batch=[{"teacher_id":"teacher","student_ids":["s1","s2"],"records":records}]
        model=torch.nn.Linear(2,4).double()
        reference=copy.deepcopy(model)
        encoded={rid:torch.tensor([i+1,.2],dtype=torch.float64) for i,rid in enumerate(records)}
        with patch("rubric.model.score_response",side_effect=lambda m,x,d:m(x)):
            stats=backward_batch(model,encoded,batch,"joint",.1,"cpu")
        logits=torch.stack([reference(encoded[rid]) for rid in records])
        targets,masks=zip(*(label_tensors(records[rid]["annotation"],"joint") for rid in records))
        full=(pair_loss(logits[0,0],logits[1,0])+pair_loss(logits[0,0],logits[2,0]))/2
        full=full+.1*auxiliary_loss(logits[:,1:],torch.tensor(targets).double(),torch.tensor(masks).double())
        full.backward()
        torch.testing.assert_close(model.weight.grad,reference.weight.grad)
        torch.testing.assert_close(model.bias.grad,reference.bias.grad)
        self.assertEqual(stats["unique_responses"],3)
        self.assertEqual(stats["pairs"],2)

    def test_quality_metrics_conflicts_and_bootstrap_unit(self):
        labels={d:label(1,True) for d in DIMENSIONS}
        records={"t":{"adv":2,"acc":0,"labels":labels},
                 "s1":{"adv":1,"acc":1,"labels":labels},
                 "s2":{"adv":0,"acc":0,"labels":labels}}
        result=metrics([{"teacher_id":"t","student_ids":["s1","s2"],"records":records}],20)
        self.assertEqual(result["d_acc"],1)
        self.assertEqual(result["teacher_wrong_student_right"]["quality_ordering"],0)
        self.assertEqual(result["best_of_2"],1)
        self.assertEqual(result["ci95_by_prompt_bootstrap"]["best_of_2"],[1,1])
        self.assertEqual(auroc([1,1],[0,1]),.5)
        self.assertIsNone(auroc([1,2],[1,1]))
        group={"uid":"fixture","teacher_id":"t","student_ids":["s1","s2"],"records":records}
        self.assertEqual(paired_deltas([group],[group],20)["best_of_2"]["ci95"],[0,0])

    def test_dry_run_any_cwd_no_gpu_import_or_artifact(self):
        script = ROOT/"rubric/pipeline.py"
        with TemporaryDirectory() as directory:
            code=("import runpy,sys; sys.argv="+repr([str(script),"--dry-run"])+"; "
                  "runpy.run_path("+repr(str(script))+",run_name='__main__'); "
                  "assert 'torch' not in sys.modules and 'vllm' not in sys.modules")
            process=subprocess.run([sys.executable,"-c",code],cwd=directory,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_weight_hash_cache_detects_same_size_content_replacement(self):
        import os
        with TemporaryDirectory() as directory:
            path=Path(directory)/"model"
            path.mkdir()
            weights=path/"model.safetensors"
            weights.write_bytes(b"initial")
            cache=Path(directory)/"cache.json"
            original=model_fingerprint(path,cache)
            self.assertEqual(model_fingerprint(path,cache),original)
            metadata=weights.stat()
            weights.write_bytes(b"changed")
            os.utime(weights,ns=(metadata.st_atime_ns,metadata.st_mtime_ns))
            changed=model_fingerprint(path,cache)
            self.assertNotEqual(original["digest"],changed["digest"])


if __name__=="__main__":
    unittest.main()
