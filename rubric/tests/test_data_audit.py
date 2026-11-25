import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from pathlib import Path
from rubric.common import *
from rubric.schema import *
from rubric.data import pack, load_groups, load_pool
from rubric.audit import audit


class DataAudit(unittest.TestCase):
    def setUp(self):
        self.temporary=TemporaryDirectory()
        self.cfg=load_config(ROOT/"rubric/configs/offline_v1.yaml")
        self.cfg["run"].update(name="fixture",output_root=str(Path(self.temporary.name)/"runs"))
        self.cfg["data"]["output_root"]=str(Path(self.temporary.name)/"data")
        self.run,self.dataset=claim_run(self.cfg)
        self.dataset.mkdir(parents=True)
        rows,annotations,groups=[],[],[]
        spec_hash=file_hash(self.cfg["rubric"]["spec"])
        for i,part in enumerate(("train","dev","test")):
            ids=[]
            for role in ("teacher","student"):
                prompt=[{"role":"user","content":f"Question {i}"}]
                response=f"{role} Answer: 2"
                row={"schema_version":TRAJECTORY_SCHEMA,"uid":str(i),"split":part,"prompt":prompt,
                     "prompt_hash":hash_raw_prompt(prompt),"response":response,
                     "response_hash":digest(response),"response_id":response_id(prompt,response),
                     "ground_truth":"2","source_role":role,"truncated":False,
                     "original_response_tokens":10,"input_eligible":True}
                rows.append(row)
                rid=row["response_id"]
                ids.append(rid)
                annotation={"schema_version":ANNOTATION_SCHEMA,"response_id":rid,"rubric_version":"math_v1",
                            "rubric_hash":spec_hash,"judge_fingerprint":"fixture","judge_prompt_hash":"fixture",
                            "labels":{d:label(1,True,"fixture") for d in DIMENSIONS},
                            "evidence":{d:"fixture evidence" for d in DIMENSIONS[1:]},
                            "review_status":"pending","judge_json_valid":True,
                            "raw_scorer_result":{"acc":1,"pred":"2","score":1}}
                annotation["cache_key"]=annotation_key(row,spec_hash,"fixture","fixture")
                annotations.append(annotation)
            groups.append({"uid":str(i),"split":part,"teacher_id":ids[0],"student_ids":[ids[1],ids[1]]})
        write_jsonl(self.dataset/"trajectories.jsonl",rows)
        write_jsonl(self.dataset/"annotations.raw.jsonl",annotations)
        write_jsonl(self.dataset/"groups.jsonl",groups)
        write_json(self.dataset/"splits.json",{"train":["0"],"dev":["1"],"test":["2"]})
        write_json(self.dataset/"manifest.json",{"collection":{
            "trajectory_hash":file_hash(self.dataset/"trajectories.jsonl"),
            "group_hash":file_hash(self.dataset/"groups.jsonl")}})

    def tearDown(self):
        self.temporary.cleanup()

    def test_audit_pending_then_explicit_human_and_locked_pack(self):
        summary=audit(self.cfg)
        self.assertFalse(summary["quality_gate_passed"])
        self.assertEqual(summary["human_review_count"],0)
        with self.assertRaises(ValueError):
            pack(self.cfg)
        reviews=read_jsonl(self.dataset/"audit/review_requests.jsonl")
        for row in reviews:
            row.update(reviewer="synthetic-test-fixture",accepted=True,severe_reverse=False,reason="test fixture")
        write_jsonl(self.dataset/"audit/reviews.jsonl",reviews)
        self.assertTrue(audit(self.cfg)["quality_gate_passed"])
        pack(self.cfg)
        group=load_groups(self.dataset,"train")[0]
        self.assertEqual(len(group["records"]),2)
        self.assertEqual(len(group["student_ids"]),2)
        self.assertEqual(group["split"],"train")
        labels=read_jsonl(self.dataset/"annotations.reviewed.jsonl")
        labels[0]["labels"][DIMENSIONS[0]]["value"]=0
        write_jsonl(self.dataset/"annotations.reviewed.jsonl",labels)
        with self.assertRaises(ValueError):
            pack(self.cfg)

    def test_cross_split_candidate_rejected_even_if_group_hash_matches(self):
        audit(self.cfg)
        groups=read_jsonl(self.dataset/"groups.jsonl")
        groups[0]["student_ids"][0]=groups[1]["student_ids"][0]
        write_jsonl(self.dataset/"groups.jsonl",groups)
        manifest=read_json(self.dataset/"manifest.json")
        manifest["collection"]["group_hash"]=file_hash(self.dataset/"groups.jsonl")
        write_json(self.dataset/"manifest.json",manifest)
        with self.assertRaisesRegex(ValueError,"cross-question"):
            pack(self.cfg,exploratory=True)

    def test_modified_trajectory_cache_rejected(self):
        rows=read_jsonl(self.dataset/"trajectories.jsonl")
        rows[0]["ground_truth"]="3"
        write_jsonl(self.dataset/"trajectories.jsonl",rows)
        with self.assertRaisesRegex(ValueError,"trajectory pool changed"):
            load_pool(self.dataset)

    def test_batched_judge_and_cache_resume_without_model_loading(self):
        from rubric.annotate import annotate
        self.cfg["judge"]["backend"]="vllm"
        batches=[]
        class Tokenizer:
            def encode(self,text,**kwargs):
                return [1]*100
        class Engine:
            tokenizer=Tokenizer()
            def __init__(self,*args,**kwargs):
                pass
            def render(self,messages):
                return json.dumps(messages)
            def close(self):
                pass
            def generate(self,messages,**kwargs):
                batches.append(len(messages))
                raw=json.dumps({dim:{"value":1,"valid":True,"evidence":"fixture"}
                                for dim in DIMENSIONS[1:]})
                return [{"seconds_batch":.01,"outputs":[{"text":raw,"truncated":False}]} for _ in messages]
        with patch("rubric.annotate.model_fingerprint",return_value="fixture"), patch("rubric.inference.Generator",Engine):
            annotate(self.cfg)
        self.assertEqual(batches,[7])
        self.assertEqual(len(read_jsonl(self.dataset/"annotations.raw.jsonl")),6)
        with patch("rubric.annotate.model_fingerprint",return_value="fixture"), patch("rubric.inference.Generator") as engine:
            annotate(self.cfg)
            engine.assert_not_called()


if __name__=="__main__":
    unittest.main()
