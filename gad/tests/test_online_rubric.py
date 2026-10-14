import json
import random
import sys
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gad/tools"))
from rubric.core.contracts import DIMENSIONS, digest, file_hash
from gad.extensions.label_cache import LabelCache
from gad.extensions.judge_client import JudgeClient, ServiceUnavailable
from gad.extensions.critic_batch import auxiliary_weights
from gad.extensions.online_rubric import OnlineRubric
from config import compose_config, validate_config
from extension_config import hydra_overrides
from pipeline import stage_step_budget, stage_signature


def pair(index, teacher="T", student=None, step=1):
    student = student or "S" + str(index)
    return {"pair_id": str(index), "prompt_id": "p", "prompt_hash": "p", "prompt_ids": [1, 2],
            "prompt": "Solve 1+1", "ground_truth": "2", "source_step": step, "producer_stage": "gad",
            "token_identity": "tok", "prompt_truncated": False,
            "teacher": {"response_id": teacher, "token_ids": [3, 4], "response": "1+1=2. Answer: 2",
                        "truncated": False, "finish_reason": "stop"},
            "student": {"response_id": student, "token_ids": [5, 6], "response": "Answer: 3",
                        "truncated": False, "finish_reason": "stop"}}


class ConfigTests(unittest.TestCase):
    def test_default_disabled_and_new_budget(self):
        cfg = compose_config()
        self.assertFalse(cfg.get("rubric.enabled"))
        self.assertFalse(cfg.get("replay.enabled"))
        self.assertEqual([], hydra_overrides(cfg, "warmup", Path("/tmp"), 0))
        cfg.data["budget.gad_new_steps"] = 80
        self.assertEqual(80, stage_step_budget(cfg, "gad", 271, handoff_step=271))

    def test_modes_fail_fast(self):
        cfg = compose_config(settings=["rubric.enabled=true", "rubric.mode=correctness_only"])
        self.assertEqual([], validate_config(cfg))
        self.assertIn(".backends_gad_extensions_v1", cfg.get("backends.root"))
        cfg.data["rubric.stages"] = ["warmup"]
        self.assertTrue(any("rubric.stages" in row for row in validate_config(cfg)))
        cfg.data["rubric.stages"] = ["warmup", "gad"]
        cfg.data["optim.critic_dynamic_bsz"] = True
        self.assertTrue(any("static" in row for row in validate_config(cfg)))

    def test_resource_overlap(self):
        cfg = compose_config(settings=["rubric.enabled=true", "resources.judge_gpus=1,2"])
        self.assertTrue(any("overlap" in row for row in validate_config(cfg)))

    def test_stage_scope(self):
        cfg = compose_config(settings=["rubric.enabled=true", "rubric.mode=correctness_only", "rubric.stages=[gad]"])
        self.assertEqual(['gad'], cfg.get('rubric.stages'))
        self.assertEqual([], validate_config(cfg))
        self.assertTrue(any("enabled:false" in arg for arg in hydra_overrides(cfg, "warmup", Path("/tmp"), 0)))
        self.assertEqual([], hydra_overrides(cfg, "seqkd", Path("/tmp"), 0))


class CacheTests(unittest.TestCase):
    def test_content_snapshot_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = LabelCache(root / "live")
            cache.put("a", {"value": 1})
            cache.put("a", {"value": 1})
            with self.assertRaises(ValueError):
                cache.put("a", {"value": 0})
            refs = cache.snapshot(root / "snapshot")
            restored = LabelCache(root / "restored")
            restored.restore(root / "snapshot", refs)
            self.assertEqual(1, restored.get("a")["value"])
            (root / "snapshot/a.json").write_text('{}')
            with self.assertRaises(ValueError):
                restored.restore(root / "snapshot", refs)


class LossTests(unittest.TestCase):
    def test_global_dedup_microbatch_gradient(self):
        import torch
        from rubric.core.losses import weighted_contribution
        for dp in (1, 2, 4):
            for micro in (1, 2):
                pairs = [pair(i, teacher="T" + str(i // 4), student="S" + str(i // 2)) for i in range(16)]
                targets = {r: torch.tensor([[.5, 1., 0.] for p in pairs]) for r in ("teacher", "student")}
                masks = {r: torch.ones(16, 3) for r in targets}
                masks["student"][8:, 1] = 0
                weights, counts = auxiliary_weights(pairs, targets, masks, 8, dp)
                per_rank = 16 // dp
                local_mini = 8 // dp
                for offset in range(0, per_rank, local_mini):
                    parameter = torch.tensor([.2, -.3, .4], requires_grad=True)
                    gradients = []
                    for rank in range(dp):
                        parameter.grad = None
                        for start in range(rank * per_rank + offset, rank * per_rank + offset + local_mini, micro):
                            ids = slice(start, start + micro)
                            for role in targets:
                                loss, _ = weighted_contribution(parameter.expand(micro, -1), targets[role][ids], masks[role][ids],
                                                               weights[role][ids], counts[start], dp)
                                loss.backward()
                        gradients.append(parameter.grad.clone())
                    unique = {}
                    for rank in range(dp):
                        for i in range(rank * per_rank + offset, rank * per_rank + offset + local_mini):
                            for role in targets:
                                unique[pairs[i][role]["response_id"]] = (targets[role][i], masks[role][i])
                    from rubric.core.losses import auxiliary_loss
                    parameter.grad = None
                    auxiliary_loss(parameter.expand(len(unique), -1), torch.stack([v[0] for v in unique.values()]),
                                   torch.stack([v[1] for v in unique.values()])).backward()
                    torch.testing.assert_close(torch.stack(gradients).mean(0), parameter.grad)

    def test_all_missing_finite_zero(self):
        import torch
        from rubric.core.losses import weighted_contribution
        logits = torch.randn(2, 3, requires_grad=True)
        loss, _ = weighted_contribution(logits, torch.zeros_like(logits), torch.zeros_like(logits), torch.ones(2), torch.zeros(3), 4)
        loss.backward()
        self.assertEqual(0, loss.item())
        self.assertEqual(0, logits.grad.abs().sum().item())

    def test_inconsistent_duplicate_rejected(self):
        import torch
        pairs = [pair(0), pair(1)]
        targets = {r: torch.ones(2, 3) for r in ("teacher", "student")}
        targets["teacher"][1, 0] = 0
        with self.assertRaises(ValueError):
            auxiliary_weights(pairs, targets, {r: torch.ones(2, 3) for r in targets}, 2, 1)


class ClientTests(unittest.TestCase):
    def options(self):
        return {"base_url": "http://127.0.0.1/v1", "model": "judge", "api_key_env": None,
                "max_output_tokens": 30, "max_context_tokens": 100, "temperature": 0,
                "request_timeout_seconds": 1, "step_timeout_seconds": 5, "max_retries": 1, "concurrency": 2}

    def test_async_response_join_and_anonymity(self):
        class Tokenizer:
            def apply_chat_template(self, *a, **k): return [1]
        c = JudgeClient(self.options(), "fp", Tokenizer())
        labels = {d: {"value": .5, "valid": True, "evidence": "algebra"} for d in DIMENSIONS[1:]}
        def request(suffix, payload=None, timeout=None):
            if suffix == "/models": return {"data": [{"id": "judge"}]}
            if suffix == "/gad_fingerprint": return {"identity_hash": "fp"}
            time.sleep(.02 if payload["messages"][0]["content"] == "slow" else 0)
            self.assertIn("guided_json", payload)
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(labels)}}]}
        c.request = request
        result = c.annotate({"first": [{"role": "user", "content": "slow"}], "second": [{"role": "user", "content": "fast"}]})
        self.assertEqual({"first", "second"}, set(result))
        self.assertTrue(all("parsed" in r for r in result.values()))
        from rubric.core.contracts import judge_messages
        message = judge_messages("p", "r", "2", "spec", "instructions")
        self.assertNotIn("teacher", json.dumps(message))
        self.assertNotIn("source_role", json.dumps(message))

    def test_invalid_json_not_zero_and_no_service_abort(self):
        class Tokenizer:
            def apply_chat_template(self, *a, **k): return [1]
        c = JudgeClient(self.options(), "fp", Tokenizer())
        c.health = lambda *a, **k: None
        c.request = lambda *a, **k: {"choices": [{"finish_reason": "stop", "message": {"content": "bad"}}]}
        result = c.annotate({"r": []})["r"]
        self.assertFalse(result["cacheable"])
        self.assertNotIn("parsed", result)
        def fail(*a, **k): raise ServiceUnavailable("down")
        c.request = fail
        with self.assertRaises(ServiceUnavailable): c.annotate({"r": []})

    def test_context_and_identity(self):
        class Tokenizer:
            def apply_chat_template(self, *a, **k): return list(range(100))
        c = JudgeClient(self.options(), "expected", Tokenizer())
        rid, row = c.annotate_one("r", [], time.monotonic() + 2)
        self.assertEqual("judge_context_exceeded", row["error"])
        c.request = lambda suffix: {"data": [{"id": "judge"}]} if suffix == "/models" else {"identity_hash": "wrong"}
        with self.assertRaises(ServiceUnavailable): c.health()

    def test_step_deadline_includes_health_requests(self):
        c = JudgeClient(self.options(), 'fp', object())
        timeouts = []
        def request(suffix, payload=None, timeout=None):
            timeouts.append(timeout)
            return {'data': [{'id': 'judge'}]} if suffix == '/models' else {'identity_hash': 'fp'}
        c.request = request
        c.health(deadline=time.monotonic() + .05)
        self.assertEqual(2, len(timeouts))
        self.assertTrue(all(0 < timeout <= .05 for timeout in timeouts))
        with self.assertRaises(ServiceUnavailable):
            c.annotate({'r': []}, deadline=time.monotonic() - 1)
        self.assertEqual(2, len(timeouts))


if __name__ == "__main__": unittest.main()
