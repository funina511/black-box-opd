"""Check the eight-GPU launcher budgets without model weights or a training process."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import h200_8gpu as node


class NodeRecipe(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.train = self.root / "train.parquet"
        pq.write_table(pa.table({"id": range(17405)}), self.train)
        original = node.compose_config

        def compose(*args, **kwargs):
            cfg = original(*args, **kwargs)
            cfg.data.update({"data.train_file": str(self.train), "validation.quick_file": str(self.train)})
            return cfg

        mock_compose = patch.object(node, "compose_config", compose)
        mock_compose.start()
        self.addCleanup(mock_compose.stop)

    def config(self, method, *options):
        target = self.root / "plan.yaml"
        if target.exists():
            target.unlink()
        argv = ["h200_8gpu.py", method, "--runs-root", str(self.root / "runs"),
                "--write-config", str(target), *options]
        with patch.object(sys, "argv", argv), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            node.main()
        return yaml.safe_load(target.read_text())

    def test_same_global_budget_across_methods(self):
        base = self.config("base")
        self.assertEqual(base["budget.rlvr_eval_steps"], [271, 542, 813])
        self.assertEqual(Path(base["run.output_root"]).name, "e3")
        options = ("--prompt-batch", "128", "--actor-tokens", "16384", "--rollout-batched-tokens", "20480")
        for method in ("seqkd", "base", "correctness", "rlvr", "naive"):
            with self.subTest(method=method):
                cfg = self.config(method, *options)
                self.assertEqual(Path(cfg["run.output_root"]).name, "e3_b128")
                self.assertEqual(cfg["data.train_batch_size"], 128)
                self.assertEqual(cfg["budget.warmup_steps"], 135)
                self.assertEqual(cfg["budget.gad_new_steps"], 270)
                self.assertEqual(cfg["budget.rlvr_steps"], 405)
                self.assertEqual(cfg["budget.seqkd_steps"], 405)
                self.assertEqual(cfg["budget.rlvr_eval_steps"], [135, 270, 405])
                self.assertEqual(cfg["optim.actor_ppo_max_token_len_per_gpu"], 16384)
                self.assertEqual(cfg["rollout.max_num_batched_tokens"], 20480)
                if method == "naive":
                    self.assertEqual(cfg["rubric.process_sampling.max_unique_per_global_minibatch"], 128)
                    self.assertFalse(cfg["rubric.judge.enable_thinking"])
        self.assertEqual(self.config("rlvr", "--epochs", "2")["budget.rlvr_eval_steps"], [271, 542])

    def test_external_handoff_uses_source_batch_and_warmup_step(self):
        cfg = self.config("base", "--prompt-batch", "128")
        source = Path(cfg["run.output_root"]) / cfg["run.run_id"]
        source.mkdir(parents=True)
        (source / "train.yaml").write_text(yaml.safe_dump(cfg))
        options = ("--smoke", "--warmup-source", str(source))
        with self.assertRaises(SystemExit):
            self.config("naive", *options)
        target = self.config("naive", "--prompt-batch", "128", *options)
        self.assertEqual(target["budget.init_step"], 135)
        self.assertEqual(target["budget.gad_new_steps"], 3)

    def test_invalid_global_batches(self):
        for b in ("0", "65", "20000"):
            with self.subTest(global_batch=b), self.assertRaises(SystemExit):
                self.config("seqkd", "--prompt-batch", b)

    def test_judge_output_budget_matches_model_mode(self):
        brief = self.config("judge")
        instruct = self.config("judge", "--judge-protocol", "direct_v1")
        thinking = self.config("judge", "--judge-model", "8b-thinking")
        self.assertEqual(brief["rubric.judge.max_output_tokens"], 768)
        self.assertEqual(brief["rubric.process_protocol"], "direct_brief_v1")
        self.assertEqual(len({s["rubric.judge.base_url"] for s in (brief, instruct, thinking)}), 3)
        self.assertEqual(len({s["run.run_id"] for s in (brief, instruct, thinking)}), 3)
        self.assertEqual(instruct["rubric.judge.max_output_tokens"], 128)
        self.assertFalse(instruct["rubric.judge.enable_thinking"])
        self.assertIsNone(instruct["rubric.judge.reasoning_parser"])
        self.assertEqual(thinking["rubric.judge.max_output_tokens"], 8192)
        self.assertTrue(thinking["rubric.judge.enable_thinking"])
        self.assertEqual(thinking["rubric.judge.reasoning_parser"], "deepseek_r1")
        for key in ("temperature", "top_p", "top_k", "min_p"):
            self.assertEqual(instruct[f"rubric.judge.{key}"], thinking[f"rubric.judge.{key}"])
            self.assertEqual(brief[f"rubric.judge.{key}"], instruct[f"rubric.judge.{key}"])


if __name__ == "__main__":
    unittest.main()
