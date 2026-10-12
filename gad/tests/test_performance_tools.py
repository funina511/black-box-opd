"""Focused checks for performance attribution and isolated benchmark safety."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import analyze_training_perf as perf  # noqa: E402
import benchmark_train as bench  # noqa: E402
import compare_training_perf as compare_perf  # noqa: E402
from config import load_config, load_run_config, validate_config  # noqa: E402


class PerformanceToolsTests(unittest.TestCase):
    def test_normal_save_and_validation_steps_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "warmup.log"
            log.write_text("\n".join([
                "step:101 - training/global_step:101.000 - response_length/mean:6000.000 - response_length/clip_ratio:0.5 - timing_s/generate_sequences:200.000 - timing_s/gen:230.000 - timing_s/step:400.000",
                "step:102 - training/global_step:102.000 - response_length/mean:8192.000 - timing_s/generate_sequences:200.000 - timing_s/gen:240.000 - timing_s/save_checkpoint:20.000 - timing_s/step:430.000",
                "step:103 - training/global_step:103.000 - timing_s/gen:250.000 - timing_s/testing:1000.000 - timing_s/step:1400.000",
            ]))
            rows = perf.parse_steps(log)
            self.assertEqual([row["kind"] for row in rows], ["normal", "save", "validation"])
            report = perf.summarize_steps(rows, stage="warmup", prompt_count=64, candidate_count=1)
            self.assertEqual(report["timing_seconds"]["normal"]["gen"]["mean"], 230)
            self.assertEqual(report["estimated_generated_tokens_per_second"]["mean"], 1920)
            self.assertEqual(report["response_length_mean_tokens"]["mean"], 6000)
            self.assertEqual(report["kind_counts"], {"normal": 1, "save": 1, "validation": 1})

    def test_benchmark_rejects_incomplete_optimizer_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "warmup/global_step_100/actor"
            path.mkdir(parents=True)
            with self.assertRaises(Exception):
                bench.require_complete_checkpoint(root, 100)

    def test_short_screen_compares_only_common_normal_steps(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for name, target, kv in (("baseline", 107, 0.4), ("kv50", 105, 0.5)):
                run = root / name
                config = run / "reports/resolved_benchmark_config.yaml"
                config.parent.mkdir(parents=True)
                config.write_text(yaml.safe_dump({
                    "run.run_id": name,
                    "data.train_batch_size": 64,
                    "rollout.n_warmup": 1,
                    "rollout.gpu_memory_utilization": kv,
                }))
                log = run / "logs/warmup.log"
                log.parent.mkdir()
                log.write_text("\n".join(
                    f"step:{step} - training/global_step:{step}.000 - "
                    f"response_length/mean:6000.000 - timing_s/generate_sequences:200.000 - "
                    f"timing_s/update_critic:80.000 - timing_s/update_actor:30.000 - "
                    + ("timing_s/save_checkpoint:20.000 - " if step == target else "")
                    + "timing_s/step:400.000"
                    for step in range(101, target + 1)
                ))
                results.append({
                    "candidate": name, "ok": True,
                    "source_checkpoint": "/source/warmup/global_step_100",
                    "source_step": 100, "target_step": target, "warmup_steps": 2,
                    "completed_steps": list(range(101, target + 1)),
                    "config": str(config), "backend_commit": "same",
                    "backend_file_sha256": {"trainer": "same"},
                    "gpu_peak_memory_mib_per_card": {"1": 20000},
                    "training": {"timing_seconds": {"save": {"save_checkpoint": {"mean": 20}}}},
                })
            text = compare_perf.compare(results)
            self.assertIn("Common measured normal steps: [103, 104]", text)
            self.assertIn("| kv50 | 5 | 2 |", text)
            config = Path(results[1]["config"])
            config.write_text(config.read_text().replace("train_batch_size: 64", "train_batch_size: 32"))
            with self.assertRaisesRegex(ValueError, "outside the A/B performance knobs"):
                compare_perf.compare(results)

    def test_critic_micro_must_divide_normalized_minibatch(self):
        cfg = load_run_config("qwen3_1.7b").for_stage("warmup")
        cfg.data["optim.critic_micro_batch_size_per_gpu"] = 3
        problems = validate_config(cfg)
        self.assertTrue(any("normalized critic minibatch" in item for item in problems), problems)
        cfg.data["optim.critic_micro_batch_size_per_gpu"] = 2
        self.assertFalse(any("normalized critic minibatch" in item for item in validate_config(cfg)))


if __name__ == "__main__":
    unittest.main()
