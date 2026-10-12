#!/usr/bin/env python3
"""CPU tests for the ``gad-math-validation`` backend patch.

The plan (section 3.3) requires every branch to report dev-set math accuracy
during validation, using the CARD-aligned scorer, with ROUGE-L kept as the
original auxiliary diagnostic. Those numbers used to be missing entirely: the
whole training-time "validation metric" was ROUGE-L.

This suite imports the **patched** ``verl.trainer.ppo.ray_trainer`` from a
backend worktree (no GPU, no Ray cluster: the module only needs to be
importable) and exercises the code the patch inserts:

* the batch scorer computes accuracy from ``acc`` and never confuses it with the
  mean of the +/-1 ``score``;
* a response that is not truncated, one that is, and one whose answer cannot be
  parsed are classified as the plan requires;
* the inserted ``_validate`` blocks - extracted verbatim from the patched source
  and executed against fake batches and a real OmegaConf config - produce the
  ``val/math/*`` metric keys and the ``validation_math_metrics.jsonl`` artifact;
* a configured-but-unusable scorer fails loudly, while an unconfigured run keeps
  the original ROUGE-only behaviour instead of crashing;
* nothing the patch adds touches ``token_level_rewards``/advantages.

Run::

    python LMOps/gad/tests/test_math_validation.py
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

GAD_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = GAD_DIR / "tools"
BACKEND_ROOT = GAD_DIR / ".backends"
SCORER = GAD_DIR.parent / "sample" / "math_scorer.py"

sys.path.insert(0, str(TOOLS_DIR))

MARKER = "PATCH(gad:math-validation)"
END = MARKER + " end"


def extract_block(source: str, *, occurrence: int) -> str:
    """Return the ``occurrence``-th marker block (start marker .. end marker), dedented."""
    lines = source.splitlines()
    starts = [
        index
        for index, line in enumerate(lines)
        if MARKER in line and " end" not in line and "dev-set math accuracy" not in line
    ]
    if occurrence >= len(starts):
        raise AssertionError(f"marker block {occurrence} not found (found {len(starts)})")
    start = starts[occurrence]
    for stop in range(start + 1, len(lines)):
        if END in lines[stop]:
            return textwrap.dedent("\n".join(lines[start:stop + 1]))
    raise AssertionError(f"no end marker after line {start}")


class FakeConfig:
    """Minimal stand-in for the trainer's OmegaConf config."""

    class _Data(dict):
        pass

    def __init__(self, local_dir: Path, *, response_length: int = 64, val_n: int = 1, scorer: str | None = None):
        from omegaconf import OmegaConf

        payload = {
            "data": {"max_response_length": response_length},
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": response_length,
                    "val_kwargs": {"n": val_n, "do_sample": False, "temperature": 0.0},
                }
            },
            "trainer": {"default_local_dir": str(local_dir)},
        }
        if scorer:
            payload["data"]["gad_math_scorer_path"] = scorer
        self._cfg = OmegaConf.create(payload)

    def __getattr__(self, item):
        return getattr(self._cfg, item)


class _Batch:
    def __init__(self, tensors: dict):
        self.batch = tensors


class _Proto:
    def __init__(self, non_tensors: dict):
        self.non_tensor_batch = non_tensors


class MathValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = BACKEND_ROOT / "gad"
        cls.trainer_source = cls.backend / "verl" / "trainer" / "ppo" / "ray_trainer.py"
        if not cls.trainer_source.exists():
            raise unittest.SkipTest(f"backend not prepared: {cls.trainer_source}")
        if not SCORER.exists():
            raise unittest.SkipTest(f"vendored scorer missing: {SCORER}")
        cls.source = cls.trainer_source.read_text(encoding="utf-8")

        sys.path.insert(0, str(cls.backend))
        try:
            import verl.trainer.ppo.ray_trainer as ray_trainer
        except Exception as exc:  # noqa: BLE001 - a missing dependency skips, a broken patch fails below
            raise unittest.SkipTest(f"cannot import the patched ray_trainer here: {type(exc).__name__}: {exc}")
        cls.ray_trainer = ray_trainer

    # ------------------------------------------------------------------ patch
    def test_patch_is_present_and_verified(self) -> None:
        import backend_patches

        patch = backend_patches.MathValidationPatch()
        ok, problems = patch.verify(self.backend)
        self.assertTrue(ok, problems)
        self.assertEqual(self.source.count(END), 3, "expected three inserted marker blocks")

    def test_patched_validate_still_computes_rouge(self) -> None:
        self.assertIn('metric_dict["val/rouge-L/mean"] = np.mean(sample_scores)', self.source)
        self.assertIn("safe_rouge_score", self.source)

    # ---------------------------------------------------------------- scorer
    def test_unusable_scorer_fails_loudly_and_absent_scorer_is_off(self) -> None:
        self.assertIsNone(self.ray_trainer._gad_load_math_scorer(None))
        with self.assertRaises(RuntimeError):
            self.ray_trainer._gad_load_math_scorer(str(self.backend / "does" / "not" / "exist.py"))

    def test_batch_scoring_separates_accuracy_from_score_mean(self) -> None:
        scorer = self.ray_trainer._gad_load_math_scorer(str(SCORER))
        rows = self.ray_trainer._gad_score_math_batch(
            scorer,
            [
                "Let me compute. Answer: 142",          # correct
                "reasoning ... Answer: 99",             # wrong
                "The answer is \\boxed{142}",           # no `Answer:` -> unparsable
                "work\nAnswer: 142",                    # correct
            ],
            ["142", "142", "142", "142"],
            [10, 20, 30, 40],
            64,
        )
        summary = self.ray_trainer._gad_math_metrics(rows)
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["acc"], 0.5, places=6)
        # mean of the +/-1 score is 0.0 here, so a score mean must never be
        # reported as accuracy (plan section 3.3).
        self.assertAlmostEqual(summary["score_mean"], 0.0, places=6)
        self.assertAlmostEqual(summary["answer_parse_failure_rate"], 0.25, places=6)
        self.assertAlmostEqual(summary["response_len_mean"], 25.0, places=6)

    def test_truncation_uses_the_rollout_response_length(self) -> None:
        scorer = self.ray_trainer._gad_load_math_scorer(str(SCORER))
        rows = self.ray_trainer._gad_score_math_batch(
            scorer, ["Answer: 1", "Answer: 1", "Answer: 1"], ["1", "1", "1"], [64, 63, 65], 64
        )
        self.assertEqual([row["truncated"] for row in rows], [1.0, 0.0, 1.0])
        self.assertAlmostEqual(self.ray_trainer._gad_math_metrics(rows)["truncated_rate"], 2 / 3, places=6)

    def test_scorer_exception_counts_the_row_as_incorrect(self) -> None:
        class Exploding:
            @staticmethod
            def compute_score(solution_str, ground_truth):
                raise ValueError("boom")

        row = self.ray_trainer._gad_score_math_row(Exploding(), "Answer: 1", "1")
        self.assertEqual(row, {"acc": 0.0, "score": -1.0, "answer_parsed": 0.0})

    # ------------------------------------------------- patched _validate path
    def _run_validate_blocks(self, *, scorer_path: str | None, local_dir: Path, out_dir: Path):
        """Execute the inserted _validate blocks against fake batches."""
        import torch

        setup = extract_block(self.source, occurrence=0)
        batch_block = extract_block(self.source, occurrence=1)
        metric_block = extract_block(self.source, occurrence=2)

        config = FakeConfig(local_dir, response_length=8, val_n=1, scorer=None)
        self_obj = type("FakeTrainer", (), {})()
        self_obj.config = config
        self_obj.global_steps = 7

        response_length = 8
        # one untruncated response, one that fills the whole window (truncated)
        responses = torch.zeros((2, response_length), dtype=torch.long)
        attention = torch.zeros((2, response_length), dtype=torch.long)
        attention[0, :4] = 1
        attention[1, :] = 1
        test_output_gen_batch = _Batch({"responses": responses, "attention_mask": attention})
        test_batch = _Proto(
            {
                "reward_model": [
                    {"ground_truth": "142", "style": "rule"},
                    {"ground_truth": "7", "style": "rule"},
                ]
            }
        )
        namespace = {
            "self": self_obj,
            "test_output_gen_batch": test_output_gen_batch,
            "test_batch": test_batch,
            "output_texts": ["Answer: 142", "Answer: 3"],
            "metric_dict": {},
            # the original _validate uses a defaultdict(list), so the patched
            # `reward_extra_infos_dict["acc"]` append must work with one
            "reward_extra_infos_dict": collections.defaultdict(list, {"reward": [0.5, 0.5]}),
            "os": os,
            "json": json,
            "time": time,
            "scope": "full",
            "_gad_started": time.perf_counter(),
            "_gad_timing": {"generation": 0.0, "decode": 0.0, "rouge_l": 0.0, "math_scoring": 0.0},
        }
        namespace.update(
            {name: getattr(self.ray_trainer, name) for name in dir(self.ray_trainer) if name.startswith("_gad_")}
        )

        old_env = os.environ.get("GAD_MATH_SCORER")
        if scorer_path:
            os.environ["GAD_MATH_SCORER"] = scorer_path
        else:
            os.environ.pop("GAD_MATH_SCORER", None)
        try:
            exec(compile(setup, "<setup>", "exec"), namespace)
            exec(compile(batch_block, "<batch>", "exec"), namespace)
            exec(compile(metric_block, "<metrics>", "exec"), namespace)
        finally:
            if old_env is None:
                os.environ.pop("GAD_MATH_SCORER", None)
            else:
                os.environ["GAD_MATH_SCORER"] = old_env
        return namespace

    def test_inserted_blocks_produce_metrics_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            namespace = self._run_validate_blocks(scorer_path=str(SCORER), local_dir=tmp_path, out_dir=tmp_path)
            metrics = namespace["metric_dict"]
            self.assertEqual(metrics["val/math/acc"], 0.5)
            self.assertEqual(metrics["val/math/count"], 2.0)
            self.assertEqual(metrics["val/math/truncated_rate"], 0.5)
            self.assertEqual(metrics["val/math/answer_parse_failure_rate"], 0.0)
            self.assertIn("val/math/score_mean", metrics)
            self.assertIn("val/math/response_len_mean", metrics)

            # the original ROUGE aggregation inputs are untouched apart from the
            # extra `acc` variable the metric dict consumes
            self.assertEqual(namespace["reward_extra_infos_dict"]["reward"], [0.5, 0.5])
            self.assertEqual(namespace["reward_extra_infos_dict"]["acc"], [1.0, 0.0])
            self.assertNotIn("token_level_rewards", namespace)

            artifact = tmp_path / "validation_math_metrics.jsonl"
            self.assertTrue(artifact.exists())
            record = json.loads(artifact.read_text(encoding="utf-8").strip())
            self.assertEqual(record["global_step"], 7)
            self.assertEqual(record["scorer"], str(SCORER))
            self.assertEqual(record["val_kwargs_n"], 1)
            self.assertFalse(record["val_do_sample"])
            self.assertEqual(record["max_response_length"], 8)
            self.assertEqual(record["acc"], 0.5)
            self.assertIn("EOS", record["truncated_definition"])
            self.assertIn("response_length", record["truncated_definition"])

            # a second validation pass appends instead of overwriting
            self._run_validate_blocks(scorer_path=str(SCORER), local_dir=tmp_path, out_dir=tmp_path)
            self.assertEqual(len(artifact.read_text(encoding="utf-8").strip().splitlines()), 2)

    def test_unconfigured_scorer_keeps_rouge_only_behaviour(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            namespace = self._run_validate_blocks(scorer_path=None, local_dir=tmp_path, out_dir=tmp_path)
            metrics = namespace["metric_dict"]
            self.assertFalse([key for key in metrics if key.startswith("val/math/")], metrics)
            self.assertEqual(namespace["reward_extra_infos_dict"], {"reward": [0.5, 0.5]})
            self.assertFalse((tmp_path / "validation_math_metrics.jsonl").exists())

    # -------------------------------------------------------------- frozen
    def test_only_known_patches_were_added_to_ray_trainer(self) -> None:
        """The backend must match the pinned source plus the known patches.

        Validation scope includes timing imports and two replaced statements,
        so a purely insertion-based assertion no longer describes that patch.
        Reward/advantage byte equality is checked separately below.
        """
        import shutil
        import backend_patches

        pristine = GAD_DIR / "verl"
        trainer = Path("verl/trainer/ppo/ray_trainer.py")
        config = Path("verl/trainer/config/ppo_trainer.yaml")
        if not (pristine / trainer).exists():
            self.skipTest(f"pinned checkout missing: {pristine / trainer}")
        with tempfile.TemporaryDirectory() as temporary:
            expected = Path(temporary)
            for relative in (trainer, config):
                (expected / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(pristine / relative, expected / relative)
            for patch in (backend_patches.ResumeWorkerStatePatch(),
                          backend_patches.MathValidationPatch(),
                          backend_patches.ValidationScopePatch()):
                patch.apply(expected)
            self.assertEqual((expected / trainer).read_text(encoding="utf-8"), self.source)

    def test_reward_and_advantage_code_is_byte_identical(self) -> None:
        pristine_path = GAD_DIR / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
        if not pristine_path.exists():
            self.skipTest(f"pinned checkout missing: {pristine_path}")
        pristine = pristine_path.read_text(encoding="utf-8")
        block = pristine[pristine.index("def compute_advantage("):pristine.index("class RayPPOTrainer:")]
        self.assertIn(block, self.source)
        for frozen in (
            "    token_level_rewards = token_level_scores - beta * kld\n",
            '    data.batch["token_level_rewards"] = token_level_rewards\n',
        ):
            self.assertIn(frozen, self.source)
        # the math numbers are validation-only: they never appear in the same
        # statement as a reward/advantage tensor
        for line in self.source.splitlines():
            if "_gad_score_math" in line or "math_rows" in line or "math_summary" in line:
                self.assertNotIn("advantages", line)
                self.assertNotIn("token_level_rewards", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
