"""RLVR milestones survive checkpoint rotation and an interrupted HF publication."""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from rlvr_checkpoint import commit_checkpoint, publish_eval_model, saved_actor_paths
from train_rlvr import main, training_budget


class RLVREvalCheckpoints(unittest.TestCase):
    def test_formal_smoke_and_explicit_step_budgets(self):
        config = {"budget.rlvr_steps": 542, "budget.rlvr_eval_steps": [271, 542]}
        self.assertEqual(training_budget(config, None, False), (542, [271, 542]))
        self.assertEqual(training_budget(config, None, True), (1, [1]))
        self.assertEqual(training_budget(config, 300, False), (300, [271, 300]))
        self.assertEqual(training_budget(config, 2, False), (2, [2]))

    def test_interrupt_and_resume_at_milestone(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            source = stage / "global_step_3/actor/huggingface"
            source.mkdir(parents=True)
            (source / "config.json").write_text("{}")
            (source / "weights").write_bytes(b"trained weights")
            commit_checkpoint(stage, 2, [3, 5])
            marker = stage / "latest_checkpointed_iteration.txt"
            self.assertEqual(marker.read_text(), "2")
            self.assertFalse((stage / "eval_models").exists())
            with patch("rlvr_checkpoint.os.link", side_effect=OSError("interrupted")):
                with self.assertRaises(shutil.Error):
                    commit_checkpoint(stage, 3, [3, 5])
            self.assertEqual(marker.read_text(), "3")
            self.assertFalse((stage / "eval_models/step_3").exists())
            destination = publish_eval_model(stage, 3)
            self.assertTrue(os.path.samefile(source / "weights", destination / "weights"))
            shutil.rmtree(source.parent)
            self.assertEqual(publish_eval_model(stage, 3), destination)
            self.assertEqual((destination / "weights").read_bytes(), b"trained weights")

    def test_restart_rotation_history_and_two_subsequent_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)

            def actor(step):
                return stage / f"global_step_{step}" / "actor"

            def write(step):
                hf = actor(step) / "huggingface"
                hf.mkdir(parents=True)
                (hf / "weights").write_text(str(step))
                (actor(step).parent / "data.pt").write_bytes(b"cursor")
                commit_checkpoint(stage, step, [3, 10])

            for step in (3, 6):
                write(step)
            actor(4).mkdir(parents=True)
            actor(15).mkdir(parents=True)
            (actor(15).parent / "data.pt").touch()
            history = saved_actor_paths(actor(6))
            self.assertEqual(history, [str(actor(3)), str(actor(6))])
            # Mirror the FSDP two-slot FIFO with the history restored at step 6.
            for step in (8, 10):
                shutil.rmtree(history.pop(0))
                write(step)
                history.append(str(actor(step)))
            self.assertFalse(actor(6).exists())
            self.assertFalse(actor(3).exists())
            for step in (3, 10):
                self.assertEqual((stage / f"eval_models/step_{step}/weights").read_text(), str(step))

    def test_completed_frozen_plans_use_their_original_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            hf = run / "rlvr/global_step_2/actor/huggingface"
            hf.mkdir(parents=True)
            (hf / "config.json").write_text("{}")
            (hf / "tokenizer.json").write_text("{}")
            save_file({"weight": torch.zeros(1)}, hf / "model.safetensors")
            (run / "rlvr/latest_checkpointed_iteration.txt").write_text("2")
            for plan in ({"steps": 2}, {"steps": 2, "eval_steps": [2]}):
                (run / "rlvr_plan.json").write_text(json.dumps(plan))
                with patch.object(sys, "argv", ["train_rlvr", "--resume", "--output", tmp]), \
                        patch("train_rlvr.run_command") as train:
                    main()
                train.assert_not_called()
                self.assertEqual((run / "rlvr/eval_models/step_2").exists(), "eval_steps" in plan)


if __name__ == "__main__":
    unittest.main()
