"""The binary GRPO reward is attached only to the final visible student token."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "gad/tools")]
from rlvr_reward import verified_rewards
from scoring import math as scorer


class VerifiedRewardTests(unittest.TestCase):
    def test_verified_wrong_and_unscorable_have_distinct_metrics(self):
        texts = {1: r"Answer: \boxed{4}", 2: r"Answer: \boxed{5}", 3: "unfinished"}
        tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: texts[tokens[0]])
        batch = SimpleNamespace(
            batch={"responses": torch.tensor([[1, 9, 0], [2, 0, 0], [3, 9, 9], [1, 0, 0]]),
                   "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]])},
            non_tensor_batch={"reward_model": [{"ground_truth": gt} for gt in ("4", "4", "4", "")]},
        )
        rewards, metrics = verified_rewards(batch, tokenizer, scorer)
        self.assertEqual(rewards.tolist(), [[0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(metrics, {"rlvr/responses": 4, "rlvr/count_correct": 1,
                                   "rlvr/count_incorrect": 1, "rlvr/count_no_answer": 1,
                                   "rlvr/count_gt_unparseable": 1})


if __name__ == "__main__":
    unittest.main()
