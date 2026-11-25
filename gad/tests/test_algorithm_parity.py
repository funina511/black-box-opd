#!/usr/bin/env python3
"""Algorithm parity checks against the three pinned backends (CPU, torch only).

Plan section 6 requires that the frozen algorithm be verified against fixed
tensors, not just by reading the diff:

* the discriminator loss must be exactly
  ``-logsigmoid(sum(teacher_values * teacher_mask) - sum(student_values * student_mask)).mean()``
  and must still produce finite gradients for both inputs;
* the Warmup/SeqKD SFT loss must be the masked token-mean negative log-likelihood,
  and crucially it must NOT contain a KL term even though the stage script sets
  ``use_kl_loss=True`` (the KL is added by the actor only for the GAD branch);
* GRPO advantages must be computed per uid group with the original normalization.

Each backend is imported in a *subprocess* with its own PYTHONPATH, so the test
also proves the backends are importable and isolated from each other.

Run::

    python LMOps/gad/tests/test_algorithm_parity.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

GAD_DIR = Path(__file__).resolve().parents[1]
BACKEND_ROOT = Path(os.environ.get("GAD_TEST_BACKEND_ROOT", GAD_DIR / ".backends"))
PYTHON = sys.executable

# The child process is intentionally self-contained: it computes the reference
# values independently of the library implementation and compares them.
CHILD = r'''
import json, math, sys
import numpy as np
import torch

backend = sys.argv[1]
stage = sys.argv[2]
results = {}

from verl.trainer.ppo import core_algos
results["core_algos_file"] = core_algos.__file__

# --- discriminator loss -----------------------------------------------------
torch.manual_seed(0)
student = torch.randn(4, 6, dtype=torch.float32)
teacher = torch.randn(4, 6, dtype=torch.float32)
s_mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 4, dtype=torch.float32)
t_mask = torch.tensor([[1, 1, 1, 1, 1, 0]] * 4, dtype=torch.float32)
student.requires_grad_(True)
teacher.requires_grad_(True)
loss = core_algos.compute_discriminator_loss(student, teacher, s_mask, t_mask)
reference = -torch.nn.functional.logsigmoid(
    (teacher * t_mask).sum(-1) - (student * s_mask).sum(-1)
).mean()
results["discriminator_loss"] = float(loss)
results["discriminator_reference"] = float(reference)
results["discriminator_matches"] = bool(torch.allclose(loss, reference, atol=0, rtol=0))
loss.backward()
results["student_grad_finite"] = bool(torch.isfinite(student.grad).all())
results["teacher_grad_finite"] = bool(torch.isfinite(teacher.grad).all())
# The gradient must treat masked positions as absent.
masked_grad_zero = bool((student.grad[0, 4:] == 0).all() and (teacher.grad[0, 5] == 0).all())
results["masked_grad_zero"] = masked_grad_zero

# A perfect discriminator (teacher far above student) must give loss -> 0 and a
# swapped pair must give a large loss: this is the ranking semantics of GAD.
try:
    big = core_algos.compute_discriminator_loss(
        torch.full((1, 2), -20.0), torch.full((1, 2), 20.0),
        torch.ones(1, 2), torch.ones(1, 2),
    )
    swapped = core_algos.compute_discriminator_loss(
        torch.full((1, 2), 20.0), torch.full((1, 2), -20.0),
        torch.ones(1, 2), torch.ones(1, 2),
    )
    results["ranking_ok"] = bool(float(big) < 1e-6 and float(swapped) > 20.0)
except AttributeError:
    results["ranking_ok"] = None

# --- SFT loss (seqkd / warmup only) ----------------------------------------
if hasattr(core_algos, "compute_sft_loss"):
    log_prob = -torch.rand(3, 5) - 0.1
    mask = torch.tensor([[1, 1, 1, 0, 0]] * 3, dtype=torch.float32)
    sft = core_algos.compute_sft_loss(log_prob=log_prob, response_mask=mask, loss_agg_mode="token-mean")
    expected = (-log_prob * mask).sum() / mask.sum()
    results["sft_loss"] = float(sft)
    results["sft_expected"] = float(expected)
    results["sft_matches"] = bool(torch.allclose(sft, expected, atol=1e-6, rtol=0))
    # token-sum mode must scale by the sequence length instead of the token count
    token_sum = core_algos.compute_sft_loss(
        log_prob=log_prob, response_mask=mask, loss_agg_mode="seq-mean-token-sum"
    )
    results["sft_seq_mean_token_sum"] = float(token_sum)
else:
    results["sft_matches"] = None

# --- GRPO advantage grouping ------------------------------------------------
rewards = torch.tensor([[1.0, 0, 0], [3.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]], dtype=torch.float32)
mask = torch.tensor([[1, 1, 1]] * 4, dtype=torch.float32)
index = np.array(["a", "a", "b", "b"], dtype=object)
adv, _ = core_algos.compute_grpo_outcome_advantage(
    token_level_rewards=rewards, response_mask=mask, index=index, norm_adv_by_std_in_grpo=True
)
# The group mean must be zero across *all rows of the same uid*, not within one row.
group_a = adv[0:2][mask[0:2].bool()].mean().item()
group_b = adv[2:4][mask[2:4].bool()].mean().item()
results["grpo_group_means"] = [group_a, group_b]
results["grpo_groups_zero_mean"] = bool(abs(group_a) < 1e-4 and abs(group_b) < 1e-4)
results["grpo_signs"] = [
    float(adv[0][0]), float(adv[1][0]),
]
results["grpo_high_reward_positive"] = bool(adv[1][0] > 0 > adv[0][0])

print("@@RESULT@@" + json.dumps(results))
'''


def run_child(stage: str) -> dict:
    backend = BACKEND_ROOT / stage
    if not (backend / "verl").is_dir():
        raise unittest.SkipTest(f"backend {stage} not prepared; run tools/prepare_backends.py first")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(backend)
    env["GAD_EXPECT_VERL_PATH"] = str(backend)
    result = subprocess.run(
        [PYTHON, "-c", CHILD, str(backend), stage],
        env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise AssertionError(f"{stage}: child failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    for line in result.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    raise AssertionError(f"{stage}: no result marker\n{result.stdout}\n{result.stderr}")


class AlgorithmParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = {}
        for stage in ("seqkd", "warmup", "gad"):
            if (BACKEND_ROOT / stage / "verl").is_dir():
                cls.results[stage] = run_child(stage)

    def test_discriminator_loss_matches_the_frozen_formula(self) -> None:
        for stage, data in self.results.items():
            self.assertTrue(data["discriminator_matches"], f"{stage}: discriminator loss drifted")
            self.assertTrue(data["student_grad_finite"], stage)
            self.assertTrue(data["teacher_grad_finite"], stage)
            self.assertTrue(data["masked_grad_zero"], f"{stage}: padding positions affect the loss")

    def test_discriminator_ranks_teacher_above_student(self) -> None:
        for stage, data in self.results.items():
            if data.get("ranking_ok") is None:
                continue
            self.assertTrue(data["ranking_ok"], f"{stage}: ranking semantics broken")

    def test_sft_loss_is_masked_nll_without_kl(self) -> None:
        checked = 0
        for stage, data in self.results.items():
            if data.get("sft_matches") is None:
                continue
            checked += 1
            self.assertTrue(data["sft_matches"], f"{stage}: SFT loss is not the masked token-mean NLL")
            self.assertGreater(data["sft_seq_mean_token_sum"], data["sft_loss"] - 1e-6, stage)
        self.assertGreater(checked, 0, "no backend exposes compute_sft_loss")

    def test_grpo_advantages_are_group_normalized(self) -> None:
        for stage, data in self.results.items():
            self.assertTrue(data["grpo_groups_zero_mean"], f"{stage}: {data['grpo_group_means']}")
            self.assertTrue(data["grpo_high_reward_positive"], f"{stage}: {data['grpo_signs']}")

    def test_each_backend_imports_its_own_verl(self) -> None:
        for stage, data in self.results.items():
            self.assertIn(str(BACKEND_ROOT / stage), data["core_algos_file"], stage)


if __name__ == "__main__":
    unittest.main(verbosity=2)
