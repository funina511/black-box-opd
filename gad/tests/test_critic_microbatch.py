"""Check paired critic loss scaling for the proposed static micro batch size."""
from __future__ import annotations

import sys
import os
import unittest
from pathlib import Path

import torch

BACKEND = Path(os.environ.get("GAD_TEST_BACKEND_ROOT", Path(__file__).resolve().parents[1] / ".backends")) / "warmup"
if BACKEND.is_dir():
    sys.path.insert(0, str(BACKEND))


class CriticMicrobatchTests(unittest.TestCase):
    def test_micro_one_and_two_keep_paired_loss_and_gradient(self):
        if not BACKEND.is_dir():
            self.skipTest("Warmup backend has not been prepared")
        from verl.trainer.ppo.core_algos import compute_discriminator_loss

        torch.manual_seed(42)
        pairs = 16  # Warmup's normalized critic minibatch on four GPUs.
        student_inputs = torch.randn(pairs, 7, 4)
        teacher_inputs = torch.randn(pairs, 9, 4)
        student_lengths = torch.arange(pairs) % 7 + 1
        teacher_lengths = torch.arange(pairs) % 9 + 1
        student_mask = (torch.arange(7)[None, :] < student_lengths[:, None]).float()
        teacher_mask = (torch.arange(9)[None, :] < teacher_lengths[:, None]).float()
        initial_weight = torch.randn(4, 1)

        def run(micro: int):
            weight = initial_weight.clone().requires_grad_()
            loss_total = torch.zeros(())
            for start in range(0, pairs, micro):
                end = start + micro
                student_values = (student_inputs[start:end] @ weight).squeeze(-1)
                teacher_values = (teacher_inputs[start:end] @ weight).squeeze(-1)
                paired_loss = compute_discriminator_loss(
                    student_values, teacher_values,
                    student_mask[start:end], teacher_mask[start:end],
                )
                scaled_loss = paired_loss / (pairs // micro)
                scaled_loss.backward()
                loss_total = loss_total + scaled_loss.detach()
            return loss_total, weight.grad.detach().clone()

        whole_loss, whole_grad = run(16)
        for micro in (1, 2):
            loss, grad = run(micro)
            self.assertTrue(torch.allclose(loss, whole_loss, rtol=1e-6, atol=1e-6))
            self.assertTrue(torch.allclose(grad, whole_grad, rtol=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
