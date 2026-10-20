#!/usr/bin/env python3
"""Micro-batch partitioning: budget cap enforcement and gradient equivalence.

Two review findings are pinned down here (CPU, torch only):

1. ``ceildiv(total, budget)`` only bounds the *mean* load, so the pinned
   ``rearrange_micro_batches`` could hand a trainer a micro-batch far above the token
   budget (measured: 36864 combined tokens against 12288, because only the student mask
   was counted). The patch must never return an over-budget partition unless a *single*
   sample does not fit, and it must say so.
2. A different partition is **not** mathematically neutral, so the patch must not claim
   it is. The trainer's dynamic-bsz accumulation
   (``dp_actor.py``: ``loss = policy_loss * (len(data) / ppo_mini_batch_size)``) gives
   every sample the weight ``w_i = (|g| / B) * (n_i / sum_{j in g} n_j)``. Equal-length
   micro-batches are exactly invariant; unequal lengths are not. The pinned losses and
   the pinned scaling rule are used verbatim so the numbers are the real ones.

Run::

    python LMOps/gad/tests/test_microbatch_equivalence.py
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
WARMUP = BACKEND_ROOT / "warmup"
PINNED_SEQLEN = GAD_DIR / "verl" / "verl" / "utils" / "seqlen_balancing.py"
PYTHON = sys.executable
SENTINEL = "@@RESULT@@"


# --------------------------------------------------------------------------- #
# child 1: partitioning / cap
# --------------------------------------------------------------------------- #
PARTITION_CHILD = r'''
import importlib.util, json, os, sys
from pathlib import Path
import torch
from tensordict import TensorDict

backend, pinned_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, backend)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


patched = load("patched_seqlen", str(Path(backend) / "verl/utils/seqlen_balancing.py"))
pinned = load("pinned_seqlen", pinned_path)
WIDTH, BUDGET = 10240, 12288
out = {}


def make_batch(student, teacher=None):
    d = {"attention_mask": torch.zeros(len(student), WIDTH, dtype=torch.long)}
    for i, n in enumerate(student):
        d["attention_mask"][i, :n] = 1
    if teacher is not None:
        d["teacher_attention_mask"] = torch.zeros(len(teacher), WIDTH, dtype=torch.long)
        for i, n in enumerate(teacher):
            d["teacher_attention_mask"][i, :n] = 1
    return TensorDict(d, batch_size=[len(student)])


def combined(batch, partitions):
    s = batch["attention_mask"].sum(1).tolist()
    t = (batch["teacher_attention_mask"].sum(1).tolist()
         if "teacher_attention_mask" in batch.keys() else [0] * len(s))
    return [sum(s[i] + t[i] for i in p) for p in partitions]


def student_only(batch, partitions):
    s = batch["attention_mask"].sum(1).tolist()
    return [sum(s[i] for i in p) for p in partitions]


# 1. the reviewer's symptom: pinned sizes by the student mask, the teacher side overflows
batch = make_batch([1024] * 4, [8192] * 4)
_, pinned_idx = pinned.rearrange_micro_batches(batch=batch, max_token_len=BUDGET, compute_teacher=False)
_, patched_idx = patched.rearrange_micro_batches(batch=batch, max_token_len=BUDGET, compute_teacher=False)
out["pinned_combined_loads"] = combined(batch, pinned_idx)
out["patched_combined_loads"] = combined(batch, patched_idx)
out["patched_partitions"] = patched_idx
out["budget"] = BUDGET

# 2. the pinned balancer can exceed the budget on the student side alone
batch2 = make_batch([8192] * 4, [2048] * 4)
_, pinned_idx2 = pinned.rearrange_micro_batches(batch=batch2, max_token_len=BUDGET, compute_teacher=False)
_, patched_idx2 = patched.rearrange_micro_batches(batch=batch2, max_token_len=BUDGET, compute_teacher=False)
out["pinned_student_loads"] = student_only(batch2, pinned_idx2)
out["patched_student_loads"] = student_only(batch2, patched_idx2)
out["patched_loads2"] = combined(batch2, patched_idx2)

# 3. one sample above the budget: singleton (nothing can split it) and reported
batch3 = make_batch([10000, 100, 100], [9216, 100, 100])
_, patched_idx3 = patched.rearrange_micro_batches(batch=batch3, max_token_len=BUDGET, compute_teacher=False)
out["singleton_partitions"] = patched_idx3
out["singleton_loads"] = combined(batch3, patched_idx3)

# 4. GAD_MICRO_BATCH_BUDGET=upstream must be bit-identical to the pinned code
os.environ["GAD_MICRO_BATCH_BUDGET"] = "upstream"
_, upstream_idx = patched.rearrange_micro_batches(batch=batch2, max_token_len=BUDGET, compute_teacher=False)
out["upstream_mode_partitions"] = upstream_idx
out["upstream_mode_matches_pinned"] = upstream_idx == pinned_idx2
os.environ.pop("GAD_MICRO_BATCH_BUDGET")

# 5. the teacher-only path (SeqKD) must be untouched when it already fits
batch5 = make_batch([512] * 8, [3000] * 8)
_, patched_idx5 = patched.rearrange_micro_batches(batch=batch5, max_token_len=BUDGET, compute_teacher=True)
_, pinned_idx5 = pinned.rearrange_micro_batches(batch=batch5, max_token_len=BUDGET, compute_teacher=True)
out["teacher_only_matches_pinned"] = patched_idx5 == pinned_idx5

# 6. never more micro-batches than samples, and never fewer than the pinned estimate
out["count_within_bounds"] = all(
    len(idx) <= 4 for idx in (patched_idx, patched_idx2, patched_idx3)
)

# 7. the pre-existing guard is unchanged: a sequence longer than the budget still asserts
try:
    patched.rearrange_micro_batches(batch=make_batch([2048], [2048]), max_token_len=1024)
    out["asserts_on_long_sequence"] = False
except AssertionError:
    out["asserts_on_long_sequence"] = True

print("@@RESULT@@" + json.dumps(out))
'''


# --------------------------------------------------------------------------- #
# child 2: accumulation rule (real pinned loss + real pinned scaling)
# --------------------------------------------------------------------------- #
GRADIENT_CHILD = r'''
import json, sys
from pathlib import Path

backend = sys.argv[1]
sys.path.insert(0, backend)

import torch
from verl.trainer.ppo.core_algos import compute_sft_loss

out = {}

def grouped_objective(token_losses, groups, batch_size):
    """Exactly dp_actor.update_policy's dynamic-bsz accumulation.

    ``token_losses[i]`` is sample i's per-token loss (constant over its tokens); the
    pinned loss aggregates each micro-batch with ``loss_agg_mode="token-mean"`` and the
    trainer scales that by ``len(micro_batch) / ppo_mini_batch_size``.
    """
    total = None
    for group in groups:
        stacked = torch.stack([token_losses[i] for i in group])
        mask = torch.ones_like(stacked)
        micro = compute_sft_loss(log_prob=-stacked, response_mask=mask, loss_agg_mode="token-mean")
        term = micro * (len(group) / batch_size)
        total = term if total is None else total + term
    return total


def weights(token_losses, groups, batch_size):
    """d(objective)/d(per-sample per-token loss) == the sample's effective weight."""
    grads = []
    for i in range(len(token_losses)):
        leaf = token_losses[i].detach().clone().requires_grad_(True)
        perturbed = [leaf if j == i else token_losses[j].detach() for j in range(len(token_losses))]
        objective = grouped_objective(perturbed, groups, batch_size)
        (grad,) = torch.autograd.grad(objective, leaf)
        grads.append(grad.item())
    return grads


BATCH = 2
# --- equal token counts: partitioning must be exactly neutral -----------------
equal = [torch.tensor(1.0), torch.tensor(2.0)]
batched = grouped_objective(equal, [[0, 1]], BATCH)
split = grouped_objective(equal, [[0], [1]], BATCH)
out["equal_objective_batched"] = batched.item()
out["equal_objective_split"] = split.item()
out["equal_weights_batched"] = weights(equal, [[0, 1]], BATCH)
out["equal_weights_split"] = weights(equal, [[0], [1]], BATCH)

# --- unequal token counts: the reviewer's 2.5 vs 2.0 --------------------------
# sample 0: 1 token at 1.0 ; sample 1: 3 tokens at 3.0
loss_mat = torch.tensor([[1.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
mask_batched = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
batched_loss = compute_sft_loss(log_prob=-loss_mat, response_mask=mask_batched,
                                loss_agg_mode="token-mean") * (2 / BATCH)
split_loss = (
    compute_sft_loss(log_prob=-loss_mat[0:1, 0:1], response_mask=torch.ones(1, 1),
                     loss_agg_mode="token-mean") * (1 / BATCH)
    + compute_sft_loss(log_prob=-loss_mat[1:2, :], response_mask=torch.ones(1, 3),
                       loss_agg_mode="token-mean") * (1 / BATCH)
)
out["unequal_objective_batched"] = batched_loss.item()
out["unequal_objective_split"] = split_loss.item()
out["unequal_gradient_differs"] = not torch.allclose(batched_loss, split_loss)

# gradient-level: autograd on the per-token losses, with the real token counts
token_counts = {0: 1, 1: 3}
losses = [torch.tensor(1.0), torch.tensor(3.0)]


def objective_with_counts(groups):
    total = None
    for group in groups:
        rows, masks = [], []
        for i in group:
            count = token_counts[i]
            rows.append(losses[i].reshape(1, 1).repeat(1, count))
            masks.append(torch.ones(1, count))
        micro = compute_sft_loss(log_prob=-torch.cat(rows, 1), response_mask=torch.cat(masks, 1),
                                 loss_agg_mode="token-mean")
        term = micro * (len(group) / BATCH)
        total = term if total is None else total + term
    return total


grads_batched, grads_split = [], []
for i in range(2):
    leaf = losses[i].detach().clone().requires_grad_(True)
    saved = losses[i]
    losses[i] = leaf
    (g,) = torch.autograd.grad(objective_with_counts([[0, 1]]), leaf)
    grads_batched.append(g.item())
    (g,) = torch.autograd.grad(objective_with_counts([[0], [1]]), leaf)
    grads_split.append(g.item())
    losses[i] = saved
out["weights_batched"] = grads_batched
out["weights_split"] = grads_split
out["weight_sum"] = sum(grads_batched)

# the closed form w_i = (|g| / B) * (n_i / sum_{j in g} n_j)
out["formula_batched"] = [(2 / BATCH) * token_counts[i] / sum(token_counts.values()) for i in range(2)]
out["formula_split"] = [1 / BATCH, 1 / BATCH]

print("@@RESULT@@" + json.dumps(out))
'''


def run_child(script: str, *args: str, backend: Path = WARMUP) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(backend)
    env.pop("GAD_MICRO_BATCH_BUDGET", None)
    env["TOKENIZERS_PARALLELISM"] = "false"
    result = subprocess.run(
        [PYTHON, "-c", script, str(backend), *args],
        capture_output=True, text=True, env=env, cwd=str(GAD_DIR),
    )
    if result.returncode != 0:
        raise AssertionError(f"child failed:\n{result.stdout[-2000:]}\n{result.stderr[-3000:]}")
    lines = [line for line in result.stdout.splitlines() if line.startswith(SENTINEL)]
    if not lines:
        raise AssertionError(f"child produced no result:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return json.loads(lines[-1][len(SENTINEL):])


@unittest.skipUnless(WARMUP.is_dir() and PINNED_SEQLEN.exists(), "backends not prepared")
class MicroBatchCapTests(unittest.TestCase):
    """The patch must never return an over-budget partition (except a lone sample)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.out = run_child(PARTITION_CHILD, str(PINNED_SEQLEN))

    def test_pinned_partition_really_overflows_the_budget(self) -> None:
        # the finding is real: 4 x (1024 student + 8192 teacher) lands in one micro-batch
        self.assertGreater(max(self.out["pinned_combined_loads"]), self.out["budget"])

    def test_patched_partition_respects_the_budget(self) -> None:
        self.assertLessEqual(max(self.out["patched_combined_loads"]), self.out["budget"])

    def test_pinned_balancer_can_exceed_the_budget_on_one_mask(self) -> None:
        self.assertGreater(max(self.out["pinned_student_loads"]), self.out["budget"])
        self.assertLessEqual(max(self.out["patched_student_loads"]), self.out["budget"])
        self.assertLessEqual(max(self.out["patched_loads2"]), self.out["budget"])

    def test_a_single_oversized_sample_stays_a_singleton(self) -> None:
        loads = self.out["singleton_loads"]
        partitions = self.out["singleton_partitions"]
        over = [i for i, load in enumerate(loads) if load > self.out["budget"]]
        self.assertEqual(len(over), 1, loads)
        self.assertEqual(len(partitions[over[0]]), 1, partitions)

    def test_upstream_mode_is_bit_identical_to_the_pinned_code(self) -> None:
        self.assertTrue(self.out["upstream_mode_matches_pinned"], self.out["upstream_mode_partitions"])

    def test_teacher_only_path_is_unchanged_when_it_fits(self) -> None:
        self.assertTrue(self.out["teacher_only_matches_pinned"])

    def test_count_stays_within_bounds_and_guard_is_intact(self) -> None:
        self.assertTrue(self.out["count_within_bounds"])
        self.assertTrue(self.out["asserts_on_long_sequence"])


@unittest.skipUnless(WARMUP.is_dir(), "backends not prepared")
class GradientAggregationTests(unittest.TestCase):
    """Partitioning is NOT gradient-neutral; this pins the exact difference."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.out = run_child(GRADIENT_CHILD)

    def test_equal_token_counts_are_exactly_invariant(self) -> None:
        self.assertAlmostEqual(self.out["equal_objective_batched"], self.out["equal_objective_split"], places=6)
        for batched, split in zip(self.out["equal_weights_batched"], self.out["equal_weights_split"]):
            self.assertAlmostEqual(batched, split, places=6)

    def test_unequal_token_counts_change_the_objective(self) -> None:
        # the reviewer's numbers, reproduced with the pinned SFT loss and scaling
        self.assertAlmostEqual(self.out["unequal_objective_batched"], 2.5, places=6)
        self.assertAlmostEqual(self.out["unequal_objective_split"], 2.0, places=6)
        self.assertTrue(self.out["unequal_gradient_differs"])

    def test_per_sample_weights_follow_the_closed_form(self) -> None:
        for got, expected in (
            (self.out["weights_batched"], self.out["formula_batched"]),
            (self.out["weights_split"], self.out["formula_split"]),
        ):
            for a, b in zip(got, expected):
                self.assertAlmostEqual(a, b, places=6)
        # weights always sum to N / B, so the difference is a re-distribution
        self.assertAlmostEqual(self.out["weight_sum"], 1.0, places=6)


class PatchDocumentationTests(unittest.TestCase):
    """The patch must not advertise a mathematical equivalence it does not have."""

    def test_no_equivalence_claim_in_the_patch_source(self) -> None:
        text = (GAD_DIR / "tools" / "backend_patches.py").read_text(encoding="utf-8")
        for claim in ("mathematically identical", "mathematically equivalent", "is mathematically"):
            self.assertNotIn(claim, text)
        self.assertIn("GAD_MICRO_BATCH_BUDGET", text)

    def test_pipeline_documents_the_deviation(self) -> None:
        text = (GAD_DIR / "PIPELINE.md").read_text(encoding="utf-8")
        self.assertIn("GAD_MICRO_BATCH_BUDGET", text)
        self.assertIn("2.5", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
