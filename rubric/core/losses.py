"""BT and globally normalized, deduplicated auxiliary BCE objectives."""
import torch
from torch.nn import functional as F

LOSS_MODES = ('legacy_active', 'fixed_dimension', 'fixed_dimension_gated')


def pair_loss(teacher, student):
    return -F.logsigmoid(teacher - student)


def dimension_weights(counts, mode='legacy_active', tau=4):
    """Per-label coefficient before/after gating for a GLOBAL logical minibatch.

    Counts already include inverse multiplicities across both roles, DP ranks
    and gradient accumulation. Never recompute them from a local microbatch.
    """
    if mode not in LOSS_MODES:raise ValueError('unknown auxiliary loss mode: ' + str(mode))
    if type(tau) not in (int, float) or not 0 < tau < float('inf'):raise ValueError('gate tau must be finite and positive')
    if counts.shape != (3,) or not torch.isfinite(counts).all() or (counts < 0).any():raise ValueError('expected three finite nonnegative global counts')
    active = (counts > 0).to(counts.dtype)
    denominator = active.sum().clamp_min(1) if mode == 'legacy_active' else 3
    before = active / counts.clamp_min(1) / denominator
    gate = (counts / tau).clamp(max=1) if mode == 'fixed_dimension_gated' else active
    return before, gate, before * gate


def auxiliary_loss(logits, targets, masks, mode='legacy_active', tau=4):
    counts = masks.sum(0)
    return weighted_contribution(logits, targets, masks, torch.ones_like(masks[:, 0]), counts,
                                 mode=mode, tau=tau)[0]


def weighted_contribution(logits, targets, masks, weights, counts, dp_size=1,
                          mode='legacy_active', tau=4):
    """Contribution to one global optimizer minibatch; DP gradients are averaged."""
    _, _, coefficient = dimension_weights(counts, mode, tau)
    terms = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
    numerator = (terms * masks * weights[:, None]).sum(0)
    return (numerator * coefficient).sum() * dp_size, numerator
