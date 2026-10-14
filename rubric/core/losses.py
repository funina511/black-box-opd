import torch
from torch.nn import functional as F


def pair_loss(teacher, student):
    return -F.logsigmoid(teacher - student)


def auxiliary_loss(logits, targets, masks):
    counts = masks.sum(0)
    active = counts > 0
    if not active.any():
        return logits.sum() * 0
    sums = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * masks).sum(0)
    return (sums[active] / counts[active]).mean()


def weighted_contribution(logits, targets, masks, weights, counts, dp_size=1):
    """Contribution to one global optimizer minibatch; DP gradients are averaged."""
    active = counts > 0
    terms = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    numerator = (terms * masks * weights[:, None]).sum(0)
    loss = (numerator * active / counts.clamp_min(1)).sum() / active.sum().clamp_min(1)
    return loss * dp_size, numerator
