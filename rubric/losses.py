"""Original pair loss plus unique-response, masked auxiliary supervision."""
import torch
from torch.nn import functional as F
from rubric.common import DIMENSIONS


def pair_loss(teacher, student):
    return -F.logsigmoid(teacher - student)


def auxiliary_loss(logits, targets, masks):
    counts = masks.sum(0)
    active = counts > 0
    if not active.any():
        return logits.sum() * 0
    sums = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * masks).sum(0)
    return (sums[active] / counts[active]).mean()


def auxiliary_contribution(logits, targets, masks, counts):
    """One UNIQUE response, normalized by the entire optimizer step's counts."""
    active = counts > 0
    if not active.any():
        return logits.sum() * 0
    terms = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * masks
    return (terms[active] / counts[active]).sum() / active.sum()


def label_tensors(annotation, mode):
    targets, masks = [], []
    for dimension in DIMENSIONS:
        item = annotation["labels"][dimension]
        enabled = mode == "joint" or (mode == "correctness_aux" and dimension == DIMENSIONS[0])
        masks.append(float(enabled and item["valid"]))
        targets.append(float(item["value"]) if item["valid"] else 0.0)
    return targets, masks
