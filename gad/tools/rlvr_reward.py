"""Binary, response-level math reward for the GRPO control."""

from collections import Counter

import torch


def verified_rewards(batch, tokenizer, scorer):
    """Reward the final visible token of each verified-correct student reply."""
    responses = batch.batch["responses"]
    masks = batch.batch["response_mask"].bool()
    rewards = torch.zeros_like(responses, dtype=torch.float32)
    statuses = Counter()
    for i, reward_model in enumerate(batch.non_tensor_batch["reward_model"]):
        indices = masks[i].nonzero(as_tuple=True)[0]
        if not len(indices):
            raise ValueError("RLVR rollout has no visible response tokens")
        text = tokenizer.decode(responses[i, indices].tolist(), skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        result = scorer.compute_score(text, reward_model["ground_truth"])
        status = result["evaluation_status"]
        statuses[status] += 1
        rewards[i, indices[-1]] = float(status == "correct")
    metrics = {f"rlvr/count_{status}": count for status, count in statuses.items()}
    metrics["rlvr/responses"] = len(responses)
    return rewards, metrics
