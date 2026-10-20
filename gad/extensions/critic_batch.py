"""Exact-token pair extraction and rebuilding; never mutate the policy batch."""
from collections import Counter
from copy import deepcopy
from .common import digest

FIELDS = ("input_ids", "responses", "attention_mask", "position_ids", "teacher_input_ids",
          "teacher_response", "teacher_attention_mask", "teacher_position_ids")


def extract_pairs(batch, tokenizer, token_identity, step, stage):
    records = []
    width = batch.batch["prompts"].shape[-1]
    for i in range(len(batch)):
        prompt_mask = batch.batch["attention_mask"][i, :width].bool()
        prompt_ids = batch.batch["prompts"][i][prompt_mask].cpu().tolist()
        extra = batch.non_tensor_batch.get("extra_info", [{}] * len(batch))[i]
        index = extra.get("index", i)
        visible_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        prompt_hash = digest({"tokens": prompt_ids, "text": visible_prompt, "template": token_identity})
        prompt_id = digest({"index": str(index), "prompt_hash": prompt_hash})
        ground_truth = str(batch.non_tensor_batch["reward_model"][i]["ground_truth"])
        pair = {"prompt_id": prompt_id, "prompt_hash": prompt_hash, "prompt_ids": prompt_ids,
                "prompt": visible_prompt, "ground_truth": ground_truth, "source_step": step,
                "producer_stage": stage, "actor_policy_step": step - 1,
                "prompt_truncated": bool(batch.non_tensor_batch.get("gad_prompt_truncated", [False] * len(batch))[i]),
                "token_identity": token_identity}
        for role, response_key, attention_key in (
                ("student", "responses", "attention_mask"),
                ("teacher", "teacher_response", "teacher_attention_mask")):
            values = batch.batch[response_key][i]
            mask = batch.batch[attention_key][i, -values.shape[-1]:].bool()
            ids = values[mask].cpu().tolist()
            if not ids:
                raise ValueError("empty response cannot enter an extension critic update")
            text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            rid = digest({"prompt": prompt_hash, "tokens": ids, "text": text, "template": token_identity})
            if role == "student":
                finish = str(batch.non_tensor_batch.get("gad_finish_reason", ["unknown"] * len(batch))[i])
                truncated = finish == "length"
            else:
                finish = "teacher_source"
                truncated = bool(batch.non_tensor_batch.get("gad_teacher_token_truncated", [False] * len(batch))[i])
                # Source index provenance is joined by the driver before annotation.
            pair[role] = {"response_id": rid, "token_ids": ids, "response": text,
                          "finish_reason": finish, "truncated": truncated}
            pair[role]['training_token_truncated'] = bool(batch.non_tensor_batch.get('gad_teacher_token_truncated', [False] * len(batch))[i]) if role == 'teacher' else False
            if role == 'teacher':
                pair[role].update({key: extra.get(key) for key in ('teacher_metadata_version', 'selected_teacher_index',
                    'selected_teacher_text_sha256', 'source_finish_reason', 'source_truncated') if key in extra})
        pair["pair_id"] = digest({"prompt": prompt_id, "teacher": pair["teacher"]["response_id"],
                                  "student": pair["student"]["response_id"], "step": step, "occurrence": i})
        pair["source_index"] = str(index)
        records.append(pair)
    return records


def pack_pairs(pairs, template):
    import numpy as np
    import torch
    from verl import DataProto
    prompt_width = template.batch["prompts"].shape[-1]
    student_width = template.batch["responses"].shape[-1]
    teacher_width = template.batch["teacher_response"].shape[-1]
    pad = int(template.meta_info["gad_pad_token_id"])
    tensors = {}
    for role, response_key, input_key, mask_key, position_key, width in (
            ("student", "responses", "input_ids", "attention_mask", "position_ids", student_width),
            ("teacher", "teacher_response", "teacher_input_ids", "teacher_attention_mask", "teacher_position_ids", teacher_width)):
        inputs, responses, masks = [], [], []
        for pair in pairs:
            prompt, response = pair["prompt_ids"], pair[role]["token_ids"]
            if len(prompt) > prompt_width or len(response) > width:
                raise ValueError("historical token sequence exceeds original backend widths")
            inputs.append([pad] * (prompt_width - len(prompt)) + prompt + response + [pad] * (width - len(response)))
            responses.append(response + [pad] * (width - len(response)))
            masks.append([0] * (prompt_width - len(prompt)) + [1] * len(prompt) + [1] * len(response) + [0] * (width - len(response)))
        tensors[input_key] = torch.tensor(inputs, dtype=template.batch[input_key].dtype)
        tensors[response_key] = torch.tensor(responses, dtype=template.batch[response_key].dtype)
        tensors[mask_key] = torch.tensor(masks, dtype=template.batch[mask_key].dtype)
        # Retain original backend's increasing response positions, including padding.
        prompt_positions = (tensors[mask_key][:, :prompt_width].cumsum(-1) - 1).clamp_min(0)
        delta = torch.arange(1, width + 1)[None, :]
        tensors[position_key] = torch.cat([prompt_positions, prompt_positions[:, -1:] + delta], dim=-1)
    metadata = {"pair_id": np.array([row["pair_id"] for row in pairs], dtype=object),
                "prompt_id": np.array([row["prompt_id"] for row in pairs], dtype=object),
                "source_step": np.array([row["source_step"] for row in pairs]),
                "teacher_response_id": np.array([row["teacher"]["response_id"] for row in pairs], dtype=object),
                "student_response_id": np.array([row["student"]["response_id"] for row in pairs], dtype=object)}
    result = DataProto.from_dict(tensors=tensors, non_tensors=metadata, meta_info=deepcopy(template.meta_info))
    result.meta_info["global_token_num"] = result.batch["attention_mask"].sum(-1).tolist()
    return result


def auxiliary_weights(pairs, targets, masks, global_mini_size, dp_size):
    """Mirror contiguous DP dispatch and each rank's local minibatch slicing."""
    import torch
    size = len(pairs)
    if size % dp_size or global_mini_size % dp_size or size % global_mini_size:
        raise ValueError("extension batches must divide DP and logical minibatch sizes")
    per_rank, local_mini = size // dp_size, global_mini_size // dp_size
    weights = {role: torch.zeros(size) for role in ("teacher", "student")}
    counts = torch.zeros(size, 3)
    for offset in range(0, per_rank, local_mini):
        rows = [rank * per_rank + index for rank in range(dp_size) for index in range(offset, offset + local_mini)]
        multiplicity = Counter(pairs[i][role]["response_id"] for i in rows for role in weights)
        by_response = {}
        for i in rows:
            for role in weights:
                rid = pairs[i][role]["response_id"]
                labels = (targets[role][i].tolist(), masks[role][i].tolist())
                if rid in by_response and by_response[rid] != labels:
                    raise ValueError("same response has inconsistent labels")
                by_response[rid] = labels
                weights[role][i] = 1 / multiplicity[rid]
        global_counts = sum((torch.tensor(value[1]) for value in by_response.values()), torch.zeros(3))
        counts[rows] = global_counts
    return weights, counts
