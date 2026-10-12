"""Strict single-head export and four-channel Qwen token classifier."""
from __future__ import annotations
import json
import re
from pathlib import Path

from rubric.common import LAYOUT, digest, file_hash, write_json


def inspect_head(path):
    import torch
    path = Path(path)
    shard = next(iter(path.glob("model_world_size_*_rank_0.pt")), None)
    if shard:
        state = torch.load(shard, map_location="cpu", weights_only=False)
        shapes = {key: list(value.shape) for key, value in state.items() if key.startswith("score.")}
    else:
        from safetensors import safe_open
        shapes = {}
        for file in sorted(path.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith("score."):
                        shapes[key] = list(handle.get_slice(key).get_shape())
    config = json.loads((path / "config.json").read_text())
    if shapes.get("score.weight") != [1, config.get("hidden_size")]:
        raise ValueError(f"source critic needs intact [1, hidden] score.weight: {shapes}")
    if shapes.get("score.bias") not in (None, [1]):
        raise ValueError("source critic score.bias shape mismatch")
    if config["model_type"] not in ("qwen3", "qwen2"):
        raise ValueError("only Qwen2/Qwen3 TokenClassification is supported")
    return shapes


def export_single_head(source, target):
    """Read shards on CPU and write into the DERIVED run; never modify source."""
    import torch
    from torch.distributed.tensor import DTensor
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
    source, target = Path(source), Path(target)
    shapes = inspect_head(source)
    source_files = sorted(source.glob("model_world_size_*_rank_*.pt")) or sorted(source.glob("*.safetensors"))
    signature = digest({p.name: file_hash(p) for p in source_files})
    manifest = target / "export_manifest.json"
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["source_signature"] != signature:
            raise ValueError("export cache source changed")
        for name, expected in info["export_files"].items():
            if file_hash(target / name) != expected:
                raise ValueError("derived single-head export changed")
        inspect_head(target)
        return target
    if target.exists() and any(target.iterdir()):
        raise ValueError("incomplete export directory; inspect it before retrying")
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    config.num_labels = 1
    config.id2label = {0: "adv"}
    config.label2id = {"adv": 0}
    if source_files[0].suffix == ".safetensors":
        from safetensors.torch import load_file
        state = {}
        for file in source_files:
            state.update(load_file(str(file)))
    else:
        ranks = sorted(source_files, key=lambda p: int(re.search(r"rank_(\d+)", p.name).group(1)))
        world = int(re.search(r"world_size_(\d+)", ranks[0].name).group(1))
        if len(ranks) != world:
            raise ValueError("missing model shards")
        states = [torch.load(file, map_location="cpu", weights_only=False) for file in ranks]
        state = {}
        for key in list(states[0]):
            values = [s.pop(key) for s in states]
            first = values[0]
            if isinstance(first, DTensor):
                if len(first.placements) != 1:
                    raise ValueError("only one-dimensional FSDP shard export is supported")
                placement = first.placements[0]
                local = [v.to_local() for v in values]
                if placement.is_shard():
                    merged = torch.cat(local, dim=placement.dim)
                    slices = tuple(slice(0, size) for size in first.shape)
                    state[key] = merged[slices].contiguous()
                elif placement.is_replicate():
                    if not all(torch.equal(local[0], v) for v in local[1:]):
                        raise ValueError(f"replicated weights differ: {key}")
                    state[key] = local[0]
                else:
                    raise ValueError("partial DTensor export is not supported")
            else:
                if not all(torch.equal(first, v) for v in values[1:]):
                    raise ValueError(f"replicated tensor differs: {key}")
                state[key] = first
        del states
    with torch.device("meta"):
        model = AutoModelForTokenClassification.from_config(config, torch_dtype=torch.bfloat16)
    model.load_state_dict(state, strict=True, assign=True)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(target)
    write_json(manifest, {"source": str(source.resolve()), "source_signature": signature,
                          "head_shapes": shapes, "num_labels": 1,
                          "export_files": {p.name: file_hash(p) for p in target.iterdir() if p.is_file()}})
    return target


def expand_head(model, seed, rubric_hash):
    import torch
    from torch import nn
    if model.config.model_type not in ("qwen2", "qwen3") or model.score.out_features != 1:
        raise ValueError("expansion requires an intact supported single-head classifier")
    old = model.score
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        new = nn.Linear(old.in_features, len(LAYOUT), bias=old.bias is not None)
        nn.init.normal_(new.weight, mean=0.0, std=model.config.initializer_range)
        if new.bias is not None:
            nn.init.zeros_(new.bias)
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight[0].copy_(old.weight[0])
        if old.bias is not None:
            new.bias[0].copy_(old.bias[0])
    model.score = new
    model.num_labels = len(LAYOUT)
    model.config.num_labels = len(LAYOUT)
    model.config.id2label = dict(enumerate(LAYOUT))
    model.config.label2id = {name: index for index, name in enumerate(LAYOUT)}
    model.config.rubric_head_layout = list(LAYOUT)
    model.config.rubric_hash = rubric_hash
    model.config.rubric_version = "math_v1"
    return model


def load_classifier(path, dtype=None, rubric_hash=None):
    from transformers import AutoModelForTokenClassification
    model, info = AutoModelForTokenClassification.from_pretrained(
        path, local_files_only=True, torch_dtype=dtype, output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"classifier load was not strict: {info}")
    if rubric_hash is not None:
        if model.config.num_labels != 4 or tuple(getattr(model.config, "rubric_head_layout", ())) != LAYOUT:
            raise ValueError("saved rubric head layout mismatch")
        if getattr(model.config, "rubric_hash", None) != rubric_hash:
            raise ValueError("saved rubric version/hash mismatch")
    elif model.config.num_labels != 1:
        raise ValueError("expected source single-head critic")
    return model


def last_response_logits(logits, response_mask):
    import torch
    if logits.shape[:2] != response_mask.shape:
        raise ValueError("response mask shape mismatch")
    if (response_mask.sum(-1) == 0).any():
        raise ValueError("empty response cannot be scored")
    positions = torch.arange(response_mask.shape[1], device=logits.device)
    index = torch.where(response_mask.bool(), positions, -1).max(-1).values
    return logits[torch.arange(logits.shape[0], device=logits.device), index]


def encode_response(tokenizer, row, generation):
    thinking = {"enable_thinking": generation["enable_thinking"]} if generation["thinking_control"] else {}
    prompt = tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True, **thinking)
    raw_ids = row.get("generated_token_ids")
    if raw_ids is not None:
        if not raw_ids or any(type(token) is not int or token < 0 for token in raw_ids):
            raise ValueError("invalid generated token IDs")
        decoded = tokenizer.decode(raw_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if decoded != row["response"]:
            raise ValueError("source generated token IDs do not decode to the complete stored response")
        response = list(raw_ids)
    else:
        response = tokenizer.encode(row["response"], add_special_tokens=False)
    original = len(response)
    if not response:
        raise ValueError("empty tokenized response")
    if tokenizer.eos_token_id is not None and response[-1] != tokenizer.eos_token_id and not row["truncated"]:
        response.append(tokenizer.eos_token_id)
    if len(prompt) > generation["max_prompt_length"] or len(response) > generation["max_response_length"]:
        raise ValueError("full response/prompt exceeds source budget; no silent truncation")
    return {"input_ids": prompt + response, "attention_mask": [1] * (len(prompt) + len(response)),
            "response_mask": [0] * len(prompt) + [1] * len(response),
            "original_response_tokens": original, "visible_response_tokens": len(response)}


def score_response(model, encoded, device):
    import torch
    inputs = {name: torch.tensor([encoded[name]], dtype=torch.long, device=device)
              for name in ("input_ids", "attention_mask")}
    inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    mask = torch.tensor([encoded["response_mask"]], device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=torch.device(device).type == "cuda"):
        output = model(**inputs, use_cache=False).logits
    return last_response_logits(output, mask)[0].float()
