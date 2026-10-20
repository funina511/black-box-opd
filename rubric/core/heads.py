def expand_head(model, seed, rubric_hash, version='math_v1'):
    import torch
    from torch import nn
    from .contracts import layout_for
    LAYOUT = layout_for(version)
    if model.config.model_type not in ("qwen2", "qwen3") or model.score.out_features != 1:
        raise ValueError("expansion requires an intact supported single-head classifier")
    old = model.score
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        new = nn.Linear(old.in_features, len(LAYOUT), bias=old.bias is not None)
        nn.init.normal_(new.weight, mean=0., std=model.config.initializer_range)
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
    model.config.rubric_version = version
    return model


def last_response_logits(logits, response_mask):
    import torch
    if logits.shape[:2] != response_mask.shape or (response_mask.sum(-1) == 0).any():
        raise ValueError("invalid or empty response mask")
    positions = torch.arange(response_mask.shape[1], device=logits.device)
    index = torch.where(response_mask.bool(), positions, -1).max(-1).values
    return logits[torch.arange(logits.shape[0], device=logits.device), index]


def validate_head(config, enabled, rubric_hash=None, version=None):
    from .contracts import layout_for
    version = version or getattr(config, 'rubric_version', 'math_v1')
    LAYOUT = layout_for(version)
    labels = getattr(config, "num_labels", len(getattr(config, "id2label", {0: "adv"})))
    if enabled:
        if labels != len(LAYOUT) or tuple(getattr(config, "rubric_head_layout", ())) != LAYOUT:
            raise ValueError(f"enabled rubric requires prepared {len(LAYOUT)}-channel critic")
        if config.rubric_hash != rubric_hash:
            raise ValueError("critic rubric hash mismatch")
        if getattr(config, 'rubric_version', None) != version:
            raise ValueError('critic rubric semantic version mismatch')
    elif labels != 1 and getattr(config, "rubric_head_layout", None):
        raise ValueError("four-channel critic cannot be resumed with rubric disabled")
