#!/usr/bin/env python3
"""Prepare deterministic original adv initialization, then expand in a derived run."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rubric.core.contracts import file_hash, digest, LAYOUT
from rubric.core.heads import expand_head, validate_head


def prepare(source, target, seed, spec, single=False):
    import torch
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
    source, target = Path(source), Path(target)
    weights = sorted(source.glob("*.safetensors"))
    signature = {"source": str(source.resolve()), "weights": {p.name: file_hash(p) for p in weights},
                 "config": file_hash(source / "config.json"), "seed": seed,
                 "rubric_hash": file_hash(spec), "single": single}
    manifest_path = target / "initialization_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_identity"] != signature:
            raise ValueError("critic initialization inputs changed")
        for name, expected in manifest["files"].items():
            if file_hash(target / name) != expected:
                raise ValueError("prepared critic artifact changed")
        return str(target)
    if target.exists() and any(target.iterdir()):
        raise ValueError("incomplete derived critic initialization")
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    if config.model_type not in ("qwen2", "qwen3"):
        raise ValueError("only verified Qwen2/Qwen3 token classifiers are supported")
    classifier = any("ForTokenClassification" in a for a in config.architectures or [])
    four = bool(getattr(config, "rubric_head_layout", None))
    if four:
        validate_head(config, True, file_hash(spec))
        if single:
            raise ValueError("implicit four-head to one-head export is unsupported")
    else:
        config.num_labels = 1
        config.id2label = {0: "adv"}
        config.label2id = {"adv": 0}
    config.classifier_dropout = 0.
    config.hidden_dropout = "0"
    config.summary_dropout_prob = 0.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, loading = AutoModelForTokenClassification.from_pretrained(
            source, config=config, torch_dtype=torch.float32, local_files_only=True,
            attn_implementation="eager", output_loading_info=True)
    missing = set(loading["missing_keys"])
    expected_missing = {"score.weight", "score.bias"} if not classifier else set()
    if missing - expected_missing or (classifier and missing) or loading["mismatched_keys"] or loading["error_msgs"]:
        raise ValueError(f"backbone/trained critic must load strictly: {loading}")
    if set(loading["unexpected_keys"]) - {"lm_head.weight", "lm_head.bias"}:
        raise ValueError(f"unexpected source weights: {loading['unexpected_keys']}")
    before_weight = model.score.weight.detach().clone()
    before_bias = model.score.bias.detach().clone() if model.score.bias is not None else None
    fixture = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        before_logits = model(fixture).logits.detach()
    if not four and not single:
        expand_head(model, seed, file_hash(spec))
    with torch.no_grad():
        after_logits = model(fixture).logits.detach()
    if not torch.equal(before_weight[0], model.score.weight[0]) or (before_bias is not None and not torch.equal(before_bias[0], model.score.bias[0])):
        raise AssertionError("adv row changed during migration")
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(target)
    manifest = {"source_identity": signature, "initialization": "strict_trained_classifier" if classifier else "causal_backbone_original_single_score_initialization",
                "loading_info": loading, "head_layout": list(LAYOUT) if not single else ["adv"],
                "adv_weight_exact": True, "adv_logits_max_error": (before_logits[:, :, 0] - after_logits[:, :, 0]).abs().max().item(),
                "fixture_logits_before": before_logits.tolist(), "fixture_logits_after": after_logits.tolist(),
                "files": {p.name: file_hash(p) for p in target.iterdir() if p.is_file()}}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return str(target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--spec", default=str(ROOT / "rubric/specs/math_v1.yaml"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single-head", action="store_true", help="Matching deterministic adv fixture for baseline experiments")
    args = p.parse_args()
    print(prepare(args.source, args.target, args.seed, args.spec, args.single_head))
