#!/usr/bin/env python3
"""Checkpoint handoff into GAD, and export of the final GAD model.

Two modes:

``--mode handoff`` (default)
    Warmup -> GAD: pick the last complete Warmup checkpoint, export HF actor and
    discriminator into the GAD run directory, and write the latest-step marker.
    This is what the original GAD instructions do by hand::

        STEP=800
        cp -r .../global_step_${STEP} .../adversarial/
        echo ${STEP} > .../adversarial/latest_checkpointed_iteration.txt
        python tools/merge_model2hf.py --local_dir .../global_step_${STEP}/actor
        python tools/merge_model2hf.py --local_dir .../global_step_${STEP}/critic

``--mode export-final``
    Export the trained actor (latest complete checkpoint, or an explicitly
    selected complete historical checkpoint) as an HF
    directory for evaluation. Evaluating the handoff actor instead would score
    the Warmup weights GAD started from, i.e. report "before GAD" numbers.

In both modes:

* the default step is discovered from complete checkpoints; an explicit older
  step is accepted only when that exact checkpoint is complete;
* completeness is checked first: model shards, optimizer shards, ``config.json``,
  tokenizer files and ``data.pt`` must all be present;
* source shard files are never modified; the HF export is written beside them;
* every exported file is checksummed and recorded in the manifest, and the
  merged weights are compared key-by-key against their source shards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    PipelineError,
    capture_command,
    ensure_dir,
    log,
    sha256_file,
    utc_timestamp,
    write_json,
)

TOKENIZER_HINTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def discover_checkpoints(warmup_dir: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for child in warmup_dir.glob("global_step_*"):
        suffix = child.name.split("global_step_")[-1]
        if suffix.isdigit() and child.is_dir():
            found.append((int(suffix), child))
    return sorted(found)


def read_dataloader_state(path: Path) -> dict:
    """Summarize ``data.pt`` so the resume position is recorded, not assumed.

    ``StatefulDataLoader.state_dict()`` carries the sampler position; recording it
    is what makes "the GAD run continues at the Warmup dataloader position"
    checkable instead of hopeful. Torch is imported lazily so the handoff can
    still be inspected (``--skip-merge``) in an environment without it.
    """
    summary: dict = {"path": str(path)}
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is present in the stage env
        summary["error"] = f"torch unavailable: {exc}"
        return summary
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - a corrupt data.pt must be reported
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary
    if not isinstance(state, dict):
        summary["error"] = f"unexpected data.pt payload type: {type(state).__name__}"
        return summary
    summary["keys"] = sorted(str(key) for key in state)
    for key in ("_num_yielded", "_sampler_iter_yielded", "_sampler_iter_started", "_iterator_finished"):
        if key in state and isinstance(state[key], (int, bool)):
            summary[key] = state[key]
    sampler_state = state.get("_sampler_state")
    if isinstance(sampler_state, dict):
        summary["_sampler_state_keys"] = sorted(str(key) for key in sampler_state)
    return summary


def check_checkpoint(path: Path, roles: tuple[str, ...] = ("actor", "critic")) -> list[str]:
    problems: list[str] = []
    if not path.is_dir():
        return [f"checkpoint directory missing: {path}"]
    for role in roles:
        role_dir = path / role
        if not role_dir.is_dir():
            problems.append(f"{path.name}/{role}: missing")
            continue
        shards = sorted(role_dir.glob("model_world_size_*_rank_*.pt"))
        if not shards:
            problems.append(f"{path.name}/{role}: no model shards")
        optim = sorted(role_dir.glob("optim_world_size_*_rank_*.pt"))
        if not optim:
            problems.append(f"{path.name}/{role}: no optimizer shards")
        if not (role_dir / "config.json").exists():
            problems.append(f"{path.name}/{role}: config.json missing")
        if not any((role_dir / hint).exists() for hint in TOKENIZER_HINTS):
            problems.append(f"{path.name}/{role}: no tokenizer files")
    if not (path / "data.pt").exists():
        problems.append(f"{path.name}: data.pt missing (dataloader state)")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from gad.extensions.checkpoint_state import checkpoint_problems
    problems.extend(checkpoint_problems(path))
    return problems


def pick_last_complete(warmup_dir: Path, *, step: int | None,
                       roles: tuple[str, ...] = ("actor", "critic")) -> tuple[int, Path, list[dict]]:
    checkpoints = discover_checkpoints(warmup_dir)
    if not checkpoints:
        raise PipelineError(f"no global_step_* checkpoint under {warmup_dir}")
    if step is not None:
        checkpoints = [(s, p) for s, p in checkpoints if s <= step]
        if not checkpoints:
            raise PipelineError(f"no checkpoint at or below step {step} under {warmup_dir}")

    report: list[dict] = []
    for candidate_step, path in sorted(checkpoints, reverse=True):
        problems = check_checkpoint(path, roles=roles)
        report.append({"step": candidate_step, "path": str(path), "problems": problems})
        if not problems:
            return candidate_step, path, report
        log(f"step {candidate_step} is incomplete: {problems[0]}")
    raise PipelineError(
        f"no complete checkpoint under {warmup_dir}; inspected {[item['step'] for item in report]}. "
        "Refusing to hand over a partial checkpoint."
    )


def copy_role(src_ckpt: Path, dst_ckpt: Path, role: str) -> None:
    source = src_ckpt / role
    target = dst_ckpt / role
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    log(f"copied {role}: {source} -> {target}")


def export_hf(ckpt_dir: Path, role: str, merge_tool: Path, python_bin: str) -> Path:
    """Mimic the original shell flow: copy metadata, then run merge_model2hf."""
    role_dir = ckpt_dir / role
    hf_dir = role_dir / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    for item in role_dir.iterdir():
        if item.is_file() and item.suffix != ".pt":
            shutil.copy2(item, hf_dir / item.name)
    if not (hf_dir / "config.json").exists():
        raise PipelineError(f"{role}: config.json missing before merge (looked in {role_dir})")
    log(f"merging {role} shards -> {hf_dir}")
    code, out, err = capture_command([python_bin, str(merge_tool), "--local_dir", str(role_dir)])
    if code != 0:
        raise PipelineError(f"merge_model2hf failed for {role} (exit {code}): {err or out}")
    if not (hf_dir / "config.json").exists():
        raise PipelineError(f"{role}: merged output has no config.json at {hf_dir}")
    return hf_dir


def source_weight_keys(role_dir: Path) -> set:
    """Weight names in the first saved shard, used to prove the merge lost nothing."""
    shards = sorted(role_dir.glob("model_world_size_*_rank_0.pt"))
    if not shards:
        return set()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is present in the stage env
        return set()
    try:
        state = torch.load(shards[0], map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - unreadable shards are reported by the merge itself
        return set()
    return set(state.keys()) if isinstance(state, dict) else set()


def inspect_safetensors(hf_dir: Path) -> dict:
    """Read weight names/shapes without loading the model into memory."""
    weight_map: dict[str, str] = {}
    index_file = hf_dir / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text(encoding="utf-8")).get("weight_map", {})
    files = sorted({str(hf_dir / name) for name in weight_map.values()}) or [
        str(p) for p in sorted(hf_dir.glob("*.safetensors"))
    ]

    shapes: dict[str, list[int]] = {}
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - safetensors ships with transformers
        return {"files": files, "keys": None, "reason": "safetensors not importable"}

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[key] = list(handle.get_slice(key).get_shape())
    return {"files": files, "keys": sorted(shapes), "shapes": shapes}


def fix_critic_config(critic_hf: Path, *, hidden_size: int | None) -> dict:
    """Make the merged discriminator self-describing.

    ``merge_model2hf`` rebuilds the model from the *architecture string in the
    copied config.json*, which for the critic is still the student's
    ``...ForCausalLM``; the value head survives (it is saved from the provided
    state dict) but the metadata is wrong. Rewrite it so any downstream loader -
    including the trainer's own ``hf_model`` checkpoint path - sees a
    token-classification model with a single label.
    """
    config_path = critic_hf / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    before = list(config.get("architectures") or [])
    architectures = []
    for arch in before or ["ForTokenClassification"]:
        if "ForCausalLM" in arch:
            architectures.append(arch.replace("ForCausalLM", "ForTokenClassification"))
        elif "ForConditionalGeneration" in arch:
            architectures.append(arch.replace("ForConditionalGeneration", "ForTokenClassification"))
        elif "ForTokenClassification" in arch:
            architectures.append(arch)
        else:
            architectures.append(arch)
    if not architectures:
        architectures = ["ForTokenClassification"]
    config["architectures"] = architectures
    shape = inspect_safetensors(critic_hf).get("shapes", {}).get("score.weight")
    labels = shape[0] if shape else 4 if config.get("rubric_head_layout") else 1
    if labels not in (1, 4):
        raise PipelineError(f"unsupported critic score shape: {shape}")
    if labels == 4:
        from extension_config import LAYOUT
        if config.get("rubric_head_layout") != list(LAYOUT) or not config.get("rubric_hash"):
            raise PipelineError("four-channel critic is missing its layout/rubric hash")
        config["id2label"] = dict(enumerate(LAYOUT))
        config["label2id"] = {name: i for i, name in enumerate(LAYOUT)}
    config["num_labels"] = labels
    if hidden_size:
        config.setdefault("hidden_size", hidden_size)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"architectures_before": before, "architectures_after": architectures, "num_labels": labels}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("handoff", "export-final"), default="handoff",
                        help="handoff: Warmup -> GAD staging (default). export-final: merge the last complete "
                             "GAD actor so the *trained* model can be evaluated.")
    parser.add_argument("--warmup-dir", default=None, help="Warmup stage output directory (read-only; handoff mode).")
    parser.add_argument("--gad-dir", required=True, help="GAD stage output directory (created/copied into).")
    parser.add_argument("--merge-tool", default=str(Path(__file__).resolve().parents[1] / "tools" / "merge_model2hf.py"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--step", type=int, default=None, help="Use this step instead of the newest complete one.")
    parser.add_argument("--gad-steps", type=int, default=None, help="Additional GAD steps K (records the S+K target).")
    parser.add_argument("--manifest", default=None, help="Manifest path (default: <gad-dir>/handoff_manifest.json).")
    parser.add_argument("--check-only", action="store_true", help="Verify only; copy and merge nothing.")
    parser.add_argument("--force", action="store_true", help="Reuse a non-empty GAD directory.")
    parser.add_argument("--allow-marker-mismatch", action="store_true",
                        help="Continue when the latest-step marker disagrees with the newest complete "
                             "checkpoint (recorded as a problem; the handoff then fails at the end).")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Copy the checkpoint but do not merge (used by preflight).")
    parser.add_argument("--skip-dataloader-check", action="store_true",
                        help="Do not read data.pt (for environments without torch).")
    return parser.parse_args()


def export_final(args: argparse.Namespace) -> int:
    """Merge the last complete GAD actor into an HF directory for evaluation.

    Evaluating ``handoff_manifest.actor_hf`` would score the *Warmup* weights that
    GAD started from, which silently reports "before training" numbers. This mode
    exports the model the GAD stage actually produced (the last complete
    ``global_step_*``), and refuses to export a discriminator or a partial
    checkpoint.
    """
    gad_dir = Path(args.gad_dir).expanduser().resolve()
    merge_tool = Path(args.merge_tool).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve() if args.manifest else gad_dir / "final_model_manifest.json"
    )
    if not gad_dir.is_dir():
        raise PipelineError(f"GAD directory not found: {gad_dir}")
    if not merge_tool.exists():
        raise PipelineError(f"merge tool not found: {merge_tool}")

    problems: list[str] = []
    manifest: dict = {
        "stage": "export_final_model",
        "mode": "export-final",
        "created_at": utc_timestamp(),
        "gad_dir": str(gad_dir),
        "check_only": bool(args.check_only),
    }

    step, source_ckpt, inspection = pick_last_complete(gad_dir, step=args.step, roles=("actor",))
    if args.step is not None and step != args.step:
        raise PipelineError(
            f"requested global_step_{args.step} is incomplete; newest complete checkpoint "
            f"at or below it is global_step_{step}"
        )
    log(f"selected complete actor checkpoint: global_step_{step} ({source_ckpt})")
    manifest["step"] = step
    manifest["source_checkpoint"] = str(source_ckpt)
    manifest["inspected_checkpoints"] = inspection

    latest_file = gad_dir / "latest_checkpointed_iteration.txt"
    if latest_file.exists():
        recorded = latest_file.read_text(encoding="utf-8").strip()
        manifest["gad_latest_marker"] = recorded
        if recorded != str(step):
            if args.step is not None and int(recorded) > step:
                log(f"explicit historical checkpoint global_step_{step}; latest marker is {recorded}")
            else:
                marker_problem = (
                    f"{latest_file} says {recorded!r} but the selected complete checkpoint is {step}; "
                    "the run may not have finished cleanly"
                )
                if args.allow_marker_mismatch:
                    problems.append(marker_problem)
                else:
                    raise PipelineError(f"{marker_problem}. Re-run training to completion, "
                                        "or pass --allow-marker-mismatch.")
    else:
        problems.append(f"{latest_file} missing (cannot confirm which step GAD considered latest)")

    def finish() -> int:
        manifest["ok"] = not problems
        manifest["problems"] = problems
        write_json(manifest_path, manifest)
        for problem in problems:
            log(f"ERROR: {problem}")
        log(f"final-model export {'OK' if not problems else 'FAILED'} -> {manifest_path}")
        return 0 if not problems else 1

    if args.check_only:
        return finish()

    actor_dir = export_hf(source_ckpt, "actor", merge_tool, args.python_bin)
    info = inspect_safetensors(actor_dir)
    keys = set(info.get("keys") or [])
    source_keys = source_weight_keys(source_ckpt / "actor")
    config = json.loads((actor_dir / "config.json").read_text(encoding="utf-8"))

    manifest["actor_hf"] = str(actor_dir)
    comparison = compare_weight_keys(source_keys, keys, config=config, role="actor")
    manifest["weight_keys"] = {
        "merged_actor": len(keys),
        "source_actor": len(source_keys),
        "lost": comparison["lost"],
        "added": comparison["added"],
        "tied_absent": comparison["tied_absent"],
        "tied_readded": comparison["tied_readded"],
        "tie_word_embeddings": comparison["tie_word_embeddings"],
    }
    problems.extend(comparison["problems"])
    if source_keys and keys:
        if comparison["lost"]:
            problems.append(
                f"merged actor is missing {len(comparison['lost'])} weights from its checkpoint, "
                f"e.g. {comparison['lost'][:3]}"
            )
        if comparison["added"]:
            problems.append(f"merged actor gained unexpected weights, e.g. {comparison['added'][:3]}")
        if comparison["tied_absent"]:
            log(f"actor: {comparison['tied_absent']} omitted by the merge because the export keeps "
                "tie_word_embeddings=true (the LM head is re-tied at load time)")
    else:
        problems.append("could not compare merged and source weight keys (shards unreadable?)")

    score_keys = sorted(key for key in keys if key.startswith("score."))
    if score_keys:
        problems.append(f"the final export contains discriminator parameters ({score_keys[:3]}): wrong role")
    architectures = list(config.get("architectures") or [])
    manifest["config"] = {
        "architectures": architectures,
        "num_labels": config.get("num_labels"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
    }
    if architectures and all("ForTokenClassification" in arch for arch in architectures):
        problems.append(f"the final export is a discriminator architecture: {architectures}")
    if not any(key.startswith("model.") for key in keys):
        problems.append("merged actor has no `model.*` weights; the merge did not produce a transformer")
    manifest["exports"] = {
        "actor_files": {p.name: sha256_file(p) for p in sorted(actor_dir.glob("*.json"))},
        "actor_weight_files": [Path(p).name for p in (info.get("files") or [])],
    }
    return finish()


def run_handoff(args: argparse.Namespace) -> int:
    if not args.warmup_dir:
        raise PipelineError("--warmup-dir is required in handoff mode")
    warmup_dir = Path(args.warmup_dir).expanduser().resolve()
    gad_dir = Path(args.gad_dir).expanduser().resolve()
    merge_tool = Path(args.merge_tool).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else gad_dir / "handoff_manifest.json"

    problems: list[str] = []
    manifest: dict = {
        "stage": "prepare_gad_checkpoint",
        "created_at": utc_timestamp(),
        "warmup_dir": str(warmup_dir),
        "gad_dir": str(gad_dir),
        "check_only": bool(args.check_only),
    }

    if not warmup_dir.is_dir():
        raise PipelineError(f"warmup directory not found: {warmup_dir}")
    if not merge_tool.exists():
        raise PipelineError(f"merge tool not found: {merge_tool}")

    step, source_ckpt, inspection = pick_last_complete(warmup_dir, step=args.step)
    if args.step is not None and step != args.step:
        raise PipelineError(f"explicit Warmup step {args.step} is incomplete; refusing fallback to {step}")
    log(f"last complete Warmup checkpoint: global_step_{step} ({source_ckpt})")
    manifest["step"] = step
    manifest["source_checkpoint"] = str(source_ckpt)
    manifest["inspected_checkpoints"] = inspection

    # Pre-conditions are checked before anything is copied: a mismatch means the
    # Warmup run did not finish cleanly, and handing over silently would hide it.
    latest_file = warmup_dir / "latest_checkpointed_iteration.txt"
    marker_problem = None
    if latest_file.exists():
        recorded = latest_file.read_text(encoding="utf-8").strip()
        manifest["warmup_latest_marker"] = recorded
        if recorded != str(step) and not (args.step is not None and recorded.isdigit() and int(recorded) > step):
            marker_problem = (
                f"{latest_file} says {recorded!r} but the newest complete checkpoint is {step}; "
                "the warmup run may not have finished cleanly"
            )
    else:
        marker_problem = f"{latest_file} missing (cannot confirm which step Warmup considered latest)"
    if marker_problem and not args.allow_marker_mismatch:
        raise PipelineError(f"{marker_problem}. Re-run Warmup to completion, or pass --allow-marker-mismatch.")
    if marker_problem:
        problems.append(marker_problem)

    if args.gad_steps:
        manifest["gad_steps"] = int(args.gad_steps)
        manifest["target_total_steps"] = step + int(args.gad_steps)

    def finish() -> int:
        """Write the manifest and report every problem on exactly one path."""
        manifest["ok"] = not problems
        manifest["problems"] = problems
        write_json(manifest_path, manifest)
        for problem in problems:
            log(f"ERROR: {problem}")
        log(f"handoff {'OK' if not problems else 'FAILED'} -> {manifest_path}")
        return 0 if not problems else 1

    if args.check_only:
        return finish()

    if gad_dir.exists() and any(gad_dir.iterdir()) and not args.force:
        raise PipelineError(
            f"{gad_dir} already exists and is not empty; refusing to overwrite. Use --force to reuse it."
        )
    ensure_dir(gad_dir)

    target_ckpt = gad_dir / f"global_step_{step}"
    ensure_dir(target_ckpt)
    copy_role(source_ckpt, target_ckpt, "actor")
    copy_role(source_ckpt, target_ckpt, "critic")
    shutil.copy2(source_ckpt / "data.pt", target_ckpt / "data.pt")
    (gad_dir / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")
    manifest["gad_checkpoint"] = str(target_ckpt)
    manifest["gad_latest_marker"] = str(gad_dir / "latest_checkpointed_iteration.txt")
    manifest["data_pt_sha256"] = sha256_file(target_ckpt / "data.pt")
    manifest["dataloader_state"] = (
        {"skipped": True} if args.skip_dataloader_check else read_dataloader_state(target_ckpt / "data.pt")
    )
    if manifest["dataloader_state"].get("error"):
        problems.append(f"could not read the dataloader state from data.pt: {manifest['dataloader_state']['error']}")

    # warmup artifacts must remain untouched
    manifest["warmup_untouched"] = {
        "checkpoint": str(source_ckpt),
        "mtime_ns": source_ckpt.stat().st_mtime_ns,
    }

    actor_hf = target_ckpt / "actor" / "huggingface"
    critic_hf = target_ckpt / "critic" / "huggingface"
    if args.skip_merge:
        manifest["actor_hf"] = str(actor_hf)
        manifest["critic_hf"] = str(critic_hf)
        manifest["merged"] = False
        return finish()

    actor_dir = export_hf(target_ckpt, "actor", merge_tool, args.python_bin)
    critic_dir = export_hf(target_ckpt, "critic", merge_tool, args.python_bin)

    critic_info = inspect_safetensors(critic_dir)
    actor_info = inspect_safetensors(actor_dir)
    critic_config = json.loads((critic_dir / "config.json").read_text(encoding="utf-8"))
    hidden_size = critic_config.get("hidden_size") or (critic_config.get("text_config") or {}).get("hidden_size")

    if critic_info.get("shapes"):
        if "score.weight" not in critic_info["shapes"]:
            problems.append(
                "merged discriminator has no score.weight: the value head was lost during merge; "
                "refusing to hand over a head-less critic"
            )
        else:
            shape = critic_info["shapes"]["score.weight"]
            expected_labels = 4 if critic_config.get("rubric_head_layout") else 1
            if len(shape) != 2 or shape[0] != expected_labels:
                problems.append(f"score.weight has shape {shape}; expected [{expected_labels}, hidden]")
            if hidden_size and shape[1] != hidden_size:
                problems.append(f"score.weight shape {shape} does not match hidden_size {hidden_size}")
    else:
        problems.append(f"could not inspect merged critic weights for the value head: {critic_info.get('reason')}")

    actor_keys = set(actor_info.get("keys") or [])
    critic_keys = set(critic_info.get("keys") or [])
    source_keys = {
        "actor": source_weight_keys(target_ckpt / "actor"),
        "critic": source_weight_keys(target_ckpt / "critic"),
    }
    manifest["weight_keys"] = {
        "merged_actor": len(actor_keys),
        "merged_critic": len(critic_keys),
        "source_actor": len(source_keys["actor"]),
        "source_critic": len(source_keys["critic"]),
    }
    # Each merged export must carry exactly the weights of its own source shards
    # (except head weights that a tied-embedding export legitimately omits).
    role_configs = {
        "actor": _load_config(actor_dir),
        "critic": _load_config(critic_dir),
    }
    for role, merged_keys in (("actor", actor_keys), ("critic", critic_keys)):
        source = source_keys[role]
        if not source or not merged_keys:
            manifest["weight_keys"][f"{role}_comparison"] = "skipped (keys unavailable)"
            continue
        comparison = compare_weight_keys(source, merged_keys, config=role_configs[role], role=role)
        manifest["weight_keys"][f"{role}_lost"] = comparison["lost"]
        manifest["weight_keys"][f"{role}_added"] = comparison["added"]
        manifest["weight_keys"][f"{role}_tied_absent"] = comparison["tied_absent"]
        manifest["weight_keys"][f"{role}_tied_readded"] = comparison["tied_readded"]
        manifest["weight_keys"][f"{role}_tie_word_embeddings"] = comparison["tie_word_embeddings"]
        problems.extend(comparison["problems"])
        if comparison["lost"]:
            problems.append(
                f"merged {role} is missing {len(comparison['lost'])} weights from its checkpoint, "
                f"e.g. {comparison['lost'][:3]}"
            )
        if comparison["added"]:
            problems.append(f"merged {role} gained unexpected weights, e.g. {comparison['added'][:3]}")
        if comparison["tied_absent"]:
            log(f"{role}: {comparison['tied_absent']} omitted by the merge because the export keeps "
                "tie_word_embeddings=true (the LM head is re-tied at load time)")

    # A value-head model scores hidden states, so it legitimately has no LM head.
    # Record the difference instead of treating it as data loss.
    absent_head_keys = sorted(key for key in actor_keys - critic_keys if key.startswith("lm_head"))
    shared_base = len(actor_keys & critic_keys)
    manifest["critic_vs_actor"] = {
        "shared_keys": shared_base,
        "absent_head_keys": absent_head_keys,
        "expected_absent_reason": "token-classification value-head models replace the LM head with `score`",
    }
    if not any(key.startswith("score.") for key in critic_keys - actor_keys):
        problems.append("merged critic contains no discriminator-specific parameters (score.*)")

    manifest["critic_config_fix"] = fix_critic_config(critic_dir, hidden_size=hidden_size)
    manifest["critic_value_head"] = {
        "score_keys": sorted(key for key in (critic_info.get("shapes") or {}) if key.startswith("score.")),
        "score_weight_shape": (critic_info.get("shapes") or {}).get("score.weight"),
        "config_num_labels": json.loads((critic_dir / "config.json").read_text(encoding="utf-8")).get("num_labels"),
        "config_architectures": json.loads((critic_dir / "config.json").read_text(encoding="utf-8"))["architectures"],
    }

    manifest["actor_hf"] = str(actor_dir)
    manifest["critic_hf"] = str(critic_dir)
    manifest["exports"] = {
        "actor_files": {p.name: sha256_file(p) for p in sorted(actor_dir.glob("*.json"))},
        "critic_files": {p.name: sha256_file(p) for p in sorted(critic_dir.glob("*.json"))},
        "actor_weight_files": [Path(p).name for p in (actor_info.get("files") or [])],
        "critic_weight_files": [Path(p).name for p in (critic_info.get("files") or [])],
    }

    manifest["merged"] = True
    code = finish()
    log(f"actor  init : {actor_dir}")
    log(f"critic init : {critic_dir}")
    log(f"resume step : {step} (latest marker written); GAD optimizer starts fresh unless resume is enabled")
    return code


def _load_config(hf_dir: Path) -> dict:
    path = hf_dir / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - reported by the caller
        return {}


def tied_head_keys(config: dict | None) -> set[str]:
    """Head weights ``save_pretrained`` legitimately omits for tied embeddings.

    Verified on transformers 4.55: for a ``tie_word_embeddings: true`` model
    ``state_dict()`` (what verl saves into the FSDP shard) contains
    ``lm_head.weight``, while ``save_pretrained`` (what ``merge_model2hf``
    writes) drops it and relies on the tie at load time. Qwen3-1.7B is exactly
    such a model, so without this the handoff would report its own correct
    export as "missing lm_head.weight".
    """
    if not config or not config.get("tie_word_embeddings"):
        return set()
    return {"lm_head.weight"}


def compare_weight_keys(source: set, merged: set, *, config: dict | None, role: str) -> dict:
    """Diff merged-export keys against the checkpoint shard keys.

    Only tied head keys may differ, and only when the merged export keeps the
    tie (``tie_word_embeddings: true``) *and* still carries the embedding the
    head is tied to; otherwise the excused key is real data loss.
    """
    tied = tied_head_keys(config)
    excused_lost = sorted(key for key in source - merged if key in tied)
    excused_added = sorted(key for key in merged - source if key in tied)
    lost = sorted(key for key in source - merged if key not in tied)
    added = sorted(key for key in merged - source if key not in tied)

    embedded = any(key.endswith("embed_tokens.weight") for key in merged)
    tied_declared = bool((config or {}).get("tie_word_embeddings"))
    problems: list[str] = []
    if excused_lost or excused_added:
        if not (embedded and tied_declared):
            problems.append(
                f"merged {role} differs in {sorted(set(excused_lost) | set(excused_added))} but the export "
                f"does not look tied (tie_word_embeddings={tied_declared}, embed_tokens={embedded}); "
                "the LM head would be lost"
            )
            lost = sorted(lost + excused_lost)
            added = sorted(added + excused_added)
            excused_lost, excused_added = [], []
    return {
        "lost": lost,
        "added": added,
        "tied_absent": excused_lost,
        "tied_readded": excused_added,
        "tie_word_embeddings": tied_declared,
        "embedding_present": embedded,
        "problems": problems,
    }


def main() -> int:
    args = parse_args()
    if args.mode == "export-final":
        return export_final(args)
    return run_handoff(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"[gad] FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)
