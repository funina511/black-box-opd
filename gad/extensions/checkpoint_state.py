"""Transactional extension state, independently verifiable without importing torch."""
import json
import os
import random
from pathlib import Path
from .common import VERSION, atomic_json, digest, file_hash


def model_state_problems(path, world_size):
    problems = []
    if not (path / 'data.pt').is_file():
        problems.append('dataloader state missing: data.pt')
    for role in ('actor', 'critic'):
        directory = path / role
        for name in ('config.json', 'tokenizer_config.json'):
            if not (directory / name).is_file():
                problems.append(f'model/tokenizer metadata missing: {role}/{name}')
        if not any((directory / name).is_file() for name in
                   ('tokenizer.json', 'tokenizer.model', 'spiece.model', 'vocab.json', 'vocab.txt')):
            problems.append(f'tokenizer vocabulary missing: {role}')
        for rank in range(world_size):
            for prefix in ('model', 'optim', 'extra_state'):
                candidate = directory / f'{prefix}_world_size_{world_size}_rank_{rank}.pt'
                if not candidate.is_file() or not candidate.stat().st_size:
                    problems.append(f'incomplete model/optimizer/RNG shards: {role}/{candidate.name}')
    return problems


def checkpoint_problems(path, required=False):
    path = Path(path)
    directory = path / "training_extensions"
    marker = path / "checkpoint_complete.json"
    if not directory.exists() and not marker.exists():
        return ["required training_extensions missing"] if required else []
    problems = []
    if not marker.is_file() or not (directory / "manifest.json").is_file():
        return ["extension checkpoint is not atomically complete"]
    try:
        complete = json.loads(marker.read_text())
        if complete["manifest_hash"] != file_hash(directory / "manifest.json"):
            problems.append("extension manifest hash mismatch")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("verify_model_shards"):
            problems.extend(model_state_problems(path, manifest['world_size']))
        for name, expected in manifest.get('model_metadata', {}).items():
            candidate = path / name
            if not candidate.is_file() or file_hash(candidate) != expected:
                problems.append(f'dataloader/model metadata missing/changed: {name}')
        for name, expected in manifest["files"].items():
            candidate = directory / name
            if not candidate.is_file() or file_hash(candidate) != expected:
                problems.append(f"extension state missing/changed: {name}")
        for key, expected in manifest.get("cache_references", {}).items():
            candidate = directory / "labels" / (key + ".json")
            if not candidate.is_file() or file_hash(candidate) != expected:
                problems.append(f"label snapshot missing/changed: {key}")
    except (KeyError, ValueError, OSError) as exc:
        problems.append(f"invalid extension completion: {exc}")
    return problems


def save(runtime, path):
    import numpy as np
    import torch
    path = Path(path)
    directory = path / "training_extensions"
    directory.mkdir(parents=True, exist_ok=True)
    (path / 'checkpoint_complete.json').unlink(missing_ok=True)
    files, references = {}, {}
    rng_file = directory / "sampling_rng.pt"
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state()}, rng_file)
    files[rng_file.name] = file_hash(rng_file)
    if runtime.replay:
        target = directory / "replay_state.pt"
        torch.save(runtime.replay.state_dict(), target)
        files[target.name] = file_hash(target)
    if runtime.rubric:
        references = runtime.rubric.cache.snapshot(directory / "labels")
        target = directory / "rubric_state.json"
        atomic_json(target, {"cache_references": references, "empty_student_steps": runtime.rubric.empty_student_steps,
                             "human_review_status": "unreviewed_exploratory"})
        files[target.name] = file_hash(target)
    manifest = {"version": VERSION, "identity": runtime.identity, "world_size": runtime.world_size,
                "verify_model_shards": hasattr(runtime, "trainer"),
                "stage": runtime.stage, "stage_start": runtime.stage_start, "stage_step": runtime.stage_step,
                "committed_global_step": runtime.committed_step, "files": files, "cache_references": references,
                "replay_pairs": len(runtime.replay.entries) if runtime.replay else 0,
                "replay_hash": digest(list(runtime.replay.entries)) if runtime.replay else None}
    if manifest["verify_model_shards"]:
        problems = model_state_problems(path, runtime.world_size)
        if problems:
            raise ValueError('refusing to publish partial extension checkpoint: ' + '; '.join(problems))
        metadata = {'data.pt': file_hash(path / 'data.pt')}
        for role in ("actor", "critic"):
            for candidate in (path / role).iterdir():
                if candidate.suffix in ('.json', '.jinja') or candidate.name in ('tokenizer.model', 'spiece.model', 'vocab.txt', 'merges.txt'):
                    metadata[str(candidate.relative_to(path))] = file_hash(candidate)
        manifest['model_metadata'] = metadata
    atomic_json(directory / "manifest.json", manifest)
    atomic_json(path / "checkpoint_complete.json", {"version": VERSION, "step": runtime.committed_step,
                                                     "manifest_hash": file_hash(directory / "manifest.json")})


def restore(runtime, path):
    import numpy as np
    import torch
    path = Path(path)
    problems = checkpoint_problems(path, required=True)
    if problems:
        raise ValueError("cannot resume extension checkpoint: " + "; ".join(problems))
    directory = path / "training_extensions"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["identity"] != runtime.identity or manifest["world_size"] != runtime.world_size or manifest["stage"] != runtime.stage:
        raise ValueError("same-stage resume algorithm/source/world-size identity mismatch")
    runtime.stage_start = manifest["stage_start"]
    runtime.stage_step = manifest["stage_step"]
    runtime.committed_step = manifest["committed_global_step"]
    if runtime.replay:
        runtime.replay.load_state_dict(torch.load(directory / "replay_state.pt", weights_only=False))
    if runtime.rubric:
        state = json.loads((directory / "rubric_state.json").read_text())
        runtime.rubric.cache.restore(directory / "labels", state["cache_references"])
        runtime.rubric.empty_student_steps = state["empty_student_steps"]
    rng = torch.load(directory / "sampling_rng.pt", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])


def latest_complete(directory, allow_handoff=False):
    directory = Path(directory)
    candidates = sorted((int(p.name.split("global_step_")[-1]), p)
                        for p in directory.glob("global_step_*") if p.name.split("global_step_")[-1].isdigit())
    for _, path in reversed(candidates):
        if not checkpoint_problems(path, required=True):
            return str(path)
        if allow_handoff and (path / "handoff_initialization.json").is_file():
            return str(path)
    return None
