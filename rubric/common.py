"""Dependency-light configuration and artifact helpers."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sample"))
sys.path.insert(0, str(ROOT / "gad" / "tools"))
DIMENSIONS = ("answer_correctness", "reasoning_validity", "condition_coverage")
LAYOUT = ("adv", *DIMENSIONS)
PATH_KEYS = ("run.output_root", "data.output_root", "source.run_config",
             "source.actor_checkpoint", "source.critic_checkpoint", "source.dataset_manifest",
             "rubric.spec", "judge.prompt", "judge.model", "judge.annotations_path", "audit.reviews_path")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def get(data, key, default=None):
    for part in key.split("."):
        if not isinstance(data, dict) or part not in data:
            return default
        data = data[part]
    return data


def put(data, key, value):
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def load_config(path, settings=(), size=None):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != "rubric.offline.v1":
        raise ValueError("expected rubric.offline.v1 config")
    for entry in settings:
        key, sep, value = entry.partition("=")
        if not sep or get(data, key, "__missing__") == "__missing__":
            raise ValueError(f"unknown override: {entry}")
        put(data, key, (None if value == "null" else value) if key in PATH_KEYS or key == "resources.gpus"
            else yaml.safe_load(value))
    if size:
        data["sampling"]["prompt_count"] = 24 if size == "smoke" else 512
        data["split"] = {"unit": "prompt", "train": 16 if size == "smoke" else 320,
                         "dev": 4 if size == "smoke" else 96, "test": 4 if size == "smoke" else 96}
    for key in PATH_KEYS:
        if value := get(data, key):
            candidate = Path(value).expanduser()
            put(data, key, str((path.parent / candidate).resolve()) if not candidate.is_absolute()
                else str(candidate.resolve()))
    if sum(data["split"][part] for part in ("train", "dev", "test")) != data["sampling"]["prompt_count"]:
        raise ValueError("split sizes must sum to prompt_count")
    if data["sampling"]["teacher_n"] != 1 or data["sampling"]["student_n"] != 2:
        raise ValueError("v1 requires one teacher and two fixed student candidates")
    if data["resources"]["strategy"] != "single_gpu":
        raise ValueError("only single_gpu training is verified")
    if data["training"]["update_actor"] or data["training"]["reward_mode"] != "original_adv":
        raise ValueError("offline v1 must not update actor or modify its reward")
    if tuple(data["model"]["layout"]) != LAYOUT:
        raise ValueError("head layout mismatch")
    if data["training"]["micro_batch_responses"] != 1:
        raise ValueError("v1 streams one response at a time; use micro_batch_responses=1")
    if data["training"]["mode"] not in ("adv_only", "correctness_aux", "joint"):
        raise ValueError("unsupported training.mode")
    for key in ("logical_batch_prompts", "max_epochs", "save_every_steps"):
        if type(data["training"][key]) is not int or data["training"][key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if data["training"]["lr"] <= 0 or data["training"]["rubric_lambda"] < 0:
        raise ValueError("lr must be positive and lambda nonnegative")
    if data["rubric"]["input_mode"] != "implicit_fixed" or data["rubric"]["unknown_policy"] != "mask":
        raise ValueError("v1 requires implicit_fixed input and masked unknown labels")
    if tuple(data["rubric"]["dimensions"]) != DIMENSIONS or not data["sampling"]["inherit_source_generation_settings"]:
        raise ValueError("fixed v1 dimensions and source generation inheritance are required")
    return data


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def write_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def code_fingerprint():
    files = sorted(Path(__file__).parent.glob("*.py"))
    files += [ROOT / "sample" / "math_scorer.py", ROOT / "sample" / "task_template.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def snapshot_code(run):
    fingerprints = code_fingerprint()
    version = digest(fingerprints)
    destination = Path(run) / "code" / version
    for name, expected in fingerprints.items():
        target = destination / name
        if target.exists():
            if file_hash(target) != expected:
                raise ValueError("saved code snapshot changed")
        else:
            atomic_text(target, (ROOT / name).read_text())
    write_json(destination / "fingerprint.json", fingerprints)
    return version


def model_file_stats(path):
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) + sorted(path.glob("model_world_size_*_rank_*.pt"))
    files += sorted(path.glob("*.json")) + sorted(path.glob("*.jinja"))
    return {p.name: {name: getattr(p.stat(), name) for name in
                     ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")} for p in files}


def model_fingerprint(path, cache_path=None):
    path = Path(path)
    files = sorted(path.glob("*.safetensors")) + sorted(path.glob("model_world_size_*_rank_*.pt"))
    files += sorted(path.glob("*.json")) + sorted(path.glob("*.jinja"))
    if not any(p.suffix in (".pt", ".safetensors") for p in files):
        raise ValueError(f"model weights missing: {path}")
    before = model_file_stats(path)
    if cache_path and Path(cache_path).exists():
        cached = read_json(cache_path)
        if cached["file_stats"] == before and cached["fingerprint"]["path"] == str(path.resolve()):
            return cached["fingerprint"]
    hashes = {p.name: file_hash(p) for p in files}
    if model_file_stats(path) != before:
        raise ValueError("model files changed during content fingerprinting")
    fingerprint = {"path": str(path.resolve()), "files": hashes, "digest": digest(hashes)}
    if cache_path:
        write_json(cache_path, {"file_stats": before, "fingerprint": fingerprint})
    return fingerprint


def source_config(cfg):
    from config import load_config as load_training_config
    if not get(cfg, "source.run_config"):
        raise ValueError("source.run_config must be explicit")
    return load_training_config(cfg["source"]["run_config"]).for_stage("warmup")


def generation_settings(cfg):
    source = source_config(cfg)
    return {"temperature": float(source.get("rollout.temperature")), "top_p": float(source.get("rollout.top_p")),
            "max_prompt_length": int(source.get("data.max_prompt_length")),
            "max_response_length": int(source.get("data.max_response_length")),
            "enable_thinking": bool(source.get("rollout.enable_thinking")),
            "thinking_control": bool(source.get("rollout.thinking_control")),
            "source_warmup_n": int(source.get("rollout.n_warmup")),
            "student_n": cfg["sampling"]["student_n"], "seed": cfg["run"]["seed"]}


def paths(cfg):
    count = cfg["sampling"]["prompt_count"]
    name = cfg["run"]["name"]
    if name == "auto":
        name = f"math_v1_{count}q_seed{cfg['run']['seed']}_{digest(cfg)[:12]}"
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("invalid run name")
    return Path(cfg["run"]["output_root"]) / name, Path(cfg["data"]["output_root"]) / name


def claim_run(cfg):
    run, dataset = paths(cfg)
    saved = run / "config.resolved.yaml"
    if saved.exists():
        if yaml.safe_load(saved.read_text()) != cfg:
            raise ValueError("run config is immutable; choose a new run.name")
    else:
        atomic_text(saved, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        write_json(run / "code_fingerprint.json", code_fingerprint())
    environment = run / "environment.json"
    current = environment_fingerprint()
    if environment.exists() and read_json(environment) != current:
        raise ValueError("runtime package versions changed; use a new run")
    if not environment.exists():
        write_json(environment, current)
    return run, dataset


def environment_fingerprint():
    import importlib.metadata
    versions = {}
    for package in ("torch", "transformers", "vllm", "numpy", "pyarrow", "safetensors", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "packages": versions}


def require_gpu(cfg, judge=False):
    value = str(cfg["resources"].get("gpus") or "")
    ids = value.split(",") if value else []
    if not ids or any(not part.isdigit() for part in ids) or len(ids) != len(set(ids)):
        raise ValueError("resources.gpus must explicitly list distinct GPU IDs")
    if judge and len(ids) < int(cfg["judge"]["tensor_parallel_size"]):
        raise ValueError("not enough GPUs for judge tensor_parallel_size")
    return value if judge else ids[0]
