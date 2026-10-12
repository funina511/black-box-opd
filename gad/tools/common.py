"""Shared helpers for the GAD pipeline tools.

Kept dependency-light (stdlib + PyYAML): these helpers run on the login node and
inside `--dry-run`, so they must not require torch/vllm/verl.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_PYTHON = sys.executable

# Stage order used by the pipeline. `sample` is the offline data stage and is
# not part of the three training backends.
TRAIN_STAGES = ("seqkd", "warmup", "gad")


class PipelineError(RuntimeError):
    """Any fatal pipeline/preflight problem."""


def log(message: str, *, component: str = "gad") -> None:
    print(f"[{component}] {message}", flush=True)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_id(prefix: str = "run") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def require_path(path: str | Path, what: str) -> Path:
    target = Path(path).expanduser()
    if not target.exists():
        raise PipelineError(f"{what} not found: {target}")
    return target


def run_command(
    argv: Sequence[str],
    *,
    env: dict | None = None,
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
    check: bool = True,
    dry_run: bool = False,
    component: str = "run",
) -> subprocess.CompletedProcess:
    """Run a command, optionally teeing stdout/stderr into a log file."""
    printable = " ".join(shlex.quote(str(a)) for a in argv)
    log(f"$ {printable}", component=component)
    if dry_run:
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items() if v is not None})

    if log_path:
        target = ensure_dir(Path(log_path).parent) / Path(log_path).name
        with target.open("w", encoding="utf-8") as handle:
            handle.write(f"# $ {printable}\n")
            handle.flush()
            process = subprocess.run(
                [str(a) for a in argv],
                env=merged_env,
                cwd=str(cwd) if cwd else None,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if check and process.returncode != 0:
            raise PipelineError(f"command failed with exit code {process.returncode}: {printable} (log: {target})")
        return process

    process = subprocess.run([str(a) for a in argv], env=merged_env, cwd=str(cwd) if cwd else None)
    if check and process.returncode != 0:
        raise PipelineError(f"command failed with exit code {process.returncode}: {printable}")
    return process


def capture_command(
    argv: Sequence[str],
    *,
    env: dict | None = None,
    cwd: str | Path | None = None,
) -> tuple[int, str, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update({k: str(v) for k, v in env.items() if v is not None})
    process = subprocess.run(
        [str(a) for a in argv],
        env=merged_env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def git_rev_parse(repo: str | Path, ref: str = "HEAD") -> str:
    code, out, err = capture_command(["git", "-C", str(repo), "rev-parse", ref])
    if code != 0:
        raise PipelineError(f"git rev-parse {ref} failed in {repo}: {err}")
    return out


def git_is_dirty(repo: str | Path) -> bool:
    code, out, _ = capture_command(["git", "-C", str(repo), "status", "--porcelain"])
    return code == 0 and bool(out.strip())


def git_show_file(repo: str | Path, ref: str, path: str) -> str | None:
    code, out, _ = capture_command(["git", "-C", str(repo), "show", f"{ref}:{path}"])
    return out if code == 0 else None


def replace_once(text: str, anchor: str, replacement: str, *, what: str) -> str:
    """Replace a unique anchor, refusing ambiguous or missing matches."""
    count = text.count(anchor)
    if count == 0:
        raise PipelineError(f"patch anchor not found ({what}): {anchor[:120]!r}")
    if count > 1:
        raise PipelineError(f"patch anchor is not unique ({what}, {count} matches): {anchor[:120]!r}")
    return text.replace(anchor, replacement, 1)


def insert_after(text: str, anchor: str, insertion: str, *, what: str) -> str:
    return replace_once(text, anchor, anchor + insertion, what=what)


def contains_all(text: str, markers: Iterable[str]) -> bool:
    return all(marker in text for marker in markers)


def python_version(python_bin: str) -> str:
    code, out, err = capture_command([python_bin, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"])
    return out if code == 0 else f"<error: {err}>"


def environment_summary() -> dict:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }


_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def slugify(text: str) -> str:
    return _SLUG.sub("_", str(text)).strip("_") or "value"


def parse_key_values(pairs: Sequence[str] | None) -> dict:
    """Parse repeated ``key=value`` arguments into scalar config values.

    Decimal numbers, booleans and null use their native types. Other values
    remain strings, including GPU lists and paths such as ``007.parquet``.
    """
    result: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise PipelineError(f"expected key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise PipelineError(f"expected a nonempty key, got {pair!r}")
        lowered = value.lower()
        if lowered in ("true", "false"):
            parsed = lowered == "true"
        elif lowered in ("null", "none"):
            parsed = None
        elif re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", value):
            parsed = int(value)
        elif re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value) and any(c in value for c in ".eE"):
            parsed = float(value)
        else:
            parsed = value
        result[key] = parsed
    return result


def flatten_dict(data: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in (data or {}).items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        else:
            flat[name] = value
    return flat


def human_bytes(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PiB"


def free_disk_gib(path: str | Path) -> float:
    usage = os.statvfs(str(path))
    return usage.f_bavail * usage.f_frsize / (1024 ** 3)


# --------------------------------------------------------------------------- #
# vLLM KV-cache budget
#
# vLLM sizes the KV cache once, while the engine is constructed
# (v1/engine/core.py:_initialize_kv_caches -> v1/core/kv_cache_utils.py:
# check_enough_kv_cache_memory) and refuses to start when the pool cannot hold a
# single sequence of ``max_model_len`` tokens. ``wake_up`` afterwards only
# re-attaches the allocator (v1/worker/gpu_worker.py:wake_up), so the pool is
# never resized: whatever the engine picks at init is what the whole run gets.
#
# The non-KV overhead below is the *measured* engine-side usage on the target
# host (4x RTX 3090 24GB, Qwen3-1.7B bf16, vLLM 0.8.5, enforce_eager=False):
# at gpu_memory_utilization=0.30 the engine reported 0.95 GiB of KV cache,
# while its bf16 weights are 3.4 GiB and 0.30 x 23.5 GiB = 7.05 GiB, leaving
# ~2.7 GiB for profiling/activation/graph overhead. Keep it a documented
# constant: it makes the check predictive instead of a guess.
# --------------------------------------------------------------------------- #
VLLM_NON_KV_OVERHEAD_GIB = 2.7

_DTYPE_BYTES = {
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "fp16": 2,
    "half": 2,
    "float32": 4,
    "fp32": 4,
    "float": 4,
}


def kv_cache_bytes_per_token(model_config: dict) -> int:
    """KV bytes for one token of one sequence: 2 (K and V) x heads x dim x layers x dtype."""
    layers = int(model_config["num_hidden_layers"])
    attention_heads = int(model_config["num_attention_heads"])
    kv_heads = int(model_config.get("num_key_value_heads") or attention_heads)
    head_dim = model_config.get("head_dim") or int(model_config["hidden_size"]) // attention_heads
    dtype = str(model_config.get("torch_dtype") or "bfloat16").split(".")[-1].lower()
    if dtype not in _DTYPE_BYTES:
        raise ValueError(f"unsupported torch_dtype for the KV estimate: {model_config.get('torch_dtype')!r}")
    return 2 * kv_heads * int(head_dim) * layers * _DTYPE_BYTES[dtype]


def kv_cache_gib(model_config: dict, max_model_len: int) -> float:
    """GiB the KV cache needs for ONE sequence of ``max_model_len`` tokens (the init gate)."""
    return kv_cache_bytes_per_token(model_config) * int(max_model_len) / (1024 ** 3)


def vllm_kv_budget(
    *,
    model_config: dict,
    max_model_len: int,
    gpu_memory_utilization: float,
    total_memory_gib: float,
    weight_gib: float,
    overhead_gib: float = VLLM_NON_KV_OVERHEAD_GIB,
) -> dict:
    """Predict vLLM's engine-construction KV-cache check for one GPU.

    ``available_gib`` mirrors vLLM's own accounting: the engine claims
    ``gpu_memory_utilization x total`` and subtracts the weights it loads plus its
    profiling overhead; whatever is left is the KV pool. ``ok`` False means the
    engine would abort with "KV cache ... larger than the available KV cache
    memory"; ``tight`` means it starts but with almost no concurrency.
    """
    need = kv_cache_gib(model_config, max_model_len)
    budget = float(gpu_memory_utilization) * float(total_memory_gib)
    available = budget - float(weight_gib) - float(overhead_gib)
    return {
        "max_model_len": int(max_model_len),
        "kv_bytes_per_token": kv_cache_bytes_per_token(model_config),
        "need_gib": round(need, 3),
        "available_gib": round(available, 3),
        "budget_gib": round(budget, 3),
        "weight_gib": round(float(weight_gib), 3),
        "overhead_gib": round(float(overhead_gib), 3),
        "total_memory_gib": round(float(total_memory_gib), 3),
        "ok": available >= need,
        "tight": available < need * 1.25,
    }
