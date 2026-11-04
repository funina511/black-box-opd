from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path: str | Path) -> str | None:
    target = Path(path)
    return sha256_file(target) if target.is_file() else None


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-.") or "run"


def atomic_write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, target)


def atomic_write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def percentile(values: list[int | float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def init_wandb(
    *,
    mode: str,
    project: str,
    entity: str | None,
    group: str | None,
    name: str,
    job_type: str,
    config: dict[str, Any],
):
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B logging was requested, but `wandb` is not installed.") from error

    return wandb.init(
        project=project,
        entity=entity or None,
        group=group or None,
        name=name,
        job_type=job_type,
        mode=mode,
        config=config,
    )


def log_wandb_artifact(
    run,
    *,
    name: str,
    artifact_type: str,
    files: Iterable[str | Path],
    metadata: dict[str, Any] | None = None,
) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(
        name=sanitize_name(name),
        type=artifact_type,
        metadata=metadata or {},
    )
    for file_path in files:
        path = Path(file_path)
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
