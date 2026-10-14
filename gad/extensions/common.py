import json
import os
from pathlib import Path
from .core.contracts import DIMENSIONS, LAYOUT, digest, file_hash

VERSION = "gad.extensions.v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def plain(config):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config


def enabled(config):
    return bool(config.trainer.get("rubric", {}).get("enabled", False)
                or config.trainer.get("critic_replay", {}).get("enabled", False))
