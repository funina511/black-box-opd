import json
import os
from pathlib import Path
from .core.contracts import DIMENSIONS, LAYOUT, digest, file_hash

VERSION = "gad.extensions.v1"


def semantic_module_hash(process_protocol):
    if process_protocol == 'direct_v1':
        # Existing direct_v1 services, label caches and annotation journals
        # include this value in their keys. The protocol never runs semantic_v24.
        return '7bd48b6379513e9d2ce453620f169466f687774fb48cc4d1d176175c4059a26d'
    from .core import semantic_v24
    return file_hash(semantic_v24.__file__)


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
