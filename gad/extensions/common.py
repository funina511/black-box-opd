import fcntl
import json
import os
from pathlib import Path
from uuid import uuid4
from .core.contracts import DIMENSIONS, LAYOUT, digest, file_hash

VERSION = "gad.extensions.v1"


def semantic_module_hash(process_protocol):
    if process_protocol == 'direct_brief_v1':
        return 'not_used'
    if process_protocol == 'direct_v1':
        # Existing direct_v1 services, label caches and annotation journals
        # include this value in their keys. The protocol never runs semantic_v24.
        return '7bd48b6379513e9d2ce453620f169466f687774fb48cc4d1d176175c4059a26d'
    from .core import semantic_v24
    return file_hash(semantic_v24.__file__)


def atomic_write(path, data, *, replace=True):
    """Publish complete bytes; return False if a create-only write lost the race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as stream:
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            if replace:
                os.replace(temporary, path)
            else:
                # Cache writers share the directory lock only for publication.
                # Atomic rename also works on volumes without hard-link support.
                lock_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                    if path.exists():
                        return False
                    os.replace(temporary, path)
                finally:
                    os.close(lock_fd)
            return True
        finally:
            temporary.unlink(missing_ok=True)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def atomic_json(path, value):
    atomic_write(path, json_bytes(value))


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
