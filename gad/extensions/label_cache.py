"""Content-addressed immutable successful annotations; transport failures are not cached."""
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
from .common import atomic_write, file_hash, json_bytes


class LabelCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.references = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            return self._read(key)

    def _read(self, key):
        path = self.directory / (key + ".json")
        expected = self.references.get(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            if expected is not None:
                raise ValueError(f"referenced cache object missing: {key}") from None
            return None
        actual = sha256(data).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f"referenced cache object changed: {key}")
        value = json.loads(data)
        if value.get("cache_key") != key:
            raise ValueError("label cache key mismatch")
        self.references[key] = actual
        return value

    def put(self, key, value):
        path = self.directory / (key + ".json")
        value = {**value, "cache_key": key}
        with self._lock:
            existing = self._read(key)
            if existing is None:
                data = json_bytes(value)
                if atomic_write(path, data, replace=False):
                    self.references[key] = sha256(data).hexdigest()
                    return
                existing = self._read(key)
            if existing != value:
                raise ValueError("immutable cache entry changed")

    def referenced_keys(self, keys):
        with self._lock:
            return self.references.keys() & keys

    def snapshot(self, target):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        with self._lock:
            references = self.references.copy()
        for key, expected in references.items():
            source = self.directory / (key + ".json")
            destination = target / source.name
            if destination.exists():
                if not source.samefile(destination) and file_hash(source) != expected:
                    raise ValueError(f"referenced cache object changed: {key}")
            else:
                try:
                    os.link(source, destination)
                except OSError:
                    # A checkpoint may be on a different filesystem.
                    data = source.read_bytes()
                    if sha256(data).hexdigest() != expected:
                        raise ValueError(f"referenced cache object changed: {key}")
                    atomic_write(destination, data)
                    continue
            if file_hash(destination) != expected:
                raise ValueError(f"checkpoint label cache snapshot changed: {key}")
        return references

    def restore(self, snapshot, references):
        with self._lock:
            for key, expected in references.items():
                source = Path(snapshot) / (key + ".json")
                if not source.is_file():
                    raise ValueError("checkpoint label cache snapshot missing/changed")
                data = source.read_bytes()
                if sha256(data).hexdigest() != expected:
                    raise ValueError("checkpoint label cache snapshot missing/changed")
                if self.references.get(key, expected) != expected:
                    raise ValueError("live cache disagrees with checkpoint")
                destination = self.directory / source.name
                if not destination.exists() and atomic_write(destination, data, replace=False):
                    continue
                if file_hash(destination) != expected:
                    raise ValueError("live cache disagrees with checkpoint")
            self.references.update(references)
