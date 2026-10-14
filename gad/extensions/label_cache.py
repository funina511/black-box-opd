"""Content-addressed immutable successful annotations; transport failures are not cached."""
import json
from pathlib import Path
from .common import atomic_json, digest, file_hash


class LabelCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.references = {}

    def get(self, key):
        path = self.directory / (key + ".json")
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value.get("cache_key") != key:
            raise ValueError("label cache key mismatch")
        self.references[key] = file_hash(path)
        return value

    def put(self, key, value):
        path = self.directory / (key + ".json")
        value = {**value, "cache_key": key}
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("immutable cache entry changed")
        atomic_json(path, value)
        self.references[key] = file_hash(path)

    def snapshot(self, target):
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        for key, expected in self.references.items():
            source = self.directory / (key + ".json")
            if file_hash(source) != expected:
                raise ValueError("referenced cache object changed")
            destination = target / source.name
            if not destination.exists():
                import os
                # Immutable content-addressed entries may be shared safely with checkpoints.
                try:
                    os.link(source, destination)
                except OSError:
                    destination.write_bytes(source.read_bytes())
        return dict(self.references)

    def restore(self, snapshot, references):
        for key, expected in references.items():
            source = Path(snapshot) / (key + ".json")
            if not source.is_file() or file_hash(source) != expected:
                raise ValueError("checkpoint label cache snapshot missing/changed")
            destination = self.directory / source.name
            if destination.exists() and file_hash(destination) != expected:
                raise ValueError("live cache disagrees with checkpoint")
            if not destination.exists():
                destination.write_bytes(source.read_bytes())
        self.references.update(references)
