"""Bounded CPU pair pool with a dedicated RNG and transactional step insertion."""
import json
import random
from collections import deque
from copy import deepcopy
from .common import digest


class ReplayBuffer:
    def __init__(self, options):
        self.options = options
        self.entries = deque()
        self.bytes = 0
        self.rng = random.Random(options["seed"])
        self.last_committed_step = 0
        self.evicted_age = self.evicted_capacity = self.skipped_oversize = 0

    @staticmethod
    def size(pair):
        return len(json.dumps(pair, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())

    def evict(self, stage_step):
        while self.entries and stage_step - self.entries[0]["stage_step"] > self.options["max_age_steps"]:
            self.bytes -= self.size(self.entries.popleft())
            self.evicted_age += 1
        while self.entries and (len(self.entries) > self.options["capacity_pairs"] or self.bytes > self.options["max_bytes"]):
            self.bytes -= self.size(self.entries.popleft())
            self.evicted_capacity += 1

    def select(self, fresh, stage_step):
        self.evict(stage_step)
        k = min(int(self.options["mix_ratio"] * len(fresh)), len(self.entries))
        if k == 0:
            return deepcopy(fresh), 0
        current = self.rng.sample(fresh, len(fresh) - k)
        historical = self.rng.sample(list(self.entries), k)
        return [deepcopy(row) for row in current + historical], k

    def commit(self, fresh, stage_step):
        if stage_step <= self.last_committed_step:
            raise ValueError("replay step committed twice or backwards")
        for pair in fresh:
            row = deepcopy(pair)
            row["stage_step"] = stage_step
            row["labels_pending"] = "labels" not in row
            size = self.size(row)
            if size > self.options["max_bytes"]:
                self.skipped_oversize += 1
                continue
            self.entries.append(row)
            self.bytes += size
            self.evict(stage_step)
        self.last_committed_step = stage_step

    def state_dict(self):
        return {"options": self.options, "entries": list(self.entries), "rng": self.rng.getstate(),
                "last_committed_step": self.last_committed_step, "bytes": self.bytes,
                "evicted_age": self.evicted_age, "evicted_capacity": self.evicted_capacity,
                "skipped_oversize": self.skipped_oversize, "content_hash": digest(list(self.entries))}

    def load_state_dict(self, state):
        if state["options"] != self.options or digest(state["entries"]) != state["content_hash"]:
            raise ValueError("replay config or content mismatch")
        self.entries = deque(deepcopy(state["entries"]))
        self.bytes = sum(self.size(row) for row in self.entries)
        if self.bytes != state["bytes"]:
            raise ValueError("replay byte count mismatch")
        self.rng.setstate(state["rng"])
        for key in ("last_committed_step", "evicted_age", "evicted_capacity", "skipped_oversize"):
            setattr(self, key, state[key])
