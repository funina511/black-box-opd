"""Offline teacher index: schema, reading, writing and integrity checks.

The index is a JSONL file with one record per training/dev prompt::

    {
      "schema_version": "gad.teacher_index.v1",
      "uid": "<extra_info.index>",
      "raw_prompt_hash": "<sha256 of canonicalize_raw_prompt(prompt)>",
      "teacher_answers": ["<teacher text 1>", ...],
      "teacher_fingerprint": { ... },          # see task_template.build_teacher_fingerprint_payload
      "teacher_fingerprint_digest": "<sha256>",
      "sampling": {                            # per-answer metadata, added by sample_teacher.py
         "finish_reasons": ["stop", ...],
         "truncated": [false, ...]
      },
      "source": "gad"                          # "gad" | "card-import"
    }

Design rules (see ``LMOps/GAD修改计划.md`` section 5):

* ``teacher_answers`` holds teacher *text* only -- never logits, never token ids.
* A record is keyed by ``(uid, raw_prompt_hash)`` so that a prompt edit cannot
  silently reuse an answer that was generated for a different question.
* Reading is strict by default: duplicate keys, empty answers, missing fields,
  bad JSON and fingerprint mismatches raise instead of being skipped.
* Nothing is deleted automatically. ``append_record`` refuses to touch a file
  whose header fingerprint differs from the requested one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from task_template import (
    PROMPT_TEMPLATE_VERSION,
    canonicalize_raw_prompt,
    hash_raw_prompt,
    hash_teacher_fingerprint,
    sha256_text,
)

GAD_INDEX_SCHEMA_VERSION = "gad.teacher_index.v1"
# Schemas produced by the CARD reference implementation that we can import.
CARD_INDEX_SCHEMAS = ("card.teacher_index.v1", "card.teacher_index.v2")

INDEX_HEADER_KEY = "_header"


class TeacherIndexError(RuntimeError):
    """Raised for any index integrity problem (never swallowed)."""


@dataclass
class TeacherIndexRecord:
    uid: str
    raw_prompt_hash: str
    teacher_answers: list[str]
    teacher_fingerprint: dict[str, Any]
    teacher_fingerprint_digest: str
    sampling: dict[str, Any] = field(default_factory=dict)
    source: str = "gad"
    schema_version: str = GAD_INDEX_SCHEMA_VERSION

    @property
    def teacher_answer(self) -> str:
        return self.teacher_answers[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "uid": self.uid,
            "raw_prompt_hash": self.raw_prompt_hash,
            "teacher_answers": list(self.teacher_answers),
            "teacher_fingerprint": dict(self.teacher_fingerprint),
            "teacher_fingerprint_digest": self.teacher_fingerprint_digest,
            "sampling": dict(self.sampling),
            "source": self.source,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any], *, path: str = "<memory>", line_number: int = 0) -> "TeacherIndexRecord":
        where = f"{path}:{line_number}"
        schema = row.get("schema_version")
        if schema in CARD_INDEX_SCHEMAS:
            answers = _card_answers(row, where=where)
            return cls(
                uid=str(row["uid"]),
                raw_prompt_hash=str(row.get("raw_prompt_hash", "")),
                teacher_answers=answers,
                teacher_fingerprint=dict(row.get("teacher_fingerprint") or {}),
                teacher_fingerprint_digest=str(row.get("teacher_fingerprint_digest", ""))
                or hash_teacher_fingerprint(dict(row.get("teacher_fingerprint") or {})),
                sampling={},
                source="card-import",
                schema_version=str(schema),
            )

        if schema != GAD_INDEX_SCHEMA_VERSION:
            raise TeacherIndexError(f"unsupported teacher index schema at {where}: {schema!r}")

        for key in ("uid", "raw_prompt_hash", "teacher_answers", "teacher_fingerprint"):
            if key not in row:
                raise TeacherIndexError(f"teacher index record missing {key!r} at {where}")
        answers = row["teacher_answers"]
        if not isinstance(answers, list) or not answers:
            raise TeacherIndexError(f"teacher index record needs a non-empty teacher_answers list at {where}")
        answers = [str(a).strip() for a in answers]
        if any(not a for a in answers):
            raise TeacherIndexError(f"teacher index record has an empty teacher answer at {where}")

        fingerprint = dict(row["teacher_fingerprint"])
        digest = str(row.get("teacher_fingerprint_digest") or hash_teacher_fingerprint(fingerprint))
        if digest != hash_teacher_fingerprint(fingerprint):
            raise TeacherIndexError(f"teacher fingerprint digest mismatch at {where}")

        uid = str(row["uid"]).strip()
        if not uid:
            raise TeacherIndexError(f"teacher index record has an empty uid at {where}")

        return cls(
            uid=uid,
            raw_prompt_hash=str(row["raw_prompt_hash"]),
            teacher_answers=answers,
            teacher_fingerprint=fingerprint,
            teacher_fingerprint_digest=digest,
            sampling=dict(row.get("sampling") or {}),
            source=str(row.get("source", "gad")),
            schema_version=str(schema),
        )


def _card_answers(row: Mapping[str, Any], *, where: str) -> list[str]:
    if row.get("schema_version") == "card.teacher_index.v1":
        answer = str(row.get("teacher_answer", "")).strip()
        if not answer:
            raise TeacherIndexError(f"CARD teacher index record has an empty teacher_answer at {where}")
        return [answer]
    answers = row.get("teacher_answers")
    if not isinstance(answers, list) or not answers:
        raise TeacherIndexError(f"CARD teacher index record needs a non-empty teacher_answers list at {where}")
    answers = [str(a).strip() for a in answers]
    if any(not a for a in answers):
        raise TeacherIndexError(f"CARD teacher index record has an empty teacher answer at {where}")
    return answers


def header_record(*, fingerprint: dict[str, Any], created_by: str, notes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """First line of a GAD index file: identifies the sampling configuration."""
    return {
        INDEX_HEADER_KEY: GAD_INDEX_SCHEMA_VERSION,
        "teacher_fingerprint": dict(fingerprint),
        "teacher_fingerprint_digest": hash_teacher_fingerprint(fingerprint),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "created_by": created_by,
        "notes": dict(notes or {}),
    }


class TeacherIndex:
    """In-memory view of an index file, with strict loading."""

    def __init__(self, records: dict[tuple[str, str], TeacherIndexRecord], header: dict[str, Any] | None = None,
                 path: Path | None = None):
        self._records = records
        self.header = header
        self.path = path

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, uid: str) -> bool:
        return any(key[0] == uid for key in self._records)

    @property
    def uids(self) -> set[str]:
        return {key[0] for key in self._records}

    def get(self, *, uid: str, raw_prompt: Any | None = None) -> TeacherIndexRecord | None:
        if raw_prompt is not None:
            key = (str(uid), hash_raw_prompt(raw_prompt))
            return self._records.get(key)
        for candidate_key, record in self._records.items():
            if candidate_key[0] == str(uid):
                return record
        return None

    def require(self, *, uid: str, raw_prompt: Any) -> TeacherIndexRecord:
        record = self.get(uid=uid, raw_prompt=raw_prompt)
        if record is None:
            raise TeacherIndexError(
                f"teacher index miss for uid={uid!r} raw_prompt_hash={hash_raw_prompt(raw_prompt)!r} "
                f"(index={self.path})"
            )
        return record

    def items(self) -> Iterable[tuple[tuple[str, str], TeacherIndexRecord]]:
        return self._records.items()

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = True) -> "TeacherIndex":
        index_path = Path(path).expanduser()
        if not index_path.exists():
            raise TeacherIndexError(f"teacher index not found: {index_path}")

        records: dict[tuple[str, str], TeacherIndexRecord] = {}
        header: dict[str, Any] | None = None
        duplicates: list[str] = []
        broken: list[str] = []

        with index_path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    if strict:
                        raise TeacherIndexError(f"corrupt JSON at {index_path}:{line_number}: {exc}") from exc
                    broken.append(f"{index_path}:{line_number}")
                    continue

                if INDEX_HEADER_KEY in row:
                    header = row
                    continue

                record = TeacherIndexRecord.from_json(row, path=str(index_path), line_number=line_number)
                key = (record.uid, record.raw_prompt_hash)
                if key in records:
                    duplicates.append(f"{record.uid}@{line_number}")
                    if strict:
                        raise TeacherIndexError(f"duplicate teacher index key at {index_path}:{line_number}: {key!r}")
                    continue
                records[key] = record

        if broken and strict:
            raise TeacherIndexError(f"corrupt teacher index lines: {broken}")

        return cls(records=records, header=header, path=index_path)

    def fingerprint_conflicts(self, expected_digest: str) -> list[str]:
        return sorted(
            {
                record.teacher_fingerprint_digest
                for record in self._records.values()
                if record.teacher_fingerprint_digest != expected_digest
            }
        )


def append_record(path: str | Path, record: TeacherIndexRecord, *, fsync: bool = False) -> None:
    """Append one record, flushing immediately so a crash keeps prior work."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_json(), ensure_ascii=False)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def read_done_uids(path: str | Path, *, expected_digest: str | None = None) -> set[str]:
    """uids that already have at least ``--m`` answers (used for resume)."""
    target = Path(path).expanduser()
    done: set[str] = set()
    if not target.exists():
        return done
    with target.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if INDEX_HEADER_KEY in row:
                continue
            try:
                record = TeacherIndexRecord.from_json(row, path=str(target), line_number=line_number)
            except TeacherIndexError:
                continue
            if expected_digest is not None and record.teacher_fingerprint_digest != expected_digest:
                continue
            done.add(record.uid)
    return done


def find_fingerprint_mismatch(path: str | Path, expected_digest: str) -> dict[str, Any] | None:
    """Return the first record whose fingerprint differs, for a fail-fast check."""
    target = Path(path).expanduser()
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if INDEX_HEADER_KEY in row:
                digest = row.get("teacher_fingerprint_digest")
                if digest and digest != expected_digest:
                    return {"line": line_number, "kind": "header", "digest": digest}
                continue
            digest = row.get("teacher_fingerprint_digest") or hash_teacher_fingerprint(
                dict(row.get("teacher_fingerprint") or {})
            )
            if digest != expected_digest:
                return {"line": line_number, "kind": "record", "uid": row.get("uid"), "digest": digest}
    return None


__all__ = [
    "CARD_INDEX_SCHEMAS",
    "GAD_INDEX_SCHEMA_VERSION",
    "INDEX_HEADER_KEY",
    "TeacherIndex",
    "TeacherIndexError",
    "TeacherIndexRecord",
    "append_record",
    "canonicalize_raw_prompt",
    "find_fingerprint_mismatch",
    "header_record",
    "read_done_uids",
    "sha256_text",
]
