"""Shared math task template, prompt canonicalization and hashing.

This module is the single source of truth for three things that MUST stay
identical across the teacher sampler, the GAD data converter, the three
training backends and the math evaluator:

1. the *task instruction* used when a bare problem statement has to be turned
   into a chat message (DAPO-Math-17k already ships its own instruction inside
   the ``prompt`` column, so those prompts are passed through untouched --
   see ``prepare_prompts.py``);
2. the ``Answer: ...`` contract that the CARD-aligned math scorer parses
   (``LMOps/gad/tools/math_metrics.py``);
3. the canonical prompt serialization used to key the teacher index, so that a
   teacher answer can be looked up again after the parquet is rewritten.

The instruction text is intentionally byte-identical to CARD's
``scripts/prepare_data.py::INSTRUCTION`` and ``scripts/prepare_eval_benchmarks.py``
so that prompts stay comparable with the CARD reference implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

# CARD-aligned DAPO instruction. Applied ONLY to bare problem statements.
TASK_INSTRUCTION_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your "
    "response should be of the form Answer: $Answer (without quotes) where "
    "$Answer is the answer to the problem.\n\n{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

# Bumped whenever the template above (or the message assembly) changes; it is
# part of the teacher fingerprint, so a bump invalidates an existing index.
PROMPT_TEMPLATE_VERSION = "gad.math.v1"

_SUPPORTED_ROLES = {"system", "user", "assistant", "developer"}


def build_user_message(problem: str) -> dict[str, str]:
    """Wrap a bare problem statement in the DAPO/CARD instruction."""
    return {"role": "user", "content": TASK_INSTRUCTION_TEMPLATE.format(problem=problem)}


def build_content_messages(problem: str) -> list[dict[str, str]]:
    """The ``content`` column value used by the GAD parquet contract."""
    return [build_user_message(problem)]


def normalize_raw_prompt(raw_prompt: Any) -> Any:
    """Convert pyarrow/numpy containers into plain Python lists of dicts."""
    if isinstance(raw_prompt, str):
        return raw_prompt

    if hasattr(raw_prompt, "tolist") and not isinstance(raw_prompt, (bytes, bytearray)):
        raw_prompt = raw_prompt.tolist()

    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)

    if isinstance(raw_prompt, list):
        messages = []
        for message in raw_prompt:
            if not isinstance(message, Mapping):
                raise TypeError(f"prompt messages must be mappings, got {type(message)!r}")
            messages.append(dict(message))
        return messages

    raise TypeError(f"unsupported raw_prompt type: {type(raw_prompt)!r}")


def extract_question_text(raw_prompt: Any) -> str:
    """Best-effort extraction of the question text (used for statistics only)."""
    normalized = normalize_raw_prompt(raw_prompt)
    if isinstance(normalized, str):
        return normalized.strip()
    if len(normalized) == 1:
        return str(normalized[0].get("content", "")).strip()
    return "\n\n".join(
        f"{str(m.get('role', 'user')).upper()}: {m.get('content', '')}" for m in normalized
    ).strip()


def canonicalize_raw_prompt(raw_prompt: Any) -> str:
    """Stable, order-preserving serialization of a chat prompt.

    Keys are sorted and separators are compact so that the same messages always
    hash to the same digest, regardless of dict insertion order or of whether
    the value came from parquet, JSON or a plain Python literal.
    """
    normalized = normalize_raw_prompt(raw_prompt)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_raw_prompt(raw_prompt: Any) -> str:
    return hashlib.sha256(canonicalize_raw_prompt(raw_prompt).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_teacher_fingerprint_payload(
    *,
    provider: str,
    model: str,
    base_url: str | None,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    num_samples: int,
    seed: int | None,
    extra_body: Mapping[str, Any] | None = None,
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION,
    engine: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything that must match for two teacher samplings to be comparable.

    ``engine`` carries backend-specific facts (in-process vLLM version, dtype,
    tensor-parallel size, ...). It is omitted entirely when absent, so a
    fingerprint produced by the HTTP path - and therefore every index digest
    written by earlier versions - stays bit-identical.
    """
    payload = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "num_samples": num_samples,
        "seed": seed,
        "extra_body": dict(extra_body) if extra_body else None,
        "prompt_template_version": prompt_template_version,
    }
    if engine:
        payload["engine"] = dict(engine)
    return payload


def hash_teacher_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


# --- answer formatting helpers (shared by eval data prep and reporting) ------

_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")


def extract_final_answer(text: str) -> str | None:
    """Last ``Answer: ...`` line of a completion, or ``None``."""
    matches = _ANSWER_LINE.findall(text or "")
    return matches[-1].strip() if matches else None


def has_answer_line(text: str) -> bool:
    return extract_final_answer(text) is not None


__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "TASK_INSTRUCTION_TEMPLATE",
    "build_content_messages",
    "build_teacher_fingerprint_payload",
    "build_user_message",
    "canonicalize_raw_prompt",
    "extract_final_answer",
    "extract_question_text",
    "has_answer_line",
    "hash_raw_prompt",
    "hash_teacher_fingerprint",
    "normalize_raw_prompt",
    "sha256_file",
    "sha256_text",
    "stable_json_dumps",
]
