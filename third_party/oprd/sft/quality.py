from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class QualityResult:
    valid: bool
    reason: str
    has_boxed_answer: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def has_boxed(text: str) -> bool:
    return "\\boxed" in text


def detect_repeated_lines(text: str, min_length: int = 20, threshold: int = 5) -> bool:
    counts: dict[str, int] = {}
    for line in text.splitlines():
        normalized = line.strip()
        if len(normalized) < min_length:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
        if counts[normalized] >= threshold:
            return True
    return False


def detect_ngram_repetition(text: str, width: int = 100, threshold: int = 3) -> bool:
    if len(text) < width * threshold:
        return False
    counts: dict[str, int] = {}
    for start in range(0, len(text) - width + 1, 10):
        block = text[start : start + width]
        counts[block] = counts.get(block, 0) + 1
        if counts[block] >= threshold:
            return True
    return False


def detect_consecutive_repetition(text: str, width: int = 50, threshold: int = 3) -> bool:
    if len(text) < width * threshold:
        return False
    for start in range(len(text) - width * threshold + 1):
        block = text[start : start + width]
        if all(
            text[start + repeat * width : start + (repeat + 1) * width] == block
            for repeat in range(1, threshold)
        ):
            return True
    return False


def validate_output(text: str, *, require_boxed: bool = True) -> QualityResult:
    stripped = text.strip()
    boxed = has_boxed(stripped)
    if not stripped:
        return QualityResult(False, "empty", boxed)
    if require_boxed and not boxed:
        return QualityResult(False, "no_boxed", boxed)
    if detect_repeated_lines(stripped):
        return QualityResult(False, "repeated_lines", boxed)
    if detect_ngram_repetition(stripped):
        return QualityResult(False, "ngram_repetition", boxed)
    if len(stripped) > 5000 and detect_consecutive_repetition(stripped):
        return QualityResult(False, "consecutive_repetition", boxed)
    return QualityResult(True, "ok", boxed)


@lru_cache(maxsize=1)
def _load_math_grader() -> tuple[Callable[[str], str], Callable[[str, str], bool], Callable[[str, str], bool]]:
    repo_root = Path(__file__).resolve().parents[1]
    verl_root = repo_root / "verl"
    if str(verl_root) not in sys.path:
        sys.path.insert(0, str(verl_root))
    try:
        from verl.utils.reward_score.ttrl_math.math_utils import (
            extract_boxed_answer,
            grade_answer_mathd,
            grade_answer_sympy,
        )
    except Exception as error:
        raise RuntimeError(
            "Cannot import the verl math grader. Run dataset preparation in the OPRD/verl environment "
            "with its math dependencies installed."
        ) from error
    return extract_boxed_answer, grade_answer_mathd, grade_answer_sympy


def extract_candidate_answer(text: str) -> str | None:
    if "\\boxed" in text:
        extract_boxed_answer, _, _ = _load_math_grader()
        boxed = extract_boxed_answer(text)
        if boxed:
            return boxed

    for line in reversed(text.splitlines()):
        candidate = line.strip()
        match = re.match(r"(?i)^(?:final\s+)?answer\s*:\s*(.+)$", candidate)
        if match:
            return match.group(1).strip()
    return None


def grade_output(text: str, ground_truth: str | None) -> bool | None:
    if ground_truth is None or str(ground_truth).strip() == "":
        return None
    prediction = extract_candidate_answer(text)
    if prediction is None:
        return False
    extract_boxed_answer, grade_answer_mathd, grade_answer_sympy = _load_math_grader()
    normalized_ground_truth = str(ground_truth)
    if "\\boxed" in normalized_ground_truth:
        extracted_ground_truth = extract_boxed_answer(normalized_ground_truth)
        if extracted_ground_truth:
            normalized_ground_truth = extracted_ground_truth
    try:
        return bool(
            grade_answer_mathd(prediction, normalized_ground_truth)
            or grade_answer_sympy(prediction, normalized_ground_truth)
        )
    except Exception:
        return False
