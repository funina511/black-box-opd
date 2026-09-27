# Vendored math scorer -- CARD-aligned.
#
# Source: ``CARD/verl/utils/reward_score/math_dapo.py`` in this workspace
# (CARD = the reference implementation used for the DAPO-Math-17k math track).
# Licence: Apache-2.0, Copyright 2024 Bytedance Ltd. and/or its affiliates,
#          Copyright 2022 EleutherAI and the HuggingFace Inc. team,
#          adapted from https://github.com/EleutherAI/lm-evaluation-harness
#
# It is vendored verbatim (minus unrelated helpers) so that the sampling /
# data-preparation package does not need CARD or ``verl`` on PYTHONPATH.
# Do not "improve" this file: the numbers reported by the evaluation tooling are
# only comparable with CARD while the implementation stays byte-equivalent.
"""CARD-aligned ``Answer:`` scorer for math responses.

``compute_score`` returns ``{"score": +/-1.0, "acc": 0/1, "pred": str}``.
Accuracy MUST use ``acc`` (or ``score > 0``); the mean of ``score`` is not an
accuracy.  Only the last 300 characters are inspected, matching CARD.
"""

from __future__ import annotations

import re
from typing import Optional

# Number of trailing characters CARD inspects (longest MATH-500 answer is 159).
TRAILING_WINDOW = 300

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = ANSWER_PATTERN
) -> tuple[bool, str]:
    """Check whether the solution matches the ground truth (Minerva criteria)."""
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> tuple[bool, Optional[str]]:
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict:
    """CARD-compatible score for one completion.

    Returns ``{"score": 1.0 or -1.0, "acc": 1 or 0, "pred": str}``.
    """
    solution_str = solution_str[-TRAILING_WINDOW:]

    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred,
    }


def extract_prediction(solution_str: str) -> str:
    """The normalized prediction the scorer would compare, without a ground truth."""
    match = re.findall(ANSWER_PATTERN, solution_str[-TRAILING_WINDOW:])
    if not match:
        return "[INVALID]"
    return normalize_final_answer(match[-1])


__all__ = [
    "ANSWER_PATTERN",
    "TRAILING_WINDOW",
    "compute_score",
    "extract_prediction",
    "is_correct_minerva",
    "is_correct_strict_box",
    "last_boxed_only_string",
    "normalize_final_answer",
    "remove_boxed",
    "verify",
]
