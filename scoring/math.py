"""math_unified_v1: visible explicit answer extraction and bounded typed grading.

Candidate eval-fix scorer (E01-E07) for the isolated evaluation repair.  The
official scoring/math.py is NOT modified while the training experiment is still
running; eval_fix/rescore_fixed.py loads this file explicitly.

Semantics of protocol revision 2026-10-22.eval-fix.2:
- extraction never sees the ground truth and keeps the original marker rules;
- the ground truth is validated on its own before any prediction is compared,
  so data problems classify every rollout of a question identically;
- pure decimal integers keep the AIME leading-zero normalisation, but numeric
  subscripts/base notation are never stripped (52_8 != 52_9, x_1 != x_2);
- symbols/constants (x, pi) are graded mathematically; plain text answers use a
  case-insensitive exact comparison;
- a single branch never equals a two-valued +/- ground truth;
- TimeoutError and internal grader failures stay unresolved instead of becoming
  an incorrect label;
- identity() hashes the source that actually runs plus its vendor dependency.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).resolve().parent
_ROOT_DIR = _Path(__file__).resolve().parents[1]
if _SCRIPT_DIR != _ROOT_DIR:
    _sys.path[:] = [p for p in _sys.path if _Path(p or '.').resolve() != _SCRIPT_DIR]
if str(_ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_DIR))

from contextlib import contextmanager
from functools import lru_cache
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VERSION = 'math_unified_v1'
# Explicit semantic marker; the fingerprint still hashes the executed sources.
PROTOCOL_REVISION = '2026-10-22.eval-fix.3'
DEFAULT_CONFIG = {'version': VERSION, 'entrypoint': str(Path(__file__).resolve()), 'timeout_seconds': 2.0}
MARKER = re.compile(r'(?im)^[ \t]*(?:(?:[-*+]|\d{1,3}[.)]|\#{1,6})[ \t]+|>[ \t]*)*(?:\*\*|__)?(?:final[ \t]+)?answer[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*')


class GroundTruthUnparseable(ValueError):
    """The ground truth cannot be graded reliably (data or parser limitation)."""


class PredUnsupported(ValueError):
    """The prediction is plausibly valid but outside the supported grader."""


class PredInvalidSyntax(ValueError):
    """The prediction is definitely malformed; evaluation counts it as wrong."""


class _GraderFailure(RuntimeError):
    """Internal grading failure (tooling/fault injection); never becomes 0."""


# ---------------------------------------------------------------------------
# Extraction: unchanged marker priority, never looks at the ground truth
# ---------------------------------------------------------------------------

def _unbox(text, start):
    opening = re.match(r'\\boxed[ \t]*\{', text[start:])
    if not opening:
        return None, len(text), 'malformed'
    begin = start + opening.end(); depth = 1
    for i in range(begin, len(text)):
        if text[i] in '{}' and (i == 0 or text[i - 1] != '\\'):
            depth += 1 if text[i] == '{' else -1
            if depth == 0:
                value = text[begin:i].strip()
                return value or None, i + 1, 'ok' if value else 'malformed'
    return None, start + opening.end(), 'malformed'


def _clean(text):
    text = str(text).strip()
    # Markdown can wrap the label, the value, or both; tolerate a remaining edge.
    text = re.sub(r'^(?:(?:\*\*|__)\s*)+|(?:\s*(?:\*\*|__))+$', '', text).strip()
    tick = chr(96)
    for pair in (('**', '**'), ('__', '__'), (tick, tick), ('\\[', '\\]'), ('\\(', '\\)'), ('$$', '$$'), ('$', '$')):
        if text.startswith(pair[0]) and text.endswith(pair[1]) and len(text) >= len(pair[0]) + len(pair[1]):
            text = text[len(pair[0]):-len(pair[1])].strip()
    wrapper = re.fullmatch(r'\\(?:text|mathrm|operatorname)\{\s*([^{}]*?)\s*\}', text)
    if wrapper:
        text = wrapper.group(1).strip()
    return text.rstrip('.').strip() if text.endswith('.') and not text.endswith('...') else text


def _placeholder(value):
    return bool(re.fullmatch(r'(?is)(?:<[^>]+>|\[.*(?:answer|number).*\]|\$?answer(?:\$|\b.*))', str(value).strip()))


def _math_candidate(value):
    """Reject prose/templates without using GT or attempting mathematical proof."""
    if not value or _placeholder(value):
        return False
    words = re.findall(r'[A-Za-z]{2,}', re.sub(r'\\[A-Za-z]+', '', value))
    allowed = {'pi', 'sqrt', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'sinh', 'cosh',
               'tanh', 'log', 'ln', 'exp', 'zeta', 'abs', 'oo', 'inf', 'infinity', 'cm', 'mm', 'km', 'kg',
               'meter', 'meters', 'second', 'seconds', 'hour', 'hours', 'degree', 'degrees', 'or', 'and',
               'pmatrix', 'bmatrix', 'vmatrix', 'matrix'}
    return all(word.lower() in allowed for word in words)


_MATH_WORDS = frozenset({'pi', 'e', 'i', 'sqrt', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos',
                         'atan', 'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'zeta', 'abs', 'oo', 'inf', 'nan'})
_TEXT_ANSWER_RE = re.compile(r"[A-Za-z][A-Za-z .'\-]*")
# A text answer is a short label ("east", "Evelyn", "no solution"), not a
# sentence copied from the prompt/template.  The historical extraction contract
# keeps template placeholders masked, so prose must stay malformed.
_TEXT_ANSWER_STOPWORDS = frozenset({'should', 'would', 'could', 'must', 'is', 'are', 'was', 'were',
                                    'be', 'been', 'being', 'written', 'write', 'answer', 'number',
                                    'placeholder', 'template', 'here', 'there', 'your', 'please',
                                    'input', 'value', 'final', 'question', 'problem', 'blank'})


def _is_plain_text_answer(value):
    text = _clean(value)
    if not text or not _TEXT_ANSWER_RE.fullmatch(text):
        return False
    words = re.findall(r'[A-Za-z]+', text)
    if not words or all(len(word) <= 1 for word in words):
        return False
    if len(words) > 4 or len(text) > 40:
        return False
    lowered = {word.lower() for word in words}
    if lowered & _MATH_WORDS or lowered & _TEXT_ANSWER_STOPWORDS:
        return False
    return True


def _strip_spacing_commands(text):
    # latex spacing commands; "\ " must not match the second backslash of a
    # matrix row separator "\\ ".
    text = text.replace('\\!', '').replace('\\,', '').replace('\\;', '').replace('\\:', '')
    text = re.sub(r'(?<!\\)\\ ', '', text)
    return text.replace('~', '')


def _is_quantity_answer(value):
    try:
        return _quantity(_strip_spacing_commands(_clean(value))) is not None
    except Exception:
        return False


def extract_answer(response):
    text = str(response or '').replace('\r\n', '\n')
    boxes = []
    for match in re.finditer(r'\\boxed\b', text):
        if any(begin <= match.start() < end and status == 'ok' for begin, end, _, status, _ in boxes):
            continue
        value, end, status = _unbox(text, match.start())
        boxes.append((match.start(), end, value, status, 'boxed'))
    candidates = []
    covered = set()
    for marker in MARKER.finditer(text):
        line_end = text.find('\n', marker.end())
        line_end = len(text) if line_end < 0 else line_end
        value = _clean(text[marker.end():line_end]); method = 'answer_line'; end = line_end
        if not value:
            tail = re.search(r'[^\s][^\n]*', text[line_end:])
            if tail:
                begin = line_end + tail.start(); end = line_end + tail.end()
                if MARKER.match(text, begin):
                    candidates.append((marker.start(), line_end, None, 'malformed', method))
                    continue
                value = _clean(tail.group()); method = 'answer_next_line'
        inside = [(i, b) for i, b in enumerate(boxes) if marker.end() <= b[0] < end]
        if inside:
            covered.update(i for i, _ in inside)
            expressions = {b[2] for _, b in inside if b[3] == 'ok'}
            status = 'malformed' if any(b[3] != 'ok' for _, b in inside) else 'ambiguous' if len(expressions) > 1 else 'ok'
            value = inside[-1][1][2] if status == 'ok' else None
        else:
            status = 'placeholder' if _placeholder(value) else 'ok' if (_math_candidate(value) or _is_quantity_answer(value)) else 'malformed'
            if status == 'malformed' and _is_plain_text_answer(value):
                status = 'ok'
        if status == 'ok' and not _math_candidate(value) and not _is_plain_text_answer(value) and not _is_quantity_answer(value):
            status = 'malformed'
        candidates.append((marker.start(), end, value, status, method))
    candidates.extend(box for i, box in enumerate(boxes) if i not in covered)
    if not candidates:
        return {'pred': None, 'parse_status': 'no_answer', 'parse_method': 'none', 'candidate_count': 0,
                'error_reason': 'no_explicit_final_answer'}
    # A quoted answer template cannot override an earlier real answer. An empty
    # or malformed final mathematical answer still masks, rather than falling back.
    actual = [c for c in candidates if c[3] != 'placeholder']
    chosen = max(actual or candidates, key=lambda item: item[0])
    if chosen[3] == 'placeholder':
        chosen = (chosen[0], chosen[1], None, 'malformed', chosen[4])
    if chosen[3] == 'ok' and not _math_candidate(chosen[2] or '') and not _is_plain_text_answer(chosen[2]) and not _is_quantity_answer(chosen[2] or ''):
        chosen = (chosen[0], chosen[1], None, 'malformed', chosen[4])
    return {'pred': chosen[2] if chosen[3] == 'ok' else None, 'parse_status': chosen[3], 'parse_method': chosen[4],
            'candidate_count': len(candidates), 'error_reason': None if chosen[3] == 'ok' else 'last_explicit_answer_' + chosen[3]}


# ---------------------------------------------------------------------------
# Normalisation: display-only rewrites; never deletes object structure
# ---------------------------------------------------------------------------

_SUPERSCRIPT_TABLE = str.maketrans({'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5',
                                    '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-', '⁺': '+',
                                    '⁼': '=', '⁽': '(', '⁾': ')', 'ⁿ': 'n'})
_SUPERSCRIPT_CHARS = '⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁼⁽⁾ⁿ'
_UNICODE_FIXES = (('⁄', '/'), ('∕', '/'), ('×', '*'), ('⋅', '*'), ('·', '*'), ('−', '-'),
                  ('–', '-'), ('—', '-'), ('％', '%'))
_TRIG_FUNCTIONS = ('sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan',
                   'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'sqrt')
_ALLOWED_NAMES = frozenset({'pi', 'sqrt', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos',
                            'atan', 'sinh', 'cosh', 'tanh', 'log', 'ln', 'exp', 'zeta', 'abs', 'oo', 'E', 'I', 'inf'})


def _expand_superscripts(value):
    if not any(character in value for character in _SUPERSCRIPT_CHARS):
        return value

    def replace(match):
        body = ''.join(match.group(0).translate(_SUPERSCRIPT_TABLE).split())
        return '^(' + body + ')'
    return re.sub('[' + _SUPERSCRIPT_CHARS + ']+', replace, value)


def _brace_group(text, start):
    """Return (content, end_index) for a balanced brace group starting at start."""
    if start >= len(text) or text[start] != '{':
        raise ValueError('expected latex group')
    depth = 0
    for index in range(start, len(text)):
        character = text[index]
        if character == '{' and (index == 0 or text[index - 1] != '\\'):
            depth += 1
        elif character == '}' and (index == 0 or text[index - 1] != '\\'):
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    raise ValueError('unbalanced latex group')


def _replace_two_argument_command(text, command, builder):
    out = []
    index = 0
    pattern = '\\' + command
    while True:
        found = text.find(pattern, index)
        if found < 0:
            out.append(text[index:])
            return ''.join(out)
        end = found + len(pattern)
        if end < len(text) and text[end].isalpha():
            out.append(text[index:end])
            index = end
            continue
        out.append(text[index:found])
        try:
            first, after_first = _brace_group(text, end)
        except ValueError:
            out.append(pattern)
            index = end
            continue
        if after_first < len(text) and text[after_first] == '{':
            try:
                second, after_second = _brace_group(text, after_first)
            except ValueError:
                out.append(pattern)
                index = end
                continue
        else:
            token = re.match(r'[+-]?\d+(?:\.\d+)?', text[after_first:])
            if token is None:
                out.append(pattern)
                index = end
                continue
            second = token.group(0)
            after_second = after_first + token.end()
        out.append(builder(first, second))
        index = after_second


def _replace_one_argument_command(text, command, builder):
    out = []
    index = 0
    pattern = '\\' + command
    while True:
        found = text.find(pattern, index)
        if found < 0:
            out.append(text[index:])
            return ''.join(out)
        end = found + len(pattern)
        if end < len(text) and text[end].isalpha():
            out.append(text[index:end])
            index = end
            continue
        out.append(text[index:found])
        try:
            first, after_first = _brace_group(text, end)
        except ValueError:
            out.append(pattern)
            index = end
            continue
        out.append(builder(first))
        index = after_first


def _latex_prepare(value):
    """Deterministic LaTeX rewrite for the structures this fixed grader supports.

    Matrices/vectors are deliberately NOT flattened here; _value_structure keeps
    their shape.  Anything still containing a backslash falls back to the
    vendored latex2text path and then to an unsupported classification.
    """
    text = str(value)
    text = text.replace('\\left', '').replace('\\right', '')
    for name in ('dfrac', 'tfrac'):
        text = text.replace('\\' + name, '\\frac')
    text = _replace_two_argument_command(text, 'frac', lambda a, b: '((' + a + ')/(' + b + '))')
    text = _replace_one_argument_command(text, 'text', lambda a: a)
    text = _replace_one_argument_command(text, 'mathrm', lambda a: a)
    text = _replace_one_argument_command(text, 'operatorname', lambda a: a)
    text = re.sub(r'\^\{([^{}]*)\}', r'^(\1)', text)
    text = re.sub(r'\\sqrt\s*([0-9A-Za-z]+)', r'sqrt(\1)', text)
    text = re.sub(r'\\sqrt\[([^\]]+)\]\{([^{}]*)\}', r'((\2)**(1/(\1)))', text)
    text = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', text)
    text = text.replace('\\pi', 'pi').replace('\\cdot', '*').replace('\\times', '*')
    text = text.replace('\\div', '/').replace('\\infty', 'oo').replace('\\neq', '!=')
    text = text.replace('\\ge', '>=').replace('\\le', '<=').replace('\\geq', '>=').replace('\\leq', '<=')
    text = text.replace('\\pm', '±').replace('\\mp', '∓')
    text = text.replace('\\%', '%')
    text = re.sub(r'\\[A-Za-z]+', lambda match: match.group(0)[1:] if match.group(0)[1:] in _TRIG_FUNCTIONS else match.group(0), text)
    for source, target in _UNICODE_FIXES:
        text = text.replace(source, target)
    # function application: "cot x" -> "cot(x)"
    text = re.sub(r'\b(' + '|'.join(_TRIG_FUNCTIONS) + r')\s+([A-Za-z0-9]+)', r'\1(\2)', text)
    return re.sub(r'\s+', ' ', text).strip()


def _normalize_math(value):
    text = _strip_spacing_commands(_clean(str(value)))
    text = text.replace('\\$', '').replace('\\%', '%')
    compact = re.sub(r'\s+', '', text)
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', compact):
        text = compact.replace(',', '')
    text = re.sub(r'\^\s*\{?\s*\\circ\s*\}?', '', text)   # 90^{circ} -> 90 (angle marker)
    text = text.replace('\\circ', '').replace('°', '')
    # mixed numbers: "137 1/2" and "137 \frac{1}{2}" are both 137 + 1/2
    text = re.sub(r'^(\d+)\s*\\frac\{(\d+)\}\{(\d+)\}$', r'(((\1)*(\3)+(\2))/(\3))', text)
    text = re.sub(r'^(\d+)\s+(\d+)\s*/\s*(\d+)$', r'(((\1)*(\3)+(\2))/(\3))', text)
    text = _expand_superscripts(text)
    text = _latex_prepare(text)
    for source, target in _UNICODE_FIXES:
        text = text.replace(source, target)
    text = re.sub(r'\binfinity\b', 'oo', text)
    text = re.sub(r'\binf\b', 'oo', text)
    option = re.fullmatch(r'\(\s*([A-Za-z])\s*\)', text)   # multiple-choice option: (C) -> C
    if option:
        text = option.group(1)
    text = text.strip()
    while len(text) > 1 and text.startswith('$'):
        text = text[1:].strip()
    while len(text) > 1 and text.endswith('$'):
        text = text[:-1].strip()
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', text):
        text = text.replace(',', '')
    # AIME format: a pure decimal integer may carry leading zero padding.
    # This must never touch subscripts/base notation or non-integer syntax.
    match = re.fullmatch(r'([+-]?)0+(\d+)', text)
    if match:
        text = match.group(1) + match.group(2)
    elif re.fullmatch(r'[+-]?0+', text):
        text = '0'
    return text


# ---------------------------------------------------------------------------
# Typed ground-truth model
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = r'[+-]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s+\d+\s*/\s*\d+|\s*/\s*\d+)?'
_FRACTION_PATTERN = r'\\frac\{[^{}]+\}\{[^{}]+\}'
_MATRIX_RE = re.compile(r'\\begin\{(?P<env>[pbv]?matrix)\}(?P<body>.*?)\\end\{(?P=env)\}', re.S)
_QUANTITY_TEXT_RE = re.compile(r'(?P<num>.+?)\s*\\(?:text|mathrm)\{\s*(?P<unit>[^{}]+?)\s*\}', re.S)
_QUANTITY_CURRENCY_RE = re.compile(r'\\?\$\s*(?P<num>' + _NUMBER_PATTERN + r')')
_QUANTITY_UNIT_RE = re.compile(r'(?P<num>' + _NUMBER_PATTERN + r')\s*(?P<unit>cm|mm|km|kg|mg|m|g|meters?|seconds?|hours?|degrees?|deg|\\text\{[^{}]+\}|\\%|%|°|\^\{?\\circ\}?)')
_QUANTITY_WORD_RE = re.compile(r'(?P<num>' + _NUMBER_PATTERN + r'|' + _FRACTION_PATTERN + r')\s+(?P<unit>[A-Za-z]{2,})')
_QUANTITY_GENERIC_RE = re.compile(r'(?P<num>.+?\d.*?)\s+(?P<unit>[A-Za-z]{2,})', re.S)
_BASE_NOTATION_RE = re.compile(r'(?<![\w.])\d+_\{?\d+\}?')
_CONDITION_RE = re.compile(r'=|[<>]|\\(?:in|neq|leq|geq|le|ge|cup|mid)(?![A-Za-z])|\bor\b|\band\b')
_UNIT_SYNONYMS = {'meter': 'm', 'meters': 'm', 'second': 's', 'seconds': 's', 'hour': 'h', 'hours': 'h',
                  'minute': 'min', 'minutes': 'min', 'degree': 'degree', 'degrees': 'degree', 'deg': 'degree',
                  'pct': 'percent', 'dollar': 'dollar', 'dollars': 'dollar', 'usd': 'dollar'}


def _has_base_notation(text):
    return bool(_BASE_NOTATION_RE.search(text))


def _normalize_unit(unit):
    text = str(unit).strip()
    text = re.sub(r'\s+', '', text)
    if text in ('°', 'degree', 'degrees', 'deg', '^\\circ', '^{\\circ}', '\\circ'):
        return 'degree'
    if text in ('%', '\\%', 'percent', 'pct'):
        return 'percent'
    if text in ('$', '\\$', 'dollar', 'dollars', 'usd'):
        return 'dollar'
    low = text.lower()
    return _UNIT_SYNONYMS.get(low, low)


def _quantity(text):
    value = str(text).strip()
    match = _QUANTITY_TEXT_RE.fullmatch(value)
    if match and match.group('num').strip():
        return {'unit': _normalize_unit(match.group('unit')), 'numeric': match.group('num').strip()}
    match = _QUANTITY_CURRENCY_RE.fullmatch(value)
    if match:
        return {'unit': 'dollar', 'numeric': match.group('num')}
    match = _QUANTITY_UNIT_RE.fullmatch(value)
    if match:
        return {'unit': _normalize_unit(match.group('unit')), 'numeric': match.group('num')}
    match = _QUANTITY_WORD_RE.fullmatch(value)
    if match and match.group('unit').lower() not in _MATH_WORDS:
        return {'unit': _normalize_unit(match.group('unit')), 'numeric': match.group('num')}
    match = _QUANTITY_GENERIC_RE.fullmatch(value)
    if match and match.group('unit').lower() not in _MATH_WORDS:
        return {'unit': _normalize_unit(match.group('unit')), 'numeric': match.group('num')}
    return None


def _is_set(text):
    if text.startswith('\\{') and text.endswith('\\}'):
        return True
    return text.startswith('{') and text.endswith('}') and len(text) >= 2


def _is_sequence(text):
    return len(text) >= 2 and text[0] in '([' and text[-1] in ')]' and ',' in text


def _pm_branches(text):
    normalised = text.replace('\\pm', '±').replace('\\mp', '∓')
    count = normalised.count('±') + normalised.count('∓')
    if count == 0:
        return None
    if normalised.count('±') > 1 or normalised.count('∓') > 1:
        return {'unsupported': 'multiple +/- signs are not a supported two-value answer'}
    branches = []
    for sign in ('+', '-'):
        branch = normalised.replace('±', sign).replace('∓', '-' if sign == '+' else '+')
        branches.append(branch.strip())
    return branches


def _normalize_pm_branch(text):
    norm = _normalize_math(text)
    if norm.startswith('+'):
        norm = norm[1:].lstrip()
    return norm


def _split(value):
    parts = []; start = 0; depth = 0
    for i, character in enumerate(value):
        if character in '({[':
            depth += 1
        elif character in ')}]':
            depth -= 1
        elif character == ',' and depth == 0:
            parts.append(value[start:i].strip()); start = i + 1
    parts.append(value[start:].strip())
    return parts


def _is_condition(text):
    return bool(_CONDITION_RE.search(text))


def _base_number(text):
    match = re.fullmatch(r'(?P<digits>\d+)_\{?(?P<base>\d{1,2})\}?', text)
    if not match:
        return None
    digits = match.group('digits')
    base = int(match.group('base'))
    if not 2 <= base <= 36:
        return {'invalid': 'base ' + str(base) + ' is outside 2..36'}
    if any(int(character) >= base for character in digits):
        return {'invalid': 'digits are not valid in base ' + str(base)}
    return {'digits': digits, 'base': base}


def _base_digits_canonical(digits):
    stripped = digits.lstrip('0')
    return stripped or '0'


def _base_value(digits, base):
    value = 0
    for character in digits:
        value = value * base + int(character)
    return value


def _expand_pm_elements(elements):
    expanded = []
    for element in elements:
        structure = _value_structure(element)
        if structure['kind'] == 'pm':
            expanded.extend(structure['branches'])
        else:
            expanded.append(element)
    return expanded


def _split_union(text):
    parts = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        character = text[index]
        if character in '([{':
            depth += 1
        elif character in ')]}':
            depth -= 1
        if depth == 0 and text.startswith('\\cup', index):
            parts.append(text[start:index].strip())
            index += 4
            start = index
            continue
        if depth == 0 and character == '∪':
            parts.append(text[start:index].strip())
            index += 1
            start = index
            continue
        index += 1
    if not parts:
        return None
    parts.append(text[start:].strip())
    if any(not part for part in parts):
        return None
    return parts


def _value_structure(value):
    """Structural classification only; this function never compares anything."""
    text = _strip_spacing_commands(_clean(str(value)))
    text = re.sub(r'\$([^$]+)\$', r'\1', text).strip()
    text = text.replace('\\left', '').replace('\\right', '').strip()
    if not text:
        return {'kind': 'invalid', 'text': ''}
    matrix = _MATRIX_RE.fullmatch(text)
    if matrix:
        rows = []
        for row in re.split(r'\\\\', matrix.group('body').strip()):
            if row.strip():
                rows.append([_clean(cell) for cell in row.split('&')])
        return {'kind': 'matrix', 'rows': rows, 'text': text}
    base_number = _base_number(text)
    if base_number is not None:
        if 'invalid' in base_number:
            return {'kind': 'unsupported', 'reason': base_number['invalid'], 'text': text}
        return {'kind': 'base_number', 'digits': base_number['digits'], 'base': base_number['base'], 'text': text}
    union_parts = _split_union(text)
    if union_parts is not None:
        return {'kind': 'union', 'parts': union_parts, 'text': text}
    if _is_set(text):
        inner = text[2:-2] if text.startswith('\\{') else text[1:-1]
        return {'kind': 'set', 'elements': _split(inner), 'text': text}
    if _is_sequence(text):
        return {'kind': 'sequence', 'open': text[0], 'close': text[-1], 'elements': _split(text[1:-1]), 'text': text}
    pm = _pm_branches(text)
    if isinstance(pm, dict):
        return {'kind': 'unsupported', 'reason': pm['unsupported'], 'text': text}
    if pm:
        return {'kind': 'pm', 'branches': pm, 'text': text}
    quantity = _quantity(text)
    if quantity is not None:
        return {'kind': 'quantity', 'unit': quantity['unit'], 'numeric': quantity['numeric'], 'text': text}
    compact = re.sub(r'\s+', '', text)
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', compact):
        return {'kind': 'scalar', 'text': compact.replace(',', '')}
    if len(_split(text)) > 1:
        return {'kind': 'list', 'elements': _split(text), 'text': text}
    if _is_plain_text_answer(text):
        return {'kind': 'text', 'text': text}
    if _is_condition(text):
        return {'kind': 'condition', 'text': text}
    return {'kind': 'scalar', 'text': text}


# ---------------------------------------------------------------------------
# Source identity
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def identity(timeout_seconds=2.0):
    source_roles = {
        'math.py': Path(__file__).resolve(),
        'scoring/__init__.py': ROOT / 'scoring' / '__init__.py',
        'vendor/__init__.py': ROOT / 'scoring' / 'vendor' / '__init__.py',
        'vendor/oprd_math.py': ROOT / 'scoring' / 'vendor' / 'oprd_math.py',
        'vendor/SOURCE.json': ROOT / 'scoring' / 'vendor' / 'SOURCE.json',
    }
    sources = {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in sorted(source_roles.items())}
    dependencies = {name: importlib.metadata.version(name) for name in ('sympy', 'pylatexenc')}
    protocol = {'version': VERSION, 'protocol_revision': PROTOCOL_REVISION, 'timeout_seconds': float(timeout_seconds),
                'sources': sources, 'dependencies': dependencies, 'python': list(sys.version_info[:2]),
                'numeric_tolerance': 'exact', 'object_policy': 'preserve_units_delimiters_order_conditions',
                'extraction_policy': 'explicit_final_answer_only',
                'ground_truth_policy': 'validated_before_prediction',
                'evaluation_mapping': 'format_failures_zero_unresolved_excluded',
                'training_label_policy': 'unresolved_masked_format_failures_masked',
                'vendor_source': json.loads((ROOT / 'scoring' / 'vendor' / 'SOURCE.json').read_text())}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return {'version': VERSION, 'fingerprint': fingerprint, 'protocol': protocol}


# ---------------------------------------------------------------------------
# Scalar parsing helpers
# ---------------------------------------------------------------------------

def _parse_expression(value):
    fault = os.environ.get('EVAL_FIX_FAULT')
    if fault == 'timeout':
        raise TimeoutError('injected grading deadline')
    if fault == 'runtime':
        raise _GraderFailure('injected internal grading failure')
    from scoring.vendor.oprd_math import _sympy_parse
    return _sympy_parse(value)


def _role_error(role, message):
    if role == 'gt':
        return GroundTruthUnparseable(message)
    return PredUnsupported(message)


def _clearly_invalid(text):
    stripped = re.sub(r'\\[{}]', '', text)
    for opening, closing in (('(', ')'), ('[', ']'), ('{', '}')):
        if stripped.count(opening) != stripped.count(closing):
            return True
    if re.search(r'[?#]', text):
        return True
    if re.fullmatch(r'[+\-*/^=.,;:]+', text.strip()):
        return True
    if re.search(r'(?:\*\*|[*+/^=+-])\s*$', text.strip()):
        return True
    return False


def _parse_scalar_value(normalised, role):
    text = normalised
    if _has_base_notation(text):
        raise _role_error(role, 'base notation such as 52_8 is not supported')
    if _clearly_invalid(text):
        if role == 'pred':
            raise PredInvalidSyntax('malformed final answer syntax')
        raise GroundTruthUnparseable('malformed ground truth syntax')
    value = text.replace('π', 'pi').replace('√', 'sqrt')
    if '\\' in value:
        try:
            from scoring.vendor.oprd_math import _parse_latex
            value = _parse_latex(value)
        except Exception as exc:
            raise _role_error(role, 'latex conversion failed: ' + str(exc))
    if '__' in value or re.search(r'[\[\]{};:\'"\\]', value):
        raise _role_error(role, 'unsupported mathematical syntax')
    names = set(re.findall(r'[A-Za-z]+', value))
    if any(len(name) != 1 and name not in _ALLOWED_NAMES for name in names):
        raise _role_error(role, 'units/text require exact structural match')
    value = re.sub(r'\bi\b', 'I', value).replace('ln(', 'log(')
    value = re.sub(r'\be(?=\s*(?:\^|\*\*))', 'E', value)
    try:
        parsed = _parse_expression(value)
    except (TimeoutError, _GraderFailure):
        raise
    except Exception as exc:
        raise _role_error(role, 'cannot parse ' + repr(value) + ': ' + str(exc))
    import sympy
    if not isinstance(parsed, sympy.Expr):
        raise _role_error(role, 'unsupported mathematical object')
    return parsed


@lru_cache(maxsize=1024)
def _parse_scalar_cached(normalised):
    return _parse_scalar_value(normalised, 'gt')


# ---------------------------------------------------------------------------
# Ground-truth semantics: prediction independent, cached per cleaned GT text
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def _gt_semantics(text):
    structure = _value_structure(text)
    kind = structure['kind']
    result = dict(structure)
    if kind == 'invalid':
        result.update(supported=False, reason='empty or invalid ground truth')
        return result
    if kind == 'unsupported':
        result.update(supported=False, reason=structure.get('reason', 'unsupported ground truth structure'))
        return result
    if kind == 'condition':
        result.update(supported=False, reason='condition/equation ground truth is outside the fixed grader')
        return result
    if kind == 'list':
        elements = _expand_pm_elements(structure['elements'])
        if not elements:
            result.update(supported=False, reason='empty answer-list ground truth')
            return result
        for element in elements:
            sub = _gt_semantics(_clean(element))
            if not sub.get('supported') or sub['kind'] not in ('scalar', 'text'):
                result.update(supported=False, reason='unsupported answer-list element: ' + repr(element))
                return result
            if sub['kind'] == 'scalar':
                try:
                    _parse_scalar_cached(sub['normalized'])
                except (GroundTruthUnparseable, PredUnsupported, PredInvalidSyntax) as exc:
                    result.update(supported=False, reason='unparseable answer-list element: ' + repr(element) + ': ' + str(exc))
                    return result
        result['elements'] = elements
        result.update(supported=True)
        return result
    if kind == 'text':
        result.update(supported=True)
        return result
    if kind == 'base_number':
        result.update(supported=True, value=_base_value(structure['digits'], structure['base']))
        return result
    if kind == 'scalar':
        try:
            result['normalized'] = _normalize_math(structure['text'])
            _parse_scalar_cached(result['normalized'])
        except (GroundTruthUnparseable, PredUnsupported, PredInvalidSyntax) as exc:
            result.update(supported=False, reason='unparseable ground truth: ' + str(exc))
            return result
        result.update(supported=True)
        return result
    if kind == 'quantity':
        try:
            result['numeric'] = _normalize_math(structure['numeric'])
            _parse_scalar_cached(result['numeric'])
        except (GroundTruthUnparseable, PredUnsupported, PredInvalidSyntax) as exc:
            result.update(supported=False, reason='unparseable quantity ground truth: ' + str(exc))
            return result
        if not result.get('unit'):
            result.update(supported=False, reason='empty unit in ground truth')
            return result
        result.update(supported=True)
        return result
    if kind == 'pm':
        branches = []
        for branch in structure['branches']:
            normalised = _normalize_pm_branch(branch)
            try:
                _parse_scalar_cached(normalised)
            except (GroundTruthUnparseable, PredUnsupported, PredInvalidSyntax) as exc:
                result.update(supported=False, reason='unsupported +/- branch ' + repr(branch) + ': ' + str(exc))
                return result
            branches.append({'text': branch, 'normalized': normalised})
        result['branches'] = branches
        result.update(supported=True)
        return result
    if kind == 'union':
        parts = structure['parts']
        for part in parts:
            sub = _gt_semantics(_clean(part))
            if not sub.get('supported') or sub['kind'] not in ('scalar', 'set', 'sequence', 'base_number'):
                result.update(supported=False, reason='unsupported union part: ' + repr(part))
                return result
        result['parts'] = parts
        result.update(supported=True)
        return result
    if kind in ('set', 'sequence'):
        elements = structure['elements']
        if kind == 'set':
            elements = _expand_pm_elements(elements)
        if not elements:
            result.update(supported=False, reason='empty ' + kind + ' ground truth')
            return result
        for element in elements:
            sub = _gt_semantics(_clean(element))
            if not sub.get('supported') or sub['kind'] not in ('scalar', 'text'):
                result.update(supported=False, reason='unsupported ' + kind + ' element: ' + repr(element))
                return result
            if sub['kind'] == 'scalar':
                try:
                    _parse_scalar_cached(sub['normalized'])
                except (GroundTruthUnparseable, PredUnsupported, PredInvalidSyntax) as exc:
                    result.update(supported=False, reason='unparseable ' + kind + ' element: ' + repr(element) + ': ' + str(exc))
                    return result
        result['elements'] = elements
        result.update(supported=True)
        return result
    if kind == 'matrix':
        rows = structure['rows']
        if not rows:
            result.update(supported=False, reason='empty matrix ground truth')
            return result
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            result.update(supported=False, reason='ragged matrix ground truth')
            return result
        for row in rows:
            for cell in row:
                sub = _gt_semantics(_clean(cell))
                if not sub.get('supported') or sub['kind'] not in ('scalar', 'text'):
                    result.update(supported=False, reason='unsupported matrix cell: ' + repr(cell))
                    return result
                if sub['kind'] == 'scalar':
                    try:
                        _parse_scalar_cached(sub['normalized'])
                    except (GroundTruthUnparseable, PredUnsupported, PredInvalidSyntax) as exc:
                        result.update(supported=False, reason='unparseable matrix cell: ' + repr(cell) + ': ' + str(exc))
                        return result
        result['shape'] = [len(rows), widths.pop()]
        result.update(supported=True)
        return result
    result.update(supported=False, reason='unsupported ground truth kind: ' + kind)
    return result


def _gt_target(value):
    if isinstance(value, bool) or value is None or not isinstance(value, (str, int, float)):
        raise GroundTruthUnparseable('missing or invalid math ground truth type')
    raw = str(value).strip()
    if not raw or raw.lower() in ('nan', 'none', 'null', '[invalid]', 'invalid', '?'):
        raise GroundTruthUnparseable('missing or invalid math ground truth')
    if '\\boxed' in raw or MARKER.search(raw):
        extracted = extract_answer(raw)
        if extracted['parse_status'] != 'ok':
            raise GroundTruthUnparseable('ground truth has malformed/ambiguous answer marker')
        raw = extracted['pred']
    return _clean(raw)


def ground_truth_semantics(value):
    """JSON-safe, prediction-independent ground-truth audit record."""
    try:
        clean = _gt_target(value)
    except GroundTruthUnparseable as exc:
        return {'kind': 'invalid', 'supported': False, 'reason': str(exc), 'text': ''}
    return copy.deepcopy(_gt_semantics(clean))


def validate_ground_truth(value):
    """Lightweight historical contract (kept for preflight compatibility).

    Full gradeability is checked by score() before any prediction comparison and
    is exposed by ground_truth_semantics(); requiring it here would reject
    otherwise valid training data that the evaluation grader does not support.
    """
    return _gt_target(value)


def require_gradeable_ground_truth(value):
    clean = _gt_target(value)
    semantics = _gt_semantics(clean)
    if not semantics.get('supported'):
        raise GroundTruthUnparseable(semantics.get('reason', 'unsupported ground truth'))
    return clean


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _digit_gap(text):
    return bool(re.search(r'(?<=[\d.])\s+(?=[\d.])', text))


def _canonical_text(text):
    return re.sub(r'\s+', '', text).replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')


def _scalar_compare_texts(gt_text, pred_text):
    gt_norm = _normalize_math(gt_text)
    pred_norm = _normalize_math(pred_text)
    if _digit_gap(gt_norm) or _digit_gap(pred_norm):
        return False
    if _canonical_text(gt_norm) == _canonical_text(pred_norm):
        return True
    if re.fullmatch(r'[0-9. /+*^()\-]+', gt_norm) and re.fullmatch(r'[0-9. /+*^()\-]+', pred_norm):
        try:
            from scoring.vendor.oprd_math import grade_answer_mathd
            if grade_answer_mathd(pred_norm, gt_norm):
                return True
        except (TimeoutError, _GraderFailure):
            raise
        except Exception:
            pass
    gt_parsed = _parse_scalar_cached(gt_norm)
    pred_parsed = _parse_scalar_value(pred_norm, 'pred')
    import sympy
    try:
        return bool(sympy.simplify(gt_parsed - pred_parsed) == 0)
    except (TimeoutError, _GraderFailure):
        raise
    except GroundTruthUnparseable:
        raise
    except Exception as exc:
        raise PredUnsupported('cannot compare expressions: ' + str(exc))


def _element_compare(gt_element, pred_element):
    gt_sem = _gt_semantics(_clean(str(gt_element)))
    if not gt_sem.get('supported'):
        raise GroundTruthUnparseable(gt_sem.get('reason', 'unsupported ground-truth element'))
    return _compare(gt_sem, _value_structure(pred_element))


def _compare_unordered(gt_elements, pred_elements):
    if len(gt_elements) != len(pred_elements):
        return False
    remaining = list(pred_elements)
    pending_error = None
    for element in gt_elements:
        matched = False
        for index, candidate in enumerate(remaining):
            try:
                if _element_compare(element, candidate):
                    remaining.pop(index)
                    matched = True
                    break
            except (PredUnsupported, PredInvalidSyntax) as exc:
                pending_error = exc
        if not matched:
            if pending_error is not None:
                raise pending_error
            return False
    return True


def _compare_pm(gt, pred):
    if pred['kind'] == 'pm':
        gt_branches = [branch['normalized'] for branch in gt['branches']]
        pred_branches = [_normalize_pm_branch(branch) for branch in pred['branches']]
        if len(pred_branches) != len(gt_branches):
            return False
        remaining = list(pred_branches)
        for branch in gt_branches:
            for index, candidate in enumerate(remaining):
                if _scalar_compare_texts(branch, candidate):
                    remaining.pop(index)
                    break
            else:
                return False
        return True
    if pred['kind'] in ('set', 'sequence', 'list'):
        elements = _expand_pm_elements(pred['elements']) if pred['kind'] == 'set' else list(pred['elements'])
        if len(elements) != len(gt['branches']):
            return False
        remaining = list(elements)
        for branch in gt['branches']:
            for index, candidate in enumerate(remaining):
                if _element_compare(branch['text'], candidate):
                    remaining.pop(index)
                    break
            else:
                return False
        return True
    return False


def _compare_quantity(gt, pred):
    if pred['kind'] == 'quantity':
        if pred['unit'] != gt['unit']:
            return False
        return _scalar_compare_texts(gt['numeric'], pred['numeric'])
    if pred['kind'] == 'scalar' and gt['unit'] == 'degree':
        # 90 (degree measure) is the established answer form for 90^\circ GTs.
        return _scalar_compare_texts(gt['numeric'], pred['text'])
    return False


def _compare_base(gt, pred):
    gt_digits = _base_digits_canonical(gt['digits'])
    if pred['kind'] == 'base_number':
        if pred['base'] != gt['base']:
            return False
        return _base_digits_canonical(pred['digits']) == gt_digits
    if pred['kind'] == 'scalar':
        # The question demanded this base; an un-subscripted digit string in the
        # same digits is the conventional answer form, not a cross-base equality.
        pred_norm = _normalize_math(pred['text'])
        if re.fullmatch(r'\d+', pred_norm):
            return _base_digits_canonical(pred_norm) == gt_digits
        return False
    return False


def _compare_scalar_to_base(gt, pred):
    gt_norm = _normalize_math(gt['text'])
    gt_parsed = _parse_scalar_cached(gt_norm)
    import sympy
    try:
        return bool(sympy.simplify(gt_parsed - sympy.Integer(_base_value(pred['digits'], pred['base']))) == 0)
    except (TimeoutError, _GraderFailure):
        raise
    except GroundTruthUnparseable:
        raise
    except Exception as exc:
        raise PredUnsupported('cannot compare expressions: ' + str(exc))


def _compare(gt, pred):
    pk = pred['kind']
    if pk == 'invalid':
        raise PredInvalidSyntax('empty final answer')
    if pk == 'unsupported':
        raise PredUnsupported(pred.get('reason', 'unsupported prediction structure'))
    gk = gt['kind']
    if gk == 'text':
        if pk != 'text':
            return False
        return ' '.join(gt['text'].split()).casefold() == ' '.join(pred['text'].split()).casefold()
    if gk == 'pm':
        return _compare_pm(gt, pred)
    if gk == 'quantity':
        return _compare_quantity(gt, pred)
    if gk == 'base_number':
        return _compare_base(gt, pred)
    if gk in ('set', 'list'):
        if pk not in ('set', 'list'):
            return False
        pred_elements = _expand_pm_elements(pred['elements']) if pk == 'set' else list(pred['elements'])
        return _compare_unordered(gt['elements'], pred_elements)
    if gk == 'union':
        if pk != 'union':
            return False
        gt_parts = gt['parts']
        pred_parts = pred['parts']
        if len(gt_parts) != len(pred_parts):
            return False
        remaining = list(pred_parts)
        for part in gt_parts:
            for index, candidate in enumerate(remaining):
                gt_sem = _gt_semantics(_clean(part))
                if _compare(gt_sem, _value_structure(candidate)):
                    remaining.pop(index)
                    break
            else:
                return False
        return True
    if gk == 'sequence':
        if pk != 'sequence':
            return False
        if gt['open'] != pred['open'] or gt['close'] != pred['close']:
            return False
        if len(gt['elements']) != len(pred['elements']):
            return False
        return all(_element_compare(a, b) for a, b in zip(gt['elements'], pred['elements']))
    if gk == 'matrix':
        if pk != 'matrix':
            return False
        if len(gt['rows']) != len(pred['rows']):
            return False
        for gt_row, pred_row in zip(gt['rows'], pred['rows']):
            if len(gt_row) != len(pred_row):
                return False
            if not all(_element_compare(a, b) for a, b in zip(gt_row, pred_row)):
                return False
        return True
    if gk == 'scalar':
        if pk == 'scalar':
            return _scalar_compare_texts(gt['text'], pred['text'])
        if pk == 'base_number':
            return _compare_scalar_to_base(gt, pred)
        return False
    raise GroundTruthUnparseable('unsupported ground truth kind: ' + str(gk))


def _grade(pred, gt):
    gt_sem = _gt_semantics(_clean(str(gt)))
    if not gt_sem.get('supported'):
        raise GroundTruthUnparseable(gt_sem.get('reason', 'unsupported ground truth'))
    return _compare(gt_sem, _value_structure(pred))


# ---------------------------------------------------------------------------
# Thread-safe entry point (POSIX alarms only exist on the main thread)
# ---------------------------------------------------------------------------

@contextmanager
def _deadline(seconds):
    handler = signal.getsignal(signal.SIGALRM)
    previous = signal.getitimer(signal.ITIMER_REAL)

    def expired(*args):
        raise TimeoutError('math grading deadline exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
        signal.signal(signal.SIGALRM, handler)


def grade_pair(pred, gt, timeout_seconds=2.0):
    if threading.current_thread() is threading.main_thread():
        with _deadline(timeout_seconds):
            return _grade(pred, gt)
    # Ray or caller threads cannot install POSIX alarms. Isolate the same CPU
    # grader. run_path keeps this file's directory off sys.path[0] so a sibling
    # file named math.py cannot shadow the standard library math module.
    bridge = 'import runpy,sys;runpy.run_path(sys.argv[1],run_name="__main__")'
    command = [sys.executable, '-c', bridge, str(Path(__file__).resolve()), '--grade-pair']
    completed = subprocess.run(command, input=json.dumps({'pred': pred, 'gt': gt, 'timeout': timeout_seconds}),
                               text=True, capture_output=True, timeout=timeout_seconds + 0.5)
    if not completed.stdout.strip():
        raise _GraderFailure('grading subprocess produced no output (rc=' + str(completed.returncode) + '): ' + completed.stderr.strip()[:400])
    row = json.loads(completed.stdout)
    error = row.get('error')
    text = str(error or '')
    if error == 'timeout' or text.startswith('TimeoutError'):
        raise TimeoutError('math grading deadline exceeded')
    if text.startswith('PredInvalidSyntax'):
        raise PredInvalidSyntax(text)
    if text.startswith('PredUnsupported'):
        raise PredUnsupported(text)
    if text.startswith('GroundTruthUnparseable'):
        raise GroundTruthUnparseable(text)
    if error:
        raise _GraderFailure(text)
    return bool(row['correct'])


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def score(response, ground_truth, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    if config['version'] != VERSION:
        raise ValueError('unsupported math scorer version')
    result = {'scorer_version': VERSION, 'scorer_fingerprint': identity(float(config['timeout_seconds']))['fingerprint'],
              'pred': None, 'parse_status': 'not_attempted', 'parse_method': 'none', 'candidate_count': 0,
              'error_reason': None, 'correct': None, 'grade_status': 'not_attempted'}
    # 1. Validate the ground truth on its own, before looking at the prediction.
    try:
        clean_gt = _gt_target(ground_truth)
        gt_sem = _gt_semantics(clean_gt)
        if not gt_sem.get('supported'):
            raise GroundTruthUnparseable(gt_sem.get('reason', 'unsupported ground truth'))
    except GroundTruthUnparseable as exc:
        result.update(grade_status='gt_unparseable', error_reason='gt_unparseable: ' + str(exc))
        return result
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        result.update(grade_status='timeout', error_reason=str(exc))
        return result
    except _GraderFailure as exc:
        result.update(grade_status='error', error_reason='internal_error: ' + str(exc))
        return result
    except Exception as exc:
        result.update(grade_status='error', error_reason='internal_error: ' + type(exc).__name__ + ': ' + str(exc))
        return result
    # 2. Extract the model answer (never uses the GT).
    result.update(extract_answer(response))
    if result['parse_status'] != 'ok':
        return result
    # 3. Grade.
    try:
        correct = grade_pair(result['pred'], clean_gt, float(config['timeout_seconds']))
        result.update(correct=bool(correct), grade_status='correct' if correct else 'incorrect')
    except PredInvalidSyntax as exc:
        # Evaluation counts a definite format failure as wrong (evaluation_value);
        # the training label stays masked like the historical format-failure path.
        result.update(correct=False, grade_status='invalid_format', error_reason='pred_invalid_syntax: ' + str(exc))
    except PredUnsupported as exc:
        result.update(grade_status='unsupported', error_reason='pred_unsupported: ' + str(exc))
    except GroundTruthUnparseable as exc:
        result.update(grade_status='gt_unparseable', error_reason='gt_unparseable: ' + str(exc))
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        result.update(grade_status='timeout', error_reason=str(exc))
    except _GraderFailure as exc:
        result.update(grade_status='error', error_reason='internal_error: ' + str(exc))
    except Exception as exc:
        result.update(grade_status='error', error_reason='internal_error: ' + type(exc).__name__ + ': ' + str(exc))
    return result


def evaluation_status(result):
    """Evaluation-only classification; training labels use correctness_label."""
    status = result.get('grade_status')
    if status == 'gt_unparseable':
        return 'gt_unparseable'
    if status == 'unsupported':
        return 'unsupported'
    if status == 'timeout':
        return 'timeout'
    if status in ('error', 'internal_error'):
        return 'internal_error'
    if status == 'correct':
        return 'correct'
    if status == 'incorrect':
        return 'incorrect'
    if status == 'invalid_format':
        return 'format_failure'
    parse = result.get('parse_status')
    if parse == 'no_answer':
        return 'no_answer'
    if parse in ('malformed', 'ambiguous', 'placeholder'):
        return 'format_failure'
    return 'internal_error'


def evaluation_value(result):
    status = evaluation_status(result)
    if status == 'correct':
        return 1
    if status in ('incorrect', 'no_answer', 'format_failure'):
        return 0
    return None


def correctness_label(result, prompt_truncated=False):
    """Historical training-label mapping; unchanged from scoring/math.py."""
    if prompt_truncated:
        return {'value': None, 'valid': False, 'source': 'prompt_truncated'}
    if result['parse_status'] == 'no_answer':
        return {'value': 0, 'valid': True, 'source': VERSION}
    if result['grade_status'] in ('correct', 'incorrect'):
        return {'value': int(result['correct']), 'valid': True, 'source': VERSION}
    return {'value': None, 'valid': False,
            'source': result['grade_status'] if result['grade_status'] != 'not_attempted' else result['parse_status']}


def compute_score(solution_str, ground_truth, **kwargs):
    """Thin legacy worker API.

    acc keeps the historical training-label mapping (None means unresolved or
    masked) so training consumers are not changed; the evaluation-only mapping
    is exposed separately for eval consumers.
    """
    config = json.loads(os.environ.get('GAD_SCORING_MATH_CONFIG', '{}'))
    result = score(solution_str, ground_truth, config)
    label = correctness_label(result)
    acc = label['value'] if label['valid'] else None
    return {**result, 'acc': acc, 'score': 2 * acc - 1 if acc is not None else None,
            'evaluation_valid': acc is not None,
            'evaluation_status': evaluation_status(result),
            'evaluation_value': evaluation_value(result)}


if __name__ == '__main__' and '--grade-pair' in sys.argv:
    payload = json.load(sys.stdin)
    try:
        with _deadline(float(payload['timeout'])):
            value = _grade(payload['pred'], payload['gt'])
        print(json.dumps({'correct': value}))
    except TimeoutError:
        print(json.dumps({'error': 'timeout'}))
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}))
