"""math_unified_v1: visible explicit answer extraction and bounded typed grading.

MathD normalization and SymPy parsing derive from the pinned OPRD CPU subset.
Extraction never sees GT. Object delimiters, units and conditions are preserved.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import importlib.metadata
import json
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
DEFAULT_CONFIG = {'version': VERSION, 'entrypoint': str(Path(__file__).resolve()), 'timeout_seconds': 2.0}
MARKER = re.compile(r'(?im)^[ \t]*(?:(?:[-*+]|\d{1,3}[.)]|\#{1,6})[ \t]+|>[ \t]*)*(?:\*\*|__)?(?:final[ \t]+)?answer[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*')


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
    text = text.strip()
    # Markdown can wrap the label, the value, or both; tolerate a remaining edge.
    text = re.sub(r'^(?:(?:\*\*|__)\s*)+|(?:\s*(?:\*\*|__))+$', '', text).strip()
    for pair in (('**', '**'), ('__', '__'), ('`', '`'), ('\\[', '\\]'), ('\\(', '\\)'), ('$$', '$$'), ('$', '$')):
        if text.startswith(pair[0]) and text.endswith(pair[1]) and len(text) >= len(pair[0]) + len(pair[1]):
            text = text[len(pair[0]):-len(pair[1])].strip()
    return text.rstrip('.').strip() if text.endswith('.') and not text.endswith('...') else text


def _placeholder(value):
    return bool(re.fullmatch(r'(?is)(?:<[^>]+>|\[.*(?:answer|number).*\]|\$?answer(?:\$|\b.*))', value.strip()))


def _math_candidate(value):
    """Reject prose/templates without using GT or attempting mathematical proof."""
    if not value or _placeholder(value):return False
    words = re.findall(r'[A-Za-z]{2,}', re.sub(r'\\[A-Za-z]+', '', value))
    allowed = {'pi','sqrt','sin','cos','tan','log','ln','exp','zeta','abs','cm','mm','km','kg',
               'meter','meters','second','seconds','hour','hours','degree','degrees','or','and'}
    return all(word.lower() in allowed for word in words)


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
            status = 'placeholder' if _placeholder(value) else 'ok' if _math_candidate(value) else 'malformed'
        if status == 'ok' and not _math_candidate(value):status = 'malformed'
        candidates.append((marker.start(), end, value, status, method))
    candidates.extend(box for i, box in enumerate(boxes) if i not in covered)
    if not candidates:
        return {'pred': None, 'parse_status': 'no_answer', 'parse_method': 'none', 'candidate_count': 0, 'error_reason': 'no_explicit_final_answer'}
    # A quoted answer template cannot override an earlier real answer. An empty
    # or malformed final mathematical answer still masks, rather than falling back.
    actual = [c for c in candidates if c[3] != 'placeholder']
    chosen = max(actual or candidates, key=lambda item: item[0])
    if chosen[3] == 'placeholder':chosen = (*chosen[:2], None, 'malformed', chosen[4])
    if chosen[3] == 'ok' and not _math_candidate(chosen[2]):chosen = (*chosen[:2], None, 'malformed', chosen[4])
    return {'pred': chosen[2] if chosen[3] == 'ok' else None, 'parse_status': chosen[3], 'parse_method': chosen[4],
            'candidate_count': len(candidates), 'error_reason': None if chosen[3] == 'ok' else 'last_explicit_answer_' + chosen[3]}


def validate_ground_truth(value):
    if isinstance(value, bool) or value is None or not isinstance(value, (str, int, float)) or not str(value).strip():
        raise ValueError('missing or invalid math ground truth')
    value = str(value).strip()
    if value.lower() in ('nan', 'none', 'null', '[invalid]', 'invalid'):
        raise ValueError('missing or invalid math ground truth')
    if '\\boxed' in value or MARKER.search(value):
        extracted = extract_answer(value)
        if extracted['parse_status'] != 'ok':
            raise ValueError('ground truth has malformed/ambiguous answer marker')
        value = extracted['pred']
    return _clean(value)


@lru_cache(maxsize=16)
def identity(timeout_seconds=2.0):
    sources = {str(p.relative_to(ROOT / 'scoring')): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / 'scoring').rglob('*.py')) if 'tests' not in p.parts}
    dependencies = {name: importlib.metadata.version(name) for name in ('sympy', 'pylatexenc')}
    protocol = {'version': VERSION, 'timeout_seconds': float(timeout_seconds), 'sources': sources,
                'dependencies': dependencies, 'python': list(sys.version_info[:2]), 'numeric_tolerance': 'exact',
                'object_policy': 'preserve_units_delimiters_order_conditions', 'vendor_source':
                json.loads((ROOT / 'scoring/vendor/SOURCE.json').read_text())}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return {'version': VERSION, 'fingerprint': fingerprint, 'protocol': protocol}


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


def _structure(value):
    value = _clean(value).replace('\\left', '').replace('\\right', '').strip()
    quantity = re.fullmatch(r'(.+?)\s*(cm|mm|km|m|kg|g|meters?|seconds?|hours?|degrees?|\\text\{[^}]+\}|\^\{?\\circ\}?|%)', value)
    if quantity:
        unit = re.sub(r'\\text\{\s*([^}]*?)\s*\}', r'\1', quantity[2])
        unit = re.sub(r'\^\{?(\\circ)\}?', r'\1', unit)
        return 'quantity_' + re.sub(r'\s+', '', unit), _clean(quantity[1])
    if value.startswith('\\{') and value.endswith('\\}'):
        return 'set', value[2:-2]
    if value.startswith('{') and value.endswith('}'):
        return 'set', value[1:-1]
    if value[:1] in ('[', '(') and value[-1:] in (']', ')') and ',' in value:
        return 'sequence_' + value[0] + value[-1], value[1:-1]
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', value):return 'scalar', value.replace(',', '')
    if len(_split(value)) > 1:return 'sequence_untyped', value
    if any(s in value for s in ('=', '<', '>', '\\in', '\\le', '\\ge', '\\cup', ' or ', ' and ')):
        return 'condition', value
    return 'scalar', value


def _split(value):
    parts = []; start = 0; depth = 0
    for i, character in enumerate(value):
        if character in '({[': depth += 1
        elif character in ')}]': depth -= 1
        elif character == ',' and depth == 0:
            parts.append(value[start:i].strip());start = i + 1
    parts.append(value[start:].strip())
    return parts


def _grade(pred, gt):
    from scoring.vendor.oprd_math import grade_answer_mathd, _parse_latex, _sympy_parse
    import sympy
    pk, p = _structure(pred);gk, g = _structure(gt)
    if pk != gk:
        return False
    if pk == 'set' or pk.startswith('sequence_'):
        pp, gg = _split(p), _split(g)
        if len(pp) != len(gg):return False
        if pk == 'set':
            remaining = list(gg)
            for element in pp:
                match = next((i for i, target in enumerate(remaining) if _grade(element, target)), None)
                if match is None:return False
                remaining.pop(match)
            return True
        return all(_grade(a, b) for a, b in zip(pp, gg))
    # Both exact normalization and MathD remove spaces; never concatenate digit
    # groups such as "1 2" into 12 on either path.
    if re.search(r'(?<=[\d.])\s+(?=[\d.])', p) or re.search(r'(?<=[\d.])\s+(?=[\d.])', g):return False
    canonical = lambda s: re.sub(r'\s+', '', s).replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    if canonical(p) == canonical(g):return True
    if pk == 'condition':return False  # Domain/equation proof is outside this fixed grader.
    # Units and prose are retained. Never let MathD remove them to manufacture equality.
    def parse(value):
        value = _parse_latex(value) if '\\' in value else value
        value = value.replace('π', 'pi').replace('√', 'sqrt')
        if '__' in value or re.search(r'[\[\]{};:\'"\\]', value):
            raise ValueError('unsupported mathematical syntax')
        names = set(re.findall(r'[A-Za-z]+', value))
        if any(len(name) != 1 and name not in {'pi', 'sqrt', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'zeta', 'abs'} for name in names):
            raise ValueError('units/text require exact structural match')
        value = re.sub(r'\bi\b', 'I', value).replace('ln(', 'log(')
        value = re.sub(r'\be(?=\s*(?:\^|\*\*))', 'E', value)
        parsed = _sympy_parse(value)
        if not isinstance(parsed, sympy.Expr):raise ValueError('unsupported mathematical object')
        return parsed
    # MathD fast equality only for numeric syntax; no unit/variable stripping.
    if re.fullmatch(r'[0-9. /+*^()\-]+', p) and re.fullmatch(r'[0-9. /+*^()\-]+', g):
        if grade_answer_mathd(p, g):return True
    a, b = parse(p), parse(g)
    return bool(sympy.simplify(a - b) == 0)


def grade_pair(pred, gt, timeout_seconds=2.0):
    if threading.current_thread() is threading.main_thread():
        with _deadline(timeout_seconds):return _grade(pred, gt)
    # Ray or caller threads cannot install POSIX alarms. Isolate the same CPU grader.
    command = [sys.executable, str(Path(__file__).resolve()), '--grade-pair']
    completed = subprocess.run(command, input=json.dumps({'pred': pred, 'gt': gt, 'timeout': timeout_seconds}),
        text=True, capture_output=True, timeout=timeout_seconds + .25)
    row = json.loads(completed.stdout)
    if row.get('error') == 'timeout':raise TimeoutError('math grading deadline exceeded')
    if row.get('error'):raise ValueError(row['error'])
    return row['correct']


def score(response, ground_truth, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    if config['version'] != VERSION:raise ValueError('unsupported math scorer version')
    gt = validate_ground_truth(ground_truth)
    info = identity(float(config['timeout_seconds']))
    result = {'scorer_version': VERSION, 'scorer_fingerprint': info['fingerprint'], **extract_answer(response),
              'correct': None, 'grade_status': 'not_attempted'}
    if result['parse_status'] == 'no_answer':return result
    if result['parse_status'] != 'ok':return result
    try:
        correct = grade_pair(result['pred'], gt, float(config['timeout_seconds']))
        result.update(correct=correct, grade_status='correct' if correct else 'incorrect')
    except (TimeoutError, subprocess.TimeoutExpired) as exc:
        result.update(grade_status='timeout', error_reason=str(exc))
    except Exception as exc:
        result.update(grade_status='error', error_reason=f'{type(exc).__name__}: {exc}')
    return result


def correctness_label(result, prompt_truncated=False):
    if prompt_truncated:return {'value': None, 'valid': False, 'source': 'prompt_truncated'}
    if result['parse_status'] == 'no_answer':return {'value': 0, 'valid': True, 'source': VERSION}
    if result['grade_status'] in ('correct', 'incorrect'):
        return {'value': int(result['correct']), 'valid': True, 'source': VERSION}
    return {'value': None, 'valid': False, 'source': result['grade_status'] if result['grade_status'] != 'not_attempted' else result['parse_status']}


def evaluation_value(result):
    label = correctness_label(result)
    return label['value'] if label['valid'] else None


def compute_score(solution_str, ground_truth, **kwargs):
    """Thin legacy worker API; full public status remains in the returned row."""
    import os
    config = json.loads(os.environ.get('GAD_SCORING_MATH_CONFIG', '{}'))
    result = score(solution_str, ground_truth, config)
    acc = evaluation_value(result)
    return {**result, 'acc': acc, 'score': 2 * acc - 1 if acc is not None else None,
            'evaluation_valid': acc is not None}


if __name__ == '__main__' and '--grade-pair' in sys.argv:
    inputs = json.load(sys.stdin)
    try:
        with _deadline(float(inputs['timeout'])):
            value = _grade(inputs['pred'], inputs['gt'])
        print(json.dumps({'correct': value}))
    except TimeoutError:print(json.dumps({'error': 'timeout'}))
    except Exception as exc:print(json.dumps({'error': f'{type(exc).__name__}: {exc}'}))
