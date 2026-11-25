"""Evaluation-only, explicit final-option grading; no mathematical equivalence."""
import hashlib
from pathlib import Path
import re

VERSION = 'choice_exact_v2'
TASKS = frozenset(('GPQA', 'GPQA-Diamond', 'MMLU-Pro-Math'))
_BOX = re.compile(r'\\boxed\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')
# "Answer: X" style labels at line start (markdown list/heading/bold allowed).
_LABEL = re.compile(r'(?im)^[ \t]*(?:(?:[-*+]|\#{1,6})[ \t]+|>[ \t]*)*(?:\*\*|__)?(?:the[ \t]+)?'
                    r'(?:final[ \t]+|correct[ \t]+)?(?:answer|option|choice)\b[ \t]*(?:\*\*|__)?[ \t]*'
                    r':[ \t]*(?:\*\*|__)?[ \t]*(.*)$')
# "(the) (final/correct/best) answer is X", in a sentence or at line start. Used only when the
# value starts with an option letter, so "the answer is not A" or "The answer is a bit tricky"
# cannot mask another answer; a value naming several options ("A and B", "A or B") is kept as
# an ambiguous final declaration.
_SENTENCE = re.compile(r'(?im)(?:\bthe[ \t]+|^[ \t]*(?:\*\*|__)?)(?:final[ \t]+|correct[ \t]+|best[ \t]+|right[ \t]+)?'
                       r'(?:answer|option|choice)[ \t]+is[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*(.*)$')
_DECLARED_LETTER = re.compile(r'\s*(?:\*\*)?\$?\s*(?:\(\s*([A-Za-z])\s*\)|\[\s*([A-Za-z])\s*\]|([A-Za-z]))(?![A-Za-z])'
                              r'\$?(?:\*\*)?')
_REASON = re.compile(r'\s*(?:[,;:]\s*|\s+)(?:because|since)\b.*', re.S | re.I)
_ANOTHER_OPTION = re.compile(r'\s*(?:,|/|&|\band\b|\bor\b|和|或者?)\s*(?:\*\*)?\$?\s*[(\[]?\s*[A-Za-z]\s*[)\]]?(?![A-Za-z])', re.I)
_DELIMITER_LINE = re.compile(r'\s*(?:\$\$|\\\[|\\\]|-{3,})\s*')
# A heading/label with nothing after it; the next non-empty line holds the answer.
_HEADING = re.compile(r'(?im)^[ \t]*(?:\#{1,6}[ \t]+)?(?:\*\*|__)?(?:the[ \t]+)?(?:final[ \t]+)?answer'
                      r'[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*$')
_WRAPPER = re.compile(r'\\(?:text|mathrm|textbf|mathbf|textit|mathit|textrm|mbox|bf|rm|boxed)\s*\{(.*)\}', re.S)
_LETTER_WITH_TEXT = re.compile(r'\(?\s*([A-Za-z])\s*\)?\s*(?:[.:)]\s*|\s+)(.+)', re.S)


def identity():
    return {'version': VERSION, 'fingerprint': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_revision': '2026-10-24.choice.2'}


def options(row, task):
    count = (row.get('extra_info') or {}).get('option_num')
    if count is None and task in ('GPQA', 'GPQA-Diamond'):
        count = 4
    if type(count) is not int or not 2 <= count <= 26:
        raise ValueError('multiple-choice sample lacks valid extra_info.option_num')
    return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]


def _unwrap(text):
    previous = None
    while previous != text:
        previous = text
        text = text.strip().replace('**', '').replace('__', '').strip()
        text = re.sub(r'^\$+|\$+$', '', text).strip()
        text = re.sub(r'^\\[(\[]|\\[)\]]$', '', text).strip()
        match = _WRAPPER.fullmatch(text)
        if match:
            text = match.group(1)
        if text.endswith(('.', '!', '。')) and not text.endswith('...'):
            text = text[:-1]
    return text.strip()


def _letter(text):
    match = re.fullmatch(r'\(\s*([A-Za-z])\s*\)|\[\s*([A-Za-z])\s*\]|([A-Za-z])\s*\)?', _unwrap(text))
    return next(group for group in match.groups() if group) if match else None


def _norm(text):
    text = _unwrap(text.replace('\\$', '$'))
    text = re.sub(r'\\left|\\right|\\[,;! ]|\\dfrac|\\tfrac', lambda m: '\\frac' if 'frac' in m.group() else '', text)
    text = re.sub(r'[\s$]|\\text\s*\{|\\mathrm\s*\{', '', text).replace('{', '').replace('}', '').casefold()
    number = text.replace(',', '')
    try:
        return repr(float(number))
    except ValueError:
        return text


def _content_match(text, choices):
    """Unique option whose text equals the candidate after display normalisation."""
    if not choices:
        return None
    target = _norm(text)
    hits = [i for i, option in enumerate(choices) if target and _norm(option) == target]
    return 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[hits[0]] if len(hits) == 1 else None


def _parse_letter(candidate, choices):
    """(letter, method suffix) for a bare letter or a letter followed by its own option text."""
    letter = _letter(candidate)
    if letter:
        return letter, ''
    match = _LETTER_WITH_TEXT.fullmatch(_unwrap(candidate))
    if match and choices:
        index = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.find(match.group(1).upper())
        if 0 <= index < len(choices) and _norm(match.group(2)) == _norm(choices[index]):
            return match.group(1), '_with_option_text'
    return None, ''


def _declared(text, choices):
    """('ok', letter, suffix), ('ambiguous', None, '') or (None, None, '') for a declared value."""
    letter, suffix = _parse_letter(text, choices)
    if letter:
        return 'ok', letter, suffix
    match = _DECLARED_LETTER.match(text)
    if match:
        rest = text[match.end():]
        if _ANOTHER_OPTION.match(rest):
            return 'ambiguous', None, ''
        # "B, since ..." / "B because ...": a single letter followed only by its justification.
        if _REASON.fullmatch(rest):
            return 'ok', next(group for group in match.groups() if group), '_with_reason'
    return None, None, ''


def _candidates(response, choices):
    found = [(m.start(), m.group(1), 'boxed') for m in _BOX.finditer(response)]
    found += [(m.start(1), m.group(1), 'answer') for m in _LABEL.finditer(response) if m.group(1).strip()]
    for m in _SENTENCE.finditer(response):
        if _declared(m.group(1), choices)[0]:
            found.append((m.start(1), m.group(1), 'sentence'))
    for m in _HEADING.finditer(response):
        lines = [line for line in re.finditer(r'[^\n]+', response[m.end():])
                 if line.group().strip() and not _DELIMITER_LINE.fullmatch(line.group())]
        if lines:
            found.append((m.end() + lines[0].start(), lines[0].group(), 'heading'))
    return found


def extract(response, choices=None, lowercase_ambiguous=False):
    """Return (letter, status, method, candidate_text); only explicit final declarations count."""
    response = response.rsplit('</think>', 1)[-1].strip()
    found = _candidates(response, choices)
    if not found and _letter(response) is None:
        return None, 'no_answer', 'no_explicit_final_option', None
    start, candidate, method = max(found, key=lambda item: item[0]) if found else (0, response, 'standalone')
    line_start = response.rfind('\n', 0, start) + 1
    line_end = response.find('\n', start)
    line = response[line_start:len(response) if line_end < 0 else line_end]
    boxes = list(_BOX.finditer(line))
    if boxes and re.search(r'(?i)\bor\b|或者|或', line):
        return None, 'ambiguous', 'multiple_final_options', candidate
    if len(boxes) > 1:
        letters = [_letter(m.group(1)) for m in boxes]
        if None in letters or len({letter.upper() for letter in letters}) > 1:
            return None, 'ambiguous', 'conflicting_final_options', candidate
    status, letter, suffix = _declared(candidate, choices) if method != 'boxed' else ('ok', *_parse_letter(candidate, choices))
    if status == 'ambiguous':
        return None, 'ambiguous', 'multiple_final_options', candidate
    if letter is None:
        return None, 'malformed', 'final_answer_is_not_one_option', candidate
    method += suffix
    if letter.islower() and lowercase_ambiguous:
        # The stem labels its own items a), b), ...; a lowercase letter may name one of those.
        return None, 'ambiguous', 'lowercase_letter_ambiguous', candidate
    return letter.upper(), 'ok', method, candidate


def score(row, task):
    info = identity()
    result = {'scorer_version': VERSION, 'scorer_fingerprint': info['fingerprint'],
              'pred': None, 'correct': None, 'candidate_count': 0,
              'parse_status': 'not_attempted', 'parse_method': 'choice_exact',
              'grade_status': 'not_attempted', 'error_reason': None}
    extra = row.get('extra_info') or {}
    try:
        allowed = options(row, task)
        truth = str(row.get('ground_truth', '')).strip().upper()
        if len(truth) != 1 or truth not in allowed:
            raise ValueError('ground truth is outside the actual option set')
        choices = extra.get('options')
        if choices is not None and (not isinstance(choices, list) or len(choices) != len(allowed)
                                    or not all(isinstance(c, str) for c in choices)):
            raise ValueError('extra_info.options does not match the option count')
    except ValueError as exc:
        return {**result, 'grade_status': 'gt_unparseable', 'error_reason': str(exc),
                'evaluation_status': 'gt_invalid', 'evaluation_value': None}
    pred, status, method, candidate = extract(row.get('response', ''), choices,
                                              bool(extra.get('lowercase_labels_in_stem')))
    if pred is not None and pred not in allowed:
        pred, status, method = None, 'malformed', 'prediction_outside_option_set'
    correct = pred == truth if status == 'ok' else False
    # Diagnostic only: a final answer written as an option's text instead of its letter.
    content = _content_match(candidate, choices) if status != 'ok' and candidate else None
    return {**result, 'pred': pred, 'parse_status': status, 'parse_method': method,
            'candidate_count': int(pred is not None), 'correct': correct,
            'grade_status': 'correct' if correct else 'incorrect',
            'error_reason': None if status == 'ok' else method,
            'evaluation_status': ('correct' if correct else 'incorrect') if status == 'ok' else
                                 ('no_answer' if status == 'no_answer' else 'format_failure'),
            'evaluation_value': int(correct),
            'content_option': content, 'content_option_correct': None if content is None else content == truth}
