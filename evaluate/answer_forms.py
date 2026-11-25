"""Evaluation-only answer-form rules for Minerva / Olympiad-Bench (answer_forms_v1).

About 11% of these questions have ground truths that math_unified_v1 refuses to grade
(equations, lists of tuples, floor/ceil, sympy-reserved letters such as N/S/E).  This
module runs after the unified scorer (and after minerva_units_v1 on Minerva) and adds:

- answer lists: top-level ',', ';' and the word "and" separate items; items match unordered;
- tuples: "(a,b),(c,d)" is a list of ordered tuples, compared element by element;
- equations: "f(x)=2x", "k=1", "\\frac{dx}{dt}=kx-a" with a simple left side compare the
  right side; a bare right side or the same left side (function variable renamed) is accepted;
  "MR=SRMC" (word = word) compares the two words as an unordered pair;
- a simple "x = " / "f(t) = " prefix on the prediction is dropped when the truth is a value;
- symbolic fallback: floor/ceil/binom/factorial, reserved letters, subscripted symbols and
  arcsin/exp written with braces are parsed directly and compared at fixed probe points
  (integers when floor/ceil/binom/factorial occur, rationals otherwise).

Ground truths with arbitrary constants (C_1, \\pi k) or non-simple equation left sides stay
unresolved.  On ground truths the unified scorer already supports, the rules can only turn an
answer into correct (never correct -> wrong, never unresolved -> wrong); a malformed final
answer can only become correct.  Training labels and scoring/math.py are untouched.
"""
import cmath
import hashlib
import itertools
import keyword
from pathlib import Path
import random
import re
import threading

from evaluate import minerva
from scoring import math as math_scorer

VERSION = 'answer_forms_v1'
TASKS = frozenset(('Minerva', 'Olympiad-Bench'))
_GREEK = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta',
          'kappa', 'lambda', 'mu', 'nu', 'xi', 'rho', 'sigma', 'tau', 'phi', 'varphi', 'chi', 'psi', 'omega',
          'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Sigma', 'Phi', 'Psi', 'Omega')
_FUNCS = frozenset(('floor', 'ceiling', 'binomial', 'factorial', 'sqrt', 'exp', 'log', 'ln', 'sin', 'cos', 'tan',
                    'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'pi', 'oo'))
_NAME = r'(?:' + '|'.join(sorted(_GREEK, key=len, reverse=True)) + r'|[A-Za-z])'
_KNOWN = r'(?:' + '|'.join(sorted(set(_GREEK) | _FUNCS, key=len, reverse=True)) + r')'
_SUBSCRIPTS = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
_SUB = r'(?:_[A-Za-z0-9]+)?'
_FUNC_LHS = re.compile(r'(' + _NAME + _SUB + r')\((' + _NAME + _SUB + r')\)')
_SIMPLE_LHS = re.compile(_NAME + _SUB + r"(?:\(" + _NAME + _SUB + r"\))?|d" + _NAME + _SUB + '/d' + _NAME
                         + '|' + _NAME + _SUB + r"'+(?:\(" + _NAME + r"\))?")
_WORD = re.compile(r'[A-Za-z]{2,}')
_ARBITRARY = re.compile(r'C_\{?\d|\\pi\s*k\b|\bk\s*\\pi')
_COMPARATOR = re.compile(r'[<>≤≥≠]|\\(?:le|ge|leq|geq|neq|in|lt|gt)(?![A-Za-z])')


class _Undecided(Exception):
    pass


def identity():
    return {'version': VERSION, 'fingerprint': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_revision': '2026-10-26.answer-forms.1', 'base_scorer': math_scorer.identity()['fingerprint'],
            'minerva_postprocess': minerva.identity()['fingerprint']}


# ---------------------------------------------------------------------------
# Text structure
# ---------------------------------------------------------------------------

def _prep(text):
    t = str(text).replace('\\$', '').replace('$', ' ')
    t = re.sub(r'\\(?:q?quad|displaystyle|left|right|[Bb]igg?)(?![A-Za-z])', ' ', t)
    t = re.sub(r'\\(?:text|mathrm|textrm|mbox)\s*\{\s*(and|or)\s*\}', r' \1 ', t)
    t = math_scorer._strip_spacing_commands(t).replace('\\geqslant', '\\geq').replace('\\leqslant', '\\leq')
    return ' '.join(math_scorer._clean(' '.join(t.split())).split())


def _mask(text):
    """Text with every character inside brackets replaced, for top-level searches."""
    out, depth = [], 0
    for character in text:
        if character in ')]}':
            depth -= 1
        out.append(character if depth == 0 else '\0')
        if character in '([{':
            depth += 1
    return ''.join(out), depth == 0


def _items(text):
    t = _prep(text)
    if t.startswith('\\{') and t.endswith('\\}'):
        t = t[2:-2].strip()
    masked, balanced = _mask(t)
    if not balanced:
        return [t] if t else []
    parts, start = [], 0
    for match in re.finditer(r',|;|\band\b', masked):
        parts.append(t[start:match.start()]); start = match.end()
    parts.append(t[start:])
    return [part.strip() for part in parts if part.strip()]


def _tuple(text):
    t = text.strip()
    if len(t) < 2 or t[0] != '(' or t[-1] != ')':
        return None
    masked, balanced = _mask(t[1:-1])
    if not balanced or ',' not in masked:
        return None
    inner, parts, start = t[1:-1], [], 0
    for index, character in enumerate(masked):
        if character == ',':
            parts.append(inner[start:index].strip()); start = index + 1
    parts.append(inner[start:].strip())
    return parts if all(parts) else None


def _equation(text):
    masked, balanced = _mask(text)
    if not balanced or _COMPARATOR.search(masked):
        return None
    positions = [i for i, c in enumerate(masked) if c == '=' and masked[i - 1:i] not in ('!', '<', '>')
                 and masked[i + 1:i + 2] != '=']
    if len(positions) != 1:
        return None
    left, right = text[:positions[0]].strip(), text[positions[0] + 1:].strip()
    return (left, right) if left and right else None


def _canon_lhs(text):
    t = re.sub(r'\s+', '', str(text)).replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    t = math_scorer._replace_two_argument_command(t, 'frac', lambda a, b: a + '/' + b)
    t = re.sub(r"\^\{?\\prime\}?|\\prime", "'", t)
    t = re.sub(r'\\(?=[A-Za-z])', '', t)
    return t.replace('{', '').replace('}', '')


def _is_word(text):
    return bool(_WORD.fullmatch(_canon_lhs(text)))


def _rename(text, old, new):
    letter = lambda name: r'(?<![A-Za-z\\])' + re.escape(name) + r'(?![A-Za-z])'
    if re.search(letter(new), text):
        raise _Undecided('function variable rename collision')
    return re.sub(letter(old), new, text)


# ---------------------------------------------------------------------------
# Symbolic fallback
# ---------------------------------------------------------------------------

def _group(text, start):
    """(content, end) of a {...}/(...) group or a bare token starting at start (after spaces)."""
    while start < len(text) and text[start] == ' ':
        start += 1
    if start < len(text) and text[start] in '{(':
        opening, closing = text[start], '}' if text[start] == '{' else ')'
        depth = 0
        for index in range(start, len(text)):
            depth += (text[index] == opening) - (text[index] == closing)
            if depth == 0:
                return text[start + 1:index], index + 1
        raise _Undecided('unbalanced group')
    match = re.compile(r'\\?[\w.]+').match(text, start)
    if not match:
        raise _Undecided('missing group')
    return match.group(0), match.end()


def _log_base(text):
    """\\log_{b} x -> log((x),(b)) for braced, parenthesised or bare base/argument."""
    while True:
        match = re.search(r'\\log\s*_', text)
        if not match:
            return text
        base, end = _group(text, match.end())
        argument, end = _group(text, end)
        text = text[:match.start()] + ' log((' + argument + '),(' + base + '))' + text[end:]


def _repeat(text, function):
    for _ in range(8):
        before, text = text, function(text)
        if text == before:
            break
    return text


def _to_sympy(text):
    import sympy
    from sympy.parsing import sympy_parser as parser
    t = _prep(text).replace('np.', '').replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    t = re.sub(r'\\(sin|cos|tan)\s*\^\s*\{?\s*-1\s*\}?', r'\\a\1', t)
    t = re.sub(r'\\?arc(sin|cos|tan)', r'\\a\1', t)
    t = re.sub(r'\\(?:text|mathrm|rm|operatorname)\s*\{\s*([^{}]*?)\s*\}', lambda m: m.group(1).replace(' ', ''), t)
    for source, target in (('\\lfloor', ' floor('), ('\\rfloor', ')'), ('\\lceil', ' ceiling('), ('\\rceil', ')'),
                           ('⌊', ' floor('), ('⌋', ')'), ('⌈', ' ceiling('), ('⌉', ')')):
        t = t.replace(source, target)
    t = _log_base(t)
    t = _repeat(t, lambda s: math_scorer._replace_two_argument_command(s, 'binom', lambda a, b: ' binomial((' + a + '),(' + b + '))'))
    t = _repeat(t, lambda s: math_scorer._replace_two_argument_command(s, 'frac', lambda a, b: '((' + a + ')/(' + b + '))'))
    t = _repeat(t, lambda s: math_scorer._replace_one_argument_command(s, 'sqrt', lambda a: ' sqrt(' + a + ')'))
    for name in ('exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'ln', 'log'):
        t = _repeat(t, lambda s, n=name: math_scorer._replace_one_argument_command(s, n, lambda a: ' ' + n + '(' + a + ')'))
    t = re.sub(r'([A-Za-z])([₀₁₂₃₄₅₆₇₈₉]+)', lambda m: m.group(1) + '_' + m.group(2).translate(_SUBSCRIPTS), t)
    t = re.sub(r'_\{\s*\\?([A-Za-z0-9]+)\s*\}', r'_\1', t)
    t = re.sub(r'_\\([A-Za-z]+)', r'_\1', t)
    t = re.sub(r'\\pi(?![A-Za-z])', ' pi ', t)
    t = re.sub(r'\\(' + '|'.join(_GREEK) + r')(?![A-Za-z])', r' \1', t)
    # "\sin theta_w" -> "\sin(theta_w)" before the unified normaliser splits off the subscript.
    t = re.sub(r'(\\(?:a?(?:sin|cos|tan)|ln|log|exp))\s*([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+)', r'\1(\2)', t)
    t = math_scorer._normalize_math(t)
    if '\\' in t or not t:
        raise _Undecided('unconverted latex')
    t = t.replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')').replace('^', '**')
    t = re.sub(r'\s+!', '!', t)
    t = re.sub(r'(?<![A-Za-z_])(\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![\w(])', r'(\1*10**(\2))', t)
    # I(0), x(1.5): a value of a named quantity, not I*0.
    t = re.sub(r'(?<![A-Za-z_])([A-Za-z](?:_[A-Za-z0-9]+)?)\(\s*(-?[0-9.]+)\s*\)',
               lambda m: m.group(1) + '__at_' + m.group(2).replace('.', 'p').replace('-', 'm'), t)
    local = {'_EULER': sympy.E, '_IMAG': sympy.I, 'ln': sympy.log}

    def symbol(token, rest):
        if token in _FUNCS:
            return token
        if token == 'e' and rest.startswith('**'):
            return '_EULER'
        if token == 'i':
            return '_IMAG'
        safe = token + '_' if keyword.iskeyword(token) else token
        local[safe] = sympy.Symbol(token)
        return safe

    def name(match):
        token = match.group(0)
        rest = t[match.end():].lstrip()
        if '_' in token or token in _FUNCS or token in _GREEK or len(token) == 1:
            pieces = [token]
        else:
            # "isqrt", "kpi", "RC": split into known names and single letters.
            pieces = re.findall(_KNOWN + r'|[A-Za-z]|\d+', token)
        out = [symbol(p, rest if i == len(pieces) - 1 else '') if p[0].isalpha() else p for i, p in enumerate(pieces)]
        call = '*' if rest.startswith('(') and pieces[-1] not in _FUNCS else ''
        return ' '.join(out) + call
    t = re.sub(r'(?<![A-Za-z_])[A-Za-z][A-Za-z0-9_]*', name, t)
    transformations = parser.standard_transformations + (parser.implicit_multiplication, parser.implicit_application,
                                                         parser.function_exponentiation)
    try:
        expr = parser.parse_expr(t, local_dict=local, transformations=transformations)
    except Exception as exc:
        raise _Undecided('cannot parse: ' + str(exc))
    if not isinstance(expr, sympy.Expr):
        raise _Undecided('not an expression')
    return expr


def _points(symbols, integer):
    if not symbols:
        return [{}]
    import sympy
    if integer and len(symbols) == 1:
        return [{symbols[0]: v} for v in range(1, 31)]
    if integer and len(symbols) == 2:
        return [dict(zip(symbols, combo)) for combo in itertools.product(range(1, 13), repeat=2)]
    rng = random.Random(0)
    count = 150 if integer else 40
    draw = (lambda: rng.randint(1, 9)) if integer else (lambda: sympy.Rational(rng.randint(30, 300), 100))
    return [{symbol: draw() for symbol in symbols} for _ in range(count)]


def _probe(gt, pred):
    import sympy
    if _ARBITRARY.search(gt):
        raise _Undecided('ground truth has arbitrary constants')
    g, p = _to_sympy(gt), _to_sympy(pred)
    integer = any(e.has(sympy.floor, sympy.ceiling, sympy.binomial, sympy.factorial) for e in (g, p))
    points = _points(sorted(g.free_symbols | p.free_symbols, key=str), integer)
    valid = 0
    for subs in points:
        try:
            a, b = complex(g.subs(subs).evalf(20)), complex(p.subs(subs).evalf(20))
        except Exception:
            continue
        if not (cmath.isfinite(a) and cmath.isfinite(b)):
            continue
        valid += 1
        if abs(a - b) > 1e-12 * max(abs(a), abs(b)):
            return False
    if valid >= min(5, len(points)):
        return True
    raise _Undecided('too few valid probe points')


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _value(gt, pred):
    elements = _tuple(gt)
    if elements is not None:
        others = _tuple(pred)
        if others is None or len(others) != len(elements):
            return False
        undecided = None
        for a, b in zip(elements, others):
            try:
                if not _value(a, b):
                    return False
            except _Undecided as exc:
                undecided = exc
        if undecided:
            raise undecided
        return True
    try:
        if math_scorer.grade_pair(pred, gt):
            return True
        # The unified scorer also says False for algebra it classifies as text ("kx - a");
        # a probe can only overturn that when both sides parse as expressions.
        refused = False
    except math_scorer.PredInvalidSyntax:
        return False
    except (math_scorer.GroundTruthUnparseable, math_scorer.PredUnsupported):
        refused = True
    except Exception as exc:
        raise _Undecided(str(exc))
    try:
        if threading.current_thread() is threading.main_thread():
            with math_scorer._deadline(5.0):
                return _probe(gt, pred)
        return _probe(gt, pred)
    except (_Undecided, TimeoutError) as exc:
        if refused:
            raise _Undecided(str(exc))
        return False


def _item(gt, pred):
    gt_eq, pred_eq = _equation(gt), _equation(pred)
    if gt_eq:
        left, right = gt_eq
        if _is_word(left) and _is_word(right):
            return bool(pred_eq) and {_canon_lhs(x).casefold() for x in gt_eq} == {_canon_lhs(x).casefold() for x in pred_eq}
        gt_left = _canon_lhs(left)
        if not _SIMPLE_LHS.fullmatch(gt_left):
            raise _Undecided('equation with a non-simple left side')
        if not pred_eq:
            return _value(right, pred)
        pred_left, pred_right = _canon_lhs(pred_eq[0]), pred_eq[1]
        if pred_left != gt_left:
            gf, pf = _FUNC_LHS.fullmatch(gt_left), _FUNC_LHS.fullmatch(pred_left)
            if not (gf and pf and gf.group(1) == pf.group(1)):
                return False
            pred_right = _rename(pred_right, pf.group(2), gf.group(2))
        return _value(right, pred_right)
    if pred_eq and _SIMPLE_LHS.fullmatch(_canon_lhs(pred_eq[0])):
        return _value(gt, pred_eq[1])
    return _value(gt, pred)


def grade(gt, pred):
    """True/False, or raises _Undecided."""
    gt_items, pred_items = _items(gt), _items(pred)
    if not gt_items or not pred_items:
        raise _Undecided('empty answer')
    if len(gt_items) != len(pred_items):
        return False
    remaining, undecided = list(pred_items), None
    for gt_item in gt_items:
        for index, candidate in enumerate(remaining):
            try:
                if _item(gt_item, candidate):
                    remaining.pop(index)
                    break
            except _Undecided as exc:
                undecided = exc
        else:
            if undecided:
                raise undecided
            return False
    return True


def _value_supported(value):
    elements = _tuple(value)
    if elements is not None:
        return all(_value_supported(e) for e in elements)
    if math_scorer.ground_truth_semantics(value).get('supported'):
        return True
    if _ARBITRARY.search(value):
        return False
    try:
        _to_sympy(value)
        return True
    except Exception:
        return False


def _item_supported(item):
    equation = _equation(item)
    if not equation:
        return _value_supported(item)
    left, right = equation
    if _is_word(left) and _is_word(right):
        return True
    return bool(_SIMPLE_LHS.fullmatch(_canon_lhs(left))) and _value_supported(right)


def ground_truth_semantics(ground_truth):
    base = math_scorer.ground_truth_semantics(ground_truth)
    if base.get('supported'):
        return base
    try:
        clean = math_scorer._gt_target(ground_truth)
    except math_scorer.GroundTruthUnparseable:
        return base
    items = _items(clean)
    if items and all(_item_supported(item) for item in items):
        return {'kind': 'answer_forms', 'supported': True, 'text': clean, 'items': items}
    return base


# ---------------------------------------------------------------------------
# Scoring entry point
# ---------------------------------------------------------------------------

def _rule_relevant(pred):
    return bool(re.search(r'=|;|\band\b|\$|\\quad|\\text\s*\{\s*and', pred))


def score(response, ground_truth, config=None, task=None):
    base = (minerva.score if task in minerva.TASKS else math_scorer.score)(response, ground_truth, config)
    info = identity()
    base.update(answer_forms=VERSION, answer_forms_fingerprint=info['fingerprint'], answer_form_applied=False)
    status = base['grade_status']
    if status == 'correct':
        return base
    malformed = base['parse_status'] == 'malformed'
    if status == 'gt_unparseable':
        if not ground_truth_semantics(ground_truth).get('supported'):
            return base
        extracted = math_scorer.extract_answer(response)
        base.update(extracted, grade_status='not_attempted', correct=None)
        malformed = extracted['parse_status'] == 'malformed'
        if extracted['parse_status'] != 'ok' and not malformed:
            return base
        unresolved_gt = True
    elif status in ('incorrect', 'unsupported', 'invalid_format') or malformed:
        unresolved_gt = False
    else:
        return base
    pred = minerva._final_answer_text(response) if malformed else base.get('pred')
    if not pred or (status == 'incorrect' and not _rule_relevant(pred)):
        return base
    try:
        verdict = grade(math_scorer._gt_target(ground_truth), pred)
    except (_Undecided, math_scorer.GroundTruthUnparseable):
        verdict = None
    if verdict:
        base.update(pred=pred.strip(), parse_status='ok', correct=True, grade_status='correct', error_reason=None,
                    answer_form_applied=True)
    elif malformed:
        pass
    elif verdict is False and unresolved_gt:
        base.update(correct=False, grade_status='incorrect', error_reason=None, answer_form_applied=True)
    elif verdict is None and unresolved_gt:
        base.update(grade_status='unsupported', error_reason='pred_unsupported: answer_forms undecided',
                    answer_form_applied=True)
    return base
