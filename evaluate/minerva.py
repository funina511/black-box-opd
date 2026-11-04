"""Evaluation-only Minerva rule: a trailing physical unit does not change a unitless numeric answer.

Minerva (OCW) ground truths are plain numbers while questions say "(in cm)" etc. The unified math
scorer keeps units, so "1.6 cm" against "1.6" is wrong. Here a final answer that is exactly a number
followed by unit tokens is re-graded on the number alone. Nothing else changes; training labels and
scoring.math are untouched.
"""
import hashlib
from pathlib import Path
import re

from scoring import math as math_scorer

VERSION = 'minerva_units_v1'
TASKS = frozenset(('Minerva',))
_NUMBER = (r'[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)'
           r'(?:\s*(?:[eE][-+]?\d+|(?:\\times|\\cdot|\*|×)\s*10\s*\^\s*(?:\{\s*[-+]?\d+\s*\}|[-+]?\d+)))?')
# SI prefixes on common base units, plus spelled-out and astronomy units used in OCW answers.
_UNIT_WORD = (r'(?:[yzafpnuµμmcdhkMGTPE]?(?:m|s|g|K|eV|J|W|Hz|Pa|N|mol|L|C|V|A|T|F|Ohm|ohm|erg|cal|bar|pc|ly|Jy|sr|rad)'
              r'|yr|yrs|years?|days?|d|h|hr|hrs|hours?|min|minutes?|sec|seconds?|meters?|grams?|atm|AU|au|Mpc|kpc'
              r'|arcsec|arcmin|mag|Angstroms?|angstroms?|degrees?|deg|percent|photons?|electrons?|M_\\odot|M_\{\\odot\}'
              r'|L_\\odot|L_\{\\odot\}|R_\\odot|R_\{\\odot\})')
_EXPONENT = r'(?:\s*\^\s*(?:\{\s*-?\d+\s*\}|-?\d+))?'
_SEPARATOR = r'(?:\s|\\[,;:! ]|~|/|\\cdot|\*)*'
_PLAIN = (r'(?:\\(?:mu|Omega|AA|circ|degree)(?![A-Za-z])' + _EXPONENT +
          r'|\^\s*(?:\{\s*\\circ\s*\}|\\circ)|\\%|%|°|' + _UNIT_WORD + r'(?![A-Za-z])' + _EXPONENT + ')')
# \text{...}/\mathrm{...} count only when their content is itself a unit expression, so
# "\text{or 6}", "\mathrm{n}" and "\text{+ 2}" are never removed.
_WRAPPED = (r'\\(?:text|mathrm|rm|textrm|mbox|operatorname)\s*\{' + _SEPARATOR + _PLAIN +
            r'(?:' + _SEPARATOR + _PLAIN + r')*' + _SEPARATOR + r'\}' + _EXPONENT)
_TOKEN = r'(?:' + _WRAPPED + '|' + _PLAIN + ')'
_WITH_UNIT = re.compile(r'\s*(?P<num>' + _NUMBER + r')' + r'(?P<unit>(?:' + _SEPARATOR + _TOKEN + r')+)'
                        + _SEPARATOR + r'\.?\s*')


def identity():
    return {'version': VERSION, 'fingerprint': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_revision': '2026-10-24.minerva.1', 'base_scorer': math_scorer.identity()['fingerprint']}


def strip_unit(answer):
    """Number part of '<number> <units>', or None when the answer is not of that form."""
    text = math_scorer._strip_spacing_commands(str(answer).replace('$', ' ')) if answer else ''
    match = _WITH_UNIT.fullmatch(text.replace('\\left', '').replace('\\right', ''))
    return match.group('num').strip() if match and match.group('unit').strip() else None


def _final_answer_text(response):
    """Raw text of the scorer's chosen final answer when the scorer rejected it as malformed."""
    text = str(response or '')
    boxes = [m.start() for m in re.finditer(r'\\boxed\b', text)]
    markers = list(math_scorer.MARKER.finditer(text))
    if markers and (not boxes or markers[-1].start() > boxes[-1]):
        end = text.find('\n', markers[-1].end())
        return text[markers[-1].end():len(text) if end < 0 else end]
    if boxes:
        return math_scorer._unbox(text, boxes[-1])[0]
    return None


def score(response, ground_truth, config=None):
    result = math_scorer.score(response, ground_truth, config)
    info = identity()
    result.update(eval_postprocess=VERSION, eval_postprocess_fingerprint=info['fingerprint'], unit_stripped=False)
    if result['grade_status'] == 'correct' or result['grade_status'] == 'gt_unparseable':
        return result
    if math_scorer.ground_truth_semantics(ground_truth).get('kind') != 'scalar':
        return result
    rejected = result['grade_status'] in ('incorrect', 'unsupported', 'invalid_format') or result['parse_status'] == 'malformed'
    candidate = result.get('pred') if result.get('pred') is not None else (
        _final_answer_text(response) if result['parse_status'] == 'malformed' else None)
    number = strip_unit(candidate) if rejected and candidate else None
    if number is None:
        return result
    retry = math_scorer.score('Answer: ' + number, ground_truth, config)
    if retry['grade_status'] not in ('correct', 'incorrect'):
        return result
    # Only the number changed; a wrong number with a unit stays wrong.
    return {**retry, 'eval_postprocess': VERSION, 'eval_postprocess_fingerprint': info['fingerprint'],
            'unit_stripped': True, 'pred_with_unit': candidate.strip(), 'parse_method': result['parse_method']}
