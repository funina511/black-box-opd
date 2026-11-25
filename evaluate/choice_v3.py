"""Opt-in choice scoring for explicit 'So Answer:' and numeric workings before a declaration."""
import hashlib
from pathlib import Path
import re

from evaluate import choice

VERSION = 'choice_exact_v3'
_SO_ANSWER = re.compile(r'(?im)(?:^|(?<=[.,;:!?]))[ \t]*so[ \t]*,?[ \t]+(?=answer[ \t]*:)')
_NUMBER_BOX = re.compile(r'\\boxed\s*\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}')


def identity():
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(choice.identity()['fingerprint'].encode('ascii'))
    return {'version': VERSION, 'fingerprint': digest.hexdigest(), 'protocol_revision': '2026-11-19.choice.3'}


def score(row, task, *, scorer_identity=None):
    # Expose the final line's label to v2, which already handles malformed and ambiguous answers.
    response = row.get('response', '').rsplit('</think>', 1)[-1].strip()
    before, separator, last = response.rpartition('\n')
    response = before + separator + _SO_ANSWER.sub('\n', last)
    result = choice.score({**row, 'response': response}, task)
    if result.get('parse_method') in ('multiple_final_options', 'conflicting_final_options') and _NUMBER_BOX.search(response):
        choices = (row.get('extra_info') or {}).get('options')
        candidates = choice._candidates(response, choices)
        start, _, method = max(candidates, key=lambda item: item[0])
        if method != 'boxed':
            # Keep option boxes and everything after the declaration for v2's ambiguity checks.
            line_start = response.rfind('\n', 0, start) + 1
            prefix = response[line_start:start]
            numeric_workings = _NUMBER_BOX.sub(r'\1', prefix)
            if numeric_workings != prefix:
                response = response[:line_start] + numeric_workings + response[start:]
                result = choice.score({**row, 'response': response}, task)
    info = scorer_identity or identity()
    return {**result, 'scorer_version': VERSION, 'scorer_fingerprint': info['fingerprint']}
