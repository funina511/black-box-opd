"""One visible mathematical finding followed by the two direct process labels."""
import json

from . import process_direct
from .contracts import digest, file_hash

PROTOCOL = 'direct_brief_v1'
SCHEMA_VERSION = 'rubric.direct.brief.v1'
INPUT_VERSION = 'rubric.direct.brief.input.v1'
FIELDS = ('brief_check', *process_direct.DIMS)
SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': list(FIELDS),
    'properties': {'brief_check': {'type': 'string', 'minLength': 1},
                   **process_direct.SCHEMA['properties']},
}


def parse_brief(text):
    def ordered_object(pairs):
        if tuple(k for k, _ in pairs) != FIELDS:
            raise process_direct.DirectProtocolError('expected brief_check, verified_progress, unresolved_error_severity in order')
        return dict(pairs)

    try:
        row = json.loads(text, object_pairs_hook=ordered_object)
    except (TypeError, ValueError) as exc:
        raise process_direct.DirectProtocolError(str(exc)) from exc
    if not isinstance(row, dict) or not isinstance(row['brief_check'], str) or not row['brief_check'].strip():
        raise process_direct.DirectProtocolError('brief_check must be a nonempty string')
    # Use the same strict ordinal, null, boolean and duplicate-label policy as
    # direct_v1. The finding is preserved for audits and has no training target.
    labels = {dim: row[dim] for dim in process_direct.DIMS}
    result = process_direct.parse_direct(json.dumps(labels))
    for dim in labels:
        result[dim]['source'] = 'judge_direct_brief_v1'
    result.update(brief_check=row['brief_check'], _raw_output=text, _process_protocol=PROTOCOL,
                  _schema_version=SCHEMA_VERSION)
    return result


def protocol_identity(spec, instruction):
    return {'process_protocol': PROTOCOL, 'schema_version': SCHEMA_VERSION, 'input_protocol': INPUT_VERSION,
            'schema_hash': digest(SCHEMA), 'field_order': list(FIELDS), 'spec_hash': digest(spec),
            'instructions_hash': digest(instruction), 'parser_hash': file_hash(__file__),
            'request_builder_hash': file_hash(process_direct.__file__)}
