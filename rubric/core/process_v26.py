"""Stable source fragments, selected evidence and local process-label validation."""
from copy import deepcopy
import json
import re
from pathlib import Path
from .contracts import V2_DIMENSIONS, V2_SCHEMA_VERSION, V2_INPUT_VERSION, validate_label, digest, file_hash

V2_SCHEMA_VERSION = 'rubric.events.v2.6'
V2_INPUT_VERSION = 'rubric.input.v2.4'
from .evidence_ranges_v1 import RANGES, expand, audit, EvidenceError

MARKER = '\n\nVISIBLE_RESPONSE\n'


class ProtocolError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(code + ': ' + message)


def line_spans(response):
    result, offset = [], 0
    for index, line in enumerate(response.splitlines(keepends=True), 1):
        result.append({'id': index, 'start': offset, 'end': offset + len(line), 'text': line})
        offset += len(line)
    return result


def source_spans(response):
    """One nonblank physical line per fragment; adjacent whitespace is preserved.

    Error and replacement lines must be independently selectable. A single-line
    mixed error/repair remains uncertifiable by this structural protocol.
    """
    result, pending = [], []
    def emit():
        if not pending:return
        result.append({'id': f's{len(result) + 1}', 'start': pending[0]['start'], 'end': pending[-1]['end'],
                       'loc': {'start': pending[0]['id'], 'end': pending[-1]['id']},
                       'text': ''.join(line['text'] for line in pending)})
        pending.clear()
    for line in line_spans(response):
        if line['text'].strip() and any(p['text'].strip() for p in pending):emit()
        pending.append(line)
    emit()
    return result


def compact_response(response):
    return json.dumps([{'id': span['id'], 'text': span['text']} for span in source_spans(response)],
                      ensure_ascii=False, separators=(',', ':'))


def restore_response(text):
    try:rows = json.loads(text)
    except (ValueError, TypeError) as exc:raise ProtocolError('input_schema_error', 'invalid source fragments') from exc
    if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != {'id', 'text'} or
            row['id'] != f's{i}' or not isinstance(row['text'], str) for i, row in enumerate(rows, 1)):
        raise ProtocolError('input_location_error', 'invalid source fragment identities')
    response = ''.join(row['text'] for row in rows)
    canonical = [{'id': span['id'], 'text': span['text']} for span in source_spans(response)]
    if rows != canonical:raise ProtocolError('input_integrity_error', 'noncanonical source fragments')
    return response


def response_from_messages(messages):
    try:
        header, packed = messages[-1]['content'].split(MARKER, 1)
        metadata = json.loads(header)
    except (KeyError, ValueError, TypeError) as exc:
        raise ProtocolError('input_schema_error', 'invalid compact request') from exc
    if metadata.get('input_protocol') != V2_INPUT_VERSION:
        raise ProtocolError('input_version_mismatch', 'new run required')
    response = restore_response(packed)
    if len(response) != metadata.get('response_characters') or digest(response) != metadata.get('response_hash'):
        raise ProtocolError('input_integrity_error', 'visible response content differs')
    return response


def render_instructions(spec, instruction):
    return instruction.strip() + '\n\n固定数学判据（唯一规则来源）：\n' + spec.strip()


def protocol_identity(spec, instruction):
    from . import contracts, evidence_ranges
    return {'input_protocol': V2_INPUT_VERSION, 'schema_version': V2_SCHEMA_VERSION,
            'schema_hash': digest(JUDGE_SCHEMA_V2), 'instructions_hash': digest(render_instructions(spec, instruction)),
            'spec_hash': digest(spec), 'prompt_hash': digest(instruction),
            'evidence_codec_hash': file_hash(evidence_ranges.__file__), 'evidence_max_units':8,
            'parser_hash': file_hash(__file__), 'request_builder_hash': file_hash(contracts.__file__)}


LABEL_BASE = {'type': 'object', 'additionalProperties': False,
              'required': ['value', 'valid', 'reason', 'event_ids'],
              'properties': {'value': {'enum': [0, .5, 1, None]}, 'valid': {'type': 'boolean'},
                             'reason': {'type': 'string', 'minLength': 1},
                             'event_ids': {'type': 'array', 'items': {'type': 'string'}}}}
# xgrammar ignores siblings of anyOf. Each branch carries the complete object.
LABEL = {'anyOf': []}
for valid in (False, True):
    alternative = deepcopy(LABEL_BASE)
    alternative['properties']['valid'] = {'const': valid}
    alternative['properties']['value'] = {'enum': [0, .5, 1]} if valid else {'type': 'null'}
    LABEL['anyOf'].append(alternative)
EVENT_BASE = {'type': 'object', 'additionalProperties': False,
         'required': ['type', 'supporting_span_ranges', 'trajectory_claim', 'judge_verification', 'related_event_ids'],
         'properties': {'type': {'enum': ['verified_progress', 'explicit_error', 'neutral', 'unjudgeable']},
                        'supporting_span_ranges': RANGES,
                        'trajectory_claim': {'type': 'string', 'minLength': 1},
                        'judge_verification': {'type': 'string', 'minLength': 1},
                        'related_event_ids': {'type': 'array', 'items': {'type': 'string'}}}}
REPAIR_EVENT = deepcopy(EVENT_BASE)
REPAIR_EVENT['required'] = ['type', 'supporting_span_ranges', 'trajectory_claim', 'judge_verification', 'error_event_id']
REPAIR_EVENT['properties']['type'] = {'const': 'corrected_error'}
REPAIR_EVENT['properties'].pop('related_event_ids')
REPAIR_EVENT['properties']['error_event_id'] = {'enum':[f'e{i}' for i in range(1,5)]}
# A required scalar link is supported by vLLM 0.8.5 xgrammar; minItems is not.
EVENT = {'anyOf': [EVENT_BASE, REPAIR_EVENT]}
JUDGE_SCHEMA_V2 = {'type': 'object', 'additionalProperties': False,
                  'required': ['schema_version', 'rubric_version', 'completion', 'analysis_complete', 'events', *V2_DIMENSIONS[1:]],
                  'properties': {'schema_version': {'const': V2_SCHEMA_VERSION}, 'rubric_version': {'const': 'math_process_v2'},
                                 'completion': {'enum': ['complete', 'prefix', 'no_math', 'unknown']},
                                 'analysis_complete': {'type': 'boolean'},
                                 'events': {'type': 'object', 'additionalProperties': False, 'properties': {f'e{i}': EVENT for i in range(1, 5)}},
                                 **{dimension: LABEL for dimension in V2_DIMENSIONS[1:]}}}


def parse_events(text, response):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:raise ProtocolError('event_identity_error', 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    try:
        row = json.loads(text, object_pairs_hook=unique_object)
    except ProtocolError:
        raise
    except (ValueError, TypeError) as exc:
        raise ProtocolError('json_unreadable', str(exc)) from exc
    if not isinstance(row, dict) or set(row) != set(JUDGE_SCHEMA_V2['required']):
        raise ProtocolError('root_schema_error', 'process root fields mismatch')
    if row['schema_version'] != V2_SCHEMA_VERSION or row['rubric_version'] != 'math_process_v2':
        raise ProtocolError('protocol_version_mismatch', 'new schema/cache/run required')
    if row['completion'] not in ('complete', 'prefix', 'no_math', 'unknown') or type(row['analysis_complete']) is not bool:
        raise ProtocolError('root_schema_error', 'invalid completion/coverage')
    if not isinstance(row['events'], dict) or len(row['events']) > 4:
        raise ProtocolError('event_budget_error', 'events must be a map of at most four named slots')
    if any(key not in {f'e{i}' for i in range(1, 5)} for key in row['events']):
        raise ProtocolError('event_identity_error', 'unknown event slot')
    response = response or ''
    segments = {s['id']: s for s in source_spans(response)}
    events, validity, resolved = {}, {}, {}
    for eid, payload in row['events'].items():
        event = {**payload, 'id': eid} if isinstance(payload, dict) else {'id': eid}
        repair = event.get('type') == 'corrected_error'
        branch = REPAIR_EVENT if repair else EVENT_BASE
        if repair:event['related_event_ids'] = [event.get('error_event_id')]
        events[eid] = event
        reasons = []
        if (not isinstance(payload, dict) or set(payload) != set(branch['required']) or
                not repair and event.get('type') not in EVENT_BASE['properties']['type']['enum']):
            reasons.append('event_schema_error')
        units = event.get('supporting_span_ranges')
        try:
            ids = expand(units, list(segments.values()))
            if any(not segments[sid]['text'].strip() for sid in ids):raise EvidenceError('event_span_empty')
            event['supporting_span_ids'] = ids # Internal only; the raw wire is kept in _record.
            resolved[eid] = {**audit(units, ids, list(segments.values())), 'supporting_span_ids': ids, 'spans': [
                {'span_id': sid, 'loc': segments[sid]['loc'], 'quote': segments[sid]['text'],
                 'start': segments[sid]['start'], 'end': segments[sid]['end']} for sid in ids]}
        except EvidenceError as exc:
            reasons.append(exc.code)
        for field in ('trajectory_claim', 'judge_verification'):
            if not isinstance(event.get(field), str) or not event[field].strip():
                reasons.append('event_' + field + '_missing')
        links = event.get('related_event_ids')
        if not isinstance(links, list) or any(not isinstance(x, str) for x in links) or len(set(links)) != len(links):
            reasons.append('event_links_malformed')
        validity[eid] = {'valid': not reasons, 'reasons': reasons}
    def invalidate(eid, reason):
        validity[eid]['valid'] = False
        if reason not in validity[eid]['reasons']:
            validity[eid]['reasons'].append(reason)
    def valid_span_ids(event):
        ids=event.get('supporting_span_ids')
        return set(ids) if isinstance(ids,list) and all(isinstance(s,str) for s in ids) else set()
    for eid, event in events.items():
        links = event.get('related_event_ids')
        if not isinstance(links, list) or any(not isinstance(x, str) for x in links):
            continue
        if any(link not in events or link == eid for link in links):
            invalidate(eid, 'event_link_missing_or_self')
        if event.get('type') == 'corrected_error' and (not links or any(events.get(link, {}).get('type') != 'explicit_error' for link in links)):
            invalidate(eid, 'correction_requires_observed_error')
        if event.get('type') == 'corrected_error' and any(
                valid_span_ids(event).issubset(valid_span_ids(events.get(link, {}))) for link in links):
            invalidate(eid, 'correction_reuses_error_span')
    # Detect dependency cycles and propagate only along explicit event links.
    visited, stack = set(), []
    def visit(eid):
        if eid in stack:
            for member in stack[stack.index(eid):]:invalidate(member, 'event_link_cycle')
            return
        if eid in visited:return
        stack.append(eid)
        links = events[eid].get('related_event_ids', [])
        for link in links if isinstance(links, list) else []:
            if isinstance(link, str) and link in events:visit(link)
        stack.pop();visited.add(eid)
    for eid in events:visit(eid)
    for _ in events:
        changed = False
        for eid, event in events.items():
            links = event.get('related_event_ids', [])
            if validity[eid]['valid'] and any(not validity[link]['valid'] for link in links):
                invalidate(eid, 'invalid_event_dependency');changed = True
        if not changed:break
    invalid = {eid for eid in events if not validity[eid]['valid']}
    repaired, uncertain_repair = set(), set()
    for eid, event in events.items():
        if event.get('type') != 'corrected_error':continue
        links = event.get('related_event_ids', [])
        targets = {link for link in links if isinstance(link, str) and link in events and events[link].get('type') == 'explicit_error'} if isinstance(links, list) else set()
        if eid not in invalid:repaired.update(targets)
        elif targets:uncertain_repair.update(targets)
        else:uncertain_repair.update(name for name, e in events.items() if e.get('type') == 'explicit_error')
    labels, mask_reasons = {}, {dimension: [] for dimension in V2_DIMENSIONS[1:]}
    for dimension in V2_DIMENSIONS[1:]:
        item = row[dimension]
        if not isinstance(item, dict) or set(item) != set(LABEL_BASE['required']):
            raise ProtocolError('label_schema_error', dimension)
        try:validate_label(item, dimension)
        except ValueError as exc:raise ProtocolError('label_schema_error', str(exc)) from exc
        if not isinstance(item['reason'], str) or not item['reason'].strip() or not isinstance(item['event_ids'], list) or any(not isinstance(x, str) for x in item['event_ids']):
            raise ProtocolError('label_schema_error', 'reason/evidence IDs required')
        labels[dimension] = {**deepcopy(item), 'evidence': item['reason']}
    def mask(dimension, reason):
        # Mathematical unknowns are preserved, not converted into scored labels.
        if labels[dimension]['valid']:
            if reason not in mask_reasons[dimension]:mask_reasons[dimension].append(reason)
            labels[dimension].update(value=None, valid=False, contract_violation=reason)
    discarded = {}
    for dimension, label in labels.items():
        if not label['valid']:continue
        refs = label['event_ids']
        if len(refs) != len(set(refs)):
            mask(dimension, 'duplicate_label_event_dependency');continue
        bad = [eid for eid in refs if eid not in events or eid in invalid]
        # Keep independent partial progress; global claims cannot discard bad math.
        removable = bad if dimension == 'verified_progress' and label['value'] == .5 else [
            eid for eid in bad if eid in events and events[eid].get('type') in ('neutral', 'unjudgeable')]
        if any(eid not in removable for eid in bad):
            mask(dimension, 'invalid_label_event_dependency');continue
        if removable:
            discarded[dimension] = removable
            refs = [eid for eid in refs if eid not in removable]
            label['event_ids'] = refs
        value = label['value']
        if value > 0:
            kinds = {'verified_progress', 'corrected_error'} if dimension == 'verified_progress' else {'explicit_error'}
            support = [eid for eid in refs if events[eid]['type'] in kinds]
            if not support:mask(dimension, 'positive_target_lacks_valid_evidence')
            if dimension == 'unresolved_error_severity':
                if any(eid in repaired for eid in support):mask(dimension, 'repaired_error_not_unresolved')
                if any(eid in uncertain_repair for eid in support):mask(dimension, 'untrusted_repair_affects_unresolved_status')
    uncertain_error = any(events[eid].get('type') in ('explicit_error', 'corrected_error') for eid in invalid)
    unrepaired_error = any(e['type'] == 'explicit_error' and eid not in repaired for eid, e in events.items() if eid not in invalid)
    unknown_math = any(e['type'] == 'unjudgeable' for eid, e in events.items() if eid not in invalid)
    if labels['verified_progress']['valid'] and labels['verified_progress']['value'] == 0 and not row['analysis_complete']:
        mask('verified_progress', 'zero_progress_requires_complete_visible_review')
    if labels['verified_progress']['valid'] and labels['verified_progress']['value'] == 1:
        if row['completion'] != 'complete':mask('verified_progress', 'full_progress_requires_complete_solution')
        elif not row['analysis_complete']:mask('verified_progress', 'full_progress_requires_complete_analysis')
        elif unknown_math:mask('verified_progress', 'full_progress_has_unjudgeable_math')
        elif uncertain_error:mask('verified_progress', 'full_progress_has_uncertain_error_evidence')
        elif unrepaired_error:mask('verified_progress', 'full_progress_has_unrepaired_error')
        if row['unresolved_error_severity']['valid'] and row['unresolved_error_severity']['value'] == 1:
            mask('verified_progress', 'full_solution_conflicts_with_severe_error')
    if labels['unresolved_error_severity']['valid'] and labels['unresolved_error_severity']['value'] == 0:
        math_events = [e for eid, e in events.items() if eid not in invalid and e['type'] in ('verified_progress', 'explicit_error', 'corrected_error')]
        if unknown_math:mask('unresolved_error_severity', 'zero_error_has_unjudgeable_math')
        elif uncertain_error:mask('unresolved_error_severity', 'zero_error_has_uncertain_error_evidence')
        elif not row['analysis_complete'] or row['completion'] in ('no_math', 'unknown') or not math_events:
            mask('unresolved_error_severity', 'zero_error_requires_checked_math')
        elif unrepaired_error:
            mask('unresolved_error_severity', 'unrepaired_error_conflicts_with_zero')
    violations = [{'dimension': dimension, 'reason': reason} for dimension, reasons in mask_reasons.items() for reason in reasons]
    return {**labels, '_record': row, '_raw_output': text, '_raw_labels': {d: deepcopy(row[d]) for d in V2_DIMENSIONS[1:]},
            '_event_validity': validity, '_event_issues': [{'event_id': eid, **validity[eid]} for eid in sorted(invalid)],
            '_source_evidence': resolved, '_evidence_audit': {'capacity_conflicts': [eid for eid in validity if 'evidence_capacity_conflict' in validity[eid]['reasons']], 'cross_event_duplicate_spans': sum(len(v['supporting_span_ids']) for v in resolved.values())-len({sid for v in resolved.values() for sid in v['supporting_span_ids']})}, '_discarded_label_event_ids': discarded,
            '_validation_scope': {'structure':True, 'source_presence':True, 'semantic_entailment':'not_checked'},
            '_judge_reported_scope': {'analysis_complete':row['analysis_complete'], 'completion':row['completion']},
            '_structural_validation_status': 'partial_failure' if invalid or violations else 'passed',
            '_semantic_review': {'status':'pending', 'review_source':None, 'human_verified':False},
            '_training_eligible': {d: labels[d]['valid'] for d in labels},
            '_mask_reasons': mask_reasons, '_violations': violations,
            '_protocol': {'input_protocol': V2_INPUT_VERSION, 'schema_version': V2_SCHEMA_VERSION}}
