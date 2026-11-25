"""Shared annotation and immutable-cache path for prefetch and the main barrier."""
import time
from .common import append_json
from .core.contracts import V2_DIMENSIONS as DIMS, validate_label, judge_messages


def score_pairs(owner, pairs, roles=('teacher', 'student')):
    """Preserve main-thread math grading; only Judge I/O needs a worker thread."""
    scores = {}
    seen = set()
    for pair in pairs:
        for role in roles:
            key = owner.cache_key(pair, role)
            if key in seen:
                continue
            seen.add(key)
            if owner.options[role + '_cache'] and owner.cache.get(key) is not None:
                continue
            scores[key] = owner.scorer(pair[role]['response'], pair['ground_truth'])
    return scores


def build_labels(owner, pairs, selected_keys, schema_version, roles=('teacher', 'student'), *, scorer_results=None, row_keys=None):
    from scoring.math import correctness_label
    unique, requests, values = {}, {}, {}
    hits = {'teacher': 0, 'student': 0}
    for i, pair in enumerate(pairs):
        for role in roles:
            key = row_keys[role][i] if row_keys is not None else owner.cache_key(pair, role)
            if key in unique:continue
            unique[key] = (pair, role)
            cached = owner.cache.get(key) if owner.options[role + '_cache'] else None
            if cached:
                if (cached.get('protocol_identity') != owner.protocol_identity or cached['scorer_fingerprint'] != owner.scorer_identity['fingerprint']
                        or cached['schema_version'] != schema_version or cached['response_id'] != pair[role]['response_id']):
                    raise ValueError('V2.1 cached protocol/scorer/response identity mismatch')
                for dim in DIMS:validate_label(cached['labels'][dim], dim)
                values[key] = cached;hits[role] += 1;continue
            result = (owner.scorer(pair[role]['response'], pair['ground_truth'])
                      if scorer_results is None else scorer_results[key])
            prompt_unavailable = pair['prompt_truncated'] or not pair['prompt'].strip()
            labels = {DIMS[0]: correctness_label(result, bool(prompt_unavailable))}
            reason = ('prompt_unavailable' if prompt_unavailable else 'correctness_only' if not owner.judge
                      else 'not_selected' if key not in selected_keys else None)
            for dim in DIMS[1:]:labels[dim] = {'value': None, 'valid': False, 'source': reason or 'pending'}
            visibility = {'prompt_truncated': pair['prompt_truncated'],
                'response_source_truncated': pair[role].get('source_truncated', pair[role]['truncated']),
                'training_token_truncated': pair[role].get('training_token_truncated', False), 'judge_context_overflow': False}
            values[key] = {'schema_version': schema_version, 'rubric_version': owner.version, 'rubric_hash': owner.spec_hash,
                'protocol_identity': owner.protocol_identity, 'response_id': pair[role]['response_id'], 'cache_key': key, 'labels': labels,
                'raw_scorer_result': result, 'scorer_fingerprint': result['scorer_fingerprint'],
                'judge_fingerprint': owner.options['judge_fingerprint'], 'evidence': {}, 'events': [],
                'visibility': visibility, 'unknown_reason': reason, 'review_status': 'pending',
                'process_unavailable_kind': 'not_annotatable' if prompt_unavailable else 'not_requested' if reason else 'pending'}
            if reason:
                if owner.options[role + '_cache'] and reason!='not_selected':owner.cache.put(key, values[key])
            else:
                requests[key] = judge_messages(pair['prompt'], pair[role]['response'], pair['ground_truth'],
                    owner.spec, owner.instruction, owner.version, visibility,process_protocol=owner.options.get('process_protocol','events_v27'))
    return unique, requests, values, hits


def execute_requests(owner, unique, requests, values, step, deadline, *, phase='main'):
    from .online_v2 import persist_results, request_belongs_to, process_failure_kind
    direct = owner.options.get('process_protocol', 'events_v27') == 'direct_v1'
    ledger_start = len(getattr(owner.judge, 'api_ledger', []))
    try:
        from .annotation_scheduler import annotate_chains
        contexts={key:(unique[key][0]['prompt'],unique[key][0][unique[key][1]]['response']) for key in requests}
        from .annotation_journal import AnnotationJournal
        journal=AnnotationJournal(owner.directory,owner.protocol_identity,step=step,
            max_attempts=owner.options['judge'].get('max_annotation_attempts',2),
            unknown_policy=owner.options['judge'].get('unknown_submission_policy','abort'),
            primary_requests=1+owner.options['judge'].get('max_retries',0),
            base_request_count=getattr(owner.judge,'request_count',0))
        recovered=journal.recover(contexts)
        if owner.judge:owner.judge.request_count=max(owner.judge.request_count if hasattr(owner.judge,'request_count') else 0,journal.recovered_request_count)
        if phase == 'main':
            for key, row in recovered.items():
                if (row.get('_execution_phase') == 'teacher_prefetch' and 'parsed' not in row
                        and not row.get('historical_submission_unknown') and not row.get('_retry_stage')
                        and row.get('annotation_attempt', 1) < owner.options['judge'].get('max_annotation_attempts', 2)):
                    # A failed speculative attempt uses the SAME journal/attempt
                    # bound. Unknown remote submissions still obey abort/retry policy.
                    row.update(_retry_stage='primary', annotation_attempt=row.get('annotation_attempt', 1) + 1,
                               attempts=0, usage={})
        def save_state(rid, state, result):
            journal.save(rid, state, {**result, '_execution_phase': phase})
        results = annotate_chains(owner.judge,requests,contexts,deadline,
            recovered=recovered, on_state=save_state,
            semantic=owner.options.get('semantic_check',False),process_protocol=owner.options.get('process_protocol','events_v27')) if requests else {}
        if results and all('parsed' not in r for r in results.values()) and not any(
                v['labels'][d]['valid'] for v in values.values() for d in DIMS[1:]):
            error=ValueError('all V2 annotations failed; refusing a silently masked training step')
            error.results=results
            raise error
    except Exception as exc:
        persist_results(owner, getattr(exc, 'results', {}), requests, step, fatal=exc)
        raise
    for key,result in results.items():
        if 'targeted_result' in result:
            append_json(owner.directory/'targeted_review_requests.jsonl',{'step':step,'request_id':key,
                'protocol_identity':owner.protocol_identity,'messages':result.get('targeted_request'),'result':result['targeted_result']})
    if requests:append_json(owner.directory/'annotation_schedules.jsonl',{'step':step,'execution_phase':phase,**owner.judge.last_schedule,
        'actual_api_peak':owner.judge.api_peak,'api_ledger':[a for a in owner.judge.api_ledger[ledger_start:]
            if request_belongs_to(a['request_id'],set(requests))],
        'historical_cost_scope':'referenced journal history; not additive across steps'})
    persist_results(owner, results, requests, step)
    for key in requests:
        result = results.get(key, {'error': 'judge_result_missing', 'attempts': 0, 'failure_kind': 'technical'})
        if 'parsed' in result:
            parsed = result['parsed']
            if direct:
                values[key].update(process_protocol='direct_v1',review_status='not_requested',
                    semantic_review=parsed['_semantic_review'],training_eligible=parsed['_training_eligible'],
                    raw_output=result.get('raw_output'),raw_labels=parsed['_raw_labels'])
                for dim in DIMS[1:]:values[key]['labels'][dim]=dict(parsed[dim])
            else:
                values[key].update(judge_reported_scope=parsed.get('_judge_reported_scope'),
                                   structural_validation_status=parsed.get('_structural_validation_status'),
                                   semantic_review=parsed.get('_semantic_review', {'status':'pending','review_source':None}),
                                   training_eligible=parsed.get('_training_eligible'),
                                   event_record=parsed['_record'], events=parsed['_record']['events'],
                                   source_evidence=parsed.get('_source_evidence', {}),
                                   review_source_evidence=parsed.get('_review_source_evidence', {}), evidence_audit=parsed.get('_evidence_audit', {}),
                                   evidence_validation_scope=parsed.get('_validation_scope', {}),
                                   discarded_label_event_ids=parsed.get('_discarded_label_event_ids', {}),
                                   raw_output=result.get('raw_output', parsed.get('_raw_output')),
                                   raw_labels=parsed.get('_raw_labels', {d: parsed['_record'][d] for d in DIMS[1:]}),
                                   event_validity=parsed.get('_event_validity', {}), mask_reasons=parsed.get('_mask_reasons', {}))
                values[key]['review_status'] = parsed.get('_semantic_review', {}).get('status', 'not_requested')
                values[key]['unknown_math_events'] = parsed.get('_unknown_math_events', [])
                for dim in DIMS[1:]:
                    label = parsed[dim]
                    values[key]['labels'][dim] = {'value': label['value'], 'valid': label['valid'],
                        'source': 'judge_contract_violation' if label.get('contract_violation') else 'judge'}
                    values[key]['evidence'][dim] = {k: label[k] for k in ('reason', 'event_ids')}
            values[key]['contract_violations'] = parsed.get('_violations', [])
            new_sends=[a for a in getattr(owner.judge,'api_ledger',[])[ledger_start:]
                       if request_belongs_to(a['request_id'],{key})]
            values[key]['annotation_cost']={'requests':result.get('historical_requests_used',0)+len(new_sends),
                'usage':{k:result.get('historical_usage',{}).get(k,0)+sum(a.get('usage',{}).get(k,0) for a in new_sends)
                         for k in ('prompt_tokens','completion_tokens','total_tokens')}}
            valid = any(parsed[d]['valid'] for d in DIMS[1:])
            values[key]['process_unavailable_kind'] = process_failure_kind(parsed)
            semantic_failure=result.get('targeted_result',{}).get('error')
            if not valid and semantic_failure:values[key]['process_unavailable_kind']='technical'
            values[key]['unknown_reason'] = semantic_failure or ('judge_contract_violation' if parsed.get('_violations') else 'judge_unknown' if any(not parsed[d]['valid'] for d in DIMS[1:]) else None)
            if owner.options[unique[key][1] + '_cache'] and result.get('cacheable',True):owner.cache.put(key, values[key])
        else:
            values[key].update(unknown_reason=result['error'], process_unavailable_kind='technical')
            if result['error'] == 'judge_context_exceeded':values[key]['visibility']['judge_context_overflow'] = True
    return results, recovered
