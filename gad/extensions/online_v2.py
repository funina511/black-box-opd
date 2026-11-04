"""Versioned annotations and final student process coverage at the optimizer barrier."""
import time
from .common import append_json
from .core.contracts import V2_DIMENSIONS as DIMS, V2_SCHEMA_VERSION, validate_label, judge_messages
from .critic_batch import auxiliary_weights, supervision_identity


class StudentProcessSupervisionError(RuntimeError):
    pass


def process_failure_kind(parsed):
    """Structural contract failures differ from a well-formed mathematical unknown."""
    if any(parsed[d]['valid'] for d in DIMS[1:]):return None
    structural = {'invalid_label_event_dependency', 'duplicate_label_event_dependency',
                  'positive_target_lacks_valid_evidence'}
    reasons={v['reason'] for v in parsed.get('_violations', [])}
    if reasons & structural:return 'contract_rejected'
    # A bad, unrelated event must not override the labels' mathematical unknown.
    if reasons or parsed.get('_unknown_math_events'):return 'mathematical_unknown'
    row=parsed.get('_record',{})
    dependencies={eid for d in DIMS[1:] if row.get(d,{}).get('valid')
                  for eid in row.get(d,{}).get('event_ids',[])}
    return 'contract_rejected' if any(e.get('event_id') in dependencies for e in parsed.get('_event_issues',[])) else 'mathematical_unknown'


def request_belongs_to(request_id, roots):
    import re
    if request_id in roots:return True
    return any(re.fullmatch(re.escape(root)+r'(?::attempt:[1-9][0-9]*)?:targeted',request_id or '') for root in roots)


def persist_results(owner, results, requests, step, fatal=None):
    for key, result in results.items():
        append_json(owner.directory / 'judge_attempts.jsonl', {'step': step, 'request_id': key,
                    'protocol_identity': owner.protocol_identity, **result})
        if 'parsed' not in result:
            append_json(owner.directory / 'judge_errors.jsonl', {'step': step, 'request_id': key, **result})
    if fatal is not None:
        append_json(owner.directory / 'judge_step_failures.jsonl', {'step': step,
                    'error': str(fatal), 'exception': type(fatal).__name__, 'received_results': len(results),
                    'requested': len(requests), 'missing_request_ids': sorted(set(requests) - set(results)),
                    'protocol_identity': owner.protocol_identity,
                    'usage': {key: sum(result.get('usage', {}).get(key, 0) for result in results.values())
                              for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}})
        for key, messages in requests.items():
            append_json(owner.directory / 'judge_failure_inputs.jsonl', {'step': step, 'request_id': key,
                        'messages': messages, 'result': results.get(key)})


def final_student_guard(owner, pairs, values, stats, step, process_selected=None):
    for role in ('teacher', 'student'):
        rows = {supervision_identity(p,role): (p, values[owner.cache_key(p, role)]) for i,p in enumerate(pairs)
                if process_selected is None or process_selected[role][i]}
        for scope in ('all', 'current', 'historical'):
            selected = [(p, r) for p, r in rows.values() if scope == 'all' or
                        ((p.get('source_step', step) == step) == (scope == 'current'))]
            annotatable = [(p, r) for p, r in selected if not p['prompt_truncated'] and bool(p['prompt'].strip())]
            prefix = f'rubric/process_{role}_{scope}_'
            stats[prefix + 'selected'] = len(selected)
            stats[prefix + 'annotatable'] = len(annotatable)
            any_valid = sum(any(r['labels'][d]['valid'] for d in DIMS[1:]) for _, r in selected)
            both_valid = sum(all(r['labels'][d]['valid'] for d in DIMS[1:]) for _, r in selected)
            stats[prefix + 'any_valid'] = any_valid
            stats[prefix + 'both_valid'] = both_valid
            stats[prefix + 'any_coverage'] = any_valid / len(selected) if selected else 0
            stats[prefix + 'both_coverage'] = both_valid / len(selected) if selected else 0
            stats[prefix + 'annotatable_any_valid'] = sum(any(r['labels'][d]['valid'] for d in DIMS[1:]) for _, r in annotatable)
            stats[prefix + 'annotatable_both_valid'] = sum(all(r['labels'][d]['valid'] for d in DIMS[1:]) for _, r in annotatable)
            stats[prefix + 'annotatable_any_coverage'] = stats[prefix + 'annotatable_any_valid'] / len(annotatable) if annotatable else 0
            for dimension in DIMS[1:]:
                stats[prefix + 'valid_' + dimension] = sum(r['labels'][dimension]['valid'] for _, r in selected)
                stats[prefix + 'coverage_' + dimension] = stats[prefix + 'valid_' + dimension] / len(selected) if selected else 0
            stats[prefix + 'technical_unavailable'] = sum(r.get('process_unavailable_kind') == 'technical' for _, r in annotatable)
            stats[prefix + 'contract_rejected'] = sum(r.get('process_unavailable_kind') == 'contract_rejected' for _, r in annotatable)
            stats[prefix + 'mathematical_unknown'] = sum(r.get('process_unavailable_kind') == 'mathematical_unknown' for _, r in annotatable)
    if owner.options['mode'] != 'joint':return
    prefix = 'rubric/process_student_all_'
    any_valid = stats[prefix + 'any_valid']
    if any_valid:
        owner.v2_empty_student_process_steps = 0
    else:
        owner.v2_empty_student_process_steps += 1
    stats['rubric/student_empty_process_steps'] = owner.v2_empty_student_process_steps
    stats['rubric/student_single_dimension_only'] = int(any_valid > 0 and stats[prefix + 'both_valid'] == 0)
    reason = None
    if stats[prefix + 'annotatable'] and stats[prefix + 'technical_unavailable'] == stats[prefix + 'annotatable'] and not any_valid:
        reason = 'student_process_all_technical_failure'
    elif stats[prefix + 'annotatable'] and stats[prefix + 'contract_rejected'] == stats[prefix + 'annotatable'] and not any_valid:
        reason = 'student_process_all_contract_rejected'
    elif not any_valid and owner.v2_empty_student_process_steps >= owner.options.get('max_empty_student_process_steps', 3):
        reason = 'student_process_empty_streak' if stats[prefix + 'annotatable'] else 'no_annotatable_students_streak'
    append_json(owner.directory / 'process_supervision_steps.jsonl', {'step': step, 'stats': stats,
                'coverage_state_version': owner.v2_coverage_state_version,
                'counter_initialization': owner.v2_counter_initialization, 'stop_reason': reason})
    if reason:
        error = StudentProcessSupervisionError(reason)
        error.stats = dict(stats)
        raise error
    if not any_valid:
        print(f"[rubric] WARNING: zero effective student process labels for {owner.v2_empty_student_process_steps} critic steps; annotatable={stats[prefix + 'annotatable']}", flush=True)


def annotate(owner, pairs, batch, global_mini, dp_size, step):
    import torch
    from scoring.math import correctness_label
    started = time.monotonic()
    direct=owner.options.get('process_protocol','events_v27')=='direct_v1'
    schema_version=getattr(owner,'annotation_schema_version',V2_SCHEMA_VERSION)
    ledger_start=len(getattr(owner.judge,'api_ledger',[]))
    health_before=getattr(owner.judge,'health_get_count',0)
    from .process_sampling import select_process
    selection,selection_groups=select_process(pairs,global_mini,dp_size,owner.options,step,
        pairs[0].get('producer_stage','warmup') if pairs else 'warmup')
    sampling=owner.options['mode']=='joint' and owner.options.get('process_sampling',{}).get('max_unique_per_global_minibatch') is not None
    selected_keys={owner.cache_key(p,r) for i,p in enumerate(pairs) for r in ('teacher','student') if selection[r][i]}
    deadline = started + owner.options['judge']['step_timeout_seconds'] if owner.judge else None
    if owner.judge:
        try:owner.judge.health(deadline=deadline)
        except Exception as exc:
            persist_results(owner, getattr(exc, 'results', {}), {}, step, fatal=exc)
            raise
    unique, requests, values = {}, {}, {}
    hits = {'teacher': 0, 'student': 0}
    for pair in pairs:
        for role in ('teacher', 'student'):
            key = owner.cache_key(pair, role)
            if key in unique:continue
            unique[key] = (pair, role)
            cached = owner.cache.get(key) if owner.options[role + '_cache'] else None
            if cached:
                if (cached.get('protocol_identity') != owner.protocol_identity or cached['scorer_fingerprint'] != owner.scorer_identity['fingerprint']
                        or cached['schema_version'] != schema_version or cached['response_id'] != pair[role]['response_id']):
                    raise ValueError('V2.1 cached protocol/scorer/response identity mismatch')
                for dim in DIMS:validate_label(cached['labels'][dim], dim)
                values[key] = cached;hits[role] += 1;continue
            result = owner.scorer(pair[role]['response'], pair['ground_truth'])
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
        results = annotate_chains(owner.judge,requests,contexts,deadline,
            recovered=recovered, on_state=journal.save,
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
    if requests:append_json(owner.directory/'annotation_schedules.jsonl',{'step':step,**owner.judge.last_schedule,
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
    # Effective reporting values are copies; cached mathematical labels stay intact.
    if sampling:
        from copy import deepcopy
        values=deepcopy(values)
        for key,row in values.items():
            if key not in selected_keys:
                for d in DIMS[1:]:row['labels'][d]={'value':None,'valid':False,'source':'not_selected'}
                row.update(unknown_reason='not_selected',process_unavailable_kind='not_selected')
    targets, masks = {}, {}
    for role in ('teacher', 'student'):
        records = [values[owner.cache_key(p, role)]['labels'] for p in pairs]
        targets[role] = torch.tensor([[float(r[d]['value']) if r[d]['valid'] else 0 for d in DIMS] for r in records]).reshape(len(pairs), len(DIMS))
        masks[role] = torch.tensor([[float(r[d]['valid']) for d in DIMS] for r in records]).reshape(len(pairs), len(DIMS))
        if owner.options['mode'] == 'correctness_only':masks[role][:, 1:] = 0
        elif sampling:
            masks[role][:,1:] *= torch.tensor(selection[role],dtype=masks[role].dtype)[:,None]
        batch.batch[role + '_rubric_targets'] = targets[role];batch.batch[role + '_rubric_mask'] = masks[role]
    weights, counts = auxiliary_weights(pairs, targets, masks, global_mini, dp_size)
    for role in weights:batch.batch[role + '_aux_weight'] = weights[role]
    batch.batch['rubric_global_counts'] = counts
    stats = {'rubric/annotation_seconds': time.monotonic() - started, 'rubric/judge_requests': len(requests),
        'rubric/json_valid_count': sum('parsed' in r for r in results.values()), 'rubric/json_requested_count': len(results),
        'rubric/new_annotation_failures': sum('parsed' not in r for r in results.values()),
        'rubric/new_contract_violations': sum(len(r.get('parsed',{}).get('_violations', [])) for r in results.values()),
        'rubric/judge_prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in results.values()),
        'rubric/judge_output_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in results.values()),
        'rubric/http_attempts': sum(r.get('attempts',0) for r in results.values()),
        'rubric/targeted_requests': sum(r.get('targeted_result',{}).get('attempts',0) for r in results.values()),
        'rubric/retry_count': sum(max(0, r.get('attempts', 0) - 1 - r.get('targeted_result',{}).get('attempts',0)) for r in results.values())}
    current_ledger=getattr(owner.judge,'api_ledger',[])[ledger_start:]
    stats['rubric/health_gets_new']=getattr(owner.judge,'health_get_count',0)-health_before
    stats['rubric/new_generation_sends']=len(current_ledger)
    stats['rubric/historical_generation_reservations']=sum(r.get('historical_requests_used',0) for r in recovered.values())
    stats['rubric/historical_submission_unknown']=sum(bool(r.get('historical_submission_unknown')) for r in recovered.values())
    # Canonical client-ledger fields distinguish recovered results from new sends.
    if owner.judge is not None and hasattr(owner.judge,'api_ledger'):
        stats['rubric/http_attempts']=len(current_ledger)
        stats['rubric/targeted_requests']=sum(a['request_id'].endswith(':targeted') for a in current_ledger)
        for metric,key in [('rubric/judge_prompt_tokens','prompt_tokens'),('rubric/judge_output_tokens','completion_tokens')]:
            stats[metric]=sum(a.get('usage',{}).get(key,0) for a in current_ledger)
        stats['rubric/retry_count']=sum(a.get('attempt',1)>1 for a in current_ledger)
        stats['rubric/recovery_retry_sends_new']=sum(
            result.get('annotation_attempt',1)>1 and request_belongs_to(a['request_id'],{key})
            for key,result in results.items() for a in current_ledger)
        for metric,key in [('rubric/historical_judge_prompt_tokens','prompt_tokens'),
                           ('rubric/historical_judge_output_tokens','completion_tokens')]:
            stats[metric]=sum(r.get('historical_usage',{}).get(key,0) for r in recovered.values())
    if sampling:
        stats['rubric/process_sampling_logical_minibatches']=len(selection_groups)
        stats['rubric/process_sampling_selected_unique_sum']=sum(g['selected_unique'] for g in selection_groups)
        stats['rubric/process_sampling_candidate_unique_sum']=sum(g['candidate_unique'] for g in selection_groups)
        stats['rubric/process_sampling_selection_rate']=stats['rubric/process_sampling_selected_unique_sum']/max(1,stats['rubric/process_sampling_candidate_unique_sum'])
    for role in ('teacher', 'student'):
        rows = {supervision_identity(p,role): (p, values[owner.cache_key(p, role)]) for p in pairs}
        stats['rubric/unique_' + role] = len(rows);stats['rubric/pair_occurrences_' + role] = len(pairs)
        stats['rubric/cache_hits_' + role] = hits[role]
        stats['rubric/truncated_' + role] = sum(p[role]['truncated'] for p, _ in rows.values())
        for dim in DIMS:
            valid = [r['labels'][dim]['value'] for _, r in rows.values() if r['labels'][dim]['valid']]
            complete = [(p, r) for p, r in rows.values() if not p[role]['truncated']]
            stats[f'rubric/valid_{dim}_{role}'] = len(valid)
            stats[f'rubric/coverage_{dim}_{role}'] = len(valid) / len(rows) if rows else 0
            stats[f'rubric/nontruncated_coverage_{dim}_{role}'] = sum(r['labels'][dim]['valid'] for _, r in complete) / len(complete) if complete else 0
            stats[f'rubric/aux_weight_mass_{dim}_{role}'] = (weights[role] * masks[role][:, DIMS.index(dim)]).sum().item()
            stats[f'rubric/same_score_{dim}_{role}'] = int(bool(valid) and len(set(valid)) == 1)
            for value in (0, .5, 1, None):
                stats[f'rubric/label_{dim}_{role}_{"unknown" if value is None else value}'] = sum(
                    (not r['labels'][dim]['valid']) if value is None else r['labels'][dim]['valid'] and r['labels'][dim]['value'] == value
                    for _, r in rows.values())
        for _, row in rows.values():
            reason = row['unknown_reason']
            if reason:stats['rubric/unknown_reason/' + reason] = stats.get('rubric/unknown_reason/' + reason, 0) + 1
            label = row['labels'][DIMS[0]]
            if not label['valid']:
                key = 'rubric/correctness_mask_reason/' + label['source'];stats[key] = stats.get(key, 0) + 1
    stats['rubric/teacher_wrong_student_right_count'] = sum(
        values[owner.cache_key(p, 'teacher')]['raw_scorer_result']['correct'] is False and
        values[owner.cache_key(p, 'student')]['raw_scorer_result']['correct'] is True for p in pairs)
    for pair in pairs:pair['labels'] = {role: values[owner.cache_key(pair, role)] for role in ('teacher', 'student')}
    append_json(owner.directory / 'rubric_label_batches.jsonl', {'step': step, 'protocol_identity': owner.protocol_identity,
                'values': values, 'cache_hits': hits, 'stats': stats,
                'process_sampling':selection_groups if sampling else None})
    from .quality_monitor import monitor
    monitor(owner, stats, values, step)
    # All new and cached labels are merged; only final effective process masks
    # participate. This barrier precedes critic/actor optimizer calls.
    final_student_guard(owner, pairs, values, stats, step,selection if sampling else None)
    return stats
