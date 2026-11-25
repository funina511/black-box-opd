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


def final_student_guard(owner, pairs, entries, stats, step, process_selected=None):
    for role in ('teacher', 'student'):
        rows = {supervision_identity(p,role): (p, entries[role][i]) for i,p in enumerate(pairs)
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


def annotate(owner, pairs, batch, global_mini, dp_size, step, *, scorer_results=None):
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
    row_keys = {role: [owner.cache_key(pair, role) for pair in pairs] for role in ('teacher', 'student')}
    selected_keys = {key for role, keys in row_keys.items()
                     for key, selected in zip(keys, selection[role]) if selected}
    deadline = started + owner.options['judge']['step_timeout_seconds'] if owner.judge else None
    if owner.judge:
        try:owner.judge.health(deadline=deadline)
        except Exception as exc:
            persist_results(owner, getattr(exc, 'results', {}), {}, step, fatal=exc)
            raise
    from .annotation_work import build_labels, execute_requests
    unique, requests, values, hits = build_labels(owner, pairs, selected_keys, schema_version,
                                                  scorer_results=scorer_results, row_keys=row_keys)
    owner.last_selected_teacher_keys = {key for key, selected in zip(row_keys['teacher'], selection['teacher']) if selected}
    results, recovered = execute_requests(owner, unique, requests, values, step, deadline)
    # Effective reporting values are copies; cached mathematical labels stay intact.
    if sampling:
        from copy import deepcopy
        values=deepcopy(values)
        for key,row in values.items():
            if key not in selected_keys:
                for d in DIMS[1:]:row['labels'][d]={'value':None,'valid':False,'source':'not_selected'}
                row.update(unknown_reason='not_selected',process_unavailable_kind='not_selected')
    entries = {role: [values[key] for key in keys] for role, keys in row_keys.items()}
    targets, masks = {}, {}
    for role in ('teacher', 'student'):
        records = [entry['labels'] for entry in entries[role]]
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
        rows = {supervision_identity(p,role): (p, entry) for p, entry in zip(pairs, entries[role])}
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
        teacher['raw_scorer_result']['correct'] is False and student['raw_scorer_result']['correct'] is True
        for teacher, student in zip(entries['teacher'], entries['student']))
    for i, pair in enumerate(pairs):
        pair['labels'] = {role: rows[i] for role, rows in entries.items()}
    append_json(owner.directory / 'rubric_label_batches.jsonl', {'step': step, 'protocol_identity': owner.protocol_identity,
                'values': values, 'cache_hits': hits, 'stats': stats,
                'process_sampling':selection_groups if sampling else None})
    from .quality_monitor import monitor
    monitor(owner, stats, values, step)
    # All new and cached labels are merged; only final effective process masks
    # participate. This barrier precedes critic/actor optimizer calls.
    final_student_guard(owner, pairs, entries, stats, step, selection if sampling else None)
    return stats
