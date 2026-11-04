#!/usr/bin/env python3
"""Pure CPU unified grading, fixed denominator metrics and bounded legacy audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from scoring.math import score, identity, evaluation_value, evaluation_status, validate_ground_truth
try:
    from scoring.math import ground_truth_semantics
except ImportError:  # pragma: no cover - only for the historical scorer without GT semantics
    ground_truth_semantics = None

#: Evaluation aggregation protocol identity (recorded in every metrics.json).
AGGREGATION_PROTOCOL = {
    'revision': 'eval-fix.2',
    'denominator': 'fixed: len(expected_questions) * n',
    'complete': 'sample_index_complete and U == 0',
    'interval': 'lower=C/N, upper=(C+U)/N, coverage=(C+W)/N',
    'status_mapping': 'scoring.math.evaluation_status',
    'format_failure': 'evaluation 0, training Mask unchanged',
    'provisional_metrics': 'compatibility fields only; all equal sample_accuracy_lower',
}


def _normalise_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _row_evaluation_value(row):
    """Evaluation value for a scored row; prefers the stored field when present."""
    if isinstance(row, dict) and 'evaluation_value' in row:
        return _normalise_value(row.get('evaluation_value'))
    return _normalise_value(evaluation_value(row))


def _row_evaluation_status(row):
    if isinstance(row, dict) and row.get('evaluation_status') is not None:
        return row['evaluation_status']
    return evaluation_status(row)


def _ground_truth_audit(records, inspector=None):
    """Prediction-independent GT audit keyed by question id (plan section 4.3)."""
    per_question = defaultdict(set)
    for row in records:
        uid = str(row.get('global_index', row.get('uid')))
        canonical = json.dumps(row.get('ground_truth'), sort_keys=True, default=str)
        per_question[uid].add(canonical)
    conflicts = sorted(uid for uid, values in per_question.items() if len(values) > 1)
    unsupported = []
    checked = True
    inspector = inspector or globals().get('ground_truth_semantics')
    if not callable(inspector):
        checked = False
    else:
        for uid in sorted(per_question):
            first = sorted(per_question[uid])[0]
            try:
                ground_truth = json.loads(first)
            except ValueError:
                ground_truth = first
            try:
                semantics = inspector(ground_truth)
            except Exception as exc:  # noqa: BLE001 - one bad GT must not abort the audit
                semantics = {'supported': False, 'reason': '%s: %s' % (type(exc).__name__, exc)}
            if isinstance(semantics, dict) and not semantics.get('supported', False):
                unsupported.append({'question_id': uid, 'reason': semantics.get('reason')})
    return {
        'checked': checked,
        'questions': len(per_question),
        'conflict_questions': len(conflicts),
        'unsupported_questions': len(unsupported),
        'details': (unsupported[:20] + [{'question_id': uid, 'reason': 'ground truth differs between rollouts'}
                                        for uid in conflicts[:20]]),
        'conflict_question_ids': conflicts[:50],
        'unsupported_question_ids': [item['question_id'] for item in unsupported[:50]],
    }


def _unresolved_result(grade_status, reason):
    try:
        info = identity()
        version, fingerprint = info.get('version'), info.get('fingerprint')
    except Exception:  # noqa: BLE001 - identity is diagnostic here
        version, fingerprint = None, None
    return {'scorer_version': version, 'scorer_fingerprint': fingerprint, 'pred': None,
            'parse_status': grade_status, 'parse_method': 'ground_truth_audit', 'candidate_count': 0,
            'error_reason': reason, 'correct': None, 'grade_status': grade_status}


def summarize(rows, *, expected_questions=None, n=None):
    if n is not None and (type(n) is not int or n < 1):raise ValueError('n must be a positive integer')
    authoritative = expected_questions is not None
    expected_list=list(map(str,expected_questions)) if authoritative else None
    if expected_list is not None and len(set(expected_list))!=len(expected_list):raise ValueError('duplicate expected question ID')
    expected=set(expected_list) if expected_list is not None else {str(r.get('global_index',r.get('uid'))) for r in rows if r.get('global_index',r.get('uid')) is not None}
    buckets=defaultdict(list);invalid_indices=0;out_of_range=0;unknown=set();unidentified=0
    if n is None:
        indices=[r.get('rollout_index',r.get('sample_index',0)) for r in rows]
        n=max((v+1 for v in indices if type(v) is int and v>=0),default=1)
    for row in rows:
        raw_uid=row.get('global_index',row.get('uid'))
        if raw_uid is None:unidentified+=1;continue
        uid=str(raw_uid)
        rollout=row.get('rollout_index',row.get('sample_index'))
        if uid not in expected:unknown.add(uid);continue
        if type(rollout) is not int:invalid_indices+=1;continue
        if not 0<=rollout<n:out_of_range+=1;continue
        buckets[(uid,rollout)].append(row)
    groups=defaultdict(dict);duplicates=0;conflicts=0
    for (uid,rollout),records in buckets.items():
        duplicates+=len(records)-1
        # Identical duplicates count once; conflicting records contribute nothing.
        if any(r!=records[0] for r in records[1:]):conflicts+=1;continue
        groups[uid][rollout]=records[0]
    missing=expected-set(groups)
    missing_slots=sum(n-len(groups[uid]) for uid in expected)
    sample_index_complete=not (not expected or duplicates or missing_slots or unknown or invalid_indices
                               or out_of_range or conflicts or unidentified or not authoritative)
    samples=[r for uid in expected for r in groups[uid].values()]
    errors=sum(r.get('grade_status') in ('error','timeout') for r in samples)
    values=[_row_evaluation_value(r) for r in samples]
    unscorable=sum(value is None for value in values)
    confirmed_correct=sum(value==1 for value in values)
    confirmed_incorrect=sum(value==0 for value in values)
    denominator=len(expected)*n
    unresolved=denominator-confirmed_correct-confirmed_incorrect
    lower=confirmed_correct/denominator if denominator else None
    upper=(confirmed_correct+unresolved)/denominator if denominator else None
    coverage=(confirmed_correct+confirmed_incorrect)/denominator if denominator else None
    complete=bool(sample_index_complete and unresolved==0)
    proportions=[sum(1 for r in groups[uid].values() if _row_evaluation_value(r)==1)/n for uid in expected]
    first=sum(1 if (0 in groups[uid] and _row_evaluation_value(groups[uid][0])==1) else 0 for uid in expected)
    passed=sum(any(_row_evaluation_value(r)==1 for r in groups[uid].values()) for uid in expected)
    lengths=[r.get('output_tokens',r.get('num_tokens',0)) for r in samples]
    unresolved_reasons=Counter(_row_evaluation_status(r) for r in samples if _row_evaluation_value(r) is None)
    sample_accuracy_note=('确定值 C/N（sample_index_complete 且 U==0）' if complete else
        'None：存在未决（U=%d）或采样索引不完整；请使用 sample_accuracy_lower/upper 与 coverage' % unresolved)
    result={'num_prompts':len(expected),'num_samples':len(samples),'expected_samples':denominator,
        'samples_per_question':n,'duplicate_samples':duplicates,'conflicting_sample_keys':conflicts,
        'invalid_rollout_indices':invalid_indices,'out_of_range_rollouts':out_of_range,'missing_samples':missing_slots,
        'missing_questions':len(missing),'unexpected_questions':len(unknown),'unidentified_rows':unidentified,
        'expected_questions_source':'explicit' if authoritative else 'observed_rows',
        'authoritative_expected':authoritative,
        'sample_index_complete':sample_index_complete,'evaluation_complete':complete,'strict_complete':complete,
        'confirmed_correct':confirmed_correct,'confirmed_incorrect':confirmed_incorrect,'unresolved_samples':unresolved,
        'sample_accuracy_lower':lower,'sample_accuracy_upper':upper,'coverage':coverage,
        'sample_accuracy':lower if complete else None,'sample_accuracy_note':sample_accuracy_note,
        'pass_at_1':lower,
        'pass_at_1_note':('sample_accuracy 的兼容别名；未决/索引不完整时等于 sample_accuracy_lower=C/N（U=%d）' % unresolved
                          if not complete else 'sample_accuracy 的兼容别名（此处等于确定值 C/N）'),
        'first_rollout_accuracy':first/len(expected) if expected else None,
        'any_success_at_n':passed/len(expected) if expected else None,
        'parse_failure_rate':sum(r.get('parse_status') in ('malformed','ambiguous') for r in samples)/denominator if denominator else 0,
        'no_answer_rate':sum(r.get('parse_status')=='no_answer' for r in samples)/denominator if denominator else 0,
        'grader_error_count':errors,'grader_error_rate':errors/denominator if denominator else 0,
        'unscorable_sample_count':unscorable,'scorable_sample_count':len(samples)-unscorable,
        'truncated_rate':sum(r.get('finish_reason')=='length' for r in samples)/denominator if denominator else 0,
        'output_tokens_mean':statistics.fmean(lengths) if lengths else 0,
        'unresolved_reason_counts':dict(unresolved_reasons),
        'scorer_version':samples[0].get('scorer_version') if samples else None,
        'scorer_fingerprint':samples[0].get('scorer_fingerprint') if samples else None,
        'aggregation_protocol':dict(AGGREGATION_PROTOCOL),
        'compatibility_mappings':{'sample_accuracy_avg_at_n':'sample_accuracy','best_of_n_accuracy':'any_success_at_n'}}
    result['provisional_metrics']={'pass_at_1':lower,'sample_accuracy':lower,
        'first_rollout_accuracy':lower,'any_success_at_n':lower}
    result['provisional_metrics_policy']=('Fixed expected slots; absent, unresolved and conflicting slots '
        'contribute zero only to provisional lower bounds (provisional_metrics 四项均等于 '
        'sample_accuracy_lower=C/N，不是另一套公式)。')
    return result


def run(raw_path, output_dir, config=None, expected_questions=None, n=None, task=None):
    from evaluate import answer_forms, choice, minerva
    # Opt-in by fingerprint so profiles planned before answer_forms keep their scoring.
    use_forms = task in answer_forms.TASKS and bool(config and config.get('answer_forms_fingerprint'))
    forms_identity = answer_forms.identity() if use_forms else None
    if use_forms and config['answer_forms_fingerprint'] != forms_identity['fingerprint']:
        raise ValueError('answer-form fingerprint mismatch')
    is_choice = task in choice.TASKS
    choice_identity = choice.identity() if is_choice else None
    if is_choice and config and config.get('choice_fingerprint') not in (None, choice_identity['fingerprint']):
        raise ValueError('multiple-choice evaluation scorer fingerprint mismatch')
    is_minerva = task in minerva.TASKS
    minerva_identity = minerva.identity() if is_minerva else None
    if is_minerva and config and config.get('minerva_fingerprint') not in (None, minerva_identity['fingerprint']):
        raise ValueError('Minerva evaluation post-process fingerprint mismatch')
    output_dir = Path(output_dir)
    if config and config.get('fingerprint') and config['fingerprint'] != identity(config.get('timeout_seconds', 2.0))['fingerprint']:
        raise ValueError('rescore snapshot scorer/dependency fingerprint mismatch')
    if (output_dir / 'metrics.json').exists():raise ValueError('new scoring output required; refusing overwrite of metrics')
    with Path(raw_path).open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    audit = _ground_truth_audit(records, inspector=(lambda gt: {
        'supported': isinstance(gt, str) and len(gt.strip()) == 1 and gt.strip().upper() in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'reason': 'expected one option letter'}) if is_choice else
        answer_forms.ground_truth_semantics if use_forms else None)
    conflict_questions = set(audit.get('conflict_question_ids') or [])
    unsupported_questions = set(audit.get('unsupported_question_ids') or [])
    scored = []
    for row in records:
        uid = str(row.get('global_index', row.get('uid')))
        if uid in conflict_questions:
            result = _unresolved_result('gt_unparseable', 'ground truth differs between rollouts')
        elif uid in unsupported_questions:
            result = _unresolved_result('gt_unparseable', 'ground truth unsupported by this evaluation scorer')
        elif is_choice:
            result = choice.score(row, task)
        elif use_forms:
            result = answer_forms.score(row.get('response', ''), row.get('ground_truth'), config, task=task)
        else:
            result = (minerva.score if is_minerva else score)(row.get('response', ''), row.get('ground_truth'), config)
        if is_choice:
            result.update(scorer_version=choice_identity['version'], scorer_fingerprint=choice_identity['fingerprint'])
        scored.append({**row, **result})
    metrics = summarize(scored, expected_questions=expected_questions, n=n)
    if is_choice:
        metrics['aggregation_protocol']['status_mapping'] = 'evaluate.choice: explicit final option exact match'
        content_correct = sum(r.get('content_option_correct') is True for r in scored)
        # Diagnostic only: option text written instead of the letter; never part of the primary interval.
        metrics['choice_content_diagnostic'] = {
            'format_failure_or_no_answer': sum(r.get('evaluation_status') in ('format_failure', 'no_answer') for r in scored),
            'content_matched': sum(r.get('content_option') is not None for r in scored),
            'content_matched_correct': content_correct,
            'accuracy_lower_if_content_counted': ((metrics['confirmed_correct'] + content_correct) / metrics['expected_samples']
                                                  if metrics['expected_samples'] else None)}
    if is_minerva:
        metrics['aggregation_protocol']['status_mapping'] += ' + evaluate.minerva trailing-unit rule'
        metrics['eval_postprocess'] = minerva_identity
        metrics['unit_stripped_samples'] = sum(bool(r.get('unit_stripped')) for r in scored)
        metrics['unit_stripped_correct'] = sum(bool(r.get('unit_stripped')) and r.get('grade_status') == 'correct' for r in scored)
    if use_forms:
        metrics['aggregation_protocol']['status_mapping'] += ' + evaluate.answer_forms rules'
        metrics['answer_forms'] = forms_identity
        metrics['answer_form_changed_samples'] = sum(bool(r.get('answer_form_applied')) for r in scored)
        metrics['answer_form_correct'] = sum(bool(r.get('answer_form_applied')) and r.get('grade_status') == 'correct' for r in scored)
    metrics['task'] = task
    metrics['ground_truth_audit'] = audit
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'scored_samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in scored))
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def audit(paths, output, limit=128):
    if not 0 < limit <= 128:raise ValueError('bounded history audit limit must be 1..128')
    from sample.math_scorer import compute_score as legacy
    candidates = []
    for path in paths:
        # Prefer actual cross-line Answer failures, then a deterministic spread.
        chosen, rest = [], []
        with Path(path).open() as stream:
            for i, line in enumerate(stream):
                row = json.loads(line)
                if '\n' in row.get('response', '') and __import__('re').search(r'(?i)Answer:[ \t]*\n', row.get('response', '')):
                    if len(chosen) < limit:chosen.append((path, row))
                elif i % 37 == 0 and len(rest) < limit:rest.append((path, row))
        per_group = max(1, limit // len(paths));candidates += (chosen + rest)[:per_group]
    rows = []
    for path, row in candidates[:limit]:
        current = score(row.get('response', ''), row['ground_truth']);old = legacy(row.get('response', ''), row['ground_truth'])
        reason = ('cross_line_answer_extraction' if current['parse_method'] == 'answer_next_line' else
                  'last_explicit_marker_priority' if current['candidate_count'] > 1 else
                  'mathematical_normalization' if current['pred'] != old.get('pred') else 'grading_or_consumer_policy')
        current_value = _row_evaluation_value(current)
        rows.append({'source': str(path), 'global_index': row.get('global_index'), 'rollout_index': row.get('rollout_index'),
            'response': row.get('response', ''), 'ground_truth': row['ground_truth'], 'legacy': old, 'unified': current,
            'disagrees': (bool(old.get('acc')) and current_value == 0) or (not bool(old.get('acc')) and current_value == 1),
            'unresolved': current_value is None, 'difference_reason': reason})
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    (output / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary = {'answers': len(rows), 'by_source': dict(Counter(r['source'] for r in rows)),
        'disagreements': sum(r['disagrees'] for r in rows), 'unresolved': sum(r['unresolved'] for r in rows),
        'reasons': dict(Counter(r['difference_reason'] for r in rows if r['disagrees'])),
        'new_grader_errors': sum(r['unified']['grade_status'] in ('error', 'timeout') for r in rows),
        'identity': identity(), 'existing_responses_unchanged': True, 'human_review': 'pending'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2));return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-jsonl', action='append', required=True)
    p.add_argument('--output-dir', required=True);p.add_argument('--config-json');p.add_argument('--n', type=int)
    expected=p.add_mutually_exclusive_group()
    expected.add_argument('--expected-prompts',type=int,help='Explicit legacy contiguous-ID mode (0..N-1)')
    expected.add_argument('--expected-ids',type=Path,help='JSON list of actual sampler question IDs')
    p.add_argument('--task');p.add_argument('--audit', action='store_true');p.add_argument('--limit', type=int, default=128)
    a = p.parse_args()
    result = audit(a.raw_jsonl, a.output_dir, a.limit) if a.audit else run(a.raw_jsonl[0], a.output_dir,
        json.loads(a.config_json) if a.config_json else None,
        json.loads(a.expected_ids.read_text()) if a.expected_ids else range(a.expected_prompts) if a.expected_prompts is not None else None, a.n, a.task)
    print(json.dumps(result, indent=2))
