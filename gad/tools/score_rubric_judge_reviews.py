#!/usr/bin/env python3
"""Measure agreement between two frozen judges or against completed blind human reviews."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

DIMENSIONS = ('verified_progress', 'unresolved_error_severity')


def raw_record(sample):
    """Expose complete API JSON for semantic review even if evidence was rejected."""
    api = sample.get('api_response') or {}
    choices = api.get('choices') or []
    if choices and choices[0].get('finish_reason') == 'stop':
        try:
            result = json.loads(choices[0]['message']['content'])
            if isinstance(result, dict):
                if set(result) in (set(DIMENSIONS), {'brief_check', *DIMENSIONS}) and all(
                    result[d] is None or type(result[d]) in (int, float) for d in DIMENSIONS
                ):
                    return {d: {'value': result[d], 'valid': result[d] is not None} for d in DIMENSIONS}
                return result
        except (ValueError, KeyError, TypeError):
            pass
    parsed = sample['result'].get('parsed', {})
    if parsed.get('_process_protocol') in ('direct_v1', 'direct_brief_v1'):
        return parsed
    return parsed.get('_record', {})


def metrics(rows, dimension, raw):
    known_reference = []
    mutual = []
    abstained = failed = 0
    judge_known_on_human_unknown = 0
    severe = Counter()
    confusion = Counter()
    for review, sample in rows:
        target = review[dimension]
        parsed = sample['result'].get('parsed', {})
        predicted = raw_record(sample).get(dimension, {}) if raw else parsed.get(dimension, {})
        known = (isinstance(predicted, dict) and predicted.get('valid') is True and
                 type(predicted.get('value')) in (int, float) and predicted['value'] in (0, .5, 1))
        if not target['valid']:
            judge_known_on_human_unknown += int(known)
            continue
        known_reference.append(target['value'])
        if dimension == 'unresolved_error_severity':
            if target['value'] == 1:
                severe['reference_severe'] += 1
                severe['correct_severe' if known and predicted['value'] == 1 else 'missed_severe' if known else 'abstained_severe'] += 1
                if not known and not parsed:
                    severe['failed_on_severe'] += 1
            else:
                severe['reference_nonsevere'] += 1
                if known and predicted['value'] == 1:
                    severe['false_severe_on_nonsevere'] += 1
            if target['value'] == 0:
                severe['reference_zero'] += 1
                if known and predicted['value'] == 1:
                    severe['false_severe_on_zero'] += 1
        if known:
            mutual.append((target['value'], predicted['value']))
            confusion[f"{target['value']}->{predicted['value']}"] += 1
        else:
            abstained += 1
            failed += int(not parsed)
    severe_rates = {}
    if dimension == 'unresolved_error_severity':
        for metric, numerator, denominator in (
            ('recall_on_human_severe', 'correct_severe', 'reference_severe'),
            ('false_severe_rate_on_human_nonsevere', 'false_severe_on_nonsevere', 'reference_nonsevere'),
        ):
            severe_rates[metric] = severe[numerator] / severe[denominator] if severe[denominator] else None
    return {'reviewed': len(rows), 'human_known': len(known_reference), 'mutually_known': len(mutual),
            'human_unknown': len(rows) - len(known_reference),
            'judge_known_on_human_unknown': judge_known_on_human_unknown,
            'judge_abstained_or_failed_on_human_known': abstained,
            'unusable_responses_on_human_known': failed,
            'invalid_dimensions_on_otherwise_parsed_human_known': abstained - failed,
            'agreement_on_mutually_known': sum(a == b for a, b in mutual) / len(mutual) if mutual else None,
            'mae_on_mutually_known': sum(abs(a-b) for a, b in mutual) / len(mutual) if mutual else None,
            'confusion_human_to_judge': dict(confusion), 'severe_error_counts': dict(severe), **severe_rates}


def real_strata(samples):
    groups = {}
    for rid, sample in samples.items():
        if sample['kind'] != 'real':
            continue
        correct = (sample.get('scoring') or {}).get('correct')
        outcome = 'undetermined' if correct is None else ('correct' if correct else 'incorrect')
        labels = {'role': sample['source_role'], 'verified_answer': outcome,
                  'truncation': 'truncated' if sample['truncated'] else 'complete'}
        if sample.get('category'):
            labels['category'] = sample['category']
        for axis, label in labels.items():
            groups.setdefault(axis, {}).setdefault(label, []).append(rid)
    return groups


def human_metrics(rows):
    return {mode: {dim: metrics(rows, dim, mode == 'raw') for dim in DIMENSIONS}
            for mode in ('raw', 'effective')}


def effective_label(sample, dimension):
    parsed = sample['result'].get('parsed')
    if not parsed:
        return 'failed'
    label = parsed[dimension]
    return label['value'] if label['valid'] else 'null'


def paired_student_ids(samples):
    """Collect completed, verifier-graded correct/incorrect Student answers by UID."""
    groups = {}
    for rid, row in sorted(samples.items()):
        uid = row.get('uid')
        scorer = row.get('scoring') or {}
        correct = scorer.get('correct')
        if (row.get('kind') != 'real' or row.get('source_role') != 'student' or row.get('truncated') is not False
                or not isinstance(uid, str) or not uid.strip() or type(correct) is not bool
                or scorer.get('grade_status') != ('correct' if correct else 'incorrect')
                or scorer.get('parse_status') != 'ok'):
            continue
        outcome = 'correct' if correct else 'incorrect'
        # Keep the first stable ID for duplicate text from another checkpoint.
        by_outcome = groups.setdefault(uid, {'correct': {}, 'incorrect': {}})
        by_outcome[outcome].setdefault(row['response'], rid)
    return {uid: {outcome: list(ids.values()) for outcome, ids in sides.items()}
            for uid, sides in groups.items() if all(sides.values())}


def student_delta(samples, sides, dimension):
    labels = {outcome: [effective_label(samples[rid], dimension) for rid in ids]
              for outcome, ids in sides.items()}
    if any('failed' in values for values in labels.values()):
        return 'failed'
    if any('null' in values for values in labels.values()):
        return 'abstained'
    progress_difference = mean(labels['correct']) - mean(labels['incorrect'])
    return progress_difference if dimension == 'verified_progress' else -progress_difference


def delta_summary(values):
    values = list(values)
    valid = [value for value in values if isinstance(value, (int, float))]
    return {'questions': len(values), 'valid': len(valid), 'abstained': values.count('abstained'),
            'failed': values.count('failed'), 'positive': sum(v > 0 for v in valid),
            'tie': sum(v == 0 for v in valid), 'negative': sum(v < 0 for v in valid),
            'mean_delta': mean(valid) if valid else None}


def within_question_student(samples, other=None):
    pairs = paired_student_ids(samples)
    report = {
        'eligible_questions': len(pairs), 'response_ids_by_question': pairs,
        'weighting': 'Average distinct response texts within each answer outcome; weight each question equally. '
                     'Require every included rating on both outcomes for an arm and dimension. '
                     'A failed response takes precedence over a null rating. Final-answer correctness groups the '
                     'responses; the sign of a process-label difference is descriptive.',
        'directions': {'verified_progress': 'correct minus incorrect',
                       'unresolved_error_severity': 'incorrect minus correct'},
        'dimensions': {},
    }
    for dim in DIMENSIONS:
        first = {uid: student_delta(samples, sides, dim) for uid, sides in pairs.items()}
        scores = {'benchmark': delta_summary(first.values())}
        if other is not None:
            second = {uid: student_delta(other, sides, dim) for uid, sides in pairs.items()}
            common = [uid for uid in pairs if isinstance(first[uid], (int, float)) and isinstance(second[uid], (int, float))]
            scores.update(comparison=delta_summary(second.values()), both_arms_known={
                'questions': len(common), 'benchmark': delta_summary(first[uid] for uid in common),
                'comparison': delta_summary(second[uid] for uid in common),
                'mean_comparison_minus_benchmark': mean(second[uid] - first[uid] for uid in common) if common else None,
            })
        report['dimensions'][dim] = scores
    return report


def compare_arms(left, right):
    """Paired agreement only; either judge can be wrong."""
    if not left or left.keys() != right.keys():
        raise ValueError('the two runs must contain the same nonempty set of response IDs')
    for rid, a in left.items():
        b = right[rid]
        if (any(a[k] != b[k] for k in ('prompt', 'response', 'ground_truth', 'truncated', 'source_role'))
                or a.get('uid') != b.get('uid') or a.get('kind') != b.get('kind')
                or any((a.get('scoring') or {}).get(k) != (b.get('scoring') or {}).get(k)
                       for k in ('correct', 'grade_status', 'parse_status'))
                or a.get('category') != b.get('category')):
            raise ValueError(f'paired inputs differ for {rid}')
    results = {}
    for dim in DIMENSIONS:
        confusion = Counter()
        differences = []
        known_left = known_right = exact = 0
        disagreement_ids = []
        for rid, a in left.items():
            x, y = effective_label(a, dim), effective_label(right[rid], dim)
            known_left += isinstance(x, (int, float))
            known_right += isinstance(y, (int, float))
            confusion[f'{x} -> {y}'] += 1
            if x != y:
                disagreement_ids.append(rid)
            if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                differences.append(abs(x-y))
                exact += int(x == y)
        mutual = len(differences)
        results[dim] = dict(rows=len(left), first_known=known_left, second_known=known_right,
                            mutually_known=mutual, mutually_known_exact=exact,
                            exact_agreement=exact/mutual if mutual else None,
                            mean_absolute_difference=sum(differences)/mutual if mutual else None,
                            first_to_second=dict(confusion), disagreement_ids=disagreement_ids)
    return results


def load_samples(directory):
    rows = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    by_id = {row['id']: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError(f'duplicate response ID in {directory}')
    return by_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reviews', help='completed blind human reviews')
    mode.add_argument('--compare', help='second judge output directory on the same frozen inputs')
    args = parser.parse_args()
    directory = Path(args.benchmark)
    samples = load_samples(directory)
    if args.compare:
        second = Path(args.compare)
        other = load_samples(second)
        report = compare_arms(samples, other)
        report['benchmark_directory'], report['comparison_directory'] = str(directory), str(second)
        report['within_question_student'] = within_question_student(samples, other)
        report['real_subgroups'] = {
            axis: {group: compare_arms({rid: samples[rid] for rid in ids}, {rid: other[rid] for rid in ids})
                   for group, ids in groups.items()} for axis, groups in real_strata(samples).items()}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    blind = {r['review_id']: r for r in (json.loads(line) for line in (directory / 'human_review_blind.jsonl').read_text().splitlines())}
    key = json.loads((directory / 'human_review_key.json').read_text())
    completed = []
    seen = set()
    for line in Path(args.reviews).read_text().splitlines():
        row = json.loads(line)
        rid = row['review_id']
        if rid not in blind or rid in seen:
            raise ValueError(f'unknown or duplicate review ID: {rid}')
        seen.add(rid)
        for field in ('problem', 'visible_response', 'reference_final_answer', 'response_truncated'):
            if row[field] != blind[rid][field]:
                raise ValueError(f'review input changed: {rid}/{field}')
        if row.get('review_status') != 'completed':
            continue
        if not isinstance(row.get('reviewer'), str) or not row['reviewer'].strip():
            raise ValueError(f'reviewer is required: {rid}')
        for dimension in DIMENSIONS:
            label = row[dimension]
            if type(label['valid']) is not bool:
                raise ValueError(f'valid must be boolean: {rid}/{dimension}')
            value = label['value']
            if label['valid'] and (type(value) not in (int, float) or value not in (0, .5, 1)):
                raise ValueError(f'invalid ordinal value: {rid}/{dimension}')
            if not label['valid'] and value is not None:
                raise ValueError(f'unknown must have null value: {rid}/{dimension}')
            if not isinstance(label.get('reason'), str) or not label['reason'].strip():
                raise ValueError(f'reason required, including unknowns: {rid}/{dimension}')
        completed.append((row, samples[key[rid]]))
    if not completed:
        raise ValueError('no completed human reviews; automatic label statistics are not semantic accuracy')
    reviewed = {sample['id']: (review, sample) for review, sample in completed}
    real_groups = real_strata({rid: sample for rid, (_, sample) in reviewed.items()})
    report = {'human_reviews_completed': len(completed), 'human_reviews_requested': len(blind),
              'human_review_status': 'complete' if len(completed) == len(blind) else 'partial',
              'groups': {kind: human_metrics([pair for pair in completed if pair[1]['kind'] == kind])
                         for kind in ('real', 'boundary')},
              'real_subgroups': {axis: {group: human_metrics([reviewed[rid] for rid in ids])
                                        for group, ids in groups.items()} for axis, groups in real_groups.items()},
              'caveat': 'Conditional agreement excludes abstentions; review all associated counts. Raw uses complete API JSON, including evidence-rejected annotations, for semantic diagnosis only. Effective uses all program checks. Stratified diagnostic sample is not population-weighted.'}
    target = directory / 'human_review_metrics.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
