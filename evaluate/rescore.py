#!/usr/bin/env python3
"""Pure CPU unified grading, fixed denominator metrics and bounded legacy audit."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from scoring.math import score, identity, evaluation_value, validate_ground_truth


def summarize(rows, *, expected_questions=None, n=None):
    groups = defaultdict(dict);duplicates = 0
    for row in rows:
        uid = str(row.get('global_index', row.get('uid'))); rollout = int(row.get('rollout_index', row.get('sample_index', 0)))
        if rollout in groups[uid]:duplicates += 1
        else:groups[uid][rollout] = row
    expected = set(map(str, expected_questions)) if expected_questions is not None else set(groups)
    unknown = set(groups) - expected;missing = expected - set(groups)
    incomplete = bool(duplicates or missing or unknown)
    for uid in expected:
        if n is not None and set(groups[uid]) != set(range(n)):incomplete = True
    samples = [r for uid in expected for r in groups[uid].values()]
    errors = sum(r['grade_status'] in ('error', 'timeout') for r in samples)
    unscorable = sum(evaluation_value(r) is None for r in samples)
    denominator = len(expected) * n if n is not None else len(samples)
    proportions = [sum(evaluation_value(r) or 0 for r in groups[uid].values()) / (n or len(groups[uid]) or 1) for uid in expected]
    first = sum((evaluation_value(groups[uid][0]) or 0) if 0 in groups[uid] else 0 for uid in expected)
    passed = sum(any(r['correct'] is True for r in groups[uid].values()) for uid in expected)
    lengths = [r.get('output_tokens', r.get('num_tokens', 0)) for r in samples]
    result = {'num_prompts': len(expected), 'num_samples': len(samples), 'expected_samples': denominator,
        'samples_per_question': n, 'duplicate_samples': duplicates, 'missing_questions': len(missing),
        'unexpected_questions': len(unknown), 'sample_index_complete': not incomplete,
        'evaluation_complete': not incomplete and unscorable == 0,
        'pass_at_1': statistics.fmean(proportions) if proportions else None,
        'sample_accuracy': sum(evaluation_value(r) or 0 for r in samples) / denominator if denominator else None,
        'first_rollout_accuracy': first / len(expected) if expected else None,
        'any_success_at_n': passed / len(expected) if expected else None,
        'parse_failure_rate': sum(r['parse_status'] in ('malformed', 'ambiguous') for r in samples) / denominator if denominator else 0,
        'no_answer_rate': sum(r['parse_status'] == 'no_answer' for r in samples) / denominator if denominator else 0,
        'grader_error_count': errors, 'grader_error_rate': errors / denominator if denominator else 0,
        'unscorable_sample_count': unscorable, 'scorable_sample_count': len(samples) - unscorable,
        'truncated_rate': sum(r.get('finish_reason') == 'length' for r in samples) / denominator if denominator else 0,
        'output_tokens_mean': statistics.fmean(lengths) if lengths else 0,
        'scorer_version': samples[0]['scorer_version'] if samples else None,
        'scorer_fingerprint': samples[0]['scorer_fingerprint'] if samples else None,
        'compatibility_mappings': {'sample_accuracy_avg_at_n': 'sample_accuracy', 'best_of_n_accuracy': 'any_success_at_n'}}
    if incomplete or unscorable:
        result['provisional_metrics'] = {k: result.pop(k) for k in ('pass_at_1', 'sample_accuracy', 'first_rollout_accuracy', 'any_success_at_n')}
        result['provisional_metrics_policy'] = 'Fixed expected denominator; unknown samples contribute zero only to provisional lower bounds.'
    return result


def run(raw_path, output_dir, config=None, expected_questions=None, n=None, task=None):
    output_dir = Path(output_dir)
    if config and config.get('fingerprint') and config['fingerprint'] != identity(config.get('timeout_seconds', 2.0))['fingerprint']:
        raise ValueError('rescore snapshot scorer/dependency fingerprint mismatch')
    if (output_dir / 'metrics.json').exists():raise ValueError('new scoring output required; refusing overwrite of metrics')
    records = [json.loads(line) for line in Path(raw_path).open() if line.strip()]
    scored = [{**row, **score(row.get('response', ''), row.get('ground_truth'), config)} for row in records]
    metrics = summarize(scored, expected_questions=expected_questions, n=n)
    metrics['task'] = task
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
        rows.append({'source': str(path), 'global_index': row.get('global_index'), 'rollout_index': row.get('rollout_index'),
            'response': row.get('response', ''), 'ground_truth': row['ground_truth'], 'legacy': old, 'unified': current,
            'disagrees': bool(old.get('acc')) != bool(evaluation_value(current)), 'difference_reason': reason})
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    (output / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary = {'answers': len(rows), 'by_source': dict(Counter(r['source'] for r in rows)),
        'disagreements': sum(r['disagrees'] for r in rows), 'reasons': dict(Counter(r['difference_reason'] for r in rows if r['disagrees'])),
        'new_grader_errors': sum(r['unified']['grade_status'] in ('error', 'timeout') for r in rows),
        'identity': identity(), 'existing_responses_unchanged': True, 'human_review': 'pending'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2));return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-jsonl', action='append', required=True)
    p.add_argument('--output-dir', required=True);p.add_argument('--config-json');p.add_argument('--n', type=int)
    p.add_argument('--expected-prompts', type=int);p.add_argument('--task');p.add_argument('--audit', action='store_true');p.add_argument('--limit', type=int, default=128)
    a = p.parse_args()
    result = audit(a.raw_jsonl, a.output_dir, a.limit) if a.audit else run(a.raw_jsonl[0], a.output_dir,
        json.loads(a.config_json) if a.config_json else None, range(a.expected_prompts) if a.expected_prompts is not None else None, a.n, a.task)
    print(json.dumps(result, indent=2))
