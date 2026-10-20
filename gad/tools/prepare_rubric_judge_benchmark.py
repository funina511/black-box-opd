#!/usr/bin/env python3
"""Freeze 48 real training pairs and 32 independent boundary variations."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rubric.core.contracts import digest, file_hash
from scoring.math import score, identity


def boundaries():
    rows = []
    for variant, (a, b, x) in enumerate(((3, 5, 4), (4, 7, 5), (5, 2, 3), (6, 7, 2))):
        rhs = a * x + b
        problem = f'Solve {a}x+{b}={rhs} over real numbers.'
        prefix = f'Subtract {b}: {a}x={a*x}.'
        bad = x + 1
        cases = [
            ('valid_prefix', problem, prefix, str(x), True, [.5], [0]),
            ('false_repair', problem, f'{prefix}\nDividing gives x={bad}.\nThe error is now corrected.', str(x), False, [.5], [1]),
            ('real_repair', problem, f'{prefix}\nI first wrote x={bad}; that division was wrong.\nCorrecting: x={x}. Then {a}*{x}+{b}={rhs}.\nAnswer: {x}', str(x), False, [1], [0]),
            ('pending_branch', f'Solve (x-{x})(x-{bad})=0 over real numbers.',
             f'For the x!={x} branch, division by (x-{x}) gives x-{bad}=0, hence x={bad} in this branch.',
             '{'+str(x)+','+str(bad)+'}', True, [.5], [0]),
            ('no_math', problem, f'We need to solve {a}x+{b}={rhs} over real numbers.' if variant % 2 == 0 else 'I will think about a way to solve this.', str(x), False, [0], [None]),
            ('repeat', problem, '\n'.join([prefix] * (variant+2)), str(x), True, [.5], [0]),
            ('partial_formula', problem, f'{prefix}\nx={a*x}/', str(x), True, [.5], [0, None]),
            ('explicit_error', problem, f'{prefix}\nDividing by {a} gives x={bad}.\nAnswer: {bad}', str(x), False, [.5], [1]),
        ]
        for category, prompt, response, gt, truncated, progress, error in cases:
            rows.append({'id': f'boundary_{category}_{variant}', 'kind': 'boundary', 'category': category,
                         'prompt': prompt, 'response': response, 'ground_truth': gt, 'truncated': truncated,
                         'expected_allowed': {'verified_progress': progress, 'unresolved_error_severity': error},
                         'expectation_status': 'model_assisted_anchor_human_pending'})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=str(ROOT / 'data/rubric_math_v1/math_v1_512q_seed42_2631677456f2/trajectories.jsonl'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'inputs.jsonl').exists():
        raise ValueError('frozen inputs already exist; use a new output directory')
    source = Path(args.source).resolve()
    groups = defaultdict(list)
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row['split'] == 'train' and row['input_eligible'] and row['response'].strip():
            groups[row['uid']].append(row)
    rng = random.Random(args.seed)
    pairs = []
    for uid, group in sorted(groups.items()):
        teacher = [r for r in group if r['source_role'] == 'teacher']
        students = sorted((r for r in group if r['source_role'] == 'student'), key=lambda r: r['response_id'])
        if len(teacher) == 1 and students:
            pairs.append((teacher[0], rng.choice(students)))
    rng.shuffle(pairs)
    buckets = defaultdict(list)
    scoring_audit = []
    # Only completed student answers need grading for selection; selected
    # truncated answers are graded later for descriptive outcome statistics.
    for teacher, student in pairs:
        if student['truncated']:
            result = None
            stratum = 'student_truncated'
        else:
            result = score(student['response'], student['ground_truth'])
            stratum = ('student_complete_correct' if result['correct'] is True else
                       'student_complete_incorrect' if result['correct'] is False else 'student_complete_grader_unknown')
        scoring_audit.append({'uid': student['uid'], 'stratum': stratum, 'scoring': result})
        if stratum != 'student_complete_grader_unknown':
            buckets[stratum].append((teacher, student, result))
    selected = []
    counts = {}
    for stratum in ('student_complete_correct', 'student_complete_incorrect', 'student_truncated'):
        candidates = sorted(buckets[stratum], key=lambda p: (len(p[1]['response']), p[1]['response_id']))
        if len(candidates) < 16:
            raise ValueError(f'insufficient {stratum}: {len(candidates)}; no silent stratum substitution')
        midpoint = len(candidates) // 2
        selected.extend((stratum, pair) for half in (candidates[:midpoint], candidates[midpoint:]) for pair in rng.sample(half, 8))
        counts[stratum] = len(candidates)
    real = []
    for stratum, (teacher, student, student_score) in selected:
        for source_row in (teacher, student):
            result = student_score if source_row is student and student_score is not None else score(source_row['response'], source_row['ground_truth'])
            real.append({'id': 'real_' + digest({'response_id': source_row['response_id'], 'seed': args.seed})[:20],
                         'kind': 'real', 'category': stratum, 'uid': source_row['uid'],
                         'prompt': source_row['prompt'], 'response': source_row['response'],
                         'ground_truth': source_row['ground_truth'], 'truncated': source_row['truncated'],
                         'source_role': source_row['source_role'], 'source_split': source_row['split'],
                         'source_response_id': source_row['response_id'], 'source_response_hash': source_row['response_hash'],
                         'source_fingerprint': source_row['source_fingerprint'], 'finish_reason': source_row['finish_reason'],
                         'original_response_tokens': source_row['original_response_tokens'],
                         'scoring': result, 'human_review': 'pending'})
    edge = boundaries()
    rng.shuffle(real)
    rng.shuffle(edge)
    # Every block of eight contains six real trajectories and two diagnostics,
    # reducing source imbalance if a wall-clock stop leaves a partial run.
    rows = [row for i in range(16) for row in real[i*6:(i+1)*6] + edge[i*2:(i+1)*2]]
    assert len(rows) == 128 and len({r['id'] for r in rows}) == 128
    assert len({r['uid'] for r in real}) == 48
    (output / 'inputs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    (output / 'selection_scoring_audit.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in scoring_audit))
    # Prepare the blind subset before seeing judge outputs. Natural and
    # deliberately challenging samples are evaluated separately.
    review_rng = random.Random(args.seed + 1)
    review_rows = review_rng.sample(real, 32) + review_rng.sample(edge, 16)
    review_rng.shuffle(review_rows)
    blind = []
    key = {}
    for i, row in enumerate(review_rows):
        rid = f'review_{i+1:03d}'
        key[rid] = row['id']
        blind.append({'review_id': rid, 'problem': row['prompt'], 'visible_response': row['response'],
                      'reference_final_answer': row['ground_truth'], 'response_truncated': row['truncated'],
                      'reviewer': '', 'review_status': 'pending',
                      'verified_progress': {'value': None, 'valid': None, 'reason': ''},
                      'unresolved_error_severity': {'value': None, 'valid': None, 'reason': ''}})
    (output / 'human_review_blind.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in blind))
    (output / 'human_review_key.json').write_text(json.dumps(key, indent=2))
    manifest = {'version': 'rubric.judge.benchmark.v1', 'source': str(source), 'source_sha256': file_hash(source),
                'seed': args.seed, 'real_questions': 48, 'real_responses': 96, 'boundary_responses': 32,
                'source_split': 'train', 'student_strata_pool_counts': counts,
                'selection': 'one seeded-random student per uid; 16 unique questions per student stratum; half lower/upper character lengths',
                'sampling_caveat': 'balanced diagnostic sample, not a population accuracy estimate',
                'source_caveat': 'historical 0.6B Warmup trajectories, not current GAD on-policy samples',
                'real_role_counts': dict(Counter(r['source_role'] for r in real)),
                'real_truncated_counts': dict(Counter(str(r['truncated']) for r in real)),
                'real_correctness_counts': dict(Counter(str(r['scoring']['correct']) for r in real)),
                'blind_review_count': 48, 'blind_selection': '32 real + 16 boundary, seeded before labels; roles and judge outputs hidden',
                'inputs_sha256': file_hash(output / 'inputs.jsonl'), 'scoring_identity': identity(),
                'human_review': 'pending'}
    (output / 'dataset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
