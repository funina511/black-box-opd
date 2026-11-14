#!/usr/bin/env python3
"""Predeclared, CPU-only n=4 selection on the completed naive shadow log.

Produces a recommendation and all candidates; never edits a launch config.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools')]
from gad.extensions.actor_reward import (DIMS, groups_for, rubric_scores,
                                         constraint_statistics)
from rubric.core.contracts import file_hash


def question_half(question_id):
    return int(hashlib.sha256(str(question_id).encode()).hexdigest(), 16) % 2


def read_shadow(path):
    by_step = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        rows = record.get('actor_reward_monitor')
        if rows:
            by_step[record['step']] = rows
    rows = [r for s in sorted(by_step) for r in by_step[s]]
    if not rows or any(not r.get('question_id') or len(r['parts']) != 4 or
                       any(x is None for x in r['parts']) for r in rows):
        raise ValueError('need complete four-head shadow rows with stable question_id')
    for ids in groups_for([(r['step'], r['uid']) for r in rows]).values():
        if len(ids) != 4 or len({rows[i]['question_id'] for i in ids}) != 1:
            raise ValueError('each complete uid group must contain four answers to one question')
    return rows


def std_distribution(rows, weights):
    parts = torch.tensor([r['parts'] for r in rows], dtype=torch.float32)
    scores = rubric_scores(parts, {'actor_reward_weights': weights})
    groups = groups_for([(r['step'], r['uid']) for r in rows])
    return torch.tensor([float(scores[ids].std()) for ids in groups.values()])


def analyze(rows):
    splits = {'selection': [r for r in rows if question_half(r['question_id']) == 0],
              'review': [r for r in rows if question_half(r['question_id']) == 1],
              'late_450_542': [r for r in rows if 450 <= r['step'] <= 542]}
    if not splits['selection'] or not splits['review']:
        raise ValueError('both question partitions must have rows')
    definitions = {'full': dict(zip(DIMS, (1., 1., 1.))),
                   'c_only': dict(zip(DIMS, (1., 0., 0.)))}
    # Freeze both component-specific floors before computing ANY weight result.
    floors = {}
    for name, weights in definitions.items():
        stds = std_distribution(splits['selection'], weights)
        floors[name] = {'q25': float(stds.quantile(.25)), 'median': float(stds.quantile(.5)),
                        'zero': 0., 'groups': len(stds)}
    candidates, sensitivity = [], []
    for name, weights in definitions.items():
        for weight in (.25, .5, 1., 2.):
            for floor_name in ('q25', 'zero', 'median'):
                options = {'reward_source': 'adv_plus_rubric_groupnorm', 'actor_reward_weights': weights,
                           'actor_reward_group_weight': weight,
                           'actor_reward_rubric_std_floor': floors[name][floor_name]}
                result = {'components': name, 'weight': weight, 'tau_rule': floor_name,
                          'tau': options['actor_reward_rubric_std_floor'], 'options': options,
                          **{split: constraint_statistics(data, options) for split, data in splits.items()}}
                (candidates if floor_name == 'q25' else sensitivity).append(result)
    eligible = [r for r in candidates if r['selection']['eligible']]
    eligible.sort(key=lambda r: (-r['selection']['net_corrections'],
                                 -r['selection']['cw_gap_increase'], r['weight']))
    best = eligible[0] if eligible else None
    return {'protocol': 'gad_n4_selection_v1', 'split': 'sha256(question_id) integer modulo 2: 0=selection, 1=review',
            'counts': {name: {'rows': len(data), 'questions': len({r['question_id'] for r in data}),
                             'steps': sorted({r['step'] for r in data})} for name, data in splits.items()},
            'floors_frozen_before_weight_results': floors, 'candidates': candidates, 'sensitivity_only': sensitivity,
            'recommendation': ({'components': best['components'], 'weight': best['weight'], 'tau': best['tau'],
                                'options': best['options'], 'selection': best['selection'],
                                'review_pass': best['review']['eligible'],
                                'late_pass': best['late_450_542']['eligible']} if best else None),
            'status': 'await_user_confirmation' if best else 'no_eligible_candidate_user_decision_required',
            'tie_policy': 'strict corrected/damaged; intervention bound includes created ties, excludes initially tied pairs; tie resolutions reported by online diagnostics',
            'limitations': 'Counterfactual on naive Critic trajectories; later Actor-rubric Critic predictions may drift. No config is modified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    metrics = run / 'gad/extension_metrics.jsonl'
    # A bounded smoke or incomplete run must never select formal parameters.
    from gad.extensions.checkpoint_state import checkpoint_problems
    checkpoint = run / 'gad/global_step_542'
    if checkpoint_problems(checkpoint, required=True):
        raise ValueError('offline formal selection requires a complete naive checkpoint at step542')
    rows = read_shadow(metrics)
    if sorted({r['step'] for r in rows}) != list(range(272, 543)):
        raise ValueError('need all 271 committed naive GAD shadow steps, 272..542')
    from config import load_run_config
    cfg = load_run_config(run)
    if not (cfg.get('rubric.enabled') and cfg.get('rubric.actor_reward_shadow') and
            cfg.get('rubric.reward_source') == 'adv_only' and cfg.get('actor_reward.fp32') and
            cfg.get('rollout.n') == 4):
        raise ValueError('selection input must be the FP32 naive n=4 shadow experiment')
    result = analyze(rows)
    result['source'] = {'run': str(run), 'metrics': str(metrics), 'sha256': file_hash(metrics)}
    output = Path(args.output) if args.output else ROOT / 'reports/gad_n4_20261105/selection'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# n=4 Actor-rubric 离线选择', '', '状态：' + result['status'], '',
             '只在选择集选参；复核集和 step450–542 单独报告。完整结果见 selection.json。', '',
             '|分量|w|τ|选择集净修正|C−W 优势差增加|满足约束|复核通过|后期通过|', '|---|---|---|---|---|---|---|---|']
    def show(x):
        return 'N/A' if x is None else f'{x:.6g}' if isinstance(x, float) else str(x)
    for r in result['candidates']:
        s = r['selection']
        lines.append('|'+ '|'.join(map(show, [r['components'], r['weight'], r['tau'], s['net_corrections'],
                     s['cw_gap_increase'], s['eligible'], r['review']['eligible'], r['late_450_542']['eligible']])) + '|')
    lines += ['', '建议：' + json.dumps(result['recommendation'], ensure_ascii=False), '',
              '需用户确认后再将分量、w、τ 写入第三组配置并将 selection_confirmed 设为 true。']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'output': str(output), 'status': result['status'], 'recommendation': result['recommendation']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
