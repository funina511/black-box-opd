#!/usr/bin/env python3
"""Predeclared, CPU-only selection on the completed naive shadow log.

Produces a recommendation and all candidates; never edits a launch config.
"""
import argparse
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


def shadow_group_size(rows, expected=None):
    groups = groups_for([(r['step'], r['uid']) for r in rows])
    if not groups:
        raise ValueError('no shadow groups')
    group_size = len(next(iter(groups.values()))) if expected is None else expected
    if group_size not in (4, 8):
        raise ValueError(f'expected shadow groups of 4 or 8 answers, got {group_size}')
    for ids in groups.values():
        if len(ids) != group_size or len({rows[i]['question_id'] for i in ids}) != 1:
            raise ValueError(f'each complete uid group must contain {group_size} answers to one question')
    return group_size


def read_shadow(path, *, expected_groups_per_step=None, group_size=None):
    by_step = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            rows = record.get('actor_reward_monitor')
            if rows:
                if any(r['step'] != record['step'] for r in rows):
                    raise ValueError('shadow row step differs from its recorded step')
                by_step[record['step']] = rows
    rows = [r for s in sorted(by_step) for r in by_step[s]]
    if not rows or any(not r.get('question_id') or len(r['parts']) != 4 or
                       any(x is None for x in r['parts']) for r in rows):
        raise ValueError('need complete four-head shadow rows with stable question_id')
    group_size = shadow_group_size(rows, group_size)
    if expected_groups_per_step is not None:
        for step, step_rows in by_step.items():
            if len(step_rows) != group_size * expected_groups_per_step:
                raise ValueError(f'step {step}: expected {expected_groups_per_step} groups of {group_size}, got {len(step_rows)} rows')
    return rows


def std_distribution(rows, weights):
    parts = torch.tensor([r['parts'] for r in rows], dtype=torch.float32)
    scores = rubric_scores(parts, {'actor_reward_weights': weights})
    groups = groups_for([(r['step'], r['uid']) for r in rows])
    return torch.tensor([float(scores[ids].std()) for ids in groups.values()])


def selection_config(run):
    """Read the frozen snapshot beside the log, including a relocated historical run."""
    from config import load_config, run_config_path
    pointer = run / 'current_resolved_config.txt'
    snapshot = None
    if not (run / 'train.yaml').is_file() and pointer.is_file():
        # Old runners saved absolute pointers into their own reports/resolved_configs.
        target = Path(pointer.read_text().strip())
        if target.parent.name == 'resolved_configs':
            snapshot = run / 'reports' / 'resolved_configs' / target.name
    if snapshot is None or not snapshot.is_file():
        snapshot = run_config_path(run)
    cfg = load_config(snapshot)
    recorded_run = Path(cfg.require('run.output_root')) / cfg.require('run.run_id')
    if recorded_run.resolve() != run.resolve() and not snapshot.resolve().is_relative_to(run.resolve()):
        raise ValueError(f'copy the frozen config snapshot into the archived run {run}')
    if Path(cfg.require('run.run_id')).name != run.name:
        raise ValueError('run directory name differs from its frozen run ID')
    return cfg.for_stage('gad')


def formal_step_range(cfg, run):
    """Use the frozen GAD plan and the same drop-last epoch calculation as training."""
    from pipeline import completion_marker, stage_step_budget, steps_per_epoch
    if cfg.get('run.engineering_smoke') or cfg.get('run.max_stage_steps') is not None:
        raise ValueError('offline formal selection does not accept a bounded training run')

    plan, verified = {}, {}
    marker = completion_marker(run, 'gad')
    stage_file = run / 'stage_gad.json'
    if marker.is_file():
        plan = json.loads(marker.read_text())
        verified = plan.get('verified', {})
        if not plan.get('training_completed') or plan.get('returncode') != 0:
            raise ValueError('GAD completion record does not confirm successful training')
    elif stage_file.is_file():
        stage = json.loads(stage_file.read_text())
        if stage['returncode'] != 0:
            raise ValueError('GAD stage did not finish successfully')
        plan = stage['plan']
        verified = stage.get('verification', {})
    if verified.get('training_status') == 'failed':
        raise ValueError('GAD completion record reports a training failure')

    train_file = Path(cfg.require('data.train_file'))
    recorded_steps = plan.get('steps_per_epoch')
    if train_file.is_file():
        per_epoch = steps_per_epoch(cfg)
        if recorded_steps is not None and per_epoch != recorded_steps:
            raise ValueError('GAD steps_per_epoch differs from the frozen data and batch size')
    elif type(recorded_steps) is int and recorded_steps > 0:
        per_epoch = recorded_steps
    else:
        raise ValueError('need the frozen training parquet or steps_per_epoch from the completed GAD stage')

    handoff = cfg.get('budget.init_step') or cfg.get('budget.handoff_step') or stage_step_budget(cfg, 'warmup', per_epoch)
    end = handoff + stage_step_budget(cfg, 'gad', per_epoch, handoff_step=handoff)
    if handoff != per_epoch or end not in (2 * per_epoch, 3 * per_epoch):
        raise ValueError(f'expected one Warmup epoch and one or two GAD epochs of {per_epoch} steps; '
                         f'got Warmup {handoff}, GAD target {end}')
    if plan and ((plan['start_step'], plan['total_steps']) != (handoff, end) or
                 verified.get('last_step', end) != end):
        raise ValueError('completed GAD stage differs from the frozen formal budget')
    return range(handoff + 1, end + 1)


def late_window_start(final_step, steps_per_epoch):
    # Preserve the same fraction of the last epoch as the original 450..542 window.
    late_steps = (93 * steps_per_epoch + 270) // 271
    return final_step - late_steps + 1


def analyze(rows, *, group_size=None, final_step=542, steps_per_epoch=271):
    late_start = late_window_start(final_step, steps_per_epoch)
    late_key = f'late_{late_start}_{final_step}'
    group_size = shadow_group_size(rows, group_size)
    splits = {'selection': [r for r in rows if question_half(r['question_id']) == 0],
              'review': [r for r in rows if question_half(r['question_id']) == 1],
              late_key: [r for r in rows if late_start <= r['step'] <= final_step]}
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
    return {'protocol': f'gad_n{group_size}_selection_v1', 'split': 'sha256(question_id) integer modulo 2: 0=selection, 1=review',
            'counts': {name: {'rows': len(data), 'questions': len({r['question_id'] for r in data}),
                             'steps': sorted({r['step'] for r in data})} for name, data in splits.items()},
            'floors_frozen_before_weight_results': floors, 'candidates': candidates, 'sensitivity_only': sensitivity,
            'recommendation': ({'components': best['components'], 'weight': best['weight'], 'tau': best['tau'],
                                'options': best['options'], 'selection': best['selection'],
                                'review_pass': best['review']['eligible'],
                                'late_pass': best[late_key]['eligible']} if best else None),
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
    cfg = selection_config(run)
    step_range = formal_step_range(cfg, run)
    final_step = step_range[-1]
    per_epoch = step_range.start - 1  # one full Warmup epoch precedes GAD
    late_start = late_window_start(final_step, per_epoch)
    late_key = f'late_{late_start}_{final_step}'
    checkpoint = run / 'gad' / f'global_step_{final_step}'
    if checkpoint_problems(checkpoint, required=True):
        raise ValueError(f'offline formal selection requires a complete naive checkpoint at step {final_step}')
    group_size = cfg.require('rollout.n')
    if not (cfg.get('rubric.enabled') and cfg.get('rubric.actor_reward_shadow') and
            cfg.get('rubric.reward_source') == 'adv_only' and cfg.get('actor_reward.fp32') and
            group_size in (4, 8)):
        raise ValueError('selection input must be an FP32 naive n=4 or n=8 shadow experiment')
    rows = read_shadow(metrics, expected_groups_per_step=cfg.require('data.train_batch_size'),
                       group_size=group_size)
    if sorted({r['step'] for r in rows}) != list(step_range):
        raise ValueError(f'need all {len(step_range)} committed naive GAD shadow steps, {step_range[0]}..{final_step}')
    result = analyze(rows, group_size=group_size, final_step=final_step, steps_per_epoch=per_epoch)
    result['source'] = {'run': str(run), 'metrics': str(metrics), 'sha256': file_hash(metrics)}
    default_output = ROOT / 'reports/gad_n4_20261105/selection' if group_size == 4 else run / 'reports/actor_reward_selection'
    output = Path(args.output) if args.output else default_output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = [f'# n={group_size} Actor-rubric 离线选择', '', '状态：' + result['status'], '',
             f'选择集用于选参。复核集和第 {late_start}–{final_step} 步的结果见 selection.json。', '',
             '|分量|w|τ|选择集净修正|C−W 优势差增加|满足约束|复核通过|后期通过|', '|---|---|---|---|---|---|---|---|']
    def show(x):
        return 'N/A' if x is None else f'{x:.6g}' if isinstance(x, float) else str(x)
    for r in result['candidates']:
        s = r['selection']
        lines.append('|'+ '|'.join(map(show, [r['components'], r['weight'], r['tau'], s['net_corrections'],
                     s['cw_gap_increase'], s['eligible'], r['review']['eligible'], r[late_key]['eligible']])) + '|')
    lines += ['', '建议：' + json.dumps(result['recommendation'], ensure_ascii=False), '',
              '需用户确认后再将分量、w、τ 写入第三组配置并将 selection_confirmed 设为 true。']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'output': str(output), 'status': result['status'], 'recommendation': result['recommendation']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
