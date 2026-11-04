"""Answer-level reward mixing and diagnostics; no Judge calls or model inference."""
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import torch

DIMS = ('answer_correctness', 'verified_progress', 'unresolved_error_severity')


def mixed_enabled(options):
    return options.get('reward_source', 'adv_only') == 'adv_plus_rubric' and options.get('actor_reward_coef', 0.) > 0


def combine_last(last, options):
    """FP32 adv/c/p/e from the *same*, pre-update four-head forward."""
    if last.shape[-1] != 4:
        raise ValueError('mixed reward requires four head logits')
    parts = torch.cat((last[:, :1].float(), last[:, 1:].float().sigmoid()), dim=-1)
    return rewards(parts, options)[2], parts


def rewards(parts, options):
    parts = parts.float()
    beta = options.get('actor_reward_coef', 0.)
    weights = options.get('actor_reward_weights', {})
    c, p, e = (parts[:, i + 1] * weights.get(d, 1.) for i, d in enumerate(DIMS))
    adv = parts[:, 0]
    return adv, adv + beta * c, adv + beta * (c + p - e)


def group_advantages(scores, uids):
    """Mirror the pinned GRPO (unbiased std, epsilon=1e-6), including ties."""
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    result = torch.zeros_like(scores)
    for ids in groups.values():
        values = scores[ids]
        if len(ids) == 1:
            mean, std = torch.zeros_like(values[0]), torch.ones_like(values[0])
        else:
            mean, std = values.mean(), values.std()
        result[ids] = (values - mean) / (std + 1e-6)
    return result


def category(target, mask, finish):
    if finish == 'length':
        return 'T'
    if finish != 'stop' or not mask:
        return 'U'
    return 'C' if target == 1 else 'W' if target == 0 else 'U'


def monitor_records(actor_batch, critic_batch, fresh, step):
    """Join labels by explicit pair/response IDs, never cache insertion order."""
    components = actor_batch.batch['rubric_reward_components'].detach().float().cpu()
    if len(fresh) != len(components):
        raise ValueError('reward components and fresh Actor batch differ')
    lookup = {str(pid): i for i, pid in enumerate(critic_batch.non_tensor_batch['pair_id'])}
    if len(lookup) != len(critic_batch):
        raise ValueError('critic pair IDs must identify occurrences uniquely')
    targets = critic_batch.batch['student_rubric_targets'].cpu()
    masks = critic_batch.batch['student_rubric_mask'].cpu()
    response_mask = actor_batch.batch['attention_mask'][:, -actor_batch.batch['responses'].shape[-1]:].bool().cpu()
    actual = actor_batch.batch['advantages'].detach().float().cpu()
    rows = []
    for i, pair in enumerate(fresh):
        j = lookup.get(pair['pair_id'])
        y, valid = [0.] * 3, [0.] * 3
        if j is not None:
            if str(critic_batch.non_tensor_batch['student_response_id'][j]) != pair['student']['response_id']:
                raise ValueError('student label identity differs from Actor response')
            y, valid = targets[j].tolist(), masks[j].tolist()
        finish = pair['student']['finish_reason']
        rows.append({'step': step, 'uid': str(actor_batch.non_tensor_batch['uid'][i]),
                     'pair_id': pair['pair_id'], 'response_id': pair['student']['response_id'],
                     'parts': components[i].tolist(), 'target': y, 'mask': valid,
                     'finish_reason': finish, 'tokens': int(response_mask[i].sum()),
                     'category': category(y[0], valid[0], finish),
                     'grpo_advantage': float(actual[i][response_mask[i]][0])})
    return rows


def auc(pred, target):
    positive = pred[target == 1]
    negative = pred[target == 0]
    if not len(positive) or not len(negative):
        return None
    # Bounded windows (10 steps) keep this small; ties count 0.5.
    return float(((positive[:, None] > negative).float() + .5 * (positive[:, None] == negative)).mean())


def summarize(rows, options, prefix='actor_reward/'):
    if not rows:
        return {}
    parts = torch.tensor([r['parts'] for r in rows], dtype=torch.float32)
    target = torch.tensor([r['target'] for r in rows], dtype=torch.float32)
    mask = torch.tensor([r['mask'] for r in rows]).bool()
    uids = [(r['step'], r['uid']) for r in rows]
    adv, correct, full = rewards(parts, options)
    stats = {'samples': len(rows), 'steps': len({r['step'] for r in rows})}
    weights = options.get('actor_reward_weights', {})
    rubric = sum(parts[:, i + 1] * weights.get(d, 1.) * (-1 if i == 2 else 1) for i, d in enumerate(DIMS))
    for name, values in [('adv', adv), ('c', parts[:, 1]), ('p', parts[:, 2]), ('e', parts[:, 3]),
                         ('rubric', rubric), ('contribution', full - adv), ('total', full)]:
        stats[name + '_mean'], stats[name + '_std'] = float(values.mean()), float(values.std(unbiased=False))
    stats['truncated_count'] = sum(r['finish_reason'] == 'length' for r in rows)
    stats['stopped_count'] = sum(r['finish_reason'] == 'stop' for r in rows)
    stats['abnormal_finish_count'] = len(rows) - stats['truncated_count'] - stats['stopped_count']
    stats['truncated_rate'] = stats['truncated_count'] / len(rows)
    stats['tokens_mean'] = sum(r['tokens'] for r in rows) / len(rows)
    stats['correctness_unknown'] = int((~mask[:, 0]).sum())
    for name, index in [('all', None), ('truncated', True), ('nontruncated', False)]:
        selected = mask[:, 0].clone()
        if index is not None:
            selected &= torch.tensor([(r['finish_reason'] == 'length') == index for r in rows])
        stem = 'calibration/c/' + name + '/'
        stats[stem + 'count'] = int(selected.sum())
        if selected.any():
            pred, y = parts[selected, 1], target[selected, 0]
            stats.update({stem + 'prediction_mean': float(pred.mean()), stem + 'target_mean': float(y.mean()),
                          stem + 'bias': float((pred - y).mean()), stem + 'mse': float((pred - y).square().mean())})
            value = auc(pred, y)
            stats[stem + 'auc_defined'] = int(value is not None)
            if value is not None:
                stats[stem + 'auc'] = value
    for i, name in [(1, 'p'), (2, 'e')]:
        selected = mask[:, i]
        stem = 'calibration/' + name + '/'
        stats[stem + 'count'] = int(selected.sum())
        if selected.any():
            delta = parts[selected, i + 1] - target[selected, i]
            stats.update({stem + 'prediction_mean': float(parts[selected, i + 1].mean()),
                          stem + 'target_mean': float(target[selected, i].mean()),
                          stem + 'bias': float(delta.mean()), stem + 'mae': float(delta.abs().mean())})
    group = defaultdict(list)
    for i, uid in enumerate(uids):
        group[uid].append(i)
    triples = [(i, j) for ids in group.values() for i, j in combinations(ids, 2)]
    advantages = [group_advantages(x, uids) for x in (adv, correct, full)]
    actual = torch.tensor([r.get('grpo_advantage', float(advantages[2][i])) for i, r in enumerate(rows)])
    stats['advantages_actual_max_error'] = float((actual - advantages[2]).abs().max())
    stats['advantages_change_mean'] = float((advantages[2] - advantages[0]).abs().mean())
    stats['groups'] = len(group)
    stats['nonzero_advantage_groups'] = sum(bool((advantages[2][ids] != 0).any()) for ids in group.values())
    stats['nonzero_advantage_group_rate'] = stats['nonzero_advantage_groups'] / len(group)
    tw = [(i, j) if rows[i]['category'] == 'T' else (j, i) for i, j in triples
          if {rows[i]['category'], rows[j]['category']} == {'T', 'W'}]
    stats['pairs/gaps/TW/count'] = len(tw)
    for name, values in [('adv', adv), ('c', parts[:, 1]), ('p', parts[:, 2]), ('e', parts[:, 3]),
                         ('rubric', rubric), ('total', full)]:
        if tw:
            delta = torch.stack([values[t] - values[w] for t, w in tw])
            stats['pairs/gaps/TW/T_minus_W_' + name + '_mean'] = float(delta.mean())
            stats['pairs/gaps/TW/T_minus_W_' + name + '_std'] = float(delta.std(unbiased=False))
    for a, b, name in [(adv, correct, 'adv_to_c'), (correct, full, 'c_to_full'), (adv, full, 'adv_to_full')]:
        buckets = defaultdict(list)
        for i, j in triples:
            buckets['all'].append((i, j))
            buckets[''.join(sorted((rows[i]['category'], rows[j]['category'])))].append((i, j))
        for bucket, pairs in buckets.items():
            stem = 'pairs/' + name + '/' + bucket + '/'
            counts = defaultdict(int)
            for i, j in pairs:
                da, db = float(a[i] - a[j]), float(b[i] - b[j])
                counts['strict_flips'] += da * db < 0
                counts['before_ties'] += da == 0
                counts['after_ties'] += db == 0
                counts['ties_resolved'] += da == 0 and db != 0
                counts['ties_created'] += da != 0 and db == 0
                counts['near_epsilon_before'] += 0 < abs(da) <= 1e-6
                counts['near_epsilon_after'] += 0 < abs(db) <= 1e-6
                if bucket == 'TW':
                    t, w = (i, j) if rows[i]['category'] == 'T' else (j, i)
                    dt_a, dt_b = float(a[t] - a[w]), float(b[t] - b[w])
                    counts['toward_T'] += dt_a < 0 and dt_b > 0
                    counts['toward_W'] += dt_a > 0 and dt_b < 0
                    counts['prefer_T_before'] += dt_a > 0
                    counts['prefer_T_after'] += dt_b > 0
                if mask[i, 0] and mask[j, 0] and {float(target[i, 0]), float(target[j, 0])} == {0., 1.}:
                    good, bad = (i, j) if target[i, 0] == 1 else (j, i)
                    x, y = float(a[good] - a[bad]), float(b[good] - b[bad])
                    counts['right_wrong_pairs'] += 1
                    counts['right_before'] += x > 0
                    counts['right_after'] += y > 0
                    counts['corrected'] += x < 0 and y > 0
                    counts['damaged'] += x > 0 and y < 0
            stats[stem + 'count'] = len(pairs)
            for key, value in counts.items():
                stats[stem + key] = value
            stats[stem + 'strict_flip_rate'] = counts['strict_flips'] / len(pairs)
            stats[stem + 'before_tie_rate'] = counts['before_ties'] / len(pairs)
            stats[stem + 'after_tie_rate'] = counts['after_ties'] / len(pairs)
    return {prefix + key: value for key, value in stats.items()}


def load_window(directory, committed_step):
    """Derive the window from existing committed logs; no new checkpoint state."""
    path = Path(directory) / 'extension_metrics.jsonl'
    by_step = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # Incomplete last line after a killed append.
            if row.get('step', committed_step + 1) <= committed_step and row.get('actor_reward_monitor'):
                by_step[row['step']] = row['actor_reward_monitor']
    return [by_step[s] for s in sorted(by_step)[-9:]]
