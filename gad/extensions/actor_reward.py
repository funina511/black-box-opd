"""Answer-level reward mixing and diagnostics; no Judge calls or model inference."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import torch

DIMS = ('answer_correctness', 'verified_progress', 'unresolved_error_severity')


def mixed_enabled(options):
    return options.get('reward_source', 'adv_only') == 'adv_plus_rubric' and options.get('actor_reward_coef', 0.) > 0


def components_enabled(options):
    return (mixed_enabled(options) or options.get('actor_reward_shadow', False) or
            options.get('reward_source') == 'adv_plus_rubric_groupnorm')


def groups_for(uids):
    groups = defaultdict(list)
    for i, uid in enumerate(uids):
        groups[uid].append(i)
    return groups


def rubric_scores(parts, options):
    weights = options.get('actor_reward_weights', {})
    return sum(parts[:, i + 1].float() * weights.get(d, 1.) * (-1 if i == 2 else 1)
               for i, d in enumerate(DIMS))


def groupnorm_scores(adv, rubric, uids, weight, floor):
    """Full uid groups only. Zero contribution bypasses the inner normalization.

    Shared by the driver, monitoring and offline selection. The outer GRPO is
    deliberately left to the original trainer (or group_advantages offline).
    """
    adv, rubric = adv.float(), rubric.float()
    if len(adv) != len(uids) or len(rubric) != len(adv):
        raise ValueError('reward rows and uids differ')
    if weight < 0 or floor < 0 or not torch.isfinite(torch.tensor([weight, floor])).all():
        raise ValueError('group weight and rubric std floor must be finite and nonnegative')
    if not torch.isfinite(adv).all() or not torch.isfinite(rubric).all():
        raise ValueError('nonfinite reward components')
    result = adv.clone()
    if weight == 0:
        return result
    for ids in groups_for(uids).values():
        if len(ids) < 2:
            continue
        a, r = adv[ids], rubric[ids]
        std_r = r.std()
        if std_r == 0:
            continue  # Includes floor=0; never evaluate 0/0.
        std_a = a.std()
        za = (a - a.mean()) / (std_a + 1e-6) if std_a != 0 else torch.zeros_like(a)
        zr = (r - r.mean()) / std_r.clamp_min(floor)
        result[ids] = za + weight * zr
    return result


def reward_scores(parts, options, uids):
    """adv, correctness counterfactual, actual method reward, all in FP32."""
    source = options.get('reward_source', 'adv_only')
    adv = parts[:, 0].float()
    if source == 'adv_plus_rubric_groupnorm':
        weight = options.get('actor_reward_group_weight', 0.)
        floor = options.get('actor_reward_rubric_std_floor', 0.)
        wc = options.get('actor_reward_weights', {}).get(DIMS[0], 1.)
        return (adv, groupnorm_scores(adv, parts[:, 1].float() * wc, uids, weight, floor),
                groupnorm_scores(adv, rubric_scores(parts, options), uids, weight, floor))
    if source == 'adv_plus_rubric':
        return rewards(parts, options)
    return adv, adv, adv


def driver_reward(batch, values, trainer, algorithm):
    """Run after all DP values are gathered and before token rewards / GRPO."""
    state = trainer.get('gad_extensions', {})
    if state.get('stage') != 'gad':
        return values
    options = trainer.get('rubric', {})
    if options.get('reward_source') == 'adv_plus_rubric_groupnorm':
        if algorithm.get('use_kl_in_reward', False):
            raise ValueError('groupnorm requires algorithm.use_kl_in_reward=false')
        parts = batch.batch['rubric_reward_components']
        scores = reward_scores(parts, options, batch.non_tensor_batch['uid'])[2]
        mask = batch.batch['response_mask'].bool()
        indices = torch.where(mask, torch.arange(mask.shape[-1], device=mask.device), -1).max(-1).values
        if (indices < 0).any():
            raise ValueError('empty response in group reward')
        return torch.zeros_like(values, dtype=torch.float32).scatter(1, indices[:, None], scores[:, None])
    return values.float() if state.get('actor_reward_fp32', False) else values


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
    groups = groups_for(uids)
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


def monitor_records(actor_batch, critic_batch, fresh, step, scorer_labels=None):
    """Join labels by explicit pair/response IDs, never cache insertion order."""
    four_heads = 'rubric_reward_components' in actor_batch.batch
    components = (actor_batch.batch['rubric_reward_components'].detach().float().cpu() if four_heads else
                  actor_batch.batch['values'].detach().float().cpu().sum(-1)[:, None])
    if len(fresh) != len(components):
        raise ValueError('reward components and fresh Actor batch differ')
    pair_ids = critic_batch.non_tensor_batch.get('pair_id')
    lookup = {str(pid): i for i, pid in enumerate(pair_ids)} if pair_ids is not None else {}
    if pair_ids is not None and len(lookup) != len(critic_batch):
        raise ValueError('critic pair IDs must identify occurrences uniquely')
    targets = critic_batch.batch.get('student_rubric_targets', None)
    masks = critic_batch.batch.get('student_rubric_mask', None)
    if targets is not None:
        targets, masks = targets.cpu(), masks.cpu()
    response_mask = actor_batch.batch['attention_mask'][:, -actor_batch.batch['responses'].shape[-1]:].bool().cpu()
    actual = actor_batch.batch['advantages'].detach().float().cpu()
    rows = []
    for i, pair in enumerate(fresh):
        j = lookup.get(pair['pair_id'])
        y, valid = [0.] * 3, [0.] * 3
        if j is not None and targets is not None:
            if str(critic_batch.non_tensor_batch['student_response_id'][j]) != pair['student']['response_id']:
                raise ValueError('student label identity differs from Actor response')
            y, valid = targets[j].tolist(), masks[j].tolist()
        if scorer_labels is not None:
            label = scorer_labels[i]
            y[0], valid[0] = (float(label['value']), 1.) if label['valid'] else (0., 0.)
        finish = pair['student']['finish_reason']
        rows.append({'step': step, 'uid': str(actor_batch.non_tensor_batch['uid'][i]),
                     'question_id': pair.get('prompt_hash', pair.get('prompt_id', pair['pair_id'])),
                     'pair_id': pair['pair_id'], 'response_id': pair['student']['response_id'],
                     'parts': components[i].tolist() if four_heads else [float(components[i, 0]), None, None, None],
                     'target': y, 'mask': valid,
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


def group_diagnostics(rows, parts, adv, full, options):
    uids = [(r['step'], r['uid']) for r in rows]
    groups = groups_for(uids)
    aa, af = group_advantages(adv, uids), group_advantages(full, uids)
    rubric = rubric_scores(parts, options)
    std_r, std_a, gaps_a, gaps_f, weak_deltas = [], [], [], [], []
    mixed, truncated, correct, zero, passthrough = 0, 0, 0, 0, 0
    below_flips = flips = 0
    floor = options.get('actor_reward_rubric_std_floor', 0.)
    weight = options.get('actor_reward_group_weight', 0.)
    mode = options.get('reward_source', 'adv_only')
    for ids in groups.values():
        a, r = adv[ids], rubric[ids]
        sa = float(a.std()) if len(ids) > 1 else 0.
        sr = float(r.std()) if len(ids) > 1 else 0.
        std_a.append(sa); std_r.append(sr)
        known = [rows[i]['target'][0] for i in ids if rows[i]['mask'][0]]
        mixed += 0 in known and 1 in known
        truncated += all(rows[i]['category'] == 'T' for i in ids)
        correct += len(known) == len(ids) and all(y == 1 for y in known)
        zero += not bool((af[ids] != 0).any())
        passthrough += mode == 'adv_only' or (mode == 'adv_plus_rubric_groupnorm' and (weight == 0 or sr == 0))
        cs, ws = [i for i in ids if rows[i]['category'] == 'C'], [i for i in ids if rows[i]['category'] == 'W']
        if cs and ws:
            gaps_a.append(float(aa[cs].mean() - aa[ws].mean()))
            gaps_f.append(float(af[cs].mean() - af[ws].mean()))
        if sa < 1e-5:
            weak_deltas.extend((af[ids] - aa[ids]).abs().tolist())
        for i, j in combinations(ids, 2):
            if float(adv[i] - adv[j]) * float(full[i] - full[j]) < 0:
                flips += 1
                below_flips += sr < floor
    stats = {'groups/mixed_correctness_count': mixed, 'groups/all_truncated_count': truncated,
             'groups/all_correct_count': correct, 'groups/zero_advantage_count': zero,
             'groups/passthrough_count': passthrough, 'groups/rubric_zero_std_count': sum(s == 0 for s in std_r),
             'groups/adv_weak_count': sum(s < 1e-5 for s in std_a),
             'groups/adv_weak_advantage_change_defined': int(bool(weak_deltas)),
             'groups/cw_gap_defined': int(bool(gaps_a)), 'groups/cw_gap_count': len(gaps_a),
             'groups/below_floor_strict_flips': below_flips, 'groups/below_floor_flip_share_defined': int(flips > 0)}
    for name in ('mixed_correctness', 'all_truncated', 'all_correct', 'zero_advantage'):
        stats['groups/' + name + '_rate'] = stats['groups/' + name + '_count'] / len(groups)
    if weak_deltas:
        stats['groups/adv_weak_advantage_change_mean'] = sum(weak_deltas) / len(weak_deltas)
    if gaps_a:
        stats['groups/cw_gap_adv'] = sum(gaps_a) / len(gaps_a)
        stats['groups/cw_gap_final'] = sum(gaps_f) / len(gaps_f)
        stats['groups/cw_gap_increase'] = stats['groups/cw_gap_final'] - stats['groups/cw_gap_adv']
    if flips:
        stats['groups/below_floor_flip_share'] = below_flips / flips
    for q in (0, .25, .5, .75, 1.):
        stats['groups/rubric_std_q' + str(int(q * 100))] = float(torch.tensor(std_r).quantile(q))
    for cat in ('C', 'W', 'T'):
        ids = [i for i, row in enumerate(rows) if row['category'] == cat]
        stats['category/' + cat + '/count'] = len(ids)
        if ids:
            stats['category/' + cat + '/advantage_adv_mean'] = float(aa[ids].mean())
            stats['category/' + cat + '/advantage_final_mean'] = float(af[ids].mean())
    return stats


def constraint_statistics(rows, options):
    """Predeclared n=4 selection constraints. Undefined ratios stay JSON null."""
    if not rows:
        return {'samples': 0, 'constraints': {str(i): False for i in range(1, 6)}, 'eligible': False,
                'corrected': 0, 'damaged': 0, 'net_corrections': 0, 'cw_gap_increase': None}
    parts = torch.tensor([r['parts'] for r in rows], dtype=torch.float32)
    uids = [(r['step'], r['uid']) for r in rows]
    adv, _, full = reward_scores(parts, options, uids)
    stats = group_diagnostics(rows, parts, adv, full, options)
    corrected = damaged = cw_net = changed = unequal = tw_t = tw_w = 0
    cw_flips = ct_flips = cw_total = ct_total = 0
    for ids in groups_for(uids).values():
        for i, j in combinations(ids, 2):
            a, b = float(adv[i] - adv[j]), float(full[i] - full[j])
            cats = {rows[i]['category'], rows[j]['category']}
            flip = a * b < 0
            unequal += a != 0
            changed += a != 0 and (a * b < 0 or b == 0)
            cw_total += cats == {'C', 'W'}
            ct_total += cats == {'C', 'T'}
            cw_flips += cats == {'C', 'W'} and flip
            ct_flips += cats == {'C', 'T'} and flip
            if cats == {'T', 'W'}:
                direction = 1 if rows[i]['category'] == 'T' else -1
                tw_t += a * direction < 0 and b * direction > 0
                tw_w += a * direction > 0 and b * direction < 0
            if (rows[i]['mask'][0] and rows[j]['mask'][0] and
                    {rows[i]['target'][0], rows[j]['target'][0]} == {0, 1}):
                direction = 1 if rows[i]['target'][0] == 1 else -1
                c, d = a * direction < 0 and b * direction > 0, a * direction > 0 and b * direction < 0
                corrected += c; damaged += d
                if cats == {'C', 'W'}:
                    cw_net += c - d
    precision = corrected / (corrected + damaged) if corrected + damaged else None
    tw_share = tw_t / (tw_t + tw_w) if tw_t + tw_w else None
    change_rate = changed / unequal if unequal else None
    delta = stats.get('groups/cw_gap_increase')
    constraints = {'1': precision is None or precision >= .8, '2': cw_net >= 0,
                   '3': tw_share is None or tw_share <= .5,
                   '4': change_rate is None or change_rate <= .25,
                   '5': corrected - damaged >= 0 and delta is not None and delta > 0}
    return {'samples': len(rows), 'groups': len(groups_for(uids)), 'corrected': corrected, 'damaged': damaged,
            'net_corrections': corrected - damaged, 'correction_precision': precision, 'cw_net_corrections': cw_net,
            'tw_toward_T': tw_t, 'tw_toward_W': tw_w, 'tw_toward_T_share': tw_share,
            'unequal_adv_pairs': unequal, 'order_changes': changed, 'order_change_rate': change_rate,
            'cw_strict_flip_rate': cw_flips / cw_total if cw_total else None,
            'ct_strict_flip_rate': ct_flips / ct_total if ct_total else None,
            'cw_gap_increase': delta, 'mainly_spacing': corrected == damaged and delta is not None and delta > 0,
            'adv_weak_groups': stats['groups/adv_weak_count'],
            'adv_weak_advantage_change_mean': stats.get('groups/adv_weak_advantage_change_mean'),
            'constraints': constraints, 'eligible': all(constraints.values()), 'diagnostics': stats}


@lru_cache(maxsize=4)
def monitor_scorer(entrypoint):
    # Ray's isolated backend need not expose the LMOps root on sys.path.
    # Use the configured file, as OnlineRubric does, and verify its fingerprint.
    if entrypoint:
        import importlib.util
        spec = importlib.util.spec_from_file_location('gad_student_monitor_scorer', entrypoint)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from scoring import math
    return math


def student_scorer_labels(fresh, config):
    """Read-only baseline monitoring, using the rubric's exact training mapping."""
    scorer = monitor_scorer(config.get('entrypoint'))
    labels = []
    for pair in fresh:
        result = scorer.score(pair['student']['response'], pair['ground_truth'], config)
        if result['scorer_fingerprint'] != config['fingerprint']:
            raise ValueError('student monitor scorer fingerprint differs')
        labels.append(scorer.correctness_label(result, pair['prompt_truncated'] or not pair['prompt'].strip()))
    return labels


def summarize(rows, options, prefix='actor_reward/'):
    if not rows:
        return {}
    four_heads = all(all(x is not None for x in r['parts']) for r in rows)
    parts = torch.tensor([[0. if x is None else x for x in r['parts']] for r in rows], dtype=torch.float32)
    target = torch.tensor([r['target'] for r in rows], dtype=torch.float32)
    mask = torch.tensor([r['mask'] for r in rows]).bool()
    uids = [(r['step'], r['uid']) for r in rows]
    adv, correct, full = reward_scores(parts, options, uids)
    stats = {'samples': len(rows), 'steps': len({r['step'] for r in rows})}
    weights = options.get('actor_reward_weights', {})
    rubric = rubric_scores(parts, options)
    contribution = full - adv
    if options.get('reward_source') == 'adv_plus_rubric_groupnorm':
        contribution = torch.zeros_like(adv)
        normalized_adv = group_advantages(adv, uids)
        for ids in groups_for(uids).values():
            if options.get('actor_reward_group_weight', 0.) > 0 and len(ids) > 1 and rubric[ids].std() != 0:
                contribution[ids] = full[ids] - normalized_adv[ids]
    distributions = [('adv', adv), ('total', full)]
    if four_heads:
        distributions += [('c', parts[:, 1]), ('p', parts[:, 2]), ('e', parts[:, 3]),
                          ('rubric', rubric), ('contribution', contribution)]
    for name, values in distributions:
        stats[name + '_mean'], stats[name + '_std'] = float(values.mean()), float(values.std(unbiased=False))
    stats['truncated_count'] = sum(r['finish_reason'] == 'length' for r in rows)
    stats['stopped_count'] = sum(r['finish_reason'] == 'stop' for r in rows)
    stats['abnormal_finish_count'] = len(rows) - stats['truncated_count'] - stats['stopped_count']
    stats['truncated_rate'] = stats['truncated_count'] / len(rows)
    stats['tokens_mean'] = sum(r['tokens'] for r in rows) / len(rows)
    stats['correctness_unknown'] = int((~mask[:, 0]).sum())
    stats['correctness_known'] = int(mask[:, 0].sum())
    stats['student_correct_count'] = int((target[:, 0].bool() & mask[:, 0]).sum())
    if stats['correctness_known']:
        stats['student_correct_rate_known'] = stats['student_correct_count'] / stats['correctness_known']
    for name, index in [('all', None), ('truncated', True), ('nontruncated', False)]:
        selected = mask[:, 0].clone()
        if index is not None:
            selected &= torch.tensor([(r['finish_reason'] == 'length') == index for r in rows])
        stem = 'calibration/c/' + name + '/'
        stats[stem + 'count'] = int(selected.sum())
        if selected.any() and four_heads:
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
        if selected.any() and four_heads:
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
    stats.update(group_diagnostics(rows, parts, adv, full, options))
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
