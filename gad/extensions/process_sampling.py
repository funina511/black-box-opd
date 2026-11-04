"""Select visible supervision identities once per global optimizer minibatch.

Selection affects occurrence masks, never cached mathematical annotations.
The row groups mirror contiguous DP dispatch in auxiliary_weights().
"""
from .common import digest
from .critic_batch import supervision_identity


def logical_minibatches(size, global_mini, dp_size):
    if size % dp_size or global_mini % dp_size or size % global_mini:
        raise ValueError('extension batches must divide DP and logical minibatch sizes')
    per_rank, local = size // dp_size, global_mini // dp_size
    return [[rank * per_rank + i for rank in range(dp_size)
             for i in range(offset, offset + local)]
            for offset in range(0, per_rank, local)]


def select_process(pairs, global_mini, dp_size, options, step, stage):
    config = options.get('process_sampling', {})
    budget = config.get('max_unique_per_global_minibatch')
    enabled = options.get('mode') == 'joint' and budget is not None
    if enabled and (type(budget) is not int or budget < 2):
        raise ValueError('process sampling needs at least two unique trajectories')
    selected = {r: [True] * len(pairs) for r in ('teacher', 'student')}
    groups = logical_minibatches(len(pairs), global_mini, dp_size)
    records = []
    for index, rows in enumerate(groups):
        candidates = {r: {supervision_identity(pairs[i], r) for i in rows} for r in selected}
        union = candidates['teacher'] | candidates['student']
        def order(identity):
            return digest({'seed': config.get('seed', 42), 'stage': stage,
                           'absolute_step': step, 'logical_minibatch': index,
                           'supervision_identity': identity})
        if not enabled:
            chosen = union
        else:
            # Both roles get a quota; identities appearing in both roles consume
            # one total slot and contribute to both observed role counts.
            teacher_quota = (budget + 1) // 3 if stage == 'gad' else budget // 2
            student_quota = budget - teacher_quota
            teacher = sorted(candidates['teacher'], key=order)[:teacher_quota]
            student = sorted(candidates['student'], key=order)[:student_quota]
            chosen = set(teacher) | set(student)
            chosen.update(sorted(union - chosen, key=order)[:max(0, budget - len(chosen))])
        for r in selected:
            for i in rows:selected[r][i] = supervision_identity(pairs[i], r) in chosen
        records.append({'logical_minibatch': index, 'rows': rows,
            'candidate_unique': len(union), 'selected_unique': len(chosen),
            'teacher_selected_unique': len(chosen & candidates['teacher']),
            'student_selected_unique': len(chosen & candidates['student']),
            'selected_identities': sorted(chosen)})
    return selected, records
