"""Speculative Teacher work uses the same labels, keys and bounded journal."""
import time
from .common import append_json
from .critic_batch import extract_pairs, supervision_identity

_FROM_META = object()


def teacher_pairs(gen_batch, raw_batch, tokenizer, token_identity, step, stage, repeat, *, eos_token_id=_FROM_META):
    import torch
    from verl import DataProto
    from verl.utils.torch_functional import get_response_mask
    prompt, teacher = gen_batch.batch['input_ids'], gen_batch.batch['teacher_response']
    eos = (gen_batch.meta_info.get('eos_token_id', tokenizer.eos_token_id)
           if eos_token_id is _FROM_META else eos_token_id)
    teacher_mask = get_response_mask(response_id=teacher,
        eos_token=eos,
        dtype=gen_batch.batch['attention_mask'].dtype)
    attention = torch.cat([gen_batch.batch['attention_mask'], teacher_mask], dim=-1)
    # Student fields are placeholders only; no Student is selected or annotated.
    view = DataProto.from_dict(tensors={'prompts': prompt, 'responses': teacher,
        'teacher_response': teacher, 'attention_mask': attention, 'teacher_attention_mask': attention},
        non_tensors={**raw_batch.non_tensor_batch, **gen_batch.non_tensor_batch})
    view = view.repeat(repeat_times=repeat, interleave=True)
    return extract_pairs(view, tokenizer, token_identity, step, stage)


def selected_teachers(pairs, global_mini, dp_size, options, step, stage):
    from .process_sampling import logical_minibatches, selection_order
    budget = options.get('process_sampling', {}).get('max_unique_per_global_minibatch')
    if options.get('mode') != 'joint' or budget is None:
        return []
    quota = (budget + 1) // 3 if stage == 'gad' else budget // 2
    chosen = {}
    for index, rows in enumerate(logical_minibatches(len(pairs), global_mini, dp_size)):
        candidates = {supervision_identity(pairs[i], 'teacher'): pairs[i] for i in rows}
        identities = sorted(candidates, key=lambda identity: selection_order(options, step, stage, index, identity))[:quota]
        for identity in identities:
            chosen[identity] = candidates[identity]
    return list(chosen.values())


def prefetch(owner, pairs, step, *, scorer_results=None):
    from .annotation_work import build_labels, execute_requests
    started = time.monotonic()
    ledger_start = len(owner.judge.api_ledger)
    health_before = getattr(owner.judge, 'health_get_count', 0)
    keys = {owner.cache_key(pair, 'teacher') for pair in pairs}
    results, requests, values = {}, {}, {}
    error = None
    try:
        deadline = started + owner.options['judge']['step_timeout_seconds']
        owner.judge.health(deadline=deadline)
        unique, requests, values, hits = build_labels(owner, pairs, keys,
            owner.annotation_schema_version, roles=('teacher',), scorer_results=scorer_results)
        results, recovered = execute_requests(owner, unique, requests, values, step, deadline, phase='teacher_prefetch')
    except Exception as exc:
        # Do not make a speculative failure into a failed optimizer step. Main
        # annotation still enforces its original health, masks and student guard.
        error = f'{type(exc).__name__}: {exc}'
        append_json(owner.directory / 'teacher_prefetch_errors.jsonl', {'step': step, 'error': error})
        results = getattr(exc, 'results', {})
    sends = owner.judge.api_ledger[ledger_start:]
    cached = {key for key in keys if owner.cache.get(key) is not None}
    record = {'step': step, 'candidate_keys': sorted(keys), 'cached_keys': sorted(cached),
              'seconds': time.monotonic() - started, 'error': error,
              'requests': len(requests), 'api_sends': len(sends),
              'health_gets': getattr(owner.judge, 'health_get_count', 0) - health_before,
              'json_valid_count': sum('parsed' in r for r in results.values()),
              'json_requested_count': len(results),
              'failures': sum('parsed' not in r for r in results.values()),
              'prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in sends),
              'output_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in sends)}
    append_json(owner.directory / 'teacher_prefetch.jsonl', record)
    return record
