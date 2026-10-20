#!/usr/bin/env python3
"""Bounded Appendix E validation or explicit frozen-cohort completion; no training."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools')]
from config import compose_config, validate_config
from rubric.core.contracts import file_hash, judge_messages


def completion_partition(canonical, previous):
    """Reuse every parsed result, including masked/incorrect ones; send only unstarted IDs."""
    ordered = {row['id']: row for row in canonical}
    prior = {row['id']: row for row in previous}
    if len(ordered) != len(canonical) or len(prior) != len(previous) or ordered.keys() != prior.keys():
        raise ValueError('completion requires the same unique frozen IDs')
    reused, remaining = [], []
    for row in canonical:
        saved = prior[row['id']]
        if any(saved.get(key) != value for key, value in row.items()):
            raise ValueError('frozen input changed: ' + row['id'])
        result = saved['result']
        if 'parsed' in result:reused.append(saved)
        elif str(result.get('error', '')).startswith('not_run_'):remaining.append(row)
        else:raise ValueError('completion only sends unstarted rows, not unchanged failed retries')
    return reused, remaining


def merge_completion(canonical, reused, new_rows):
    """Keep original order and provenance; do not overwrite a previously completed response."""
    frozen = {row['id']: row for row in canonical}
    all_rows = {}
    for origin, group in [('previous_thinking', reused), ('completion', new_rows)]:
        for row in group:
            rid = row['id']
            if rid not in frozen or rid in all_rows or any(row.get(key) != value for key, value in frozen[rid].items()):
                raise ValueError('duplicate, foreign or changed completion row: ' + rid)
            all_rows[rid] = {**row, 'result_origin': row.get('result_origin', origin)}
    return [all_rows.get(row['id'], {**row, 'result': {'error': 'not_run_completion_interrupted', 'cacheable': False},
        'api_response': None, 'result_origin': 'completion'}) for row in canonical]


def complete_existing(args):
    """A new, explicitly requested budget; earlier Appendix E ledgers remain frozen."""
    import shutil
    from extension_config import service_identity
    from rubric.core.contracts import digest
    from calibrate_rubric_judge import benchmark_summary
    source = Path(args.complete_from).resolve()
    directory = Path(args.output).resolve()
    if directory.exists():raise ValueError('completion output must be a new directory')
    if not 0 < args.wall_seconds <= 3600:raise ValueError('completion wall budget must be in 1..3600')
    protocol = json.loads((source / 'protocol.json').read_text())
    canonical_path = Path(protocol['inputs_path'])
    canonical = [json.loads(s) for s in canonical_path.read_text().splitlines()]
    previous = [json.loads(s) for s in (source / 'samples.jsonl').read_text().splitlines()]
    if len(canonical) != 24 or digest(canonical) != protocol['fixture_sha256']:
        raise ValueError('expected unchanged original 24-case cohort')
    reused, remaining = completion_partition(canonical, previous)
    if len(reused) != 6 or len(remaining) != 18:
        raise ValueError('this authorized completion expects exactly six completed and eighteen unstarted rows')
    cfg_source = Path(protocol['config'])
    cfg = compose_config(cfg_source)
    if service_identity(cfg) != protocol['service_identity']:
        raise ValueError('judge/model/prompt/parser/decoding identity changed; cannot claim same-configuration completion')
    if cfg.get('resources.judge_gpus') != '6,7' or cfg.get('rubric.judge.concurrency') != 1:
        raise ValueError('completion must keep GPU6,7 and concurrency1')
    directory.mkdir(parents=True)
    cfg_path = directory / 'candidate_14b.yaml'
    shutil.copy2(cfg_source, cfg_path)
    shutil.copy2(canonical_path, directory / 'selected24.jsonl')
    shutil.copy2(source / 'samples.jsonl', directory / 'previous_samples.jsonl')
    (directory / 'remaining18.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in remaining))
    manifest = {'authorization': 'user requested completing the first 24-case experiment, reusing six existing results',
        'source': str(source), 'source_samples_sha256': file_hash(source / 'samples.jsonl'),
        'source_protocol_sha256': file_hash(source / 'protocol.json'), 'config_sha256': file_hash(cfg_path),
        'inputs_sha256': file_hash(canonical_path), 'reused_ids': [r['id'] for r in reused],
        'pending_ids': [r['id'] for r in remaining], 'identity': protocol['service_identity'],
        'known_semantic_failures_do_not_stop_fixed_completion': True,
        'source_ledger_unchanged': True, 'human_review': 'pending'}
    (directory / 'completion_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for path in ['gad/tools/validate_thinking_judge.py', 'gad/tools/calibrate_rubric_judge.py',
                 'gad/extensions/judge_client.py', 'gad/tools/extension_config.py',
                 'rubric/core/process_v2.py', 'rubric/core/contracts.py',
                 'rubric/prompts/judge_math_process_v2.txt', 'rubric/specs/math_process_v2.yaml']:
        target = directory / 'code' / path;target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    ledger = {'phase': 'appendix_E_complete_fixed24', 'started_epoch': time.time(),
        'wall_clock_limit_seconds': args.wall_seconds, 'maximum_applications': 18, 'maximum_http_attempts': 18,
        'actual_applications': 0, 'actual_http_attempts': 0, 'actual_training_steps': 0,
        'reused_previous_results': 6, 'source_ledger_preserved': str(source.parent / 'phase_budget.json')}
    def save():
        ledger['elapsed_seconds'] = time.time() - ledger['started_epoch']
        (directory / 'phase_budget.json').write_text(json.dumps(ledger, indent=2))
    save()
    output = directory / 'new18';output.mkdir()
    argv = [cfg.require('run.python_bin'), str(ROOT / 'gad/tools/calibrate_rubric_judge.py'),
        '--config', str(cfg_path), '--samples', str(directory / 'remaining18.jsonl'), '--output', str(output),
        '--wall-seconds', str(max(1, args.wall_seconds - 15)), '--finish-cohort']
    try:
        with (output / 'run.log').open('w') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'})
            try:ledger['exit_code'] = process.wait(timeout=max(1, args.wall_seconds - 5))
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid, signal.SIGKILL);process.wait()
                raise
        if ledger['exit_code']:raise RuntimeError('completion child failed; preserve partial outputs')
    except BaseException as exc:
        ledger['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Only clean a judge launched by this exact child; a startup rejection
        # must not stop somebody else's pre-existing service on the same port.
        import re
        from urllib.parse import urlparse
        owner_file = ROOT / 'runs/_rubric_judge' / str(urlparse(cfg.get('rubric.judge.base_url')).port) / 'owner.json'
        startup = output / 'service_start.log'
        launched = re.search(r'Started judge pid=(\d+)', startup.read_text()) if startup.exists() else None
        owner = json.loads(owner_file.read_text()) if owner_file.exists() else None
        if launched and owner and owner['pid'] == int(launched.group(1)):
            stopped = subprocess.run(['bash', str(ROOT / 'scripts/rubric_judge.sh'), '--config', str(cfg_path),
                '--action', 'stop'], cwd=ROOT, capture_output=True, text=True, timeout=15)
            ledger['cleanup'] = {'exit_code': stopped.returncode, 'stdout': stopped.stdout.strip()}
        else:
            ledger['cleanup'] = {'action': 'no_matching_completion_owner', 'owner_file_exists': bool(owner)}
        if (output / 'budget.json').exists():
            budget = json.loads((output / 'budget.json').read_text());ledger['new18_budget'] = budget
            ledger['actual_applications'] = budget['actual_applications'];ledger['actual_http_attempts'] = budget['actual_http_attempts']
        samples_path = next((q for q in [output / 'samples.jsonl', output / 'samples.partial.jsonl',
                            output / 'samples.stream.jsonl'] if q.exists()), None)
        current = [json.loads(s) for s in samples_path.read_text().splitlines()] if samples_path else []
        combined = merge_completion(canonical, reused, current)
        (directory / 'combined24.samples.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in combined))
        summary = benchmark_summary(combined)
        summary.update(reused_previous_results=6, new_application_requests=ledger['actual_applications'],
            combined_usage_includes_six_previous_results=True, semantic_accuracy=None, human_review='pending')
        (directory / 'combined24.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        ledger['status'] = 'failed' if 'error' in ledger else ledger.get('new18_budget', {}).get('status', 'unknown')
        save();print(json.dumps(ledger, indent=2))


def remaining_budget(ledger, applications, now=None):
    if ledger['actual_applications'] + applications > 48:
        raise ValueError('Appendix E 48-application limit exceeded')
    remaining = math.floor(1800 - ((time.time() if now is None else now) - ledger['started_epoch']))
    if remaining <= 30:raise TimeoutError('shared Appendix E wall budget exhausted; no judge launched')
    return remaining


def prepare(directory):
    import yaml
    from transformers import AutoTokenizer
    from extension_config import service_identity
    inputs = directory / 'selected24.jsonl'
    rows = [json.loads(line) for line in inputs.read_text().splitlines()]
    source = ROOT / 'reports/rubric_v2/judge_protocol_v21_20261015/selected24.jsonl'
    original = [json.loads(line) for line in source.read_text().splitlines()]
    if len(rows) != 24 or {r['id'] for r in rows} != {r['id'] for r in original}:
        raise ValueError('Appendix E must reuse the frozen 24 IDs')
    old = {r['id']: r for r in original}
    if any(row != old[row['id']] for row in rows):raise ValueError('frozen response content changed')
    for model in ('14b', '8b'):
        target = directory / ('candidate_' + model + '.yaml')
        cfg = compose_config(model='qwen3_0.6b', dataset='dapo_math_5k', overrides=[
            str(ROOT / 'configs/overrides/rubric_v2_joint.yaml'),
            str(ROOT / f'configs/overrides/rubric_thinking_{model}_candidate.yaml')])
        problems = validate_config(cfg)
        if problems:raise ValueError(problems)
        text = yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=True)
        if target.exists() and target.read_text() != text:raise ValueError('frozen candidate config changed')
        target.write_text(text)
        options = {k[13:]: v for k, v in cfg.data.items() if k.startswith('rubric.judge.')}
        tokenizer = AutoTokenizer.from_pretrained(options['tokenizer_path'], local_files_only=True)
        spec, prompt = Path(cfg.require('rubric.spec')).read_text(), Path(cfg.require('rubric.prompt')).read_text()
        audits = []
        examples = {}
        for row in rows:
            messages = judge_messages(row['prompt'], row['response'], row['ground_truth'], spec, prompt,
                cfg.get('rubric.version'), {'response_source_truncated': row['truncated'],
                    'training_token_truncated': False, 'prompt_truncated': False, 'judge_context_overflow': False})
            count = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=options['enable_thinking']))
            audits.append({'id': row['id'], 'input_tokens': count, 'reserved_total_output': options['max_output_tokens'],
                'max_context_tokens': options['max_context_tokens'],
                'exceeds': count + options['max_output_tokens'] > options['max_context_tokens']})
            if len(examples) < 2:examples[row['id']] = messages
        (directory / f'capacity_{model}.json').write_text(json.dumps({'model': options['model'],
            'actor_enable_thinking': cfg.get('rollout.enable_thinking'), 'judge_enable_thinking': options['enable_thinking'],
            'service_identity': service_identity(cfg), 'requests': audits, 'request_examples': examples}, ensure_ascii=False, indent=2))
    print('Prepared frozen thinking candidates and exact-template CPU capacity audit; no model started.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'reports/rubric_v2/thinking_judge_20261015'))
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--model', choices=('14b', '8b'), default='14b')
    parser.add_argument('--screening', help='Required independent/model-assisted semantic review gate before 8B')
    parser.add_argument('--compatibility-retry', action='store_true', help='One documented affected 14B retry in the SAME window')
    parser.add_argument('--fix-note')
    parser.add_argument('--complete-from', help='Explicitly requested new window to finish the 18 unstarted rows of a previous six-result run')
    parser.add_argument('--wall-seconds', type=int, default=1800, help='New completion window only; at most 3600, does not reset old Appendix E windows')
    args = parser.parse_args()
    if args.complete_from:return complete_existing(args)
    directory = Path(args.output).resolve()
    if args.prepare_only:return prepare(directory)
    inputs = directory / 'selected24.jsonl'
    if not (directory / 'semantic_review_plan.jsonl').exists():raise ValueError('prepare semantic review before model outputs')
    ledger_path = directory / 'phase_budget.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger['inputs_sha256'] != file_hash(inputs):raise ValueError('frozen cohort changed')
        if args.compatibility_retry:
            if args.model != '14b' or len(ledger['rounds']) != 1 or not args.fix_note or ledger['rounds'][0].get('exit_code') == 0:
                raise ValueError('only one documented compatibility fix may retry an incomplete 14B round')
        else:
            if args.model != '8b' or any(r['model'] != '14b' for r in ledger['rounds']):
                raise ValueError('existing phase cannot be restarted or reset')
            if not args.screening:raise ValueError('8B is gated on 14B mathematical screening, not JSON success')
            screening = json.loads(Path(args.screening).read_text())
            if screening.get('critical_semantics_passed') is not True or screening.get('review_basis') not in ('independent_human', 'model_assisted'):
                raise ValueError('14B has not passed independent mathematical screening')
            prior = directory / ledger['rounds'][-1].get('name', '14b') / 'samples.jsonl'
            if screening.get('samples_sha256') != file_hash(prior):raise ValueError('screening identity mismatch')
    else:
        if args.compatibility_retry:raise ValueError('compatibility retry requires a failed first round')
        if args.model != '14b':raise ValueError('14B thinking must be screened first')
        ledger = {'phase': 'appendix_E', 'started_epoch': time.time(), 'wall_clock_limit_seconds': 1800,
            'maximum_applications': 48, 'actual_applications': 0, 'actual_http_attempts': 0,
            'actual_training_steps': 0, 'inputs_sha256': file_hash(inputs),
            'semantic_plan_sha256': file_hash(directory / 'semantic_review_plan.jsonl'), 'rounds': []}
    remaining = remaining_budget(ledger, 24)
    name = args.model + ('_compat_retry' if args.compatibility_retry else '')
    output = directory / name
    if output.exists():raise ValueError('round already exists; refusing to overwrite')
    output.mkdir()
    cfg_path = directory / f'candidate_{args.model}.yaml'
    record = {'model': args.model, 'name': name, 'fix_note': args.fix_note,
        'maximum_applications': 24, 'config_sha256': file_hash(cfg_path)}
    ledger['rounds'].append(record)
    def save():
        ledger['elapsed_seconds'] = time.time() - ledger['started_epoch']
        ledger_path.write_text(json.dumps(ledger, indent=2))
    save()
    cfg = compose_config(cfg_path)
    stop_file = directory / f'stop_{name}.json'
    if stop_file.exists():raise ValueError('review stop marker already exists')
    argv = [cfg.require('run.python_bin'), str(ROOT / 'gad/tools/calibrate_rubric_judge.py'),
        '--config', str(cfg_path), '--samples', str(inputs), '--output', str(output),
        '--wall-seconds', str(remaining - 10), '--stop-file', str(stop_file)]
    try:
        with (output / 'run.log').open('w') as stream:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'})
            try:record['exit_code'] = process.wait(timeout=remaining - 5)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid, signal.SIGKILL);process.wait()
                raise
        if record['exit_code']:
            raise RuntimeError('judge check did not complete; inspect round budget and raw stream')
    except BaseException as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        stopped = subprocess.run(['bash', str(ROOT / 'scripts/rubric_judge.sh'), '--config', str(cfg_path),
            '--action', 'stop'], cwd=ROOT, capture_output=True, text=True, timeout=15)
        record['cleanup'] = {'exit_code': stopped.returncode, 'stdout': stopped.stdout.strip()}
        if (output / 'budget.json').exists():record['budget'] = json.loads((output / 'budget.json').read_text())
        ledger['actual_applications'] = sum(r.get('budget', {}).get('actual_applications', 0) for r in ledger['rounds'])
        ledger['actual_http_attempts'] = sum(r.get('budget', {}).get('actual_http_attempts', 0) for r in ledger['rounds'])
        ledger['status'] = ('review_or_compatibility_stopped' if stop_file.exists() else
            'failed' if 'error' in record else 'round_completed')
        save()
        print(json.dumps(ledger, indent=2))


if __name__ == '__main__':main()
