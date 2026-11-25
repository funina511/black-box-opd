#!/usr/bin/env python3
"""Bounded judge evaluation on fixed diagnostics or a frozen response pool."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import signal
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'gad/tools'))
from config import compose_config
from extension_config import service_identity
from rubric.core.contracts import digest, file_hash, judge_messages, schema_for
from gad.extensions.judge_client import JudgeClient, RequestBudget
from validate_rubric_v2 import fixtures

DIMENSIONS = ('verified_progress', 'unresolved_error_severity')
# Model-assisted anchors, not human acceptance or a benchmark accuracy score.
ANCHORS = {'complete': (1, 0), 'prefix': (.5, 0), 'explicit_error': (.5, 1),
           'claimed_repair': (.5, 1), 'real_repair': (1, 0), 'restatement': (0, None),
           'repeat': (.5, 0), 'alternative': (1, 0), 'pending_branch': (.5, 0),
           'no_math': (0, None)}


def labels(row):
    parsed = row.get('revalidated_contract') or row.get('result', {}).get('parsed', {})
    return {d: parsed.get(d, {}).get('value') for d in DIMENSIONS}


def benchmark_summary(samples):
    def statistics(rows):
        parsed = [r for r in rows if 'parsed' in r['result']]
        attempted = [r for r in rows if r['result'].get('attempts',0)>0 or r['result'].get('api_attempts')]
        return {
            'planned': len(rows), 'attempted': len(attempted), 'parsed': len(parsed),
            'rows_with_contract_violations': sum(bool(r['result']['parsed'].get('_violations')) for r in parsed),
            'rows_with_local_event_failures': sum(bool(r['result']['parsed'].get('_event_issues')) for r in parsed),
            'errors': dict(Counter(r['result']['error'] for r in rows if 'error' in r['result'])),
            'effective_coverage_of_planned': {d: sum(bool(r['result']['parsed'][d]['valid']) for r in parsed) / len(rows) if rows else None for d in DIMENSIONS},
            'effective_labels': {d: dict(Counter(str(r['result']['parsed'][d]['value']) for r in parsed)) for d in DIMENSIONS},
            'raw_labels': {d: dict(Counter(str(r['result']['parsed']['_record'][d]['value']) for r in parsed)) for d in DIMENSIONS},
            'mathematical_unknowns': {d: sum(not r['result']['parsed']['_record'][d]['valid'] for r in parsed) for d in DIMENSIONS},
            'program_discarded_dimensions': {d: sum(bool(r['result']['parsed'].get('_mask_reasons', {}).get(d)) for r in parsed) for d in DIMENSIONS},
            'cache_hits': 0,
            'http_generation_attempts':sum(r['result'].get('attempts',0) for r in rows),
            'targeted_generation_attempts':sum(r['result'].get('targeted_result',{}).get('attempts',0) for r in rows),
            'unique_responses':len({r.get('source_response_id',r.get('response_id',r['id'])) for r in rows}),
            'independent_questions':len({r.get('uid',digest(r.get('prompt',''))) for r in rows}),
            'both_dimensions_valid':sum(all(r['result']['parsed'][d]['valid'] for d in DIMENSIONS) for r in parsed),
            'human_verified':sum(r['result']['parsed'].get('_semantic_review',{}).get('human_verified',False) for r in parsed),
            'model_supported_only':sum(any(v=='model_supported' for v in r['result']['parsed'].get('_semantic_review',{}).get('dimensions',{}).values()) for r in parsed),
            'usage': {key: sum(r['result'].get('usage', {}).get(key, 0) for r in rows)
                      for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        }
    by_kind = {kind: statistics([r for r in samples if r['kind'] == kind]) for kind in ('real', 'boundary')}
    anchor_rows = []
    for row in samples:
        if row['kind'] != 'boundary':
            continue
        parsed = row['result'].get('parsed', {})
        raw = parsed.get('_record', {})
        anchor_rows.append({'id': row['id'], 'category': row['category'], 'expected_allowed': row['expected_allowed'],
                            'raw': {d: raw.get(d, {}).get('value') for d in DIMENSIONS},
                            'effective': labels(row),
                            'comparison_status': 'available' if parsed else 'not_attempted' if str(row['result'].get('error', '')).startswith('not_run_') else 'technical_failure',
                            'raw_mismatch_dimensions': [d for d in DIMENSIONS if raw and raw[d]['value'] not in row['expected_allowed'][d]],
                            'effective_mismatch_dimensions': [d for d in DIMENSIONS if parsed and parsed[d]['value'] not in row['expected_allowed'][d]],
                            'contract_violations': parsed.get('_violations', []), 'local_event_failures': parsed.get('_event_issues', []),
                            'unknown_or_masked_dimensions': [d for d in DIMENSIONS if not parsed.get(d, {}).get('valid')],
                            'error': row['result'].get('error')})
    cross_tab = Counter()
    for row in samples:
        if row['kind'] == 'real':
            result = labels(row)
            cross_tab[(str(row['scoring']['correct']), str(result[DIMENSIONS[0]]), str(result[DIMENSIONS[1]]))] += 1
    return {'by_kind': by_kind, 'overall': statistics(samples),
            'boundary_raw_anchor_mismatch_count': sum(bool(r['raw_mismatch_dimensions']) for r in anchor_rows),
            'boundary_effective_anchor_mismatch_count': sum(bool(r['effective_mismatch_dimensions']) for r in anchor_rows),
            'boundary_not_evaluable_count': sum(r['comparison_status'] != 'available' for r in anchor_rows),
            'boundary_details': anchor_rows,
            'real_by_role': {role: statistics([r for r in samples if r['kind'] == 'real' and r['source_role'] == role]) for role in ('teacher', 'student')},
            'real_by_truncation': {str(flag): statistics([r for r in samples if r['kind'] == 'real' and r['truncated'] == flag]) for flag in (False, True)},
            'real_by_length': {group: statistics([r for r in samples if r['kind'] == 'real' and r.get('length_group') == group]) for group in ('short', 'long')},
            'outcome_cross_tab': [{'answer_correct': key[0], 'progress': key[1], 'error': key[2], 'count': count} for key, count in sorted(cross_tab.items())],
            'human_review': 'pending', 'semantic_accuracy': None,
            'interpretation': 'Contract validity and rule-scorer answer correctness are not human process-label accuracy. Boundary anchors are model-assisted; real sample is stratified, not population weighted.'}


def wait_for_futures(futures, timeout):
    """Python 3.10 futures.TimeoutError is distinct from builtin TimeoutError."""
    try:
        for future in as_completed(futures, timeout=timeout):
            future.result()
    except (TimeoutError, FuturesTimeoutError):
        return False
    return True


def stop_owned_service(service, pid, birth, hard_deadline):
    """Bound cleanup and still kill our verified group if graceful stop times out."""
    from rubric_judge import process_birth
    def verify():
        if Path(f'/proc/{pid}').exists() and process_birth(pid) != birth:
            raise RuntimeError('judge process birth changed; refusing foreign process cleanup')
    verify()
    timeout=min(30.,max(.1,hard_deadline-time.monotonic()-8))
    try:
        result=subprocess.run(service+['--action','stop'],cwd=ROOT,capture_output=True,text=True,timeout=timeout)
        record={'exit_code':result.returncode,'stdout':result.stdout.strip(),'stderr':result.stderr.strip()}
    except subprocess.TimeoutExpired:
        record={'exit_code':None,'error':'graceful_stop_timeout','timeout_seconds':timeout}
    end=min(time.monotonic()+8,hard_deadline-1)
    while time.monotonic()<end:
        try:os.killpg(pid,0)
        except ProcessLookupError:return record
        time.sleep(.2)
    verify()
    try:
        os.killpg(pid,signal.SIGKILL)
        record['forced_owned_group_cleanup']=True
    except ProcessLookupError:pass
    return record


def main():
    def terminate(signum, frame):raise KeyboardInterrupt('bounded judge task terminated')
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous', default=str(ROOT / 'reports/rubric_v2/judge_samples_guarded.jsonl'))
    parser.add_argument('--samples', help='Frozen inputs.jsonl; at most 128 rows, one attempt per row')
    parser.add_argument('--max-requests', type=int, default=24, help='Total HTTP generation calls, including retries and targeted review')
    parser.add_argument('--wall-seconds', type=int, default=1800)
    parser.add_argument('--concurrency-groups', help='Shared-service/shared-budget arms, e.g. 1,2; separate results and review dedup per arm')
    parser.add_argument('--finish-cohort', action='store_true', help='User-authorized fixed-cohort completion: retain per-row mode/budget failures and continue; stop on request rejection or repeated transport failures')
    parser.add_argument('--stop-file', help='Stop queue after the current request; retain completed outputs and summarize unstarted rows')
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 3600:
        parser.error('wall-seconds must be in 1..3600 (default 1800)')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'budget.json').exists():
        raise ValueError('output already contains a calibration; choose a new directory')
    groups=[int(v) for v in args.concurrency_groups.split(',')] if args.concurrency_groups else None
    if groups and (not args.samples or groups != [1,2]):parser.error('bounded comparison requires --samples and --concurrency-groups 1,2')
    cfg = compose_config(args.config)
    if cfg.get('rubric.version') != 'math_process_v2':
        raise ValueError('this fixed comparison requires math_process_v2')
    options = {k[13:]: v for k, v in cfg.data.items() if k.startswith('rubric.judge.')}
    if args.max_requests < 1:parser.error('max-requests must be positive')
    configured_budget=options.get('request_budget')
    options.update(rubric_version=cfg.get('rubric.version'), seed=cfg.get('rubric.seed'), max_retries=0, request_budget=min(args.max_requests,configured_budget) if configured_budget is not None else args.max_requests)
    examples = ([json.loads(line) for line in Path(args.samples).read_text().splitlines() if line.strip()]
                if args.samples else fixtures())
    if not 0 < len(examples) <= (128 if args.samples else 12) or len({r['id'] for r in examples}) != len(examples):
        raise ValueError('invalid frozen sample size or duplicate sample IDs')
    if groups and len(examples)!=4:parser.error('Appendix A comparison requires the four frozen responses')
    from urllib.parse import urlparse
    port = urlparse(options['base_url']).port
    directory = ROOT / 'runs/_rubric_judge' / str(port)
    owner_path = directory / 'owner.json'
    if owner_path.exists():
        owner = json.loads(owner_path.read_text())
        if Path(f"/proc/{owner['pid']}").exists():
            raise ValueError('dedicated port already has an owned process; choose another port')
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))

    started = time.monotonic()
    hard_deadline = started + args.wall_seconds
    cleanup_reserve = min(45, max(3, args.wall_seconds // 4))
    deadline = hard_deadline - cleanup_reserve
    budget = {'status': 'running', 'authorization': 'user requested frozen bounded response cohort' if args.samples else 'user requested downloaded model test',
              'maximum_applications': len(examples)*(len(groups) if groups else 1), 'actual_applications': 0, 'maximum_http_attempts': options['request_budget'],
              'actual_http_attempts': 0, 'completed_applications': 0,
              'maximum_training_steps': 0, 'actual_training_steps': 0,
              'wall_clock_limit_seconds': args.wall_seconds, 'cleanup_reserve_seconds': cleanup_reserve, 'gpu_ids': cfg.get('resources.judge_gpus')}
    stop_monitor = threading.Event()
    gpu_samples = []
    def save_budget():
        budget['elapsed_seconds'] = time.monotonic() - started
        (output / 'budget.json').write_text(json.dumps(budget, indent=2))
    def monitor():
        ids = set(str(cfg.require('resources.judge_gpus')).split(','))
        while not stop_monitor.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                for line in result.stdout.splitlines():
                    index, memory, utilization = [part.strip() for part in line.split(',')]
                    if index in ids:
                        gpu_samples.append({'elapsed_seconds': time.monotonic() - started, 'gpu': index,
                                            'memory_mib': int(memory), 'utilization': int(utilization)})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            stop_monitor.wait(2)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    save_budget()
    service = ['bash', str(ROOT / 'scripts/rubric_judge.sh'), '--config', str(Path(args.config).resolve())]
    owned_pid = None
    owned_birth = None
    raw_responses = {}
    results = {}
    io_lock = threading.Lock()
    try:
        before = time.monotonic()
        with (output / 'service_start.log').open('w') as stream:
            subprocess.run(service + ['--action', 'start'], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=max(1, deadline - time.monotonic()), check=True)
        owner_record = json.loads(owner_path.read_text())
        owned_pid, owned_birth = owner_record['pid'], owner_record['birth']
        spec = Path(cfg.require('rubric.spec')).read_text()
        prompt = Path(cfg.require('rubric.prompt')).read_text()
        requests = {e['id']: judge_messages(e['prompt'], e['response'], e['ground_truth'], spec, prompt,
                    'math_process_v2', {'response_source_truncated': e['truncated'],
                    'training_token_truncated': False, 'prompt_truncated': False, 'judge_context_overflow': False})
                    for e in examples}
        request_ids = {digest(messages): key for key, messages in requests.items()}
        class RecordingClient(JudgeClient):
            def request(self, suffix, payload=None, timeout=None):
                if suffix == '/chat/completions':
                    with io_lock:
                        budget['actual_http_attempts'] += 1
                        save_budget()
                result = super().request(suffix, payload, timeout)
                if suffix == '/chat/completions':
                    rid = request_ids.get(digest(payload['messages']), 'targeted_' + digest(payload['messages'])[:20])
                    with io_lock:
                        raw_responses[rid] = result
                        with (output / 'raw_api_stream.jsonl').open('a') as stream:
                            stream.write(json.dumps({'id': rid, 'group':getattr(self,'group',None), 'api_response': result}, ensure_ascii=False)+'\n')
                return result
        client = RecordingClient(options, digest(service_identity(cfg)))
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('judge startup exhausted wall budget')
            if not Path(f'/proc/{owned_pid}').exists():
                raise RuntimeError('judge process exited; inspect copied server log')
            try:
                client.health(deadline=min(deadline, time.monotonic() + 3))
                break
            except RuntimeError:
                time.sleep(min(2, max(0, deadline - time.monotonic())))
        budget['startup_seconds'] = time.monotonic() - before
        (output / 'protocol.json').write_text(json.dumps({
            'config': str(Path(args.config).resolve()), 'config_sha256': file_hash(args.config),
            'spec_sha256': file_hash(cfg.require('rubric.spec')), 'prompt_sha256': file_hash(cfg.require('rubric.prompt')),
            'schema_sha256': digest(schema_for('math_process_v2')), 'fixture_sha256': digest(examples),
            'service_identity': service_identity(cfg), 'client_options': options,
            'human_review': 'pending', 'prompt_mutated_during_run': False, 'training_started': False,
            'inputs_path': str(Path(args.samples).resolve()) if args.samples else None,
            'script_sha256': file_hash(__file__)}, indent=2))
        (output / 'requests.json').write_text(json.dumps(requests, ensure_ascii=False, indent=2))
        before = time.monotonic()
        from gad.extensions.annotation_scheduler import annotate_chains
        shared_budget=RequestBudget(options['request_budget'])
        arm_results={};arm_clients={}
        for group_index,concurrency in enumerate(groups or [options['concurrency']],1):
            arm=f'B{group_index}' if groups else 'single'
            arm_output=output/arm if groups else output
            arm_output.mkdir(exist_ok=True)
            if groups:
                client=RecordingClient({**options,'concurrency':concurrency},digest(service_identity(cfg)),
                                       tokenizer=client.tokenizer,request_budget=shared_budget)
                client.group=arm
            else:client.shared_budget=shared_budget
            arm_clients[arm]=client
            results={};abort_requests=threading.Event()
            def stop():
                if args.stop_file and Path(args.stop_file).exists():
                    budget['stop_reason']='semantic_or_compatibility_review_stop';abort_requests.set()
                return abort_requests.is_set()
            def completed(rid,result):
                results[rid]=result
                if result.get('error')=='judge_transport_error':
                    budget['consecutive_transport_failures']=budget.get('consecutive_transport_failures',0)+1
                    if budget['consecutive_transport_failures']>=4:
                        budget['stop_reason']='repeated_transport_failures';abort_requests.set()
                else:budget['consecutive_transport_failures']=0
                fatal=('judge_request_rejected','judge_request_budget_exhausted')
                if not args.finish_cohort:fatal+=('judge_thinking_not_observed','judge_reasoning_budget_exhausted')
                if result.get('error') in fatal or result.get('targeted_result',{}).get('error')=='judge_request_budget_exhausted':
                    budget['stop_reason']=result.get('error') or 'judge_request_budget_exhausted';abort_requests.set()
                example=next(e for e in examples if e['id']==rid)
                with (arm_output/'samples.stream.jsonl').open('a') as stream:
                    stream.write(json.dumps({**example,'result':result,'human_review':'pending'},ensure_ascii=False)+'\n')
                save_budget()
                print(f"{arm} completed {len(results)}/{len(examples)} {rid} error={result.get('error')}",flush=True)
            contexts={e['id']:(e['prompt'],e['response']) for e in examples}
            results=annotate_chains(client,requests,contexts,deadline-2,
                semantic=cfg.get('rubric.semantic_check'),on_result=completed,stop=stop)
            arm_samples=[{**e,'result':results[e['id']],'human_review':'pending'} for e in examples]
            summary=benchmark_summary(arm_samples)
            summary.update(concurrency=concurrency, scheduler=client.last_schedule,
                actual_api_peak=client.api_peak, api_ledger=client.api_ledger,
                batch_wall_seconds=client.last_schedule['batch_wall_seconds'],
                responses_per_second=sum(r.get('attempts',0)>0 and not r.get('pending_scheduler_state') for r in results.values())/max(client.last_schedule['batch_wall_seconds'],1e-9),
                output_tokens_per_second=summary['overall']['usage']['completion_tokens']/max(client.last_schedule['batch_wall_seconds'],1e-9))
            (arm_output/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in arm_samples))
            (arm_output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
            arm_results[arm]=summary
            budget['arms']=arm_results;budget['shared_reserved_attempts']=shared_budget.count
            budget['actual_applications']=sum(s['overall']['attempted'] for s in arm_results.values())
            budget['completed_applications']=sum(s['overall']['parsed'] for s in arm_results.values())
            save_budget()
            if budget.get('stop_reason'):break
        if groups:
            budget['status']='completed' if len(arm_results)==2 and all(not s['scheduler']['unfinished'] for s in arm_results.values()) else 'partial_budget_or_stop'
            (output/'comparison.json').write_text(json.dumps(arm_results,ensure_ascii=False,indent=2))
            return
        budget['annotation_seconds'] = time.monotonic() - before
        not_run = 'not_run_' + budget['stop_reason'] if budget.get('stop_reason') else 'not_run_wall_budget'
        samples = [{**example, 'result': results.get(example['id'], {'error': not_run, 'cacheable': False}),
                    'api_response': raw_responses.get(example['id']), 'human_review': 'pending'} for example in examples]
        (output / 'samples.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in samples))
        if args.samples:
            summary = benchmark_summary(samples)
            summary.update(model=options['model'], annotation_seconds=budget['annotation_seconds'])
            (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            budget['status'] = ('completed' if not client.last_schedule['unfinished'] and not budget.get('stop_reason') else
                'review_stopped' if budget.get('stop_reason') == 'semantic_or_compatibility_review_stop' else
                'judge_stopped' if budget.get('stop_reason') else 'wall_budget_stopped')
            print(json.dumps(summary['by_kind'], ensure_ascii=False, indent=2), flush=True)
            return
        previous = {row['id']: row for row in (json.loads(line) for line in Path(args.previous).read_text().splitlines())}
        comparison = []
        for row in samples:
            parsed = row['result'].get('parsed', {})
            actual = labels(row)
            comparison.append({'id': row['id'], 'previous': labels(previous[row['id']]), 'current': actual,
                               'anchor': dict(zip(DIMENSIONS, ANCHORS[row['id']])) if row['id'] in ANCHORS else None,
                               'violations': parsed.get('_violations', []), 'error': row['result'].get('error')})
        mismatches = [row for row in comparison if row['anchor'] is not None and row['current'] != row['anchor']]
        failures = sum('parsed' not in r for r in results.values())
        violations = sum(bool(r.get('parsed', {}).get('_violations')) for r in results.values())
        summary = {'model': options['model'], 'sample_count': len(samples), 'parsed_count': len(samples) - failures,
                   'syntax_or_evidence_failures': failures, 'rows_with_contract_violations': violations,
                   'anchor_case_count': len(ANCHORS), 'anchor_mismatches': mismatches,
                   'all_model_assisted_anchors_matched': not failures and not violations and not mismatches,
                   'human_review': 'pending', 'annotation_seconds': budget['annotation_seconds'],
                   'usage': {key: sum(r.get('usage', {}).get(key, 0) for r in results.values())
                             for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
                   'label_distributions': {d: dict(Counter(str(labels(row)[d]) for row in samples)) for d in DIMENSIONS},
                   'comparison': comparison}
        (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        budget['status'] = 'completed'
        budget['model_assisted_anchors_matched'] = summary['all_model_assisted_anchors_matched']
        print(json.dumps({key: value for key, value in summary.items() if key != 'comparison'}, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        budget.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        # The launcher may have created an owner before its command timed out.
        if owned_pid is None and owner_path.exists():
            owner_record = json.loads(owner_path.read_text())
            owned_pid, owned_birth = owner_record['pid'], owner_record['birth']
        if owned_pid is not None:
            budget['service_stop'] = stop_owned_service(service,owned_pid,owned_birth,hard_deadline)
            if args.samples and 'executor' in locals():executor.shutdown(wait=True, cancel_futures=True)
        if args.samples and not groups and not (output / 'samples.jsonl').exists():
            # Interruption/fatal startup must not erase successful or failed rows.
            partial = [{**example, 'result': results.get(example['id'],
                {'error': 'not_run_interrupted', 'cacheable': False}),
                'api_response': raw_responses.get(example['id']), 'human_review': 'pending'} for example in examples]
            (output / 'samples.partial.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in partial))
            summary = benchmark_summary(partial)
            summary.update(model=options['model'], run_status=budget['status'], actual_completed_set=False)
            (output / 'summary.partial.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        if 'arm_clients' in locals():
            budget['final_arms']={arm:{'api_ledger':c.api_ledger,'api_active':c.api_active,'actual_requests':c.request_count,'targeted_dedup_size':len(c.targeted_ids)} for arm,c in arm_clients.items()}
        if 'shared_budget' in locals():budget['shared_reserved_attempts']=shared_budget.count
        if 'client' in locals():
            budget['final_client_api_ledger']=client.api_ledger
            budget['remaining_client_inflight']=client.api_active
        stop_monitor.set()
        monitor_thread.join(timeout=6)
        (output / 'gpu_samples.json').write_text(json.dumps(gpu_samples, indent=2))
        budget['peak_memory_mib'] = {gpu: max(s['memory_mib'] for s in gpu_samples if s['gpu'] == gpu)
                                    for gpu in sorted({s['gpu'] for s in gpu_samples})}
        if (directory / 'server.log').exists():
            import shutil
            shutil.copy2(directory / 'server.log', output / 'server.log')
        save_budget()
        print(json.dumps(budget, indent=2), flush=True)


if __name__ == '__main__':
    main()
