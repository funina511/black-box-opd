"""Driver-only OpenAI-compatible client for verified vLLM 0.8.5 guided JSON."""
import json
import os
import time
import threading
import urllib.error
import urllib.request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from .core.contracts import schema_for, parse_judge, V2_INPUT_VERSION, V2_SCHEMA_VERSION


class ServiceUnavailable(RuntimeError):
    pass


class JudgeRequestRejected(ValueError):
    code = 'judge_request_rejected'


def guidance_schema(version, schema_override=None):
    """Decoder hints only. The unchanged full wire/parser remains authoritative.

    v0 falls back to an impractical Outlines FSM on string minLength. Its
    xgrammar backend can constrain the same fields without that hint; final
    parse_events still rejects missing/empty evidence and reasons.
    """
    from copy import deepcopy
    schema = deepcopy(schema_override if schema_override is not None else schema_for(version))
    if version == 'math_process_v2':
        def visit(value):
            if isinstance(value, dict):
                if value.get('type') == 'string':value.pop('minLength', None)
                for child in value.values():visit(child)
            elif isinstance(value, list):
                for child in value:visit(child)
        visit(schema)
    return schema


def guidance_backend(version):
    return 'xgrammar:no-fallback' if version == 'math_process_v2' else 'xgrammar'


class RequestBudget:
    """A shared ledger across benchmark arms; only actual send attempts reserve."""
    def __init__(self,limit):
        self.limit=limit;self.count=0;self.lock=threading.Lock()
    def reserve(self):
        with self.lock:
            if self.limit is not None and self.count>=self.limit:return False
            self.count+=1;return True


class JudgeClient:
    def __init__(self, options, expected_fingerprint, tokenizer=None, request_budget=None):
        self.options = options
        self.expected = expected_fingerprint
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(options["tokenizer_path"], local_files_only=True)
        self.tokenizer = tokenizer
        self.budget_lock = threading.Lock()
        self.request_count = 0
        self.targeted_ids = set()
        self.shared_budget = request_budget
        self.api_permits = threading.BoundedSemaphore(options.get('concurrency',1))
        self.api_active = 0
        self.api_peak = 0
        self.api_ledger = []

    def reserve_request(self):
        with self.budget_lock:
            budget = self.options.get('request_budget')
            if budget is not None and self.request_count >= budget:return False
            if self.shared_budget is not None and not self.shared_budget.reserve():return False
            self.request_count += 1
            return True

    def output_diagnostics(self, message, usage):
        """Provider total is authoritative; split retokenization is explicitly estimated."""
        reasoning, final = message.get('reasoning_content'), message.get('content')
        details = usage.get('completion_tokens_details') or {}
        provider_reasoning = details.get('reasoning_tokens')
        def estimate(value):
            if not isinstance(value, str) or not hasattr(self.tokenizer, 'encode'):return None
            return len(self.tokenizer.encode(value, add_special_tokens=False))
        return {'reasoning_output': reasoning, 'output_lengths': {
            'reasoning_characters': len(reasoning) if isinstance(reasoning, str) else 0,
            'final_characters': len(final) if isinstance(final, str) else 0,
            'reasoning_tokens': provider_reasoning if provider_reasoning is not None else estimate(reasoning),
            'reasoning_token_source': 'api.completion_tokens_details.reasoning_tokens' if provider_reasoning is not None else 'local_retokenization_estimate',
            'final_tokens': estimate(final), 'final_token_source': 'local_retokenization_estimate',
            'total_tokens': usage.get('completion_tokens'), 'total_token_source': 'api.usage.completion_tokens',
            'split_counts_are_not_additive': True}}

    def request(self, suffix, payload=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        variable = self.options.get("api_key_env")
        if variable:
            key = os.environ.get(variable)
            if not key:
                raise ServiceUnavailable("configured judge key environment variable is unset")
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.options["base_url"].rstrip("/") + suffix,
                                         data=json.dumps(payload).encode() if payload is not None else None,
                                         headers=headers)
        try:
            opener = (urllib.request.build_opener(urllib.request.ProxyHandler({}))
                      if urlparse(self.options["base_url"]).hostname in ("127.0.0.1", "localhost", "::1")
                      else urllib.request.build_opener())
            with opener.open(request, timeout=timeout or self.options["request_timeout_seconds"]) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429) and exc.code < 500:
                raise JudgeRequestRejected(f'judge request rejected: HTTP {exc.code}') from exc
            raise ServiceUnavailable(f'judge HTTP service failure: {exc.code}') from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServiceUnavailable(f"judge transport failed: {type(exc).__name__}") from exc

    def health(self, deadline=None):
        def get(suffix):
            if deadline is None:
                return self.request(suffix)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceUnavailable('whole-step judge deadline exceeded')
            return self.request(suffix, timeout=min(remaining, self.options['request_timeout_seconds']))
        models = get("/models")
        if self.options["model"] not in [row["id"] for row in models.get("data", [])]:
            raise ServiceUnavailable("judge served model mismatch")
        fingerprint = get("/gad_fingerprint")
        if fingerprint.get("identity_hash") != self.expected:
            raise ServiceUnavailable("judge revision/tokenizer/template/schema/precision fingerprint mismatch")
        return fingerprint

    def annotate_one(self, request_id, messages, deadline, *, schema_override=None, parse_override=None, max_retries=None):
        started = time.monotonic()
        version = self.options.get('rubric_version', 'math_v1')
        thinking = self.options.get('enable_thinking', False)
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=thinking)
        capacity = {'input_tokens': len(tokens), 'reserved_output_tokens': self.options['max_output_tokens'],
                    'max_context_tokens': self.options['max_context_tokens']}
        metadata = {'request_id': request_id, **capacity, 'input_protocol': V2_INPUT_VERSION if version == 'math_process_v2' else None,
                    'schema_version': V2_SCHEMA_VERSION if version == 'math_process_v2' else None,
                    'enable_thinking': thinking, 'reasoning_parser': self.options.get('reasoning_parser')}
        if len(tokens) + self.options['max_output_tokens'] > self.options['max_context_tokens']:
            return request_id, {**metadata, 'error': 'judge_context_exceeded', 'error_detail': capacity,
                                'failure_kind': 'technical', 'cacheable': False, 'attempts': 0, 'usage': {},
                                'elapsed_seconds': time.monotonic() - started, 'api_attempts': []}
        failure, detail, usage, api_attempts, raw = None, None, {}, [], None
        output_diagnostics = {}
        retries = (min(self.options['max_retries'], 1) if version == 'math_process_v2' else self.options['max_retries']) if max_retries is None else max_retries
        attempts = 0
        for attempt in range(retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = 'judge_deadline_exceeded'
                break
            payload = {'model': self.options['model'], 'messages': messages,
                       'temperature': self.options['temperature'], 'seed': self.options.get('seed', 42),
                       'top_p': self.options.get('top_p', 1.), 'top_k': self.options.get('top_k', -1),
                       'min_p': self.options.get('min_p', 0.), 'max_tokens': self.options['max_output_tokens'],
                       'guided_json': guidance_schema(version, schema_override), 'guided_decoding_backend': guidance_backend(version),
                       'chat_template_kwargs': {'enable_thinking': thinking}}
            # The same client-wide permit bounds primary, review and retry sends.
            permit_started=time.monotonic()
            if not self.api_permits.acquire(timeout=max(0,deadline-time.monotonic())):
                failure='judge_deadline_exceeded';break
            remaining=deadline-time.monotonic()
            if remaining<=0 or not self.reserve_request():
                self.api_permits.release()
                failure='judge_deadline_exceeded' if remaining<=0 else 'judge_request_budget_exhausted'
                break
            before = time.monotonic()
            attempts += 1
            with self.budget_lock:
                self.api_active+=1;self.api_peak=max(self.api_peak,self.api_active)
                send_state={'request_id':request_id,'attempt':attempts,'state':'inflight',
                            'permit_wait_seconds':before-permit_started,
                            'timeout_seconds':min(remaining,self.options['request_timeout_seconds']),
                            'timeout_scope':'batch_deadline' if remaining<=self.options['request_timeout_seconds'] else 'single_request',
                            'server_cancellation_confirmed':False}
                self.api_ledger.append(send_state)
            try:
                raw = None
                output_diagnostics = {}
                try:
                    result = self.request('/chat/completions', payload, min(remaining, self.options['request_timeout_seconds']))
                    send_state['state']='response_received'
                except Exception:
                    send_state['state']='client_failed_server_state_unknown'
                    raise
                finally:
                    with self.budget_lock:
                        self.api_active-=1;send_state['api_seconds']=time.monotonic()-before
                    self.api_permits.release()
                api_attempts.append({'attempt': attempts, 'seconds': time.monotonic() - before, 'response': result})
                for key, value in result.get('usage', {}).items():
                    if isinstance(value, (int, float)):usage[key] = usage.get(key, 0) + value
                choice = result['choices'][0]
                raw = choice['message']['content']
                output_diagnostics = self.output_diagnostics(choice['message'], result.get('usage', {}))
                if choice['finish_reason'] == 'length':
                    if thinking and not (isinstance(raw, str) and raw.strip()):
                        raise ValueError('judge_reasoning_budget_exhausted')
                    raise ValueError('judge_output_truncated')
                if thinking and not (isinstance(output_diagnostics['reasoning_output'], str) and output_diagnostics['reasoning_output'].strip()):
                    raise ValueError('judge_thinking_not_observed')
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError('judge_final_content_missing')
                response = None
                if version == 'math_process_v2' and parse_override is None:
                    from .core.process_v2 import response_from_messages
                    response = response_from_messages(messages)
                parsed = parse_override(raw) if parse_override is not None else parse_judge(raw, version, response)
                return request_id, {**metadata, **output_diagnostics, 'parsed': parsed, 'raw_output': raw, 'api_attempts': api_attempts,
                                    'cacheable': True, 'attempts': attempts, 'usage': usage,
                                    'elapsed_seconds': time.monotonic() - started}
            except ServiceUnavailable as exc:
                failure, detail = 'judge_transport_error', str(exc)
                api_attempts.append({'attempt': attempts, 'seconds': time.monotonic() - before, 'transport_error': str(exc)})
            except (ValueError, KeyError, TypeError) as exc:
                failure = getattr(exc, 'code', str(exc))
                detail = str(exc)
                if version == 'math_process_v2':break  # No unchanged format/evidence retries.
        return request_id, {**metadata, **output_diagnostics, 'error': failure, 'error_detail': detail, 'failure_kind': 'technical',
                            'raw_output': raw, 'api_attempts': api_attempts, 'cacheable': False,
                            'attempts': attempts, 'usage': usage, 'elapsed_seconds': time.monotonic() - started}

    def targeted(self, request_id, messages, deadline):
        from .core.semantic_v24 import SCHEMA, parse_review
        with self.budget_lock:
            if request_id in self.targeted_ids:
                return {'request_id':request_id,'error':'targeted_request_already_attempted','attempts':0,'usage':{}}
            self.targeted_ids.add(request_id)
        # One additional request, no transport or format retry.
        return self.annotate_one(request_id, messages, deadline, schema_override=SCHEMA,
                                 parse_override=parse_review, max_retries=0)[1]

    def annotate(self, requests, deadline=None, health_checked=False):
        if not requests:return {}
        deadline = deadline if deadline is not None else time.monotonic() + self.options['step_timeout_seconds']
        results = {}
        try:
            if not health_checked:self.health(deadline=deadline)
            if deadline <= time.monotonic():raise ServiceUnavailable('whole-step judge deadline exceeded')
            executor = ThreadPoolExecutor(max_workers=self.options['concurrency'])
            futures = [executor.submit(self.annotate_one, rid, messages, deadline) for rid, messages in requests.items()]
            try:
                for future in as_completed(futures, timeout=max(0, deadline - time.monotonic())):
                    rid, value = future.result();results[rid] = value
            except (TimeoutError, FuturesTimeoutError) as exc:
                raise ServiceUnavailable('whole-step judge deadline exceeded') from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                for future in futures:
                    if future.done() and not future.cancelled() and future.exception() is None:
                        rid, value = future.result();results[rid] = value
            if results and all(row.get('error') == 'judge_transport_error' for row in results.values()):
                raise ServiceUnavailable('all judge requests failed; no optimizer update allowed')
            if self.options.get('rubric_version') == 'math_process_v2' and results and all('parsed' not in row for row in results.values()):
                raise ValueError('all V2 annotations failed; refusing a silently masked training step')
            return results
        except Exception as exc:
            exc.results = results
            raise
