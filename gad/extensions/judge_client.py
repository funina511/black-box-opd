"""Driver-only OpenAI-compatible client for verified vLLM 0.8.5 guided JSON."""
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from .core.contracts import JUDGE_SCHEMA, parse_judge, digest


class ServiceUnavailable(RuntimeError):
    pass


class JudgeClient:
    def __init__(self, options, expected_fingerprint, tokenizer=None):
        self.options = options
        self.expected = expected_fingerprint
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(options["tokenizer_path"], local_files_only=True)
        self.tokenizer = tokenizer

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

    def annotate_one(self, request_id, messages, deadline):
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
        if len(tokens) + self.options["max_output_tokens"] > self.options["max_context_tokens"]:
            return request_id, {"error": "judge_context_exceeded", "cacheable": False, "attempts": 0}
        failure = None
        for attempt in range(self.options["max_retries"] + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceUnavailable("whole-step judge deadline exceeded")
            payload = {"model": self.options["model"], "messages": messages,
                       "temperature": self.options["temperature"], "seed": self.options.get("seed", 42),
                       "top_p": 1., "top_k": -1,
                       "max_tokens": self.options["max_output_tokens"],
                       "guided_json": JUDGE_SCHEMA, "guided_decoding_backend": "xgrammar",
                       "chat_template_kwargs": {"enable_thinking": False}}
            try:
                raw = None
                result = self.request("/chat/completions", payload,
                                      min(remaining, self.options["request_timeout_seconds"]))
                choice = result["choices"][0]
                raw = choice["message"]["content"]
                if choice["finish_reason"] == "length":
                    raise ValueError("judge_output_truncated")
                parsed = parse_judge(raw)
                return request_id, {"parsed": parsed, "cacheable": True, "attempts": attempt + 1,
                                    "usage": result.get("usage", {})}
            except ServiceUnavailable:
                failure = "judge_transport_error"
            except (ValueError, KeyError, TypeError) as exc:
                failure = str(exc)
                raw = locals().get("raw")
        return request_id, {"error": failure, "raw_output": locals().get("raw"),
                            "cacheable": False, "attempts": self.options["max_retries"] + 1}

    def annotate(self, requests, deadline=None, health_checked=False):
        if not requests:
            return {}
        deadline = deadline if deadline is not None else time.monotonic() + self.options['step_timeout_seconds']
        if not health_checked:
            self.health(deadline=deadline)
        if deadline <= time.monotonic():
            raise ServiceUnavailable('whole-step judge deadline exceeded')
        executor = ThreadPoolExecutor(max_workers=self.options["concurrency"])
        futures = [executor.submit(self.annotate_one, rid, messages, deadline) for rid, messages in requests.items()]
        results = {}
        try:
            for future in as_completed(futures, timeout=max(0, deadline - time.monotonic())):
                rid, value = future.result()
                results[rid] = value
        except TimeoutError as exc:
            raise ServiceUnavailable("whole-step judge deadline exceeded") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        if results and all(row.get("error") == "judge_transport_error" for row in results.values()):
            raise ServiceUnavailable("all judge requests failed; no optimizer update allowed")
        return results
