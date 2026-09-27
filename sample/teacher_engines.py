#!/usr/bin/env python3
"""Teacher sampling engines: HTTP (OpenAI-compatible) and in-process vLLM.

The sampler's data plane (index schema, resume, flushing, failure list) does not
care where the teacher text comes from, so the transport lives behind one small
interface::

    engine.provider            -> str    # recorded in the fingerprint
    engine.describe()          -> dict   # engine facts for the header/summary
    engine.fingerprint_fields()-> dict   # fields that change the sampling result
    engine.sample(rows, on_result)       # on_result(row, payload|None, error|None)

``payload`` is exactly what the writer expects::

    {"answers": [...], "finish_reasons": [...], "truncated": [...], "attempts": int}

Two engines are provided:

``OpenAIEngine``
    Today's behaviour: HTTP requests to an OpenAI-compatible server
    (``serve_teacher.sh``) with a thread pool, per-row retries and timeouts.

``VllmOfflineEngine``
    The single-process mode: load the teacher weights with ``vllm.LLM`` and
    generate directly, chunk by chunk. No server, no port, no HTTP, nothing to
    keep alive next to the sampler - which also removes the failure mode where a
    detached terminal stops draining the server's stdout and the engine blocks in
    ``write(2)``.

``vllm`` is imported lazily: the CPU-only data-contract tests and the HTTP path
never need it installed.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Row payloads are dicts from the prompt parquet; the writer owns the schema.
Row = Mapping[str, Any]
ResultFn = Callable[[Row, dict | None, str | None], None]


class TeacherEngine:
    """Interface every sampling backend implements."""

    provider = "unknown"
    name = "unknown"

    def describe(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def fingerprint_fields(self) -> dict:
        """Fields that must match for two samplings to be comparable."""
        return {}

    def sample(self, rows: Sequence[Row], on_result: ResultFn) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:
        """Release GPU memory / connections if the engine owns any."""


class OpenAIEngine(TeacherEngine):
    """Sample through an OpenAI-compatible ``/chat/completions`` endpoint."""

    provider = "openai_compatible"
    name = "openai"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        teacher_model: str,
        m: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        concurrency: int,
        timeout: float,
        retries: int,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.teacher_model = teacher_model
        self.m = m
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.concurrency = max(1, concurrency)
        self.timeout = timeout
        self.retries = max(1, retries)
        self.extra_body = dict(extra_body) if extra_body else {}

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "concurrency": self.concurrency,
            "timeout": self.timeout,
            "retries": self.retries,
            "extra_body": self.extra_body or None,
        }

    def fingerprint_fields(self) -> dict:
        # The HTTP payload already records base_url/extra_body, and concurrency is
        # irrelevant to the result: keeping this empty means indexes produced by
        # earlier versions of this sampler stay resumable.
        return {}

    def _sample_one(self, client, row: Row) -> dict:
        request = {
            "model": self.teacher_model,
            "messages": row["prompt"],
            "n": self.m,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            request["extra_body"] = self.extra_body
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                response = client.chat.completions.create(**request)
                answers, finish_reasons, truncated = [], [], []
                for choice in response.choices:
                    content = (choice.message.content or "").strip() if choice.message else ""
                    if not content:
                        continue
                    answers.append(content)
                    reason = getattr(choice, "finish_reason", None)
                    finish_reasons.append(reason)
                    truncated.append(reason == "length")
                if not answers:
                    raise RuntimeError("teacher returned no non-empty answers")
                return {
                    "answers": answers,
                    "finish_reasons": finish_reasons,
                    "truncated": truncated,
                    "attempts": attempt,
                }
            except Exception as exc:  # noqa: BLE001 - network and server errors alike
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(min(30.0, 2.0 ** attempt) + random.random())
        raise RuntimeError(last_error or "unknown error")

    def sample(self, rows: Sequence[Row], on_result: ResultFn) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError(
                "the 'openai' package is required for --backend openai (see requirements.txt)"
            ) from exc

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._sample_one, client, row): row for row in rows}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    on_result(row, future.result(), None)
                except Exception as exc:  # noqa: BLE001
                    on_result(row, None, str(exc))


def render_prompt(tokenizer, messages, chat_template_kwargs: Mapping[str, Any]) -> tuple[str, bool]:
    """Chat-template render, mirroring what the HTTP path asks the server to do.

    Returns ``(text, kwargs_applied)``. A template that rejects the extra keyword
    (some tokenizers are strict) falls back to the plain call and the caller
    records that the switch was NOT applied instead of pretending it was.
    """
    kwargs = {k: v for k, v in (chat_template_kwargs or {}).items() if v is not None}
    if kwargs:
        try:
            text = tokenizer.apply_chat_template(
                list(messages), add_generation_prompt=True, tokenize=False, **kwargs
            )
            return text, True
        except Exception as exc:  # noqa: BLE001 - fall back, but say so
            sys.stderr.write(
                f"[sample_teacher] WARNING: chat template rejected {sorted(kwargs)} ({type(exc).__name__}: {exc}); "
                "rendering without it\n"
            )
    text = tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=False)
    return text, False


class VllmOfflineEngine(TeacherEngine):
    """Load the teacher in-process with ``vllm.LLM`` and generate directly."""

    provider = "vllm_offline"
    name = "vllm_offline"

    def __init__(
        self,
        *,
        model_path: str,
        m: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        tensor_parallel_size: int = 1,
        max_model_len: int = 12288,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        batch_size: int = 128,
        seed: int | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        trust_remote_code: bool = True,
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        # absolute so the fingerprint is stable regardless of the working directory
        self.model_path = str(Path(model_path).expanduser().resolve()) if model_path else str(model_path)
        self.m = m
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.max_model_len = int(max_model_len)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.dtype = dtype
        self.enforce_eager = bool(enforce_eager)
        self.batch_size = max(1, int(batch_size))
        self.seed = seed
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.trust_remote_code = bool(trust_remote_code)
        self.engine_kwargs = dict(engine_kwargs or {})
        self.chat_template_kwargs_applied: bool | None = None
        self.vllm_version: str | None = None
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        if self._llm is not None:
            return
        if not Path(self.model_path).is_dir():
            raise RuntimeError(f"teacher model path is not a directory: {self.model_path}")
        try:
            import vllm  # noqa: PLC0415 - deliberately lazy: only the offline engine needs vLLM
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - dependency check
            raise RuntimeError(
                "the 'vllm' package is required for --backend vllm-offline; it ships with the training "
                "environment (see LMOps/sample/requirements.txt)"
            ) from exc

        self.vllm_version = getattr(vllm, "__version__", "unknown")
        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "tokenizer": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "enforce_eager": self.enforce_eager,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.seed is not None:
            kwargs["seed"] = int(self.seed)
        kwargs.update(self.engine_kwargs)
        sys.stderr.write(
            f"[sample_teacher] loading the teacher in-process: {self.model_path} "
            f"(tp={self.tensor_parallel_size}, max_model_len={self.max_model_len}, "
            f"dtype={self.dtype}, vllm={self.vllm_version})\n"
        )
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()
        params: dict[str, Any] = {
            "n": self.m,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            params["seed"] = int(self.seed)
        self._sampling_params = SamplingParams(**params)
        probe, applied = render_prompt(self._tokenizer, [{"role": "user", "content": "ping"}],
                                        self.chat_template_kwargs)
        if not probe:
            raise RuntimeError("chat template rendered an empty prompt")
        self.chat_template_kwargs_applied = applied

    # ------------------------------------------------------------- interface
    def describe(self) -> dict:
        return {
            "backend": self.name,
            "provider": self.provider,
            "model_path": self.model_path,
            "vllm_version": self.vllm_version,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "batch_size": self.batch_size,
            "chat_template_kwargs": self.chat_template_kwargs or None,
            "chat_template_kwargs_applied": self.chat_template_kwargs_applied,
            "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }

    def fingerprint_fields(self) -> dict:
        # Anything that can change the generated text. batch_size/concurrency and
        # the memory knobs are deliberately absent: they only change throughput.
        return {
            "backend": self.provider,
            "model_path": self.model_path,
            "vllm_version": self.vllm_version,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "max_model_len": self.max_model_len,
            "chat_template_kwargs": self.chat_template_kwargs or None,
        }

    def sample(self, rows: Sequence[Row], on_result: ResultFn) -> None:
        self._build()
        for start in range(0, len(rows), self.batch_size):
            chunk = list(rows[start:start + self.batch_size])
            prompts = []
            for row in chunk:
                text, applied = render_prompt(self._tokenizer, row["prompt"], self.chat_template_kwargs)
                self.chat_template_kwargs_applied = applied
                prompts.append(text)
            try:
                outputs = self._llm.generate(prompts, self._sampling_params, use_tqdm=False)
            except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the run
                message = f"{type(exc).__name__}: {exc}"
                for row in chunk:
                    on_result(row, None, message)
                continue
            if len(outputs) != len(chunk):
                message = f"engine returned {len(outputs)} outputs for {len(chunk)} prompts"
                for row in chunk:
                    on_result(row, None, message)
                continue
            for row, output in zip(chunk, outputs):
                answers, finish_reasons, truncated = [], [], []
                for candidate in getattr(output, "outputs", []) or []:
                    content = (getattr(candidate, "text", "") or "").strip()
                    if not content:
                        continue
                    reason = getattr(candidate, "finish_reason", None)
                    answers.append(content)
                    finish_reasons.append(reason)
                    truncated.append(reason == "length")
                if not answers:
                    on_result(row, None, "teacher returned no non-empty answers")
                    continue
                on_result(row, {
                    "answers": answers,
                    "finish_reasons": finish_reasons,
                    "truncated": truncated,
                    "attempts": 1,
                }, None)

    def close(self) -> None:
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None


def engine_from_args(args) -> TeacherEngine:
    """Build the engine selected by ``--backend`` (CLI contract)."""
    backend = getattr(args, "backend", "openai")
    chat_template_kwargs: dict[str, Any] = {}
    if getattr(args, "enable_thinking", "unset") != "unset":
        chat_template_kwargs["enable_thinking"] = args.enable_thinking == "true"
    if backend == "vllm-offline":
        engine_kwargs = {}
        if getattr(args, "engine_kwargs_json", None):
            import json

            engine_kwargs = json.loads(args.engine_kwargs_json)
        return VllmOfflineEngine(
            model_path=args.model_path,
            m=args.m,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype=args.dtype,
            enforce_eager=args.enforce_eager,
            batch_size=args.batch_size,
            seed=args.seed,
            chat_template_kwargs=chat_template_kwargs,
            engine_kwargs=engine_kwargs,
        )
    return OpenAIEngine(
        base_url=args.base_url,
        api_key=args.api_key,
        teacher_model=args.teacher_model,
        m=args.m,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
        extra_body=getattr(args, "extra_body", None),
    )


def load_engine_factory(spec: str):
    """Import ``module:callable`` and return it (testing / custom backends)."""
    import importlib

    if ":" not in spec:
        raise ValueError(f"expected module:callable, got {spec!r}")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if factory is None:
        raise ValueError(f"{module_name} has no attribute {attr!r}")
    return factory


__all__ = [
    "OpenAIEngine",
    "render_prompt",
    "TeacherEngine",
    "VllmOfflineEngine",
    "engine_from_args",
    "load_engine_factory",
]
