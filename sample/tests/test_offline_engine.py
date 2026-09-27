#!/usr/bin/env python3
"""CPU tests for the in-process (`vllm-offline`) sampling engine.

The sampling data plane must not depend on the transport, so these tests drive
the real CLI with a **fake engine** (``--engine-factory``) and assert the
properties the offline mode was added for:

* one process, no HTTP endpoint and no `vllm` import on the HTTP path;
* the same writer/resume/failure semantics as the HTTP backend;
* the engine is part of the fingerprint: an index produced by another backend is
  refused *and left untouched* instead of being silently extended;
* throughput knobs (batch size, concurrency, memory) do not change the digest,
  while anything that changes the generated text does;
* the chat template is applied exactly like the HTTP path asks the server to;
* ``--log-file`` really redirects fd 1/2 to a file (an unread terminal can
  otherwise block a detached run with a full pty buffer).

Run::

    python LMOps/sample/tests/test_offline_engine.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE_DIR))

import sample_teacher  # noqa: E402
import teacher_engines  # noqa: E402

PROMPT_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.string())])),
    ]
)

FAKE_ENGINES = '''
"""Fake engines for the offline-engine CLI tests (no GPU, no vLLM)."""
from teacher_engines import TeacherEngine


class FakeEngine(TeacherEngine):
    name = "fake"

    def __init__(self, args, provider):
        self.args = args
        self.provider = provider
        self.calls = 0
        self.closed = False

    def describe(self):
        return {
            "backend": self.name,
            "provider": self.provider,
            "batch_size": self.args.batch_size,
            "concurrency": self.args.concurrency,
            "gpu_memory_utilization": self.args.gpu_memory_utilization,
        }

    def fingerprint_fields(self):
        if self.provider != "vllm_offline":
            return {}
        return {"backend": self.provider, "dtype": self.args.dtype,
                "tensor_parallel_size": self.args.tensor_parallel_size,
                "model_path": self.args.model_path}

    def sample(self, rows, on_result):
        self.calls += 1
        for row in rows:
            if row["uid"].endswith("077"):
                on_result(row, None, "engine exploded for this uid")
                continue
            on_result(row, {
                "answers": [f"reasoning\\nAnswer: {row['uid']}"],
                "finish_reasons": ["stop"],
                "truncated": [False],
                "attempts": 1,
            }, None)

    def close(self):
        self.closed = True


def build(args):
    return FakeEngine(args, "vllm_offline")


def build_http(args):
    return FakeEngine(args, "openai_compatible")
'''


def write_prompts(path: Path, rows: int = 5) -> None:
    payload = [
        {
            "data_source": "math_dapo",
            "prompt": [{"role": "user", "content": f"problem {i}"}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(i), "style": "rule"},
            "extra_info": {"index": f"u{i:04d}"},
        }
        for i in range(rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(payload, schema=PROMPT_SCHEMA), path)


class EngineCliTests(unittest.TestCase):
    """Drive the real CLI with a fake engine; assert data-plane behaviour."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.prompts = self.root / "train.parquet"
        write_prompts(self.prompts, rows=5)
        self.engine_dir = self.root / "engines"
        self.engine_dir.mkdir()
        (self.engine_dir / "fake_engines.py").write_text(FAKE_ENGINES, encoding="utf-8")
        self.out = self.root / "teacher_index.jsonl"
        self.model_dir = self.root / "teacher_weights"
        self.model_dir.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_cli(self, *extra: str, expect: int | None = 0) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.engine_dir)
        env["NO_PROXY"] = "localhost,127.0.0.1"
        env["no_proxy"] = "localhost,127.0.0.1"
        argv = [
            sys.executable, str(SAMPLE_DIR / "sample_teacher.py"),
            "--prompt-parquet", str(self.prompts), "--out", str(self.out),
            "--backend", "vllm-offline", "--model-path", str(self.model_dir),
            "--engine-factory", "fake_engines:build",
            "--m", "1", "--max-tokens", "64", "--batch-size", "2",
            *extra,
        ]
        result = subprocess.run(argv, capture_output=True, text=True, env=env)
        if expect is not None:
            self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
        return result

    # ------------------------------------------------------------------ writer
    def test_offline_run_writes_records_and_resumes(self) -> None:
        result = self.run_cli()
        self.assertIn("engine: fake", result.stdout)
        records = [line for line in self.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(records), 5 + 1, "header + 5 answers")
        header = json.loads(records[0])
        self.assertEqual(header["teacher_fingerprint"]["provider"], "vllm_offline")
        self.assertIn("engine", header["teacher_fingerprint"])
        self.assertEqual(header["notes"]["engine"]["backend"], "fake")
        first = json.loads(records[1])
        self.assertEqual(first["teacher_answers"][0].splitlines()[-1], f"Answer: {first['uid']}")
        self.assertEqual(first["sampling"]["num_answers"], 1)

        # second run: everything already sampled, nothing rewritten
        before = self.out.read_bytes()
        again = self.run_cli()
        self.assertIn("already sampled; 0 to go", again.stdout)
        self.assertEqual(self.out.read_bytes(), before)

    def test_failures_are_recorded_and_exit_nonzero(self) -> None:
        write_prompts(self.prompts, rows=78)  # u0000..u0077; the fake engine fails u0077
        rows = pq.read_table(self.prompts).to_pylist()
        self.assertEqual(rows[-1]["extra_info"]["index"], "u0077")

        result = self.run_cli(expect=1)
        self.assertIn("u0077", result.stdout + result.stderr)
        failures = Path(str(self.out) + ".failures.jsonl")
        self.assertTrue(failures.exists())
        payload = [json.loads(line) for line in failures.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["uid"] for item in payload], ["u0077"])
        self.assertIn("engine exploded", payload[0]["error"])
        summary = json.loads(Path(str(self.out) + ".summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["prompts_written"], 77)
        self.assertEqual(summary["prompts_failed"], 1)
        self.assertEqual(summary["backend"], "vllm-offline")

        allowed = self.run_cli("--allow-incomplete", expect=0)
        self.assertIn("failures ->", allowed.stdout)

    def test_limit_is_honoured_and_progress_line_count_is_bounded(self) -> None:
        result = self.run_cli("--limit", "3", "--progress", "log", "--log-every", "1")
        self.assertIn("3 prompts from 1 file(s)", result.stdout)
        self.assertEqual(result.stdout.count("written "), 3)
        records = [line for line in self.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(records), 4)

        quiet_out = self.root / "quiet.jsonl"
        argv = [
            sys.executable, str(SAMPLE_DIR / "sample_teacher.py"),
            "--prompt-parquet", str(self.prompts), "--out", str(quiet_out),
            "--backend", "vllm-offline", "--model-path", str(self.model_dir),
            "--engine-factory", "fake_engines:build",
            "--m", "1", "--progress", "quiet", "--limit", "5",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.engine_dir)
        quiet = subprocess.run(argv, capture_output=True, text=True, env=env)
        self.assertEqual(quiet.returncode, 0, quiet.stdout + quiet.stderr)
        self.assertEqual(quiet.stdout.count("written "), 0)

    def test_log_file_captures_stdout_and_fingerprint(self) -> None:
        log_path = self.root / "logs" / "sample.log"
        result = self.run_cli("--log-file", str(log_path), "--limit", "1")
        self.assertEqual(result.stdout, "", "stdout must be redirected into the log file")
        text = log_path.read_text(encoding="utf-8")
        self.assertIn("stdout/stderr ->", text)
        self.assertIn("teacher fingerprint digest:", text)
        self.assertIn("summary ->", text)
        records = [line for line in self.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(records), 2)

    # ------------------------------------------------------------- fingerprint
    def test_engine_is_part_of_the_fingerprint_and_old_index_is_refused(self) -> None:
        """An HTTP index must not be extended by the offline engine."""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.engine_dir)
        http_argv = [
            sys.executable, str(SAMPLE_DIR / "sample_teacher.py"),
            "--prompt-parquet", str(self.prompts), "--out", str(self.out),
            "--backend", "vllm-offline", "--model-path", str(self.model_dir),
            "--engine-factory", "fake_engines:build_http",
            "--m", "1", "--limit", "2",
        ]
        first = subprocess.run(http_argv, capture_output=True, text=True, env=env)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        header = json.loads(self.out.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(header["teacher_fingerprint"]["provider"], "openai_compatible")
        self.assertNotIn("engine", header["teacher_fingerprint"])
        before = self.out.read_bytes()

        second = self.run_cli(expect=None)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("fingerprint", (second.stdout + second.stderr).lower())
        self.assertEqual(self.out.read_bytes(), before, "a refused run must not touch the index")

    def test_throughput_knobs_do_not_change_the_digest(self) -> None:
        def digest(*extra: str) -> str:
            result = self.run_cli("--dry-run", *extra)
            for line in result.stdout.splitlines():
                if "teacher fingerprint digest:" in line:
                    return line.rsplit(":", 1)[-1].strip()
            raise AssertionError(f"no digest in output: {result.stdout}")

        base = digest()
        self.assertEqual(base, digest("--batch-size", "7"))          # not part of the payload
        self.assertEqual(base, digest("--concurrency", "3"))
        self.assertEqual(base, digest("--gpu-memory-utilization", "0.5"))
        self.assertNotEqual(base, digest("--m", "2"))
        self.assertNotEqual(base, digest("--temperature", "0.9"))
        self.assertNotEqual(base, digest("--max-tokens", "128"))
        self.assertNotEqual(base, digest("--dtype", "float16"))
        self.assertNotEqual(base, digest("--tensor-parallel-size", "2"))

    def test_usage_errors_exit_two(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.engine_dir)
        openai_missing = subprocess.run(
            [sys.executable, str(SAMPLE_DIR / "sample_teacher.py"),
             "--prompt-parquet", str(self.prompts), "--out", str(self.out), "--backend", "openai"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(openai_missing.returncode, 2)
        self.assertIn("--base-url is required", openai_missing.stdout + openai_missing.stderr)

        offline_missing = subprocess.run(
            [sys.executable, str(SAMPLE_DIR / "sample_teacher.py"),
             "--prompt-parquet", str(self.prompts), "--out", str(self.out), "--backend", "vllm-offline"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(offline_missing.returncode, 2)
        self.assertIn("--model-path is required", offline_missing.stdout + offline_missing.stderr)


class FingerprintFieldTests(unittest.TestCase):
    """Field-level checks that the CLI-level fake engine cannot express."""

    class Args:
        teacher_model = "Qwen3-30B-A3B-Instruct-2507"
        model_path = "/models/Qwen3-30B-A3B-Instruct-2507"
        base_url = None
        temperature = 0.7
        top_p = 0.95
        max_tokens = 8192
        m = 1
        seed = None
        enable_thinking = "false"

    def _engine(self, **overrides):
        kwargs = dict(
            model_path=self.Args.model_path, m=1, temperature=0.7, top_p=0.95, max_tokens=8192,
            tensor_parallel_size=4, max_model_len=12288, gpu_memory_utilization=0.9,
            dtype="bfloat16", enforce_eager=False, batch_size=128,
            chat_template_kwargs={"enable_thinking": False},
        )
        kwargs.update(overrides)
        return teacher_engines.VllmOfflineEngine(**kwargs)

    def _payload(self, engine) -> dict:
        return sample_teacher.build_fingerprint(self.Args(), {}, engine)

    def test_engine_fields_that_change_the_text_are_recorded(self) -> None:
        payload = self._payload(self._engine())
        self.assertEqual(payload["provider"], "vllm_offline")
        engine_fields = payload["engine"]
        for key in ("model_path", "dtype", "tensor_parallel_size", "max_model_len", "chat_template_kwargs"):
            self.assertIn(key, engine_fields)
        # throughput/memory knobs must not be able to invalidate a sampling
        for key in ("batch_size", "gpu_memory_utilization", "enforce_eager", "gpus"):
            self.assertNotIn(key, engine_fields)

    def test_engine_fields_change_the_digest(self) -> None:
        from task_template import hash_teacher_fingerprint

        base = self._payload(self._engine())
        variants = [
            self._engine(dtype="float16"),
            self._engine(tensor_parallel_size=2),
            self._engine(max_model_len=8192),
            self._engine(chat_template_kwargs={}),
            self._engine(model_path="/models/other"),
        ]
        base_digest = hash_teacher_fingerprint(base)
        for variant in variants:
            self.assertNotEqual(base_digest, hash_teacher_fingerprint(self._payload(variant)))

    def test_http_payload_is_unchanged_by_the_engine_feature(self) -> None:
        """Backward compatibility: HTTP digests must stay bit-identical."""
        http_payload = sample_teacher.build_fingerprint(self.Args2(), {}, teacher_engines.OpenAIEngine(
            base_url="http://127.0.0.1:18080/v1", api_key="EMPTY", teacher_model="m",
            m=4, temperature=0.7, top_p=0.95, max_tokens=8192,
            concurrency=16, timeout=600.0, retries=3,
        ))
        self.assertNotIn("engine", http_payload)
        self.assertEqual(http_payload["provider"], "openai_compatible")
        expected = {
            "provider": "openai_compatible",
            "model": "m",
            "base_url": "http://127.0.0.1:18080/v1",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 8192,
            "num_samples": 4,
            "seed": None,
            "extra_body": None,
            "prompt_template_version": http_payload["prompt_template_version"],
        }
        self.assertEqual(http_payload, expected)

    class Args2(Args):
        teacher_model = "m"
        base_url = "http://127.0.0.1:18080/v1"
        m = 4


class LazyImportTests(unittest.TestCase):
    def test_vllm_is_not_imported_by_the_http_path(self) -> None:
        """The data-contract suite must not need vLLM installed."""
        code = (
            "import sys;"
            f"sys.path.insert(0, {str(SAMPLE_DIR)!r});"
            "import sample_teacher, teacher_engines;"
            "print('VLLM_IMPORTED' if 'vllm' in sys.modules else 'clean')"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clean", result.stdout)


class ChatTemplateTests(unittest.TestCase):
    """The offline path must render the same prompt the HTTP path asked for."""

    TOKENIZER = Path("/mnt/jinbo/RLRM/model/Qwen/Qwen3-1.7B")

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.TOKENIZER.is_dir():
            raise unittest.SkipTest(f"local tokenizer not available: {cls.TOKENIZER}")
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(cls.TOKENIZER, trust_remote_code=True)

    def test_thinking_switch_reaches_the_rendered_prompt(self) -> None:
        messages = [{"role": "user", "content": "What is 2+2?"}]
        plain, applied_plain = teacher_engines.render_prompt(self.tokenizer, messages, {})
        off, applied_off = teacher_engines.render_prompt(self.tokenizer, messages, {"enable_thinking": False})
        on, applied_on = teacher_engines.render_prompt(self.tokenizer, messages, {"enable_thinking": True})
        self.assertFalse(applied_plain, "no kwargs were forwarded, so nothing was 'applied'")
        self.assertTrue(applied_off and applied_on)
        # Qwen3's template defaults to thinking enabled, so the explicit switch is
        # what changes the prompt: the disabled mode appends an EMPTY reasoning
        # block right after the assistant turn.
        self.assertEqual(plain, on)
        self.assertTrue(off.startswith(plain), repr((plain[-60:], off[-60:])))
        self.assertGreater(len(off), len(plain))
        for text in (plain, off):
            self.assertIn("What is 2+2?", text)

    def test_a_template_that_rejects_the_kwarg_is_reported_not_silently_ignored(self) -> None:
        class StrictTokenizer:
            def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
                if kwargs:
                    raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
                return "<prompt>"

        text, applied = teacher_engines.render_prompt(
            StrictTokenizer(), [{"role": "user", "content": "x"}], {"enable_thinking": False}
        )
        self.assertFalse(applied)
        self.assertEqual(text, "<prompt>")

    def test_none_valued_kwargs_are_not_forwarded(self) -> None:
        class RecordingTokenizer:
            def __init__(self):
                self.seen: dict = {}

            def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
                self.seen = kwargs
                return "<prompt>"

        tokenizer = RecordingTokenizer()
        teacher_engines.render_prompt(tokenizer, [{"role": "user", "content": "x"}],
                                     {"enable_thinking": None, "other": 1})
        self.assertEqual(tokenizer.seen, {"other": 1})


class VllmOfflineEngineTests(unittest.TestCase):
    """Exercise the engine internals with a fake ``vllm`` module (no GPU, no weights).

    This covers what the CLI-level fake engine cannot: chunking, the exact
    ``SamplingParams`` we ask vLLM for, the chat-template render, mapping of
    ``RequestOutput`` to the writer payload, and per-chunk failure handling.
    """

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kwargs):
            assert add_generation_prompt and tokenize is False
            self.kwargs = kwargs
            return "PROMPT:" + messages[0]["content"]

    class FakeOutput:
        def __init__(self, text, finish_reason):
            self.text = text
            self.finish_reason = finish_reason

    class FakeRequestOutput:
        def __init__(self, outputs):
            self.outputs = outputs

    class FakeLLM:
        """Records construction kwargs and returns scripted generations."""

        last = None
        sampling_params: dict = {}
        scripted_queue: list = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[list[str]] = []
            # the engine constructs the LLM lazily inside sample(), so scripts are
            # handed over through a class-level queue
            self.scripted = list(type(self).scripted_queue)
            type(self).scripted_queue = []
            self.tokenizer = VllmOfflineEngineTests.FakeTokenizer()
            type(self).last = self

        def get_tokenizer(self):
            return self.tokenizer

        def generate(self, prompts, sampling_params, use_tqdm=False):
            self.calls.append(list(prompts))
            assert use_tqdm is False
            type(self).sampling_params = sampling_params
            if self.scripted:
                return self.scripted.pop(0)
            return [
                VllmOfflineEngineTests.FakeRequestOutput(
                    [VllmOfflineEngineTests.FakeOutput(f"reasoning\nAnswer: {prompt.split(':')[1]}", "stop")]
                )
                for prompt in prompts
            ]

    def setUp(self) -> None:
        self._saved = sys.modules.get("vllm")
        fake_module = type(sys)("vllm")
        fake_module.__version__ = "0.8.5-fake"
        fake_module.LLM = self.FakeLLM
        fake_module.SamplingParams = lambda **kwargs: {"params": kwargs}
        sys.modules["vllm"] = fake_module
        self.FakeLLM.scripted_queue = []
        self.FakeLLM.last = None
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        if self._saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = self._saved

    def _rows(self, count: int) -> list[dict]:
        return [
            {"uid": f"u{i}", "raw_prompt_hash": f"h{i}", "prompt": [{"role": "user", "content": f"q{i}"}]}
            for i in range(count)
        ]

    def _engine(self, **overrides):
        kwargs = dict(
            model_path=str(SAMPLE_DIR),  # must exist; the fake loader never reads it
            m=2, temperature=0.7, top_p=0.95, max_tokens=8192,
            tensor_parallel_size=4, max_model_len=12288, gpu_memory_utilization=0.9,
            dtype="bfloat16", batch_size=2, chat_template_kwargs={"enable_thinking": False},
        )
        kwargs.update(overrides)
        return teacher_engines.VllmOfflineEngine(**kwargs)

    def test_build_passes_the_expected_llm_kwargs(self) -> None:
        engine = self._engine()
        results: list = []
        engine.sample(self._rows(1), lambda row, payload, error: results.append((row, payload, error)))
        built = self.FakeLLM.last
        self.assertEqual(built.kwargs["model"], str(SAMPLE_DIR))
        self.assertEqual(built.kwargs["tensor_parallel_size"], 4)
        self.assertEqual(built.kwargs["max_model_len"], 12288)
        self.assertEqual(built.kwargs["gpu_memory_utilization"], 0.9)
        self.assertEqual(built.kwargs["dtype"], "bfloat16")
        self.assertNotIn("seed", built.kwargs)
        self.assertEqual(self.FakeLLM.sampling_params, {"params": {"n": 2, "temperature": 0.7,
                                                                  "top_p": 0.95, "max_tokens": 8192}})
        self.assertEqual(engine.vllm_version, "0.8.5-fake")
        self.assertTrue(engine.chat_template_kwargs_applied)
        self.assertIn("chat_template_kwargs", engine.fingerprint_fields())

    def test_prompts_are_rendered_chunked_and_mapped(self) -> None:
        engine = self._engine(batch_size=2)
        seen: list[tuple] = []
        engine.sample(self._rows(5), lambda row, payload, error: seen.append((row["uid"], payload, error)))
        self.assertEqual([call for call in self.FakeLLM.last.calls],
                         [["PROMPT:q0", "PROMPT:q1"], ["PROMPT:q2", "PROMPT:q3"], ["PROMPT:q4"]])
        self.assertEqual([uid for uid, _, _ in seen], ["u0", "u1", "u2", "u3", "u4"])
        for uid, payload, error in seen:
            self.assertIsNone(error)
            self.assertEqual(payload["answers"], [f"reasoning\nAnswer: q{uid[1:]}"])
            self.assertEqual(payload["finish_reasons"], ["stop"])
            self.assertEqual(payload["truncated"], [False])
            self.assertEqual(payload["attempts"], 1)

    def test_truncation_and_empty_answers_are_reported(self) -> None:
        engine = self._engine(batch_size=3)
        # one RequestOutput per prompt, each carrying its own candidate list
        self.FakeLLM.scripted_queue.append([
            self.FakeRequestOutput([self.FakeOutput("cut off", "length")]),
            self.FakeRequestOutput([self.FakeOutput("   ", "stop")]),   # empty -> dropped
            self.FakeRequestOutput([self.FakeOutput("ok\nAnswer: 3", "stop")]),
        ])
        seen: list[tuple] = []
        engine.sample(self._rows(3), lambda row, payload, error: seen.append((row["uid"], payload, error)))
        self.assertEqual(seen[0][1]["truncated"], [True])
        self.assertEqual(seen[2][1]["answers"], ["ok\nAnswer: 3"])
        self.assertIsNone(seen[1][1])
        self.assertIn("no non-empty answers", seen[1][2])

    def test_a_failing_chunk_only_fails_that_chunk(self) -> None:
        class Exploding(VllmOfflineEngineTests.FakeLLM):
            def generate(self, prompts, sampling_params, use_tqdm=False):
                self.calls.append(list(prompts))
                if len(self.calls) == 1:
                    raise RuntimeError("CUDA error: out of memory")
                return super().generate(prompts, sampling_params, use_tqdm)

        module = sys.modules["vllm"]
        module.LLM = Exploding
        engine = self._engine(batch_size=2)
        seen: list[tuple] = []
        engine.sample(self._rows(4), lambda row, payload, error: seen.append((row["uid"], payload, error)))
        self.assertIsNone(seen[0][1])
        self.assertIn("out of memory", seen[0][2])
        self.assertIsNone(seen[1][1])
        self.assertIsNotNone(seen[2][1], "the second chunk must still be sampled")
        self.assertIsNotNone(seen[3][1])

    def test_a_short_output_list_fails_the_chunk_instead_of_misassigning(self) -> None:
        engine = self._engine(batch_size=2)
        self.FakeLLM.scripted_queue.append([
            self.FakeRequestOutput([self.FakeOutput("only one", "stop")])  # 1 output for 2 prompts
        ])
        seen: list[tuple] = []
        engine.sample(self._rows(2), lambda row, payload, error: seen.append((row["uid"], payload, error)))
        self.assertTrue(all(payload is None for _, payload, _ in seen))
        self.assertIn("engine returned 1 outputs for 2 prompts", seen[0][2])

    def test_missing_weights_directory_is_a_clear_error(self) -> None:
        engine = self._engine(model_path="/definitely/not/here")
        with self.assertRaises(RuntimeError) as ctx:
            engine.sample(self._rows(1), lambda *args: None)
        self.assertIn("not a directory", str(ctx.exception))


class VllmImportTests(unittest.TestCase):
    def test_missing_vllm_gives_an_actionable_error(self) -> None:
        engine = teacher_engines.VllmOfflineEngine(
            model_path=str(SAMPLE_DIR), m=1, temperature=0.7, top_p=0.95, max_tokens=16,
        )
        saved = sys.modules.pop("vllm", None)
        self.addCleanup(lambda: sys.modules.__setitem__("vllm", saved) if saved else None)
        blocked = type(sys)("blocked")
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "vllm":
                raise ImportError("No module named 'vllm'")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = fake_import
        try:
            with self.assertRaises(RuntimeError) as ctx:
                engine.sample([{"uid": "u", "raw_prompt_hash": "h", "prompt": []}], lambda *args: None)
        finally:
            builtins.__import__ = real_import
        self.assertIn("vllm", str(ctx.exception))
        self.assertIn("vllm-offline", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
