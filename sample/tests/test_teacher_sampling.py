#!/usr/bin/env python3
"""End-to-end tests for the teacher sampler against a local mock endpoint.

`sample_teacher.py` is the only component that talks to an external service, so
its failure paths are exactly the ones that are easy to get wrong. These tests
run a stdlib HTTP server that speaks the OpenAI chat-completions protocol and
check the guarantees from plan section 5:

* a normal run writes one header + one record per prompt, and a re-run resumes
  instead of re-sampling (the file must not change at all);
* a changed sampling fingerprint aborts the run and leaves the existing index
  untouched;
* prompts that fail on the server side produce a failures file and a non-zero
  exit code, unless ``--allow-incomplete`` is passed explicitly;
* a prompt that already contains the answer contract (``Answer: <ground truth>``)
  is refused before anything is sent to the teacher;
* truncated answers are recorded per answer, not hidden.

Run::

    python LMOps/sample/tests/test_teacher_sampling.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE_DIR))

from task_template import hash_raw_prompt  # noqa: E402
from teacher_index import TeacherIndex  # noqa: E402

PROMPT_SCHEMA = None  # built lazily in write_prompts


class MockTeacherHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the test output
        return

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        text = "\n".join(str(m.get("content", "")) for m in messages)
        n = int(payload.get("n") or 1)

        if "FAILME" in text:
            body = json.dumps({"error": {"message": "synthetic server failure"}}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        finish = "length" if "TRUNCATE" in text else "stop"
        choices = [
            {
                "index": i,
                "message": {"role": "assistant", "content": f"mock answer {i} for: {text[:24]}"},
                "finish_reason": finish,
            }
            for i in range(n)
        ]
        body = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", "mock"),
                "choices": choices,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockTeacher:
    def __enter__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MockTeacherHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        # Build the URL from the *bound port* rather than the address tuple, which
        # may be IPv6-shaped depending on the resolver.
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def write_prompts(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
            ("extra_info", pa.struct([("index", pa.string())])),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def prompt_row(uid: str, text: str, ground_truth: str = "42") -> dict:
    return {
        "prompt": [{"role": "user", "content": text}],
        "reward_model": {"ground_truth": ground_truth, "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": uid},
    }


def run_sampler(*args: str) -> subprocess.CompletedProcess:
    # The sandbox exports a NO_PROXY value that some httpx versions cannot parse
    # ("::1,[::1]"), which would break any OpenAI client construction. Point the
    # child at the loopback mock with a clean, proxy-free environment.
    env = dict(os.environ)
    env.update(
        {
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
            "http_proxy": "",
            "https_proxy": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
        }
    )
    return subprocess.run(
        [sys.executable, str(SAMPLE_DIR / "sample_teacher.py"), *args],
        capture_output=True, text=True, env=env,
    )


class TeacherSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.prompts = self.root / "prompts.parquet"
        write_prompts(
            self.prompts,
            [
                prompt_row("u1", "What is 2+2?"),
                prompt_row("u2", "What is 3+4?"),
                prompt_row("u3", "What is 5+5?"),
            ],
        )
        self.index = self.root / "teacher_index.jsonl"
        self.teacher = MockTeacher()
        self.teacher.__enter__()

    def tearDown(self) -> None:
        self.teacher.__exit__(None, None, None)
        self._tmp.cleanup()

    def base_args(self, *extra: str) -> list[str]:
        return [
            "--prompt-parquet", str(self.prompts),
            "--out", str(self.index),
            "--base-url", self.teacher.base_url,
            "--teacher-model", "mock-teacher",
            "--m", "2",
            "--temperature", "0.7",
            "--retries", "1",
            "--concurrency", "4",
            *extra,
        ]

    # ------------------------------------------------------------------ tests
    def test_happy_path_writes_records_and_resumes_without_resampling(self) -> None:
        result = run_sampler(*self.base_args())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        first_bytes = self.index.read_bytes()
        index = TeacherIndex.load(self.index)
        self.assertEqual(len(index), 3)
        texts = {"u1": "What is 2+2?", "u2": "What is 3+4?", "u3": "What is 5+5?"}
        for uid, text in texts.items():
            record = index.require(uid=uid, raw_prompt=[{"role": "user", "content": text}])
            self.assertEqual(len(record.teacher_answers), 2)
            self.assertEqual(record.sampling["truncated"], [False, False])

        again = run_sampler(*self.base_args())
        self.assertEqual(again.returncode, 0, again.stdout + again.stderr)
        self.assertIn("nothing to do", again.stdout)
        self.assertEqual(self.index.read_bytes(), first_bytes, "a resume must not rewrite the index")

    def test_changed_fingerprint_aborts_and_leaves_the_file_untouched(self) -> None:
        result = run_sampler(*self.base_args())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        before = self.index.read_bytes()

        result = run_sampler(*self.base_args("--teacher-model", "a-different-teacher"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fingerprint changed", result.stdout + result.stderr)
        self.assertEqual(self.index.read_bytes(), before, "the existing index must not be modified")

    def test_server_failure_reports_nonzero_and_records_failures(self) -> None:
        write_prompts(
            self.prompts,
            [prompt_row("u1", "What is 2+2?"), prompt_row("bad", "FAILME please fail")],
        )
        result = run_sampler(*self.base_args())
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(Path(str(self.index) + ".failures.jsonl").exists())
        failures = [
            json.loads(line)
            for line in Path(str(self.index) + ".failures.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual([item["uid"] for item in failures], ["bad"])

        # an explicit opt-in turns the same situation into a successful (but recorded) run
        result = run_sampler(*self.base_args("--allow-incomplete"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(Path(str(self.index) + ".summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["prompts_failed"], 1)

    def test_ground_truth_in_prompt_is_refused_before_sampling(self) -> None:
        write_prompts(self.prompts, [prompt_row("u1", "What is 6+1? Answer: 7", ground_truth="7")])
        result = run_sampler(*self.base_args())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reference answer", result.stdout + result.stderr)
        self.assertFalse(self.index.exists(), "nothing may be written when the check fails")

    def test_truncated_answers_are_recorded(self) -> None:
        write_prompts(self.prompts, [prompt_row("u1", "TRUNCATE this one")])
        result = run_sampler(*self.base_args())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        index = TeacherIndex.load(self.index)
        record = index.require(uid="u1", raw_prompt=[{"role": "user", "content": "TRUNCATE this one"}])
        self.assertEqual(record.sampling["truncated"], [True, True])
        self.assertEqual(record.sampling["finish_reasons"], ["length", "length"])

    def test_same_uid_with_a_different_prompt_is_refused(self) -> None:
        result = run_sampler(*self.base_args())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        # simulate the prompt file changing under an existing index
        write_prompts(self.prompts, [prompt_row("u1", "A completely different question?")])
        result = run_sampler(*self.base_args())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("different prompt", result.stdout + result.stderr)

    def test_dry_run_does_not_touch_the_network_or_the_index(self) -> None:
        result = run_sampler(*self.base_args("--dry-run"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("dry-run", result.stdout)
        self.assertFalse(self.index.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
