#!/usr/bin/env python3
"""Train/dev split and CLI stage-selection tests (CPU only, no teacher endpoint).

Covers two reported defects:

* **split leakage** - the split used to draw *uids* at random, so the 495
  DAPO questions that appear under two ``extra_info.index`` values could land on
  opposite sides of the boundary (the real artifact leaked 27 dev questions into
  training, and ``validate_data.py`` then refused the data). ``prepare_prompts``
  now draws whole prompt groups;
* **``--stage convert``** - the documented stage name did not exist (the script
  runs ``convert_train`` and ``convert_dev``), so the run silently completed
  without converting anything.

Run::

    python LMOps/sample/tests/test_split_no_leak.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SAMPLE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = SAMPLE_DIR / "scripts"
sys.path.insert(0, str(SAMPLE_DIR))

import prepare_prompts  # noqa: E402
from task_template import (  # noqa: E402
    build_teacher_fingerprint_payload,
    hash_raw_prompt,
    hash_teacher_fingerprint,
)

FINGERPRINT = build_teacher_fingerprint_payload(
    provider="openai_compatible",
    model="teacher-test",
    base_url="http://127.0.0.1:1/v1",
    temperature=0.7,
    top_p=0.95,
    max_output_tokens=8192,
    num_samples=2,
    seed=None,
)
DIGEST = hash_teacher_fingerprint(FINGERPRINT)


def write_teacher_index(path: Path, rows: list[dict]) -> None:
    """Minimal but schema-valid teacher index for the converted GAD parquets."""
    from teacher_index import GAD_INDEX_SCHEMA_VERSION, header_record

    lines = [json.dumps(header_record(fingerprint=FINGERPRINT, created_by="test"))]
    for row in rows:
        prompt = row["prompt"]
        lines.append(
            json.dumps(
                {
                    "schema_version": GAD_INDEX_SCHEMA_VERSION,
                    "uid": row["extra_info"]["index"],
                    "raw_prompt_hash": hash_raw_prompt(prompt),
                    "teacher_answers": [f"reasoning\nAnswer: {row['reward_model']['ground_truth']}"],
                    "teacher_fingerprint": FINGERPRINT,
                    "teacher_fingerprint_digest": DIGEST,
                    "sampling": {"finish_reasons": ["stop"], "truncated": [False]},
                    "source": "gad",
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

PROMPT_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.string())])),
    ]
)


def source_row(uid: str, problem: str, answer: str) -> dict:
    return {
        "data_source": "math_dapo",
        "prompt": [{"role": "user", "content": problem}],
        "ability": "MATH",
        "reward_model": {"ground_truth": answer, "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": uid},
    }


def write_source(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=PROMPT_SCHEMA), path)


def build_unique(count: int, *, twins: set[int], rows_per_problem: int = 1) -> dict[str, dict]:
    """`count` problems where the given indices are duplicated under a second uid."""
    unique: dict[str, dict] = {}
    for index in range(count):
        problem = f"problem {index}"
        for replica in range(2 if index in twins else rows_per_problem):
            uid = f"u{index}" if replica == 0 else f"u{index}-dup"
            prompt = [{"role": "user", "content": problem}]
            unique[uid] = {
                "index": uid,
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(
                    json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "ground_truth": str(index),
            }
    return unique


class GroupedSplitTests(unittest.TestCase):
    def test_duplicate_prompts_always_land_on_the_same_side(self) -> None:
        for seed in (0, 1, 42, 1234):
            unique = build_unique(60, twins=set(range(0, 60, 3)))
            train, dev, info = prepare_prompts.split(unique, seed=seed, dev_size=12)
            train_hashes = {unique[uid]["prompt_hash"] for uid in train}
            dev_hashes = {unique[uid]["prompt_hash"] for uid in dev}
            self.assertEqual(train_hashes & dev_hashes, set(), f"seed {seed} leaked a prompt")
            self.assertEqual(info["leaked_prompt_hashes"], [])
            self.assertEqual(len(train) + len(dev), len(unique))

    def test_a_shared_prompt_is_never_split_across_sides(self) -> None:
        unique = build_unique(20, twins={0, 1, 2})
        train, dev, info = prepare_prompts.split(unique, seed=7, dev_size=3)
        for index in (0, 1, 2):
            uids = {f"u{index}", f"u{index}-dup"}
            self.assertTrue(uids <= set(train) or uids <= set(dev), f"problem {index} was split")
        self.assertEqual(info["unit"], "prompt_hash_group")
        self.assertGreaterEqual(info["multi_id_groups_in_dev"]
                                + (info["same_prompt_groups_total"] - info["multi_id_groups_in_dev"]), 0)

    def test_dev_size_overshoot_is_reported_not_hidden(self) -> None:
        # every problem has a twin, so a drawn group adds two problems
        unique = build_unique(30, twins=set(range(30)))
        train, dev, info = prepare_prompts.split(unique, seed=3, dev_size=5)
        self.assertEqual(len(dev), info["dev_size_actual"])
        self.assertEqual(info["dev_size_requested"], 5)
        self.assertEqual(info["dev_size_overshoot"], len(dev) - 5)
        self.assertGreaterEqual(info["dev_size_overshoot"], 0)

    def test_split_is_deterministic_and_seed_sensitive(self) -> None:
        unique = build_unique(40, twins={1, 5, 9})
        first = prepare_prompts.split(unique, seed=42, dev_size=8)
        again = prepare_prompts.split(unique, seed=42, dev_size=8)
        other = prepare_prompts.split(unique, seed=43, dev_size=8)
        self.assertEqual(first[0], again[0])
        self.assertEqual(first[1], again[1])
        self.assertNotEqual(first[1], other[1])

    def test_impossible_dev_size_is_rejected(self) -> None:
        unique = build_unique(4, twins=set())
        with self.assertRaises(SystemExit):
            prepare_prompts.split(unique, seed=1, dev_size=4)


class PreparePromptsEndToEndTests(unittest.TestCase):
    def _run(self, source: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SAMPLE_DIR / "prepare_prompts.py"),
             "--input", str(source), "--out-dir", str(out), "--dev-size", "6", *extra],
            capture_output=True, text=True,
        )

    def test_manifests_record_a_leak_free_grouped_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dapo.parquet"
            rows = []
            for index in range(40):
                rows.append(source_row(f"u{index}", f"problem {index}", str(index)))
                if index % 4 == 0:  # same question under a second id
                    rows.append(source_row(f"u{index}-dup", f"problem {index}", str(index)))
            write_source(source, rows)

            out = root / "prepared"
            result = self._run(source, out, "--expected-unique", "50")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["train_dev_prompt_leak"], 0)
            self.assertEqual(stats["split"]["unit"], "prompt_hash_group")
            self.assertEqual(stats["split"]["leaked_prompt_hashes"], [])

            train = pq.read_table(out / "train.parquet").to_pylist()
            dev = pq.read_table(out / "dev.parquet").to_pylist()
            train_hashes = {prepare_prompts.hash_raw_prompt(row["prompt"]) for row in train}
            dev_hashes = {prepare_prompts.hash_raw_prompt(row["prompt"]) for row in dev}
            self.assertEqual(train_hashes & dev_hashes, set())
            self.assertEqual(len(train) + len(dev), 50)

            # The downstream contract check must accept the pair. It runs on the
            # *converted* GAD parquets, which is where the original leak surfaced
            # ("train/val prompt leak" -> validate_data.py refused the data).
            index_path = root / "teacher_index.jsonl"
            write_teacher_index(index_path, train + dev)
            converted = []
            for name, prompts in (("train", out / "train.parquet"), ("dev", out / "dev.parquet")):
                target = root / f"gad_{name}.parquet"
                built = subprocess.run(
                    [sys.executable, str(SAMPLE_DIR / "build_gad_data.py"),
                     "--prompt-parquet", str(prompts), "--index", str(index_path),
                     "--out", str(target), "--select", "first"],
                    capture_output=True, text=True,
                )
                self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
                converted.append(target)

            report = root / "validation_report.json"
            validated = subprocess.run(
                [sys.executable, str(SAMPLE_DIR / "validate_data.py"),
                 "--train", str(converted[0]), "--val", str(converted[1]), "--report", str(report)],
                capture_output=True, text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stdout + validated.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["leakage"]["shared_prompt_hashes"], 0)
            self.assertEqual(payload["leakage"]["shared_uids"], 0)

    def test_external_dev_leak_is_refused_unless_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dapo.parquet"
            write_source(source, [source_row("u1", "problem 1", "1"), source_row("u2", "problem 2", "2")])
            external = root / "external.parquet"
            write_source(external, [source_row("e1", "problem 1", "1")])

            out = root / "prepared"
            result = self._run(source, out, "--card-full-train", "--dev-file", str(external))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt leak", result.stdout + result.stderr)

            out_allowed = root / "prepared_allowed"
            result = self._run(source, out_allowed, "--card-full-train", "--dev-file", str(external),
                               "--allow-dev-leak")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((out_allowed / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["train_dev_prompt_leak"], 1)
            self.assertEqual(manifest["split"]["unit"], "external_dev_file")


class RunSampleCliTests(unittest.TestCase):
    """`--stage convert` must select both conversion steps (and unknown names fail)."""

    def _config(self, root: Path) -> Path:
        config = root / "sample.yaml"
        config.write_text(
            "\n".join(
                [
                    "prompts:",
                    f"  out_dir: {root / 'prepared'}",
                    "  repo_id: test/repo",
                    "  filename: data/dapo.parquet",
                    "  revision: deadbeef",
                    "  seed: 42",
                    "  dev_size: 8",
                    "  expected_unique: 0",
                    "  on_conflict: error",
                    "  card_full_train: false",
                    "teacher:",
                    "  index_out: " + str(root / "teacher_index.jsonl"),
                    "  enable_thinking: false",
                    "gad_data:",
                    "  select: first",
                    f"  out_train: {root / 'gad_train.parquet'}",
                    f"  out_val: {root / 'gad_dev.parquet'}",
                    "student:",
                    "  model_path: ''",
                    "  tokenizer_path: ''",
                    "lengths:",
                    "  max_prompt_length: 2048",
                    "  max_response_length: 512",
                    "eval:",
                    "  out_dir: ''",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return config

    def _run(self, config: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPTS_DIR / "run_sample.sh"), "--config", str(config), "--python", sys.executable,
             "--dry-run", *extra],
            capture_output=True, text=True,
        )

    def test_convert_alias_runs_both_conversion_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            result = self._run(config, "--stage", "convert")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("stage: convert_train", result.stdout)
            self.assertIn("stage: convert_dev", result.stdout)
            self.assertIn("build_gad_data.py", result.stdout)
            self.assertNotIn("stage: prompts", result.stdout)
            self.assertNotIn("stage: validate", result.stdout)
            self.assertIn("executed stages: convert_train convert_dev", result.stdout)

    def test_single_conversion_step_is_selectable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            result = self._run(config, "--stage", "convert_dev")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("skipping stage 'convert_train'", result.stdout)
            self.assertIn("stage: convert_dev", result.stdout)

    def test_unknown_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            result = self._run(config, "--stage", "convert-both")
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("unknown --stage", result.stdout + result.stderr)
            self.assertIn("convert_train", result.stdout + result.stderr)

    def test_repeated_stage_flags_are_all_honoured(self) -> None:
        """`--stage a --stage b` must not silently drop the earlier one."""
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            result = self._run(config, "--stage", "convert", "--stage", "validate")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("stage: convert_train", result.stdout)
            self.assertIn("stage: convert_dev", result.stdout)
            self.assertIn("stage: validate", result.stdout)
            self.assertNotIn("stage: prompts", result.stdout)

            # a comma separated list is equivalent
            result = self._run(config, "--stage", "convert_dev,validate")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("stage: convert_dev", result.stdout)
            self.assertIn("stage: validate", result.stdout)
            self.assertIn("skipping stage 'convert_train'", result.stdout)

            # and an unknown element is still rejected
            result = self._run(config, "--stage", "convert, bogus")
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("unknown --stage", result.stdout + result.stderr)


FAKE_SERVER = '''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL = sys.argv[2]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class RunSampleTeacherWrapperTests(unittest.TestCase):
    """`run_sample_teacher.sh` -- one command instead of two terminals.

    The teacher is a separate long-running HTTP server, so the wrapper starts
    (or reuses) it, waits for readiness and checks the served model name before
    handing over to `run_sample.sh`.
    """

    def _fake_endpoint(self, model: str):
        port = free_port()
        proc = subprocess.Popen(
            [sys.executable, "-c", FAKE_SERVER, str(port), model],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=0.5) as response:
                    if response.status == 200:
                        return port, proc
            except Exception:
                time.sleep(0.1)
        proc.kill()
        proc.wait(timeout=5)
        raise unittest.SkipTest("could not start the fake OpenAI endpoint")

    def _config(self, root: Path, port: int, model: str) -> Path:
        source = root / "source.parquet"
        write_source(source, [source_row(f"u{i}", f"problem {i}", str(i)) for i in range(24)])
        config = root / "sample.yaml"
        config.write_text(
            "\n".join(
                [
                    "prompts:",
                    f"  input_parquet: {source}",
                    f"  out_dir: {root / 'prepared'}",
                    "  expected_unique: 0",
                    "  seed: 42",
                    "  dev_size: 6",
                    "  on_conflict: error",
                    "teacher:",
                    f"  base_url: http://127.0.0.1:{port}/v1",
                    f"  model: {model}",
                    f"  index_out: {root / 'teacher_index.jsonl'}",
                    "  enable_thinking: false",
                    "gad_data:",
                    "  select: first",
                    f"  out_train: {root / 'gad_train.parquet'}",
                    f"  out_val: {root / 'gad_dev.parquet'}",
                    "student:",
                    "  model_path: ''",
                    "  tokenizer_path: ''",
                    "lengths:",
                    "  max_prompt_length: 2048",
                    "  max_response_length: 512",
                    "eval:",
                    '  out_dir: ""',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return config

    def _run(self, config: Path, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPTS_DIR / "run_sample_teacher.sh"), "--config", str(config),
             "--python", sys.executable, "--no-server", *extra],
            capture_output=True, text=True,
        )

    def test_reuses_a_running_endpoint_and_runs_the_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port, proc = self._fake_endpoint("teacher-test")
            try:
                config = self._config(root, port, "teacher-test")
                result = self._run(config, "--stage", "prompts")
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("reusing the endpoint already running", result.stdout)
                self.assertIn("executed stages: prompts", result.stdout)
                self.assertIn("done (stages: prompts)", result.stdout)
                self.assertTrue((root / "prepared" / "train.parquet").exists())
                # a server this script did not start must never be stopped
                self.assertIsNone(proc.poll())
            finally:
                proc.kill()

    def test_served_model_name_mismatch_is_refused(self) -> None:
        """A wrong teacher.model would fail all 17,917 requests - fail fast."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            port, proc = self._fake_endpoint("some-other-model")
            try:
                config = self._config(root, port, "expected-teacher")
                result = self._run(config, "--stage", "prompts")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("serves a different model name", result.stdout + result.stderr)
                self.assertIn("expected-teacher", result.stdout + result.stderr)
                self.assertFalse((root / "prepared" / "train.parquet").exists())
            finally:
                proc.kill()

    def test_unknown_argument_is_rejected(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPTS_DIR / "run_sample_teacher.sh"), "--nope"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown argument", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
