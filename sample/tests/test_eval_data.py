#!/usr/bin/env python3
"""Tests for the external evaluation-set builder.

`prepare_eval_data.py` must never pretend a benchmark was included when its data
could not be read, and must not silently drop duplicate prompts. These tests use
tiny local fixtures (a parquet directory and a JSONL file) so the behaviour is
verified without downloading AIME/AMC/MATH500.

Run::

    python LMOps/sample/tests/test_eval_data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SAMPLE_DIR))

from task_template import build_content_messages  # noqa: E402

GAD_PROMPT_SCHEMA = pa.schema(
    [
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.string())])),
    ]
)


def write_parquet_dir(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), root / "data.parquet")
    return root


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def run_builder(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SAMPLE_DIR / "prepare_eval_data.py"), *args],
        capture_output=True, text=True,
    )


class EvalDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.aime = write_parquet_dir(
            self.root / "aime24",
            [
                {"id": "1", "problem": "Compute 1+1.", "answer": "2"},
                {"id": "2", "problem": "Compute 2+2.", "answer": "4.0"},
            ],
        )
        self.amc = write_jsonl(
            self.root / "amc24.jsonl",
            [
                {"problem": "Compute 3+3.", "answer": "6"},
                # duplicate of the first AIME problem with a different id
                {"problem": "Compute 1+1.", "answer": "2"},
            ],
        )
        self.sources = self.root / "sources.json"
        self.sources.write_text(
            json.dumps(
                {
                    "sources": [
                        {"name": "aime24", "kind": "parquet_dir", "path": str(self.aime)},
                        {"name": "amc24", "kind": "jsonl", "path": str(self.amc)},
                        {"name": "math500", "kind": "load_from_disk", "path": str(self.root / "does-not-exist")},
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_builds_available_benchmarks_and_lists_the_missing_one(self) -> None:
        out = self.root / "out"
        result = run_builder("--out-dir", str(out), "--sources", str(self.sources))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

        table = pq.read_table(out / "test.parquet")
        self.assertEqual(table.num_rows, 4)
        sources = sorted(set(table.column("data_source").to_pylist()))
        self.assertEqual(sources, ["eval_aime24", "eval_amc24"])

        manifest = json.loads((out / "eval_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["benchmarks_built"], {"aime24": 2, "amc24": 2})
        self.assertEqual(manifest["benchmarks_unavailable"], ["math500"])
        self.assertTrue(manifest["skipped"][0]["reason"])

        # '4.0' is normalised for cosmetic comparison; the scorer does the rest
        answers = {
            row["extra_info"]["index"]: row["reward_model"]["ground_truth"]
            for row in table.to_pylist()
        }
        self.assertEqual(answers["aime24-2"], "4")

    def test_strict_mode_fails_when_a_benchmark_is_unavailable(self) -> None:
        result = run_builder("--out-dir", str(self.root / "out_strict"), "--sources", str(self.sources), "--strict")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unavailable", result.stdout + result.stderr)

    def test_duplicate_prompts_within_eval_are_reported(self) -> None:
        out = self.root / "out_dupes"
        result = run_builder("--out-dir", str(out), "--sources", str(self.sources))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((out / "eval_manifest.json").read_text(encoding="utf-8"))
        duplicates = manifest["duplicate_prompts_within_eval"]
        self.assertEqual(len(duplicates), 1, duplicates)
        group = next(iter(duplicates.values()))
        self.assertEqual(sorted(group), ["aime24-1", "amc24-1"])

    def test_leakage_against_training_prompts_is_reported(self) -> None:
        # a training prompt identical to the first AIME problem
        train = self.root / "train.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "prompt": build_content_messages("Compute 1+1."),
                        "reward_model": {"ground_truth": "2", "style": "rule-lighteval/MATH_v2"},
                        "extra_info": {"index": "train-1"},
                    }
                ],
                schema=GAD_PROMPT_SCHEMA,
            ),
            train,
        )
        out = self.root / "out_leak"
        result = run_builder(
            "--out-dir", str(out), "--sources", str(self.sources), "--train-parquet", str(train)
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = json.loads((out / "eval_manifest.json").read_text(encoding="utf-8"))
        leaks = manifest["leakage_against_training_data"]
        self.assertTrue(leaks, "an identical eval/train prompt must be reported")
        self.assertEqual(leaks[0]["matches"], "train:train-1")

    def test_benchmark_filter_selects_a_subset(self) -> None:
        out = self.root / "out_subset"
        result = run_builder(
            "--out-dir", str(out), "--sources", str(self.sources), "--benchmarks", "amc24"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        table = pq.read_table(out / "test.parquet")
        self.assertEqual(sorted(set(table.column("data_source").to_pylist())), ["eval_amc24"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
