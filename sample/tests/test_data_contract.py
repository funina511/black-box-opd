#!/usr/bin/env python3
"""Data-contract tests for the sampling / GAD data pipeline (CPU only, no torch).

Run with either::

    python -m unittest discover -s LMOps/sample/tests -v
    python LMOps/sample/tests/test_data_contract.py

The cases map 1:1 onto plan section 6 "数据契约": normal records, missing/empty
teacher answers, duplicate uid, prompt mismatch, fingerprint change, truncation
and train/val leakage.
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

from teacher_index import (  # noqa: E402
    GAD_INDEX_SCHEMA_VERSION,
    TeacherIndex,
    TeacherIndexError,
    find_fingerprint_mismatch,
    header_record,
    read_done_uids,
)
from task_template import (  # noqa: E402
    build_teacher_fingerprint_payload,
    build_user_message,
    canonicalize_raw_prompt,
    hash_raw_prompt,
    hash_teacher_fingerprint,
)

PROMPT_SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("ability", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
        ("extra_info", pa.struct([("index", pa.string())])),
    ]
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


def prompt_row(uid: str, problem: str, answer: str, *, tag: str | None = None) -> dict:
    problem = problem if tag is None else f"{problem} {tag}"
    return {
        "data_source": "math_dapo",
        "prompt": [build_user_message(problem)],
        "ability": "MATH",
        "reward_model": {"ground_truth": answer, "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": uid},
    }


def write_prompts(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=PROMPT_SCHEMA), path)


def index_record(uid: str, prompt: list[dict], answers: list[str], *, fingerprint=FINGERPRINT, digest=DIGEST,
                 truncated: list[bool] | None = None) -> dict:
    flags = list(truncated or [False] * len(answers))
    return {
        "schema_version": GAD_INDEX_SCHEMA_VERSION,
        "uid": uid,
        "raw_prompt_hash": hash_raw_prompt(prompt),
        "teacher_answers": answers,
        "teacher_fingerprint": fingerprint,
        "teacher_fingerprint_digest": digest,
        "sampling": {
            "finish_reasons": ["length" if flag else "stop" for flag in flags],
            "truncated": flags,
        },
        "source": "gad",
    }


def write_index(path: Path, records: list[dict], *, header: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if header:
        lines.append(json.dumps(header_record(fingerprint=FINGERPRINT, created_by="test")))
    lines.extend(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_script(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SAMPLE_DIR / script), *args],
        capture_output=True,
        text=True,
    )


class TeacherIndexTests(unittest.TestCase):
    def test_roundtrip_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = [build_user_message("1 + 1 = ?")]
            write_index(root / "index.jsonl", [index_record("u1", prompt, ["a", "b"])])
            index = TeacherIndex.load(root / "index.jsonl")
            self.assertEqual(len(index), 1)
            record = index.require(uid="u1", raw_prompt=prompt)
            self.assertEqual(record.teacher_answers, ["a", "b"])
            self.assertEqual(read_done_uids(root / "index.jsonl", expected_digest=DIGEST), {"u1"})
            self.assertIsNone(find_fingerprint_mismatch(root / "index.jsonl", DIGEST))

    def test_empty_teacher_answer_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = [build_user_message("q")]
            write_index(root / "index.jsonl", [index_record("u1", prompt, [""])])
            with self.assertRaises(TeacherIndexError):
                TeacherIndex.load(root / "index.jsonl")

    def test_duplicate_uid_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt = [build_user_message("q")]
            write_index(
                root / "index.jsonl",
                [index_record("u1", prompt, ["a"]), index_record("u1", prompt, ["b"])],
            )
            with self.assertRaises(TeacherIndexError):
                TeacherIndex.load(root / "index.jsonl")

    def test_prompt_mismatch_is_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_index(root / "index.jsonl", [index_record("u1", [build_user_message("q1")], ["a"])])
            index = TeacherIndex.load(root / "index.jsonl")
            self.assertIsNotNone(index.get(uid="u1", raw_prompt=[build_user_message("q1")]))
            self.assertIsNone(index.get(uid="u1", raw_prompt=[build_user_message("q2")]))
            self.assertIsNotNone(index.get(uid="u1"))  # uid-only lookup still works

    def test_corrupt_jsonl_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "index.jsonl"
            write_index(path, [])
            with path.open("a", encoding="utf-8") as handle:
                handle.write("{not json}\n")
            with self.assertRaises(TeacherIndexError):
                TeacherIndex.load(path)

    def test_fingerprint_change_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = build_teacher_fingerprint_payload(
                provider="openai_compatible",
                model="teacher-other",
                base_url=None,
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=8192,
                num_samples=2,
                seed=None,
            )
            other_digest = hash_teacher_fingerprint(other)
            write_index(
                root / "index.jsonl",
                [index_record("u1", [build_user_message("q")], ["a"], fingerprint=other, digest=other_digest)],
            )
            mismatch = find_fingerprint_mismatch(root / "index.jsonl", DIGEST)
            self.assertIsNotNone(mismatch)
            self.assertEqual(read_done_uids(root / "index.jsonl", expected_digest=DIGEST), set())

    def test_card_v2_record_is_importable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "card.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "card.teacher_index.v2",
                        "uid": "u1",
                        "raw_prompt_hash": "whatever-card-used",
                        "teacher_answers": ["card answer"],
                        "teacher_fingerprint": {"model": "card-teacher"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            index = TeacherIndex.load(path)
            record = index.get(uid="u1")
            self.assertIsNotNone(record)
            self.assertEqual(record.teacher_answers, ["card answer"])
            self.assertEqual(record.source, "card-import")


class BuildGadDataTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        rows = [
            prompt_row("u1", "What is 2+2?", "4"),
            prompt_row("u2", "What is 3+3?", "6"),
        ]
        prompt_path = root / "prompts.parquet"
        write_prompts(prompt_path, rows)
        records = [
            index_record("u1", rows[0]["prompt"], ["first-answer-u1", "second-answer-u1"]),
            index_record("u2", rows[1]["prompt"], ["first-answer-u2"]),
        ]
        index_path = root / "index.jsonl"
        write_index(index_path, records)
        return prompt_path, index_path

    def test_default_select_first_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path, index_path = self._fixture(root)
            out = root / "gad.parquet"
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path), "--out", str(out),
                 "--expect-fingerprint-digest", DIGEST],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            table = pq.read_table(out)
            self.assertEqual(table.num_rows, 2)
            self.assertEqual(set(table.column_names), {"content", "teacher_response", "data_source", "ability",
                                                       "reward_model", "extra_info"})
            answers = table.column("teacher_response").to_pylist()
            self.assertEqual(answers, ["first-answer-u1", "first-answer-u2"])

    def test_missing_teacher_answer_fails_without_allow_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path, index_path = self._fixture(root)
            out = root / "gad.parquet"
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path), "--out", str(out)],
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            records = [index_record("u1", [build_user_message("What is 2+2?")], ["a"])]
            write_index(index_path, records, header=False)
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path), "--out", str(out)],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no usable teacher answer", result.stderr)

            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path), "--out", str(out),
                 "--allow-missing"],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(pq.read_table(out).num_rows, 1)

    def test_wrong_expected_fingerprint_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_path, index_path = self._fixture(root)
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path),
                 "--out", str(root / "gad.parquet"), "--expect-fingerprint-digest", "deadbeef"],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match", result.stderr)

    def test_select_correct_drops_incorrect_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [prompt_row("u1", "What is 2+2?", "4")]
            prompt_path = root / "prompts.parquet"
            write_prompts(prompt_path, rows)
            index_path = root / "index.jsonl"
            write_index(
                index_path,
                [index_record("u1", rows[0]["prompt"], ["I think the answer is 5.\nAnswer: 5",
                                                        "The sum is four.\nAnswer: 4"])],
            )
            out = root / "gad.parquet"
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path), "--out", str(out),
                 "--select", "correct"],
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(pq.read_table(out).column("teacher_response").to_pylist(),
                             ["The sum is four.\nAnswer: 4"])

    def test_select_all_keeps_incorrect_answers_and_is_the_default(self) -> None:
        """`all` is the source pool's selection=all: no correctness filter."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [prompt_row("u1", "What is 2+2?", "4"), prompt_row("u2", "What is 3+3?", "6")]
            prompt_path = root / "prompts.parquet"
            write_prompts(prompt_path, rows)
            index_path = root / "index.jsonl"
            write_index(
                index_path,
                [
                    index_record("u1", rows[0]["prompt"], ["I think the answer is 5.\nAnswer: 5"]),
                    index_record("u2", rows[1]["prompt"], ["The sum is six.\nAnswer: 6"]),
                ],
            )
            explicit = root / "all.parquet"
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path), "--out", str(explicit),
                 "--select", "all"],
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            default = root / "default.parquet"
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path), "--out", str(default)],
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            expected = ["I think the answer is 5.\nAnswer: 5", "The sum is six.\nAnswer: 6"]
            self.assertEqual(pq.read_table(explicit).column("teacher_response").to_pylist(), expected)
            self.assertEqual(pq.read_table(default).column("teacher_response").to_pylist(), expected)

    def test_truncation_aware_selections_read_the_sampler_flags(self) -> None:
        """`nontruncated` / `correct-nontruncated` must use sampling.truncated, not the score."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                prompt_row("u1", "What is 2+2?", "4"),   # right answer, complete
                prompt_row("u2", "What is 3+3?", "6"),   # right answer, TRUNCATED (top-level Answer:)
                prompt_row("u3", "What is 5+5?", "10"),  # wrong answer, complete
            ]
            prompt_path = root / "prompts.parquet"
            write_prompts(prompt_path, rows)
            index_path = root / "index.jsonl"
            write_index(
                index_path,
                [
                    index_record("u1", rows[0]["prompt"], ["The sum is four.\nAnswer: 4"]),
                    index_record("u2", rows[1]["prompt"], ["The sum is six.\nAnswer: 6"], truncated=[True]),
                    index_record("u3", rows[2]["prompt"], ["I think it is 11.\nAnswer: 11"]),
                ],
            )

            def convert(select: str) -> list[str]:
                out = root / f"{select}.parquet"
                result = run_script(
                    "build_gad_data.py",
                    ["--prompt-parquet", str(prompt_path), "--index", str(index_path),
                     "--out", str(out), "--select", select],
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return pq.read_table(out).column("teacher_response").to_pylist()

            self.assertEqual(convert("all"), [
                "The sum is four.\nAnswer: 4",
                "The sum is six.\nAnswer: 6",
                "I think it is 11.\nAnswer: 11",
            ])
            self.assertEqual(convert("nontruncated"), [
                "The sum is four.\nAnswer: 4",
                "I think it is 11.\nAnswer: 11",
            ])
            self.assertEqual(convert("correct"), [
                "The sum is four.\nAnswer: 4",
                "The sum is six.\nAnswer: 6",
            ])
            self.assertEqual(convert("correct-nontruncated"), ["The sum is four.\nAnswer: 4"])

    def test_deliberate_filtering_is_not_reported_as_missing_data(self) -> None:
        """`--select correct` drops rows on purpose; that must not be an error."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [prompt_row("u1", "What is 2+2?", "4"), prompt_row("u2", "What is 3+3?", "6")]
            prompt_path = root / "prompts.parquet"
            write_prompts(prompt_path, rows)
            index_path = root / "index.jsonl"
            write_index(
                index_path,
                [
                    index_record("u1", rows[0]["prompt"], ["The sum is four.\nAnswer: 4"]),
                    index_record("u2", rows[1]["prompt"], ["I think it is 7.\nAnswer: 7"]),
                ],
            )
            filtered = root / "filtered.parquet"
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(index_path),
                 "--out", str(filtered), "--select", "correct"],
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(pq.ParquetFile(filtered).metadata.num_rows, 1)
            sidecar = json.loads(Path(str(filtered) + ".selection.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["num_missing"], 0)
            self.assertEqual(sidecar["reasons"]["no_correct_answer"], 1)

            # ... but a prompt the teacher never answered IS missing data
            missing_index = root / "missing.jsonl"
            write_index(missing_index, [index_record("u1", rows[0]["prompt"], ["Answer: 4"])])
            result = run_script(
                "build_gad_data.py",
                ["--prompt-parquet", str(prompt_path), "--index", str(missing_index),
                 "--out", str(root / "bad.parquet"), "--select", "correct"],
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no usable teacher answer at all", result.stdout + result.stderr)


class ValidateDataTests(unittest.TestCase):
    def _gad_parquet(self, root: Path, name: str, rows: list[dict]) -> Path:
        path = root / name
        table_schema = pa.schema(
            [
                ("content", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
                ("teacher_response", pa.string()),
                ("data_source", pa.string()),
                ("ability", pa.string()),
                ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
                ("extra_info", pa.struct([("index", pa.string())])),
            ]
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=table_schema), path)
        return path

    @staticmethod
    def _row(uid: str, problem: str, answer: str, teacher: str) -> dict:
        return {
            "content": [build_user_message(problem)],
            "teacher_response": teacher,
            "data_source": "math_dapo",
            "ability": "MATH",
            "reward_model": {"ground_truth": answer, "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"index": uid},
        }

    def test_clean_data_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._gad_parquet(root, "train.parquet", [self._row("u1", "q1", "4", "reasoning\nAnswer: 4")])
            dev = self._gad_parquet(root, "dev.parquet", [self._row("u2", "q2", "6", "reasoning\nAnswer: 6")])
            result = run_script("validate_data.py", ["--train", str(train), "--val", str(dev)])
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_train_val_leak_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._gad_parquet(root, "train.parquet", [self._row("u1", "same question", "4", "Answer: 4")])
            dev = self._gad_parquet(root, "dev.parquet", [self._row("u9", "same question", "4", "Answer: 4")])
            result = run_script("validate_data.py", ["--train", str(train), "--val", str(dev)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt leak", result.stdout + result.stderr)

    def test_val_leak_can_be_recorded_instead_of_failing(self) -> None:
        """`card_full_train` uses an external dev set that may repeat DAPO questions."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._gad_parquet(root, "train.parquet", [self._row("u1", "same question", "4", "Answer: 4")])
            dev = self._gad_parquet(root, "dev.parquet", [self._row("u9", "same question", "4", "Answer: 4")])
            report = root / "report.json"
            result = run_script(
                "validate_data.py",
                ["--train", str(train), "--val", str(dev), "--allow-val-leak", "--report", str(report)],
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["leakage"]["shared_prompt_hashes"], 1)
            self.assertTrue(payload["leakage"]["allowed"])

    def test_ground_truth_in_prompt_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_row = self._row("u1", "q1", "4", "Answer: 4")
            bad_row["content"] = [{"role": "user", "content": "What is 2+2? Answer: 4"}]
            train = self._gad_parquet(root, "train.parquet", [bad_row])
            result = run_script("validate_data.py", ["--train", str(train)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ground truth", result.stdout + result.stderr)

    def test_empty_teacher_response_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = self._gad_parquet(root, "train.parquet", [self._row("u1", "q1", "4", "   ")])
            result = run_script("validate_data.py", ["--train", str(train)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("empty teacher_response", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
