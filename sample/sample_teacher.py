#!/usr/bin/env python3
"""Sample teacher *text* answers for every prompt.

Two interchangeable backends (``--backend``):

``openai`` (default)
    HTTP requests to an OpenAI-compatible endpoint (``scripts/serve_teacher.sh``).
    Black-box by construction: only ``/v1/chat/completions`` text responses are
    used, never logits, and the teacher service is not needed during GAD training.

``vllm-offline``
    Load the teacher in-process with ``vllm.LLM`` and generate directly: one
    process, one command, no port and nothing to keep alive next to the sampler.
    The chat template and sampling parameters are the same, so the produced index
    is interchangeable - only the fingerprint digest differs (the engine is part
    of the fingerprint, so an HTTP index is never silently extended offline).

Guarantees (plan section 5.2 - 5.9), independent of the backend:

* resume is keyed on ``uid`` **and** ``raw_prompt_hash`` **and** the sampling
  fingerprint digest, so a changed prompt or a changed teacher configuration can
  never reuse an old answer;
* the existing index file is never deleted or rewritten: a fingerprint conflict
  aborts the run and leaves the file untouched;
* every record is flushed as it is produced (``--fsync`` for extra safety) and
  failures land in ``<out>.failures.jsonl``;
* ``finish_reason`` and truncation are recorded per answer;
* a run with unhandled failures exits non-zero unless ``--allow-incomplete`` is
  given explicitly;
* ``--log-file`` sends this process's stdout/stderr (and those of the vLLM
  workers, via ``dup2``) to a file, so an unread terminal can never block the run
  with a full pty buffer.

Examples::

    # in-process teacher (single process, recommended on a machine with the weights)
    python sample_teacher.py \
        --prompt-parquet data/dapo/train.parquet --prompt-parquet data/dapo/dev.parquet \
        --out data/dapo/teacher_index.jsonl \
        --backend vllm-offline --model-path /models/Qwen3-30B-A3B-Instruct-2507 \
        --tensor-parallel-size 4 --max-model-len 12288 \
        --m 1 --temperature 0.7 --top-p 0.95 --max-tokens 8192 \
        --student-tokenizer /models/Qwen3-1.7B --enable-thinking false \
        --progress log --log-every 100 --log-file data/dapo/logs/sample.log

    # remote / separately served teacher
    python sample_teacher.py \
        --prompt-parquet data/dapo/train.parquet \
        --out data/dapo/teacher_index.jsonl \
        --base-url http://127.0.0.1:18080/v1 --teacher-model Qwen3-30B-A3B-Instruct-2507 \
        --m 4 --temperature 0.7 --top-p 0.95 --max-tokens 8192 \
        --student-tokenizer /models/Qwen3-1.7B --enable-thinking false
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from teacher_engines import (  # noqa: E402
    TeacherEngine,
    engine_from_args,
    load_engine_factory,
)
from teacher_index import (  # noqa: E402
    INDEX_HEADER_KEY,
    TeacherIndexError,
    TeacherIndexRecord,
    append_record,
    find_fingerprint_mismatch,
    header_record,
    read_done_uids,
)
from task_template import (  # noqa: E402
    build_teacher_fingerprint_payload,
    extract_question_text,
    hash_raw_prompt,
    hash_teacher_fingerprint,
    normalize_raw_prompt,
    sha256_text,
    stable_json_dumps,
)


def log(message: str) -> None:
    print(f"[sample_teacher] {message}", flush=True)


_LOG_HANDLES: list = []


def redirect_output(path: str | Path) -> Path:
    """Send fd 1/2 (and therefore the vLLM worker processes) to a file.

    Python-level redirection would not cover the vLLM workers that inherit the
    descriptors, and a pty whose reader went away is exactly what made a detached
    run block in ``write(2)``. ``dup2`` fixes both.
    """
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a", buffering=1, encoding="utf-8", errors="replace")
    _LOG_HANDLES.append(handle)
    os.dup2(handle.fileno(), 1)
    os.dup2(handle.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    sys.stderr = os.fdopen(2, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    log(f"stdout/stderr -> {target}")
    return target


def load_prompts(paths: list[str], *, prompt_key: str) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    seen: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"[sample_teacher] ERROR: prompt parquet not found: {path}")
        table = pq.read_table(path)
        columns = set(table.column_names)
        key = prompt_key if prompt_key in columns else ("prompt" if "prompt" in columns else "content")
        if key not in columns:
            raise SystemExit(f"[sample_teacher] ERROR: {path} has no prompt column (tried {prompt_key}/prompt/content)")
        for row in table.select([key, "extra_info"]).to_pylist():
            uid = str(row["extra_info"]["index"])
            prompt = normalize_raw_prompt(row[key])
            prompt_hash = hash_raw_prompt(prompt)
            if uid in seen:
                if seen[uid] != prompt_hash:
                    raise SystemExit(
                        f"[sample_teacher] ERROR: uid {uid!r} appears with two different prompts; "
                        "fix the prompt files before sampling"
                    )
                continue
            seen[uid] = prompt_hash
            rows.append({"uid": uid, "prompt": prompt, "raw_prompt_hash": prompt_hash, "source": str(path)})
    return rows


def check_ground_truth_leak(index_path: Path, rows: list[dict]) -> dict:
    """The teacher must never be shown the reference answer.

    A plain substring test would false-positive on short answers such as "34",
    so we only flag the *answer contract* form -- an ``Answer: <gt>`` line or a
    ``\\boxed{<gt>}`` form inside the prompt -- which is what an accidental
    concatenation of the reference solution would look like.
    """
    import pyarrow.parquet as pq

    ground_truths: dict[str, str] = {}
    for path in {row["source"] for row in rows}:
        table = pq.read_table(path)
        for row in table.select(["extra_info", "reward_model"]).to_pylist():
            ground_truths[str(row["extra_info"]["index"])] = str(row["reward_model"]["ground_truth"])

    suspicious = []
    checked = 0
    for row in rows:
        ground_truth = ground_truths.get(row["uid"], "")
        if not ground_truth:
            continue
        checked += 1
        text = extract_question_text(row["prompt"])
        lowered = text.lower()
        if f"answer: {ground_truth.lower()}" in lowered or f"\\boxed{{{ground_truth.lower()}}}" in lowered:
            suspicious.append(row["uid"])
    return {"checked": checked, "suspicious_uids": suspicious}


def build_fingerprint(args: argparse.Namespace, extra_body: dict, engine: TeacherEngine | None = None) -> dict:
    """Everything that must match for two samplings to be comparable.

    The engine's own fields are recorded only when the engine declares them, so
    indexes produced by the HTTP backend before this refactor keep their digest
    and stay resumable.
    """
    model = args.teacher_model or (Path(args.model_path).name if getattr(args, "model_path", None) else None)
    return build_teacher_fingerprint_payload(
        provider=getattr(engine, "provider", "openai_compatible") if engine else "openai_compatible",
        model=model,
        base_url=getattr(args, "base_url", None) or None,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_tokens,
        num_samples=args.m,
        seed=args.seed,
        extra_body=extra_body,
        engine=engine.fingerprint_fields() if engine else None,
    )


def prepare_index(path: Path, header: dict) -> None:
    """Create the index with a header, or verify the existing header matches."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"created index {path}")
        return
    with path.open(encoding="utf-8") as handle:
        first = handle.readline().strip()
    if not first:
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    try:
        row = json.loads(first)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"[sample_teacher] ERROR: existing index {path} is not valid JSONL ({exc}). "
            "Refusing to modify it; move it aside or choose another --out."
        ) from exc
    if INDEX_HEADER_KEY not in row:
        raise SystemExit(
            f"[sample_teacher] ERROR: existing index {path} has no header line. "
            f"Expected the first line to contain {INDEX_HEADER_KEY!r}. Refusing to modify it."
        )
    if row.get("teacher_fingerprint_digest") != header["teacher_fingerprint_digest"]:
        raise SystemExit(
            f"[sample_teacher] ERROR: sampling fingerprint changed for {path}.\n"
            f"  existing: {row.get('teacher_fingerprint_digest')}\n"
            f"  current : {header['teacher_fingerprint_digest']}\n"
            "The file was left untouched. Use a new --out path (recommended) or move the old index aside."
        )


def check_resume_consistency(path: Path, rows: list[dict], digest: str) -> None:
    """Detect same-uid/different-prompt and fingerprint problems before sampling."""
    mismatch = find_fingerprint_mismatch(path, digest)
    if mismatch is not None:
        raise SystemExit(
            f"[sample_teacher] ERROR: record at {path}:{mismatch['line']} was produced with a different "
            f"sampling fingerprint ({mismatch['digest']}). Refusing to resume onto it."
        )
    expected = {row["uid"]: row["raw_prompt_hash"] for row in rows}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if INDEX_HEADER_KEY in row:
                continue
            uid = str(row.get("uid"))
            if uid in expected and str(row.get("raw_prompt_hash")) != expected[uid]:
                raise SystemExit(
                    f"[sample_teacher] ERROR: uid {uid!r} at {path}:{line_number} was sampled from a different "
                    "prompt than the current parquet. The prompt file changed; use a new index."
                )


def student_token_stats(texts: list[str], tokenizer_path: str, *, limit: int, max_tokens: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths = []
    truncated = 0
    for text in texts[:limit] if limit else texts:
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(n_tokens)
        if n_tokens >= max_tokens:
            truncated += 1
    lengths.sort()

    def quantile(q: float) -> int:
        if not lengths:
            return 0
        return lengths[min(len(lengths) - 1, max(0, int(round(q * (len(lengths) - 1)))))]

    return {
        "count": len(lengths),
        "p50": quantile(0.50),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": lengths[-1] if lengths else 0,
        "max_tokens": max_tokens,
        "at_or_over_limit": truncated,
        "at_or_over_limit_rate": (truncated / len(lengths)) if lengths else 0.0,
        "tokenizer_path": tokenizer_path,
    }


def run_engine(
    engine: TeacherEngine,
    todo: Sequence[Mapping[str, Any]],
    *,
    out_path: Path,
    fingerprint: dict,
    digest: str,
    m: int,
    fsync: bool = False,
    progress: str = "log",
    log_every: int = 100,
) -> dict:
    """Write every engine result to the index, with one code path for all backends.

    The engine owns concurrency/batching and calls ``on_result`` for each prompt;
    everything about durability (append + flush, failure list, counters) lives
    here, so the two backends cannot drift apart.
    """
    counters = {"ok": 0, "fail": 0, "truncated": 0}
    failures: list[dict] = []
    texts_for_stats: list[str] = []
    write_lock = threading.Lock()
    total = len(todo)
    bar = None
    if progress == "tqdm":
        try:
            from tqdm import tqdm

            bar = tqdm(total=total, desc="teacher", unit="prompt")
        except ImportError:  # pragma: no cover - tqdm is optional
            bar = None

    def handle(row: Mapping[str, Any], payload: dict | None, error: str | None) -> None:
        with write_lock:
            if error is not None or payload is None:
                counters["fail"] += 1
                failures.append(
                    {"uid": row["uid"], "raw_prompt_hash": row["raw_prompt_hash"], "error": str(error)}
                )
                sys.stderr.write(f"[fail] uid={row['uid']}: {error}\n")
                if bar is not None:
                    bar.update(1)
                return

            record = TeacherIndexRecord(
                uid=row["uid"],
                raw_prompt_hash=row["raw_prompt_hash"],
                teacher_answers=payload["answers"],
                teacher_fingerprint=fingerprint,
                teacher_fingerprint_digest=digest,
                sampling={
                    "finish_reasons": payload["finish_reasons"],
                    "truncated": payload["truncated"],
                    "attempts": payload["attempts"],
                    "num_answers": len(payload["answers"]),
                },
                source="gad",
            )
            append_record(out_path, record, fsync=fsync)
            counters["ok"] += 1
            counters["truncated"] += sum(1 for flag in payload["truncated"] if flag)
            texts_for_stats.extend(payload["answers"])
            if bar is not None:
                bar.update(1)
            elif progress == "log" and log_every and counters["ok"] % log_every == 0:
                log(f"  written {counters['ok']}/{total} (failed {counters['fail']})")

    log(f"engine: {engine.name} ({stable_json_dumps(engine.describe())})")
    try:
        engine.sample(todo, handle)
    finally:
        if bar is not None:
            bar.close()
        engine.close()
    return {"counters": counters, "failures": failures, "texts_for_stats": texts_for_stats, "total": total}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-parquet", action="append", required=True,
                        help="Prompt parquet (repeatable: train and dev).")
    parser.add_argument("--out", required=True, help="Teacher index JSONL (created or resumed).")
    parser.add_argument("--backend", choices=("openai", "vllm-offline"), default="openai",
                        help="openai (default): HTTP endpoint. vllm-offline: load the teacher in this process.")
    # --- openai backend ------------------------------------------------------
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (openai backend).")
    parser.add_argument("--api-key", default=os.environ.get("GAD_TEACHER_API_KEY", "EMPTY"))
    parser.add_argument("--concurrency", type=int, default=16, help="openai backend only.")
    parser.add_argument("--timeout", type=float, default=600.0, help="openai backend only.")
    parser.add_argument("--retries", type=int, default=3, help="openai backend only.")
    # --- vllm-offline backend ------------------------------------------------
    parser.add_argument("--model-path", default=None, help="Teacher weights directory (vllm-offline backend).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=12288)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128, help="Prompts per llm.generate call (vllm-offline).")
    parser.add_argument("--gpus", default=None,
                        help="CUDA_VISIBLE_DEVICES for this process (vllm-offline); normally set by run_sample.sh.")
    parser.add_argument("--engine-kwargs-json", default=None, help="Extra kwargs merged into vllm.LLM(...).")
    # --- shared --------------------------------------------------------------
    parser.add_argument("--teacher-model", default=None,
                        help="Model id recorded in the fingerprint (served name, or the weights path).")
    parser.add_argument("--m", type=int, default=4, help="Answers per prompt (CARD uses 4).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=None, help="Recorded in the fingerprint and passed to the engine.")
    parser.add_argument("--limit", type=int, default=0, help="Only sample the first N prompts (smoke tests).")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--enable-thinking", choices=["true", "false", "unset"], default="false",
                        help="Forwarded as chat_template_kwargs.enable_thinking (recorded in the fingerprint).")
    parser.add_argument("--extra-body-json", default=None, help="Extra JSON merged into the HTTP request body.")
    parser.add_argument("--engine-factory", default=None, metavar="MODULE:CALLABLE",
                        help="Advanced/testing: build the engine yourself from the parsed args.")
    parser.add_argument("--student-tokenizer", default=None, help="Optional tokenizer for answer length stats.")
    parser.add_argument("--tokenizer-limit", type=int, default=500)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Exit 0 even when some prompts failed (recorded in the summary).")
    parser.add_argument("--fsync", action="store_true", help="fsync after every record (slower, safer).")
    parser.add_argument("--progress", choices=("log", "tqdm", "quiet"), default="log",
                        help="How to report progress. 'log' writes one line per --log-every prompts.")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-file", default=None,
                        help="Send this process's stdout/stderr (including vLLM workers) to a file. A full pty "
                             "buffer on an unread terminal can otherwise block the run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and fingerprint without sampling.")
    parser.add_argument("--summary-out", default=None, help="Where to write the run summary JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # argument validation before anything expensive: exit code 2 = usage error
    if args.backend == "openai" and not args.base_url:
        log("ERROR: --base-url is required for --backend openai")
        return 2
    if args.backend == "vllm-offline" and not args.model_path:
        log("ERROR: --model-path is required for --backend vllm-offline")
        return 2
    if args.m <= 0:
        raise SystemExit("[sample_teacher] ERROR: --m must be >= 1")
    if args.m > 1 and args.temperature <= 0:
        raise SystemExit("[sample_teacher] ERROR: --temperature must be > 0 when --m > 1")

    if args.log_file:
        redirect_output(args.log_file)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
        log(f"CUDA_VISIBLE_DEVICES={args.gpus}")

    if args.backend == "vllm-offline" and args.teacher_model is None:
        args.teacher_model = Path(args.model_path).name
    if args.teacher_model is None:
        log("ERROR: --teacher-model is required (or --model-path for the offline backend)")
        return 2

    extra_body: dict = {}
    if args.enable_thinking != "unset":
        extra_body["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking == "true"}
    if args.extra_body_json:
        extra_body.update(json.loads(args.extra_body_json))
    if args.seed is not None:
        extra_body["seed"] = int(args.seed)
    args.extra_body = extra_body
    if args.backend == "vllm-offline":
        # the offline engine consumes the template switch directly
        extra_body.pop("chat_template_kwargs", None)

    out_path = Path(args.out).expanduser().resolve()
    rows = load_prompts(args.prompt_parquet, prompt_key=args.prompt_key)
    if args.limit:
        rows = rows[: args.limit]
    log(f"{len(rows)} prompts from {len(args.prompt_parquet)} file(s)")

    # The engine is built (and therefore the fingerprint is fixed) before the
    # index is touched: a mismatch must abort without modifying anything.
    factory = load_engine_factory(args.engine_factory) if args.engine_factory else engine_from_args
    engine = factory(args)
    if args.backend == "vllm-offline" and args.concurrency != 16:
        log("note: --concurrency is ignored by the vllm-offline backend (vLLM schedules internally)")
    fingerprint = build_fingerprint(args, extra_body, engine)
    digest = hash_teacher_fingerprint(fingerprint)
    log(f"teacher fingerprint digest: {digest}")
    log(f"teacher fingerprint: {stable_json_dumps(fingerprint)}")
    log("training must use the same digest, or the data converter refuses to build the dataset")

    if args.dry_run:
        leak = check_ground_truth_leak(out_path, rows)
        log(f"dry-run: backend={args.backend} engine={engine.name}")
        log(f"dry-run: would sample {len(rows)} prompts x {args.m} answers -> {out_path}")
        log(f"dry-run: ground-truth contract leak check -> {len(leak['suspicious_uids'])} suspicious prompt(s)")
        return 0

    leak = check_ground_truth_leak(out_path, rows)
    if leak["suspicious_uids"]:
        raise SystemExit(
            f"[sample_teacher] ERROR: {len(leak['suspicious_uids'])} prompts contain the reference answer in the "
            f"answer-contract form (first: {leak['suspicious_uids'][:3]}). Refusing to send them to the teacher."
        )

    prepare_index(out_path, header_record(fingerprint=fingerprint, created_by="sample_teacher.py", notes={
        "prompt_parquets": [str(Path(p).resolve()) for p in args.prompt_parquet],
        "prompt_key": args.prompt_key,
        "enable_thinking": None if args.enable_thinking == "unset" else args.enable_thinking == "true",
        "engine": engine.describe(),
    }))
    check_resume_consistency(out_path, rows, digest)

    done = read_done_uids(out_path, expected_digest=digest)
    todo = [row for row in rows if row["uid"] not in done]
    log(f"{len(done)} prompts already sampled; {len(todo)} to go")

    if not todo:
        log("nothing to do")
        return 0

    result = run_engine(
        engine, todo, out_path=out_path, fingerprint=fingerprint, digest=digest, m=args.m,
        fsync=args.fsync, progress=args.progress, log_every=args.log_every,
    )
    counters = result["counters"]
    failures = result["failures"]
    texts_for_stats = result["texts_for_stats"]

    failures_path = Path(str(out_path) + ".failures.jsonl")
    if failures:
        with failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        log(f"{len(failures)} failures -> {failures_path} (re-run the same command to retry them)")

    summary = {
        "teacher_fingerprint": fingerprint,
        "teacher_fingerprint_digest": digest,
        "backend": args.backend,
        "engine": engine.describe(),
        "prompts_total": len(rows),
        "prompts_already_done": len(done),
        "prompts_written": counters["ok"],
        "prompts_failed": counters["fail"],
        "answers_truncated": counters["truncated"],
        "answers_per_prompt": args.m,
        "index": str(out_path),
        "failures_file": str(failures_path) if failures else None,
        "index_sha256": sha256_text(out_path.read_text(encoding="utf-8")),
        "ground_truth_leak_check": leak,
    }
    if args.student_tokenizer and texts_for_stats:
        summary["student_token_length"] = student_token_stats(
            texts_for_stats, args.student_tokenizer, limit=args.tokenizer_limit, max_tokens=args.max_tokens
        )
    summary_path = Path(args.summary_out).expanduser() if args.summary_out else Path(str(out_path) + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"summary -> {summary_path}")

    if counters["fail"] and not args.allow_incomplete:
        log(f"ERROR: {counters['fail']} prompts have no teacher answer; exiting non-zero")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
