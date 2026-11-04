from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import statistics
import time
from pathlib import Path
from typing import Any

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    git_commit,
    init_wandb,
    log_wandb_artifact,
    optional_file_sha256,
    package_version,
    percentile,
    sanitize_name,
    sha256_file,
)
from .quality import grade_output, validate_output


def parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--gpu-ids must be a comma-separated list of integers") from error
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids must contain at least one non-negative GPU id")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("--gpu-ids contains duplicate ids")
    return gpu_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample offline teacher responses with one data-parallel vLLM instance per GPU."
    )
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3"))
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Use only the first N prompts for a smoke test; 0 means the complete dataset.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=10480)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--request-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the tokenizer's thinking chat-template branch.",
    )
    parser.add_argument(
        "--basic-rejection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retry empty/no-boxed/repetitive generations.",
    )
    parser.add_argument(
        "--regenerate-until-correct",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Grade each generation immediately and retry incorrect answers. "
            "The retry budget is shared with --max-attempts-per-rollout."
        ),
    )
    parser.add_argument("--max-attempts-per-rollout", type=int, default=3)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")

    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "OPRD-High-Entropy"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.getenv("WANDB_RUN_GROUP"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-job-type", default="teacher_sampling")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.getenv("WANDB_MODE", "disabled"),
    )
    return parser


def _to_builtin(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "as_py"):
        return _to_builtin(value.as_py())
    if hasattr(value, "tolist"):
        return _to_builtin(value.tolist())
    return str(value)


def _normalize_messages(raw_prompt: Any) -> list[dict[str, str]]:
    value = _to_builtin(raw_prompt)
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if not isinstance(value, list):
        raise ValueError(f"Unsupported prompt type: {type(value).__name__}")

    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError(f"Invalid chat message: {message!r}")
        messages.append(
            {
                "role": str(message.get("role", "user")),
                "content": str(message["content"]),
            }
        )
    if not messages:
        raise ValueError("Prompt contains no messages")
    return messages


def _ground_truth(reward_model: Any) -> str | None:
    value = _to_builtin(reward_model)
    if isinstance(value, dict) and value.get("ground_truth") is not None:
        return str(value["ground_truth"])
    return None


def load_source_samples(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Teacher sampling requires pandas and pyarrow to read Parquet.") from error

    frame = pd.read_parquet(path)
    if "prompt" not in frame.columns:
        raise ValueError(f"{path} has no `prompt` column; columns={list(frame.columns)}")

    samples: list[dict[str, Any]] = []
    for global_index, (_, row) in enumerate(frame.iterrows()):
        samples.append(
            {
                # Row order gives stable resume keys even when the Parquet index
                # is not an integer RangeIndex.
                "global_index": global_index,
                "raw_prompt": _normalize_messages(row["prompt"]),
                "ground_truth": _ground_truth(row.get("reward_model")),
                "data_source": _to_builtin(row.get("data_source")),
                "ability": _to_builtin(row.get("ability")),
                "extra_info": _to_builtin(row.get("extra_info")),
            }
        )
    return samples


def _format_prompt(tokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as error:
        if "enable_thinking" not in str(error):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def assess_candidate(
    text: str,
    ground_truth: str | None,
    *,
    basic_rejection: bool,
    regenerate_until_correct: bool,
):
    """Return quality, optional correctness, and the reason to retry."""
    quality = validate_output(text)
    teacher_correct = grade_output(text, ground_truth) if regenerate_until_correct else None
    if basic_rejection and not quality.valid:
        rejection_reason = quality.reason
    elif regenerate_until_correct and teacher_correct is not True:
        rejection_reason = "incorrect"
    else:
        rejection_reason = None
    return quality, teacher_correct, rejection_reason


def retry_action(
    rejection_reason: str | None,
    *,
    attempt: int,
    max_attempts: int,
    preserve_exhausted: bool,
) -> str:
    """Choose accept, retry, terminal exhaustion, or legacy abandonment."""
    if rejection_reason is None:
        return "accept"
    if attempt + 1 < max_attempts:
        return "retry"
    return "exhaust" if preserve_exhausted else "abandon"


def regeneration_metadata(
    *,
    teacher_correct: bool | None,
    attempts_used: int,
    retry_exhausted: bool,
) -> dict[str, Any]:
    return {
        "teacher_correct": teacher_correct is True,
        "attempts_used": attempts_used,
        "correctness_exhausted": retry_exhausted and teacher_correct is not True,
        "retry_exhausted": retry_exhausted,
    }


def validate_regeneration_ground_truths(samples: list[dict[str, Any]], enabled: bool) -> None:
    if not enabled:
        return
    missing = [
        int(sample["global_index"])
        for sample in samples
        if sample.get("ground_truth") is None or not str(sample["ground_truth"]).strip()
    ]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            "--regenerate-until-correct requires ground truth for every prompt; "
            f"missing={len(missing)} indices=[{preview}{suffix}]"
        )


def validate_regeneration_grader(enabled: bool) -> None:
    if not enabled:
        return
    try:
        grader_ready = grade_output("Sanity check: \\boxed{0}", "0")
    except Exception as error:
        raise RuntimeError(
            "--regenerate-until-correct requires the verl math grader and its dependencies "
            "in SAMPLE_PYTHON."
        ) from error
    if grader_ready is not True:
        raise RuntimeError("Math grader preflight failed on a trivial boxed-answer sanity check.")


def _load_worker_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "generation_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "abandoned": 0,
            "rejection_reasons": {},
            "wall_time_seconds": 0.0,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "generation_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "abandoned": 0,
            "rejection_reasons": {},
            "wall_time_seconds": 0.0,
        }


def worker_main(
    *,
    rank: int,
    gpu_id: int,
    port: int,
    tasks: list[dict[str, Any]],
    model_path: str,
    temp_jsonl: str,
    stats_json: str,
    generation_config: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["VLLM_PORT"] = str(port)
    os.environ["MASTER_PORT"] = str(port)
    os.environ["NCCL_PORT"] = str(port)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    started = time.time()
    stats_path = Path(stats_json)
    stats = _load_worker_stats(stats_path)
    reason_counts = dict(stats.get("rejection_reasons", {}))
    pending = [{**task, "attempt": 0} for task in tasks]
    model = None
    try:
        model = LLM(
            model=model_path,
            tensor_parallel_size=1,
            max_model_len=generation_config["max_model_len"],
            trust_remote_code=generation_config["trust_remote_code"],
            gpu_memory_utilization=generation_config["gpu_memory_utilization"],
            dtype=generation_config["dtype"],
            seed=generation_config["seed"] + rank,
            enforce_eager=generation_config["enforce_eager"],
        )
        sampling_params = SamplingParams(
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            top_k=generation_config["top_k"],
            repetition_penalty=generation_config["repetition_penalty"],
            max_tokens=generation_config["max_new_tokens"],
        )
        tokenizer = model.get_tokenizer()

        output_path = Path(temp_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        needs_leading_newline = False
        if output_path.is_file() and output_path.stat().st_size > 0:
            with output_path.open("rb") as check_handle:
                check_handle.seek(-1, os.SEEK_END)
                needs_leading_newline = check_handle.read(1) != b"\n"
        with output_path.open("a", encoding="utf-8") as output_handle:
            # A killed worker can leave a partial trailing JSON object. Separate
            # new records from that fragment so a later resume can recover them.
            if needs_leading_newline:
                output_handle.write("\n")
                output_handle.flush()
            while pending:
                current = pending[: generation_config["request_batch_size"]]
                pending = pending[generation_config["request_batch_size"] :]
                formatted = [
                    _format_prompt(tokenizer, item["raw_prompt"], generation_config["enable_thinking"])
                    for item in current
                ]
                outputs = model.generate(formatted, sampling_params, use_tqdm=False)
                stats["generation_attempts"] += len(current)

                for item, request_output in zip(current, outputs):
                    candidate = request_output.outputs[0]
                    text = candidate.text
                    quality, teacher_correct, rejection_reason = assess_candidate(
                        text,
                        item.get("ground_truth"),
                        basic_rejection=generation_config["basic_rejection"],
                        regenerate_until_correct=generation_config["regenerate_until_correct"],
                    )
                    action = retry_action(
                        rejection_reason,
                        attempt=item["attempt"],
                        max_attempts=generation_config["max_attempts_per_rollout"],
                        preserve_exhausted=generation_config["regenerate_until_correct"],
                    )
                    if rejection_reason is not None:
                        stats["rejected"] += 1
                        reason_counts[rejection_reason] = reason_counts.get(rejection_reason, 0) + 1
                        if action == "retry":
                            pending.append({**item, "attempt": item["attempt"] + 1})
                            continue
                        if action == "abandon":
                            stats["abandoned"] += 1
                            continue

                    attempts_used = item["attempt"] + 1
                    retry_exhausted = action == "exhaust"
                    record = {
                        "global_index": item["global_index"],
                        "rollout_index": item["rollout_index"],
                        "raw_prompt": item["raw_prompt"],
                        "response": text,
                        "ground_truth": item.get("ground_truth"),
                        "data_source": item.get("data_source"),
                        "ability": item.get("ability"),
                        "extra_info": item.get("extra_info"),
                        "attempt": item["attempt"],
                        "is_valid": quality.valid,
                        "validation_reason": quality.reason,
                        "has_boxed_answer": quality.has_boxed_answer,
                        "prompt_tokens": len(request_output.prompt_token_ids or []),
                        "output_tokens": len(candidate.token_ids or []),
                        "finish_reason": _to_builtin(candidate.finish_reason),
                    }
                    if generation_config["regenerate_until_correct"]:
                        record.update(
                            regeneration_metadata(
                                teacher_correct=teacher_correct,
                                attempts_used=attempts_used,
                                retry_exhausted=retry_exhausted,
                            )
                        )
                    output_handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output_handle.flush()
                    stats["accepted"] += 1

                print(
                    f"[worker {rank} gpu={gpu_id}] accepted={stats['accepted']} "
                    f"pending={len(pending)} rejected={stats['rejected']} abandoned={stats['abandoned']}",
                    flush=True,
                )
    finally:
        stats["rejection_reasons"] = reason_counts
        stats["wall_time_seconds"] = float(stats.get("wall_time_seconds", 0.0)) + (time.time() - started)
        atomic_write_json(stats_path, stats)
        if model is not None:
            try:
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            except Exception:
                pass
            try:
                import gc
                import torch

                del model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass


def load_completed(temp_dir: Path) -> dict[tuple[int, int], dict[str, Any]]:
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(temp_dir.glob("worker_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            records: list[dict[str, Any]] = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    print(f"Skipping partial worker record {path}:{line_number}: {error}", flush=True)
                    continue
                if not isinstance(record, dict):
                    print(f"Skipping non-object worker record {path}:{line_number}", flush=True)
                    continue
                records.append(record)
        for record in records:
            if "global_index" not in record or "rollout_index" not in record:
                continue
            key = (int(record["global_index"]), int(record["rollout_index"]))
            completed.setdefault(key, record)
    return completed


def aggregate_worker_stats(temp_dir: Path) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "generation_attempts": 0,
        "accepted": 0,
        "rejected": 0,
        "abandoned": 0,
        "wall_time_seconds_sum_workers": 0.0,
        "rejection_reasons": {},
    }
    for path in sorted(temp_dir.glob("worker_*.stats.json")):
        try:
            stats = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("generation_attempts", "accepted", "rejected", "abandoned"):
            aggregate[key] += int(stats.get(key, 0))
        aggregate["wall_time_seconds_sum_workers"] += float(stats.get("wall_time_seconds", 0.0))
        for reason, count in stats.get("rejection_reasons", {}).items():
            reasons = aggregate["rejection_reasons"]
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return aggregate


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_parquet.is_file():
        raise FileNotFoundError(args.input_parquet)
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be positive")
    if args.max_prompts < 0:
        raise ValueError("--max-prompts must be non-negative")
    if args.max_new_tokens <= 0 or args.max_model_len <= 0:
        raise ValueError("Token limits must be positive")
    if args.max_new_tokens > args.max_model_len:
        raise ValueError("--max-new-tokens cannot exceed --max-model-len")
    if args.request_batch_size <= 0:
        raise ValueError("--request-batch-size must be positive")
    if args.max_attempts_per_rollout <= 0:
        raise ValueError("--max-attempts-per-rollout must be positive")
    if args.basic_rejection and args.regenerate_until_correct:
        raise ValueError(
            "--basic-rejection and --regenerate-until-correct cannot be enabled together; "
            "the minimal correctness mode retries only on mathematical correctness."
        )
    if not 0.0 <= args.temperature:
        raise ValueError("--temperature must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.repetition_penalty <= 0.0:
        raise ValueError("--repetition-penalty must be positive")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output_dir / "temp_rollout"
    temp_dir.mkdir(parents=True, exist_ok=True)

    samples = load_source_samples(args.input_parquet)
    if args.max_prompts > 0:
        samples = samples[: args.max_prompts]
    if not samples:
        raise ValueError(f"No prompts found in {args.input_parquet}")
    validate_regeneration_ground_truths(samples, args.regenerate_until_correct)
    validate_regeneration_grader(args.regenerate_until_correct)
    model_config_path = args.model_path / "config.json"
    run_name = args.wandb_run_name or f"teacher-sampling-{args.output_dir.name}"
    generation_config = {
        "input_parquet": str(args.input_parquet.resolve()),
        "input_sha256": sha256_file(args.input_parquet),
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": optional_file_sha256(model_config_path),
        "gpu_ids": args.gpu_ids,
        "num_rollouts": args.num_rollouts,
        "max_prompts": args.max_prompts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "request_batch_size": args.request_batch_size,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "basic_rejection": args.basic_rejection,
        "regenerate_until_correct": args.regenerate_until_correct,
        "max_attempts_per_rollout": args.max_attempts_per_rollout,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "git_commit": git_commit(),
        "vllm_version": package_version("vllm"),
        "transformers_version": package_version("transformers"),
        "sampler_sha256": sha256_file(Path(__file__)),
        "quality_sha256": sha256_file(Path(__file__).with_name("quality.py")),
    }

    config_path = args.output_dir / "sampling_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        ignored = {"gpu_ids", "request_batch_size"}
        mismatches = {
            key: (previous.get(key), generation_config.get(key))
            for key in generation_config
            if key not in ignored and previous.get(key) != generation_config.get(key)
        }
        if mismatches and not args.allow_config_mismatch:
            raise ValueError(
                "Refusing to resume an output directory with different sampling config. "
                f"Mismatches: {mismatches}. Use a new --output-dir or --allow-config-mismatch."
            )
    atomic_write_json(config_path, generation_config)

    run = init_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=run_name,
        job_type=args.wandb_job_type,
        config=generation_config,
    )
    exit_code = 0
    try:
        completed = load_completed(temp_dir)
        tasks: list[dict[str, Any]] = []
        for sample in samples:
            for rollout_index in range(args.num_rollouts):
                key = (sample["global_index"], rollout_index)
                if key not in completed:
                    tasks.append({**sample, "rollout_index": rollout_index})

        print(
            f"Loaded {len(samples)} prompts; expected={len(samples) * args.num_rollouts}, "
            f"completed={len(completed)}, remaining={len(tasks)}",
            flush=True,
        )
        chunks: list[list[dict[str, Any]]] = [[] for _ in args.gpu_ids]
        for index, task in enumerate(tasks):
            chunks[index % len(chunks)].append(task)

        processes: list[multiprocessing.Process] = []
        context = multiprocessing.get_context("spawn")
        configured_port_base = int(os.getenv("VLLM_PORT_BASE", "0"))
        port_base = configured_port_base or 20000 + (os.getpid() % 2000) * 16
        if port_base < 1024 or port_base + len(args.gpu_ids) >= 65535:
            raise ValueError(f"Invalid VLLM_PORT_BASE={port_base}")
        print(f"vLLM worker port base: {port_base}", flush=True)
        for rank, (gpu_id, chunk) in enumerate(zip(args.gpu_ids, chunks)):
            if not chunk:
                continue
            process = context.Process(
                target=worker_main,
                kwargs={
                    "rank": rank,
                    "gpu_id": gpu_id,
                    "port": port_base + rank,
                    "tasks": chunk,
                    "model_path": str(args.model_path),
                    "temp_jsonl": str(temp_dir / f"worker_{rank}.jsonl"),
                    "stats_json": str(temp_dir / f"worker_{rank}.stats.json"),
                    "generation_config": generation_config,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
        failed = [process.pid for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"Teacher sampling workers failed: pids={failed}")

        completed = load_completed(temp_dir)
        records = [completed[key] for key in sorted(completed)]
        raw_path = args.output_dir / "raw_samples.jsonl"
        atomic_write_jsonl(raw_path, records)

        expected = len(samples) * args.num_rollouts
        lengths = [int(record.get("output_tokens", 0)) for record in records]
        valid_count = sum(bool(record.get("is_valid")) for record in records)
        boxed_count = sum(bool(record.get("has_boxed_answer")) for record in records)
        truncated_count = sum(record.get("finish_reason") == "length" for record in records)
        attempts_used = [int(record.get("attempts_used", 1)) for record in records]
        stats = {
            "num_prompts": len(samples),
            "num_rollouts": args.num_rollouts,
            "expected_samples": expected,
            "generated_samples": len(records),
            "completion_rate": len(records) / expected if expected else 0.0,
            "valid_rate": valid_count / len(records) if records else 0.0,
            "boxed_rate": boxed_count / len(records) if records else 0.0,
            "truncated_rate": truncated_count / len(records) if records else 0.0,
            "output_tokens_mean": statistics.fmean(lengths) if lengths else 0.0,
            "output_tokens_p50": percentile(lengths, 0.50),
            "output_tokens_p95": percentile(lengths, 0.95),
            "wall_time_seconds": time.time() - started,
            **aggregate_worker_stats(temp_dir),
        }
        if args.regenerate_until_correct:
            correct_count = sum(record.get("teacher_correct") is True for record in records)
            exhausted_count = sum(bool(record.get("correctness_exhausted")) for record in records)
            first_attempt_correct_count = sum(
                record.get("teacher_correct") is True and int(record.get("attempts_used", 1)) == 1
                for record in records
            )
            stats.update(
                {
                    "correct_samples": correct_count,
                    "correct_rate": correct_count / len(records) if records else 0.0,
                    "exhausted_samples": exhausted_count,
                    "exhausted_rate": exhausted_count / len(records) if records else 0.0,
                    "attempts_used_mean": statistics.fmean(attempts_used) if attempts_used else 0.0,
                    "attempts_used_p50": percentile(attempts_used, 0.50),
                    "attempts_used_p95": percentile(attempts_used, 0.95),
                    "first_attempt_correct_rate": (
                        first_attempt_correct_count / len(records) if records else 0.0
                    ),
                }
            )
        stats_path = args.output_dir / "sampling_stats.json"
        atomic_write_json(stats_path, stats)
        manifest_path = args.output_dir / "manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "kind": "teacher_raw_samples",
                "run_name": run_name,
                "config": generation_config,
                "stats": stats,
                "raw_samples_sha256": sha256_file(raw_path),
            },
        )

        print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
        if run is not None:
            run.log({f"sample/{key}": value for key, value in stats.items() if isinstance(value, (int, float))})
            run.summary.update({f"sample/{key}": value for key, value in stats.items() if isinstance(value, (int, float))})
            try:
                import wandb

                table = wandb.Table(
                    columns=[
                        "global_index",
                        "rollout_index",
                        "ground_truth",
                        "response",
                        "valid",
                        "teacher_correct",
                        "attempts_used",
                        "output_tokens",
                    ]
                )
                for record in records[:64]:
                    table.add_data(
                        record["global_index"],
                        record["rollout_index"],
                        record.get("ground_truth"),
                        record.get("response", "")[:4000],
                        record.get("is_valid"),
                        record.get("teacher_correct"),
                        record.get("attempts_used", 1),
                        record.get("output_tokens", 0),
                    )
                run.log({"sample/examples": table})
            except Exception as error:
                print(f"W&B sample table skipped: {error}", flush=True)
            log_wandb_artifact(
                run,
                name=f"{sanitize_name(run_name)}-raw",
                artifact_type="teacher_raw_samples",
                files=[raw_path, config_path, stats_path, manifest_path],
                metadata=stats,
            )

        if len(records) != expected and not args.allow_incomplete:
            print(
                "Sampling is incomplete. Re-run the same command to resume, or pass --allow-incomplete.",
                flush=True,
            )
            exit_code = 2
    except Exception:
        exit_code = 1
        raise
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    raise SystemExit(main())
