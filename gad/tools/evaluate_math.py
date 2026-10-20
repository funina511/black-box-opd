#!/usr/bin/env python3
"""External math evaluation for the initial student and every stage product.

Two protocols, both parameterised (plan sections 3.4 and 5):

* **dev monitor** - ``--n 1 --temperature 0`` on the held-out DAPO dev set: cheap,
  deterministic, used to compare initial student / SeqKD / Warmup / GAD;
* **paper protocol** - the CARD settings (``temperature 1.0``, ``top_p 0.95``,
  ``max_tokens 32768``, 16 samples per question, 2 for MATH-500). Per question we
  take the success *rate* over the samples and average over questions to estimate
  pass@1; "solved at least once" is reported separately as ``pass_at_n_any_success``
  and must never be called pass@1.

The scorer is the CARD-aligned one from ``tools/math_metrics.py``; it never feeds
back into training. Every generation is written to ``predictions.jsonl`` so a
report can be recomputed and audited without re-running the GPU job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PipelineError, ensure_dir, log, sha256_text, utc_timestamp, write_json  # noqa: E402
from math_metrics import load_scorer, score_response, summarize_sampling  # noqa: E402


def load_prompts(parquet_path: Path, *, prompt_key: str | None, benchmarks: set[str] | None,
                 limit: int) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = set(table.column_names)
    key = prompt_key if prompt_key and prompt_key in columns else (
        "prompt" if "prompt" in columns else ("content" if "content" in columns else None)
    )
    if key is None:
        raise PipelineError(f"{parquet_path} has no prompt column (columns: {sorted(columns)})")

    rows: list[dict] = []
    for row in table.to_pylist():
        data_source = str(row.get("data_source", "unknown"))
        benchmark = data_source[5:] if data_source.startswith("eval_") else data_source
        if benchmarks and benchmark not in benchmarks:
            continue
        rows.append(
            {
                "uid": str(row.get("extra_info", {}).get("index", len(rows))),
                "benchmark": benchmark,
                "messages": row[key],
                "ground_truth": str((row.get("reward_model") or {}).get("ground_truth", "")),
            }
        )
    if limit:
        rows = rows[:limit]
    return rows


def build_prompts(rows: list[dict], tokenizer, *, enable_thinking: bool | None) -> list[str]:
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    prompts = []
    for row in rows:
        try:
            prompts.append(
                tokenizer.apply_chat_template(
                    row["messages"], add_generation_prompt=True, tokenize=False, **kwargs
                )
            )
        except TypeError:
            # The model's template does not accept the switch: fall back and warn once.
            prompts.append(tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False))
    return prompts


def run_vllm(model: str, prompts: list[str], *, n_per_prompt: list[int], temperature: float, top_p: float,
             max_tokens: int, max_model_len: int, tp: int, gpu_memory_utilization: float,
             seed: int | None) -> list[list[dict]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        seed=seed,
    )
    results: list[list[dict]] = []
    # vLLM applies one SamplingParams per request, so group by sample count.
    order = sorted(range(len(prompts)), key=lambda i: n_per_prompt[i])
    grouped: dict[int, list[int]] = {}
    for index in order:
        grouped.setdefault(n_per_prompt[index], []).append(index)

    outputs_by_index: dict[int, list[dict]] = {}
    for n, indices in grouped.items():
        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        chunk = [prompts[i] for i in indices]
        outputs = llm.generate(chunk, params, use_tqdm=True)
        for index, output in zip(indices, outputs):
            samples = []
            for candidate in output.outputs:
                samples.append(
                    {
                        "response": candidate.text,
                        "finish_reason": getattr(candidate, "finish_reason", None),
                        "num_tokens": len(candidate.token_ids),
                    }
                )
            outputs_by_index[index] = samples
    for index in range(len(prompts)):
        results.append(outputs_by_index.get(index, []))
    return results


def run_hf(model: str, prompts: list[str], *, n_per_prompt: list[int], temperature: float, top_p: float,
           max_tokens: int, seed: int | None) -> list[list[dict]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    hf_model.eval()
    results: list[list[dict]] = []
    for prompt, samples in zip(prompts, n_per_prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        with torch.no_grad():
            generated = hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        entries = []
        for sequence in generated:
            new_tokens = sequence[inputs["input_ids"].shape[-1]:]
            entries.append(
                {
                    "response": tokenizer.decode(new_tokens, skip_special_tokens=True),
                    "finish_reason": "length" if len(new_tokens) >= max_tokens else "stop",
                    "num_tokens": int(len(new_tokens)),
                }
            )
        results.append(entries)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model directory (merged actor).")
    parser.add_argument("--parquet", required=True, help="Eval parquet (prompt + reward_model.ground_truth).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--benchmarks", default="", help="Comma-separated subset; empty = every benchmark present.")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--math500-n", type=int, default=None, help="Override samples/question for MATH-500.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--prompt-key", default=None)
    parser.add_argument("--enable-thinking", choices=["true", "false", "unset"], default="false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None, help="Report label, e.g. 'seqkd' or 'initial-student'.")
    parser.add_argument("--scorer-path", default=None)
    parser.add_argument("--scoring-config-json", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan; generate nothing.")
    args = parser.parse_args()
    if args.scoring_config_json:
        scoring_config = json.loads(args.scoring_config_json)
        if scoring_config.get('version') == 'math_unified_v1':
            os.environ["GAD_SCORING_MATH_CONFIG"] = args.scoring_config_json
            if not args.scorer_path:args.scorer_path = scoring_config.get('entrypoint')
        elif scoring_config.get('version') == 'legacy_card_v1':
            os.environ.pop('GAD_SCORING_MATH_CONFIG',None)
            if not args.scorer_path:args.scorer_path = str(Path(__file__).resolve().parents[2]/'sample/math_scorer.py')
        else:raise PipelineError('unsupported scoring config version')
    elif not args.scorer_path:
        inherited = json.loads(os.environ.get('GAD_SCORING_MATH_CONFIG','{}'))
        if inherited.get('version') == 'legacy_card_v1':
            args.scorer_path = str(Path(__file__).resolve().parents[2]/'sample/math_scorer.py')
            os.environ.pop('GAD_SCORING_MATH_CONFIG',None)
    # Resolve the scorer before GPU generation. A legacy config must never be
    # sent to the unified implementation after paying for all model outputs.
    scorer_module = load_scorer(args.scorer_path)
    requested = json.loads(args.scoring_config_json or '{}').get('version')
    actual = getattr(scorer_module,'VERSION','legacy_card_v1')
    if requested and requested != actual:
        raise PipelineError('selected scorer conflicts with requested scoring config version')
    if getattr(scorer_module,'VERSION',None) == 'math_unified_v1':
        effective = json.loads(os.environ.get('GAD_SCORING_MATH_CONFIG','{}'))
        if effective.get('version','math_unified_v1') != 'math_unified_v1':
            raise PipelineError('selected unified scorer conflicts with legacy scoring config')
        if effective.get('fingerprint') and effective['fingerprint'] != scorer_module.identity(effective.get('timeout_seconds',2.0))['fingerprint']:
            raise PipelineError('scorer fingerprint changed; create a new evaluation config before generation')

    parquet_path = Path(args.parquet).expanduser().resolve()
    if not parquet_path.exists():
        raise PipelineError(f"eval parquet not found: {parquet_path}")
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise PipelineError(f"model directory not found: {model_path}")

    benchmarks = {b.strip() for b in args.benchmarks.split(",") if b.strip()} or None
    rows = load_prompts(parquet_path, prompt_key=args.prompt_key, benchmarks=benchmarks, limit=args.limit)
    if not rows:
        raise PipelineError(f"no prompts selected from {parquet_path}")

    n_per_prompt = []
    for row in rows:
        n = args.n
        if args.math500_n and row["benchmark"] == "math500":
            n = args.math500_n
        n_per_prompt.append(max(1, int(n)))

    out_dir = ensure_dir(args.out_dir)
    plan = {
        "model": str(model_path),
        "parquet": str(parquet_path),
        "questions": len(rows),
        "benchmarks": sorted({row["benchmark"] for row in rows}),
        "samples_total": sum(n_per_prompt),
        "n": args.n,
        "math500_n": args.math500_n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "backend": args.backend,
        "enable_thinking": args.enable_thinking,
    }
    log(f"plan: {json.dumps(plan, ensure_ascii=False)}")
    if args.dry_run:
        return 0

    enable_thinking = None if args.enable_thinking == "unset" else args.enable_thinking == "true"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    prompts = build_prompts(rows, tokenizer, enable_thinking=enable_thinking)

    if args.backend == "vllm":
        generated = run_vllm(
            str(model_path), prompts, n_per_prompt=n_per_prompt, temperature=args.temperature,
            top_p=args.top_p, max_tokens=args.max_tokens,
            max_model_len=args.max_model_len or (args.max_tokens + 4096), tp=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
        )
    else:
        generated = run_hf(
            str(model_path), prompts, n_per_prompt=n_per_prompt, temperature=args.temperature,
            top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed,
        )

    predictions_path = out_dir / "predictions.jsonl"
    scored_records: list[dict] = []
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row, samples in zip(rows, generated):
            for sample_index, sample in enumerate(samples):
                scored = score_response(sample["response"], row["ground_truth"], scorer_module=scorer_module)
                record = {
                    "uid": row["uid"],
                    "benchmark": row["benchmark"],
                    "sample_index": sample_index,
                    "ground_truth": row["ground_truth"],
                    "response": sample["response"],
                    "finish_reason": sample.get("finish_reason"),
                    "truncated": sample.get("finish_reason") == "length",
                    "num_tokens": sample.get("num_tokens"),
                    **scored,
                }
                scored_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    grouped: dict[str, list[dict]] = {}
    for record in scored_records:
        grouped.setdefault(record["benchmark"], []).append(record)
    if getattr(scorer_module, 'VERSION', None) == 'math_unified_v1':
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from evaluate.rescore import summarize
        from collections import defaultdict
        expected = defaultdict(list)
        sample_counts = {}
        for row, n in zip(rows, n_per_prompt):
            expected[row['benchmark']].append(row['uid']);sample_counts[row['benchmark']] = n
        report = {'benchmarks': {}, 'overall': {}}
        for benchmark in expected:
            report['benchmarks'][benchmark] = summarize(grouped.get(benchmark, []), expected_questions=expected[benchmark], n=sample_counts[benchmark])
        complete = all(item['evaluation_complete'] for item in report['benchmarks'].values())
        report['overall'] = {'evaluation_complete': complete, 'aggregation': 'benchmark macro mean',
            'pass_at_1': sum(item['pass_at_1'] for item in report['benchmarks'].values()) / len(expected) if complete else None}
    else:
        report = summarize_sampling(grouped, answer_token_limit=args.max_tokens)
    report["label"] = args.label or model_path.name
    report["protocol"] = plan
    report["generated_at"] = utc_timestamp()
    write_json(out_dir / "metrics.json", report)

    write_json(
        out_dir / "manifest.json",
        {
            "label": args.label,
            "model": str(model_path),
            "parquet": str(parquet_path),
            "predictions": str(predictions_path),
            "protocol": plan,
            "prompt_template_hash": sha256_text(json.dumps(prompts[:1], ensure_ascii=False)),
            "scorer": str(getattr(scorer_module, "__gad_source__", "unknown")),
            "generated_at": utc_timestamp(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )

    log(f"predictions -> {predictions_path}")
    log(f"report      -> {out_dir / 'metrics.json'}")
    for benchmark, entry in report['benchmarks'].items():
        log(f"  {benchmark}: pass_at_1={entry.get('pass_at_1')}, evaluation_complete={entry.get('evaluation_complete', True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
