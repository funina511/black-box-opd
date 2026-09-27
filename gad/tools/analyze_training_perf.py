#!/usr/bin/env python3
"""Summarize stage training and validation timing without double counting timers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"(?:^|\s)step:(\d+)\s+-")
FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_./-]*):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
TIMERS = ("step", "gen", "generate_sequences", "reshard", "update_critic",
          "update_actor", "ref", "reward", "old_log_prob", "testing", "save_checkpoint")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_steps(log: Path) -> list[dict]:
    records: dict[int, dict] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", line).replace("\r", "")
        matched = STEP.search(line)
        if not matched or "timing_s/step:" not in line:
            continue
        step = int(matched.group(1))
        fields = {key: float(value) for key, value in FIELD.findall(line)}
        if int(fields.get("training/global_step", step)) != step:
            continue
        records[step] = {"step": step, "fields": fields,
                         "kind": "validation" if "timing_s/testing" in fields else
                                 "save" if "timing_s/save_checkpoint" in fields else "normal"}
    return [records[key] for key in sorted(records)]


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(values), "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p90": round(ordered[min(len(ordered)-1, int((len(ordered)-1)*0.9 + 0.5))], 3),
            "max": round(max(values), 3)}


def summarize_steps(records: list[dict], *, stage: str, last_n: int | None = None,
                    skip_first: int = 0, resume_step: int | None = None,
                    prompt_count: int | None = None, candidate_count: int | None = None) -> dict:
    selected = [record for record in records if resume_step is None or record["step"] > resume_step]
    if skip_first:
        selected = selected[skip_first:]
    if last_n is not None:
        selected = selected[-last_n:]
    by_kind = {kind: [record for record in selected if record["kind"] == kind]
               for kind in ("normal", "save", "validation")}
    fields = {}
    for kind, group in by_kind.items():
        fields[kind] = {timer: summary([r["fields"][f"timing_s/{timer}"] for r in group
                                         if f"timing_s/{timer}" in r["fields"]]) for timer in TIMERS}
    gen_rates = []
    lengths = []
    clip = []
    memory = []
    # Match throughput and response statistics to the same normal steps used
    # for training timers; the final checkpoint step is accounted for separately.
    for record in by_kind["normal"]:
        f = record["fields"]
        if "response_length/mean" in f:
            lengths.append(f["response_length/mean"])
        if "response_length/clip_ratio" in f:
            clip.append(f["response_length/clip_ratio"])
        if "perf/max_memory_allocated_gb" in f:
            memory.append(f["perf/max_memory_allocated_gb"])
        if "timing_s/generate_sequences" in f and f.get("response_length/mean") is not None:
            # Batch and rollout size are read from the run config by the caller.
            gen_rates.append(f["response_length/mean"] / f["timing_s/generate_sequences"])
    generated_tokens_per_sec = (
        [rate * prompt_count * candidate_count for rate in gen_rates]
        if prompt_count and candidate_count else []
    )
    return {"stage": stage, "selected_steps": [r["step"] for r in selected],
            "prompt_count_per_step": prompt_count, "candidate_count_per_prompt": candidate_count,
            "kind_counts": {kind: len(group) for kind, group in by_kind.items()},
            "timing_seconds": fields,
            "response_length_mean_tokens": summary(lengths),
            "response_truncated_fraction": summary(clip),
            "generated_tokens_per_sequence_per_second": summary(gen_rates),
            "estimated_generated_tokens_per_second": summary(generated_tokens_per_sec),
            "reported_distributed_max_memory_allocated_gb": summary(memory),
            "note": "timing_s/gen contains generate_sequences and reshard; do not sum these timers"}


def read_validation(path: Path) -> dict:
    groups: dict[str, list[dict]] = {"quick": [], "full": []}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            groups.setdefault(row.get("validation_scope") or "legacy_full", []).append(row)
    result = {}
    for scope, rows in groups.items():
        result[scope] = {"count": len(rows), "steps": [r.get("global_step") for r in rows],
                         "sample_count": summary([float(r["count"]) for r in rows if "count" in r]),
                         "timing_seconds": {key: summary([float(r["timing_seconds"][key]) for r in rows
                                                          if key in r.get("timing_seconds", {})])
                                            for key in ("generation", "decode", "rouge_l", "math_scoring", "total")}}
    return result


def render_markdown(payload: dict) -> str:
    steps = payload["training"]
    normal = steps["timing_seconds"]["normal"]
    lines = [f"# {payload['stage']} performance: {payload['label']}", "",
             f"Source: `{payload['source_log']}`; selected steps: {steps['selected_steps']}.", "",
             "| Timer | Mean s | Median s | P90 s | Max s |", "|---|---:|---:|---:|---:|"]
    for timer in TIMERS:
        item = normal[timer]
        if item:
            lines.append(f"| {timer} | {item['mean']} | {item['median']} | {item['p90']} | {item['max']} |")
    lines += ["", "`gen` contains `generate_sequences` and `reshard`; rows are not additive.",
              "The reported distributed memory value is not a per-GPU nvidia-smi peak.", ""]
    for key in ("response_length_mean_tokens", "response_truncated_fraction",
                "generated_tokens_per_sequence_per_second",
                "estimated_generated_tokens_per_second"):
        lines.append(f"- {key}: {steps[key]}")
    lines += ["", "## Validation", "", "```json", json.dumps(payload["validation"], indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("seqkd", "warmup", "gad"), required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    log = args.log or run_dir / "logs" / f"{args.stage}.log"
    if not log.is_file():
        parser.error(f"log does not exist: {log}")
    records = parse_steps(log)
    if not records:
        parser.error(f"no completed training steps found in {log}")
    config_pointer = run_dir / "current_resolved_config.txt"
    config_path = Path(config_pointer.read_text().strip()) if config_pointer.is_file() else run_dir / "resolved_config.yaml"
    import yaml
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    batch = (config.get("data") or {}).get("train_batch_size")
    rollout = config.get("rollout") or {}
    candidates = rollout.get("n") if args.stage == "gad" else rollout.get(f"n_{args.stage}")
    gpu_count = (config.get("resources") or {}).get("n_gpus_per_node")
    backend = Path(__file__).with_name("backend_patches.py")
    payload = {"label": args.label, "stage": args.stage, "created_at": datetime.now(timezone.utc).isoformat(),
               "run_id": run_dir.name, "source_log": str(log), "source_log_sha256": sha256(log),
               "config_snapshot": str(config_path) if config_path.exists() else None,
               "config_sha256": sha256(config_path) if config_path.is_file() else None,
               "backend_patch_source_sha256": sha256(backend),
               "configured_gpu_count": gpu_count,
               "latest_checkpoint": (run_dir / args.stage / "latest_checkpointed_iteration.txt").read_text().strip()
               if (run_dir / args.stage / "latest_checkpointed_iteration.txt").is_file() else None,
               "training": summarize_steps(records, stage=args.stage, last_n=args.last_n,
                                           skip_first=args.skip_first, resume_step=args.resume_step,
                                           prompt_count=int(batch) if batch else None,
                                           candidate_count=int(candidates) if candidates else None),
               "validation": read_validation(run_dir / args.stage / "validation_math_metrics.jsonl")}
    out_dir = args.out_dir or run_dir / "reports/performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{args.stage}_{args.label}_{stamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(payload))
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
