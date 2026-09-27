#!/usr/bin/env python3
"""Compare isolated benchmarks on the same measured normal training steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from analyze_training_perf import parse_steps, summarize_steps

ALLOWED_CHANGES = {"run.run_id", "rollout.gpu_memory_utilization",
                   "optim.critic_dynamic_bsz", "optim.critic_micro_batch_size_per_gpu"}


def mean(training: dict, timer: str) -> float | None:
    value = training["timing_seconds"]["normal"].get(timer)
    return value["mean"] if value else None


def summary_mean(training: dict, key: str) -> float | None:
    value = training.get(key)
    return value["mean"] if value else None


def compare(reports: list[dict]) -> str:
    if not reports or reports[0]["candidate"] != "baseline":
        raise ValueError("the first result must be the baseline")
    baseline = reports[0]
    start = baseline["source_checkpoint"]
    source_step = baseline["source_step"]
    configs = [yaml.safe_load(Path(report["config"]).read_text()) for report in reports]
    normal_records = []
    for report, config in zip(reports, configs):
        name = report["candidate"]
        expected_steps = list(range(source_step + 1, report["target_step"] + 1))
        if not report["ok"] or report["completed_steps"] != expected_steps:
            raise ValueError(f"{name} did not complete its exact step budget")
        if report["source_checkpoint"] != start or report["source_step"] != source_step:
            raise ValueError(f"{name} uses a different source checkpoint")
        if ({key: value for key, value in config.items() if key not in ALLOWED_CHANGES}
                != {key: value for key, value in configs[0].items() if key not in ALLOWED_CHANGES}):
            raise ValueError(f"{name} changed settings outside the A/B performance knobs")
        if (not report.get("backend_file_sha256")
                or report.get("backend_commit") != baseline.get("backend_commit")
                or report["backend_file_sha256"] != baseline.get("backend_file_sha256")):
            raise ValueError(f"{name} used a different or unverified training backend")
        log = Path(report["config"]).parents[1] / "logs/warmup.log"
        rows = parse_steps(log)
        normal_records.append({row["step"]: row for row in rows
                               if row["kind"] == "normal"
                               and row["step"] > source_step + report["warmup_steps"]})

    common_steps = sorted(set.intersection(*(set(rows) for rows in normal_records)))
    if len(common_steps) < 2:
        raise ValueError("fewer than two common normal training steps remain for comparison")
    summaries = [summarize_steps([rows[step] for step in common_steps], stage="warmup",
                                 prompt_count=int(config["data.train_batch_size"]),
                                 candidate_count=int(config["rollout.n_warmup"]))
                 for rows, config in zip(normal_records, configs)]
    base_step = mean(summaries[0], "step")
    lines = ["# Warmup isolated benchmark comparison", "",
             f"Source: `{start}`. Common measured normal steps: {common_steps}. "
             "Each run also completed its own final checkpoint save.", "",
             "| Candidate | Updates | Normal steps | Step s | Speedup | Generate s | Tokens/s est. | "
             "Response tokens | Clipped | Critic s | Actor s | Final save s | Peak MiB/card |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for report, training in zip(reports, summaries):
        step = mean(training, "step")
        speed = f"{(base_step / step - 1) * 100:+.1f}%" if base_step and step else "n/a"
        peaks = report["gpu_peak_memory_mib_per_card"]
        peak_text = "/".join(f"{gpu}:{peaks[gpu]}" for gpu in sorted(peaks, key=int)) or "n/a"
        save = report["training"]["timing_seconds"]["save"]["save_checkpoint"]
        values = [report["candidate"], str(len(report["completed_steps"])),
                  str(training["kind_counts"]["normal"]),
                  f"{step:.1f}" if step is not None else "n/a", speed,
                  f"{mean(training, 'generate_sequences'):.1f}" if mean(training, "generate_sequences") is not None else "n/a",
                  f"{summary_mean(training, 'estimated_generated_tokens_per_second'):.1f}" if summary_mean(training, "estimated_generated_tokens_per_second") is not None else "n/a",
                  f"{summary_mean(training, 'response_length_mean_tokens'):.0f}" if summary_mean(training, "response_length_mean_tokens") is not None else "n/a",
                  f"{summary_mean(training, 'response_truncated_fraction'):.2%}" if summary_mean(training, "response_truncated_fraction") is not None else "n/a",
                  f"{mean(training, 'update_critic'):.1f}" if mean(training, "update_critic") is not None else "n/a",
                  f"{mean(training, 'update_actor'):.1f}" if mean(training, "update_actor") is not None else "n/a",
                  f"{save['mean']:.1f}" if save else "n/a", peak_text]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "`Generate s` is `timing_s/generate_sequences`, which is contained within "
              "`timing_s/gen`. Tokens/s uses each step's actual mean response length. "
              "Peak MiB/card is sampled by nvidia-smi every three seconds over each run's full "
              "duration; a shorter run may miss later memory peaks. Comparisons with only two "
              "normal steps are screening results and need confirmation before formal adoption.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="baseline result first, then candidates")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    content = compare([json.loads(path.read_text()) for path in args.results])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
        print(args.output)
    else:
        print(content)


if __name__ == "__main__":
    main()
