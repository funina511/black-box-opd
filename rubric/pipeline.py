#!/usr/bin/env python3
"""Small offline experiment entry point; GPU stages use isolated subprocesses."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubric.common import *

VARIANTS = ("baseline", "adv_only", "correctness_aux", "joint")
STAGES = ("preflight", "prepare", "collect", "annotate", "audit", "pack", "train", "evaluate", "report", "all")


def plan(cfg):
    run, dataset = paths(cfg)
    return {"schema_version": "rubric.plan.v1", "run": str(run), "dataset": str(dataset),
            "source": cfg["source"], "generation": generation_settings(cfg), "sampling": cfg["sampling"],
            "split": cfg["split"], "judge": {k:v for k,v in cfg["judge"].items() if k != "api_key_env"},
            "gpu_visibility": {"collect_train_evaluate": require_gpu(cfg), "judge": require_gpu(cfg, judge=True)
                               if cfg["judge"]["backend"] == "vllm" else None},
            "variants": list(VARIANTS), "training": cfg["training"],
            "strategy": "single_gpu, full responses, unique auxiliary labels",
            "audit_behavior": "all stops pending human review unless --exploratory is explicit"}


def preflight(cfg):
    import shutil
    from rubric.model import inspect_head
    result = plan(cfg)
    for key in ("source.run_config","source.actor_checkpoint","source.critic_checkpoint",
                "source.dataset_manifest","rubric.spec","judge.prompt"):
        if not Path(get(cfg,key)).exists():
            raise ValueError(f"missing {key}: {get(cfg,key)}")
    if cfg["judge"]["backend"] == "vllm" and not Path(cfg["judge"]["model"]).exists():
        raise ValueError("configured local judge model missing")
    if cfg["judge"]["backend"] == "api":
        for key in ("api_base_url","api_model","api_version"):
            if not cfg["judge"][key]:
                raise ValueError(f"explicit API {key} required")
    if cfg["judge"]["backend"] == "import" and not cfg["judge"]["annotations_path"]:
        raise ValueError("explicit judge.annotations_path required")
    result["source_head"] = inspect_head(cfg["source"]["critic_checkpoint"])
    gpu_query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    inventory = {line.split(",")[0].strip(): [int(item.strip()) for item in line.split(",")[1:]]
                 for line in gpu_query.stdout.splitlines()}
    requested = require_gpu(cfg, judge=cfg["judge"]["backend"] == "vllm").split(",")
    if any(gpu not in inventory for gpu in requested):
        raise ValueError("configured GPU does not exist")
    result["gpu_inventory_mib"] = {gpu: {"total":inventory[gpu][0],"used":inventory[gpu][1]} for gpu in requested}
    result["disk_free_gib"] = shutil.disk_usage(ROOT).free / 2**30
    run, _ = claim_run(cfg)
    write_json(run / "preflight.json", result)
    return result


def report(cfg):
    run, dataset = paths(cfg)
    audit = read_json(dataset / "audit/summary.json") if (dataset / "audit/summary.json").exists() else None
    manifest = read_json(dataset / "manifest.json") if (dataset / "manifest.json").exists() else None
    variants = {}
    for variant in VARIANTS:
        path = run / "variants" / variant / "test.metrics.json"
        variants[variant] = read_json(path) if path.exists() else {"status": "not_evaluated"}
    complete = all("metrics" in result for result in variants.values())
    result = {"run": str(run), "dataset": str(dataset), "sample_size": cfg["sampling"]["prompt_count"],
              "audit": audit, "data": manifest, "variants": variants, "all_variants_evaluated": complete,
              "conclusion": ("engineering smoke completed; label quality pending human verification; no efficacy claim"
                             if complete and (not audit or not audit["quality_gate_passed"])
                             else "offline engineering experiment completed; larger validation required"
                             if complete else "incomplete; inspect per-stage status and missing variant evaluations"),
              "missing_evaluations": [key for key,value in variants.items() if "metrics" not in value]}
    pairs = read_jsonl(dataset / "groups.jsonl")
    annotations = {row["response_id"]:row for row in read_jsonl(dataset / "annotations.raw.jsonl")}
    conflicts = []
    correctness_groups = {}
    for group in pairs:
        teacher = annotations.get(group["teacher_id"])
        if teacher:
            for rid in group["student_ids"]:
                student = annotations.get(rid)
                if student:
                    category = f"teacher_{int(bool(teacher['raw_scorer_result']['acc']))}_student_{int(bool(student['raw_scorer_result']['acc']))}"
                    correctness_groups[category] = correctness_groups.get(category,0) + 1
                if student and not teacher["raw_scorer_result"]["acc"] and student["raw_scorer_result"]["acc"]:
                    conflicts.append({"uid":group["uid"],"teacher_id":group["teacher_id"],"student_id":rid,
                                      "split":group["split"]})
    write_jsonl(run / "failure_examples.jsonl", conflicts)
    result["teacher_wrong_student_right_pairs_in_full_pool"] = len(conflicts)
    result["full_pool_pair_correctness_strata"] = correctness_groups
    result["failure_examples"] = str(run / "failure_examples.jsonl")
    if audit:
        result["limitations"] = ["human label quality unverified" if not audit["quality_gate_passed"] else "one judge only",
                                 f"{audit['truncated_responses']} truncated responses have masked process labels",
                                 "whitespace duplicate exclusion cannot rule out semantic paraphrases",
                                 "24-question smoke is not an efficacy experiment" if cfg["sampling"]["prompt_count"] == 24 else
                                 "single seed; further training seeds and independent validation required"]
    if complete:
        from rubric.evaluate import paired_deltas
        predictions = {variant:read_jsonl(run/"variants"/variant/"test.predictions.jsonl") for variant in VARIANTS}
        result["paired_comparisons"] = {
            f"{candidate}_minus_{reference}":paired_deltas(predictions[candidate],predictions[reference],
                                                          cfg["training"]["bootstrap_samples"],cfg["run"]["seed"])
            for candidate,reference in (("adv_only","baseline"),("correctness_aux","adv_only"),
                                         ("joint","adv_only"),("joint","correctness_aux"))}
    write_json(run / "report.json", result)
    lines = ["# Rubric offline experiment", "", result["conclusion"], "",
             f"Questions: {cfg['sampling']['prompt_count']}; run: {run.name}", "",
             "| Variant | d_acc | Quality ordering | Best-of-2 | Oracle |",
             "| --- | --- | --- | --- | --- |"]
    for variant, item in variants.items():
        m = item.get("metrics", {})
        lines.append("| " + " | ".join([variant] + [str(m.get(k,"pending")) for k in
                                                    ("d_acc","quality_ordering","best_of_2","oracle")]) + " |")
    lines += ["", "Intervals, denominators, strata, process metrics and human-reviewed metrics: report.json.",
              "This evaluates a fixed response pool. The actor was not updated.",
              "Source Warmup has already seen DAPO train questions. This is auxiliary-training holdout only."]
    atomic_text(run / "report.md", "\n".join(lines) + "\n")
    return {"report": str(run / "report.md"), "conclusion": result["conclusion"],
            "missing_evaluations": result["missing_evaluations"]}


def execute_stage(cfg, stage, args):
    if stage == "preflight":
        return preflight(cfg)
    if stage == "prepare":
        from rubric.data import prepare
        run, dataset = prepare(cfg)
        return {"run": str(run), "dataset": str(dataset)}
    if stage == "collect":
        from rubric.collect import collect
        return collect(cfg)
    if stage == "annotate":
        from rubric.annotate import annotate
        return annotate(cfg)
    if stage == "audit":
        from rubric.audit import audit
        return audit(cfg)
    if stage == "pack":
        from rubric.data import pack
        pack(cfg, args.exploratory)
        return {"packed": True}
    if stage == "train":
        from rubric.train_offline import train
        return train(cfg, args.variant, args.resume, args.max_steps, args.exploratory)
    if stage == "evaluate":
        from rubric.evaluate import evaluate
        return evaluate(cfg, args.variant, args.split)
    return report(cfg)


def launch(cfg, stage, args, variant=None):
    run, _ = claim_run(cfg)
    command = [sys.executable, str(Path(__file__).resolve()), "--config", str(run / "config.resolved.yaml"),
               "--stage", stage, "--worker", "--variant", variant or args.variant, "--split", args.split]
    if args.exploratory:
        command.append("--exploratory")
    if stage == "train" and args.resume:
        command += ["--resume", args.resume]
    if stage == "train" and args.max_steps:
        command += ["--max-steps", str(args.max_steps)]
    env = dict(os.environ)
    if stage in ("collect","train","evaluate","annotate"):
        env["CUDA_VISIBLE_DEVICES"] = require_gpu(cfg, judge=stage == "annotate" and cfg["judge"]["backend"] == "vllm")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    finally:
        # The new process group belongs solely to this stage, including vLLM
        # workers. Reap residual GPU children on normal exit and interruption.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "rubric/configs/offline_v1.yaml"))
    parser.add_argument("--stage", choices=STAGES, default="preflight")
    parser.add_argument("--size", choices=("smoke","pilot"))
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--split", choices=("dev","test"), default="test")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--max-steps", type=int, help="limit this training invocation; checkpoint remains resumable")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set, args.size)
    args.variant = args.variant or cfg["training"]["mode"]
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "variant": args.variant, **plan(cfg)}, ensure_ascii=False, indent=2))
        return
    run, dataset = claim_run(cfg)
    if args.stage == "all":
        for stage in ("preflight","collect","annotate","audit"):
            launch(cfg, stage, args)
        quality = read_json(dataset / "audit/summary.json")
        if not quality["quality_gate_passed"] and not args.exploratory:
            print("[rubric] pending human review: " + quality["review_template"], flush=True)
            report(cfg)
            return
        launch(cfg, "pack", args)
        for variant in VARIANTS:
            if variant != "baseline":
                launch(cfg, "train", args, variant)
                selection_path = run / "variants" / variant / "selection.json"
                if not selection_path.exists() or not read_json(selection_path)["training_complete"]:
                    print("[rubric] training limit reached; resume before test evaluation")
                    report(cfg)
                    return
            launch(cfg, "evaluate", args, variant)
        print(json.dumps(report(cfg), ensure_ascii=False))
    elif not args.worker and args.stage in ("collect","annotate","train","evaluate"):
        launch(cfg, args.stage, args)
    else:
        started = time.monotonic()
        version = snapshot_code(run)
        try:
            result = execute_stage(cfg, args.stage, args)
        except Exception as exc:
            append_jsonl(run / "stages.jsonl", {"stage": args.stage, "variant": args.variant,
                                              "seconds": time.monotonic()-started, "success": False,
                                              "code_version": version,
                                              "error": f"{type(exc).__name__}: {exc}"})
            raise
        append_jsonl(run / "stages.jsonl", {"stage": args.stage, "variant": args.variant,
                                          "code_version": version,
                                          "seconds": time.monotonic()-started, "success": True})
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
