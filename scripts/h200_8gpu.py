#!/usr/bin/env python3
"""Launch the n=8, DAPO-17k series on one H200 node, using the existing trainers."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "gad/tools"
sys.path.insert(0, str(TOOLS))
from config import compose_config, load_run_config, validate_config

JUDGE_MODELS = {
    "30b-instruct": "Qwen3-30B-A3B-Instruct-2507",
    "8b-thinking": "Qwen3-8B",
}


def check_judge_source(source, preset, parser):
    expected = JUDGE_MODELS[preset]
    if source.get("rubric.judge.model") != expected or source.get("rubric.judge.enable_thinking") != (preset == "8b-thinking"):
        parser.error(f"source run uses a different Judge from {expected}; select the matching run/model")


def judge_fingerprint(config):
    from extension_config import service_identity
    from rubric.core.contracts import digest
    return digest(service_identity(config))


def recorded_judge(run, config):
    record = run / "rubric_judge_identity.txt"
    # Early 8B runs predate this launcher record. Their trainer still verifies
    # the frozen service identity in the checkpoint during resume.
    return record.read_text().strip() if record.is_file() else judge_fingerprint(config)


def gpu_list(value):
    ids = value.split(",")
    if any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("expected comma-separated distinct GPU indices")
    return ids


def actor_options(report, naive_dir):
    selection = json.loads(report.read_text(encoding="utf-8"))
    if Path(selection["source"]["run"]).resolve() != naive_dir or selection["protocol"] != "gad_n8_selection_v1":
        raise ValueError(f"expected n=8 selection from {naive_dir}")
    choice = selection["recommendation"]
    if not choice:
        raise ValueError("no eligible Actor reward was selected; inspect the selection report")
    print(f"Confirmed {choice['components']} w={choice['weight']} tau={choice['tau']}, "
          f"review_pass={choice['review_pass']}, late_pass={choice['late_pass']}", flush=True)
    opts = {
        "rubric.actor_reward_group_weight": choice["weight"],
        "rubric.actor_reward_rubric_std_floor": choice["tau"],
        "rubric.actor_reward_selection_confirmed": True,
    }
    for dim, weight in choice["options"]["actor_reward_weights"].items():
        opts[f"rubric.actor_reward_weights.{dim}"] = weight
    return opts


def run_pipeline(cmd, run, stages, args):
    if args.dry_run:
        subprocess.run([*cmd, "--dry-run"], check=True)
        return
    with (run / ".node-training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"training already active for {run}") from None
        subprocess.run(cmd, check=True, pass_fds=(lock.fileno(),))
        if args.smoke:
            for stage in stages.split(","):
                marker = run / stage / "COMPLETED.json"
                if not marker.is_file():
                    raise RuntimeError(f"bounded {stage} run exited before its target; inspect {run}/stage_{stage}.json")
                print(f"{stage} completed at step {json.loads(marker.read_text())['verified']['last_step']}")


def launch(args, parser):
    root = args.runs_root.expanduser().resolve()
    if args.smoke:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.tag):
            parser.error("--tag must be a single directory name")
        root = root / "gray" / args.tag
    root = root / f"e{args.epochs}"
    method = args.method
    # Keep prior 8B experiment paths resumable while keeping the new labels and
    # Actor reward selection in their own run family.
    suffix = "_30b_instruct" if args.judge_model == "30b-instruct" else ""
    naive = root / ("naive" + suffix)
    run = root / (method + suffix if method in ("naive", "actor", "judge") else method)
    selection = naive / "reports/actor_reward_selection"
    py = sys.executable
    if args.warmup_only and (method not in ("base", "correctness", "naive") or args.warmup_source):
        parser.error("--warmup-only is for the first stage of base, correctness or naive")
    if method == "select":
        if args.smoke or args.resume or args.dry_run or args.write_config:
            parser.error("select operates on completed formal naive trajectories")
        check_judge_source(load_run_config(naive), args.judge_model, parser)
        subprocess.run([py, TOOLS / "select_actor_reward.py", "--run", naive, "--output", selection], check=True)
        return

    stages = {"seqkd": "seqkd", "base": "warmup,gad", "correctness": "warmup,gad",
              "naive": "warmup,gad", "actor": "gad"}
    if args.warmup_only:
        stages[method] = "warmup"
    if args.resume:
        if args.write_config:
            parser.error("--write-config materializes a new recipe; --resume uses the existing frozen config")
        if method == "rlvr":
            cmd = [py, TOOLS / "train_rlvr.py", "--output", run, "--resume"]
        elif method in stages:
            frozen = load_run_config(run)
            if method in ("naive", "actor"):
                check_judge_source(frozen, args.judge_model, parser)
                if recorded_judge(run, frozen) != judge_fingerprint(frozen):
                    parser.error("the current Judge differs from the service identity saved at training launch")
            stage = "gad" if frozen.get("budget.init_from_warmup_run") else stages[method]
            cmd = [py, TOOLS / "pipeline.py", "--run", run, "--resume", "--stages", stage]
            return run_pipeline(cmd, run, stage, args)
        else:
            parser.error("--resume applies to a training run")
        subprocess.run(cmd + (["--dry-run"] if args.dry_run else []), check=True)
        return

    joint = method in ("naive", "actor", "judge")
    if joint and set(args.train_gpus) & set(args.judge_gpus):
        parser.error("assign separate --train-gpus and --judge-gpus for this node recipe")
    overlays = []
    if joint or method == "correctness":
        label = "naive" if joint else "correctness_only"
        overlays.append(ROOT / f"configs/overrides/h200_{label}_n8.yaml")
    if method == "actor":
        overlays.append(ROOT / "configs/overrides/h200_actor_n8.yaml")
    cfg = compose_config(ROOT / "configs/h200_baseline_n8.yaml", overrides=overlays)
    model_root = args.model_root.expanduser().resolve()
    total = 271 * args.epochs
    cfg.data.update({
        "run.run_id": run.name, "run.output_root": str(root),
        "models.student": str(model_root / "Qwen3-0.6B"),
        "rubric.judge.tokenizer_path": str(model_root / "Qwen3-8B"),
        "resources.n_gpus_per_node": len(args.train_gpus),
        "resources.gpus": ",".join(args.train_gpus),
        "resources.judge_gpus": ",".join(args.judge_gpus),
        "resources.allow_judge_gpu_sharing": False,
        "rubric.judge.tensor_parallel_size": len(args.judge_gpus),
        "rubric.judge.gpu_memory_utilization": args.judge_memory,
        "rollout.gpu_memory_utilization": args.kv, "rollout.max_num_seqs": args.max_seqs,
        "optim.critic_micro_batch_size_per_gpu": args.critic_micro,
        "budget.stage_epochs.seqkd": args.epochs, "budget.stage_epochs.gad": args.epochs,
        "budget.seqkd_steps": total, "budget.gad_steps": total - 271,
        "budget.gad_new_steps": total - 271, "budget.rlvr_steps": total,
        "budget.rlvr_eval_steps": list(range(271, total + 1, 271)),
    })
    if joint:
        model = JUDGE_MODELS[args.judge_model]
        cfg.data.update({"rubric.judge.model": model, "rubric.judge.tokenizer_path": str(model_root / model)})
        if args.judge_model == "30b-instruct":
            cfg.data.update({
                "rubric.judge.enable_thinking": False, "rubric.judge.reasoning_parser": None,
                "rubric.judge.base_url": "http://127.0.0.1:8153/v1",
                "rubric.prompt": str(ROOT / "rubric/prompts/judge_math_process_direct_v1_instruct.txt"),
            })
    if args.smoke:
        cfg.data.update({"run.engineering_smoke": True, "run.disable_validation": True,
                         "run.log_backend": "console", "optim.critic_warmup": 0,
                         "budget.stage_epochs.seqkd": None, "budget.seqkd_steps": 2,
                         "budget.stage_epochs.warmup": None, "budget.warmup_steps": 2,
                         "budget.stage_epochs.gad": None, "budget.gad_steps": 3,
                         "budget.gad_new_steps": 3})
    if args.warmup_source:
        if not args.smoke or method not in ("actor", "base", "correctness", "naive"):
            parser.error("--warmup-source is for GAD smoke runs")
        source_cfg = load_run_config(args.warmup_source)
        if source_cfg.get("rubric.enabled") and "warmup" in source_cfg.get("rubric.stages"):
            if method == "base":
                parser.error("base needs a single-head Warmup source; this run trained a four-head Critic")
            if joint and source_cfg.get("rubric.mode") == "joint":
                check_judge_source(source_cfg, args.judge_model, parser)
        stages[method] = "gad"
        cfg.data.update({"budget.init_from_warmup_run": str(args.warmup_source.expanduser().resolve()),
                         "budget.init_step": args.init_step})
    if method == "actor":
        if not args.warmup_source:
            cfg.data.update({"budget.init_from_warmup_run": str(naive),
                             "budget.init_step": 2 if args.smoke else 271})
        if args.smoke:
            cfg.data.update({"run.max_stage_steps": 3, "rubric.actor_reward_group_weight": 0.25,
                             "rubric.actor_reward_rubric_std_floor": 0.045992180705070496,
                             "rubric.actor_reward_weights.verified_progress": 0.,
                             "rubric.actor_reward_weights.unresolved_error_severity": 0.})
        else:
            if not args.confirm_selection:
                parser.error("first run select and inspect selection.json, then pass actor --confirm-selection")
            source_cfg = load_run_config(naive)
            check_judge_source(source_cfg, args.judge_model, parser)
            if recorded_judge(naive, source_cfg) != judge_fingerprint(cfg):
                parser.error("Actor Judge settings differ from the selected naive run")
            cfg.data.update(actor_options(selection / "selection.json", naive))
    problems = validate_config(cfg)
    if problems:
        parser.error("\n".join(problems))
    if args.write_config:
        output = args.write_config.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(cfg.data, handle, sort_keys=True)
        except FileExistsError:
            parser.error(f"{output} already exists; choose a new --write-config path")
        print(output)
        return
    if method != "judge" and not args.dry_run and run.exists():
        parser.error(f"{run} already exists; use --resume, a new smoke tag, or a different --runs-root")

    # Actual launches keep their resolved input (also for RLVR); plans and
    # service control need a config only for the lifetime of the subprocess.
    with tempfile.TemporaryDirectory(prefix="lmops-h200-node-") as temp:
        config_dir = Path(temp)
        if method != "judge" and not args.dry_run:
            config_dir = root / "_launch_configs"
            config_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=config_dir, prefix=f"{method}-", suffix=".yaml",
                                         mode="w", encoding="utf-8", delete=False) as handle:
            yaml.safe_dump(cfg.data, handle, sort_keys=True)
            config = handle.name
        if method == "judge":
            cmd = [py, TOOLS / "rubric_judge.py", "--config", config, "--action", args.action]
        elif method == "rlvr":
            cmd = [py, TOOLS / "train_rlvr.py", "--config", config, "--output", run]
            if args.smoke:
                cmd.append("--smoke")
        else:
            cmd = [py, TOOLS / "pipeline.py", "--config", config, "--stages", stages[method]]
            if not args.dry_run:
                fingerprint = judge_fingerprint(cfg) if joint else None
                run.mkdir(parents=True, exist_ok=False)
                (run / "current_resolved_config.txt").write_text(config + "\n", encoding="utf-8")
                if fingerprint:
                    (run / "rubric_judge_identity.txt").write_text(fingerprint + "\n", encoding="utf-8")
            return run_pipeline(cmd, run, stages[method], args)
        if args.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("method", choices=("seqkd", "base", "correctness", "rlvr", "naive", "select", "actor", "judge"))
    p.add_argument("--epochs", type=int, choices=(2, 3), default=2, help="total data passes")
    p.add_argument("--runs-root", type=Path, default=os.environ.get("H200_RUNS_ROOT", ROOT / "runs/h200_8gpu"))
    p.add_argument("--model-root", type=Path, default=ROOT / "assets/models")
    p.add_argument("--judge-model", choices=JUDGE_MODELS, default="30b-instruct",
                   help="online rubric Judge; 30B uses the same model release as the offline Teacher")
    p.add_argument("--train-gpus", type=gpu_list, default="0,1,2,3")
    p.add_argument("--judge-gpus", type=gpu_list, default="4", help="dedicated physical indices, TP = count")
    p.add_argument("--judge-memory", type=float, default=.70, help="vLLM memory fraction on dedicated Judge cards")
    p.add_argument("--kv", type=float, default=.40, help="student vLLM memory fraction")
    p.add_argument("--max-seqs", type=int, default=128)
    p.add_argument("--critic-micro", type=int, default=4)
    p.add_argument("--smoke", action="store_true", help="full batch/length, bounded engineering budgets")
    p.add_argument("--tag", default="baseline", help="independent smoke output directory label")
    p.add_argument("--warmup-only", action="store_true",
                   help="stop at completed Warmup, e.g. to run mature GAD comparisons before --resume")
    p.add_argument("--warmup-source", type=Path, help="read-only completed Warmup for a mature GAD smoke")
    p.add_argument("--init-step", type=int, default=271, help="warmup checkpoint for --warmup-source")
    p.add_argument("--action", default="health", choices=("start", "stop", "health", "logs", "restart"),
                   help="judge service action")
    p.add_argument("--confirm-selection", action="store_true", help="use this series' recommendation for formal Actor")
    preview = p.add_mutually_exclusive_group()
    preview.add_argument("--dry-run", action="store_true")
    preview.add_argument("--write-config", type=Path, help="write resolved YAML to a new file and exit")
    p.add_argument("--resume", action="store_true", help="resume using the run's frozen configuration")
    args = p.parse_args()
    os.environ.setdefault("VLLM_USE_V1", "1")
    launch(args, p)


if __name__ == "__main__":
    main()
