#!/usr/bin/env python3
"""Preflight: everything that must be true *before* a GPU is allocated.

Checks (each one is reported, never silently skipped):

1. the interpreter runs and can import torch / ray / vllm / transformers / hydra /
   omegaconf / datasets / pyarrow / rouge_score;
2. every requested stage backend exists, sits at the pinned commit, carries the
   engineering patches and resolves ``verl`` from its own worktree (a foreign
   ``verl`` on the path is reported even when it is shadowed);
3. GPUs are visible and match ``resources.n_gpus_per_node`` (``--allow-no-gpu``
   downgrades this to a warning for CPU-only planning runs);
4. model directories exist and look loadable (``config.json`` + tokenizer);
5. the vLLM engine can size its KV cache: one ``max_model_len`` sequence must fit in
   ``gpu_memory_utilization x device memory - weights - engine overhead`` (the engine
   sizes this pool once, at construction, and refuses to start when it does not fit);
6. train/dev parquet exist, are readable, contain the required columns and do
   not share uids or prompts (leak check on a bounded sample);
7. the teacher index exists, its sampling fingerprint matches the expectation in
   the config and it covers every training uid;
8. the eval parquet exists when evaluation is enabled;
9. the output root exists and is writable, with a disk-space warning threshold;
10. when ``--stage gad`` is given, the Warmup handoff artifacts are checked.

Exit code 0 means "no failures"; warnings are printed and recorded but do not
fail the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    TRAIN_STAGES,
    PipelineError,
    capture_command,
    environment_summary,
    free_disk_gib,
    git_rev_parse,
    kv_cache_gib,
    log,
    read_json,
    sha256_text,
    utc_timestamp,
    vllm_kv_budget,
    write_json,
)
from config import Config, load_config, validate_config, validate_paths, add_config_arguments, select_config  # noqa: E402
from pipeline import validation_metrics_expected  # noqa: E402

REQUIRED_PACKAGES = {
    "torch": "training backend",
    "ray": "distributed driver",
    "transformers": "model loading",
    "hydra": "config composition",
    "omegaconf": "config composition",
    "datasets": "parquet loading",
    "pyarrow": "parquet IO",
    "rouge_score": "the original validation metric",
}
OPTIONAL_PACKAGES = {"vllm": "rollout engine (required unless rollout.name is not vllm)", "wandb": "optional logger"}


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[dict] = []
        self.sections: dict = {}

    def fail(self, message: str) -> None:
        self.failures.append(message)
        log(f"FAIL: {message}")

    def warn(self, message: str, **extra) -> None:
        self.warnings.append({"message": message, **extra})
        log(f"WARN: {message}")

    def ok(self, message: str) -> None:
        log(f"ok  : {message}")

    def to_dict(self) -> dict:
        return {
            "created_at": utc_timestamp(),
            "environment": environment_summary(),
            "ok": not self.failures,
            "failures": self.failures,
            "warnings": self.warnings,
            "sections": self.sections,
        }


def check_interpreter(cfg: Config, report: Report, *, allow_no_gpu: bool) -> None:
    python_bin = str(cfg.require("run.python_bin"))
    if not Path(python_bin).exists():
        report.fail(f"run.python_bin does not exist: {python_bin}")
        return
    code, out, err = capture_command([python_bin, "-c", "import sys; print(sys.version.split()[0])"])
    if code != 0:
        report.fail(f"cannot run {python_bin}: {err}")
        return
    report.ok(f"interpreter {python_bin} (python {out})")
    report.sections["interpreter"] = {"python_bin": python_bin, "version": out}

    for package, why in REQUIRED_PACKAGES.items():
        code, _, err = capture_command([python_bin, "-c", f"import {package}"])
        if code != 0:
            report.fail(f"missing required package '{package}' ({why}) in {python_bin}: {err.splitlines()[-1] if err else ''}")
    for package, why in OPTIONAL_PACKAGES.items():
        code, _, _ = capture_command([python_bin, "-c", f"import {package}"])
        if code != 0:
            report.warn(f"optional package '{package}' not importable ({why})")

    gpu_code = (
        "import torch;"
        "print(torch.cuda.device_count());"
        "print(torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'none')"
    )
    code, out, err = capture_command([python_bin, "-c", gpu_code])
    if code != 0:
        if allow_no_gpu:
            report.warn(f"could not query CUDA (--allow-no-gpu): {err.splitlines()[-1] if err else ''}")
        else:
            report.fail(f"could not query CUDA: {err.splitlines()[-1] if err else ''}")
        report.sections["gpu"] = {"available": False}
        return
    lines = out.splitlines()
    count = int(lines[0]) if lines and lines[0].isdigit() else 0
    report.sections["gpu"] = {"available": count > 0, "count": count, "name": lines[1] if len(lines) > 1 else None}
    expected = int(cfg.get("resources.n_gpus_per_node"))
    if count == 0:
        message = "no CUDA devices visible"
        report.warn(message) if allow_no_gpu else report.fail(message)
    elif count < expected:
        report.fail(f"requested {expected} GPUs per node but only {count} are visible")
    else:
        report.ok(f"{count} CUDA device(s) visible: {lines[1] if len(lines) > 1 else ''}")

    expected_verl = None
    code, out, err = capture_command([python_bin, "-c", "import verl, pathlib; print(pathlib.Path(verl.__file__).resolve())"])
    if code == 0:
        expected_verl = out
        report.warn(
            f"a verl is importable without a stage PYTHONPATH: {out}. Stages pin PYTHONPATH to their own "
            "worktree and the import guard aborts if it resolves elsewhere."
        )
    report.sections["installed_verl"] = expected_verl


def check_backends(cfg: Config, report: Report, stages: list[str]) -> dict:
    results = {}
    manifest_path = Path(cfg.require("backends.root")) / "backend_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for stage in stages:
        backend = Path(cfg.require("backends.root")) / stage
        entry = {"path": str(backend)}
        problems: list[str] = []
        if not (backend / "verl").is_dir():
            problems.append(f"{backend} is not a verl worktree; run tools/prepare_backends.py")
        else:
            head = git_rev_parse(backend, "HEAD")
            entry["commit"] = head
            pinned = (manifest.get("backends", {}).get(stage) or {}).get("requested_commit")
            if pinned and head != pinned:
                problems.append(f"{stage} worktree is at {head[:10]} but the manifest pins {pinned[:10]}")
        if (backend / "verl").is_dir():
            from backend_patches import verify_patches

            ok, patch_problems = verify_patches(backend)
            entry["patches_ok"] = ok
            entry["patch_problems"] = patch_problems
            problems.extend(patch_problems)
            if cfg.get("backends.profile") == "gad_extensions_v1" and stage in ("warmup", "gad"):
                try:
                    from algorithm_patches import verify, code_hash
                    algorithm = verify(backend)
                    if algorithm["extension_code_hash"] != code_hash():
                        raise ValueError("backend extension source is stale")
                    entry["algorithm"] = algorithm
                except (ValueError, RuntimeError, OSError) as exc:
                    problems.append(f"algorithm profile validation failed: {exc}")
        results[stage] = {**entry, "problems": problems}
        if problems:
            for problem in problems:
                report.fail(f"[{stage}] {problem}")
        else:
            report.ok(f"[{stage}] backend {backend} @ {entry.get('commit', '?')[:10]} with patches verified")
    report.sections["backends"] = results
    return results


def check_models(cfg: Config, report: Report) -> None:
    models = {}
    for key in ("models.student", "models.discriminator_init"):
        path = Path(str(cfg.get(key) or "")).expanduser()
        entry = {"path": str(path)}
        problems = []
        if not path.is_dir():
            problems.append(f"{key}: not a directory: {path}")
        else:
            if not (path / "config.json").exists():
                problems.append(f"{key}: config.json missing in {path}")
            tokenizer_hits = [
                name for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model")
                if (path / name).exists()
            ]
            entry["tokenizer_files"] = tokenizer_hits
            if not tokenizer_hits:
                problems.append(f"{key}: no tokenizer files in {path}")
            weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
            entry["weight_files"] = len(weights)
            if not weights:
                problems.append(f"{key}: no weight files in {path}")
        entry["problems"] = problems
        models[key] = entry
        for problem in problems:
            report.fail(problem)
    if not any(models.get(key, {}).get("problems") for key in models):
        report.ok("model directories look loadable")
    report.sections["models"] = models


def check_vllm_kv_budget(cfg: Config, report: Report) -> None:
    """Predict vLLM's engine-construction KV-cache gate on CPU, before touching a GPU.

    This is the check that would have caught the ``seqkd`` failure at
    gpu_memory_utilization=0.30 ("1.09 GiB KV cache is needed ... larger than the
    available KV cache memory (0.95 GiB)"): the pool must hold one full
    ``max_model_len`` sequence and is sized once, at engine construction.
    """
    entry: dict = {"checked": False}
    report.sections["vllm_kv_budget"] = entry

    if str(cfg.get("rollout.name") or "").lower() != "vllm":
        entry["reason"] = f"rollout.name={cfg.get('rollout.name')!r} is not vllm"
        return

    model_dir = Path(str(cfg.get("models.student") or "")).expanduser()
    config_path = model_dir / "config.json"
    if not config_path.exists():
        entry["reason"] = f"{config_path} not found"
        return
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: BLE001
        entry["reason"] = f"cannot read {config_path}: {exc}"
        return

    weight_bytes = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    if not weight_bytes:
        weight_bytes = sum(f.stat().st_size for f in model_dir.glob("pytorch_model*.bin"))
    if not weight_bytes:
        entry["reason"] = f"no weight files under {model_dir} to size the engine"
        return

    length_budget = int(cfg.get("data.max_prompt_length") or 0) + int(cfg.get("data.max_response_length") or 0)
    max_model_len = int(cfg.get("rollout.max_model_len") or length_budget)
    if not max_model_len:
        entry["reason"] = "no length budget (data.max_prompt_length + data.max_response_length)"
        return

    try:
        need_gib = kv_cache_gib(model_config, max_model_len)
    except (KeyError, TypeError, ValueError) as exc:  # noqa: BLE001 - a partial config.json cannot be sized
        entry["reason"] = f"cannot size the KV cache from {config_path}: {exc}"
        report.warn(entry["reason"])
        return

    python_bin = str(cfg.require("run.python_bin"))
    gpu_ids = [part.strip() for part in str(cfg.get("resources.gpus") or "").split(",") if part.strip()]
    device_index = int(gpu_ids[0]) if gpu_ids and gpu_ids[0].isdigit() else 0
    probe = f"import torch;print(int(torch.cuda.get_device_properties({device_index}).total_memory))"
    code, out, err = capture_command([python_bin, "-c", probe])
    if code != 0 or not out.strip().isdigit():
        entry["reason"] = f"cannot query device memory: {(err or out).strip().splitlines()[-1] if (err or out).strip() else 'no output'}"
        report.warn(
            "could not check the vLLM KV-cache budget (device memory unavailable): "
            f"{entry['reason']}. The engine needs ~{need_gib:.2f} GiB of KV cache "
            f"for one {max_model_len}-token sequence."
        )
        return

    budget = vllm_kv_budget(
        model_config=model_config,
        max_model_len=max_model_len,
        gpu_memory_utilization=float(cfg.get("rollout.gpu_memory_utilization")),
        total_memory_gib=int(out.strip()) / (1024 ** 3),
        weight_gib=weight_bytes / (1024 ** 3),
    )
    entry.update({"checked": True, "gpu_index": device_index, **budget})
    utilization = float(cfg.get("rollout.gpu_memory_utilization"))
    if not budget["ok"]:
        report.fail(
            f"vLLM cannot size its KV cache: one {max_model_len}-token sequence needs "
            f"{budget['need_gib']:.2f} GiB but only {budget['available_gib']:.2f} GiB would be available at "
            f"rollout.gpu_memory_utilization={utilization} on GPU {device_index} ({budget['total_memory_gib']:.1f} GiB) "
            f"(weights {budget['weight_gib']:.2f} GiB + ~{budget['overhead_gib']:.2f} GiB engine overhead). "
            "Raise rollout.gpu_memory_utilization (the KV pool is discarded again before every actor update, so it "
            "does not compete with training memory) or lower data.max_response_length."
        )
    elif budget["tight"]:
        report.warn(
            f"vLLM KV cache will be tight: {budget['available_gib']:.2f} GiB available for a "
            f"{budget['need_gib']:.2f} GiB single-sequence requirement at gpu_memory_utilization={utilization} "
            f"on GPU {device_index}; generation will be serialised (expect slow rollouts)."
        )
    else:
        report.ok(
            f"vLLM KV budget: {budget['available_gib']:.2f} GiB available >= {budget['need_gib']:.2f} GiB needed "
            f"for one {max_model_len}-token sequence (gpu_memory_utilization={utilization}, GPU {device_index})"
        )


def check_data(cfg: Config, report: Report, *, sample_rows: int) -> dict:
    import pyarrow.parquet as pq

    def inspect(path_key: str, split: str) -> dict:
        path = Path(str(cfg.require(path_key))).expanduser()
        entry: dict = {"path": str(path)}
        if not path.exists():
            report.fail(f"{path_key} not found: {path}")
            return entry
        parquet = pq.ParquetFile(str(path))
        entry["rows"] = parquet.metadata.num_rows
        columns = set(parquet.schema_arrow.names)
        entry["columns"] = sorted(columns)
        prompt_key = str(cfg.get("data.prompt_key"))
        required = [prompt_key, "teacher_response", "reward_model", "extra_info"]
        missing = [name for name in required if name not in columns]
        if missing:
            report.fail(f"{split}: {path} is missing required column(s) {missing}")
        batch = next(parquet.iter_batches(batch_size=min(sample_rows, max(1, entry["rows"]))), None)
        if batch is not None:
            rows = batch.to_pylist()
            uids = [str(row["extra_info"]["index"]) for row in rows] if "extra_info" in columns else []
            empty_teacher = sum(1 for row in rows if not str(row.get("teacher_response") or "").strip())
            entry["sampled_rows"] = len(rows)
            entry["sampled_empty_teacher"] = empty_teacher
            entry["sampled_duplicate_uids"] = len(uids) - len(set(uids))
            if "teacher_response" in columns and empty_teacher:
                report.fail(f"{split}: {empty_teacher}/{len(rows)} sampled rows have an empty teacher_response")
            if uids and entry["sampled_duplicate_uids"]:
                report.fail(f"{split}: {entry['sampled_duplicate_uids']} duplicate uids in the sampled rows")
            entry["sample_uids"] = uids[:2000]
        return entry

    train = inspect("data.train_file", "train")
    val = inspect("data.val_file", "dev")
    overlap = len(set(train.get("sample_uids", [])) & set(val.get("sample_uids", [])))
    if overlap:
        report.fail(f"train/dev share {overlap} uids in the sampled window (leak)")
    else:
        report.ok(
            f"data ok: train={train.get('rows')} rows, dev={val.get('rows')} rows, no sampled uid overlap"
        )
    batch_size = int(cfg.require("data.train_batch_size"))
    if train.get("rows"):
        steps = train["rows"] // batch_size
        entry = {"steps_per_epoch": steps, "dropped_per_epoch": train["rows"] % batch_size}
        if steps < 2:
            report.warn(
                f"train dataloader yields only {steps} step(s) per epoch; "
                "the warmup actor-update gate is at step 10, so a smoke run needs several epochs"
            )
        report.sections["steps"] = entry
    report.sections["data"] = {"train": {k: v for k, v in train.items() if k != "sample_uids"},
                               "dev": {k: v for k, v in val.items() if k != "sample_uids"}}
    return {"train": train, "dev": val}


def check_teacher_index(cfg: Config, report: Report, data: dict) -> None:
    index_path = cfg.get("data.teacher_index")
    if not index_path:
        report.warn("data.teacher_index is not set; teacher fingerprint/coverage cannot be verified")
        return
    path = Path(str(index_path)).expanduser()
    if not path.exists():
        report.fail(f"data.teacher_index not found: {path}")
        return
    header = None
    first_record = None
    record_count = 0
    uids: set[str] = set()
    header_mismatch = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                report.fail(f"teacher index {path}:{line_number} is corrupt: {exc}")
                return
            if "_header" in row and header is None:
                header = row
                continue
            if first_record is None:
                first_record = row
            record_count += 1
            if "uid" in row:
                uids.add(str(row["uid"]))
            digest = row.get("teacher_fingerprint_digest")
            if header and digest and digest != header.get("teacher_fingerprint_digest"):
                header_mismatch += 1

    entry = {"path": str(path), "records": record_count, "fingerprint_digest": (header or {}).get(
        "teacher_fingerprint_digest"
    )}
    if header is None:
        report.warn(f"teacher index {path} has no header line; the sampling fingerprint cannot be confirmed")
    expected = cfg.get("data.expected_fingerprint_digest")
    if expected and entry["fingerprint_digest"] != expected:
        report.fail(
            f"teacher index fingerprint {entry['fingerprint_digest']!r} != data.expected_fingerprint_digest {expected!r}"
        )
    if header_mismatch:
        report.fail(f"{header_mismatch} teacher index records carry a different fingerprint than the header")

    thinking = ((header or {}).get("notes") or {}).get("enable_thinking")
    if isinstance(thinking, str):  # indexes written before this notes field became a bool
        thinking = {"true": True, "false": False}.get(thinking.strip().lower())
    configured = cfg.get("rollout.enable_thinking")
    if isinstance(thinking, bool) and bool(thinking) != bool(configured):
        report.fail(
            f"teacher index was sampled with enable_thinking={thinking} but training uses {bool(configured)}; "
            "teacher text and student prompts would use different templates"
        )
    elif isinstance(thinking, bool):
        report.ok(f"teacher index thinking mode matches training (enable_thinking={thinking})")

    train_uids = set(data["train"].get("sample_uids", []))
    if train_uids:
        missing = len(train_uids - uids)
        if missing:
            report.fail(f"{missing} of {len(train_uids)} sampled train uids are missing from the teacher index")
        else:
            report.ok(f"teacher index covers all {len(train_uids)} sampled train uids")
    report.sections["teacher_index"] = entry


def check_validation_metrics(cfg: Config, report: Report) -> None:
    """The training-validation math scorer must exist and agree with CARD fixtures."""
    expected, reason = validation_metrics_expected(cfg)
    path = cfg.get("validation.scorer_path")
    entry: dict = {"expected": bool(expected), "reason": reason, "scorer_path": str(path) if path else None}
    if not path:
        entry["status"] = "disabled"
        if cfg.get("validation.require_math_metrics"):
            report.warn(
                "no math scorer is configured: the stages will only report the original ROUGE-L "
                "validation metric, not the plan's dev-set accuracy"
            )
        report.sections["validation_metrics"] = entry
        return

    scorer = Path(str(path)).expanduser()
    entry["exists"] = scorer.exists()
    if not scorer.exists():
        report.fail(f"validation.scorer_path does not exist: {scorer}")
        report.sections["validation_metrics"] = entry
        return

    # Run the scorer's own CARD parity fixtures with the stage interpreter.
    code = (
        "import importlib.util, json;"
        f"spec = importlib.util.spec_from_file_location('gad_scorer', {str(scorer)!r});"
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module);"
        "print('@@SCORER@@' + json.dumps(module.compute_score('work\\nAnswer: 42', '42')))"
    )
    code_out, out, err = capture_command([str(cfg.require("run.python_bin")), "-c", code])
    entry["probe_ok"] = code_out == 0
    if code_out != 0:
        report.fail(f"could not import the math scorer with {cfg.get('run.python_bin')}: {err[:200]}")
    else:
        for line in out.splitlines():
            if line.startswith("@@SCORER@@"):
                entry["probe"] = json.loads(line[len("@@SCORER@@"):])
        probe = entry.get("probe") or {}
        if probe.get("acc") != 1 or float(probe.get("score", 0)) != 1.0:
            report.fail(f"the math scorer did not score a correct answer as correct: {probe}")
        else:
            report.ok(f"math scorer ready: {scorer} (CARD-aligned compute_score)")
    entry["status"] = "enabled" if expected else "configured-but-unused"
    report.sections["validation_metrics"] = entry


def check_quick_dev(cfg: Config, report: Report) -> None:
    if not cfg.get("validation.quick_enabled"):
        report.sections["quick_dev"] = {"enabled": False}
        return
    import hashlib
    import pyarrow.parquet as pq
    full_path = Path(str(cfg.require("data.val_file")))
    quick_path = Path(str(cfg.require("validation.quick_file")))
    manifest_path = quick_path.with_suffix(".manifest.json")
    entry = {"enabled": True, "full": str(full_path), "quick": str(quick_path)}
    if not quick_path.is_file() or not manifest_path.is_file():
        report.fail(f"quick validation file or manifest is missing: {quick_path}, {manifest_path}")
        report.sections["quick_dev"] = entry
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    full_uids = [str(item["extra_info"]["index"]) for item in pq.read_table(full_path, columns=["extra_info"]).to_pylist()]
    quick_uids = [str(item["extra_info"]["index"]) for item in pq.read_table(quick_path, columns=["extra_info"]).to_pylist()]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    entry.update({"full_rows": len(full_uids), "quick_rows": len(quick_uids),
                  "full_sha256": sha(full_path), "quick_sha256": sha(quick_path)})
    if len(quick_uids) != int(cfg.get("validation.quick_size")) or len(set(quick_uids)) != len(quick_uids):
        report.fail("quick validation size or UID uniqueness is wrong")
    if not set(quick_uids).issubset(full_uids) or quick_uids != manifest.get("uids"):
        report.fail("quick validation UIDs differ from full dev or manifest")
    if entry["full_sha256"] != manifest.get("full_sha256") or entry["quick_sha256"] != manifest.get("quick_sha256"):
        report.fail("quick validation parent/subset SHA256 differs from manifest")
    if not report.failures:
        report.ok(f"quick dev fixed subset: {len(quick_uids)}/{len(full_uids)} UIDs, hashes verified")
    report.sections["quick_dev"] = entry


def check_eval(cfg: Config, report: Report) -> None:
    if not cfg.get("eval.enabled"):
        report.sections["eval"] = {"enabled": False}
        return
    path = cfg.get("eval.parquet")
    if not path:
        report.fail("eval.enabled is true but eval.parquet is not set")
        return
    parquet_path = Path(str(path)).expanduser()
    if not parquet_path.exists():
        report.fail(f"eval parquet not found: {parquet_path}")
        return
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(parquet_path))
    benchmarks = sorted(
        {
            str(row["data_source"])
            for row in next(parquet.iter_batches(batch_size=min(5000, parquet.metadata.num_rows)), None).to_pylist()
        }
    )
    requested = {b.strip() for b in str(cfg.get("eval.benchmarks")).split(",") if b.strip()}
    present = {b[5:] if b.startswith("eval_") else b for b in benchmarks}
    missing = sorted(requested - present)
    entry = {"path": str(parquet_path), "rows": parquet.metadata.num_rows, "benchmarks": sorted(present),
             "missing_benchmarks": missing}
    if missing:
        report.fail(f"eval parquet is missing requested benchmark(s): {missing}")
    else:
        report.ok(f"eval parquet ok: {parquet.metadata.num_rows} rows across {sorted(present)}")
    report.sections["eval"] = entry


def check_teacher_pool(cfg: Config, report: Report) -> None:
    """Report the composition of the teacher pool that will actually be trained on.

    ``build_gad_data`` writes ``<parquet>.stats.json`` next to every pool with the
    selection rule and the truncation accounting. With ``gad_data.select: all`` (the
    default) the pool deliberately keeps answers that ran into the sampling cap; those
    answers have no final ``Answer:`` line, which affects both the SFT teacher text and
    the math scoring of that sample. That choice is fine, but it must be visible here
    instead of only in the README.
    """
    entry: dict = {}
    for key, split in (("data.train_file", "train"), ("data.val_file", "val")):
        parquet = Path(str(cfg.get(key) or "")).expanduser()
        stats_path = parquet.with_suffix(parquet.suffix + ".stats.json")
        if not stats_path.exists():
            continue
        stats = read_json(stats_path, default={}) or {}
        lengths = stats.get("teacher_token_length") or {}
        select = stats.get("select")
        truncated = int(lengths.get("truncated_before_final_answer") or 0)
        rate = float(lengths.get("truncated_before_final_answer_rate") or 0.0)
        entry[split] = {
            "select": select,
            "rows": stats.get("rows_out"),
            "stats": str(stats_path),
            "truncated_before_final_answer": truncated,
            "truncated_before_final_answer_rate": rate,
            "at_or_over_limit": lengths.get("at_or_over_limit"),
            "max_tokens": lengths.get("max_tokens"),
        }
        if truncated and select not in ("nontruncated", "correct-nontruncated"):
            report.warn(
                f"{split} pool uses select={select!r} and keeps {truncated} teacher answers that were cut "
                f"off by the {lengths.get('max_tokens')}-token sampling cap ({rate * 100:.1f}% of "
                f"{stats.get('rows_out')} rows; they have no final 'Answer:' line). That is a deliberate "
                "choice, but it weakens both the SFT teacher text and the math scoring for those samples; "
                "re-run LMOps/sample with gad_data.select='nontruncated'/'correct-nontruncated' (or a larger "
                "--max-tokens) if that was not intended."
            )
    report.sections["teacher_pool"] = entry


def check_output(cfg: Config, report: Report) -> None:
    root = Path(cfg.require("run.output_root")).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".gad_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        report.fail(f"output root is not writable: {root} ({exc})")
        return
    free = free_disk_gib(root)
    report.sections["output"] = {"root": str(root), "free_gib": free}
    if free < 50:
        report.warn(f"only {free:.1f} GiB free under {root}; checkpoints may not fit")
    else:
        report.ok(f"output root writable, {free:.1f} GiB free")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arguments(parser)
    parser.add_argument("--report", default=None, help="Where to write preflight_report.json.")
    parser.add_argument("--stages", default=None, help="Backends to check (default: all three).")
    parser.add_argument("--stage", default=None, help="Also require the inputs for this stage to be ready.")
    parser.add_argument("--allow-no-gpu", action="store_true", help="Downgrade the GPU check to a warning.")
    parser.add_argument("--skip-packages", action="store_true", help="Skip the import checks (fast planning runs).")
    parser.add_argument("--sample-rows", type=int, default=2000)
    args = parser.parse_args()

    cfg = select_config(args)
    stages = [s.strip() for s in (args.stages.split(",") if args.stages else TRAIN_STAGES) if s.strip()]
    for stage in stages:
        if stage not in TRAIN_STAGES:
            raise PipelineError(f"unknown stage {stage!r}; known: {TRAIN_STAGES}")

    report = Report()
    problems = validate_config(cfg)
    problems += validate_paths(cfg, check_exists=True)
    for problem in problems:
        report.fail(f"config: {problem}")

    check_output(cfg, report)
    check_backends(cfg, report, stages)
    check_models(cfg, report)
    check_vllm_kv_budget(cfg, report)
    data = check_data(cfg, report, sample_rows=args.sample_rows)
    check_teacher_index(cfg, report, data)
    check_teacher_pool(cfg, report)
    check_validation_metrics(cfg, report)
    check_quick_dev(cfg, report)
    check_eval(cfg, report)
    from extension_config import active, service_identity
    if any(active(cfg, "rubric", stage) for stage in stages):
        if cfg.get("rubric.mode") == "joint":
            try:
                from gad.extensions.judge_client import JudgeClient
                from rubric.core.contracts import digest
                options = {key[13:]: value for key, value in cfg.data.items() if key.startswith("rubric.judge.")}
                health = JudgeClient(options, digest(service_identity(cfg)), tokenizer=object()).health()
                if set(str(health.get("gpu_ids", "")).split(",")) & set(str(cfg.get("resources.gpus")).split(",")):
                    raise ValueError("running judge uses training GPUs")
                if set(str(health.get("gpu_ids", "")).split(",")) != set(str(cfg.get("resources.judge_gpus")).split(",")):
                    raise ValueError("running judge GPU allocation differs from declared resources.judge_gpus")
                report.sections["online_judge"] = health
                report.ok("online judge model/revision/template/schema/precision verified")
            except (ValueError, RuntimeError, OSError) as exc:
                report.fail(f"online judge: {exc}")
        else:
            report.sections["online_judge"] = {"mode": "correctness_only", "service_required": False}
    if not args.skip_packages:
        check_interpreter(cfg, report, allow_no_gpu=args.allow_no_gpu)

    if args.stage == "gad":
        warmup_dir = Path(cfg.require("run.output_root")) / str(cfg.get("run.run_id")) / "warmup"
        report.sections["gad_inputs"] = {
            "note": "the GAD stage additionally requires a completed Warmup checkpoint; "
                    "tools/prepare_gad_checkpoint.py --check-only verifies it at run time",
            "expected_warmup_dir": str(warmup_dir),
        }

    payload = report.to_dict()
    if args.report:
        report_path = Path(args.report).expanduser()
    elif cfg.existing_run:
        run_dir = Path(cfg.require("run.output_root")) / str(cfg.require("run.run_id"))
        report_path = run_dir / "reports" / f"preflight_{utc_timestamp().replace(':', '').replace('-', '')}.json"
    else:
        report_path = Path(cfg.require("run.output_root")) / "_preflight" / f"preflight_{utc_timestamp().replace(':', '').replace('-', '')}.json"
    write_json(report_path, payload)
    log(f"report -> {report_path}")

    if report.failures:
        log(f"preflight FAILED with {len(report.failures)} failure(s)")
        return 1
    log(f"preflight passed with {len(report.warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        log(f"FATAL: {exc}")
        raise SystemExit(2)
