#!/usr/bin/env python3
"""CPU tests for the GAD pipeline tooling (no GPU, no torch, no vLLM).

Run::

    python LMOps/gad/tests/test_pipeline_cpu.py

Covered plan requirements:

* P0  - the pipeline dry-run must print the three stages with the right backends,
        budgets and dependency chain (SeqKD has no discriminator, Warmup starts
        from the student, GAD continues from the Warmup step);
* P0  - the backend worktrees must be at the pinned commits with the engineering
        patches applied and idempotent;
* P0  - the sitecustomize guard must refuse a foreign ``verl``;
* P2/P3 - the Warmup handoff must pick the last *complete* checkpoint (not the
        highest directory), copy it without touching Warmup, and write the GAD
        latest-checkpoint marker;
* config validation must report every missing key instead of falling back to a
        default model or dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GAD_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = GAD_DIR / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import backend_patches  # noqa: E402
import config as gad_config  # noqa: E402
import pipeline  # noqa: E402
from common import PipelineError, kv_cache_bytes_per_token, kv_cache_gib, parse_key_values, replace_once, vllm_kv_budget  # noqa: E402
import preflight  # noqa: E402

PYTHON = sys.executable
BACKEND_ROOT = Path(os.environ.get("GAD_TEST_BACKEND_ROOT", GAD_DIR / ".backends"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_prompt_parquet(path: Path, rows: int = 32) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("data_source", pa.string()),
            ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
            ("extra_info", pa.struct([("index", pa.string())])),
        ]
    )
    payload = [
        {
            "data_source": "math_dapo",
            "prompt": [{"role": "user", "content": f"Solve problem {i}. Answer with 'Answer: N'."}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(i), "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"index": f"u{i}"},
        }
        for i in range(rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(payload, schema=schema), path)


def make_gad_parquet(path: Path, rows: int = 32, *, prefix: str = "u") -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("content", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("teacher_response", pa.string()),
            ("data_source", pa.string()),
            ("ability", pa.string()),
            ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
            ("extra_info", pa.struct([("index", pa.string())])),
        ]
    )
    payload = [
        {
            "content": [{"role": "user", "content": f"Solve problem {i}."}],
            "teacher_response": f"Work...\nAnswer: {i}",
            "data_source": "math_dapo",
            "ability": "MATH",
            "reward_model": {"ground_truth": str(i), "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"index": f"{prefix}{i}"},
        }
        for i in range(rows)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(payload, schema=schema), path)


def make_model_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "config.json", {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"\x00")


def base_config(tmp: Path, **overrides) -> dict:
    train = tmp / "data" / "train.parquet"
    val = tmp / "data" / "dev.parquet"
    make_gad_parquet(train, rows=32, prefix="t")
    make_gad_parquet(val, rows=8, prefix="v")
    model = tmp / "model"
    make_model_dir(model)
    data = {
        "backends": {"root": str(BACKEND_ROOT)},
        "run": {"output_root": str(tmp / "runs"), "python_bin": PYTHON, "log_backend": "console"},
        "models": {"student": str(model)},
        "data": {
            "train_file": str(train),
            "val_file": str(val),
            "prompt_key": "content",
            "train_batch_size": 8,
            "val_batch_size": 8,
            "max_prompt_length": 2048,
            "max_response_length": 512,
        },
        "rollout": {"n": 4, "tensor_model_parallel_size": 1, "enable_thinking": False},
        "budget": {"seqkd_steps": 4, "warmup_steps": 10, "gad_steps": 6},
        "resources": {"n_gpus_per_node": 1, "nnodes": 1, "save_freq": 4, "test_freq": 4},
    }
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        data.setdefault(section, {})[field] = value
    return data


def write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class ConfigTests(unittest.TestCase):
    def test_missing_required_keys_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "run.yaml"
            write_yaml(cfg_path, {"run": {"output_root": tmp}})
            cfg = gad_config.load_config(cfg_path)
            problems = gad_config.validate_config(cfg)
            joined = " ".join(problems)
            for key in ("models.student", "data.train_file", "data.val_file", "data.train_batch_size"):
                self.assertIn(key, joined)

    def test_bad_gpu_and_batch_combinations_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["resources"]["n_gpus_per_node"] = 3
            payload["rollout"]["tensor_model_parallel_size"] = 2
            payload["data"]["train_batch_size"] = 8
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            problems = " ".join(gad_config.validate_config(cfg))
            self.assertIn("must be divisible by tensor_model_parallel_size", problems)
            self.assertIn("should be divisible by n_gpus_per_node", problems)

    def test_relative_paths_resolve_against_the_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "nested" / "run.yaml"
            payload = base_config(tmp_path)
            payload["data"]["train_file"] = "data/train.parquet"
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertEqual(Path(cfg.get("data.train_file")), (tmp_path / "nested" / "data" / "train.parquet").resolve())

    def test_missing_eval_parquet_only_fails_when_eval_is_enabled(self) -> None:
        """A staged eval path must not block training while eval.enabled is false."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["eval"] = {"enabled": False, "parquet": str(tmp_path / "missing" / "test.parquet")}
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertEqual(
                [p for p in gad_config.validate_paths(cfg, check_exists=True) if "eval.parquet" in p],
                [],
            )

            payload["eval"]["enabled"] = True
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertTrue(
                any("eval.parquet" in p for p in gad_config.validate_paths(cfg, check_exists=True)),
                "an enabled evaluation with a missing parquet must still be reported",
            )

    def test_epoch_budget_translates_to_steps(self) -> None:
        """The original recipe sizes stages in epochs; the pipeline must convert them."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["budget"]["stage_epochs"] = {"seqkd": 3, "warmup": 2, "gad": 4}
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            steps_per_epoch = 4  # 32 rows / batch 8

            self.assertEqual(pipeline.stage_step_budget(cfg, "seqkd", steps_per_epoch), 12)
            self.assertEqual(pipeline.stage_step_budget(cfg, "warmup", steps_per_epoch), 8)
            # GAD continues Warmup's global step, so its epoch budget is an absolute target.
            self.assertEqual(pipeline.stage_step_budget(cfg, "gad", steps_per_epoch, handoff_step=8), 8)
            self.assertEqual(pipeline.stage_step_budget(cfg, "gad", steps_per_epoch, handoff_step=4), 12)

    def test_gad_epoch_budget_must_exceed_the_warmup_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["budget"]["stage_epochs"] = {"seqkd": 3, "warmup": 2, "gad": 2}
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            with self.assertRaises(pipeline.PipelineError):
                pipeline.stage_step_budget(cfg, "gad", 4, handoff_step=8)

    def test_vllm_engine_limits_must_cover_the_length_budget(self) -> None:
        """vLLM aborts when chunked prefill is on and max_num_batched_tokens < max_model_len."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)  # 2048 prompt + 512 response = 2560
            payload["rollout"]["max_num_batched_tokens"] = 2048
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertIn(
                "rollout.max_num_batched_tokens (2048) must be >=",
                " ".join(gad_config.validate_config(cfg)),
            )

            payload["rollout"]["max_num_batched_tokens"] = 2560
            payload["rollout"]["max_model_len"] = 512
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            problems = " ".join(gad_config.validate_config(cfg))
            self.assertNotIn("rollout.max_num_batched_tokens (", problems)
            self.assertIn("rollout.max_model_len (512) must be >=", problems)

    def test_micro_batch_budget_must_cover_the_padded_sequence(self) -> None:
        """Below prompt+response the backend assert fires; catch it in preflight."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)  # 2048 + 512 = 2560
            payload["optim"] = {"actor_ppo_max_token_len_per_gpu": 2048, "critic_ppo_max_token_len_per_gpu": 12288}
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            problems = " ".join(gad_config.validate_config(cfg))
            self.assertIn("optim.actor_ppo_max_token_len_per_gpu (2048) must be >=", problems)
            self.assertNotIn("optim.critic_ppo_max_token_len_per_gpu", problems)

    def test_static_discriminator_micro_batch_must_be_positive(self) -> None:
        """R5: static discriminator batching needs micro >= 1 (reference CRITIC_MICRO=1)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["optim"] = {"critic_dynamic_bsz": False, "critic_micro_batch_size_per_gpu": 0}
            write_yaml(cfg_path, payload)
            problems = " ".join(gad_config.validate_config(gad_config.load_config(cfg_path)))
            self.assertIn("optim.critic_micro_batch_size_per_gpu must be >= 1", problems)

            # dynamic batching makes the micro size irrelevant (its own budget is checked instead)
            payload["optim"] = {"critic_dynamic_bsz": True, "critic_micro_batch_size_per_gpu": 0}
            write_yaml(cfg_path, payload)
            problems = " ".join(gad_config.validate_config(gad_config.load_config(cfg_path)))
            self.assertNotIn("optim.critic_micro_batch_size_per_gpu", problems)

    def test_free_cache_engine_requires_enforce_eager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["rollout"] = {"free_cache_engine": True, "enforce_eager": False}
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertIn(
                "rollout.free_cache_engine=True requires rollout.enforce_eager=True",
                " ".join(gad_config.validate_config(cfg)),
            )

            payload["rollout"]["enforce_eager"] = True
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertNotIn(
                "free_cache_engine=True requires",
                " ".join(gad_config.validate_config(cfg)),
            )

    def test_expandable_segments_is_rejected(self) -> None:
        """vLLM's CuMemAllocator asserts against it (cumem.py:146); fail in preflight."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["run"] = {"output_root": str(tmp_path / "runs"), "python_bin": PYTHON,
                              "pytorch_cuda_alloc_conf": "expandable_segments:True"}
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertIn(
                "must not use expandable_segments",
                " ".join(gad_config.validate_config(cfg)),
            )

            # a vLLM-compatible setting passes, and the default injects nothing
            payload["run"]["pytorch_cuda_alloc_conf"] = "max_split_size_mb:128"
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertNotIn("expandable_segments", " ".join(gad_config.validate_config(cfg)))
            self.assertEqual(
                pipeline.stage_env(cfg, tmp_path / "run", "seqkd", BACKEND_ROOT / "seqkd",
                                   log_backend="console").get("PYTORCH_CUDA_ALLOC_CONF"),
                "max_split_size_mb:128",
            )
            payload["run"]["pytorch_cuda_alloc_conf"] = ""
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertNotIn(
                "PYTORCH_CUDA_ALLOC_CONF",
                pipeline.stage_env(cfg, tmp_path / "run", "seqkd", BACKEND_ROOT / "seqkd",
                                   log_backend="console"),
            )

    def test_cli_overrides_coerce_booleans(self) -> None:
        """`--set x=false` must not be a truthy string: bool("false") is True."""
        parsed = parse_key_values(["run.val_before_train=false", "models.param_offload=TRUE",
                                   "resources.test_freq=0", "data.train_file=007.parquet"])
        self.assertIs(parsed["run.val_before_train"], False)
        self.assertIs(parsed["models.param_offload"], True)
        self.assertEqual(parsed["resources.test_freq"], 0)
        self.assertEqual(parsed["data.train_file"], "007.parquet")

    def test_reference_model_is_offloaded_by_default(self) -> None:
        """The forward-only KL reference stays on CPU, like the original harness."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path))
            cfg = gad_config.load_config(cfg_path)
            self.assertTrue(cfg.get("models.ref_param_offload"))
            overrides = pipeline.build_overrides(
                cfg, "warmup", model_path=str(tmp_path / "model"), critic_path=str(tmp_path / "model"),
                out_dir=tmp_path / "out", total_steps=4, epochs=1,
            )
            self.assertIn("actor_rollout_ref.ref.fsdp_config.param_offload=True", overrides)
            self.assertIn("actor_rollout_ref.actor.fsdp_config.param_offload=False", overrides)

    def test_stage_epoch_budget_is_validated_and_replaces_the_step_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["budget"]["stage_epochs"] = {"seqkd": 0}
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertIn(
                "budget.stage_epochs.seqkd must be > 0",
                " ".join(gad_config.validate_config(cfg)),
            )

            # A stage with an epoch budget does not need its *_steps value any more.
            payload["budget"]["stage_epochs"] = {"seqkd": 3}
            payload["budget"]["seqkd_steps"] = 0
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            self.assertNotIn(
                "budget.seqkd_steps must be > 0",
                " ".join(gad_config.validate_config(cfg)),
            )


class MicroBatchBudgetTests(unittest.TestCase):
    """The Warmup actor and the discriminator forward *both* sides in one micro-batch.

    ``rearrange_micro_batches`` used to size the budget by the student mask only, so a
    micro-batch that also forwards the teacher text could carry many times the token
    budget (long math answers make this the first thing to OOM).
    """

    def _load(self):
        backend = BACKEND_ROOT / "gad"
        if not (backend / "verl").is_dir():
            self.skipTest("backend gad not prepared")
        try:
            import tensordict  # noqa: F401
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"torch/tensordict unavailable: {exc}")
        path = str(backend)
        sys.path.insert(0, path)
        self.addCleanup(sys.path.remove, path)
        self.addCleanup(self._drop_verl_modules)
        return backend

    @staticmethod
    def _drop_verl_modules() -> None:
        for name in [n for n in sys.modules if n == "verl" or n.startswith("verl.")]:
            del sys.modules[name]

    def test_budget_covers_student_plus_teacher(self) -> None:
        self._load()
        import torch
        from tensordict import TensorDict

        from verl.utils.seqlen_balancing import rearrange_micro_batches

        rows, seq, budget = 4, 4096, 12288
        batch = TensorDict(
            {
                "attention_mask": torch.ones(rows, seq, dtype=torch.long),
                "teacher_attention_mask": torch.ones(rows, seq, dtype=torch.long),
            },
            batch_size=rows,
        )
        micro_batches, _ = rearrange_micro_batches(batch=batch, max_token_len=budget)
        worst = max(
            mb["attention_mask"].sum().item() + mb["teacher_attention_mask"].sum().item()
            for mb in micro_batches
        )
        # Student-only sizing (4 x 4096 = 16384) would give 2 micro-batches, each
        # forwarding 4 x 4096 teacher tokens as well = 32768 tokens per micro-batch.
        self.assertGreaterEqual(len(micro_batches), 3, "the teacher side was not counted")
        self.assertLess(worst, rows * seq * 2, "a micro-batch still carries both sides unbudgeted")

    def test_teacher_only_batches_keep_the_explicit_behaviour(self) -> None:
        """SeqKD selects only the teacher fields and asks for compute_teacher=True."""
        self._load()
        import torch
        from tensordict import TensorDict

        from verl.utils.seqlen_balancing import rearrange_micro_batches

        rows, seq = 4, 2048
        batch = TensorDict({"teacher_attention_mask": torch.ones(rows, seq, dtype=torch.long)}, batch_size=rows)
        micro_batches, _ = rearrange_micro_batches(batch=batch, max_token_len=4096, compute_teacher=True)
        self.assertEqual(sum(mb["teacher_attention_mask"].sum().item() for mb in micro_batches), rows * seq)
        self.assertGreaterEqual(len(micro_batches), rows * seq // 4096)


class BackendPatchTests(unittest.TestCase):
    def test_patches_are_applied_and_verified_on_every_backend(self) -> None:
        for stage in ("seqkd", "warmup", "gad"):
            worktree = BACKEND_ROOT / stage
            if not (worktree / "verl").is_dir():
                self.skipTest(f"backend {stage} not prepared; run tools/prepare_backends.py first")
            ok, problems = backend_patches.verify_patches(worktree)
            self.assertTrue(ok, f"{stage}: {problems}")

    def test_reapplying_patches_is_a_noop(self) -> None:
        worktree = BACKEND_ROOT / "gad"
        if not (worktree / "verl").is_dir():
            self.skipTest("backend gad not prepared")
        results = backend_patches.apply_patches(worktree)
        self.assertTrue(all(not result.changed for result in results), [r.name for r in results if r.changed])

    def test_frozen_algorithm_files_are_untouched(self) -> None:
        source = GAD_DIR / "verl"
        worktree = BACKEND_ROOT / "gad"
        if not (worktree / "verl").is_dir():
            self.skipTest("backend gad not prepared")
        for relative in (
            "verl/trainer/ppo/core_algos.py",
            "verl/workers/critic/dp_critic.py",
            "verl/workers/actor/dp_actor.py",
        ):
            self.assertEqual(
                (source / relative).read_bytes(),
                (worktree / relative).read_bytes(),
                f"{relative} must be byte-identical to the pinned backend",
            )

    def test_replace_once_refuses_ambiguous_anchors(self) -> None:
        with self.assertRaises(PipelineError):
            replace_once("a\na\n", "a\n", "b\n", what="ambiguous")
        with self.assertRaises(PipelineError):
            replace_once("a\n", "zzz", "b\n", what="missing")


class GuardTests(unittest.TestCase):
    def test_guard_rejects_a_foreign_verl(self) -> None:
        worktree = BACKEND_ROOT / "gad"
        if not (worktree / "verl").is_dir():
            self.skipTest("backend gad not prepared")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "run"
            guard_dir = pipeline.write_guard(None, run_dir, "gad", worktree)
            foreign = tmp_path / "foreign"
            (foreign / "verl").mkdir(parents=True)
            (foreign / "verl" / "__init__.py").write_text("", encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(guard_dir), str(foreign)])
            env["GAD_EXPECT_VERL_PATH"] = str(worktree)
            env["GAD_BACKEND_STAGE"] = "gad"
            bad = subprocess.run(
                [PYTHON, "-c", "import verl"], env=env, capture_output=True, text=True
            )
            self.assertNotEqual(bad.returncode, 0, "the guard must refuse a foreign verl")
            self.assertIn("gad-guard", bad.stderr)

            env["PYTHONPATH"] = os.pathsep.join([str(guard_dir), str(worktree)])
            good = subprocess.run(
                [PYTHON, "-c", "import verl, pathlib; print(pathlib.Path(verl.__file__).resolve())"],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertIn(str(worktree), good.stdout)


class RayGuardToolTests(unittest.TestCase):
    """`tools/check_ray_guard.py` must never claim success it did not observe."""

    def test_guard_marker_is_found_beyond_any_fixed_prefix(self) -> None:
        """Ray wraps worker exceptions, so the marker sits far past 400 chars.

        The original code truncated the message *before* searching it, which
        turned a correctly refused import into a reported failure.
        """
        import check_ray_guard

        wrapped = (
            "TaskError: ray::check() ...\n"
            + "while executing task, file sitecustomize.py line 129\n" * 40
            + "[gad-guard] stage gad must import verl from /a but resolved /b"
        )
        self.assertGreater(len(wrapped), 400)
        summary = check_ray_guard.summarize_worker_error(wrapped)
        self.assertTrue(summary["worker_error_has_guard"])
        self.assertIn("gad-guard", summary["worker_error"])
        self.assertEqual(summary["worker_error_len"], len(wrapped))
        self.assertLess(len(summary["worker_error"]), len(wrapped))

        clean = check_ray_guard.summarize_worker_error("ValueError: unrelated failure")
        self.assertFalse(clean["worker_error_has_guard"])
        self.assertEqual(clean["worker_error"], "ValueError: unrelated failure")

    def test_reports_inconclusive_when_ray_cannot_start(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            result = subprocess.run(
                [
                    PYTHON, str(TOOLS_DIR / "check_ray_guard.py"),
                    "--backend", str(BACKEND_ROOT / "gad"),
                    "--python-bin", PYTHON,
                    "--report", str(report_path),
                ],
                capture_output=True, text=True,
            )
            self.assertIn(result.returncode, (0, 2), result.stdout + result.stderr)
            if not report_path.exists():
                self.skipTest("backend not prepared")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["ok"], result.returncode == 0)
            if result.returncode == 2:
                self.assertFalse(report["ray_available"])
                self.assertIn("INCONCLUSIVE", result.stdout + result.stderr)
            else:
                # On a host where Ray starts, a refused import must be reported
                # as a pass - that is the case the truncation bug broke.
                self.assertTrue(report["ray_available"])
                self.assertTrue(report["negative_ok"], report["negative_case"])
                self.assertTrue(report["positive_ok"], report["positive_case"])


class DryRunTests(unittest.TestCase):
    def _run(self, cfg_path: Path) -> str:
        result = subprocess.run(
            [PYTHON, str(TOOLS_DIR / "pipeline.py"), "--config", str(cfg_path), "--dry-run"],
            capture_output=True, text=True, cwd=str(GAD_DIR),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_epoch_budget_drives_the_printed_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["budget"]["stage_epochs"] = {"seqkd": 3, "warmup": 2, "gad": 4}
            write_yaml(cfg_path, payload)
            if not (BACKEND_ROOT / "gad" / "verl").is_dir():
                self.skipTest("backends not prepared")
            out = self._run(cfg_path)

            seqkd = out.split("stage seqkd", 1)[1].split("stage warmup", 1)[0]
            warmup = out.split("stage warmup", 1)[1].split("stage gad", 1)[0]
            gad = out.split("stage gad", 1)[1]

            # 32 rows / batch 8 = 4 steps per epoch
            self.assertIn("trainer.total_training_steps=12", seqkd)   # 3 epochs
            self.assertIn("trainer.total_training_steps=8", warmup)   # 2 epochs
            # GAD's 4 epochs is an absolute target: 16 steps, continuing Warmup's step 8.
            self.assertIn("trainer.total_training_steps=16", gad)
            self.assertIn("global_step_8/actor/huggingface", gad)
            self.assertIn("epoch budget : 3 epoch(s) x 4 steps/epoch", out)

    def test_vllm_engine_limits_are_always_forwarded(self) -> None:
        """vLLM defaults (8192 / prompt+response) would abort; both must be explicit."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path))
            if not (BACKEND_ROOT / "gad" / "verl").is_dir():
                self.skipTest("backends not prepared")
            out = self._run(cfg_path)

            # 2048 + 512 = 2560
            self.assertIn("actor_rollout_ref.rollout.max_model_len=2560", out)
            self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=2560", out)
            self.assertIn("actor_rollout_ref.rollout.max_num_seqs=128", out)

    def test_stage_dependency_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path))
            if not (BACKEND_ROOT / "gad" / "verl").is_dir():
                self.skipTest("backends not prepared")
            out = self._run(cfg_path)

            seqkd = out.split("stage seqkd", 1)[1].split("stage warmup", 1)[0]
            warmup = out.split("stage warmup", 1)[1].split("stage gad", 1)[0]
            gad = out.split("stage gad", 1)[1]

            # SeqKD is an independent SFT baseline: no discriminator, no KL.
            self.assertNotIn("critic.model.path=", seqkd)
            self.assertIn("actor_rollout_ref.actor.use_kl_loss=False", seqkd)
            self.assertIn("actor_rollout_ref.actor.optim.lr=5e-06", seqkd)
            self.assertIn("trainer.critic_warmup=10", seqkd)
            self.assertIn("trainer.total_training_steps=4", seqkd)

            # Warmup starts from the student, warms the discriminator, then updates the actor.
            self.assertIn("critic.model.path=", warmup)
            self.assertIn("actor_rollout_ref.actor.use_kl_loss=True", warmup)
            self.assertIn("actor_rollout_ref.actor.kl_loss_coef=0.001", warmup)
            self.assertIn("trainer.critic_warmup=10", warmup)
            self.assertIn("trainer.total_training_steps=10", warmup)

            # GAD continues from S=10 and stops at S+K=16.
            self.assertIn("trainer.total_training_steps=16", gad)
            self.assertIn("trainer.critic_warmup=0", gad)
            self.assertIn("trainer.resume_mode=auto", gad)
            self.assertIn("global_step_10/actor/huggingface", gad)
            self.assertIn("global_step_10/critic/huggingface", gad)
            self.assertNotIn("model.path=" + str(tmp_path / "model") + " \\", gad)

            # every stage must forward the thinking switch and pin its backend
            for stage, name in ((seqkd, "seqkd"), (warmup, "warmup"), (gad, "gad")):
                self.assertIn("+data.apply_chat_template_kwargs.enable_thinking=false", stage)
                self.assertIn(str(BACKEND_ROOT / name), stage)

    def test_thinking_control_off_omits_the_switch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["rollout"]["thinking_control"] = False
            write_yaml(cfg_path, payload)
            out = self._run(cfg_path)
            self.assertNotIn("apply_chat_template_kwargs", out)

    def test_resume_flag_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path, **{"run.resume_worker_state": True}))
            out = self._run(cfg_path)
            self.assertIn("trainer.resume_load_worker_state=True", out)

    def test_stage_rollout_n_follows_the_reference_split(self) -> None:
        """Reference STAGE_N_SEQKD/STAGE_N_WARMUP=1 (uid dedup discards extra candidates)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path))
            if not (BACKEND_ROOT / "gad" / "verl").is_dir():
                self.skipTest("backends not prepared")
            out = self._run(cfg_path)
            seqkd = out.split("stage seqkd", 1)[1].split("stage warmup", 1)[0]
            warmup = out.split("stage warmup", 1)[1].split("stage gad", 1)[0]
            gad = out.split("stage gad", 1)[1]
            self.assertIn("actor_rollout_ref.rollout.n=1", seqkd)
            self.assertIn("actor_rollout_ref.rollout.n=1", warmup)
            self.assertIn("actor_rollout_ref.rollout.n=4", gad)  # base_config rollout.n = 4

    def test_gad_group_size_below_two_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["rollout"] = {"n": 1, "n_seqkd": 1, "n_warmup": 1}
            write_yaml(cfg_path, payload)
            problems = " ".join(gad_config.validate_config(gad_config.load_config(cfg_path)))
            self.assertIn("rollout.n (1) is the GAD group size and must be >= 2", problems)

    def test_discriminator_batches_statically_like_the_reference(self) -> None:
        """R5: CRITIC_DYNAMIC_BSZ=False / CRITIC_MICRO=1, decoupled from the actor."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path))
            if not (BACKEND_ROOT / "gad" / "verl").is_dir():
                self.skipTest("backends not prepared")
            out = self._run(cfg_path)
            warmup = out.split("stage warmup", 1)[1].split("stage gad", 1)[0]
            gad = out.split("stage gad", 1)[1]
            for block in (warmup, gad):
                self.assertIn("critic.use_dynamic_bsz=False", block)
                self.assertIn("critic.ppo_micro_batch_size_per_gpu=1", block)
                # the dynamic-only budget must not be forwarded in static mode
                self.assertNotIn("critic.ppo_max_token_len_per_gpu=", block)
            self.assertIn("actor_rollout_ref.actor.use_dynamic_bsz=True", warmup)

    def test_discriminator_can_opt_back_into_dynamic_batching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path)
            payload["optim"] = {"critic_dynamic_bsz": True,
                                "critic_ppo_max_token_len_per_gpu": 4096}
            write_yaml(cfg_path, payload)
            if not (BACKEND_ROOT / "gad" / "verl").is_dir():
                self.skipTest("backends not prepared")
            out = self._run(cfg_path)
            warmup = out.split("stage warmup", 1)[1].split("stage gad", 1)[0]
            self.assertIn("critic.use_dynamic_bsz=True", warmup)
            self.assertIn("critic.ppo_max_token_len_per_gpu=4096", warmup)
            self.assertNotIn("critic.ppo_micro_batch_size_per_gpu=", warmup)


class StageSelectionTests(unittest.TestCase):
    """--from-stage/--to-stage and the signature-gated skip logic."""

    def _write_cfg(self, tmp_path: Path, **overrides) -> Path:
        cfg_path = tmp_path / "run.yaml"
        write_yaml(cfg_path, base_config(tmp_path, **overrides))
        return cfg_path

    def _dry_run(self, cfg_path: Path, *extra: str) -> str:
        result = subprocess.run(
            [PYTHON, str(TOOLS_DIR / "pipeline.py"), "--config", str(cfg_path), "--dry-run", *extra],
            capture_output=True, text=True, cwd=str(GAD_DIR),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_stage_window_selects_the_expected_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = self._write_cfg(tmp_path)

            out = self._dry_run(cfg_path, "--from-stage", "warmup", "--to-stage", "gad")
            self.assertIn("stage warmup", out)
            self.assertIn("stage gad", out)
            self.assertNotIn("stage seqkd", out)

            out = self._dry_run(cfg_path, "--stages", "seqkd")
            self.assertIn("stage seqkd", out)
            self.assertNotIn("stage warmup", out)
            self.assertNotIn("stage gad", out)

            out = self._dry_run(cfg_path, "--to-stage", "warmup")
            self.assertIn("stage seqkd", out)
            self.assertIn("stage warmup", out)
            self.assertNotIn("stage gad", out)

    def test_unknown_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = self._write_cfg(tmp_path)
            result = subprocess.run(
                [PYTHON, str(TOOLS_DIR / "pipeline.py"), "--config", str(cfg_path), "--dry-run",
                 "--stages", "seqkd,nope"],
                capture_output=True, text=True, cwd=str(GAD_DIR),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown stage", result.stdout + result.stderr)

    def test_skip_requires_a_matching_signature_and_a_verified_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = self._write_cfg(tmp_path)
            cfg = gad_config.load_config(cfg_path)
            run_dir = tmp_path / "runs" / "r1"
            marker = pipeline.completion_marker(run_dir, "seqkd")

            self.assertEqual(pipeline.check_skip(cfg, run_dir, "seqkd"), (False, "no successful training exit record"))

            StageArtifactTests()._complete_checkpoint(pipeline.stage_out_dir(cfg,run_dir,"seqkd"),4)
            write_json(marker, {"total_steps":4,"verified": {"ok": True}, "inputs": pipeline.stage_inputs_digest(cfg, "seqkd")})
            skip, reason = pipeline.check_skip(cfg, run_dir, "seqkd")
            self.assertTrue(skip, reason)

            write_json(marker, {"verified": {"ok": True}, "inputs": {"different": True}})
            skip, reason = pipeline.check_skip(cfg, run_dir, "seqkd")
            self.assertFalse(skip)
            self.assertIn("signature changed", reason)

            write_json(marker, {"verified": {"ok": False}, "inputs": pipeline.stage_inputs_digest(cfg, "seqkd")})
            skip, reason = pipeline.check_skip(cfg, run_dir, "seqkd")
            self.assertFalse(skip)
            self.assertIn("not verified", reason)

    def test_legacy_seqkd_marker_ignores_warmup_performance_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = gad_config.load_config(self._write_cfg(tmp_path))
            run_dir = tmp_path / "runs/r1"
            snapshot = run_dir / "reports/resolved_configs/before_performance.yaml"
            write_yaml(snapshot, cfg.to_dict())
            StageArtifactTests()._complete_checkpoint(pipeline.stage_out_dir(cfg,run_dir,"seqkd"),10)
            old_inputs = pipeline.stage_inputs_digest(cfg, "seqkd")
            old_inputs["config_signature"] = pipeline.legacy_stage_signature(cfg, "seqkd")["signature"]
            write_json(pipeline.completion_marker(run_dir, "seqkd"),
                       {"verified": {"ok": True, "last_step": 10}, "inputs": old_inputs})

            cfg.data["rollout.gpu_memory_utilization"] = 0.50
            cfg.data["optim.critic_dynamic_bsz"] = False
            cfg.data["optim.critic_micro_batch_size_per_gpu"] = 2
            skip, reason = pipeline.check_skip(cfg, run_dir, "seqkd")
            self.assertTrue(skip, reason)

            cfg.data["data.train_batch_size"] = 128
            skip, reason = pipeline.check_skip(cfg, run_dir, "seqkd")
            self.assertFalse(skip)
            self.assertIn("signature changed", reason)

    def test_config_change_invalidates_the_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = self._write_cfg(tmp_path)
            cfg = gad_config.load_config(cfg_path)
            digest_before = pipeline.stage_inputs_digest(cfg, "warmup")

            write_yaml(cfg_path, base_config(tmp_path, **{"budget.warmup_steps": 11}))
            cfg_after = gad_config.load_config(cfg_path)
            digest_after = pipeline.stage_inputs_digest(cfg_after, "warmup")
            self.assertNotEqual(digest_before["config_signature"], digest_after["config_signature"])


class HandoffTests(unittest.TestCase):
    def _make_checkpoint(self, root: Path, step: int, *, complete: bool, roles=("actor", "critic")) -> Path:
        ckpt = root / f"global_step_{step}"
        for role in roles:
            role_dir = ckpt / role
            role_dir.mkdir(parents=True, exist_ok=True)
            (role_dir / f"model_world_size_1_rank_0.pt").write_bytes(b"shard")
            if complete:
                (role_dir / f"optim_world_size_1_rank_0.pt").write_bytes(b"optim")
            write_json(role_dir / "config.json", {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 8})
            (role_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        if complete:
            # a real torch payload: the handoff reads the dataloader position from it
            import torch

            torch.save({"_num_yielded": step, "_sampler_iter_yielded": step}, ckpt / "data.pt")
        return ckpt

    def test_picks_last_complete_checkpoint_and_stages_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            warmup = tmp_path / "warmup"
            self._make_checkpoint(warmup, 2, complete=True)
            self._make_checkpoint(warmup, 3, complete=False)  # newest but incomplete
            (warmup / "latest_checkpointed_iteration.txt").write_text("3", encoding="utf-8")

            gad_dir = tmp_path / "gad"
            result = subprocess.run(
                [
                    PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                    "--warmup-dir", str(warmup), "--gad-dir", str(gad_dir),
                    "--skip-merge", "--manifest", str(tmp_path / "handoff.json"),
                ],
                capture_output=True, text=True,
            )
            # step 3 is incomplete -> the tool must refuse, not silently hand over step 2
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("warmup run may not have finished cleanly", result.stdout + result.stderr)

            (warmup / "latest_checkpointed_iteration.txt").write_text("2", encoding="utf-8")
            result = subprocess.run(
                [
                    PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                    "--warmup-dir", str(warmup), "--gad-dir", str(gad_dir),
                    "--skip-merge", "--manifest", str(tmp_path / "handoff.json"),
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((tmp_path / "handoff.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["step"], 2)
            self.assertTrue((gad_dir / "global_step_2" / "actor" / "config.json").exists())
            self.assertTrue((gad_dir / "global_step_2" / "critic" / "config.json").exists())
            self.assertTrue((gad_dir / "global_step_2" / "data.pt").exists())
            self.assertEqual((gad_dir / "latest_checkpointed_iteration.txt").read_text().strip(), "2")
            # the dataloader position must be recorded, not assumed
            self.assertEqual(manifest["dataloader_state"]["_num_yielded"], 2)
            # the warmup directory must not gain a huggingface/ directory
            self.assertFalse((warmup / "global_step_2" / "actor" / "huggingface").exists())

    def test_corrupt_dataloader_state_fails_the_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            warmup = tmp_path / "warmup"
            ckpt = self._make_checkpoint(warmup, 1, complete=True)
            (ckpt / "data.pt").write_bytes(b"not-a-torch-file")
            (warmup / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
            result = subprocess.run(
                [
                    PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                    "--warmup-dir", str(warmup), "--gad-dir", str(tmp_path / "gad"), "--skip-merge",
                    "--manifest", str(tmp_path / "handoff.json"),
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dataloader state", result.stdout + result.stderr)

    def test_gad_steps_record_the_target_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            warmup = tmp_path / "warmup"
            self._make_checkpoint(warmup, 5, complete=True)
            (warmup / "latest_checkpointed_iteration.txt").write_text("5", encoding="utf-8")
            result = subprocess.run(
                [
                    PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                    "--warmup-dir", str(warmup), "--gad-dir", str(tmp_path / "gad"),
                    "--skip-merge", "--gad-steps", "7",
                    "--manifest", str(tmp_path / "handoff.json"),
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((tmp_path / "handoff.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["target_total_steps"], 12)

    def test_refuses_to_overwrite_a_non_empty_gad_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            warmup = tmp_path / "warmup"
            self._make_checkpoint(warmup, 1, complete=True)
            (warmup / "latest_checkpointed_iteration.txt").write_text("1", encoding="utf-8")
            gad_dir = tmp_path / "gad"
            gad_dir.mkdir(parents=True)
            (gad_dir / "leftover.txt").write_text("x", encoding="utf-8")
            result = subprocess.run(
                [
                    PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                    "--warmup-dir", str(warmup), "--gad-dir", str(gad_dir), "--skip-merge",
                ],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stdout + result.stderr)


class StageArtifactTests(unittest.TestCase):
    def test_failed_stage_log_is_preserved_before_the_next_attempt(self) -> None:
        """A restart truncates logs/<stage>.log, so a failure must leave a copy behind."""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "logs" / "seqkd.log"
            log.parent.mkdir(parents=True)
            log.write_text(
                "\n".join(f"line {i}" for i in range(120))
                + "\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.12 GiB\n",
                encoding="utf-8",
            )
            info = pipeline.preserve_failed_stage_log({"log": str(log)}, finished="2026-09-22T16:01:35Z")

            self.assertTrue(info["preserved"])
            preserved = Path(info["preserved"])
            self.assertTrue(preserved.exists())
            self.assertIn("failed-20260922T160135Z", preserved.name)
            self.assertIn("CUDA out of memory", preserved.read_text(encoding="utf-8"))
            # the tail travels inside stage_<stage>.json, so it survives even if the copy fails
            self.assertEqual(len(info["tail"]), 60)
            self.assertIn("OutOfMemoryError", info["tail"][-1])
            self.assertTrue(log.exists())

    def _complete_checkpoint(self, out: Path, step: int, *, stage: str = "warmup") -> Path:
        roles = ("actor",) if stage == "seqkd" else ("actor", "critic")
        ckpt = out / f"global_step_{step}"
        for role in roles:
            role_dir = ckpt / role
            role_dir.mkdir(parents=True, exist_ok=True)
            (role_dir / "model_world_size_1_rank_0.pt").write_bytes(b"shard")
            (role_dir / "optim_world_size_1_rank_0.pt").write_bytes(b"optim")
            write_json(role_dir / "config.json", {"architectures": ["Qwen3ForCausalLM"]})
            (role_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        import torch

        torch.save({"_num_yielded": step}, ckpt / "data.pt")
        (out / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")
        return ckpt

    def test_artifact_verification_rejects_incomplete_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "warmup"
            (out / "global_step_3" / "actor").mkdir(parents=True)
            (out / "global_step_3" / "actor" / "model_world_size_1_rank_0.pt").write_bytes(b"x")
            (out / "global_step_3" / "actor" / "config.json").write_text("{}", encoding="utf-8")
            report = pipeline.verify_stage_artifacts("warmup", out, expected_step=3)
            self.assertFalse(report["ok"])
            joined = " ".join(report["problems"])
            self.assertIn("critic", joined)
            self.assertIn("data.pt", joined)
            self.assertIn("latest_checkpointed_iteration.txt", joined)

    def test_math_metrics_are_required_and_recorded(self) -> None:
        """A stage that only reported ROUGE-L must not be accepted as complete."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "warmup"
            self._complete_checkpoint(out, 10)

            report = pipeline.verify_stage_artifacts(
                "warmup", out, expected_step=10, math_metrics_expected=True, strict_evaluation=True, math_metrics_reason="test"
            )
            self.assertFalse(report["ok"])
            self.assertIn("validation_math_metrics.jsonl", " ".join(report["problems"]))

            artifact = out / "validation_math_metrics.jsonl"
            artifact.write_text(
                json.dumps({"global_step": 0, "acc": 0.25, "count": 8, "truncated_rate": 0.1,
                            "answer_parse_failure_rate": 0.0, "scorer": "/s.py"}) + "\n"
                + json.dumps({"global_step": 10, "acc": 0.5, "score_mean": 0.0, "count": 512,
                              "truncated_rate": 0.2, "answer_parse_failure_rate": 0.05,
                              "scorer": "/s.py"}) + "\n",
                encoding="utf-8",
            )
            report = pipeline.verify_stage_artifacts(
                "warmup", out, expected_step=10, math_metrics_expected=True, strict_evaluation=True, math_metrics_reason="test"
            )
            self.assertTrue(report["ok"], report["problems"])
            self.assertEqual(report["math_metrics"]["records"], 2)
            self.assertEqual(report["math_metrics"]["last"]["acc"], 0.5)
            self.assertEqual(report["math_metrics"]["steps"], [0, 10])

            artifact.write_text("{not json}\n", encoding="utf-8")
            report = pipeline.verify_stage_artifacts(
                "warmup", out, expected_step=10, math_metrics_expected=True, strict_evaluation=True, math_metrics_reason="test"
            )
            self.assertFalse(report["ok"])

    def test_quick_record_cannot_complete_a_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "warmup"
            self._complete_checkpoint(out, 10)
            artifact = out / "validation_math_metrics.jsonl"
            artifact.write_text(json.dumps({"global_step": 10, "acc": 0.5, "count": 128,
                                            "validation_scope": "quick"}) + "\n")
            report = pipeline.verify_stage_artifacts(
                "warmup", out, expected_step=10, math_metrics_expected=True, strict_evaluation=True,
                require_final_full=True, full_count=512)
            self.assertFalse(report["ok"])
            self.assertIn("final validation must be full", " ".join(report["problems"]))
            artifact.write_text(json.dumps({"global_step": 10, "acc": 0.5, "count": 512,
                                            "validation_scope": "full"}) + "\n")
            report = pipeline.verify_stage_artifacts(
                "warmup", out, expected_step=10, math_metrics_expected=True, strict_evaluation=True,
                require_final_full=True, full_count=512)
            self.assertTrue(report["ok"], report["problems"])

    def test_math_metrics_are_not_required_when_validation_is_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path))
            cfg = gad_config.load_config(cfg_path)
            expected, reason = pipeline.validation_metrics_expected(cfg)
            self.assertTrue(expected, reason)
            self.assertEqual(cfg.get("validation.scorer_path"), str(GAD_DIR.parent / "sample" / "math_scorer.py"))

            payload = base_config(tmp_path)
            payload["run"]["val_before_train"] = False
            payload["resources"]["test_freq"] = 0
            write_yaml(cfg_path, payload)
            cfg = gad_config.load_config(cfg_path)
            expected, reason = pipeline.validation_metrics_expected(cfg)
            self.assertFalse(expected)
            self.assertIn("validation never runs", reason)


class GadResumeTests(unittest.TestCase):
    """`--resume-gad` must resume in place instead of re-running the handoff.

    ``prepare_gad_checkpoint.py`` refuses to write into a non-empty GAD
    directory, so an interrupted run could never be resumed with the same run id.
    """

    def _run_dir(self, tmp_path: Path, *, handoff_step: int = 12, gad_steps: int = 6, **overrides):
        cfg_path = tmp_path / "run.yaml"
        write_yaml(cfg_path, base_config(tmp_path, **{"budget.warmup_steps": handoff_step,
                                                      "budget.gad_steps": gad_steps, **overrides}))
        cfg = gad_config.load_config(cfg_path)
        run_dir = tmp_path / "runs" / "smoke"
        gad_dir = pipeline.stage_out_dir(cfg, run_dir, "gad")
        return cfg, run_dir, gad_dir

    def _handoff_manifest(self, run_dir: Path, gad_dir: Path, step: int) -> dict:
        manifest = {
            "actor_hf": str(gad_dir / f"global_step_{step}" / "actor" / "huggingface"),
            "critic_hf": str(gad_dir / f"global_step_{step}" / "critic" / "huggingface"),
            "step": step,
            "ok": True,
        }
        write_json(run_dir / "handoff_manifest.json", manifest)
        return manifest

    def _gad_checkpoint(self, gad_dir: Path, step: int) -> None:
        for role in ("actor", "critic"):
            role_dir = gad_dir / f"global_step_{step}" / role
            role_dir.mkdir(parents=True, exist_ok=True)
            (role_dir / "model_world_size_1_rank_0.pt").write_bytes(b"shard")
            (role_dir / "optim_world_size_1_rank_0.pt").write_bytes(b"optim")
            write_json(role_dir / "config.json", {"architectures": ["Qwen3ForCausalLM"]})
            (role_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        import torch

        torch.save({"_num_yielded": step}, gad_dir / f"global_step_{step}" / "data.pt")
        (gad_dir / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")

    def test_first_run_does_the_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, run_dir, gad_dir = self._run_dir(Path(tmp))
            decision = pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=True)
            self.assertEqual(decision["mode"], "fresh")
            self.assertIn("no complete checkpoint", decision["reason"])

    def test_resume_reuses_the_recorded_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg, run_dir, gad_dir = self._run_dir(tmp_path)
            manifest = self._handoff_manifest(run_dir, gad_dir, 12)
            self._gad_checkpoint(gad_dir, 14)

            decision = pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=True)
            self.assertEqual(decision["mode"], "resume")
            self.assertEqual(decision["from_step"], 14)
            self.assertEqual(decision["total_steps"], 18)
            self.assertEqual(decision["handoff"]["actor_hf"], manifest["actor_hf"])

            plan = pipeline.plan_stage(
                cfg, run_dir, "gad", steps_per_epoch_value=4, handoff=decision["handoff"], resume=decision
            )
            self.assertEqual(plan["resume"]["mode"], "resume")
            self.assertEqual(plan["total_steps"], 18)
            self.assertEqual(plan["model_path"], manifest["actor_hf"])

    def test_resume_ignores_an_incomplete_newer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg, run_dir, gad_dir = self._run_dir(tmp_path)
            self._handoff_manifest(run_dir, gad_dir, 12)
            self._gad_checkpoint(gad_dir, 14)
            (gad_dir / "global_step_16" / "actor").mkdir(parents=True)  # crashed mid-save

            decision = pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=True)
            self.assertEqual(decision["mode"], "resume")
            self.assertEqual(decision["from_step"], 14)

    def test_resume_falls_back_to_the_stage_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg, run_dir, gad_dir = self._run_dir(tmp_path)
            self._gad_checkpoint(gad_dir, 13)
            write_json(
                run_dir / "stage_gad.json",
                {"plan": {"model_path": str(gad_dir / "global_step_12" / "actor" / "huggingface"),
                          "critic_path": str(gad_dir / "global_step_12" / "critic" / "huggingface"),
                          "start_step": 12}},
            )
            decision = pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=True)
            self.assertEqual(decision["mode"], "resume")
            self.assertEqual(decision["from_step"], 13)
            self.assertIn("stage_gad.json", decision["handoff"]["source"])

    def test_resume_without_any_recorded_handoff_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, run_dir, gad_dir = self._run_dir(Path(tmp))
            self._gad_checkpoint(gad_dir, 14)
            with self.assertRaises(PipelineError) as ctx:
                pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=True)
            self.assertIn("cannot be resumed", str(ctx.exception))

    def test_resume_after_the_target_step_skips_the_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, run_dir, gad_dir = self._run_dir(Path(tmp))
            self._handoff_manifest(run_dir, gad_dir, 12)
            self._gad_checkpoint(gad_dir, 18)  # S + K already reached

            decision = pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=True)
            self.assertEqual(decision["mode"], "resume-complete")
            self.assertIn("already reached", decision["reason"])

    def test_no_resume_flag_keeps_the_normal_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, run_dir, gad_dir = self._run_dir(Path(tmp))
            decision = pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=False)
            self.assertEqual(decision["mode"], "fresh")
            self.assertEqual(decision["reason"], "no resume requested")
            self.assertNotIn("conflict", decision)

    def test_existing_checkpoint_without_the_flag_explains_the_conflict(self) -> None:
        """A bare re-run must not die with the handoff's "refusing to overwrite"."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg, run_dir, gad_dir = self._run_dir(Path(tmp))
            self._gad_checkpoint(gad_dir, 14)

            decision = pipeline.resolve_gad_resume(cfg, run_dir, added_steps=6, resume_requested=False)
            self.assertEqual(decision["conflict"]["step"], 14)
            self.assertIn("--resume-gad", decision["reason"])

            with self.assertRaises(PipelineError) as ctx:
                pipeline.run_stage(cfg, run_dir, "gad", dry_run=False, resume_requested=False)
            self.assertIn("--resume-gad", str(ctx.exception))

            # a dry run still prints the plan instead of aborting
            import contextlib
            import io

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                plan = pipeline.run_stage(cfg, run_dir, "gad", dry_run=True, resume_requested=False)
            self.assertIn("already holds a complete checkpoint", buffer.getvalue())
            self.assertEqual(plan["resume"]["mode"], "fresh")

    def test_resume_dry_run_never_schedules_the_handoff_again(self) -> None:
        """The reported failure: resume used to re-run the handoff and abort."""
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg, run_dir, gad_dir = self._run_dir(tmp_path, **{"run.resume_worker_state": True})
            manifest = self._handoff_manifest(run_dir, gad_dir, 12)
            self._gad_checkpoint(gad_dir, 14)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                plan = pipeline.run_stage(cfg, run_dir, "gad", dry_run=True, resume_requested=True)
            printed = buffer.getvalue()

            self.assertNotIn("prepare_gad_checkpoint.py", printed, printed)
            self.assertIn("resuming from global_step_14", printed)
            self.assertEqual(plan["model_path"], manifest["actor_hf"])
            self.assertEqual(plan["resume"]["mode"], "resume")
            self.assertEqual(plan["total_steps"], 18)
            self.assertIn("trainer.resume_load_worker_state=True", plan["argv"])


class FinalModelEvalTests(unittest.TestCase):
    """The final evaluation must score the trained GAD actor, not the Warmup init."""

    def test_dry_run_targets_the_gad_final_step_not_the_handoff_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(cfg_path, base_config(tmp_path, **{"budget.warmup_steps": 12, "budget.gad_steps": 6}))
            cfg = gad_config.load_config(cfg_path)
            run_dir = tmp_path / "runs" / "smoke"
            write_json(
                run_dir / "handoff_manifest.json",
                {
                    "actor_hf": str(run_dir / "gad" / "global_step_12" / "actor" / "huggingface"),
                    "critic_hf": str(run_dir / "gad" / "global_step_12" / "critic" / "huggingface"),
                    "step": 12,
                },
            )
            target = pipeline.export_final_actor(cfg, run_dir, dry_run=True)
            self.assertIn("global_step_18", target)
            self.assertNotIn("global_step_12", target)

    def test_exported_step_must_match_the_verified_gad_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_json(run_dir / "stage_gad.json", {"verification": {"ok": True, "last_step": 18}})
            with self.assertRaises(PipelineError) as ctx:
                pipeline.verify_final_model_step(run_dir, {"step": 12, "actor_hf": "/x"})
            self.assertIn("refusing to evaluate the wrong checkpoint", str(ctx.exception))
            self.assertEqual(pipeline.verify_final_model_step(run_dir, {"step": 18}), 18)

    def test_completion_marker_overrides_the_stage_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            write_json(run_dir / "stage_gad.json", {"verification": {"last_step": 18}})
            write_json(pipeline.completion_marker(run_dir, "gad"), {"verified": {"ok": True, "last_step": 20}})
            self.assertEqual(pipeline.expected_gad_final_step(run_dir), 20)


class HydraComposeTests(unittest.TestCase):
    """Every override the pipeline emits must resolve against the real config.

    ``--dry-run`` prints the commands without composing them, so a typo in a
    Hydra key would only surface on the GPU box, minutes into a launch. This
    composes the backend's own ``ppo_trainer`` config with the exact override
    list of each stage using Hydra's compose API (CPU only, no trainer start).
    """

    def _plan_overrides(self, tmp_path: Path, stage: str) -> list[str]:
        cfg_path = tmp_path / "run.yaml"
        payload = base_config(tmp_path)
        write_yaml(cfg_path, payload)
        cfg = gad_config.load_config(cfg_path)
        handoff = None
        if stage == "gad":
            gad_dir = pipeline.stage_out_dir(cfg, tmp_path / "runs" / "r1", "gad")
            handoff = {
                "actor_hf": str(gad_dir / "global_step_10" / "actor" / "huggingface"),
                "critic_hf": str(gad_dir / "global_step_10" / "critic" / "huggingface"),
                "step": 10,
            }
        plan = pipeline.plan_stage(cfg, tmp_path / "runs" / "r1", stage, steps_per_epoch_value=4,
                                   handoff=handoff)
        argv = plan["argv"]
        return argv[argv.index("-m") + 2:]

    def test_every_stage_override_composes(self) -> None:
        backend = BACKEND_ROOT / "gad"
        if not (backend / "verl").is_dir():
            self.skipTest("backends not prepared")
        script = r'''
import json, sys
from hydra import compose, initialize_config_dir
overrides = json.loads(sys.argv[1])
config_dir = sys.argv[2]
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=overrides)
payload = {
    "total_training_steps": int(cfg.trainer.total_training_steps),
    "critic_warmup": int(cfg.trainer.critic_warmup),
    "use_kl_loss": bool(cfg.actor_rollout_ref.actor.use_kl_loss),
    "kl_loss_coef": float(cfg.actor_rollout_ref.actor.kl_loss_coef),
    "val_n": int(cfg.actor_rollout_ref.rollout.val_kwargs.n),
    "val_do_sample": bool(cfg.actor_rollout_ref.rollout.val_kwargs.do_sample),
    "train_batch_size": int(cfg.data.train_batch_size),
    "max_response_length": int(cfg.data.max_response_length),
    "optimizer_offload_actor": bool(cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload),
    "optimizer_offload_critic": bool(cfg.critic.model.fsdp_config.optimizer_offload),
    "chat_template_kwargs": dict(cfg.data.get("apply_chat_template_kwargs", {}) or {}),
    "critic_dynamic_bsz": bool(cfg.critic.use_dynamic_bsz),
    "critic_micro": cfg.critic.get("ppo_micro_batch_size_per_gpu", None),
}
print("@@CFG@@" + json.dumps(payload))
'''
        expected_steps = {"seqkd": 4, "warmup": 10, "gad": 16}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for stage in ("seqkd", "warmup", "gad"):
                overrides = self._plan_overrides(tmp_path, stage)
                env = dict(os.environ)
                env["PYTHONPATH"] = str(backend)
                env["PYTHONWARNINGS"] = "ignore"
                result = subprocess.run(
                    [PYTHON, "-c", script, json.dumps(overrides), str(backend / "verl" / "trainer" / "config")],
                    capture_output=True, text=True, env=env, cwd=str(tmp_path),
                )
                self.assertEqual(result.returncode, 0, f"{stage}: {result.stderr[-1500:]}")
                payload = None
                for line in result.stdout.splitlines():
                    if line.startswith("@@CFG@@"):
                        payload = json.loads(line[len("@@CFG@@"):])
                self.assertIsNotNone(payload, f"{stage}: no config payload in {result.stdout[-500:]}")
                self.assertEqual(payload["total_training_steps"], expected_steps[stage], stage)
                self.assertEqual(payload["critic_warmup"], 0 if stage == "gad" else 10, stage)
                self.assertEqual(payload["use_kl_loss"], stage != "seqkd", stage)
                self.assertEqual(payload["val_n"], 1, stage)
                self.assertFalse(payload["val_do_sample"], stage)
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False}, stage)
                # R5: the discriminator batches statically with micro=1 (reference CRITIC_MICRO).
                # SeqKD has no discriminator at all, so no critic override is emitted there.
                if stage == "seqkd":
                    self.assertNotIn("critic.use_dynamic_bsz", " ".join(overrides))
                else:
                    self.assertFalse(payload["critic_dynamic_bsz"], stage)
                    self.assertEqual(payload["critic_micro"], 1, stage)

    def test_offload_and_batch_overrides_survive_composition(self) -> None:
        """The 24 GB-card settings must actually reach the trainer config."""
        backend = BACKEND_ROOT / "gad"
        if not (backend / "verl").is_dir():
            self.skipTest("backends not prepared")
        script = (
            "import json,sys\n"
            "from hydra import compose, initialize_config_dir\n"
            "overrides = json.loads(sys.argv[1])\n"
            "with initialize_config_dir(config_dir=sys.argv[2], version_base=None):\n"
            "    cfg = compose(config_name='ppo_trainer', overrides=overrides)\n"
            "print('@@CFG@@' + json.dumps({\n"
            "  'actor_offload': bool(cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload),\n"
            "  'critic_offload': bool(cfg.critic.model.fsdp_config.optimizer_offload),\n"
            "  'param_offload': bool(cfg.actor_rollout_ref.actor.fsdp_config.param_offload),\n"
            "  'batch': int(cfg.data.train_batch_size),\n"
            "  'val_batch': int(cfg.data.val_batch_size),\n"
            "  'max_tokens_per_gpu': int(cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu),\n"
            "  'gpu_mem': float(cfg.actor_rollout_ref.rollout.gpu_memory_utilization),\n"
            "  'tp': int(cfg.actor_rollout_ref.rollout.tensor_model_parallel_size),\n"
            "}))\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            write_yaml(
                cfg_path,
                base_config(
                    tmp_path,
                    **{
                        "models.optimizer_offload": True,
                        "data.train_batch_size": 64,
                        "data.val_batch_size": 16,
                        "rollout.tensor_model_parallel_size": 2,
                        "rollout.gpu_memory_utilization": 0.45,
                        "optim.actor_ppo_max_token_len_per_gpu": 16384,
                    },
                ),
            )
            cfg = gad_config.load_config(cfg_path)
            plan = pipeline.plan_stage(cfg, tmp_path / "runs" / "r1", "warmup", steps_per_epoch_value=4)
            overrides = plan["argv"][plan["argv"].index("-m") + 2:]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(backend)
            result = subprocess.run(
                [PYTHON, "-c", script, json.dumps(overrides), str(backend / "verl" / "trainer" / "config")],
                capture_output=True, text=True, env=env, cwd=str(tmp_path),
            )
            self.assertEqual(result.returncode, 0, result.stderr[-1500:])
            payload = json.loads([line for line in result.stdout.splitlines() if line.startswith("@@CFG@@")][0][7:])
            self.assertTrue(payload["actor_offload"])
            self.assertTrue(payload["critic_offload"])
            self.assertFalse(payload["param_offload"])
            self.assertEqual(payload["batch"], 64)
            self.assertEqual(payload["val_batch"], 16)
            self.assertEqual(payload["max_tokens_per_gpu"], 16384)
            self.assertEqual(payload["gpu_mem"], 0.45)
            self.assertEqual(payload["tp"], 2)


class VllmKvBudgetTests(unittest.TestCase):
    """The KV-cache gate must be predicted on CPU instead of discovered on GPU.

    vLLM sizes the pool once, at engine construction, and refuses to start when it
    cannot hold a single ``max_model_len`` sequence. That is exactly how the
    gpu_memory_utilization=0.30 run died: "1.09 GiB KV cache is needed ... larger
    than the available KV cache memory (0.95 GiB)".
    """

    QWEN3_1_7B = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 2048,
        "torch_dtype": "bfloat16",
    }

    def test_kv_bytes_per_token_matches_the_engine(self) -> None:
        # 2(K,V) x 8(kv heads) x 128(head_dim) x 28(layers) x 2B = 114688 B/token
        self.assertEqual(kv_cache_bytes_per_token(self.QWEN3_1_7B), 114688)
        # 114688 x 10240 = 1.0938 GiB == the engine's own "1.09 GiB" requirement
        self.assertAlmostEqual(kv_cache_gib(self.QWEN3_1_7B, 10240), 1.0938, places=3)

    def test_mha_and_head_dim_fallbacks_and_dtype(self) -> None:
        # no num_key_value_heads -> full MHA; no head_dim -> hidden_size / heads
        cfg = {"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64, "torch_dtype": "float16"}
        self.assertEqual(kv_cache_bytes_per_token(cfg), 2 * 4 * 16 * 2 * 2)
        fp32 = dict(cfg, torch_dtype="torch.float32")
        self.assertEqual(kv_cache_bytes_per_token(fp32), 2 * kv_cache_bytes_per_token(cfg))
        with self.assertRaises(ValueError):
            kv_cache_bytes_per_token(dict(cfg, torch_dtype="int8"))

    def test_budget_reproduces_the_measured_030_failure(self) -> None:
        low = vllm_kv_budget(
            model_config=self.QWEN3_1_7B, max_model_len=10240,
            gpu_memory_utilization=0.30, total_memory_gib=23.5, weight_gib=3.4,
        )
        # the engine really reported 0.95 GiB available and refused to start
        self.assertAlmostEqual(low["available_gib"], 0.95, places=2)
        self.assertFalse(low["ok"])

        high = vllm_kv_budget(
            model_config=self.QWEN3_1_7B, max_model_len=10240,
            gpu_memory_utilization=0.40, total_memory_gib=23.5, weight_gib=3.4,
        )
        self.assertTrue(high["ok"])
        self.assertFalse(high["tight"])
        self.assertAlmostEqual(high["available_gib"], 3.30, places=2)

    def _fake_device_python(self, tmp_path: Path, total_bytes: int) -> Path:
        """A stand-in interpreter that only answers the device-memory probe."""
        script = tmp_path / "fake_python"
        script.write_text(f'#!/bin/sh\necho {total_bytes}\n', encoding="utf-8")
        script.chmod(0o755)
        return script

    def test_preflight_flags_a_too_small_utilization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            payload = base_config(tmp_path, **{"run.python_bin": str(self._fake_device_python(tmp_path, 25_232_932_864))})
            model = Path(payload["models"]["student"])  # base_config already made the directory
            (model / "config.json").write_text(json.dumps(self.QWEN3_1_7B), encoding="utf-8")
            weights = model / "model.safetensors"
            with open(weights, "wb") as handle:
                handle.truncate(3_650_727_936)  # sparse: 3.4 GiB, no disk usage
            payload["data"]["max_prompt_length"] = 2048
            payload["data"]["max_response_length"] = 8192
            payload["rollout"]["name"] = "vllm"

            for utilization, should_fail in ((0.30, True), (0.40, False)):
                payload["rollout"]["gpu_memory_utilization"] = utilization
                write_yaml(cfg_path, payload)
                report = preflight.Report()
                preflight.check_vllm_kv_budget(gad_config.load_config(cfg_path), report)
                entry = report.sections["vllm_kv_budget"]
                self.assertTrue(entry["checked"], entry)
                self.assertEqual(entry["ok"], not should_fail, entry)
                if should_fail:
                    self.assertTrue(report.failures)
                    self.assertIn("gpu_memory_utilization", report.failures[0])
                    self.assertIn("1.09", report.failures[0])
                else:
                    self.assertFalse(report.failures, report.failures)

    def test_preflight_degrades_gracefully(self) -> None:
        """A partial model config (or no device query) must warn, never crash."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg_path = tmp_path / "run.yaml"
            # make_model_dir writes a stub config.json without the KV dimensions
            write_yaml(cfg_path, base_config(tmp_path, **{"rollout.name": "vllm"}))
            report = preflight.Report()
            preflight.check_vllm_kv_budget(gad_config.load_config(cfg_path), report)
            self.assertFalse(report.failures, report.failures)
            self.assertFalse(report.sections["vllm_kv_budget"]["checked"])
            self.assertTrue(report.warnings)

            # a non-vLLM rollout has no engine KV pool to size
            write_yaml(cfg_path, base_config(tmp_path, **{"rollout.name": "hf"}))
            report = preflight.Report()
            preflight.check_vllm_kv_budget(gad_config.load_config(cfg_path), report)
            self.assertFalse(report.sections["vllm_kv_budget"]["checked"])
            self.assertIn("not vllm", report.sections["vllm_kv_budget"]["reason"])

    def test_shipped_config_passes_its_own_gate(self) -> None:
        """The committed run config must survive the gate on a 24GB card."""
        config_path = GAD_DIR.parent / "train.yaml"
        if not config_path.exists():
            self.skipTest("shipped run config not found")
        cfg = gad_config.load_config(config_path)
        model_dir = Path(str(cfg.get("models.student")))
        model_config_path = model_dir / "config.json"
        if not model_config_path.exists():
            self.skipTest("student weights not present")
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        weight_bytes = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
        if not weight_bytes:
            self.skipTest("student safetensors not present")
        length_budget = int(cfg.get("data.max_prompt_length")) + int(cfg.get("data.max_response_length"))
        budget = vllm_kv_budget(
            model_config=model_config,
            max_model_len=int(cfg.get("rollout.max_model_len") or length_budget),
            gpu_memory_utilization=float(cfg.get("rollout.gpu_memory_utilization")),
            total_memory_gib=23.5,
            weight_gib=weight_bytes / (1024 ** 3),
        )
        self.assertTrue(budget["ok"], budget)
        self.assertFalse(budget["tight"], budget)
        # enforce_eager must stay on: the CUDA-graph pool is not covered by the engine sleep.
        self.assertTrue(bool(cfg.get("rollout.enforce_eager")))
        # free_cache_engine must not be advertised as a memory lever on vLLM 0.8.5.
        self.assertFalse(bool(cfg.get("rollout.free_cache_engine")))
        self.assertNotIn("expandable_segments", str(cfg.get("run.pytorch_cuda_alloc_conf") or ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
