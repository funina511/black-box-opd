"""Formal training config for the opt-in online algorithm profile."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rubric.core.contracts import LAYOUT, digest, file_hash

EXTENSION_DEFAULTS = {
    "rollout.seed": None,
    "backends.profile": "original",
    "budget.gad_new_steps": None,
    "budget.init_from_warmup_run": None,
    "budget.init_step": None,
    "run.engineering_smoke": False,
    "run.max_stage_steps": None,
    "optim.critic_ppo_mini_batch_size": None,
    "rubric.enabled": False, "rubric.stages": ["warmup", "gad"], "rubric.mode": "joint",
    "rubric.loss_coef": .1, "rubric.spec": str(ROOT / "rubric/specs/math_v1.yaml"),
    "rubric.prompt": str(ROOT / "rubric/prompts/judge_math_v1.txt"),
    "rubric.head_layout": list(LAYOUT), "rubric.reward_source": "adv_only",
    "rubric.label_source": "online", "rubric.label_scope": "selected_critic_batch",
    "rubric.teacher_cache": True, "rubric.student_cache": True,
    "rubric.truncated_process_policy": "mask", "rubric.on_annotation_error": "mask",
    "rubric.on_service_unavailable": "abort_step", "rubric.seed": 42,
    "rubric.judge.backend": "openai_compatible", "rubric.judge.base_url": None,
    "rubric.judge.model": None, "rubric.judge.model_revision": None,
    "rubric.judge.tokenizer_path": None, "rubric.judge.api_key_env": None,
    "rubric.judge.temperature": 0., "rubric.judge.max_output_tokens": 768,
    "rubric.judge.max_context_tokens": 16384, "rubric.judge.structured_output": "required",
    "rubric.judge.concurrency": 8, "rubric.judge.request_timeout_seconds": 120,
    "rubric.judge.step_timeout_seconds": 600, "rubric.judge.max_retries": 2,
    "rubric.judge.dtype": "bfloat16", "rubric.judge.tensor_parallel_size": 2,
    "rubric.judge.gpu_memory_utilization": .85, "rubric.judge.cpu_offload_gb": 0.,
    "replay.enabled": False, "replay.stages": ["gad"], "replay.mix_ratio": .25,
    "replay.max_age_steps": 8, "replay.capacity_pairs": 1024, "replay.max_bytes": 536870912,
    "replay.sampling": "uniform_without_replacement", "replay.eviction": "oldest_first",
    "replay.insert_after_step": True, "replay.store_labels": True, "replay.seed": 43,
    "resources.judge_gpus": None,
}


def active(cfg, feature, stage):
    return bool(cfg.get(feature + ".enabled") and stage in cfg.get(feature + ".stages"))


def extension_enabled(cfg):
    return bool(cfg.get("rubric.enabled") or cfg.get("replay.enabled"))


def validate(cfg):
    problems = []
    if cfg.get("backends.profile") not in ("original", "gad_extensions_v1"):
        problems.append("backends.profile must be original or gad_extensions_v1")
    for feature in ("rubric", "replay"):
        if type(cfg.get(feature + ".enabled")) is not bool:
            problems.append(feature + ".enabled must be boolean")
    if cfg.get("rubric.enabled"):
        if cfg.get("rubric.stages") not in (["warmup", "gad"], ["gad"]):
            problems.append("rubric.stages supports [warmup,gad] or [gad]; warmup-only cannot discard auxiliary heads")
        if cfg.get("rubric.mode") not in ("joint", "correctness_only"):
            problems.append("rubric.mode must be joint or correctness_only")
        if cfg.get("rubric.head_layout") != list(LAYOUT) or cfg.get("rubric.reward_source") != "adv_only":
            problems.append("rubric requires fixed four-channel layout and adv_only reward")
        for key, expected in {"label_source": "online", "label_scope": "selected_critic_batch",
                              "truncated_process_policy": "mask", "on_annotation_error": "mask",
                              "on_service_unavailable": "abort_step"}.items():
            if cfg.get("rubric." + key) != expected:
                problems.append("unsupported rubric." + key)
        if cfg.get("rubric.loss_coef") < 0:
            problems.append("rubric.loss_coef must be nonnegative")
        if cfg.get("rubric.mode") == "joint":
            for key in ("base_url", "model", "model_revision", "tokenizer_path"):
                if not cfg.get("rubric.judge." + key):
                    problems.append("joint requires rubric.judge." + key)
            if cfg.get("rubric.judge.backend") != "openai_compatible" or cfg.get("rubric.judge.structured_output") != "required":
                problems.append("joint requires fingerprinted OpenAI-compatible guided JSON service")
            training = set(str(cfg.get("resources.gpus")).split(","))
            judging = set(str(cfg.get("resources.judge_gpus") or "").split(","))
            if not cfg.get("resources.judge_gpus"):
                problems.append("joint requires explicit resources.judge_gpus")
            if training & judging:
                problems.append("judge GPUs and training GPUs overlap")
            if cfg.get("rubric.judge.tensor_parallel_size") != len(judging):
                problems.append("judge TP must match its explicit GPU count")
            for key in ("concurrency", "max_output_tokens", "max_context_tokens", "request_timeout_seconds", "step_timeout_seconds"):
                if cfg.get("rubric.judge." + key) <= 0:
                    problems.append("rubric.judge." + key + " must be positive")
    if cfg.get("replay.enabled"):
        if cfg.get("replay.stages") != ["gad"]:
            problems.append("replay v1 supports only stages=[gad]")
        if not 0 <= cfg.get("replay.mix_ratio") < 1:
            problems.append("replay.mix_ratio must be in [0,1)")
        for key in ("max_age_steps", "capacity_pairs", "max_bytes"):
            if type(cfg.get("replay." + key)) is not int or cfg.get("replay." + key) < 1:
                problems.append("replay." + key + " must be a positive integer")
        if cfg.get("replay.sampling") != "uniform_without_replacement" or cfg.get("replay.eviction") != "oldest_first":
            problems.append("unsupported replay sampling/eviction")
        if not cfg.get("replay.insert_after_step") or not cfg.get("replay.store_labels"):
            problems.append("v1 requires insert_after_step=true and store_labels=true")
    if extension_enabled(cfg):
        if cfg.get("optim.critic_dynamic_bsz") or cfg.get("resources.nnodes") != 1:
            problems.append("extensions v1 requires static critic batching and a single node (FSDP SP=1)")
        mini = cfg.get("optim.critic_ppo_mini_batch_size") or cfg.get("data.train_batch_size")
        if not isinstance(mini, int) or mini <= 0 or cfg.get("data.train_batch_size") % mini:
            problems.append("critic logical minibatch prompts must divide train_batch_size")
        else:
            for stage in ("warmup", "gad"):
                n = cfg.get("rollout.n_warmup") if stage == "warmup" else cfg.get("rollout.n")
                world = cfg.get("resources.n_gpus_per_node")
                micro = cfg.get("optim.critic_micro_batch_size_per_gpu")
                if mini * n % world or (mini * n // world) % micro:
                    problems.append(stage + ": critic minibatch must divide DP and microbatch sizes")
    if cfg.get("budget.gad_new_steps") is not None:
        if type(cfg.get("budget.gad_new_steps")) is not int or cfg.get("budget.gad_new_steps") < 1:
            problems.append("budget.gad_new_steps must be a positive integer")
    if cfg.get("run.max_stage_steps") is not None:
        if not extension_enabled(cfg) or not cfg.get("run.engineering_smoke") or type(cfg.get("run.max_stage_steps")) is not int or cfg.get("run.max_stage_steps") < 1:
            problems.append("run.max_stage_steps requires a positive engineering extension invocation limit")
    if bool(cfg.get("budget.init_from_warmup_run")) != bool(cfg.get("budget.init_step")):
        problems.append("--init-from-warmup-run and explicit --init-step are required together")
    return problems


def configure_profile(cfg):
    if extension_enabled(cfg):
        cfg.data["backends.profile"] = "gad_extensions_v1"
        original = ROOT / "gad/.backends"
        if Path(cfg.get("backends.root")).resolve() == original:
            from algorithm_patches import code_hash
            fingerprint = digest({'code': code_hash(), 'patch': file_hash(ROOT / 'gad/tools/algorithm_patches.py')})[:12]
            cfg.data["backends.root"] = str(ROOT / ('gad/.backends_gad_extensions_v1_' + fingerprint))


def service_identity(cfg, full=False):
    """Configuration identity plus local model content; caches validated file stats."""
    options = {key[13:]: value for key, value in cfg.data.items() if key.startswith("rubric.judge.")}
    model_dir = Path(cfg.require("rubric.judge.tokenizer_path"))
    files = sorted(model_dir.glob("*.json")) + sorted(model_dir.glob("*.jinja")) + sorted(model_dir.glob("*.safetensors"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("judge local model weights missing")
    stats = {p.name: [p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns] for p in files}
    import importlib.metadata
    identity = {"model": options["model"], "revision": options["model_revision"], "dtype": options["dtype"],
                "tokenizer_path": str(model_dir.resolve()), "model_files": stats,
                "vllm": importlib.metadata.version("vllm"), "nonthinking": True,
                "schema": digest(__import__('rubric.core.contracts', fromlist=['JUDGE_SCHEMA']).JUDGE_SCHEMA),
                "TP": options["tensor_parallel_size"], "context": options["max_context_tokens"],
                "decoding": "guided_json/xgrammar", "cpu_offload_gb": options["cpu_offload_gb"]}
    identity["request_defaults"] = {key: options[key] for key in ("temperature", "max_output_tokens", "max_context_tokens")}
    identity["judge_seed"] = cfg.get("rubric.seed")
    if full:
        identity["model_sha256"] = {p.name: file_hash(p) for p in files}
    return identity


def hydra_overrides(cfg, stage, out_dir, start_step):
    from common import sha256_file
    from algorithm_patches import code_hash
    rubric = active(cfg, "rubric", stage)
    replay = active(cfg, "replay", stage)
    if cfg.get("backends.profile") != "gad_extensions_v1" or stage == "seqkd":
        return []
    declared = {key: value for key, value in cfg.data.items() if key.startswith(("rubric.", "replay."))}
    worker = {"enabled": rubric}
    driver = {"enabled": rubric}
    if rubric:
        driver = {key[7:]: value for key, value in cfg.data.items() if key.startswith("rubric.")}
        driver["enabled"] = True
        judge = {key[6:]: driver.pop(key) for key in list(driver) if key.startswith("judge.")}
        driver["judge"] = judge
        driver["scorer_path"] = cfg.require("validation.scorer_path")
        driver["spec_hash"] = file_hash(driver["spec"])
        driver["judge_fingerprint"] = digest(service_identity(cfg)) if cfg.get("rubric.mode") == "joint" else "rule_only"
        worker = {"enabled": True, "mode": driver["mode"], "loss_coef": driver["loss_coef"],
                  "head_layout": list(LAYOUT), "spec_hash": driver["spec_hash"]}
    replay_config = {key[7:]: value for key, value in cfg.data.items() if key.startswith("replay.")}
    replay_config["enabled"] = replay
    token_files = {p.name: file_hash(p) for p in Path(cfg.require("models.student")).iterdir()
                   if p.name.startswith("tokenizer") or p.suffix == ".jinja"}
    state = {"enabled": rubric or replay, "stage": stage, "start_step": start_step,
             "max_invocation_steps": cfg.get("run.max_stage_steps"),
             "sampling_seed": cfg.get("rollout.seed") if cfg.get("rollout.seed") is not None else cfg.get("rubric.seed") if rubric or replay else None,
             "token_identity": digest({"tokenizer": token_files, "thinking": cfg.get("rollout.enable_thinking")}),
             "teacher_index": cfg.get("data.teacher_index"),
             "identity": digest({"declared": declared, "code": code_hash(), "stage": stage,
                                 "scorer": file_hash(cfg.require("validation.scorer_path")),
                                 "train_data": file_hash(cfg.require("data.train_file")),
                                 "judge": driver.get("judge_fingerprint"),
                                 "initial_model": cfg.require("models.student"),
                                 "rollout_seed": cfg.get("rollout.seed"),
                                 "initial_critic": cfg.require("models.discriminator_init")})}
    def encode(value):
        if isinstance(value, dict):
            return "{" + ",".join(str(k) + ":" + encode(v) for k, v in value.items()) + "}"
        if isinstance(value, list):
            return "[" + ",".join(encode(v) for v in value) + "]"
        return json.dumps(value, ensure_ascii=False)
    return [f"trainer.rubric={encode(driver)}", f"critic.rubric={encode(worker)}",
            f"trainer.critic_replay={encode(replay_config)}", f"trainer.gad_extensions={encode(state)}",
            f"critic.extensions_code_hash={code_hash()}",
            f"critic.extensions_enabled={str(rubric or replay).lower()}",
            f"data.gad_extensions.enabled={str(rubric or replay).lower()}",
            f"actor_rollout_ref.rollout.gad_extensions.enabled={str(rubric or replay).lower()}",
            f"critic.ppo_mini_batch_size={cfg.get('optim.critic_ppo_mini_batch_size') or cfg.get('data.train_batch_size')}"]
