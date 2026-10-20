"""Formal training config for the opt-in online algorithm profile."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rubric.core.contracts import LAYOUT, layout_for, schema_for, digest, file_hash

EXTENSION_DEFAULTS = {
    'scoring.math.version': 'legacy_card_v1', 'scoring.math.entrypoint': None, 'scoring.math.timeout_seconds': 2.0,
    'scoring.math.fingerprint': None, 'rubric.version': 'math_v1',
    'run.seed': 42, 'run.disable_validation': False,
    "rollout.seed": None,
    "backends.profile": "original",
    "budget.gad_new_steps": None,
    "budget.init_from_warmup_run": None,
    "budget.init_step": None,
    "run.engineering_smoke": False,
    "run.max_stage_steps": None,
    "optim.critic_ppo_mini_batch_size": None,
    "rubric.enabled": False, "rubric.stages": ["warmup", "gad"], "rubric.mode": "joint",
    "rubric.loss_mode": "legacy_active", "rubric.count_gate_tau": 4,
    "rubric.gradient_diagnostics_interval": 0, "rubric.gradient_diagnostics_max_pairs": 4,
    "rubric.semantic_check": False, "rubric.semantic_max_extra_requests": 1,
    "rubric.judge.request_budget": None,
    "rubric.loss_coef": .1, "rubric.spec": str(ROOT / "rubric/specs/math_v1.yaml"),
    "rubric.prompt": str(ROOT / "rubric/prompts/judge_math_v1.txt"),
    "rubric.head_layout": list(LAYOUT), "rubric.reward_source": "adv_only",
    "rubric.label_source": "online", "rubric.label_scope": "selected_critic_batch",
    "rubric.teacher_cache": True, "rubric.student_cache": True,
    "rubric.truncated_process_policy": "mask", "rubric.on_annotation_error": "mask",
    "rubric.on_service_unavailable": "abort_step", "rubric.seed": 42,
    "rubric.max_empty_student_process_steps": 3,
    "rubric.judge.backend": "openai_compatible", "rubric.judge.base_url": None,
    "rubric.judge.model": None, "rubric.judge.model_revision": None,
    "rubric.judge.tokenizer_path": None, "rubric.judge.api_key_env": None,
    "rubric.judge.temperature": 0., "rubric.judge.max_output_tokens": 768,
    "rubric.judge.enable_thinking": False, "rubric.judge.reasoning_parser": None,
    "rubric.judge.top_p": 1., "rubric.judge.top_k": -1, "rubric.judge.min_p": 0.,
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
        if cfg.get("rubric.head_layout") != list(layout_for(cfg.get('rubric.version'))) or cfg.get("rubric.reward_source") != "adv_only":
            problems.append("rubric requires fixed four-channel layout and adv_only reward")
        for key, expected in {"label_source": "online", "label_scope": "selected_critic_batch",
                              "truncated_process_policy": 'visible_prefix' if cfg.get('rubric.version') == 'math_process_v2' else 'mask', "on_annotation_error": "mask",
                              "on_service_unavailable": "abort_step"}.items():
            if cfg.get("rubric." + key) != expected:
                problems.append("unsupported rubric." + key)
        from rubric.core.losses import LOSS_MODES
        if cfg.get('rubric.loss_mode') not in LOSS_MODES:
            problems.append('rubric.loss_mode must be legacy_active, fixed_dimension or fixed_dimension_gated')
        tau = cfg.get('rubric.count_gate_tau')
        if type(tau) not in (int, float) or not 0 < tau < float('inf'):
            problems.append('rubric.count_gate_tau must be finite and positive')
        interval = cfg.get('rubric.gradient_diagnostics_interval')
        if type(interval) is not int or interval < 0:
            problems.append('rubric.gradient_diagnostics_interval must be a nonnegative integer')
        if type(cfg.get('rubric.gradient_diagnostics_max_pairs')) is not int or cfg.get('rubric.gradient_diagnostics_max_pairs') < 1:
            problems.append('rubric.gradient_diagnostics_max_pairs must be positive')
        if type(cfg.get('rubric.semantic_check')) is not bool or type(cfg.get('rubric.semantic_max_extra_requests')) is not int or cfg.get('rubric.semantic_max_extra_requests') != 1:
            problems.append('semantic_check must be boolean; at most one extra request per trajectory')
        if cfg.get('rubric.semantic_check') and (cfg.get('rubric.version') != 'math_process_v2' or cfg.get('rubric.mode') != 'joint'):
            problems.append('semantic_check requires math_process_v2 joint mode')
        request_budget = cfg.get('rubric.judge.request_budget')
        if request_budget is not None and (type(request_budget) is not int or request_budget < 1):
            problems.append('judge request_budget must be null or a positive integer')
        if cfg.get("rubric.loss_coef") < 0:
            problems.append("rubric.loss_coef must be nonnegative")
        limit = cfg.get('rubric.max_empty_student_process_steps')
        if type(limit) is not int or limit < 1:
            problems.append('rubric.max_empty_student_process_steps must be a positive integer')
        if cfg.get("rubric.mode") == "joint":
            import math
            thinking = cfg.get('rubric.judge.enable_thinking')
            parser = cfg.get('rubric.judge.reasoning_parser')
            if type(thinking) is not bool:
                problems.append('rubric.judge.enable_thinking must be boolean, independent of rollout.enable_thinking')
            if thinking and parser != 'deepseek_r1':
                problems.append('thinking judge on vLLM 0.8.5 requires reasoning_parser=deepseek_r1')
            if not thinking and parser is not None:
                problems.append('non-thinking judge on vLLM 0.8.5 requires reasoning_parser=null')
            for key, low, high, inclusive in [('temperature', 0, 2, True), ('top_p', 0, 1, False), ('min_p', 0, 1, True)]:
                value = cfg.get('rubric.judge.' + key)
                if type(value) not in (int, float) or not math.isfinite(value) or value > high or value < low or (not inclusive and value == low):
                    problems.append('invalid rubric.judge.' + key)
            if thinking and cfg.get('rubric.judge.temperature') == 0:
                problems.append('thinking judge requires non-greedy temperature > 0')
            top_k = cfg.get('rubric.judge.top_k')
            if type(top_k) is not int or (top_k != -1 and top_k < 1):
                problems.append('rubric.judge.top_k must be -1 or a positive integer')
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
            if cfg.get('rubric.judge.max_output_tokens') >= cfg.get('rubric.judge.max_context_tokens'):
                problems.append('judge reserved output must be smaller than total context')
            if cfg.get('rubric.version') == 'math_process_v2' and (type(cfg.get('rubric.judge.max_retries')) is not int or cfg.get('rubric.judge.max_retries') not in (0, 1)):
                problems.append('V2.1 permits at most one transport retry and no format retry')
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
                stage_cfg = cfg.for_stage(stage)
                n = stage_cfg.get("rollout.n_warmup") if stage == "warmup" else stage_cfg.get("rollout.n")
                world = cfg.get("resources.n_gpus_per_node")
                micro = cfg.get("optim.critic_micro_batch_size_per_gpu")
                if any(type(v) is not int or v < 1 for v in (n,world,micro)):
                    problems.append(stage + ': rollout.n, DP and microbatch sizes must be positive integers')
                    continue
                if mini * n % world or (mini * n // world) % micro:
                    problems.append(stage + ": critic minibatch must divide DP and microbatch sizes")
    if cfg.get('scoring.math.version') == 'math_unified_v1':
        from scoring.math import identity
        if cfg.get('rollout.seed') is None:problems.append('new unified experiments require explicit rollout.seed')
        if cfg.get('scoring.math.fingerprint') != identity(cfg.get('scoring.math.timeout_seconds'))['fingerprint']:
            problems.append('saved math scorer/dependency fingerprint differs; new run required')
        if cfg.get('validation.scorer_path') != cfg.get('scoring.math.entrypoint'):
            problems.append('validation scorer must derive from scoring.math.entrypoint')
        if cfg.get('rubric.enabled') and cfg.get('rubric.version') != 'math_process_v2':
            problems.append('unified new rubric experiments require math_process_v2')
        if cfg.get('run.disable_validation') and not cfg.get('run.engineering_smoke'):
            problems.append('disable_validation is reserved for bounded engineering smoke')
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
    if extension_enabled(cfg) or cfg.get('scoring.math.version') == 'math_unified_v1':
        cfg.data["backends.profile"] = "gad_extensions_v1"
        original = ROOT / "gad/.backends"
        if Path(cfg.get("backends.root")).resolve() == original:
            from algorithm_patches import code_hash
            fingerprint = digest({'code': code_hash(), 'patch': file_hash(ROOT / 'gad/tools/algorithm_patches.py'), 'backend_patch': file_hash(ROOT / 'gad/tools/backend_patches.py')})[:12]
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
    from gad.extensions.judge_client import guidance_schema, guidance_backend
    identity = {"model": options["model"], "revision": options["model_revision"], "dtype": options["dtype"],
                "tokenizer_path": str(model_dir.resolve()), "model_files": stats,
                "vllm": importlib.metadata.version("vllm"),
                "enable_thinking": options['enable_thinking'], "reasoning_parser": options['reasoning_parser'],
                "engine": "v0", "server_sha256": file_hash(ROOT / 'gad/tools/rubric_judge_server.py'),
                "client_sha256": file_hash(ROOT / 'gad/extensions/judge_client.py'),
                "schema": digest(schema_for(cfg.get('rubric.version'))),
                'rubric_version': cfg.get('rubric.version'),
                "TP": options["tensor_parallel_size"], "context": options["max_context_tokens"],
                "decoding": "guided_json", "guided_backend_request": guidance_backend(cfg.get('rubric.version')),
                "guidance_revision": "v0_xgrammar_string_length_hint_projection_v1",
                "guidance_schema_hash": digest(guidance_schema(cfg.get('rubric.version'))),
                "cpu_offload_gb": options["cpu_offload_gb"]}
    identity["request_defaults"] = {key: options[key] for key in ("temperature", "top_p", "top_k", "min_p", "max_output_tokens", "max_context_tokens")}
    identity["judge_seed"] = cfg.get("rubric.seed")
    if cfg.get('rubric.version') == 'math_process_v2':
        from rubric.core.process_v2 import protocol_identity
        identity['semantic_policy'] = {key: cfg.get('rubric.' + key) for key in ('semantic_check', 'semantic_max_extra_requests')}
        identity['semantic_module_hash'] = file_hash(ROOT / 'rubric/core/semantic_v24.py')
        identity['protocol_identity'] = protocol_identity(Path(cfg.require('rubric.spec')).read_text(), Path(cfg.require('rubric.prompt')).read_text())
    if full:
        identity["model_sha256"] = {p.name: file_hash(p) for p in files}
    return identity


def deployed_code_identity(cfg, stage):
    """Used backends are immutable; validate their deployed package before resume."""
    from algorithm_patches import code_hash, verify
    backend = Path(cfg.require('backends.root')) / stage
    if (backend / '.gad_backend_used').exists():
        return verify(backend)
    return {'extension_code_hash':code_hash(), 'patch_generator_hash':file_hash(ROOT/'gad/tools/algorithm_patches.py')}


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
        driver['scoring_math'] = {key[13:]: value for key, value in cfg.data.items() if key.startswith('scoring.math.')}
        worker = {"enabled": True, 'gradient_diagnostics_dir':str(Path(out_dir)/'gradient_diagnostics'), "mode": driver["mode"], "loss_coef": driver["loss_coef"],
                  **{key: driver[key] for key in ('loss_mode', 'count_gate_tau', 'gradient_diagnostics_interval', 'gradient_diagnostics_max_pairs')},
                  'version': driver['version'], "head_layout": list(layout_for(driver['version'])), "spec_hash": driver["spec_hash"]}
    replay_config = {key[7:]: value for key, value in cfg.data.items() if key.startswith("replay.")}
    replay_config["enabled"] = replay
    token_files = {p.name: file_hash(p) for p in Path(cfg.require("models.student")).iterdir()
                   if p.name.startswith("tokenizer") or p.suffix == ".jinja"}
    state = {"enabled": rubric or replay, "stage": stage, "start_step": start_step,
             "max_invocation_steps": cfg.get("run.max_stage_steps"),
             "sampling_seed": cfg.get("rollout.seed"),
             'scorer_identity': {key[13:]: value for key, value in cfg.data.items() if key.startswith('scoring.math.')},
             "token_identity": digest({"tokenizer": token_files, "thinking": cfg.get("rollout.enable_thinking")}),
             "teacher_index": cfg.get("data.teacher_index"),
             "identity": digest({"declared": declared, "code": getattr(cfg, "_recovery_code_hash", None) or deployed_code_identity(cfg,stage)["extension_code_hash"], "stage": stage,
                                 "scorer": {key: value for key, value in cfg.data.items() if key.startswith('scoring.math.')},
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
            f"critic.extensions_code_hash={getattr(cfg, '_recovery_code_hash', None) or deployed_code_identity(cfg,stage)['extension_code_hash']}",
            f"critic.extensions_enabled={str(rubric or replay).lower()}",
            f"data.gad_extensions.enabled={str(rubric or replay).lower()}",
            f"actor_rollout_ref.rollout.gad_extensions.enabled={str(rubric or replay).lower()}",
            f"critic.ppo_mini_batch_size={cfg.get('optim.critic_ppo_mini_batch_size') or cfg.get('data.train_batch_size')}"]
