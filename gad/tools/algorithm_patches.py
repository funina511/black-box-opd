"""Explicit, isolated algorithm profile. Engineering patches stay unchanged."""
import json
import py_compile
import shutil
from pathlib import Path
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "gad_extensions_v1"
VERSION = "gad.extensions.v1"


def source_files():
    files = {str(p.relative_to(ROOT / "gad/extensions")): p
             for p in (ROOT / "gad/extensions").rglob("*.py") if "core" not in p.relative_to(ROOT / "gad/extensions").parts}
    files.update({"core/" + p.name: p for p in (ROOT / "rubric/core").glob("*.py")})
    files.update({'public_scoring/' + str(p.relative_to(ROOT / 'scoring')): p
                  for p in (ROOT / 'scoring').rglob('*.py') if 'tests' not in p.parts})
    return files


def code_hash():
    import sys
    sys.path.insert(0, str(ROOT))
    from rubric.core.contracts import digest, file_hash
    return digest({name: file_hash(path) for name, path in sorted(source_files().items())})


def replace(text, old, new, name):
    if text.count(old) != 1:
        raise RuntimeError(f"algorithm patch anchor {name!r} expected once, got {text.count(old)}")
    return text.replace(old, new, 1)


def patch_warmup_sft(texts):
    from textwrap import indent

    ray = "verl/trainer/ppo/ray_trainer.py"
    start = texts[ray].index("                    # recompute old_log_probs")
    end = texts[ray].index("                    # update critic", start)
    # The reference configuration enables KL, but this pinned vanilla SFT
    # actor never adds it to its loss. Keep the full path for score consumers.
    texts[ray] = texts[ray][:start] + (
        "                    from verl.experimental.gad_extensions.warmup import skip_policy_diagnostics, length_metrics\n"
        "                    skip_diagnostics = skip_policy_diagnostics(self.config, batch)\n"
        "                    if skip_diagnostics:\n"
        "                        batch.meta_info['skip_warmup_policy_diagnostics'] = True\n"
        "                        batch.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature\n"
        "                        batch.meta_info['compute_teacher'] = False\n"
        "                    else:\n"
    ) + indent(texts[ray][start:end], "    ") + texts[ray][end:]
    metrics = "compute_data_metrics(batch=batch, use_critic=self.use_critic)"
    texts[ray] = replace(texts[ray], metrics, "length_metrics(batch) if skip_diagnostics else " + metrics,
                         "warmup data metrics")

    actor = "verl/workers/actor/dp_actor.py"
    texts[actor] = replace(texts[actor], '        if multi_turn:\n            select_keys.append("loss_mask")\n',
        '        skip_diagnostics = data.meta_info.get("skip_warmup_policy_diagnostics", False)\n'
        '        if not skip_diagnostics:\n'
        '            select_keys.extend(["old_log_probs", "advantages"])\n'
        '        if multi_turn:\n            select_keys.append("loss_mask")\n', "warmup policy fields")
    texts[actor] = replace(texts[actor], '"position_ids", "old_log_probs", "advantages",',
                           '"position_ids",', "warmup SFT fields")
    texts[actor] = replace(texts[actor], '        if self.config.use_kl_loss:\n            select_keys.append("ref_log_prob")',
        '        if not skip_diagnostics and self.config.use_kl_loss:\n            select_keys.append("ref_log_prob")',
        "warmup reference field")
    reads = '                    old_log_prob = data["old_log_probs"]\n                    advantages = data["advantages"]\n'
    # Both tensors are used exclusively in the non-vanilla policy loss.
    texts[actor] = replace(texts[actor], reads, "", "warmup PPO reads")
    loss = '                        policy_loss_fn = get_policy_loss_fn(loss_mode)'
    texts[actor] = replace(texts[actor], loss, indent(reads, "    ") + loss, "warmup PPO loss")


def apply(worktree, stage):
    worktree = Path(worktree)
    if stage == "seqkd":
        return {"profile": PROFILE, "stage": stage, "algorithm_enabled": False}
    marker = worktree / ".gad_algorithm_manifest.json"
    current_hash = code_hash()
    generator_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    if marker.exists():
        existing = json.loads(marker.read_text())
        # Updating a used worktree is forbidden; use a new profile root after code changes.
        if existing["extension_code_hash"] == current_hash and existing.get("patch_generator_hash") == generator_hash:
            verify(worktree)
            return existing
        if (worktree / ".gad_backend_used").exists():
            raise RuntimeError(f"extension code changed: create a new backends.root instead of modifying used {worktree}")
        # This isolated generated backend has never launched training.
        import subprocess
        subprocess.run(["git", "-C", str(worktree), "checkout", "--", *existing["patched_files"]], check=True)
        shutil.rmtree(worktree / "verl/experimental/gad_extensions")
        marker.unlink()
        from backend_patches import apply_patches
        apply_patches(worktree)
    paths = ["verl/workers/critic/dp_critic.py", "verl/workers/fsdp_workers.py",
             "verl/trainer/ppo/ray_trainer.py", "verl/utils/dataset/rl_dataset.py",
             "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py", "verl/trainer/config/ppo_trainer.yaml",
             "verl/trainer/main_ppo.py"]
    if stage == "warmup":
        paths.append("verl/workers/actor/dp_actor.py")
    texts = {name: (worktree / name).read_text() for name in paths}
    key = paths[0]
    texts[key] = replace(texts[key], "    def _forward_micro_batch(self, micro_batch, compute_teacher):\n",
        "    def _forward_micro_batch(self, micro_batch, compute_teacher):\n"
        "        # ALGORITHM(gad_extensions_v1): original adv reward or opt-in mixture.\n"
        "        if self.config.get('rubric', {}).get('enabled', False):\n"
        "            from verl.experimental.gad_extensions.worker import reward_forward\n"
        "            return reward_forward(self, micro_batch, compute_teacher)\n", "critic reward")
    texts[key] = replace(texts[key], "    def compute_values(self, data: DataProto) -> torch.Tensor:\n",
        "    def compute_values(self, data: DataProto) -> torch.Tensor:\n"
        "        from verl.experimental.gad_extensions.actor_reward import components_enabled\n"
        "        if self.config.get('rubric', {}).get('enabled', False) and components_enabled(self.config.rubric):\n"
        "            from verl.experimental.gad_extensions.worker import compute_mixed_values\n"
        "            return compute_mixed_values(self, data)\n", "mixed sequence components")
    texts[key] = replace(texts[key], "    def update_critic(self, data: DataProto):\n",
        "    def update_critic(self, data: DataProto):\n"
        "        if self.config.get('rubric', {}).get('enabled', False):\n"
        "            from verl.experimental.gad_extensions.worker import update_joint\n"
        "            return update_joint(self, data)\n", "critic update")
    key = paths[1]
    # Warmup wraps this line in a teacher/student branch (one deeper indent).
    texts[key] = replace(texts[key], 'output = DataProto.from_dict(tensors={"values": values})',
        'output = DataProto.from_dict(tensors=values if isinstance(values, dict) else {"values": values})',
        'mixed worker payload')
    texts[key] = replace(texts[key], "        critic_model_config.num_labels = 1\n",
        "        # ALGORITHM(gad_extensions_v1): structure prepared before FSDP/meta construction.\n"
        "        if config.get('rubric', {}).get('enabled', False):\n"
        "            from verl.experimental.gad_extensions.worker import check_model, record_worker_source\n"
        "            check_model(critic_model_config, config.rubric, local_path)\n"
        "            record_worker_source(config)\n"
        "        else:\n"
        "            if getattr(critic_model_config, 'rubric_head_layout', None):\n"
        "                raise ValueError('four-head critic cannot run with rubric disabled')\n"
        "            critic_model_config.num_labels = 1\n"
        "            if config.get('extensions_enabled', False):\n"
        "                from verl.experimental.gad_extensions.worker import record_worker_source\n"
        "                record_worker_source(config)\n", "worker model")
    key = paths[2]
    texts[key] = replace(texts[key], '                with marked_timer("step", timing_raw):\n',
        '                with marked_timer("step", timing_raw):\n'
        '                    if self.gad_extensions is not None:\n'
        '                        self.gad_extensions.prefetch_teachers(gen_batch, batch, self.global_steps)\n',
        'teacher prefetch before rollout')
    texts[key] = replace(texts[key], '                    # recompute old_log_probs\n',
        '                    if self.gad_extensions is not None and self.gad_extensions.execution.get("early_annotation"):\n'
        '                        self.gad_extensions.begin_prepare(batch, self.global_steps)\n'
        '                    # recompute old_log_probs\n', 'student annotation after rollout')
    texts[key] = replace(texts[key], '            self.train_dataloader.load_state_dict(dataloader_state_dict)\n',
        '            from verl.experimental.gad_extensions.sampler_state import prepare_state\n'
        '            dataloader_state_dict = prepare_state(self.train_dataloader, dataloader_state_dict, global_step_folder)\n'
        '            self.train_dataloader.load_state_dict(dataloader_state_dict)\n', 'exact sampler restore')
    texts[key] = replace(texts[key], '                    progress_bar.close()\n                    return\n',
        '                    progress_bar.close()\n'
        '                    if self.gad_extensions is not None:\n'
        '                        self.gad_extensions.close()\n'
        '                    return\n', 'annotation pool close')
    texts[key] = replace(texts[key], '    def _validate(self):\n',
        '    def _validate(self):\n'
        "        if self.config.trainer.get('disable_validation', False):\n"
        '            return {}\n', 'bounded smoke validation')
    texts[key] = replace(texts[key], '        # load checkpoint before doing anything\n        self._load_checkpoint()',
        "        import random\n"
        "        if self.config.trainer.get('seed') is not None:\n"
        "            random.seed(self.config.trainer.seed)\n"
        "            np.random.seed(self.config.trainer.seed)\n"
        "            torch.manual_seed(self.config.trainer.seed)\n"
        '        # load checkpoint before doing anything\n        self._load_checkpoint()', 'explicit driver seed')
    texts[key] = replace(texts[key], "        # load checkpoint before doing anything\n        self._load_checkpoint()",
        "        # ALGORITHM(gad_extensions_v1): lazy driver state; disabled is inert.\n"
        "        self.gad_extensions = None\n"
        "        if self.config.trainer.get('gad_extensions', {}).get('enabled', False):\n"
        "            from verl.experimental.gad_extensions.runtime import Runtime\n"
        "            self.gad_extensions = Runtime(self)\n"
        "        # load checkpoint before doing anything\n        self._load_checkpoint()\n"
        "        _gad_invocation_start_step = self.global_steps\n", "driver init")
    texts[key] = replace(texts[key], '                        batch.batch["token_level_scores"] = reward_tensor\n',
        "                        # All uid rows are gathered here, before the unchanged GRPO.\n"
        "                        from verl.experimental.gad_extensions.actor_reward import driver_reward\n"
        "                        reward_tensor = driver_reward(batch, reward_tensor, self.config.trainer, self.config.algorithm)\n"
        '                        batch.batch["token_level_scores"] = reward_tensor\n', 'driver group reward')
    texts[key] = replace(texts[key], "                            critic_output = self.critic_wg.update_critic(batch)",
        "                            critic_batch = batch\n"
        "                            if self.gad_extensions is not None:\n"
        "                                critic_batch, extension_metrics = self.gad_extensions.prepare(batch, self.global_steps)\n"
        "                                metrics.update(extension_metrics)\n"
        "                            critic_output = self.critic_wg.update_critic(critic_batch)", "driver update")
    texts[key] = replace(texts[key], "                    # Log rollout generations if enabled\n",
        "                    if self.gad_extensions is not None:\n"
        "                        metrics.update(self.gad_extensions.commit(self.config.trainer.critic_warmup <= self.global_steps))\n"
        "                    # Log rollout generations if enabled\n", "driver commit")
    texts[key] = replace(texts[key], "                is_last_step = self.global_steps >= self.total_training_steps\n",
        "                is_last_step = self.global_steps >= self.total_training_steps\n"
        "                _gad_invocation_limit = self.config.trainer.get('gad_extensions', {}).get('max_invocation_steps')\n"
        "                extension_stop = bool(_gad_invocation_limit and self.global_steps - _gad_invocation_start_step >= _gad_invocation_limit)\n", "invocation limit")
    texts[key] = replace(texts[key], "                    if self.global_steps in self.config.trainer.checkpoint_steps or (self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)):\n",
        "                    if is_last_step or extension_stop or self.global_steps in self.config.trainer.checkpoint_steps or (self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0):\n", "limit checkpoint")
    texts[key] = replace(texts[key], "                if is_last_step:\n                    pprint(f\"Final validation metrics: {last_val_metrics}\")",
        "                if is_last_step or extension_stop:\n                    pprint(f\"Final validation metrics: {last_val_metrics}\")", "limit return")
    texts[key] = replace(texts[key], "        # latest checkpointed iteration tracker (for atomic usage)\n",
        "        if getattr(self, 'gad_extensions', None) is not None:\n"
        "            from verl.experimental.gad_extensions.checkpoint_state import save\n"
        "            save(self.gad_extensions, local_global_step_folder)\n"
        "        # latest checkpointed iteration tracker (for atomic usage)\n", "save commit")
    texts[key] = replace(texts[key], "            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest",
        "            if getattr(self, 'gad_extensions', None) is not None:\n"
        "                from verl.experimental.gad_extensions.checkpoint_state import latest_complete\n"
        "                global_step_folder = latest_complete(checkpoint_folder, allow_handoff=True)\n"
        "                if global_step_folder is None and any(__import__('pathlib').Path(checkpoint_folder).glob('global_step_*')):\n"
        "                    raise ValueError('extension checkpoints exist but none is complete; refusing to restart from scratch')\n"
        "            else:\n"
        "                global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest", "load selection")
    texts[key] = replace(texts[key], "        actor_path = os.path.join(global_step_folder, \"actor\")\n",
        "        if getattr(self, 'gad_extensions', None) is not None:\n"
        "            from pathlib import Path\n"
        "            if (Path(global_step_folder) / 'checkpoint_complete.json').exists():\n"
        "                from verl.experimental.gad_extensions.checkpoint_state import restore\n"
        "                restore(self.gad_extensions, global_step_folder)\n"
        "                self.config.trainer.resume_load_worker_state = True\n"
        "            elif (Path(global_step_folder) / 'handoff_initialization.json').exists():\n"
        "                self.config.trainer.resume_load_worker_state = False\n"
        "            else:\n"
        "                raise ValueError('missing same-stage extension state')\n"
        "        actor_path = os.path.join(global_step_folder, \"actor\")\n", "load state")
    # A wall-time trace aligns the external three-second physical GPU samples
    # with rollout, optimizer updates and checkpoint saving; native timers stay.
    import re
    before_fit, fit = texts[key].split('    def fit(self):', 1)
    def trace_phase(match):
        indent, arguments = match.group(1), match.group(2)
        return (indent + 'with (self.gad_extensions.timed_phase(marked_timer, ' + arguments + ') '
                'if self.gad_extensions is not None else marked_timer(' + arguments + ')):')
    fit = re.sub(r'^(\s*)with marked_timer\(("(?:step|gen|old_log_prob|ref|reward|adv|update_critic|update_actor|save_checkpoint)"[^\n]*)\):',
                 trace_phase, fit, flags=re.M)
    texts[key] = before_fit + '    def fit(self):' + fit
    key = paths[3]
    texts[key] = replace(texts[key], "        self.max_prompt_length = config.get(\"max_prompt_length\", 1024)\n",
        "        self.max_prompt_length = config.get(\"max_prompt_length\", 1024)\n"
        "        self.gad_extensions_enabled = config.get('gad_extensions', {}).get('enabled', False)\n", "dataset switch")
    texts[key] = replace(texts[key], "        if not self.processor_type == \"MiniCPMVImageProcessor\":\n",
        "        if self.gad_extensions_enabled:\n"
        "            row_dict['gad_prompt_truncated'] = bool(input_ids.shape[-1] > self.max_prompt_length)\n"
        "        if not self.processor_type == \"MiniCPMVImageProcessor\":\n", "dataset provenance")
    texts[key] = replace(texts[key], '            teacher_response = self.tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)\n',
        '            teacher_response = self.tokenizer(teacher_response, return_tensors="pt", add_special_tokens=False)\n'
        "            if self.gad_extensions_enabled:\n"
        "                row_dict['gad_teacher_token_truncated'] = bool(teacher_response['input_ids'].shape[-1] > self.max_response_length)\n", 'teacher pre-truncation provenance')
    key = paths[4]
    texts[key] = replace(texts[key], "            response = []\n            rollout_log_probs = []\n",
        "            response = []\n            rollout_log_probs = []\n"
        "            gad_finish_reasons = []\n"
        "            gad_response_lengths = []\n", "rollout list")
    texts[key] = replace(texts[key], "                    response.append(response_ids)\n",
        "                    response.append(response_ids)\n"
        "                    if self.config.get('gad_extensions', {}).get('enabled', False):\n"
        "                        gad_finish_reasons.append(output.outputs[sample_id].finish_reason)\n"
        "                        gad_response_lengths.append(len(response_ids))\n", "rollout finish")
    texts[key] = replace(texts[key], "        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)\n",
        "        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)\n"
        "        if self.config.get('gad_extensions', {}).get('enabled', False):\n"
        "            lengths = torch.tensor(gad_response_lengths, device=response.device)\n"
        "            response_attention_mask = (torch.arange(response_length, device=response.device)[None, :] < lengths[:, None]).to(attention_mask.dtype)\n", "actual generated lengths")
    texts[key] = replace(texts[key], "        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)\n",
        "        if self.config.get('gad_extensions', {}).get('enabled', False):\n"
        "            non_tensor_batch['gad_finish_reason'] = np.array(gad_finish_reasons, dtype=object)\n"
        "            non_tensor_batch['gad_generated_length'] = np.array(gad_response_lengths, dtype=object)\n"
        "        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)\n", "rollout result")
    key = paths[5]
    texts[key] = replace(texts[key], "data:\n", "data:\n  gad_extensions: {enabled: false}\n", "data yaml")
    start = texts[key].index("actor_rollout_ref:\n")
    end = texts[key].index("\ncritic:\n", start)
    section = replace(texts[key][start:end], "  rollout:\n", "  rollout:\n    gad_extensions: {enabled: false}\n    seed: 0\n", "rollout yaml")
    texts[key] = texts[key][:start] + section + texts[key][end:]
    import yaml
    from extension_config import EXTENSION_DEFAULTS
    def nested(prefix):
        result = {}
        for name, value in EXTENSION_DEFAULTS.items():
            if not name.startswith(prefix + "."):
                continue
            parts = name[len(prefix) + 1:].split(".")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return result
    def block(value):
        return "".join("  " + line + "\n" for line in yaml.safe_dump(value, sort_keys=False).splitlines())
    driver = nested("rubric")
    driver.update({"spec_hash": None, "prompt_hash": None, "protocol_identity": None,
                   "scorer_path": None, "judge_fingerprint": None, 'scoring_math': None})
    worker = {"enabled": False, "gradient_diagnostics_dir": None, "mode": "joint", "loss_coef": .1,
              **{key: EXTENSION_DEFAULTS["rubric." + key] for key in ("loss_mode", "count_gate_tau", "gradient_diagnostics_interval", "gradient_diagnostics_max_pairs", "reward_source", "actor_reward_coef", "actor_reward_shadow", "actor_reward_group_weight", "actor_reward_rubric_std_floor")},
              'actor_reward_weights': driver['actor_reward_weights'],
              'version': 'math_v1',
              "head_layout": driver["head_layout"], "spec_hash": None}
    texts[key] = replace(texts[key], "critic:\n", "critic:\n" + block({"rubric": worker, "extensions_code_hash": None, "extensions_enabled": False}), "critic yaml")
    state = {"enabled": False, "identity": None, "stage": None, "start_step": 0,
             "execution": {"teacher_prefetch": False, "early_annotation": False},
             'scorer_identity': None,
             "token_identity": None, "teacher_index": None, "sampling_seed": None, "max_invocation_steps": None,
             "actor_reward_fp32": False, "student_scorer_monitor": False}
    texts[key] = replace(texts[key], "trainer:\n", "trainer:\n" + block(
        {"rubric": driver, "critic_replay": nested("replay"), "gad_extensions": state}), "trainer yaml")
    if stage == "warmup":
        texts[key] = replace(texts[key], "trainer:\n", "trainer:\n  warmup_forward_diagnostics: false\n",
                             "warmup forward diagnostics")
    key = paths[6]
    texts[key] = replace(texts[key], '        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)\n',
        '        if data_config.get("gad_extensions", {}).get("enabled", False):\n'
        '            from verl.experimental.gad_extensions.sampler_state import ExactRandomSampler\n'
        '            sampler = ExactRandomSampler(data_source=dataset, generator=train_dataloader_generator)\n'
        '        else:\n'
        '            sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)\n',
        'exact random sampler')
    if stage == "warmup":
        patch_warmup_sft(texts)
    # Validate all anchors and compile in memory before writing any file.
    for name, text in texts.items():
        if name.endswith(".py"):
            compile(text, str(worktree / name), "exec")
    destination = worktree / "verl/experimental/gad_extensions"
    for name, source in source_files().items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name, text in texts.items():
        (worktree / name).write_text(text)
    manifest = {"profile": PROFILE, "version": VERSION, "stage": stage,
                "patch_generator_hash": generator_hash,
                "extension_code_hash": current_hash,
                "patched_files": {name: sha256(text.encode()).hexdigest() for name, text in texts.items()}}
    marker.write_text(json.dumps(manifest, indent=2))
    verify(worktree)
    return manifest


def verify(worktree):
    worktree = Path(worktree)
    manifest = json.loads((worktree / ".gad_algorithm_manifest.json").read_text())
    for name, expected in manifest["patched_files"].items():
        path = worktree / name
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"algorithm backend source changed: {path}")
        if path.suffix == ".py":
            compile(path.read_text(), str(path), "exec")
    from sys import path as sys_path
    sys_path.insert(0, str(ROOT))
    from rubric.core.contracts import digest, file_hash
    directory = worktree / "verl/experimental/gad_extensions"
    actual = {str(p.relative_to(directory)): file_hash(p) for p in directory.rglob("*.py")}
    if digest(actual) != manifest["extension_code_hash"]:
        raise RuntimeError("deployed extension package hash mismatch")
    return manifest
