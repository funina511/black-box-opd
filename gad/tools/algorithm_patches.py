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
             "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py", "verl/trainer/config/ppo_trainer.yaml"]
    texts = {name: (worktree / name).read_text() for name in paths}
    key = paths[0]
    texts[key] = replace(texts[key], "    def _forward_micro_batch(self, micro_batch, compute_teacher):\n",
        "    def _forward_micro_batch(self, micro_batch, compute_teacher):\n"
        "        # ALGORITHM(gad_extensions_v1): adv-only external reward contract.\n"
        "        if self.config.get('rubric', {}).get('enabled', False):\n"
        "            from verl.experimental.gad_extensions.worker import reward_forward\n"
        "            return reward_forward(self, micro_batch, compute_teacher)\n", "critic reward")
    texts[key] = replace(texts[key], "    def update_critic(self, data: DataProto):\n",
        "    def update_critic(self, data: DataProto):\n"
        "        if self.config.get('rubric', {}).get('enabled', False):\n"
        "            from verl.experimental.gad_extensions.worker import update_joint\n"
        "            return update_joint(self, data)\n", "critic update")
    key = paths[1]
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
    texts[key] = replace(texts[key], "        # load checkpoint before doing anything\n        self._load_checkpoint()",
        "        # ALGORITHM(gad_extensions_v1): lazy driver state; disabled is inert.\n"
        "        self.gad_extensions = None\n"
        "        if self.config.trainer.get('rubric', {}).get('enabled', False) or self.config.trainer.get('critic_replay', {}).get('enabled', False):\n"
        "            from verl.experimental.gad_extensions.runtime import Runtime\n"
        "            self.gad_extensions = Runtime(self)\n"
        "        # load checkpoint before doing anything\n        self._load_checkpoint()", "driver init")
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
        "                extension_stop = bool(self.gad_extensions is not None and self.gad_extensions.options.get('max_invocation_steps') and self.gad_extensions.invocation_steps + 1 >= self.gad_extensions.options['max_invocation_steps'])\n", "invocation limit")
    texts[key] = replace(texts[key], "                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):\n",
        "                    if self.config.trainer.save_freq > 0 and (is_last_step or extension_stop or self.global_steps % self.config.trainer.save_freq == 0):\n", "limit checkpoint")
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
    key = paths[3]
    texts[key] = replace(texts[key], "        self.max_prompt_length = config.get(\"max_prompt_length\", 1024)\n",
        "        self.max_prompt_length = config.get(\"max_prompt_length\", 1024)\n"
        "        self.gad_extensions_enabled = config.get('gad_extensions', {}).get('enabled', False)\n", "dataset switch")
    texts[key] = replace(texts[key], "        if not self.processor_type == \"MiniCPMVImageProcessor\":\n",
        "        if self.gad_extensions_enabled:\n"
        "            row_dict['gad_prompt_truncated'] = bool(input_ids.shape[-1] > self.max_prompt_length)\n"
        "            row_dict['gad_teacher_token_truncated'] = bool(teacher_response is not None and teacher_response['input_ids'].shape[-1] > self.max_response_length)\n"
        "        if not self.processor_type == \"MiniCPMVImageProcessor\":\n", "dataset provenance")
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
    section = replace(texts[key][start:end], "  rollout:\n", "  rollout:\n    gad_extensions: {enabled: false}\n    seed: null\n", "rollout yaml")
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
    driver.update({"spec_hash": None, "scorer_path": None, "judge_fingerprint": None})
    worker = {"enabled": False, "mode": "joint", "loss_coef": .1,
              "head_layout": driver["head_layout"], "spec_hash": None}
    texts[key] = replace(texts[key], "critic:\n", "critic:\n" + block({"rubric": worker, "extensions_code_hash": None, "extensions_enabled": False}), "critic yaml")
    state = {"enabled": False, "identity": None, "stage": None, "start_step": 0,
             "token_identity": None, "teacher_index": None, "sampling_seed": None, "max_invocation_steps": None}
    texts[key] = replace(texts[key], "trainer:\n", "trainer:\n" + block(
        {"rubric": driver, "critic_replay": nested("replay"), "gad_extensions": state}), "trainer yaml")
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
