"""FSDP-compatible four-channel forward and globally normalized auxiliary gradients."""
import torch
from .core.heads import last_response_logits, validate_head
from .core.losses import weighted_contribution, dimension_weights


def rubric_enabled(critic):
    return critic.config.get("rubric", {}).get("enabled", False)


def forward_channels(critic, data, teacher):
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    prefix = "teacher_" if teacher else ""
    ids = data[prefix + "input_ids"]
    attention = data[prefix + "attention_mask"]
    positions = data[prefix + "position_ids"]
    length = data["teacher_response" if teacher else "responses"].shape[-1]
    if positions.dim() != 2 or critic.ulysses_sequence_parallel_size != 1:
        raise ValueError("online rubric v1 supports text-only, SP=1")
    with torch.autocast(device_type=critic.device_name, dtype=torch.bfloat16):
        if critic.use_remove_padding:
            compact, indices, *_ = unpad_input(ids.unsqueeze(-1), attention)
            compact = compact.transpose(0, 1)
            compact_positions = index_first_axis(positions.reshape(-1, 1), indices).transpose(0, 1)
            output = critic.critic_module(input_ids=compact, attention_mask=None,
                                          position_ids=compact_positions, use_cache=False).logits.squeeze(0)
            logits = pad_input(output, indices, batch=ids.shape[0], seqlen=ids.shape[1])
        else:
            logits = critic.critic_module(input_ids=ids, attention_mask=attention,
                                          position_ids=positions, use_cache=False).logits
    return logits[:, -length:, :], attention[:, -length:]


def reward_forward(critic, data, teacher):
    logits, mask = forward_channels(critic, data, teacher)
    last = last_response_logits(logits, mask)
    positions = torch.arange(mask.shape[-1], device=mask.device)
    indices = torch.where(mask.bool(), positions, -1).max(-1).values
    reward = torch.zeros_like(mask, dtype=logits.dtype)
    return reward.scatter(1, indices[:, None], last[:, :1])


def update_joint(critic, proto):
    from verl.utils.device import get_device_id
    critic.critic_module.train()
    if critic.config.use_dynamic_bsz or critic.config.ppo_epochs != 1 or critic.ulysses_sequence_parallel_size != 1:
        raise ValueError("rubric requires static batching, ppo_epochs=1 and SP=1")
    fields = ["input_ids", "responses", "attention_mask", "position_ids", "teacher_input_ids",
              "teacher_response", "teacher_attention_mask", "teacher_position_ids", "rubric_global_counts"]
    for role in ("teacher", "student"):
        fields += [role + "_rubric_targets", role + "_rubric_mask", role + "_aux_weight"]
    batch = proto.select(batch_keys=fields).batch
    size = critic.config.ppo_mini_batch_size
    micro = critic.config.ppo_micro_batch_size_per_gpu
    dp = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    metrics = {}
    def record(key, value):
        metrics.setdefault(key, []).append(float(value))
    for mini in batch.split(size):
        if len(mini) != size or size % micro:
            raise ValueError("incomplete logical minibatch is unsupported")
        counts = mini["rubric_global_counts"][0].to(get_device_id())
        if not torch.equal(mini["rubric_global_counts"], mini["rubric_global_counts"][0].expand_as(mini["rubric_global_counts"])):
            raise ValueError("global logical minibatch counts are inconsistent")
        critic.critic_optimizer.zero_grad()
        numerator = torch.zeros_like(counts)
        adv_total = acc_total = student_mean = teacher_mean = 0.
        mode=critic.config.rubric.get('loss_mode','legacy_active')
        tau=critic.config.rubric.get('count_gate_tau',4)
        interval=critic.config.rubric.get('gradient_diagnostics_interval',0)
        if interval:
            # Derive the schedule from restored optimizer state, not an
            # invocation-local counter that resets on resume.
            steps=[float(v.get('step',0)) for v in critic.critic_optimizer.state.values()]
            completed=int(max(steps,default=0))
            if completed % interval==0:
                from .gradient_diagnostics import diagnose
                for key,value in diagnose(critic,mini,forward_channels,get_device_id()).items():record(key,value)
        # FP32 master weights are preserved by the original CriticWorker.
        for data in mini.split(micro):
            data = data.to(get_device_id())
            student_logits, student_mask = forward_channels(critic, data, False)
            student = last_response_logits(student_logits, student_mask).float()
            teacher_logits, teacher_mask = forward_channels(critic, data, True)
            teacher = last_response_logits(teacher_logits, teacher_mask).float()
            adv = -torch.nn.functional.logsigmoid(teacher[:, 0] - student[:, 0]).mean()
            aux = student.sum() * 0
            for role, logits in (("student", student), ("teacher", teacher)):
                contribution, local = weighted_contribution(
                    logits[:, 1:], data[role + "_rubric_targets"], data[role + "_rubric_mask"],
                    data[role + "_aux_weight"], counts, dp, mode, tau)
                aux = aux + contribution
                numerator += local.detach() * dp
            loss = adv / (size // micro) + critic.config.rubric.loss_coef * aux
            if not torch.isfinite(loss):
                raise ValueError("nonfinite joint critic loss; step must be rolled back")
            loss.backward()
            adv_total += adv.detach().item() / (size // micro)
            acc_total += (teacher[:, 0] > student[:, 0]).float().mean().item() / (size // micro)
            student_mean += student[:, 0].mean().detach().item() / (size // micro)
            teacher_mean += teacher[:, 0].mean().detach().item() / (size // micro)
        norm = critic._optimizer_step()
        if not torch.isfinite(norm):
            raise ValueError("critic optimizer rejected nonfinite gradient; actor must not update")
        active = counts > 0
        per_dimension = numerator / counts.clamp_min(1)
        before, gate, after = dimension_weights(counts,mode,tau)
        auxiliary = (numerator * after).sum().item()
        record("critic/d_loss", adv_total)
        record("critic/loss_adv", adv_total)
        record("critic/d_acc", acc_total)
        record("critic/student_value_mean", student_mean)
        record("critic/teacher_value_mean", teacher_mean)
        record("critic/grad_norm", norm.detach().item())
        record("rubric/loss_aux", auxiliary)
        record("rubric/loss_aux_weighted", auxiliary * critic.config.rubric.loss_coef)
        record('critic/loss_total', adv_total + auxiliary * critic.config.rubric.loss_coef)
        record('critic/optimizer_steps', 1)
        record('critic/microbatches_per_optimizer_step', size // micro)
        for index, dimension in enumerate(critic.config.rubric.get('head_layout', ['adv', 'answer_correctness', 'reasoning_validity', 'condition_coverage'])[1:]):
            record("rubric/loss_" + dimension, per_dimension[index].item())
            record("rubric/global_count_" + dimension, counts[index].item())
            record('rubric/gate_' + dimension,gate[index].item())
            record('rubric/coefficient_before_gate_' + dimension,before[index].item())
            record('rubric/coefficient_after_gate_' + dimension,after[index].item())
            record('rubric/loss_contribution_' + dimension,(numerator[index]*after[index]).item())
        for candidate in ('legacy_active','fixed_dimension','fixed_dimension_gated'):
            record('rubric/loss_mode/'+candidate,int(mode==candidate))
    critic.critic_optimizer.zero_grad()
    return metrics


def check_model(config, rubric, local_path):
    validate_head(config, bool(rubric.get("enabled", False)), rubric.get("spec_hash"), rubric.get('version', 'math_v1'))
    from pathlib import Path
    from safetensors import safe_open
    shapes = {}
    for path in Path(local_path).glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key.startswith("score."):
                    shapes[key] = list(stream.get_slice(key).get_shape())
    if shapes.get("score.weight") != [4, config.hidden_size] or shapes.get("score.bias") not in (None, [4]):
        raise ValueError(f"four-head worker initialization requires intact score tensors: {shapes}")


def record_worker_source(config):
    import inspect
    from pathlib import Path
    from .common import digest, file_hash
    from .core import contracts
    directory = Path(__file__).parent
    actual = {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*.py"))}
    expected = config.get("extensions_code_hash")
    if expected and digest(actual) != expected:
        raise ValueError("Ray worker extension source hash differs from driver")
    print(f"[gad-extensions-worker] source={directory} hash={digest(actual)}", flush=True)
