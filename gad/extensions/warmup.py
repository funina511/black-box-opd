"""Dependencies of the pinned Warmup worker's single-turn Teacher SFT update."""
from functools import lru_cache
import json
from pathlib import Path

from .actor_reward import components_enabled


@lru_cache(maxsize=8)
def _standard_qwen3(path, architecture):
    # Custom model eval forwards can consume RNG or update persistent buffers.
    config_file = Path(path) / "config.json"
    if not config_file.is_file():
        return False
    model = json.loads(config_file.read_text())
    # Dynamic RoPE can update frequency buffers even in eval mode.
    return (model.get("model_type") == "qwen3" and not model.get("rope_scaling") and not model.get("auto_map")
            and model.get("architectures") == [architecture])


def skip_policy_diagnostics(config, batch):
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    actor_model = config.actor_rollout_ref.model
    critic_model = config.critic.model
    return (
        actor.strategy in ("fsdp", "fsdp2") and config.critic.strategy in ("fsdp", "fsdp2")
        and actor.policy_loss.loss_mode == "vanilla" and actor.entropy_coeff == 0
        and config.algorithm.adv_estimator == "grpo" and not config.algorithm.use_kl_in_reward
        and not rollout.multi_turn.enable and "multi_modal_inputs" not in batch.non_tensor_batch
        and "rollout_log_probs" not in batch.batch
        and not config.trainer.get("warmup_forward_diagnostics", False)
        and not config.trainer.rollout_data_dir and not config.reward_model.launch_reward_fn_async
        and not config.trainer.gad_extensions.get("student_scorer_monitor", False)
        and not components_enabled(config.critic.rubric)
        and not actor_model.external_lib and not critic_model.external_lib
        and not actor_model.override_config and not critic_model.override_config
        and _standard_qwen3(actor_model.path, "Qwen3ForCausalLM")
        and _standard_qwen3(critic_model.path, "Qwen3ForTokenClassification")
    )


def length_metrics(batch):
    from verl.trainer.ppo.metric_utils import _compute_response_info

    info = _compute_response_info(batch)
    response_width = batch.batch["responses"].shape[-1]
    widths = {"response_length": response_width,
              "prompt_length": batch.batch["attention_mask"].shape[-1] - response_width}
    metrics = {}
    for name, width in widths.items():
        lengths = info[name]
        metrics.update({name + "/mean": lengths.mean().item(), name + "/max": lengths.max().item(),
                        name + "/min": lengths.min().item(), name + "/clip_ratio": (lengths == width).float().mean().item()})
    return metrics
