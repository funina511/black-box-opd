#!/usr/bin/env python3
"""Real Qwen four-rank handoff, disabled regression, and 10k-token backward."""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def merge_master_checkpoint(directory):
    import torch
    from torch.distributed.tensor import DTensor
    files = sorted(Path(directory).glob('model_world_size_*_rank_*.pt'))
    states = [torch.load(path, map_location='cpu', weights_only=False, mmap=True) for path in files]
    result = {}
    for key in states[0]:
        values = [state[key] for state in states]
        first = values[0]
        if isinstance(first, DTensor):
            placement = first.placements[0]
            if placement.is_shard():
                merged = torch.cat([v.to_local() for v in values], dim=placement.dim)
                result[key] = merged[tuple(slice(0, size) for size in first.shape)].contiguous()
            elif placement.is_replicate():
                result[key] = first.to_local().clone()
            else:
                raise ValueError('partial DTensor unsupported')
        else:
            result[key] = first.clone()
    return result


def tensors(prompt_length, response_length, rank):
    import torch
    result = {}
    for teacher in (False, True):
        prefix = 'teacher_' if teacher else ''
        ids = torch.arange(prompt_length + response_length)[None, :] % 60 + 10
        ids[0, -1] = (20 if teacher else 25) + rank
        result[prefix + 'input_ids'] = ids
        result['teacher_response' if teacher else 'responses'] = ids[:, -response_length:]
        result[prefix + 'attention_mask'] = torch.ones_like(ids)
        result[prefix + 'position_ids'] = torch.arange(ids.shape[1])[None, :]
    return result


def main():
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoModelForTokenClassification
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', default=str(ROOT / 'gad/.backends_gad_extensions_v1/gad'))
    p.add_argument('--source-critic', required=True)
    p.add_argument('--hf-critic', required=True)
    p.add_argument('--output', default=str(ROOT / 'runs/_online_contracts/model_handoff.json'))
    args = p.parse_args()
    sys.path.insert(0, args.backend)
    from verl import DataProto
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.experimental.gad_extensions.worker import forward_channels, update_joint
    from verl.experimental.gad_extensions.core.heads import last_response_logits
    from verl.workers.critic.dp_critic import DataParallelPPOCritic
    from verl.utils.fsdp_utils import init_fn, get_init_weight_context_manager
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    world = dist.get_world_size()
    if world != 4:
        raise ValueError('requires the formal four-rank geometry')
    cfg = AutoConfig.from_pretrained(args.hf_critic, local_files_only=True)
    if cfg.num_labels != 4:
        raise ValueError('handoff lost auxiliary heads')
    def load_model():
        model = AutoModelForTokenClassification.from_pretrained(
            args.hf_critic, torch_dtype=torch.float32, local_files_only=True, attn_implementation='flash_attention_2')
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1)
        return model
    def facade(model):
        return SimpleNamespace(critic_module=model, use_remove_padding=True, ulysses_sequence_parallel_size=1, device_name='cuda')
    fixture = {k: v.cuda() for k, v in tensors(8, 16, 0).items()}
    master_logits = None
    if rank == 0:
        master = load_model()
        master.load_state_dict(merge_master_checkpoint(args.source_critic), strict=True)
        from safetensors import safe_open
        with safe_open(str(Path(args.hf_critic) / 'model.safetensors'), framework='pt') as exported:
            state = master.state_dict()
            if set(exported.keys()) != set(state):
                raise AssertionError('handoff state keys changed')
            for key in exported.keys():
                torch.testing.assert_close(exported.get_tensor(key), state[key].to(torch.bfloat16), rtol=0, atol=0)
        # Compare the same BF16 parameter precision used by the actual worker;
        # standalone FP32 RMSNorm plus autocast is a different computation.
        master.cuda().to(torch.bfloat16).eval()
        with torch.no_grad():
            logits, mask = forward_channels(facade(master), fixture, False)
            master_logits = last_response_logits(logits, mask).float().cpu()
        del master
        torch.cuda.empty_cache()
    dist.barrier()
    raw = load_model().cuda().to(torch.bfloat16).eval()
    with torch.no_grad():
        logits, mask = forward_channels(facade(raw), fixture, False)
        hf_logits = last_response_logits(logits, mask).float().cpu()
    raw.float()  # Restore FP32 masters before FSDP performs BF16 forward casts.
    # Exercise the real worker's rank-zero load / other-rank meta broadcast path.
    # Qwen3's tied-embedding config normally selects CPU initialization instead.
    if rank != 0:
        del raw
        torch.cuda.empty_cache()
        with get_init_weight_context_manager(use_meta_tensor=True)():
            raw = AutoModelForTokenClassification.from_config(cfg, attn_implementation='flash_attention_2')
        if not raw.score.weight.is_meta or raw.score.weight.shape[0] != 4:
            raise AssertionError('four-head meta construction failed')
        apply_monkey_patch(raw, use_remove_padding=True, ulysses_sp_size=1)
        raw.eval()
    model = FSDP(raw, device_id=rank, sync_module_states=True, param_init_fn=init_fn,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32))
    with torch.no_grad():
        logits, mask = forward_channels(facade(model), fixture, False)
        loaded_logits = last_response_logits(logits, mask).float().cpu()
    torch.testing.assert_close(hf_logits, loaded_logits, atol=.001, rtol=.001)
    if rank == 0:
        torch.testing.assert_close(master_logits, hf_logits, atol=.001, rtol=.001)
    handoff = {'master_logits': master_logits.tolist() if rank == 0 else None,
               'hf_logits': hf_logits.tolist(), 'loaded_logits': loaded_logits.tolist(),
               'hf_to_fsdp_max_error': (hf_logits - loaded_logits).abs().max().item(),
               'master_to_hf_max_error': (master_logits - hf_logits).abs().max().item() if rank == 0 else None,
               'meta_rank_broadcast': True,
               'all_exported_weights_exact_bf16': True,
               'tolerance': {'atol': .001, 'rtol': .001}}
    # Exercise the original source and extension source with both switches disabled.
    del model, raw
    torch.cuda.empty_cache()
    raw = load_model()
    old = raw.score
    raw.score = nn.Linear(old.in_features, 1, bias=old.bias is not None)
    with torch.no_grad():
        raw.score.weight.copy_(old.weight[:1])
        if old.bias is not None: raw.score.bias.copy_(old.bias[:1])
    raw.num_labels = 1
    model = FSDP(raw.cuda(), device_id=rank, sync_module_states=True,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32))
    legacy_path = ROOT / 'gad/.backends/gad/verl/workers/critic/dp_critic.py'
    spec = importlib.util.spec_from_file_location('legacy_dp_critic_fixture', legacy_path)
    legacy = importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    config = OmegaConf.create({'model': {'use_remove_padding': True}, 'use_dynamic_bsz': False,
                              'ppo_epochs': 1, 'ppo_mini_batch_size': 1, 'ppo_micro_batch_size_per_gpu': 1,
                              'grad_clip': .2, 'rubric': {'enabled': False}})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    proto = DataProto.from_dict(tensors=tensors(8, 16, rank))
    captured = []
    def capture():
        captured.append(torch.cat([parameter.grad.reshape(-1).clone() for parameter in model.parameters()]))
        return torch.tensor(0., device=f'cuda:{rank}')
    metrics = []
    for cls in (legacy.DataParallelPPOCritic, DataParallelPPOCritic):
        critic = cls(config, model, optimizer);critic._optimizer_step = capture
        metrics.append(critic.update_critic(proto))
    error = (captured[0] - captured[1]).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if error.item() != 0 or metrics[0]['critic/d_loss'] != metrics[1]['critic/d_loss']:
        raise AssertionError('disabled loss/gradient regression')
    disabled = {'gradient_max_error': error.item(), 'loss_legacy': metrics[0]['critic/d_loss'],
                'loss_extensions_disabled': metrics[1]['critic/d_loss']}
    del model, raw, optimizer, critic, captured
    torch.cuda.empty_cache()
    # Actual 0.6B, full 2048+8192 budget, static microbatch=1, original FSDP precision.
    raw = load_model().cuda()
    model = FSDP(raw, device_id=rank, sync_module_states=True,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    inputs = tensors(2048, 8192, rank)
    for role in ('teacher', 'student'):
        inputs[role + '_rubric_targets'] = torch.tensor([[1., .5, 1.] if role == 'teacher' else [0., 0., 0.]])
        inputs[role + '_rubric_mask'] = torch.tensor([[1., 1., 1.] if role == 'teacher' else [1., 0., 0.]])
        inputs[role + '_aux_weight'] = torch.ones(1)
    inputs['rubric_global_counts'] = torch.tensor([[8., 4., 4.]])
    parameter_change = []
    def step():
        norm = model.clip_grad_norm_(.2)
        before = next(model.parameters()).detach().clone()
        optimizer.step()
        delta = (next(model.parameters()).detach() - before).abs().max()
        dist.all_reduce(delta, op=dist.ReduceOp.MAX)
        parameter_change.append(delta.item())
        return norm
    critic = facade(model)
    critic.config = OmegaConf.create({'use_dynamic_bsz': False, 'ppo_epochs': 1, 'ppo_mini_batch_size': 1,
                                     'ppo_micro_batch_size_per_gpu': 1, 'rubric': {'loss_coef': .1}})
    critic.critic_optimizer = optimizer;critic._optimizer_step = step
    torch.cuda.reset_peak_memory_stats()
    metrics = update_joint(critic, DataProto.from_dict(tensors=inputs))
    if parameter_change[0] == 0:
        raise AssertionError('no real tiny-lr update')
    long_case = {'prompt_tokens': 2048, 'response_tokens': 8192, 'total_input_tokens': 10240,
                 'parameter_max_update': parameter_change[0], 'loss_adv': metrics['critic/loss_adv'][0],
                 'loss_aux': metrics['rubric/loss_aux'][0], 'gradient_norm': metrics['critic/grad_norm'][0],
                 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    if rank == 0:
        output = Path(args.output);output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'passed': True, 'world_size': world, 'source_critic': args.source_critic,
                                     'hf_critic': args.hf_critic, 'handoff': handoff, 'disabled': disabled,
                                     'long_backward': long_case, 'fixture_labels': 'synthetic engineering labels only'}, indent=2))
        print(output.read_text())
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':main()
