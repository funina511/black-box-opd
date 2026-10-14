#!/usr/bin/env python3
"""Four-rank FSDP gradient/packing contract. Run with torchrun, never unittest."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from omegaconf import OmegaConf
    from transformers.modeling_outputs import TokenClassifierOutput
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=str(ROOT / "gad/.backends_gad_extensions_v1/gad"))
    parser.add_argument("--output", default=str(ROOT / "runs/_online_contracts/gpu_contract.json"))
    args = parser.parse_args()
    sys.path.insert(0, args.backend)
    from verl import DataProto
    from verl.experimental.gad_extensions.worker import update_joint, reward_forward
    from verl.experimental.gad_extensions.critic_batch import auxiliary_weights
    from verl.experimental.gad_extensions.core.heads import last_response_logits
    from verl.experimental.gad_extensions.core.losses import weighted_contribution
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    if world != 4:
        raise ValueError("this acceptance contract requires four FSDP ranks")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 16)
            self.score = nn.Linear(16, 4)
        def forward(self, input_ids, **kwargs):
            return TokenClassifierOutput(logits=self.score(self.embedding(input_ids)))

    results = []
    for remove_padding in (False, True):
        for micro in (1, 2):
            torch.manual_seed(42)
            raw = Model()
            reference = Model();reference.load_state_dict(raw.state_dict())
            reference.eval()
            model = FSDP(raw.cuda(), device_id=rank,
                         mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                        buffer_dtype=torch.float32), sync_module_states=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0)
            pairs = [{"teacher": {"response_id": "T"}, "student": {"response_id": "S" + str(i // 2)}} for i in range(8)]
            targets = {r: torch.tensor([[1., .5, 1.] if r == "teacher" else [0., .5, 0.] for i in range(8)]) for r in ("teacher", "student")}
            masks = {r: torch.ones(8, 3) for r in targets}
            masks["student"][4:, 1:] = 0
            if remove_padding and micro == 2:
                masks["teacher"][:] = 0
                masks["student"][4:, :] = 0  # Two ranks contribute no auxiliary supervision.
            weights, counts = auxiliary_weights(pairs, targets, masks, 8, 4)
            tensors = {}
            for role, prefix, response_name in (("student", "", "responses"), ("teacher", "teacher_", "teacher_response")):
                responses = [[7 + i // 2, 9, 0, 0] if role == "student" else [3, 4, 0, 0] for i in range(8)]
                tensors[response_name] = torch.tensor(responses)
                tensors[prefix + "input_ids"] = torch.tensor([[0, 1, 2, *row] for row in responses])
                tensors[prefix + "attention_mask"] = torch.tensor([[0, 1, 1, 1, 1, 0, 0]] * 8)
                tensors[prefix + "position_ids"] = torch.tensor([[0, 0, 1, 2, 3, 4, 5]] * 8)
                tensors[role + "_rubric_targets"] = targets[role]
                tensors[role + "_rubric_mask"] = masks[role]
                tensors[role + "_aux_weight"] = weights[role]
            tensors["rubric_global_counts"] = counts
            full = DataProto.from_dict(tensors=tensors)
            local = full[rank * 2:(rank + 1) * 2]
            reference.zero_grad()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                student = reference(full.batch['input_ids']).logits[:, 4, :].float()
                teacher = reference(full.batch['teacher_input_ids']).logits[:, 4, :].float()
            adv = -torch.nn.functional.logsigmoid(teacher[:, 0] - student[:, 0]).mean()
            aux = student.sum() * 0
            for role, logits in (("student", student), ("teacher", teacher)):
                term, _ = weighted_contribution(logits[:, 1:], targets[role], masks[role], weights[role], counts[0])
                aux = aux + term
            (adv + .1 * aux).backward()
            expected_gradient = torch.cat([p.grad.reshape(-1) for p in reference.parameters()])
            observed = []
            def step():
                local_gradient = next(model.parameters()).grad.detach().clone()
                pieces = [torch.empty_like(local_gradient) for _ in range(world)]
                dist.all_gather(pieces, local_gradient)
                gradient = torch.cat(pieces).cpu()[:expected_gradient.numel()]
                error = (gradient - expected_gradient).abs().max().item()
                torch.testing.assert_close(gradient, expected_gradient, atol=.004, rtol=.035)
                norm = model.clip_grad_norm_(.2)
                before = next(model.parameters()).detach().clone()
                optimizer.step()
                difference = (next(model.parameters()).detach() - before).abs().max()
                dist.all_reduce(difference, op=dist.ReduceOp.MAX)
                if difference.item() == 0:
                    raise AssertionError("tiny-lr FP32 master parameter update rounded away")
                observed.append({"gradient_max_error": error, "parameter_update": difference.item(), "master_dtype": str(before.dtype)})
                return norm
            critic = SimpleNamespace(config=OmegaConf.create({'use_dynamic_bsz':False,'ppo_epochs':1,
                        'ppo_mini_batch_size':2,'ppo_micro_batch_size_per_gpu':micro,'rubric':{'loss_coef':.1}}),
                        critic_module=model,critic_optimizer=optimizer,use_remove_padding=remove_padding,
                        ulysses_sequence_parallel_size=1,device_name='cuda',_optimizer_step=step)
            metrics = update_joint(critic, local)
            data = {key: value.cuda() for key, value in local.batch.items()}
            with torch.no_grad():
                rewards = reward_forward(critic, data, False)
            if rewards.shape != (2, 4) or rewards[:, [0, 2, 3]].abs().sum().item() != 0:
                raise AssertionError("external reward includes non-final tokens/auxiliary channels")
            results.append({'remove_padding':remove_padding,'micro':micro,**observed[0],
                            'loss_adv':metrics['critic/loss_adv'][0]})
            del model, optimizer, critic
            torch.cuda.empty_cache()
    if rank == 0:
        path = Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'world_size':world,'backend':args.backend,'passed':True,'cases':results},indent=2))
        print(path.read_text())
    dist.barrier();dist.destroy_process_group()


if __name__ == '__main__':main()
