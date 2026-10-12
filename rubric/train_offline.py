"""Single-GPU grouped critic training with exact logical-batch denominators."""
from __future__ import annotations
import math
import os
import random
import shutil
import time
from rubric.common import *


def batch_records(batch):
    return {rid: record for group in batch for rid, record in group["records"].items()}


def backward_batch(model, encoded, batch, mode, weight, device, diagnose=False):
    import torch
    from rubric.model import score_response
    from rubric.losses import pair_loss, label_tensors, auxiliary_contribution
    records = batch_records(batch)
    targets, masks = {}, {}
    for rid, record in records.items():
        target, mask = label_tensors(record["annotation"], mode)
        targets[rid] = torch.tensor(target, device=device)
        masks[rid] = torch.tensor(mask, device=device)
    counts = torch.stack(list(masks.values())).sum(0)
    total_pairs = sum(len(group["student_ids"]) for group in batch)
    seen, adv_sum, aux_sum, diagnostics = set(), 0.0, 0.0, {}
    for group in batch:
        for sid in group["student_ids"]:
            tid = group["teacher_id"]
            teacher = score_response(model, encoded[tid], device)
            student = score_response(model, encoded[sid], device)
            adv = pair_loss(teacher[0], student[0]) / total_pairs
            auxiliary = adv * 0
            for rid, logits in ((tid, teacher), (sid, student)):
                if rid not in seen:
                    auxiliary = auxiliary + auxiliary_contribution(logits[1:], targets[rid], masks[rid], counts)
                    seen.add(rid)
            loss = adv + weight * auxiliary
            if not torch.isfinite(loss):
                raise ValueError("non-finite offline training loss")
            if diagnose and not diagnostics:
                # Measure both losses on the SAME first pair graph. This is a sampled
                # gradient diagnostic, explicitly not a full-step decomposition.
                parameters = [p for p in model.parameters() if p.requires_grad]
                for name, part in (("adv", adv), ("auxiliary_weighted", weight * auxiliary)):
                    gradients = torch.autograd.grad(part, parameters, retain_graph=True, allow_unused=True)
                    norm = sum(float(g.detach().float().square().sum()) for g in gradients if g is not None) ** .5
                    diagnostics[name + "_sample_gradient_norm"] = norm
                    del gradients
            loss.backward()
            adv_sum += float(adv.detach())
            aux_sum += float(auxiliary.detach())
    return {"adv_loss": adv_sum, "auxiliary_loss": aux_sum, "total_loss": adv_sum + weight * aux_sum,
            "unique_responses": len(records), "pairs": total_pairs, "valid_counts": counts.tolist(),
            "gradient_diagnostic_scope": "first pair, logical-step normalization", **diagnostics}


def save_checkpoint(directory, model, tokenizer, optimizer, scheduler, state):
    import numpy as np
    import torch
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"checkpoint already exists: {directory}")
    temporary = directory.with_name(directory.name + ".incomplete")
    if temporary.exists():
        raise ValueError(f"incomplete checkpoint exists: {temporary}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()},
               temporary / "training_state.pt")
    write_json(temporary / "offline_manifest.json",
               {key: value for key, value in state.items() if key not in ("optimizer","scheduler")})
    os.replace(temporary, directory)
    return directory


def restore_training_state(optimizer, scheduler, restored):
    import numpy as np
    import torch
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    random.setstate(restored["rng_python"])
    np.random.set_state(restored["rng_numpy"])
    torch.set_rng_state(restored["rng_torch"])
    torch.cuda.set_rng_state_all(restored["rng_cuda"])


def prune_checkpoints(directory, state, keep=5):
    import re
    checkpoints = sorted(p for p in Path(directory).glob("step_*")
                         if re.fullmatch(r"step_[0-9]+",p.name) and p.is_dir())
    protected = set(checkpoints[-keep:])
    if state.get("best_checkpoint"):
        protected.add(Path(state["best_checkpoint"]))
    for path in checkpoints:
        if path in protected:
            continue
        if path.is_symlink() or read_json(path/"offline_manifest.json")["identity"] != state["identity"]:
            raise ValueError("refusing to prune an unrelated checkpoint")
        shutil.rmtree(path)


def train(cfg, variant, resume=None, max_steps=None, exploratory=False):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from rubric.model import export_single_head, load_classifier, expand_head, encode_response, score_response
    from rubric.data import load_groups
    from rubric.evaluate import predict_groups, metrics
    if variant not in ("adv_only","correctness_aux","joint"):
        raise ValueError("train variant must be adv_only, correctness_aux or joint")
    run, dataset = claim_run(cfg)
    manifest = read_json(dataset / "manifest.json")
    if manifest["packed"]["exploratory"] and not exploratory:
        raise ValueError("labels lack human verification; explicitly pass --exploratory")
    directory = run / "variants" / variant
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "selection.json").exists() and read_json(directory / "selection.json")["training_complete"]:
        return read_json(directory / "selection.json")
    packed_hash = manifest["packed"]["data_hash"]
    train_groups, dev_groups = load_groups(dataset, "train"), load_groups(dataset, "dev")
    if not train_groups or not dev_groups:
        raise ValueError("nonempty train and dev groups required after quarantine")
    options = cfg["training"]
    if options["dtype"] != "bfloat16":
        raise ValueError("v1 verified training dtype is bfloat16")
    rubric_hash = file_hash(cfg["rubric"]["spec"])
    identity = digest({"config": cfg, "dataset": packed_hash, "variant": variant, "code": code_fingerprint()})
    random.seed(cfg["run"]["seed"])
    np.random.seed(cfg["run"]["seed"])
    torch.manual_seed(cfg["run"]["seed"])
    torch.cuda.manual_seed_all(cfg["run"]["seed"])
    source = export_single_head(cfg["source"]["critic_checkpoint"], run / "exports/critic")
    state = {"identity": identity, "variant": variant, "dataset_hash": packed_hash,
             "epoch": 0, "offset": 0, "step": 0, "order": None, "best_objective": None,
             "best_checkpoint": None, "dev_history": [], "rubric_hash": rubric_hash,
             "fresh_offline_optimizer": True}
    if resume == "auto" and not (directory / "latest.json").exists():
        resume = None
    if resume:
        checkpoint = Path(read_json(directory / "latest.json")["checkpoint"]) if resume == "auto" else Path(resume)
        restored = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        if restored["identity"] != identity:
            raise ValueError("resume code/config/data/variant identity changed")
        state = {key: value for key, value in restored.items() if not key.startswith("rng_")
                 and key not in ("optimizer","scheduler")}
        model = load_classifier(checkpoint, torch.float32, rubric_hash)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    else:
        if (directory / "latest.json").exists():
            raise ValueError("unfinished run exists; use --resume auto")
        # FP32 master parameters and optimizer states avoid rounding away 1e-6
        # updates. Forward uses BF16 autocast in score_response.
        model = load_classifier(source, torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        longest = max((r["trajectory"] for g in train_groups for r in g["records"].values()),
                      key=lambda r: r["original_response_tokens"])
        encoded_longest = encode_response(tokenizer, longest, manifest["source"]["generation"])
        model.to("cuda").eval()
        with torch.inference_mode():
            before = score_response(model, encoded_longest, "cuda")[0].clone()
        expand_head(model, cfg["run"]["seed"], rubric_hash)
        with torch.inference_mode():
            after = score_response(model, encoded_longest, "cuda")[0]
        difference = float((before-after).abs())
        # BF16 GEMM shape changes can change rounding, though the saved row is exact.
        if difference > max(.02, abs(float(before)) * .01):
            raise ValueError(f"adv initialization score drift: {difference}")
        write_json(directory / "head_migration.json", {"adv_before": float(before), "adv_after": float(after),
                                                       "absolute_difference": difference, "longest_tokens": len(encoded_longest["input_ids"]),
                                                       "weight_row_copied_exactly": True})
    model.to("cuda")
    state["precision"] = {"master_parameters": "float32", "forward_autocast": "bfloat16",
                          "optimizer_states": "float32"}
    model.config.use_cache = False
    if options["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], betas=tuple(options["betas"]),
                                 weight_decay=options["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if resume:
        restore_training_state(optimizer, scheduler, restored)
        del restored
    encoded = {rid: encode_response(tokenizer, rec["trajectory"], manifest["source"]["generation"])
               for rid, rec in batch_records(train_groups).items()}
    weight = 0.0 if variant == "adv_only" else options["rubric_lambda"]
    batch_size = options["logical_batch_prompts"]
    stop_at = state["step"] + max_steps if max_steps else math.inf
    torch.cuda.reset_peak_memory_stats()
    while state["epoch"] < options["max_epochs"]:
        if state["order"] is None:
            state["order"] = list(range(len(train_groups)))
            random.Random(cfg["run"]["seed"] + state["epoch"]).shuffle(state["order"])
        while state["offset"] < len(state["order"]):
            start = state["offset"]
            batch = [train_groups[index] for index in state["order"][start:start+batch_size]]
            started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            stats = backward_batch(model, encoded, batch, variant, weight, "cuda", diagnose=state["step"] == 0)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options["grad_clip"])
            if not torch.isfinite(norm):
                raise ValueError("non-finite gradient norm")
            optimizer.step()
            scheduler.step()
            state["step"] += 1
            state["offset"] += len(batch)
            stats.update(step=state["step"], epoch=state["epoch"], seconds=time.monotonic()-started,
                         gradient_norm=float(norm), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                         response_tokens=sum(encoded[rid]["visible_response_tokens"] for rid in batch_records(batch)))
            stats["unique_responses_per_second"] = stats["unique_responses"]/stats["seconds"]
            append_jsonl(directory / "training.jsonl", stats)
            print(f"[rubric] {variant} step={state['step']} loss={stats['total_loss']:.5f} seconds={stats['seconds']:.1f}", flush=True)
            if state["offset"] < len(state["order"]) and (
                    state["step"] % options["save_every_steps"] == 0 or state["step"] >= stop_at):
                checkpoint = save_checkpoint(directory / f"step_{state['step']:06d}", model, tokenizer, optimizer, scheduler, state)
                write_json(directory / "latest.json", {"checkpoint": str(checkpoint)})
                prune_checkpoints(directory,state)
                if state["step"] >= stop_at:
                    return {"training_complete": False, "checkpoint": str(checkpoint), "step": state["step"]}
        prediction = predict_groups(model, tokenizer, dev_groups, manifest["source"]["generation"], "cuda", variant)
        dev = metrics(prediction)
        # Predeclared: adv loss for B1; adv + lambda mean valid auxiliary MAE for B2/B3.
        errors = [item["mae"] for item in dev["auxiliary"].values()]
        objective = dev["adv_loss"] + weight * (sum(errors)/len(errors) if errors else 0)
        state["dev_history"].append({"epoch": state["epoch"] + 1, "step": state["step"], "objective": objective, "metrics": dev})
        improved = state["best_objective"] is None or objective < state["best_objective"]
        checkpoint = directory / f"step_{state['step']:06d}"
        if improved:
            state["best_objective"], state["best_checkpoint"] = objective, str(checkpoint)
        state["epoch"] += 1
        state["offset"], state["order"] = 0, None
        save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, state)
        write_json(directory / "latest.json", {"checkpoint": str(checkpoint)})
        prune_checkpoints(directory,state)
        selection = {"training_complete": state["epoch"] >= options["max_epochs"],
                     "checkpoint": state["best_checkpoint"], "selection_split": "dev",
                     "selection_rule": "adv_loss + lambda * mean valid auxiliary MAE",
                     "best_objective": state["best_objective"], "dev_history": state["dev_history"],
                     "identity": identity, "step": state["step"]}
        write_json(directory / "selection.json", selection)
        if state["step"] >= stop_at:
            return selection
    return read_json(directory / "selection.json")
