"""One driver instance per stage; all mutation commits after successful model updates."""
import json
from pathlib import Path
from .common import plain, append_json
from .critic_batch import extract_pairs, pack_pairs, FIELDS
from .replay_buffer import ReplayBuffer


class Runtime:
    def __init__(self, trainer):
        config = trainer.config
        self.trainer = trainer
        self.options = plain(config.trainer.gad_extensions)
        self.identity = self.options["identity"]
        self.stage = self.options["stage"]
        self.world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
        self.directory = Path(config.trainer.default_local_dir)
        self.stage_start = self.options["start_step"]
        self.stage_step = 0
        self.committed_step = self.stage_start
        self.invocation_steps = 0
        self.rubric = None
        if config.trainer.rubric.enabled:
            from .online_rubric import OnlineRubric
            self.rubric = OnlineRubric(plain(config.trainer.rubric), self.directory)
        self.replay = ReplayBuffer(plain(config.trainer.critic_replay)) if config.trainer.critic_replay.enabled else None
        self.teacher_provenance = {}
        if self.options.get("teacher_index"):
            for line in Path(self.options["teacher_index"]).open():
                record = json.loads(line)
                if not record.get("_header"):
                    uid = str(record.get("uid", record.get("index", "")))
                    self.teacher_provenance[uid] = {"sampling": record.get("sampling", {}),
                                                    'candidate_hashes': [__import__('hashlib').sha256(text.encode()).hexdigest() for text in record.get('teacher_answers', [])],
                                                    "finish_reason": record.get("finish_reason"),
                                                    "truncated": record.get("truncated", False)}
        self.pending = None
        self.calibration_ids = set()
        calibration = self.directory / "rubric_calibration.jsonl"
        if calibration.exists():
            self.calibration_ids = {json.loads(line)["response_id"] for line in calibration.read_text().splitlines()}

    def prepare(self, actor_batch, step):
        # Persist no advantages/logprobs/scores; the policy batch remains untouched.
        fresh = extract_pairs(actor_batch, self.trainer.tokenizer, self.options["token_identity"], step, self.stage)
        for pair in fresh:
            source = self.teacher_provenance.get(pair["source_index"], {})
            sampling = source.get("sampling", {})
            selected = pair['teacher'].get('selected_teacher_index')
            hashes = source.get('candidate_hashes', [])
            if selected is None:
                if len(hashes) > 1:
                    raise ValueError('teacher multi-candidate provenance missing; migrate data before training')
                selected = 0
            if hashes and (selected >= len(hashes) or pair['teacher'].get('selected_teacher_text_sha256', hashes[selected]) != hashes[selected]):
                raise ValueError('selected teacher text hash differs from source candidate')
            def source_field(name, default):
                values = sampling.get(name) or []
                return values[selected] if selected < len(values) else default
            finish = pair['teacher'].get('source_finish_reason', source_field('finish_reasons', source.get('finish_reason')))
            truncated = bool(pair['teacher'].get('source_truncated', source_field('truncated', source.get('truncated', False)))) or finish == 'length'
            pair["teacher"]["source_truncated"] = truncated
            pair["teacher"]["truncated"] |= truncated
            pair["teacher"]["source_finish_reason"] = finish
        stage_step = self.stage_step + 1
        chosen, sampled = self.replay.select(fresh, stage_step) if self.replay else (fresh, 0)
        metadata = dict(actor_batch.meta_info)
        metadata["gad_pad_token_id"] = self.trainer.tokenizer.pad_token_id
        # A shallow DataProto view carries width/meta; rebuilding creates new tensors.
        from verl import DataProto
        template = DataProto(batch=actor_batch.batch, non_tensor_batch=actor_batch.non_tensor_batch, meta_info=metadata)
        critic_batch = pack_pairs(chosen, template)
        stats = {"replay/sampled_pairs": sampled, "replay/realized_ratio": sampled / len(fresh),
                 "replay/current_tokens": sum(len(p["prompt_ids"]) * 2 + len(p["teacher"]["token_ids"]) + len(p["student"]["token_ids"])
                                              for p in chosen if p["source_step"] == step),
                 "replay/historical_tokens": sum(len(p["prompt_ids"]) * 2 + len(p["teacher"]["token_ids"]) + len(p["student"]["token_ids"])
                                                 for p in chosen if p["source_step"] != step)}
        ages = [stage_step - p["stage_step"] for p in chosen if p["source_step"] != step]
        stats["replay/age_mean"] = sum(ages) / len(ages) if ages else 0
        stats["replay/age_max"] = max(ages) if ages else 0
        if self.rubric:
            global_mini = self.trainer.config.critic.ppo_mini_batch_size * self.trainer.config.actor_rollout_ref.rollout.n
            stats.update(self.rubric.annotate(chosen, critic_batch, global_mini, self.world_size, step))
            # Selected fresh labels are retained; other fresh pairs stay labels_pending.
            labels = {p["pair_id"]: p.get("labels") for p in chosen}
            for pair in fresh:
                if labels.get(pair["pair_id"]):
                    pair["labels"] = labels[pair["pair_id"]]
            for pair in chosen:
                for role in ("teacher", "student"):
                    rid = pair[role]["response_id"]
                    if len(self.calibration_ids) < 32 and rid not in self.calibration_ids:
                        append_json(self.directory / "rubric_calibration.jsonl", {
                            "step": step, "response_id": rid, "source_role": role, "prompt": pair["prompt"],
                            "ground_truth": pair["ground_truth"], "response": pair[role]["response"],
                            "tokens": len(pair[role]["token_ids"]), "truncated": pair[role]["truncated"],
                            "annotation": pair["labels"][role], "human_review": "pending"})
                        self.calibration_ids.add(rid)
        self.pending = (fresh, stage_step, step, stats)
        return critic_batch, stats

    def commit(self, actor_updated):
        if self.pending is None:
            raise ValueError("no prepared extension step")
        fresh, stage_step, step, stats = self.pending
        if self.replay:
            self.replay.commit(fresh, stage_step)
            stats.update({"replay/size_pairs": len(self.replay.entries), "replay/serialized_bytes": self.replay.bytes,
                          "replay/evicted_age": self.replay.evicted_age,
                          "replay/evicted_capacity": self.replay.evicted_capacity})
        self.stage_step, self.committed_step = stage_step, step
        self.invocation_steps += 1
        stats.update({"training/actor_updated": int(actor_updated), "training/critic_updated": 1,
                      "training/stage_step": stage_step, "training/absolute_global_step": step,
                      "training/rollout_sampling_seed": self.options["sampling_seed"]})
        append_json(self.directory / "extension_metrics.jsonl", {"step": step, "stage": self.stage, **stats})
        self.pending = None
        return stats
