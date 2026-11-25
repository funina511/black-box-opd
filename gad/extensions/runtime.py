"""One driver instance per stage; all mutation commits after successful model updates."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
            with Path(self.options["teacher_index"]).open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not record.get("_header"):
                        uid = str(record.get("uid", record.get("index", "")))
                        self.teacher_provenance[uid] = {"sampling": record.get("sampling", {}),
                                                        'candidate_hashes': [__import__('hashlib').sha256(text.encode()).hexdigest() for text in record.get('teacher_answers', [])],
                                                        "finish_reason": record.get("finish_reason"),
                                                        "truncated": record.get("truncated", False)}
        self.execution = self.options.get('execution', {})
        self.executor = None
        self.prepared = None
        self.annotation_future = None
        self.teacher_future = None
        self.pending = None
        self.pending_reward_monitor = None
        self.reward_monitor_history = None
        self.calibration_ids = set()
        calibration = self.directory / "rubric_calibration.jsonl"
        if calibration.exists():
            self.calibration_ids = {json.loads(line)["response_id"] for line in calibration.read_text().splitlines()}

    def _fresh_pairs(self, actor_batch, step):
        # Persist no advantages/logprobs/scores; the policy batch remains untouched.
        fresh = extract_pairs(actor_batch, self.trainer.tokenizer, self.options["token_identity"], step, self.stage)
        self._attach_teacher_source(fresh)
        return fresh

    def _attach_teacher_source(self, pairs):
        for pair in pairs:
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

    def _executor(self):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gad-annotation')
        return self.executor

    @contextmanager
    def timed_phase(self, timer, name, *args, **kwargs):
        """Keep native timers and add wall intervals for the external GPU sampler."""
        path = self.directory / 'training_phases.jsonl'
        step = self.trainer.global_steps
        append_json(path, {'step': step, 'phase': name, 'event': 'start', 'epoch': time.time()})
        complete = False
        try:
            with timer(name, *args, **kwargs):
                yield
            complete = True
        finally:
            append_json(path, {'step': step, 'phase': name, 'event': 'end', 'epoch': time.time(),
                               'complete': complete})

    def prefetch_teachers(self, gen_batch, raw_batch, step):
        if not (self.rubric and self.execution.get('teacher_prefetch')):
            return
        if self.teacher_future is not None:
            raise ValueError('previous Teacher prefetch was not joined')
        from .prefetch import teacher_pairs, selected_teachers, prefetch
        try:
            if not hasattr(self, '_prefetch_eos'):
                # Match FSDPWorker exactly, including multiple EOS tokens and
                # the model-config fallback. Do not alter the rollout batch.
                from verl.utils.model import get_generation_config
                model = self.trainer.config.actor_rollout_ref.model
                generation = get_generation_config(model.path, trust_remote_code=model.get('trust_remote_code', False))
                self._prefetch_eos = generation.eos_token_id if generation is not None else self.trainer.tokenizer.eos_token_id
            pairs = teacher_pairs(gen_batch, raw_batch, self.trainer.tokenizer,
                self.options['token_identity'], step, self.stage,
                self.trainer.config.actor_rollout_ref.rollout.n, eos_token_id=self._prefetch_eos)
            # The provenance join must be exactly the one used after rollout.
            self._attach_teacher_source(pairs)
            global_mini = self.trainer.config.critic.ppo_mini_batch_size * self.trainer.config.actor_rollout_ref.rollout.n
            selected = selected_teachers(pairs, global_mini, self.world_size, self.rubric.options, step, self.stage)
            from .annotation_work import score_pairs
            scores = score_pairs(self.rubric, selected, roles=('teacher',))
            self.teacher_future = self._executor().submit(prefetch, self.rubric, selected, step, scorer_results=scores)
        except Exception as exc:
            append_json(self.directory / 'teacher_prefetch_errors.jsonl', {'step': step, 'error': str(exc)})

    def _annotate(self, chosen, critic_batch, step, scorer_results=None):
        stats = {}
        prefetch = self.teacher_future.result() if self.teacher_future is not None else None
        if self.rubric:
            global_mini = self.trainer.config.critic.ppo_mini_batch_size * self.trainer.config.actor_rollout_ref.rollout.n
            kwargs = {'scorer_results': scorer_results} if scorer_results is not None else {}
            stats.update(self.rubric.annotate(chosen, critic_batch, global_mini, self.world_size, step, **kwargs))
        if prefetch is not None:
            main_keys = getattr(self.rubric, 'last_selected_teacher_keys', set())
            stats.update({'execution/teacher_prefetch_candidates': len(prefetch['candidate_keys']),
                'execution/teacher_prefetch_hits': len(main_keys.intersection(prefetch['cached_keys'])),
                'execution/teacher_prefetch_seconds': prefetch['seconds'],
                'execution/teacher_prefetch_requests': prefetch['requests'],
                'execution/teacher_prefetch_api_sends': prefetch['api_sends']})
            for key, value in [('judge_requests', prefetch['requests']),
                    ('json_valid_count', prefetch['json_valid_count']),
                    ('json_requested_count', prefetch['json_requested_count']),
                    ('new_annotation_failures', prefetch['failures']),
                    ('judge_prompt_tokens', prefetch['prompt_tokens']),
                    ('judge_output_tokens', prefetch['output_tokens']), ('http_attempts', prefetch['api_sends']),
                    ('new_generation_sends', prefetch['api_sends']), ('health_gets_new', prefetch['health_gets'])]:
                stats['rubric/' + key] = stats.get('rubric/' + key, 0) + value
        return stats

    def begin_prepare(self, actor_batch, step):
        if self.prepared is not None:
            raise ValueError('extension preparation scheduled twice')
        fresh = self._fresh_pairs(actor_batch, step)
        stage_step = self.stage_step + 1
        chosen, sampled = self.replay.select(fresh, stage_step) if self.replay else (fresh, 0)
        metadata = dict(actor_batch.meta_info)
        metadata["gad_pad_token_id"] = self.trainer.tokenizer.pad_token_id
        # A shallow DataProto view carries width/meta; rebuilding creates new tensors.
        from verl import DataProto
        template = DataProto(batch=actor_batch.batch, non_tensor_batch=actor_batch.non_tensor_batch, meta_info=metadata)
        # Baseline monitoring is a read-only observer: preserve the original
        # Critic batch, including every tensor and its batching metadata.
        critic_batch = pack_pairs(chosen, template) if self.rubric or self.replay else actor_batch
        stats = {"replay/sampled_pairs": sampled, "replay/realized_ratio": sampled / len(fresh),
                 "replay/current_tokens": sum(len(p["prompt_ids"]) * 2 + len(p["teacher"]["token_ids"]) + len(p["student"]["token_ids"])
                                              for p in chosen if p["source_step"] == step),
                 "replay/historical_tokens": sum(len(p["prompt_ids"]) * 2 + len(p["teacher"]["token_ids"]) + len(p["student"]["token_ids"])
                                                 for p in chosen if p["source_step"] != step)}
        ages = [stage_step - p["stage_step"] for p in chosen if p["source_step"] != step]
        stats["replay/age_mean"] = sum(ages) / len(ages) if ages else 0
        stats["replay/age_max"] = max(ages) if ages else 0
        if self.execution.get('early_annotation'):
            from .annotation_work import score_pairs
            before_score = time.monotonic()
            scores = score_pairs(self.rubric, chosen)
            stats['execution/scorer_precompute_seconds'] = time.monotonic() - before_score
        self.prepared = (fresh, chosen, critic_batch, stats, stage_step, step, time.monotonic())
        if self.execution.get('early_annotation'):
            self.annotation_future = self._executor().submit(self._annotate, chosen, critic_batch, step, scores)
        else:
            stats.update(self._annotate(chosen, critic_batch, step))

    def prepare(self, actor_batch, step):
        if self.prepared is None:
            self.begin_prepare(actor_batch, step)
        fresh, chosen, critic_batch, stats, stage_step, prepared_step, started = self.prepared
        if prepared_step != step:
            raise ValueError('prepared extension step differs from optimizer step')
        before_wait = time.monotonic()
        if self.annotation_future is not None:
            stats.update(self.annotation_future.result())
        stats['execution/annotation_wait_seconds'] = time.monotonic() - before_wait
        stats['execution/annotation_lead_seconds'] = before_wait - started
        if self.rubric:
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
        if 'rubric_reward_components' in actor_batch.batch or self.options.get('student_scorer_monitor'):
            from .actor_reward import monitor_records, student_scorer_labels
            labels = (student_scorer_labels(fresh, self.options['scorer_identity'])
                      if self.rubric is None and self.options.get('student_scorer_monitor') else None)
            self.pending_reward_monitor = monitor_records(actor_batch, critic_batch, fresh, step, labels)
            stats['actor_reward/fp32'] = int(str(actor_batch.batch['token_level_rewards'].dtype) == 'torch.float32')
        self.pending = (fresh, stage_step, step, stats)
        self.prepared = self.annotation_future = self.teacher_future = None
        return critic_batch, stats

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def commit(self, actor_updated):
        if self.pending is None:
            raise ValueError("no prepared extension step")
        fresh, stage_step, step, stats = self.pending
        if self.replay:
            self.replay.commit(fresh, stage_step)
            stats.update({"replay/size_pairs": len(self.replay.entries), "replay/serialized_bytes": self.replay.bytes,
                          "replay/evicted_age": self.replay.evicted_age,
                          "replay/evicted_capacity": self.replay.evicted_capacity})
        record = {}
        if self.pending_reward_monitor is not None:
            from .actor_reward import load_window, summarize
            if self.reward_monitor_history is None:
                self.reward_monitor_history = load_window(self.directory, self.committed_step)
            self.reward_monitor_history.append(self.pending_reward_monitor)
            self.reward_monitor_history = self.reward_monitor_history[-10:]
            options = plain(self.trainer.config.trainer.rubric)
            stats.update(summarize(self.pending_reward_monitor, options))
            stats.update(summarize([row for batch in self.reward_monitor_history for row in batch], options,
                                   prefix='actor_reward/window/'))
            record['actor_reward_monitor'] = self.pending_reward_monitor
        self.stage_step, self.committed_step = stage_step, step
        self.invocation_steps += 1
        stats.update({"training/actor_updated": int(actor_updated), "training/critic_updated": 1,
                      "training/stage_step": stage_step, "training/absolute_global_step": step,
                      "training/rollout_sampling_seed": self.options["sampling_seed"]})
        append_json(self.directory / "extension_metrics.jsonl", {"step": step, "stage": self.stage, **stats, **record})
        self.pending = None
        self.pending_reward_monitor = None
        return stats
