"""Current-response supervision at the driver barrier; no actor reward changes."""
import importlib.util
import time
from pathlib import Path
from .common import DIMENSIONS, append_json, digest, file_hash
from .core.contracts import ANNOTATION_SCHEMA, judge_messages, validate_label
from .judge_client import JudgeClient
from .label_cache import LabelCache
from .critic_batch import auxiliary_weights


class OnlineRubric:
    def __init__(self, options, directory):
        self.options = options
        self.directory = Path(directory)
        self.spec_hash = file_hash(options["spec"])
        self.prompt_hash = file_hash(options["prompt"])
        if self.spec_hash != options["spec_hash"]:
            raise ValueError("rubric spec changed after launch")
        self.spec = Path(options["spec"]).read_text()
        self.instruction = Path(options["prompt"]).read_text()
        self.scorer_hash = file_hash(options["scorer_path"])
        module_spec = importlib.util.spec_from_file_location("gad_online_math_scorer", options["scorer_path"])
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.scorer = module.compute_score
        self.version = options.get('version', 'math_v1')
        if self.version == 'math_process_v2':
            from scoring.math import identity, score
            from .core import process_v2
            self.annotation_contract_hash = file_hash(process_v2.__file__)
            self.protocol_identity = process_v2.protocol_identity(self.spec, self.instruction)
            self.protocol_identity['semantic_policy'] = {key:options.get(key, default) for key,default in (('semantic_check',False),('semantic_max_extra_requests',1))}
            from .core import semantic_v24
            self.protocol_identity['semantic_module_hash'] = file_hash(semantic_v24.__file__)
            self.scoring_config = options['scoring_math']
            self.scorer_identity = identity(self.scoring_config['timeout_seconds'])
            if self.scorer_identity['fingerprint'] != self.scoring_config['fingerprint']:
                raise ValueError('V2 scorer/dependency identity changed')
            self.scorer = lambda response, gt: score(response, gt, self.scoring_config)
        self.cache = LabelCache(self.directory.parent / "rubric_label_cache")
        if self.version == 'math_process_v2':
            from .core.contracts import V2_SCHEMA_VERSION
            namespace = 'rubric_label_cache_' + V2_SCHEMA_VERSION.replace('rubric.', '').replace('.', '_')
            if options['mode'] == 'joint' and options['judge'].get('enable_thinking', False):
                namespace += '_thinking_' + options['judge_fingerprint'][:12]
            self.cache = LabelCache(self.directory.parent / namespace)
        self.judge = None
        if options["mode"] == "joint":
            self.judge = JudgeClient({**options["judge"], "seed": options["seed"], 'rubric_version': self.version}, options["judge_fingerprint"])
            try:self.judge.health()
            except Exception as exc:
                append_json(self.directory / 'judge_step_failures.jsonl', {'phase': 'initialize', 'error': str(exc),
                            'protocol_identity': getattr(self, 'protocol_identity', None), 'received_results': 0})
                raise
        self.empty_student_steps = 0
        self.v2_empty_student_process_steps = 0
        self.v2_coverage_state_version = 'student.process.coverage.v2.1'
        self.v2_counter_initialization = 'new_stage_zero'

    def cache_key(self, pair, role):
        return digest({"prompt": pair["prompt_hash"], "response": pair[role]["response_id"],
                       "GT": pair["ground_truth"], "spec": self.spec_hash, "prompt_version": self.prompt_hash,
                       "judge": self.options["judge_fingerprint"], "scorer": self.scorer_hash,
                       "prompt_truncated": pair["prompt_truncated"], "response_truncated": pair[role]["truncated"],
                       "mode": self.options["mode"], 'version': self.version,
                       'annotation_contract_hash': getattr(self, 'annotation_contract_hash', None),
                       'protocol_identity': getattr(self, 'protocol_identity', None),
                       'scorer_identity': getattr(self, 'scorer_identity', None),
                       'visibility': {key: pair[role].get(key) for key in ('source_truncated', 'training_token_truncated', 'source_finish_reason', 'selected_teacher_index')}})

    def annotate(self, pairs, batch, global_mini, dp_size, step):
        if self.version == 'math_process_v2':
            from .online_v2 import annotate
            return annotate(self, pairs, batch, global_mini, dp_size, step)
        import torch
        started = time.monotonic()
        deadline = started + self.options['judge']['step_timeout_seconds'] if self.judge else None
        if self.judge:
            self.judge.health(deadline=deadline)  # Cached-only steps also verify service identity.
        unique, requests, values, reasons = {}, {}, {}, {}
        hits = {"teacher": 0, "student": 0}
        for pair in pairs:
            for role in ("teacher", "student"):
                key = self.cache_key(pair, role)
                if key in unique:
                    continue
                unique[key] = (pair, role)
                allow_cache = self.options[role + "_cache"]
                cached = self.cache.get(key) if allow_cache else None
                if cached:
                    for dimension in DIMENSIONS:
                        validate_label(cached["labels"][dimension], dimension)
                    if cached["response_id"] != pair[role]["response_id"] or cached["rubric_hash"] != self.spec_hash:
                        raise ValueError("label cache response/spec identity mismatch")
                    values[key] = cached
                    hits[role] += 1
                    continue
                answer = self.scorer(pair[role]["response"], pair["ground_truth"])
                pred = answer.get("pred")
                ambiguous = pred in (None, "", "INVALID", "[INVALID]") and (
                    "boxed" in pair[role]["response"] or "Answer:" in pair[role]["response"])
                correctness_valid = not ambiguous and not pair["prompt_truncated"]
                labels = {DIMENSIONS[0]: {"value": int(bool(answer["acc"])) if correctness_valid else None,
                                          "valid": correctness_valid, "source": "math_scorer"}}
                reason = ("prompt_truncated" if pair["prompt_truncated"] else "response_truncated"
                          if pair[role]["truncated"] else "correctness_only" if not self.judge else None)
                for dim in DIMENSIONS[1:]:
                    labels[dim] = {"value": None, "valid": False, "source": reason or "pending"}
                values[key] = {"schema_version": ANNOTATION_SCHEMA, "response_id": pair[role]["response_id"],
                               "rubric_version": "math_v1", "rubric_hash": self.spec_hash,
                               "judge_fingerprint": self.options["judge_fingerprint"], "cache_key": key,
                               "labels": labels, "raw_scorer_result": answer, "evidence": {},
                               "review_status": "pending", "unknown_reason": reason}
                if reason:
                    reasons[reason] = reasons.get(reason, 0) + 1
                    if allow_cache:
                        self.cache.put(key, values[key])
                else:
                    requests[key] = judge_messages(pair["prompt"], pair[role]["response"], pair["ground_truth"],
                                                   self.spec, self.instruction)
        results = self.judge.annotate(requests, deadline=deadline, health_checked=True) if requests else {}
        for key, result in results.items():
            if result.get("parsed"):
                for dim, label in result["parsed"].items():
                    values[key]["labels"][dim] = {"value": label["value"], "valid": label["valid"], "source": "judge"}
                    values[key]["evidence"][dim] = label["evidence"]
                values[key]["unknown_reason"] = "judge_unknown" if any(not x["valid"] for x in result["parsed"].values()) else None
                if self.options[unique[key][1] + "_cache"]:
                    self.cache.put(key, values[key])
            else:
                values[key]["unknown_reason"] = result["error"]
                append_json(self.directory / "judge_errors.jsonl", {"step": step, "request_id": key, **result})
            reason = values[key]["unknown_reason"]
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        targets, masks = {}, {}
        for role in ("teacher", "student"):
            labels = [values[self.cache_key(pair, role)]["labels"] for pair in pairs]
            targets[role] = torch.tensor([[float(row[dim]["value"]) if row[dim]["valid"] else 0 for dim in DIMENSIONS] for row in labels])
            masks[role] = torch.tensor([[float(row[dim]["valid"]) for dim in DIMENSIONS] for row in labels])
            if self.options["mode"] == "correctness_only":
                masks[role][:, 1:] = 0
            batch.batch[role + "_rubric_targets"] = targets[role]
            batch.batch[role + "_rubric_mask"] = masks[role]
        weights, counts = auxiliary_weights(pairs, targets, masks, global_mini, dp_size)
        for role in weights:
            batch.batch[role + "_aux_weight"] = weights[role]
        batch.batch["rubric_global_counts"] = counts
        stats = {"rubric/judge_requests": len(requests), "rubric/judge_seconds": time.monotonic() - started,
                 "rubric/json_valid_count": sum("parsed" in row for row in results.values()),
                 "rubric/json_requested_count": len(results),
                 "rubric/cache_hits_teacher": hits["teacher"], "rubric/cache_hits_student": hits["student"],
                 "rubric/retry_count": sum(max(0, row["attempts"] - 1) for row in results.values()),
                 "rubric/judge_prompt_tokens": sum(row.get("usage", {}).get("prompt_tokens", 0) for row in results.values()),
                 "rubric/judge_output_tokens": sum(row.get("usage", {}).get("completion_tokens", 0) for row in results.values()),
                 "rubric/json_valid_rate": sum("parsed" in row for row in results.values()) / len(results) if results else 1.,
                 "rubric/teacher_wrong_student_right_count": sum(
                     not values[self.cache_key(p, "teacher")]["raw_scorer_result"]["acc"]
                     and values[self.cache_key(p, "student")]["raw_scorer_result"]["acc"] for p in pairs)}
        for role in ("teacher", "student"):
            entries = {pair[role]["response_id"]: values[self.cache_key(pair, role)] for pair in pairs}
            stats["rubric/unique_" + role] = len(entries)
            stats["rubric/truncated_" + role] = sum(pair[role]["truncated"] for pair in pairs)
            for dim in DIMENSIONS:
                valid = [r["labels"][dim]["value"] for r in entries.values() if r["labels"][dim]["valid"]]
                stats[f"rubric/valid_{dim}_{role}"] = len(valid)
                stats[f"rubric/coverage_{dim}_{role}"] = len(valid) / len(entries) if entries else 0
                stats[f"rubric/same_score_{dim}_{role}"] = int(bool(valid) and len(set(valid)) == 1)
        stats.update({"rubric/unknown_reason/" + key: value for key, value in reasons.items()})
        if self.judge and masks["student"][:, 1:].sum().item() == 0:
            self.empty_student_steps += 1
            if self.empty_student_steps >= 3:
                print("[rubric] WARNING: student process supervision is zero for 3+ consecutive steps", flush=True)
        else:
            self.empty_student_steps = 0
        for pair in pairs:
            pair["labels"] = {role: values[self.cache_key(pair, role)] for role in ("teacher", "student")}
        return stats
