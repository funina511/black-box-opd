"""Minimal, reversible engineering patches applied to the three pinned backends.

The algorithm is frozen (see plan section 2): these patches must not touch the
discriminator loss, the GRPO advantage, the Warmup SFT loss or the update order.
Each patch is:

* **named** and recorded in the handoff manifest with the diff it produced;
* **idempotent** - re-applying it is a no-op;
* **anchor-verified** - every insertion site must match exactly once, otherwise
  the patch aborts instead of silently mangling the source;
* **independently verifiable** - ``verify()`` re-checks the marker strings in a
  fresh worktree (used by preflight before any GPU time is spent).

Patches
-------
``gad-chat-template-kwargs``
    Forward ``data.apply_chat_template_kwargs`` (e.g. ``enable_thinking: false``)
    to every ``apply_chat_template`` call in ``RLHFDataset``. The original loader
    never forwarded the switch, so a Hydra option would have been silently
    ignored - which would have made teacher data and student training disagree.

``gad-resume-worker-state``
    Opt-in same-stage full-state resume: restore actor/critic model, optimizer,
    scheduler and RNG from the checkpoint. Off by default, so the original
    behaviour (weights come from ``model.path``, only step + dataloader resume)
    is preserved exactly.

``gad-math-validation``
    Add the plan's primary validation metric: CARD-aligned math accuracy on the
    student texts that ``_validate()`` already generates (no extra generation, no
    gradient, nothing written back into rewards/advantages/discriminator labels).
    ROUGE-L is kept as the original auxiliary diagnostic. The scorer path comes
    from ``GAD_MATH_SCORER`` (set by the pipeline for every stage); a configured
    but unusable scorer is a hard error instead of a silent ROUGE-only run. Each
    validation pass also appends one line to
    ``<trainer.default_local_dir>/validation_math_metrics.jsonl`` so the
    pipeline can verify the metric was really produced, independently of the
    logging backend.

``gad-teacher-aware-micro-batch``
    Size a micro-batch by *every* sequence it will forward, and refuse to hand out a
    micro-batch above the token budget. ``rearrange_micro_batches`` only ever looked at the
    student mask (unless ``compute_teacher=True``), so the Warmup actor (SFT on the teacher
    text) and the discriminator (student and teacher both forward in one micro-batch) could
    be handed a micro-batch whose teacher side alone was many times the token budget - long
    math answers make this the first thing to OOM. ``ceildiv(total, budget)`` only bounds
    the mean load, so the split count now grows until *every* partition fits, with the DP
    ranks kept in step by collectives all of them join.

    This is an engineering change, not a no-op: the trainer's dynamic-bsz accumulation
    (``loss * len(micro_batch) / ppo_mini_batch_size``) is not partition-invariant, so
    re-grouping samples re-weights them (``(|g| / B) * (n_i / sum_{j in g} n_j)``). The
    effect is measured in ``tests/test_microbatch_equivalence.py`` and documented in
    ``PIPELINE.md`` 3.2; ``GAD_MICRO_BATCH_BUDGET=upstream`` restores the pinned
    partitioning exactly.

``gad-vllm-engine-limits``
    Forward ``rollout.max_num_seqs`` to vLLM. It was present in the Hydra config
    but never passed to ``LLM(...)``, so tuning it silently did nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from common import PipelineError, insert_after, log, replace_once, sha256_text

MARKER_PREFIX = "PATCH(gad:"


@dataclass
class PatchResult:
    name: str
    changed: bool
    files: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class BackendPatch:
    name = "patch"
    description = ""

    def apply(self, worktree: Path) -> PatchResult:  # pragma: no cover - interface
        raise NotImplementedError

    def verify(self, worktree: Path) -> tuple[bool, list[str]]:  # pragma: no cover - interface
        raise NotImplementedError


def _read(path: Path) -> str:
    if not path.exists():
        raise PipelineError(f"patch target not found: {path}")
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _assert_compiles(path: Path, *, what: str) -> None:
    """Fail loudly if a patch left the target file syntactically broken."""
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except SyntaxError as exc:  # pragma: no cover - a broken patch must abort the run
        raise PipelineError(f"{what}: patched file does not compile ({path}): {exc}") from exc


class ChatTemplateKwargsPatch(BackendPatch):
    name = "gad-chat-template-kwargs"
    description = "forward data.apply_chat_template_kwargs to every apply_chat_template call"

    DATASET = "verl/utils/dataset/rl_dataset.py"
    CONFIG = "verl/trainer/config/ppo_trainer.yaml"

    MARKER_INIT = f"{MARKER_PREFIX}chat-template-kwargs) forward tokenizer chat-template options"
    MARKER_FILTER = f"{MARKER_PREFIX}chat-template-kwargs) local alias for the filter pass"
    MARKER_CALL = f"{MARKER_PREFIX}chat-template-kwargs)"
    MARKER_CONFIG = f"{MARKER_PREFIX}chat-template-kwargs) extra kwargs forwarded to"

    def apply(self, worktree: Path) -> PatchResult:
        dataset_path = worktree / self.DATASET
        config_path = worktree / self.CONFIG
        text = _read(dataset_path)
        changed = False

        if self.MARKER_INIT not in text:
            text = insert_after(
                text,
                '        self.chat_template_func = config.get("chat_template_func", None)\n',
                "        # " + self.MARKER_INIT + "\n"
                "        # (e.g. enable_thinking) to every apply_chat_template call. Without this\n"
                "        # the configured template switch was silently ignored.\n"
                "        self.apply_chat_template_kwargs = dict(config.get(\"apply_chat_template_kwargs\", {}) or {})\n",
                what="rl_dataset.__init__ chat template kwargs",
            )
            changed = True

        if self.MARKER_FILTER not in text:
            text = insert_after(
                text,
                "            tokenizer = self.tokenizer\n"
                "            processor = self.processor\n"
                "            prompt_key = self.prompt_key\n"
                "            image_key = self.image_key\n"
                "            video_key = self.video_key\n",
                "            # " + self.MARKER_FILTER + "\n"
                "            apply_chat_template_kwargs = self.apply_chat_template_kwargs\n",
                what="rl_dataset filter locals",
            )
            changed = True

        processor_filter_anchor = (
            "                def doc2len(doc) -> int:\n"
            "                    messages = self._build_messages(doc)\n"
            "                    raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)\n"
        )
        if processor_filter_anchor in text:
            text = replace_once(
                text,
                processor_filter_anchor,
                "                def doc2len(doc) -> int:\n"
                "                    messages = self._build_messages(doc)\n"
                "                    raw_prompt = self.processor.apply_chat_template(\n"
                "                        messages, add_generation_prompt=True, tokenize=False, **apply_chat_template_kwargs\n"
                "                    )  # " + self.MARKER_CALL + "\n",
                what="doc2len processor template",
            )
            changed = True

        tokenizer_filter_anchor = (
            "                def doc2len(doc) -> int:\n"
            "                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))\n"
        )
        if tokenizer_filter_anchor in text:
            text = replace_once(
                text,
                tokenizer_filter_anchor,
                "                def doc2len(doc) -> int:\n"
                "                    return len(\n"
                "                        tokenizer.apply_chat_template(\n"
                "                            doc[prompt_key], add_generation_prompt=True, **apply_chat_template_kwargs\n"
                "                        )\n"
                "                    )  # " + self.MARKER_CALL + "\n",
                what="doc2len tokenizer template",
            )
            changed = True

        getitem_processor_anchor = (
            "            else:\n"
            "                raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)\n"
            "                multi_modal_data = {}\n"
        )
        if getitem_processor_anchor in text:
            text = replace_once(
                text,
                getitem_processor_anchor,
                "            else:\n"
                "                raw_prompt = self.processor.apply_chat_template(\n"
                "                    messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs\n"
                "                )  # " + self.MARKER_CALL + "\n"
                "                multi_modal_data = {}\n",
                what="__getitem__ processor template",
            )
            changed = True

        getitem_tokenizer_anchor = (
            "            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)\n"
            "            model_inputs = self.tokenizer(raw_prompt, return_tensors=\"pt\", add_special_tokens=False)\n"
        )
        if getitem_tokenizer_anchor in text:
            text = replace_once(
                text,
                getitem_tokenizer_anchor,
                "            raw_prompt = self.tokenizer.apply_chat_template(\n"
                "                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs\n"
                "            )  # " + self.MARKER_CALL + "\n"
                "            model_inputs = self.tokenizer(raw_prompt, return_tensors=\"pt\", add_special_tokens=False)\n",
                what="__getitem__ tokenizer template",
            )
            changed = True

        if changed:
            _write(dataset_path, text)

        config_text = _read(config_path)
        if self.MARKER_CONFIG not in config_text:
            config_text = replace_once(
                config_text,
                "  # The field in the dataset where the prompt is located. Default is 'prompt'.\n  prompt_key: prompt\n",
                "  # The field in the dataset where the prompt is located. Default is 'prompt'.\n"
                "  prompt_key: prompt\n\n"
                "  # " + self.MARKER_CONFIG + "\n"
                "  # tokenizer/processor.apply_chat_template (e.g. {enable_thinking: false}).\n"
                "  # Empty by default so behaviour is identical to the original loader.\n"
                "  apply_chat_template_kwargs: {}\n",
                what="data.apply_chat_template_kwargs config key",
            )
            _write(config_path, config_text)
            changed = True

        return PatchResult(
            name=self.name,
            changed=changed,
            files=[self.DATASET, self.CONFIG],
            details={"dataset_sha256": sha256_text(_read(dataset_path))},
        )

    def verify(self, worktree: Path) -> tuple[bool, list[str]]:
        problems: list[str] = []
        dataset = _read(worktree / self.DATASET)
        config = _read(worktree / self.CONFIG)

        if self.MARKER_INIT not in dataset:
            problems.append(f"{self.DATASET}: missing init marker")
        if "self.apply_chat_template_kwargs = dict(config.get(" not in dataset:
            problems.append(f"{self.DATASET}: init assignment missing")
        if self.MARKER_FILTER not in dataset:
            problems.append(f"{self.DATASET}: filter local alias missing")

        # The four original call sites must all be forwarded; the un-patched forms
        # must be gone. This is exact (no marker counting ambiguity).
        unpatched_forms = (
            "self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)",
            "self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)",
            "tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)",
        )
        for form in unpatched_forms:
            if form in dataset:
                problems.append(f"{self.DATASET}: un-patched call site remains: {form[:60]}...")
        if dataset.count("apply_chat_template(") != 4:
            problems.append(
                f"{self.DATASET}: expected exactly 4 apply_chat_template call sites, "
                f"found {dataset.count('apply_chat_template(')}"
            )
        if dataset.count("**apply_chat_template_kwargs") != 2:
            problems.append(f"{self.DATASET}: the overlong-prompt filter does not forward the kwargs")
        if dataset.count("**self.apply_chat_template_kwargs") != 2:
            problems.append(f"{self.DATASET}: __getitem__ does not forward the kwargs")
        if "apply_chat_template_kwargs: {}" not in config:
            problems.append(f"{self.CONFIG}: data.apply_chat_template_kwargs key missing")

        # The frozen algorithm files must stay untouched by this patch.
        for frozen in ("verl/trainer/ppo/core_algos.py", "verl/workers/critic/dp_critic.py"):
            if not (worktree / frozen).exists():
                problems.append(f"{frozen}: expected file missing from the worktree")
        return not problems, problems


class ResumeWorkerStatePatch(BackendPatch):
    name = "gad-resume-worker-state"
    description = "opt-in same-stage resume of actor/critic optimizer, scheduler and RNG state"

    RAY_TRAINER = "verl/trainer/ppo/ray_trainer.py"
    CONFIG = "verl/trainer/config/ppo_trainer.yaml"
    MARKER = f"{MARKER_PREFIX}resume-worker-state)"

    def apply(self, worktree: Path) -> PatchResult:
        trainer_path = worktree / self.RAY_TRAINER
        config_path = worktree / self.CONFIG
        text = _read(trainer_path)
        changed = False

        if self.MARKER not in text:
            anchor = (
                "        # load dataloader,\n"
                "        # TODO: from remote not implemented yet\n"
                "        dataloader_local_path = os.path.join(global_step_folder, \"data.pt\")\n"
            )
            insertion = (
                "        # " + self.MARKER + " same-stage full-state resume. Disabled by default, so\n"
                "        # the original behaviour (weights from model.path + step/dataloader resume only)\n"
                "        # is unchanged. When enabled, actor/critic model, optimizer, lr scheduler and\n"
                "        # RNG state are restored from this checkpoint.\n"
                "        if self.config.trainer.get(\"resume_load_worker_state\", False):\n"
                "            if not os.path.exists(actor_path):\n"
                "                raise FileNotFoundError(f\"actor checkpoint missing for full-state resume: {actor_path}\")\n"
                "            print(f\"[gad] full-state resume: loading actor worker state from {actor_path}\")\n"
                "            self.actor_rollout_wg.load_checkpoint(\n"
                "                actor_path, del_local_after_load=self.config.trainer.get(\"del_local_ckpt_after_load\", False)\n"
                "            )\n"
                "            if self.use_critic:\n"
                "                if not os.path.exists(critic_path):\n"
                "                    raise FileNotFoundError(f\"critic checkpoint missing for full-state resume: {critic_path}\")\n"
                "                print(f\"[gad] full-state resume: loading critic worker state from {critic_path}\")\n"
                "                self.critic_wg.load_checkpoint(\n"
                "                    critic_path, del_local_after_load=self.config.trainer.get(\"del_local_ckpt_after_load\", False)\n"
                "                )\n"
                "\n"
            )
            text = replace_once(text, anchor, insertion + anchor, what="ray_trainer._load_checkpoint resume hook")
            _write(trainer_path, text)
            changed = True

        config_text = _read(config_path)
        if "resume_load_worker_state" not in config_text:
            config_text = replace_once(
                config_text,
                '  # "resume_path": resume from a user-defined path\n  resume_mode: auto\n',
                '  # "resume_path": resume from a user-defined path\n'
                "  resume_mode: auto\n\n"
                "  # " + self.MARKER + " restore actor/critic model+optimizer+scheduler+RNG on resume.\n"
                "  # False keeps the original weights/dataloader-only semantics.\n"
                "  resume_load_worker_state: False\n",
                what="trainer.resume_load_worker_state config key",
            )
            _write(config_path, config_text)
            changed = True

        return PatchResult(
            name=self.name,
            changed=changed,
            files=[self.RAY_TRAINER, self.CONFIG],
            details={"ray_trainer_sha256": sha256_text(_read(trainer_path))},
        )

    def verify(self, worktree: Path) -> tuple[bool, list[str]]:
        problems: list[str] = []
        trainer = _read(worktree / self.RAY_TRAINER)
        config = _read(worktree / self.CONFIG)

        if self.MARKER not in trainer:
            problems.append(f"{self.RAY_TRAINER}: resume hook marker missing")
        if "resume_load_worker_state" not in trainer:
            problems.append(f"{self.RAY_TRAINER}: resume_load_worker_state branch missing")
        if "self.actor_rollout_wg.load_checkpoint(" not in trainer:
            problems.append(f"{self.RAY_TRAINER}: actor load_checkpoint call missing")
        if "self.critic_wg.load_checkpoint(" not in trainer:
            problems.append(f"{self.RAY_TRAINER}: critic load_checkpoint call missing")
        if "resume_load_worker_state: False" not in config:
            problems.append(f"{self.CONFIG}: trainer.resume_load_worker_state key missing")
        return not problems, problems


class MathValidationPatch(BackendPatch):
    """Add CARD-aligned math accuracy to the validation metrics of every branch.

    The plan (section 3.3) requires the three branches to report dev-set math
    accuracy on the texts ``_validate()`` already generates. This patch adds the
    metric without touching the training reward, the advantage, the discriminator
    labels or the update order: ROUGE-L stays exactly where it was and the math
    numbers are additional validation output.
    """

    name = "gad-math-validation"
    description = "report CARD-aligned dev-set math accuracy during validation"

    RAY_TRAINER = "verl/trainer/ppo/ray_trainer.py"
    MARKER = f"{MARKER_PREFIX}math-validation)"

    HELPERS = '''# @@MARKER@@ dev-set math accuracy (added by tools/backend_patches.py).
# The metric is computed on the student texts _validate() already generated: no
# extra generation, no gradient, and nothing is written back into
# token_level_rewards, advantages or discriminator labels. The original ROUGE-L
# diagnostic is unchanged. The scorer is the vendored CARD implementation named
# by GAD_MATH_SCORER - the same file the offline evaluation uses, so the
# training-time and evaluation numbers come from one implementation.
_GAD_MATH_SCORER_CACHE = {}
_GAD_MATH_SCORE_ERRORS = [0]


def _gad_math_scorer_path(config):
    """GAD_MATH_SCORER wins; data.gad_math_scorer_path is accepted as a fallback."""
    path = os.environ.get("GAD_MATH_SCORER")
    if path:
        return path
    try:
        return config.data.get("gad_math_scorer_path", None)
    except Exception:  # noqa: BLE001 - the config shape differs between branches
        return None


def _gad_load_math_scorer(path):
    """Return the scorer module for ``path``, or None when math metrics are off."""
    if not path:
        return None
    if path in _GAD_MATH_SCORER_CACHE:
        return _GAD_MATH_SCORER_CACHE[path]
    import importlib.util
    import sys

    if not os.path.isfile(path):
        raise RuntimeError(
            f"[gad] GAD_MATH_SCORER points at {path}, which does not exist; refusing to validate "
            "without the configured math metric"
        )
    spec = importlib.util.spec_from_file_location("gad_math_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gad_math_scorer", module)
    spec.loader.exec_module(module)
    if not hasattr(module, "compute_score"):
        raise RuntimeError(f"[gad] math scorer {path} does not expose compute_score()")
    _GAD_MATH_SCORER_CACHE[path] = module
    return module


def _gad_score_math_row(scorer, response, ground_truth):
    """One row's CARD-aligned score; a scorer failure counts as incorrect."""
    try:
        result = scorer.compute_score(response or "", str(ground_truth or ""))
        prediction = str(result.get("pred", ""))
        return {
            "acc": float(bool(result.get("acc"))),
            "score": float(result.get("score", -1.0)),
            "answer_parsed": 0.0 if prediction == "[INVALID]" else 1.0,
        }
    except Exception as exc:  # noqa: BLE001 - one bad row must not abort validation
        _GAD_MATH_SCORE_ERRORS[0] += 1
        if _GAD_MATH_SCORE_ERRORS[0] <= 5:
            print(f"[gad] math scoring failed ({type(exc).__name__}: {exc}); row counted as incorrect")
        return {"acc": 0.0, "score": -1.0, "answer_parsed": 0.0}


def _gad_math_metrics(rows):
    """Aggregate per-row math scores into the numbers the plan asks for.

    ``acc`` (mean of the 0/1 correctness flags) is the accuracy; the mean of the
    +/-1 ``score`` is reported separately and is never called accuracy.
    """
    if not rows:
        return {}
    count = len(rows)
    return {
        "count": count,
        "acc": float(sum(row["acc"] for row in rows) / count),
        "score_mean": float(sum(row["score"] for row in rows) / count),
        "answer_parse_failure_rate": float(sum(1.0 - row["answer_parsed"] for row in rows) / count),
        "response_len_mean": float(sum(row["length"] for row in rows) / count),
        "truncated_rate": float(sum(row["truncated"] for row in rows) / count),
    }


def _gad_write_math_metrics(directory, payload):
    """Append one JSON line per validation pass (independent of the log backend)."""
    if not directory:
        return None
    try:
        os.makedirs(str(directory), exist_ok=True)
        target = os.path.join(str(directory), "validation_math_metrics.jsonl")
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\\n")
        return target
    except Exception as exc:  # noqa: BLE001 - a reporting failure must not kill training
        print(f"[gad] could not append validation_math_metrics.jsonl under {directory}: {exc}")
        return None


def _gad_score_math_batch(scorer, output_texts, ground_truths, response_lengths, max_response_length):
    """Score one validation batch (student text vs ground truth).

    ``response_lengths`` are the true response lengths (tokens up to and
    including EOS, taken from the rollout's own attention mask), so truncation is
    ``length >= max_response_length`` - the same convention the branch uses when
    it builds its response masks.
    """
    rows = []
    for text, ground_truth, length in zip(output_texts, ground_truths, response_lengths):
        row = _gad_score_math_row(scorer, text, ground_truth)
        row["length"] = float(length)
        row["truncated"] = float(length >= max_response_length)
        rows.append(row)
    return rows


'''

    ANCHOR_CLASS = "class RayPPOTrainer:"
    ANCHOR_SCORER = "        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)\n"
    ANCHOR_ROUGE = (
        "            scores = [safe_rouge_score(ref, cand) for ref, cand in zip(teacher_texts, output_texts)]\n"
        "            sample_scores.extend(scores)\n"
    )
    ANCHOR_METRIC = '        metric_dict["val/rouge-L/mean"] = np.mean(sample_scores)\n'

    # A worktree that carries an older revision of this patch must be rebuilt
    # instead of patched twice; this is the fingerprint that detects it.
    FINGERPRINT = (
        "def _gad_score_math_batch(scorer, output_texts, ground_truths, response_lengths, max_response_length)",
        "_gad_score_math_batch(\n",
    )
    EXPECTED_END_MARKERS = 3

    def _outdated_revision(self, text: str) -> bool:
        """Marker present but the payload is from an older revision of the patch."""
        if text.count(self.MARKER + " end") != self.EXPECTED_END_MARKERS:
            return True
        return not all(snippet in text for snippet in self.FINGERPRINT)

    def apply(self, worktree: Path) -> PatchResult:
        path = worktree / self.RAY_TRAINER
        text = _read(path)
        changed = False

        if self.MARKER in text and self._outdated_revision(text):
            raise PipelineError(
                f"{path} carries an older revision of {self.name}. Rebuild the backend worktree "
                "(tools/prepare_backends.py --recreate) instead of patching on top of it."
            )

        if self.MARKER not in text:
            text = replace_once(
                text,
                self.ANCHOR_CLASS,
                self.HELPERS.replace("@@MARKER@@", self.MARKER) + self.ANCHOR_CLASS,
                what="ray_trainer math metric helpers",
            )

            # --- per-validation setup ------------------------------------- #
            text = insert_after(
                text,
                self.ANCHOR_SCORER,
                "\n"
                "        # " + self.MARKER + " math accuracy on the same generated texts.\n"
                "        scorer_path = _gad_math_scorer_path(self.config)\n"
                "        math_scorer = _gad_load_math_scorer(scorer_path)\n"
                "        if math_scorer is None:\n"
                "            print(\n"
                "                \"[gad] WARNING: validation math metrics are DISABLED (no GAD_MATH_SCORER); \"\n"
                "                \"only the original ROUGE-L diagnostic is reported\"\n"
                "            )\n"
                "        math_rows = []\n"
                "        math_max_response_length = int(\n"
                "            self.config.actor_rollout_ref.rollout.get(\"response_length\", None)\n"
                "            or self.config.data.max_response_length\n"
                "        )\n"
                "        # " + self.MARKER + " end\n",
                what="ray_trainer._validate math setup",
            )

            # --- per-batch scoring ----------------------------------------- #
            text = insert_after(
                text,
                self.ANCHOR_ROUGE,
                "\n"
                "            # " + self.MARKER + " score the student texts generated just above.\n"
                "            if math_scorer is not None:\n"
                "                batch_response_length = test_output_gen_batch.batch[\"responses\"].shape[-1]\n"
                "                response_mask = test_output_gen_batch.batch[\"attention_mask\"][:, -batch_response_length:]\n"
                "                batch_rows = _gad_score_math_batch(\n"
                "                    math_scorer,\n"
                "                    output_texts,\n"
                "                    [str(item.get(\"ground_truth\", \"\")) for item in test_batch.non_tensor_batch[\"reward_model\"]],\n"
                "                    [int(value) for value in response_mask.sum(dim=-1).tolist()],\n"
                "                    math_max_response_length,\n"
                "                )\n"
                "                math_rows.extend(batch_rows)\n"
                "                # Feeding `acc` into the original per-data-source aggregation is what\n"
                "                # promotes math accuracy to the val-core slot instead of ROUGE-L.\n"
                "                reward_extra_infos_dict[\"acc\"].extend(row[\"acc\"] for row in batch_rows)\n"
                "            # " + self.MARKER + " end\n",
                what="ray_trainer._validate per-batch math scoring",
            )

            # --- metrics + artifact ---------------------------------------- #
            text = insert_after(
                text,
                self.ANCHOR_METRIC,
                "\n"
                "        # " + self.MARKER + " primary metric, plus the diagnostics the plan asks for.\n"
                "        math_summary = _gad_math_metrics(math_rows)\n"
                "        if math_summary:\n"
                "            metric_dict[\"val/math/acc\"] = math_summary[\"acc\"]\n"
                "            metric_dict[\"val/math/score_mean\"] = math_summary[\"score_mean\"]\n"
                "            metric_dict[\"val/math/answer_parse_failure_rate\"] = math_summary[\"answer_parse_failure_rate\"]\n"
                "            metric_dict[\"val/math/response_len_mean\"] = math_summary[\"response_len_mean\"]\n"
                "            metric_dict[\"val/math/truncated_rate\"] = math_summary[\"truncated_rate\"]\n"
                "            metric_dict[\"val/math/count\"] = float(math_summary[\"count\"])\n"
                "            _gad_write_math_metrics(\n"
                "                self.config.trainer.get(\"default_local_dir\", None),\n"
                "                {\n"
                "                    \"global_step\": int(self.global_steps),\n"
                "                    \"scorer\": scorer_path,\n"
                "                    \"ground_truth_field\": \"reward_model.ground_truth\",\n"
                "                    \"val_kwargs_n\": int(self.config.actor_rollout_ref.rollout.val_kwargs.n),\n"
                "                    \"val_do_sample\": bool(self.config.actor_rollout_ref.rollout.val_kwargs.do_sample),\n"
                "                    \"val_temperature\": float(self.config.actor_rollout_ref.rollout.val_kwargs.temperature),\n"
                "                    \"max_response_length\": math_max_response_length,\n"
                "                    \"truncated_definition\": (\n"
                "                        \"response tokens up to and including EOS >= rollout.response_length\"\n"
                "                    ),\n"
                "                    **math_summary,\n"
                "                },\n"
                "            )\n"
                "        # " + self.MARKER + " end\n",
                what="ray_trainer._validate math metric dict",
            )

            _write(path, text)
            _assert_compiles(path, what=f"{self.name} patch")
            changed = True

        return PatchResult(
            name=self.name,
            changed=changed,
            files=[self.RAY_TRAINER],
            details={"ray_trainer_sha256": sha256_text(_read(path))},
        )

    def verify(self, worktree: Path) -> tuple[bool, list[str]]:
        problems: list[str] = []
        path = worktree / self.RAY_TRAINER
        text = _read(path)

        for marker in (
            "def _gad_math_metrics(rows)",
            "def _gad_load_math_scorer(path)",
            "def _gad_write_math_metrics(directory, payload)",
            "def _gad_score_math_batch(scorer, output_texts, ground_truths, response_lengths, max_response_length)",
            "_gad_math_scorer_path(self.config)",
            'reward_extra_infos_dict["acc"].extend',
            'metric_dict["val/math/acc"] = math_summary["acc"]',
            "validation_math_metrics.jsonl",
        ):
            if marker not in text:
                problems.append(f"{self.RAY_TRAINER}: missing {marker!r}")
        if text.count(self.MARKER + " end") != self.EXPECTED_END_MARKERS:
            problems.append(
                f"{self.RAY_TRAINER}: expected {self.EXPECTED_END_MARKERS} {self.MARKER} end markers, "
                f"found {text.count(self.MARKER + ' end')}"
            )

        # The reward/advantage path must be byte-identical to the original: the
        # math metric is validation output only and may never feed the tensors
        # the trainer optimizes (plan section 3.3). These two lines exist in all
        # three branches; the critic-only reward shortcut is not required here.
        for frozen in (
            "    token_level_rewards = token_level_scores - beta * kld\n",
            '    data.batch["token_level_rewards"] = token_level_rewards\n',
        ):
            if frozen not in text:
                problems.append(f"{self.RAY_TRAINER}: frozen reward line changed or missing: {frozen.strip()!r}")

        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            problems.append(f"{self.RAY_TRAINER}: does not compile after patching: {exc}")
        return not problems, problems


class ValidationScopePatch(BackendPatch):
    """Use a fixed quick dev loader between checkpoints and full dev at stage end."""

    name = "gad-validation-scope"
    description = "separate fixed quick and final full validation with timing"
    RAY_TRAINER = "verl/trainer/ppo/ray_trainer.py"
    MARKER = f"{MARKER_PREFIX}validation-scope)"

    def apply(self, worktree: Path) -> PatchResult:
        path = worktree / self.RAY_TRAINER
        source = _read(path)
        if self.MARKER in source:
            return PatchResult(self.name, False, [self.RAY_TRAINER])
        text = source
        text = replace_once(text, "import os\n", "import os\nimport time\n", what="validation timing import")
        text = replace_once(
            text,
            '        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"\n',
            '        # ' + self.MARKER + ' fixed quick loader; full loader above is unchanged.\n'
            '        quick_file = os.environ.get("GAD_QUICK_DEV")\n'
            '        self.quick_val_dataloader = None\n'
            '        if quick_file:\n'
            '            quick_dataset = create_rl_dataset([quick_file], self.config.data, self.tokenizer, self.processor)\n'
            '            if len(quick_dataset) < 1:\n'
            '                raise ValueError("quick validation dataset is empty")\n'
            '            self.quick_val_dataloader = StatefulDataLoader(\n'
            '                dataset=quick_dataset, batch_size=val_batch_size,\n'
            '                num_workers=self.config.data.get("dataloader_num_workers", 8),\n'
            '                shuffle=False, drop_last=False, collate_fn=collate_fn,\n'
            '            )\n'
            '        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"\n',
            what="quick validation loader",
        )
        text = replace_once(
            text,
            '    def _validate(self):\n        data_source_lst = []\n',
            '    def _validate(self):\n'
            '        # ' + self.MARKER + ' timings include all validation batches.\n'
            '        _gad_started = time.perf_counter()\n'
            '        _gad_timing = {"generation": 0.0, "decode": 0.0, "rouge_l": 0.0, "math_scoring": 0.0}\n'
            '        data_source_lst = []\n',
            what="validation timing setup",
        )
        text = replace_once(
            text,
            '        for test_data in self.val_dataloader:\n',
            '        # ' + self.MARKER + ' select one loader for this pass.\n'
            '        scope = getattr(self, "_gad_validation_scope", "full")\n'
            '        loader = self.quick_val_dataloader if scope == "quick" else self.val_dataloader\n'
            '        for test_data in loader:\n',
            what="validation loader selection",
        )
        text = replace_once(
            text,
            '            if not self.async_rollout_mode:\n                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)\n',
            '            _gad_tick = time.perf_counter()\n'
            '            if not self.async_rollout_mode:\n                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)\n',
            what="generation timing start",
        )
        text = replace_once(
            text,
            '                self.async_rollout_manager.sleep()\n\n            # unpad\n',
            '                self.async_rollout_manager.sleep()\n'
            '            _gad_timing["generation"] += time.perf_counter() - _gad_tick\n\n'
            '            # unpad\n',
            what="generation timing end",
        )
        text = replace_once(text, '            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]\n',
                            '            _gad_tick = time.perf_counter()\n'
                            '            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]\n',
                            what="decode timing start")
        text = replace_once(text, '            teacher_outputs.extend(teacher_texts)\n',
                            '            teacher_outputs.extend(teacher_texts)\n'
                            '            _gad_timing["decode"] += time.perf_counter() - _gad_tick\n',
                            what="decode timing end")
        text = replace_once(text, '            scores = [safe_rouge_score(ref, cand) for ref, cand in zip(teacher_texts, output_texts)]\n',
                            '            _gad_tick = time.perf_counter()\n'
                            '            scores = [safe_rouge_score(ref, cand) for ref, cand in zip(teacher_texts, output_texts)]\n'
                            '            _gad_timing["rouge_l"] += time.perf_counter() - _gad_tick\n',
                            what="rouge timing")
        text = replace_once(text, '                batch_rows = _gad_score_math_batch(\n',
                            '                _gad_tick = time.perf_counter()\n'
                            '                batch_rows = _gad_score_math_batch(\n',
                            what="math timing start")
        text = replace_once(text, '                math_rows.extend(batch_rows)\n',
                            '                math_rows.extend(batch_rows)\n'
                            '                _gad_timing["math_scoring"] += time.perf_counter() - _gad_tick\n',
                            what="math timing end")
        text = replace_once(text, '                    **math_summary,\n',
                            '                    "validation_scope": scope,\n'
                            '                    "subset_sha256": os.environ.get("GAD_QUICK_SHA256" if scope == "quick" else "GAD_FULL_SHA256"),\n'
                            '                    "timing_seconds": {**_gad_timing, "total": time.perf_counter() - _gad_started},\n'
                            '                    **math_summary,\n',
                            what="validation provenance")
        text = replace_once(text, '        return metric_dict\n',
                            '        # ' + self.MARKER + ' keep quick and full series distinct.\n'
                            '        return {f"{scope}/{key}": value for key, value in metric_dict.items()}\n',
                            what="scoped validation metric keys")
        text = replace_once(
            text,
            '                        with marked_timer("testing", timing_raw, color="green"):\n                            val_metrics: dict = self._validate()\n',
            '                        with marked_timer("testing", timing_raw, color="green"):\n'
            '                            # ' + self.MARKER + ' final step always uses all dev rows.\n'
            '                            self._gad_validation_scope = "full" if is_last_step or self.quick_val_dataloader is None else "quick"\n'
            '                            val_metrics: dict = self._validate()\n',
            what="final full validation selection",
        )
        _write(path, text)
        _assert_compiles(path, what=self.name)
        return PatchResult(self.name, True, [self.RAY_TRAINER], {"ray_trainer_sha256": sha256_text(text)})

    def verify(self, worktree: Path) -> tuple[bool, list[str]]:
        path = worktree / self.RAY_TRAINER
        source = _read(path)
        problems = [f"{path}: missing {needle}" for needle in
                    (self.MARKER, 'self.quick_val_dataloader = StatefulDataLoader(',
                     'self._gad_validation_scope = "full" if is_last_step',
                     '"validation_scope": scope', '"timing_seconds":',
                     'return {f"{scope}/{key}": value for key, value in metric_dict.items()}')
                    if needle not in source]
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            problems.append(f"{path}: {exc}")
        return not problems, problems


class TeacherAwareMicroBatchPatch(BackendPatch):
    """Size micro-batches by every mask they forward, and never exceed the budget.

    Two changes, both inside ``rearrange_micro_batches``:

    1. the budget covers every sequence the micro-batch forwards (student + teacher when
       both are present). The pinned code sizes by the student mask unless
       ``compute_teacher=True``, so the Warmup actor (SFT on the teacher text) and the
       discriminator could be handed a micro-batch whose teacher side alone is many times
       the budget - an 8192-token student micro-batch can carry 131072 teacher tokens;
    2. the partition is capped. ``ceildiv(total, budget)`` bounds the *mean* load, not the
       worst partition, so the Karmarkar-Karp balancer could still return a micro-batch
       above the budget (measured: 1024 student + 16384 teacher tokens against 12288).
       The split count now grows - through collectives every rank joins, so the DP ranks
       keep a single micro-batch count - until every partition fits. A single sample that
       alone exceeds the budget stays a singleton (nothing can split a sample further;
       each of its forwards still fits, because ``max_seq_len <= max_token_len`` is
       asserted) and is reported.

    **This is not a mathematically neutral change and this patch does not claim it is.**
    With ``use_dynamic_bsz=True`` the trainer scales each micro-batch's token-normalised
    loss by that micro-batch's *sample count* (``dp_actor.py``:
    ``loss = policy_loss * (len(data) / ppo_mini_batch_size)``), so the accumulated
    objective is ``sum_g (|g| / B) * tokenmean(g)`` - re-grouping samples re-weights them
    by their token share inside their micro-batch. Equal-length samples in a micro-batch
    are invariant; unequal lengths are not. ``GAD_MICRO_BATCH_BUDGET=upstream`` restores
    the pinned partitioning bit-for-bit (no teacher-aware sizing, no cap) for those who
    want parity and can afford the memory. Impact measured by
    ``tests/test_microbatch_equivalence.py`` and documented in ``PIPELINE.md`` 3.2.
    """

    name = "gad-teacher-aware-micro-batch"
    description = "size micro-batches by every mask they forward and cap the budget"

    SEQLEN = "verl/utils/seqlen_balancing.py"
    MARKER = f"{MARKER_PREFIX}teacher-aware-micro-batch)"
    MARKER_MODE = f"{MARKER_PREFIX}teacher-aware-micro-batch:mode)"
    MARKER_CAP = f"{MARKER_PREFIX}teacher-aware-micro-batch:cap)"
    #: Last line of the region this patch owns; lets a stale region be replaced wholesale.
    MARKER_END = f"{MARKER_PREFIX}teacher-aware-micro-batch:end)"
    REGION_START = "    # this is per local micro_bsz\n"
    #: First line of untouched pinned code *after* the region (unique in the file).
    REGION_END = "\n    micro_batches = []\n"

    #: Runtime switch for the budget model. ``teacher-aware`` (default) applies changes 1+2;
    #: ``upstream`` restores the pinned backend's sizing and partitioning exactly.
    MODE_ENV = "GAD_MICRO_BATCH_BUDGET"

    ANCHOR = (
        "    # this is per local micro_bsz\n"
        "    if compute_teacher:\n"
        '        max_seq_len = batch["teacher_attention_mask"].shape[-1]\n'
        "    else:\n"
        '        max_seq_len = batch["attention_mask"].shape[-1]\n'
        '    assert max_token_len >= max_seq_len, f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"\n'
        "    if compute_teacher:\n"
        '        seq_len_effective: torch.Tensor = batch["teacher_attention_mask"].sum(dim=1)\n'
        "    else:\n"
        '        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)\n'
    )

    REPLACEMENT_V1 = (
        "    # this is per local micro_bsz\n"
        "    # " + MARKER + " Size the budget by every sequence this micro-batch will\n"
        "    # forward. `compute_teacher=True` keeps its original meaning (teacher side only,\n"
        "    # used by SeqKD's teacher SFT). When the batch carries both sides and the teacher\n"
        "    # is not the explicit choice, the budget must cover student + teacher, because\n"
        "    # those micro-batches forward both (Warmup/GAD discriminator) or the teacher\n"
        "    # (Warmup actor SFT). Sizing by the student alone overflows the budget on long\n"
        "    # math answers: an 8192-token student micro-batch can carry 131072 teacher tokens.\n"
        '    _has_teacher = "teacher_attention_mask" in batch.keys()\n'
        '    _has_student = "attention_mask" in batch.keys()\n'
        "    if compute_teacher and _has_teacher:\n"
        '        max_seq_len = batch["teacher_attention_mask"].shape[-1]\n'
        '        seq_len_effective: torch.Tensor = batch["teacher_attention_mask"].sum(dim=1)\n'
        "    elif _has_teacher and _has_student:\n"
        '        max_seq_len = max(batch["attention_mask"].shape[-1], batch["teacher_attention_mask"].shape[-1])\n'
        '        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1) + batch["teacher_attention_mask"].sum(dim=1)\n'
        "    else:\n"
        '        _mask_key = "teacher_attention_mask" if _has_teacher else "attention_mask"\n'
        "        if _mask_key not in batch.keys():\n"
        '            raise ValueError("rearrange_micro_batches needs attention_mask or teacher_attention_mask")\n'
        "        max_seq_len = batch[_mask_key].shape[-1]\n"
        "        seq_len_effective: torch.Tensor = batch[_mask_key].sum(dim=1)\n"
        '    assert max_token_len >= max_seq_len, f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"\n'
    )

    #: Unchanged between the pinned revision and v1: the split-count estimate, the DP
    #: all-reduce that keeps one count per rank, and the balanced partition call.
    MIDDLE_TAIL = (
        "    total_seqlen = seq_len_effective.sum().item()\n"
        "    # NOTE: num_microbatches <= batch_size, so take the min of this two.\n"
        "    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))\n"
        "    if min_num_micro_batch is not None:\n"
        "        # used to support pp\n"
        "        num_micro_batches = max(min_num_micro_batch, num_micro_batches)\n"
        "    if dist.is_initialized() and same_micro_num_in_dp:\n"
        "        num_micro_batches = torch.tensor([num_micro_batches], device=get_device_name())\n"
        "        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)\n"
        "        num_micro_batches = num_micro_batches.cpu().item()\n"
        "    if num_batches_divided_by is not None:\n"
        "        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)\n"
        "\n"
        "    seq_len_effective = seq_len_effective.tolist()\n"
        "    assert num_micro_batches <= len(seq_len_effective)\n"
        "\n"
        "    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)\n"
    )

    REPLACEMENT_V2 = (
        "    # this is per local micro_bsz\n"
        "    # " + MARKER + " Budget model. `teacher-aware` (default) sizes a micro-batch by\n"
        "    # every sequence it will forward and caps the result; `upstream` restores the\n"
        "    # pinned backend's sizing and partitioning exactly (see the class docstring).\n"
        "    import os as _gad_os\n"
        '    _gad_budget_mode = _gad_os.environ.get("' + MODE_ENV + '", "teacher-aware").strip().lower()\n'
        '    _has_teacher = "teacher_attention_mask" in batch.keys()\n'
        '    _has_student = "attention_mask" in batch.keys()\n'
        '    if _gad_budget_mode == "upstream":\n'
        "        # " + MARKER_MODE + " byte-for-byte the pinned behaviour\n"
        "        if compute_teacher:\n"
        '            max_seq_len = batch["teacher_attention_mask"].shape[-1]\n'
        '            seq_len_effective: torch.Tensor = batch["teacher_attention_mask"].sum(dim=1)\n'
        "        else:\n"
        '            max_seq_len = batch["attention_mask"].shape[-1]\n'
        '            seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1)\n'
        "    elif compute_teacher and _has_teacher:\n"
        "        # teacher-side SFT (SeqKD): the pinned code already sized by the teacher mask\n"
        '        max_seq_len = batch["teacher_attention_mask"].shape[-1]\n'
        '        seq_len_effective: torch.Tensor = batch["teacher_attention_mask"].sum(dim=1)\n'
        "    elif _has_teacher and _has_student:\n"
        "        # both sides are forwarded (Warmup actor SFT on the teacher text, discriminator\n"
        "        # scoring student and teacher): the budget has to cover both\n"
        '        max_seq_len = max(batch["attention_mask"].shape[-1], batch["teacher_attention_mask"].shape[-1])\n'
        '        seq_len_effective: torch.Tensor = batch["attention_mask"].sum(dim=1) + batch["teacher_attention_mask"].sum(dim=1)\n'
        "    else:\n"
        '        _mask_key = "teacher_attention_mask" if _has_teacher else "attention_mask"\n'
        "        if _mask_key not in batch.keys():\n"
        '            raise ValueError("rearrange_micro_batches needs attention_mask or teacher_attention_mask")\n'
        "        max_seq_len = batch[_mask_key].shape[-1]\n"
        "        seq_len_effective: torch.Tensor = batch[_mask_key].sum(dim=1)\n"
        '    assert max_token_len >= max_seq_len, f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"\n'
    )

    #: v1 sizing + the split-count estimate + the cap loop.
    CAP_TAIL = (
        "    total_seqlen = seq_len_effective.sum().item()\n"
        "    # NOTE: num_microbatches <= batch_size, so take the min of this two.\n"
        "    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))\n"
        "    if min_num_micro_batch is not None:\n"
        "        # used to support pp\n"
        "        num_micro_batches = max(min_num_micro_batch, num_micro_batches)\n"
        "    if num_batches_divided_by is not None:\n"
        "        num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)\n"
        "\n"
        "    seq_len_effective = seq_len_effective.tolist()\n"
        "    assert num_micro_batches <= len(seq_len_effective)\n"
        "\n"
        "    # " + MARKER_CAP + " `ceildiv(total, budget)` bounds the MEAN load, not the worst\n"
        "    # partition: the balancer could still return a micro-batch above max_token_len\n"
        "    # (measured: 1024 student + 16384 teacher tokens against a 12288 budget). Grow the\n"
        "    # split count until every partition fits. Every rank executes the same rounds and\n"
        "    # joins the same collectives - the fit flag is MIN-reduced, so the ranks always\n"
        "    # agree on the count - and the loop stops when all ranks fit or when every sample\n"
        "    # already has its own micro-batch.\n"
        "    def _gad_oversized(partitions, costs, budget):\n"
        "        return any(sum(costs[i] for i in part) > budget for part in partitions)\n"
        "\n"
        "    _gad_cap = 0 if _gad_budget_mode == \"upstream\" else 1\n"
        "    while True:\n"
        "        if dist.is_initialized() and same_micro_num_in_dp:\n"
        "            _gad_sync = torch.tensor([num_micro_batches], device=get_device_name())\n"
        "            dist.all_reduce(_gad_sync, op=dist.ReduceOp.MAX, group=dp_group)\n"
        "            num_micro_batches = _gad_sync.cpu().item()\n"
        "        if num_batches_divided_by is not None:\n"
        "            num_micro_batches = roundup_divisible(num_micro_batches, num_batches_divided_by)\n"
        "        micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)\n"
        "        _gad_fits = 0 if (_gad_cap and _gad_oversized(micro_bsz_idx, seq_len_effective, max_token_len)) else 1\n"
        "        if dist.is_initialized() and same_micro_num_in_dp:\n"
        "            _gad_fits_t = torch.tensor([_gad_fits], device=get_device_name())\n"
        "            dist.all_reduce(_gad_fits_t, op=dist.ReduceOp.MIN, group=dp_group)\n"
        "            _gad_fits = int(_gad_fits_t.cpu().item())\n"
        "        if _gad_fits or num_micro_batches >= len(seq_len_effective):\n"
        "            break\n"
        "        num_micro_batches += 1\n"
        "    _gad_worst = max(sum(seq_len_effective[i] for i in part) for part in micro_bsz_idx)\n"
        "    if _gad_cap and not _gad_fits:\n"
        "        # Unavoidable: one sample costs more than the whole budget, so no partition can\n"
        "        # shrink it below a single sample. Each of its forwards still fits the engine\n"
        "        # budget (max_seq_len <= max_token_len is asserted above).\n"
        "        print(\n"
        "            f\"[" + MARKER_CAP + " a single sample needs {_gad_worst} tokens but the budget is \"\n"
        "            f\"{max_token_len}; it stays a singleton micro-batch. Lower \"\n"
        "            \"data.max_response_length or data.max_prompt_length to fit it.\"\n"
        "        )\n"
        "    elif not _gad_cap and _gad_worst > max_token_len:\n"
        "        print(\n"
        "            f\"[" + MARKER_MODE + " upstream partitioning kept a {_gad_worst}-token micro-batch \"\n"
        "            f\"(budget {max_token_len}); set " + MODE_ENV + "=teacher-aware to cap it.\"\n"
        "        )\n"
        "    # " + MARKER_END + "\n"
    )

    #: The region as it exists in worktrees patched by v1 (migration anchor).
    V1_REGION = REPLACEMENT_V1 + MIDDLE_TAIL
    #: The region this version installs.
    V2_REGION = REPLACEMENT_V2 + CAP_TAIL

    def apply(self, worktree: Path) -> PatchResult:
        path = worktree / self.SEQLEN
        text = _read(path)
        changed = False
        if self.V2_REGION not in text:
            if self.V1_REGION in text:
                # migrate a worktree that carries the older teacher-aware sizing
                text = replace_once(text, self.V1_REGION, self.V2_REGION, what="micro-batch budget cap")
            elif (self.ANCHOR + self.MIDDLE_TAIL) in text:
                text = replace_once(
                    text, self.ANCHOR + self.MIDDLE_TAIL, self.V2_REGION, what="micro-batch token budget"
                )
            elif self.MARKER_END in text:
                # a stale version of this patch owns the region: replace it wholesale
                start = text.index(self.REGION_START)
                end = text.index(self.MARKER_END) + len(self.MARKER_END) + 1
                text = text[:start] + self.V2_REGION + text[end:]
            elif self.REGION_START in text and self.REGION_END in text:
                # a stale region from an unknown revision: the patch owns everything between
                # its start marker and the first untouched pinned line after it
                start = text.index(self.REGION_START)
                end = text.index(self.REGION_END, start)
                text = text[:start] + self.V2_REGION + text[end:]
            else:
                raise PipelineError(
                    f"{path}: neither the pinned text, the v1 teacher-aware region nor this patch's region "
                    f"({self.MARKER_END}) was found; recreate the worktree before patching"
                )
            _write(path, text)
            _assert_compiles(path, what=self.name)
            changed = True
        return PatchResult(
            name=self.name,
            changed=changed,
            files=[self.SEQLEN],
            details={"seqlen_balancing_sha256": sha256_text(_read(path))},
        )

    def verify(self, worktree: Path) -> tuple[bool, list[str]]:
        problems: list[str] = []
        text = _read(worktree / self.SEQLEN)
        if self.MARKER not in text:
            problems.append(f"{self.SEQLEN}: teacher-aware micro-batch marker missing")
        if self.MARKER_CAP not in text:
            problems.append(f"{self.SEQLEN}: micro-batch budget cap missing (stale v1 patch?)")
        if self.MARKER_MODE not in text or self.MODE_ENV not in text:
            problems.append(f"{self.SEQLEN}: {self.MODE_ENV} escape hatch missing")
        if 'batch["attention_mask"].sum(dim=1) + batch["teacher_attention_mask"].sum(dim=1)' not in text:
            problems.append(f"{self.SEQLEN}: student+teacher budget not present")
        if "if compute_teacher and _has_teacher:" not in text:
            problems.append(f"{self.SEQLEN}: compute_teacher branch no longer explicit")
        if "_gad_oversized(micro_bsz_idx, seq_len_effective, max_token_len)" not in text:
            problems.append(f"{self.SEQLEN}: per-partition budget check missing")
        try:
            compile(text, str(worktree / self.SEQLEN), "exec")
        except SyntaxError as exc:
            problems.append(f"{self.SEQLEN}: does not compile after patching: {exc}")
        return not problems, problems


class VllmEngineLimitsPatch(BackendPatch):
    """Forward ``rollout.max_num_seqs`` to the vLLM engine."""

    name = "gad-vllm-engine-limits"
    description = "pass rollout.max_num_seqs through to LLM(...)"

    ROLLOUT = "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    MARKER = f"{MARKER_PREFIX}vllm-engine-limits)"

    ANCHOR = (
        "            max_num_batched_tokens=max_num_batched_tokens,\n"
        "            enable_chunked_prefill=config.enable_chunked_prefill,\n"
    )
    REPLACEMENT = (
        "            max_num_batched_tokens=max_num_batched_tokens,\n"
        "            # " + MARKER + " rollout.max_num_seqs existed in the Hydra config but was\n"
        "            # never passed to the engine, so lowering it had no effect on vLLM's\n"
        "            # concurrency (and therefore on its KV-cache pressure).\n"
        '            max_num_seqs=int(config.get("max_num_seqs") or 1024),\n'
        "            enable_chunked_prefill=config.enable_chunked_prefill,\n"
    )

    def apply(self, worktree: Path) -> PatchResult:
        path = worktree / self.ROLLOUT
        text = _read(path)
        changed = False
        if self.MARKER not in text:
            text = replace_once(text, self.ANCHOR, self.REPLACEMENT, what="vLLM engine kwargs")
            _write(path, text)
            _assert_compiles(path, what=self.name)
            changed = True
        return PatchResult(
            name=self.name,
            changed=changed,
            files=[self.ROLLOUT],
            details={"vllm_rollout_sha256": sha256_text(_read(path))},
        )

    def verify(self, worktree: Path) -> tuple[bool, list[str]]:
        problems: list[str] = []
        text = _read(worktree / self.ROLLOUT)
        if self.MARKER not in text:
            problems.append(f"{self.ROLLOUT}: vLLM engine limits marker missing")
        if 'max_num_seqs=int(config.get("max_num_seqs") or 1024),' not in text:
            problems.append(f"{self.ROLLOUT}: max_num_seqs is not forwarded to LLM(...)")
        try:
            compile(text, str(worktree / self.ROLLOUT), "exec")
        except SyntaxError as exc:
            problems.append(f"{self.ROLLOUT}: does not compile after patching: {exc}")
        return not problems, problems


ALL_PATCHES: tuple[BackendPatch, ...] = (
    ChatTemplateKwargsPatch(),
    ResumeWorkerStatePatch(),
    MathValidationPatch(),
    ValidationScopePatch(),
    TeacherAwareMicroBatchPatch(),
    VllmEngineLimitsPatch(),
)


def apply_patches(worktree: Path, *, patch_names: list[str] | None = None) -> list[PatchResult]:
    results = []
    for patch in ALL_PATCHES:
        if patch_names and patch.name not in patch_names:
            continue
        result = patch.apply(worktree)
        log(f"{'applied' if result.changed else 'already applied'}: {patch.name} -> {worktree.name}")
        results.append(result)
    return results


def verify_patches(worktree: Path, *, patch_names: list[str] | None = None) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for patch in ALL_PATCHES:
        if patch_names and patch.name not in patch_names:
            continue
        ok, patch_problems = patch.verify(worktree)
        if not ok:
            problems.extend(f"{patch.name}: {problem}" for problem in patch_problems)
    return not problems, problems


__all__ = [
    "ALL_PATCHES",
    "BackendPatch",
    "ChatTemplateKwargsPatch",
    "MathValidationPatch",
    "ValidationScopePatch",
    "PatchResult",
    "ResumeWorkerStatePatch",
    "TeacherAwareMicroBatchPatch",
    "VllmEngineLimitsPatch",
    "apply_patches",
    "verify_patches",
]
