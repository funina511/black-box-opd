> Qwen3-1.7B 当前实验请从 `LMOps/scripts/sample.sh` 和
> `LMOps/sample/configs/sample.yaml` 进入；专用数据在
> `LMOps/data/dapo_math_17k_gad/`。下文的脚本说明适用于通用采样实现。

# `LMOps/sample` — offline data production for GAD

Turns **DAPO-Math-17k** into the three artifacts the three GAD stages consume:

```
DAPO parquet ──dedupe/split──▶ train.parquet + dev.parquet        (prepare_prompts.py)
                              ──teacher sampling──▶ teacher_index.jsonl  (sample_teacher.py)
                              ──conversion────────▶ gad_train.parquet + gad_dev.parquet (build_gad_data.py)
                              ──validation────────▶ validation_report.json (validate_data.py)
```

Everything here is CPU-only and independent: it never imports CARD, never imports
`verl`, and does not require the CARD checkout on `PYTHONPATH`.

## Quick start

Two interchangeable teacher backends, selected by `teacher.engine`:

| engine | processes | when to use |
| --- | --- | --- |
| `vllm_offline` | **one** (teacher loaded inside the sampler) | the weights are on this machine: no port, no server to babysit, no sudo needed, logs go to a file |
| `openai` | two (server + sampler) | the teacher is served elsewhere / by someone else, or you reuse a CARD index |

```bash
# 0) install the (small) dependency set (vllm only for the offline engine)
pip install -r LMOps/sample/requirements.txt

# 1) copy and edit the example config
cp LMOps/sample/configs/sample.example.yaml /abs/path/sample.yaml
$EDITOR /abs/path/sample.yaml          # set teacher.engine + paths

# 2) dry-run first: prints every command, downloads nothing, samples nothing
bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml --dry-run

# 3a) in-process teacher (recommended): one command, detach it and walk away
setsid nohup bash LMOps/sample/scripts/run_sample.sh \
  --config /abs/path/sample.yaml --python /abs/path/envs/gad/bin/python --stage sample \
  > /dev/null 2>&1 < /dev/null &
tail -f /abs/path/data/logs/sample.log       # progress; safe to close the terminal

# 3b) HTTP teacher instead: start the server once, then sample
TEACHER_MODEL_PATH=/models/Qwen3-30B-A3B-Instruct-2507 TEACHER_TP=4 \
  bash LMOps/sample/scripts/serve_teacher.sh > /abs/path/data/logs/teacher.log 2>&1 &
bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml

# 4) build the GAD parquets and check them
bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml --stage convert
bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml --stage validate
```

The engine is part of the fingerprint, so an index produced through HTTP is
**not** silently extended by the offline engine (and vice versa): switch engines
by moving the old index aside.

## Scripts

| Script | Purpose |
| --- | --- |
| `prepare_prompts.py` | download/read DAPO, de-duplicate by `extra_info.index`, check same-id conflicts, fix the train/dev split, write manifests |
| `sample_teacher.py` | sample `m` teacher answers per prompt (in-process vLLM or an OpenAI-compatible API), write the index, resume safely |
| `teacher_engines.py` | the two sampling backends behind one interface (`OpenAIEngine`, `VllmOfflineEngine`) |
| `build_gad_data.py` | join prompts with the index and emit the GAD `content` + `teacher_response` parquet |
| `validate_data.py` | enforce the data contract (uniqueness, leakage, empty answers, ground-truth leakage, length budget) |
| `prepare_eval_data.py` | build the external AIME/AMC/MATH500 eval parquet with an explicit availability report |
| `make_smoke_data.py` | build a tiny **synthetic** dataset (clearly marked) for CPU checks and GPU smoke runs |
| `serve_teacher.sh` | launch the vLLM OpenAI-compatible teacher endpoint |
| `run_sample.sh` | run the whole data pipeline from one YAML config |

## Library modules

* `task_template.py` — the shared math instruction, prompt canonicalization and
  hashing, teacher-fingerprint payload. The instruction is byte-identical to
  CARD's so prompts remain comparable.
* `teacher_index.py` — strict index reader/writer (never deletes, never skips
  silently) plus CARD `card.teacher_index.v1/v2` import support.
* `math_scorer.py` — vendored CARD `math_dapo.compute_score` (Apache-2.0, see the
  file header). Accuracy must use `acc`; the mean of `score` (±1) is not an
  accuracy.

## Data contract

GAD train/val parquet columns:

| Column | Meaning |
| --- | --- |
| `content` | `[{role, content}]` chat prompt — used via `data.prompt_key=content` |
| `teacher_response` | teacher text for this prompt (never token ids) |
| `data_source`, `ability` | provenance labels |
| `reward_model.ground_truth` | reference answer, used by scoring/diagnostics only |
| `extra_info.index` | stable uid (also the teacher-index key) |

The ground truth never enters the prompt and never becomes the GAD reward.

## Notes / limits

* DAPO ships ~100 copies of every problem. The de-duplicated count is expected to
  be **17,917**; `prepare_prompts.py --expected-unique 17917` fails when it is not,
  which is how a changed upstream file is caught.
* The train/dev split is drawn **per prompt, not per uid**: 495 DAPO questions
  appear under two different `extra_info.index` values, and splitting uids
  directly put 27 of them on both sides (dev questions seen during training, and
  `validate_data.py` refusing the pair). A prompt-hash group is now drawn as a
  whole, so the two ids of a question always land on the same side; the split
  manifest records `unit`, group counts, the actual dev size and any overshoot
  (`dev_size_overshoot`), and `leaked_prompt_hashes` must be empty.
* An external dev set (`prompts.dev_file`) is checked for overlap with the
  training prompts as well: the run aborts unless `--allow-dev-leak` is passed,
  which records the overlap instead (`validate_data.py --allow-val-leak` does the
  same on the contract check).
* The default teacher selection is `all` (alias `first`): the **first valid answer**,
  deterministic, no correctness filtering, so all three stages see identical teacher text.
  `run_sample.sh` picks it with `gad_data.select`, and `gad_data.select_val` controls the
  dev split separately (the original leaves its validation set unfiltered). The other modes
  are deliberate experiments that drop prompts, matching the original harness's pools:
  `nontruncated` (drop answers the sampler marked `truncated`), `correct` (teacher answer
  must match the ground truth) and `correct-nontruncated` (both - the original's default
  pool). Deliberate filtering is reported in the sidecar (`num_dropped`, `reasons`,
  `teacher_truncated_selected`), not treated as missing data; only prompts the teacher never
  answered, or prompts absent from the index, are errors.
* `prompts.card_full_train: true` reproduces CARD's "all prompts in train" budget:
  there is no held-out dev split from DAPO, so an independent dev set
  (`prompts.dev_file`) must be provided. The two configurations do **not** have
  the same data budget and reports must say which one was used.
* `run_sample.sh --stage` accepts `prompts`, `sample`, `convert_train`,
  `convert_dev`, `validate`, `eval`, plus the group alias `convert`
  (= `convert_train` + `convert_dev`). An unknown name exits with code 2 instead
  of silently skipping every stage.
