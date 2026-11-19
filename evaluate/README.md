# 通用外部评测入口

外部评测脚本默认回答上限为 **16384 tokens**，总上下文上限为 **18432 tokens**
（为提示词预留 2048）。`scripts/eval.sh` 和 `run_extended_eval.sh` 均继承此设置，
默认 profile 名称也改为 `16k`。可用 `--max-tokens`、`--max-model-len` 临时覆盖；
训练 rollout 的长度与已保存的实验配置不受影响。

## H200：单入口八任务、统一 n=4

2026-11-17新增 [run_all_eval.sh](run_all_eval.sh)，复用现有 `evaluate.py`，默认GPU0、八项任务全部每题采样4次（包含MATH-500），输出16384/上下文18432，non-thinking和原评分规则不变。按模型→任务顺序串行执行；默认依次评base、SeqKD、GAD，每个模型跑完八任务后再切换模型。每项生成仍使用现有独立sampler进程，不新增常驻推理服务。

```bash
cd /root/siton-tmp/home/jinbo/bloxopd/LMOps
export GAD_PYTHON=/root/siton-tmp/home/jinbo/miniconda3/envs/gad/bin/python
bash evaluate/run_all_eval.sh \
  --run /tmp/bloxopd_h200_runs/h200_baseline_06b_17k_n4 --dry-run
# 正式运行：去掉 --dry-run。只评GAD时追加 --methods gad。
```

任务顺序为AMC23、AIME24、AIME25、MATH-500、Minerva、Olympiad-Bench、GPQA、MMLU-Pro-Math。扩展四项仍使用`extended_dapo_v2`；新结果统一写到run下的`evaluation/all8_dapo_v2_af1_n4_16k_h200_graph/`，不与旧core4/ext4不同采样次数的profile混用。当前数据共3096题，每模型12384条回答，默认三模型合计37152条。重复同一命令沿用已有身份和完整性检查，跳过已完整评完的任务；`--summary`只读取汇总，`--max-prompts 2`会使用独立的`_smoke2`目录。旧入口和历史协议保持原样。

单卡H200默认传入`--no-enforce-eager`，允许vLLM V1使用torch.compile和CUDA Graph；显存预算保持`0.85`，采样器提交批量保持256，未指定时设置`OMP_NUM_THREADS=4`。`gpu-memory-utilization`表示显存预算比例，并非GPU计算利用率；继续调高不一定更快。Qwen3-0.6B在当前16k输出设置下仍需要较大的KV缓存，因此先保留这些批量和显存设置。每个任务仍会单独初始化引擎、捕获CUDA Graph，首次还会编译；编译缓存默认放在`/tmp/bloxopd_h200_cache/vllm`，可通过`VLLM_CACHE_ROOT`覆盖。评测次数、长度、温度、seed和评分规则不变，但执行方式变化可能导致采样结果不同，故使用独立profile。

2026-11-17实测最终GAD step542的AMC23前8题×4，三轮均正常退出、32条回答完整且评分完成：eager采样247.5秒，Graph首次冷启动256.8秒，复用编译缓存后176.9秒。缓存复用后耗时减少28.5%，按实际输出token数归一化的吞吐提高约30%（两种模式回答长度不同）。计时包含sampler/引擎启动、编译及图捕获，不含checkpoint导出与外部评分；这只是32条回答的短测，不代表完整八任务或256条提交批量的固定加速比。详细数据见[单卡性能报告](../reports/h200_evaluation_20261117/single_gpu_speed_comparison.json)。正式全量评测尚未启动。

## 原有分组入口

在 [run_eval.sh](run_eval.sh) 顶部设置 `RUN`、`MODELS`、`MODEL_PATHS`、`DATASETS`、GPU 和采样参数，然后运行：

```bash
cd /mnt/jinbo/RLRM/LMOps
bash evaluate/run_eval.sh --dry-run
bash evaluate/run_eval.sh
bash evaluate/run_eval.sh --summary
```

也可以临时选择实验，无需改脚本：

```bash
bash evaluate/run_eval.sh --run qwen3_0.6b_5k --dry-run
bash evaluate/run_eval.sh --run qwen3_0.6b_17k --dry-run
```

当前 1.7B GAD 探索实验有意停在 step 542（Warmup step 271 后再训 1 epoch），可显式评测该步：

```bash
bash evaluate/run_eval.sh --run qwen3_1.7b \
  --checkpoint-step gad=542 \
  --profile oprd_dapo_nonthinking_gad542_n8_8k --dry-run
# 确认计划后去掉 --dry-run 运行
```

顶部的 `RUN` 选择已有运行目录；命令行 `--run` 可传运行目录或已登记的历史别名。模型、训练数据、non-thinking 和阶段预算从该 run 保存的 `train.yaml` 读取，全局模板的后续修改不会改变已有实验的评测依据。`MODELS=(base seqkd warmup gad)` 决定评测哪些模型；删除条目即可缩小范围。若要评测任意本地 Hugging Face checkpoint，加入 `MODEL_PATHS=("candidate=/绝对路径/模型目录")` 和 `MODELS=(candidate)`。`MODEL_PATHS` 指定的自定义 checkpoint 检查 `config.json` 和权重文件；自动定位的训练阶段 checkpoint 还会核对阶段完成标记。`DATASETS` 中每项直接指定数据集名称和测试 Parquet 路径。命令行也接受 `--methods gad --tasks AMC23 --max-prompts 2 --dry-run` 等临时覆盖。

评测固定使用与当前 GAD 训练一致的 **non-thinking** 提示，并核对实验配置及运行快照中的模型和训练数据。已完成的训练阶段若只有分片 checkpoint，正式运行时会先在 CPU 上合并为 Hugging Face 模型并核对权重；`--dry-run` 会显示 `needs_export`。若一个阶段有意停在配置目标之前，可在脚本顶部设置 `CHECKPOINT_STEPS=("gad=542")`，或运行时传入 `--checkpoint-step gad=542`；脚本会核对指定步数的完整 checkpoint 分片并导出对应模型，在计划里标记为显式选择的 checkpoint。未显式指定时，未完成阶段不会被当成最终模型评测，报错会列出最新 checkpoint 步数和配置目标步数。改变评测步数时请更换 `PROFILE`。各数据集默认每题生成 `N=8` 次；MATH-500 使用 `MATH500_N=2`。输出位于该实验 run 目录下的 `evaluation/<PROFILE>/`。更换模型、数据或采样设置时更新 `PROFILE`；脚本会拒绝把不同设置写进已经存在的同名结果目录。设置 `MAX_PROMPTS=2` 时会自动使用单独的 `_smoke2` 目录。

生成复用 OPRD 的 `sft.sample_teacher`；数学题统一使用 `math_unified_v1`，不再同时输出 OPRD/CARD 两套竞争口径。选择题使用评估专用的 `choice_exact_v2`，按明确的最终选项精确匹配，记录任务自己的评分器身份；Minerva 另有评估侧单位规则 `minerva_units_v1`。`--task-n TASK=N` 可按任务设置每题采样次数。评测在训练进程占用同一 run 时拒绝启动；`--dry-run` 和 `--summary` 不占用 GPU。完整评测结束后脚本会自动打印汇总表。

新增的 Minerva、Olympiad-Bench、GPQA Diamond 和 MMLU-Pro 数学子集通过 [run_extended_eval.sh](run_extended_eval.sh) 选择，默认使用与训练相同 DAPO 提示词的 `extended_dapo_v2` 数据（选择题另要求只输出选项字母）；原 `run_eval.sh` 的四任务默认设置保留。数据范围、评分限制及命令见 [扩展评测说明](EXTENDED_EVALUATION.md)。

`evaluate.py` 和 `summarize_eval.py` 是通用实现；当前H200八任务实验使用 `run_all_eval.sh`，原有分组协议分别使用 `run_eval.sh`、`run_extended_eval.sh`。跨 run、跨 profile 的各任务准确率及均值表用 `python evaluate/summarize_runs.py RUN [RUN ...]`（或 `--all`），说明见 [扩展评测说明](EXTENDED_EVALUATION.md#运行)。

## 单独导出模型

不启动评测时，使用通用导出入口。默认导出该阶段最新完整 checkpoint，也可用 `--step` 选择保留的历史步数：

```bash
bash scripts/export_model.sh --run qwen3_1.7b --stage gad --step 542 --dry-run
bash scripts/export_model.sh --run qwen3_1.7b --stage gad --step 542
```

脚本会核对训练分片，将 actor 合并为 `actor/huggingface/`，逐项检查合并后的权重，并记录 `eval_export_step_<步数>.json`。已通过核对的同一步模型再次调用会直接复用。
