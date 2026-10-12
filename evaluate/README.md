# 通用外部评测入口

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

生成与原始评分复用 OPRD 的 `sft.sample_teacher` 和 `sft.score_math`，同一批结果再用 LMOps 训练时的 CARD 评分器复算，汇总时分别列出两种口径。评测在训练进程占用同一 run 时拒绝启动；`--dry-run` 和 `--summary` 不占用 GPU。完整评测结束后脚本会自动打印汇总表。

`evaluate.py` 和 `summarize_eval.py` 是通用实现，日常只需使用和编辑 `run_eval.sh`。

## 单独导出模型

不启动评测时，使用通用导出入口。默认导出该阶段最新完整 checkpoint，也可用 `--step` 选择保留的历史步数：

```bash
bash scripts/export_model.sh --run qwen3_1.7b --stage gad --step 542 --dry-run
bash scripts/export_model.sh --run qwen3_1.7b --stage gad --step 542
```

脚本会核对训练分片，将 actor 合并为 `actor/huggingface/`，逐项检查合并后的权重，并记录 `eval_export_step_<步数>.json`。已通过核对的同一步模型再次调用会直接复用。
