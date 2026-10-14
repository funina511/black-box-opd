# GAD 在线 Rubric 与历史 Replay

统一配置为 `LMOps/train.yaml`，启动入口为 `scripts/train.sh`。
`rubric.enabled` 与 `replay.enabled` 均默认关闭。
工程证据见 [ONLINE_RUBRIC_VERIFICATION.md](ONLINE_RUBRIC_VERIFICATION.md)；
工程冒烟不用于判断 actor 数学性能。完整对照实验由用户启动。

## 训练行为

| 配置 | Warmup critic | GAD critic | Judge |
| --- | --- | --- | --- |
| 两开关关闭 | 原成对损失、单头 | 原成对损失、单头 | 不请求 |
| rubric / correctness_only | 成对损失＋答案监督、四通道 | 延续本组 Warmup 的四通道 | 不请求 |
| rubric / joint | 成对损失＋三个辅助维度 | 延续四通道与联合目标 | 独立服务 |
| replay-only | 原单头训练 | 固定数量的当前/历史比较 | 不请求 |
| rubric＋replay | rubric 联合训练 | 历史比较＋rubric 联合训练 | 只标注选中比较 |

主方案 rubric 阶段为 `[warmup,gad]`，从 Warmup 首次 critic 更新生效。
Warmup actor 保留 teacher SFT 和原门槛；GAD actor 保留当前 rollout、adv 奖励、
GRPO/PPO 与 KL 设置。三个辅助分数不混入 actor reward。
`rubric.stages=[gad]` 是阶段消融；Warmup-only、Warmup replay、动态 critic 批次、
SP>1、多节点及改变 world size 的恢复暂不支持。

扩展采用独立 worktree，源码位于 `gad/extensions/` 和 `rubric/core/`，
由 `gad/tools/algorithm_patches.py` 部署到目标 backend。
已经启动训练的 backend 不随源码更新而热修改；更新算法需新 backend root 和新实验。
从全局配置新建实验时会自动按代码/补丁指纹选择新的 backend root；已保存run的root保持固定。

## Judge 服务

当前已配置本地 Qwen3-8B、BF16、TP=2、non-thinking；judge GPU 为 6,7，
训练 GPU 为 1,2,3,4。服务和 Ray 训练资源分开。
API/权重/tokenizer/模板/schema 指纹都需要匹配，健康检查会核实模型身份。

```bash
cd /mnt/jinbo/RLRM/LMOps
bash scripts/rubric_judge.sh --action start --dry-run
bash scripts/rubric_judge.sh --action start
# 模型加载后，health 成功才启动 joint 训练。
bash scripts/rubric_judge.sh --action health
bash scripts/rubric_judge.sh --action logs
```

```bash
# 结束使用时，仅停止此脚本拥有的进程组。
bash scripts/rubric_judge.sh --action stop
```

服务使用本机 vLLM 0.8.5 的 `guided_json` / `xgrammar`，不将格式错误猜成分值。
正确性使用原 CARD 规则评分器；截断或未知过程标签 mask，单条标注错误保留原因。
整步服务不可用时，在 optimizer 更新前失败；从最后完整 checkpoint 恢复。

## A/B/C 对照启动

为各组使用相同的初始 adv 行，先准备一次确定性单头 critic。该目录是派生初始化，
不会覆盖原学生模型；已有正确的同名初始化产物会核验后复用。

```bash
cd /mnt/jinbo/RLRM/LMOps
/mnt/jinbo/miniconda3/envs/gad/bin/python gad/tools/prepare_rubric_critic.py \
  --source /mnt/jinbo/RLRM/model/Qwen/Qwen3-0.6B \
  --target /mnt/jinbo/RLRM/LMOps/runs/_initialization/qwen3_0.6b_adv_seed42 \
  --single-head --seed 42
CRITIC_INIT=/mnt/jinbo/RLRM/LMOps/runs/_initialization/qwen3_0.6b_adv_seed42
```

以下继承全局相同的数据、完整 Warmup 预算、长度和温度；GAD 各增加 80 步。
显式设置相同 rollout seed，避免扩展组的确定性采样设置成为混淆。
每条命令先带 `--dry-run` 检查，正式执行时去掉它。

```bash
# A：原版两阶段基线。
bash scripts/train.sh --stages warmup,gad --name gad_baseline \
  --gad-new-steps 80 --set models.discriminator_init="$CRITIC_INIT" \
  --set rollout.seed=42 --set rubric.enabled=false --set replay.enabled=false --dry-run

# B：两阶段正确性辅助，不需要 judge。
bash scripts/train.sh --stages warmup,gad --name gad_correctness \
  --gad-new-steps 80 --set models.discriminator_init="$CRITIC_INIT" \
  --set rollout.seed=42 --set rubric.enabled=true \
  --set rubric.mode=correctness_only --set replay.enabled=false --dry-run

# C：两阶段完整 rubric，先启动 judge。
bash scripts/train.sh --stages warmup,gad --name gad_rubric \
  --gad-new-steps 80 --set models.discriminator_init="$CRITIC_INIT" \
  --set rollout.seed=42 --set rubric.enabled=true \
  --set rubric.mode=joint --set replay.enabled=false --dry-run
```

Warmup 结束于实际绝对步 S 时，`--gad-new-steps 80` 的终点为 S+80，
优先于 `budget.stage_epochs.gad`；未完成 Warmup 时 dry-run 给出预计 S。
各组各自重新训练 Warmup，SeqKD 是独立基线，不初始化 Warmup。
replay 的对照在相同 Warmup 来源和相同 GAD 新步数下追加，设置 `--set replay.enabled=true`。

## Resume 与阶段消融

每个新 run 保存自己的 `train.yaml`。以后用 run 目录选配置：

```bash
bash scripts/train.sh --run /绝对路径/实验目录 --resume --stages warmup,gad
bash scripts/train.sh --run /绝对路径/实验目录 --resume --stages gad
```

恢复不允许修改模型、数据、rubric/judge 版本、功能开关和预算。
同阶段恢复模型、optimizer、scheduler、dataloader、worker RNG、driver RNG、
replay 和标签索引；Warmup→GAD 按原约定新建 optimizer，不加载 Warmup optimizer。
标签按实际回答精确匹配，两阶段可复用匹配缓存；GAD replay 从空池开始。

```bash
# 显式新建一个 GAD 消融，源 Warmup 只读。
bash scripts/train.sh --stages gad --name gad_rubric_gad_only \
  --init-from-warmup-run qwen3_0.6b_17k --init-step 271 --gad-new-steps 80 \
  --set rollout.seed=42 --set rubric.enabled=true \
  --set 'rubric.stages=[gad]' --set rubric.mode=joint --dry-run
```

源模型和训练数据必须匹配；旧 Warmup 的训练条件不同则只用于工程检查。
单头源在派生 run 中扩头一次，四头源严格核验后完整加载。

## 工程冒烟与日志

`gad/tools/prepare_online_smoke.py` 创建独立的 16 题工程数据，选有完整短 teacher 的题目，
随后仍使用当前 actor 在线采样。该集合不能作为数学能力评价。
`--warmup-smoke-steps` 显式缩短新 run 的 Warmup，并标为 engineering smoke。
`--max-stage-steps` 只限制本次调用，不缩短计划预算，便于测试新进程恢复。
解除已保存的调用限制使用 `--set run.max_stage_steps=null`。

```bash
bash scripts/train.sh --run /绝对路径/工程实验目录 --resume --stages warmup \
  --set run.max_stage_steps=null
```

每阶段保留 `extension_metrics.jsonl`、前32条唯一回答的 `rubric_calibration.jsonl`、
judge 错误原文；人工复核状态保持 pending。
`rubric/json_valid_count` / `json_requested_count` 提供格式检查分母，
逐维 coverage、unknown 原因、teacher/student 截断、replay 比例/年龄/token 数均可见。
MFU 沿用原 student token 口径，额外 token 指标包含 teacher 与 student。
`rubric/judge_seconds`包含整个标签准备屏障；judge请求数为0时该耗时仍可含规则评分。

完整 checkpoint 位于 `global_step_<S>/training_extensions/`，由
`checkpoint_complete.json` 原子发布；残缺目录不能当作同阶段恢复点。
默认保留最近5次完整保存，交接/导出目录额外保护。

## Actor 效果评价由用户执行

每组训练后使用现有通用评测入口，记录 base、Warmup末和GAD末三点：

```bash
bash evaluate/run_eval.sh --run /绝对路径/实验目录 \
  --methods base,warmup,gad --profile rubric_actor_comparison --dry-run
# 去掉 --dry-run 正式评测；各组采用相同数据和解码设置。
```

工程冒烟不比较数学准确率。正式实验比较相同 Warmup＋GAD预算后的 actor 性能，
同时报告各自 Warmup 起点、输出长度、截断和额外 judge 成本。
