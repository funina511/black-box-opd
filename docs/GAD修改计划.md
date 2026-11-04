# GAD 修改计划（官方算法 + DAPO-Math-17k 数学推理）

本次交付为源码对照与实施计划；尚未修改训练算法、创建可运行的采样/训练脚本或启动训练。

训练任务已明确为 **DAPO-Math-17k 数学推理**，替代 LMSYS。此次修订将 DAPO 数据准备、长推理模板与长度适配、数学正确率评测列为首版必做项。数学标准答案用于评测，GAD 训练奖励仍来自原版判别器。

已确认的实验设置：按 SeqKD → Warmup → GAD 的时间顺序执行；SeqKD 是独立基线，Warmup 从初始学生模型开始，GAD 接续 Warmup。SeqKD 的输出不作为 Warmup 初始化。

## 1. 修改原则与源码基线

- 以 `LMOps/gad` 及其嵌套的 `verl` 仓库为算法基准，CARD 提供 DAPO 数据准备、数学提示、采样与评测协议、参数组织和运行管理参考。
- 保留官方判别器结构、损失、Warmup actor 更新、GAD actor 更新、更新顺序和阶段初始化语义。
- 优先增加外围脚本和数据转换；确实需要修改后端时，将数据/模板兼容补丁与 checkpoint 工程补丁分开，逐项记录原因。
- 不把 CARD 的新版 `verl` 整体复制进来，不直接采用 `main_gad.py` / `RayGADTrainer` 替换原版训练器。

本地已存在完整的三个远程分支，无需为了制定计划再次下载：

| 阶段 | 官方分支 | 当前对应 commit |
| --- | --- | --- |
| SeqKD | `origin/seqkd` | `c8b3a62f5edd4994e13a15a461cf9527973695bd` |
| Warmup | `origin/warmup` | `b6e88286c3e16d6f364e698f60c3be608f1a36cc` |
| GAD | `origin/gad` | `707757b9202c466d986e529a4658434d0ac6a7ee` |

`LMOps` 当前 commit 为 `6f53ef41c951de0962bbf27fe3f5819eabc8ce0e`；嵌套 `gad/verl` 当前检出 `gad` 且工作树干净。父仓库将 `gad/verl/` 显示为未跟踪目录，后续需单独记录其版本，不能只记录父仓库版本。

## 2. 源码对照结论

| 项目 | 原版实际实现 | CARD 实现及采用决定 |
| --- | --- | --- |
| 阶段组织 | SeqKD、Warmup、GAD 位于不同分支，均从 `verl.trainer.main_ppo` 启动 | CARD 使用独立 GAD trainer；仅参考启动配置组织 |
| 判别器 | critic 被改作单标量 value head，对有效 response token 的分数求和 | CARD 同样采用 teacher/student 排序损失，但微批实现等有差异；保留原版 |
| Warmup | student 在线生成负样本；判别器训练；actor 按 uid 去重后对教师回答做 SFT，并受 `critic_warmup` 门控 | CARD 用普通 SFT 权重初始化 actor 和新判别器，再短暂预热判别器；不采用这种替代流程 |
| GAD 更新 | 判别器分数 → GRPO 优势 → 更新判别器 → 更新 actor；actor 使用官方策略损失和 KL 设置 | 保留原版实际调用顺序、mask、聚合方式 |
| 阶段衔接 | Warmup 的 actor 和 critic 分别合并为 HF 权重后加载；恢复全局步数和 dataloader 状态 | CARD 从 SFT 初始化新判别器，缺失时可回退 base；不采用这种回退 |
| 教师数据 | `content` 消息列表 + `teacher_response` 字符串，后者由学生 tokenizer 编码 | CARD 使用 `prompt` + 教师索引；通过离线转换适配 |
| 验证 | 读取教师回答并计算 ROUGE-L | 新增数学正确率作为主要指标，ROUGE-L 保留为辅助诊断；外部数学基准评测列为首版工作 |
| 依赖 | 原版 README 指定 Python 3.10.12 / PyTorch 2.6.0 / vLLM 0.8.5 | CARD 使用另一套后端；环境隔离，不直接照搬依赖或 SDPA 开关 |

必须冻结的算法细节：

1. 判别器使用 `num_labels=1` 的 value-head 模型；序列分数是有效响应 token 分数之和，不改成均值、最后一个 token 分数或二分类概率。
2. 损失保持 `-logsigmoid(sum(teacher_values * teacher_mask) - sum(student_values * student_mask)).mean()`。
3. Warmup 保留 `deduplicate_by_uid()`、teacher forward、`compute_sft_loss()` 和判别器训练。其 actor 实际采用 SFT 损失，不能因为启动脚本带 `use_kl_loss=True` 就擅自往 Warmup actor 损失中增加 KL。
4. GAD 保留 rollout 分组、GRPO 归一化、PPO clipping、KL、梯度累积及原有更新频率；同一步奖励来自判别器更新前的前向结果。
5. 保留原版 teacher/student 的 padding、EOS、position ids 和 response mask 语义；工程适配必须验证 tensor 对齐。

关键源码位置：

- 原版：`gad/verl/verl/workers/critic/dp_critic.py`、`trainer/ppo/core_algos.py`、`workers/actor/dp_actor.py`、`trainer/ppo/ray_trainer.py`、`workers/fsdp_workers.py`（后四项相对 `gad/verl/verl/`）。
- 数据：`gad/verl/verl/utils/dataset/rl_dataset.py`、`gad/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`。
- CARD 参考：`../CARD/scripts/220-run-gad.sh`、`../CARD/scripts/build_teacher_index.py`、`../CARD/scripts/build_gad_data.py`、`../CARD/algo/card_teacher_index.py`。

## 3. 先解决的工程问题

### 3.1 三个算法分支的安全调用

为三个固定 commit 创建独立的 detached worktree，例如 `gad/.backends/{seqkd,warmup,gad}`。原有 `gad/verl` 保持为源码基准；不在运行中的工作目录反复执行 `git checkout`。

每个阶段启动独立 Python/Ray 进程，显式绑定对应 backend 的 `PYTHONPATH`，并在 driver 与 Ray worker 检查 `verl.__file__`、阶段和 commit，防止误导入 CARD 或其他项目安装的 `verl`。必须避免连接上一阶段残留的 Ray 集群；只管理本次任务创建的进程。worktree 目录、输出和缓存加入合适的忽略规则。

第一版以单机多 GPU 为验收范围；节点数、GPU 数、TP 和 batch 参数均可配置，多机作为独立验证项，不能把 `nnodes` 参数存在视为多机已验证。

### 3.2 参数与训练预算

新增统一配置，显式提供学生模型、判别器初始模型、训练/验证文件、输出目录、各阶段训练预算、GPU、TP、batch、token 长度和日志后端。

- 原版训练脚本硬编码 `/tmp`、8 GPU、TP=2 和 W&B 占位值；这些改为参数，不修改 loss 代码。
- 原版 Warmup README 写 1 epoch，而 shell 脚本写 `total_epochs=2`。计划的默认实验配置采用 README 的 **1 epoch**，并在最终配置中明确记录；另保留复用脚本值的配置方式，避免混淆“逻辑保留”和“训练预算一致”。
- 原版 SeqKD / GAD shell 的 epoch 值均为 4，但 GAD 恢复了 Warmup 的 global step，不能把 4 直接解释为额外训练 4 epoch。
- 对用户暴露 `seqkd_steps`、`warmup_steps`、`gad_steps` 为各阶段新增步数；若 Warmup 最后一步是 S，GAD 新增 K 步，原版 trainer 的全局停止目标应为 S+K，并保证 epoch 循环足以到达该目标。
- 首轮保留原版学习率：SeqKD actor 5e-6；Warmup/GAD actor 与 critic 1e-6。保留原版脚本的 rollout n=8、temperature=0.8、grad_clip=0.2；GAD actor KL 系数 0.001。资源缩小或长度调整单独记录。
- 保留原版阶段门控：SeqKD/Warmup 脚本 `critic_warmup=10`，GAD 为 0。短烟测必须覆盖门控后的 actor 更新，不能仅跑两步就声称 Warmup 验证通过。

### 3.3 Warmup → GAD 和中断恢复

从 Warmup 完成清单中读取最后一个**完整** checkpoint；不写死 STEP=800，不以目录名字典序猜最后一步。

自动执行：

1. 检查 actor、critic 分片、模型配置、tokenizer、`data.pt` 和实际 global step；缺失时报错。
2. 使用原版 `tools/merge_model2hf.py` 分别导出 actor 与 critic；核验 critic 仍是单标量判别器，且判别头权重被完整保留。
3. 建立独立 GAD 输出目录，复制衔接所需状态，设置明确的 checkpoint 路径及 latest 标记，禁止覆盖 Warmup 产物。
4. GAD actor 从 Warmup actor 加载，critic 从 Warmup critic 加载；reference 的初始化沿用原版 GAD worker 路径，不额外固定为最初学生模型。
5. 验证起始步数、dataloader 位置和 S+K 目标步数；记录导出来源和文件校验值。

**需要特别区分：**原版 GAD `_load_checkpoint()` 中 actor/critic 的 `load_checkpoint()` 已被注释，作者脚本依赖预先合并的 HF 权重。默认跨阶段衔接忠实保留这一行为：权重承接、步数和 dataloader 恢复，优化器重新初始化。不能宣称原版跨阶段恢复了完整 optimizer/scheduler。

同阶段中断恢复单独实现和验证：优先设计显式恢复模式，恢复该阶段 actor/critic optimizer、scheduler、随机状态和 dataloader，并保持原来的 reference 锚点。该模式属于 checkpoint 工程补丁，不改变正常无中断运行的算法；若尚未验证，脚本必须拒绝将仅恢复步数的行为标成“完整续训”。

### 3.4 数学任务适配（本次修订新增，首版必做）

**数据源与去重。**参考 CARD 的 `scripts/prepare_data.py`，输入支持本地 DAPO parquet，或显式下载其使用的 `BytedTsinghua-SIA/DAPO-Math-17k`、`data/dapo-math-17k.parquet`；下载时固定 revision 并记录文件哈希。CARD 源码说明该文件含重复题目，按 `extra_info.index` 去重后为 17,917 题；这是本地代码中的预期值，本次未下载实际文件验证。实施时打印实际原始/去重数量，检查同 id 的 prompt 和 ground truth 是否一致，不机械截取前 17,000 行；相同题面不同 id 也要检查并保存 id 映射。

**训练/验证划分。**默认建议用固定 seed=42 从唯一题目中留出 512 题作为 `dapo_dev`，其余训练，并保存划分清单；此为计划默认值，可配置。若实际唯一题数为 17,917，则清理其他异常前为 17,405 train / 512 dev。三阶段使用完全相同的 train/dev 和教师回答。该划分与 CARD 全量 DAPO 训练不同，报告中明确标注；另提供 `card_full_train` 配置，使用全部训练题和显式提供的独立开发集。不能声称两种配置具有完全相同的数据预算。

**教师解答。**DAPO 的数学答案标签不是教师推理轨迹。每道 train/dev 题仍需教师生成完整解答，或导入匹配当前题面与模板的 CARD 教师索引。`teacher_response` 保存完整推理和最终答案；`reward_model.ground_truth` 单独保存标准答案字符串。不得把 ground truth 拼进模型输入或作为教师解答替代品。沿用 `select=all` 的固定第一条有效回答，保留错误但有效的教师解答；缺失、损坏、截断等数据质量问题单独统计处理，不做默认正确性筛选。

**提示与答案格式。**DAPO 已有的消息模板原样保留，避免重复套数学提示；需从裸题构造 prompt 时复用 CARD 的逐步解题及末行 `Answer: ...` 约定。教师采样、SeqKD、Warmup、GAD 和评测使用同一任务指令。记录 tokenizer/chat-template 哈希；针对支持 thinking 开关的学生，提供 CARD 对齐配置 `enable_thinking=False`，在三个 backend 的长度过滤、实际编码及评测路径一致生效。教师采用其模型支持的模板，保留完整文本输出并记录推理模式；不假设教师和学生 tokenizer 相同。原版 loader 未直接转发该模板参数，需加最小兼容补丁，不能只添加一个未被实际使用的 Hydra 配置。

**长度与资源。**数学配置以 CARD 的 prompt=2048、response=8192 为起点，要求学生、判别器和 rollout 支持至少 10240 的总上下文；三阶段保持一致。教师采样上限以 8192 起步，但教师和学生 token 数不同，仍需用学生 tokenizer 统计 p50/p95/p99、超限率及最终答案被截断比例。过长样本采用显式策略并生成共同数据清单，不静默只在某一阶段丢弃。短长度 smoke 只用于接口检查，另做真实长序列烟测；原版 1536 不再是数学训练默认值。先通过 microbatch、offload、token budget 调整适配显存，保留有效 batch/梯度累积语义；不能直接将判别器求和分数改成长度均值。同步记录 teacher/student 长度差与判别分数，观察判别器是否依赖截断差异区分二者。

**训练预算。**DAPO 去重、划分和长度检查后，按实际 dataloader 长度计算 epoch 步数，记录 drop-last 丢弃数量。不能继续使用 LMSYS 的约 800 步 Warmup；小数据 smoke 模式显式配置足够 epoch，使其越过原版第 10 步 actor 更新门控。长响应配置也会改变 token 预算，不能照搬原版每卡 max-token 配置后假定显存可用。

**数学验证。**在三个分支 `_validate()` 已生成的 student 文本上，增加统一、无梯度的数学评分。训练期间 `dapo_dev` 每题单次确定性生成，报告 accuracy、答案解析失败率、输出长度、截断率；保留教师回答用于原版 ROUGE-L 和现有 rollout 接口。正确率绝不回填 `token_level_rewards`、GRPO advantage 或判别器标签。初始学生、SeqKD、Warmup、GAD 使用相同开发集和解码设置比较。原版的训练奖励和参数更新无需为此改变。

**评分协议。**训练中诊断参考 CARD `verl/utils/reward_score/math_dapo.py` 的 `compute_score()`：默认提取末次 `Answer:`，做规范化后比较；其返回 `score` 为 ±1，accuracy 应使用 `acc`，不能把 score 均值直接称正确率。记录其仅检查响应末尾 300 字符等行为；对长答案、无 `Answer:`、仅 `\boxed{}`、小数与整数形式等建立对照样例。该规则不保证通用符号等价；若另加更强 verifier，指标分开命名，不静默改变 CARD 对齐指标。最终外部评测另核对 `scripts/eval_paper_math.py` 的实际解析/评分实现，不假定与训练中 scorer 完全相同。

**外部评测。**新增独立 HF/vLLM 数学评测入口，无需给外部测试题生成教师回答。参考 CARD `scripts/prepare_eval_benchmarks.py` 的 AIME24、AIME25、AMC23、AMC24、MATH500；路径全部参数化，检测跨 train/dev/test 的题面重复并记录处理结果。数据缺失时明确列出，不能静默跳过后仍称 all5。开发集用于调参，外部测试集用于最终报告；报告各数据集结果及明确的汇总方式。

CARD `scripts/250-run-gad-eval.sh` 的最终评测配置为 temperature=1.0、top-p=0.95、输出上限 32768，通常每题 16 次、MATH500 每题 2 次，与训练中的确定性验证分开。提供对应配置，但先检查模型上下文与计算预算；对每题多次采样取成功比例，再按题平均估计 pass@1，不能将“任一次成功”的 pass@n 当 pass@1。统一评测学生初始模型和三个阶段产物，保存逐题输出、答案和评分。

新增源码参考：`../CARD/scripts/prepare_data.py`、`../CARD/algo/card_prompts.py`、`../CARD/scripts/prepare_eval_benchmarks.py`、`../CARD/scripts/250-run-gad-eval.sh`、`../CARD/scripts/eval_paper_math.py`、`../CARD/verl/utils/reward_score/math_dapo.py`。

## 4. 一键启动设计

拟新增以下文件（此列表是实施目标，文件尚未创建）：

```text
LMOps/gad/
  configs/pipeline.example.yaml
  scripts/pipeline/run_all.sh
  scripts/pipeline/run_seqkd.sh
  scripts/pipeline/run_warmup.sh
  scripts/pipeline/run_gad.sh
  tools/pipeline.py
  tools/prepare_backends.py
  tools/prepare_gad_checkpoint.py
  tools/preflight.py
  tools/evaluate_math.py
  tools/math_metrics.py
  scripts/pipeline/run_math_eval.sh
  configs/math_eval.example.yaml
```

用户入口：

```bash
# 预期接口，实施完成后可用
bash LMOps/gad/scripts/pipeline/run_all.sh --config /absolute/path/run.yaml --dry-run
bash LMOps/gad/scripts/pipeline/run_all.sh --config /absolute/path/run.yaml
```

执行顺序：DAPO/教师数据预检 → SeqKD → 校验 SeqKD 产物 → Warmup → 校验/导出 Warmup actor+critic → GAD → 导出最终学生模型与运行摘要。三个阶段均包含开发集数学验证；完整外部评测支持单独一键调用，或通过显式 `eval_after_train` 配置在训练结束后串联。

模型依赖关系：

```text
初始学生 ──→ SeqKD ──→ 独立基线模型
初始学生 ──→ Warmup actor ──→ GAD actor ──→ 最终学生模型
判别器初始模型 ──→ Warmup critic ──→ GAD critic
```

一键脚本必须具备：

- `set -euo pipefail`、完整退出码传播、阶段日志和错误阶段报告；上一步失败立即停止。
- `--dry-run` 输出最终配置、backend、完整命令、checkpoint 输入输出和阶段预算，不占 GPU、不采样。
- `--from-stage` / `--to-stage` 支持分段执行；从 GAD 开始必须显式提供已验证的 Warmup checkpoint。
- 保存 resolved config、源码 SHA、数据 manifest、依赖版本、阶段开始/结束步数、模型来源和退出状态。
- 完成标记只在产物验证通过后写入；跳过已完成阶段前比较配置和输入签名，不能仅检测目录存在。
- 输出目录使用独立 run id；配置冲突、checkpoint 缺失、导出失败均报错，不静默回退初始模型。
- 默认 console/file 日志；W&B 作为可选项，从环境读取凭据，不在代码或日志中写密钥。

## 5. `LMOps/sample` 采样方案

可行。本地 CARD 中找到了采样与索引构建代码，但未发现可直接复用的 `.parquet` / `.jsonl` 采样数据文件。因此参考的是 CARD 的采样流程与数据协议；数据源固定为 DAPO-Math-17k，实际本地文件、数据 revision、模型和输出路径由运行配置提供。

拟新增：

```text
LMOps/sample/
  README.md
  requirements.txt
  configs/sample.example.yaml
  prepare_prompts.py
  prepare_eval_data.py
  sample_teacher.py
  teacher_index.py
  build_gad_data.py
  validate_data.py
  scripts/serve_teacher.sh
  scripts/run_sample.sh
  tests/test_data_contract.py
```

数据流程：

```text
DAPO parquet → 去重/保留数学消息与标准答案/固定 train-dev 切分
           → 教师离线采样 → teacher_index.jsonl
           → GAD 数据转换 → train.parquet + val.parquet + manifest.json
```

具体实现要求：

1. 采样使用独立的 OpenAI-compatible 教师接口（可由本地 vLLM 服务提供），只依赖教师文本，不依赖教师 logits；训练阶段不需要教师服务常驻。
2. 参考 CARD 的多回答索引，保存 `schema_version`、`uid`、`raw_prompt_hash`、`teacher_answers`、`teacher_fingerprint`；采样次数 m 可配置，示例按 CARD 使用 m=4。转换默认确定性选择第一个有效回答，三阶段共用同一选择结果。
3. 默认不按答案正确性筛选，不引入 CARD 的多数票/置信度 reward。以后若增加 `select=correct`，标记为独立实验配置。
4. 续采样按 uid + prompt hash + 采样配置指纹核对；指纹包含实际请求使用的模型、模板、温度、top-p、最大输出长度、m 和种子（服务支持时）。兼容导入 CARD 已有索引，同时独立记录新增元数据。
5. 比 CARD 脚本加强检查：相同 uid 不同 prompt、重复记录、损坏 JSONL、指纹冲突、少于 m 个有效回答时不默默跳过；不自动删除旧索引，冲突时报错并保留原文件。
6. 支持并发、超时、有限重试、逐条落盘、失败清单和断点续采样；记录 `finish_reason` 和截断情况。存在未处理失败时返回非零退出码，除非用户显式选择允许不完整数据。
7. 从 DAPO 唯一题目提前固定 train/dev 划分，检测 prompt 泄漏；为 dev 采样教师回答，兼容原版验证接口，同时以数学正确率为主要指标。外部数学测试集无需教师回答，不写入训练数据。
8. 模板管理同时记录教师请求消息和学生实际训练消息，采用第 3.4 节数学模板协议；Qwen3 的 thinking 开关由配置显式控制，不能采样和训练分别使用不同默认值。
9. 使用学生 tokenizer 统计 prompt/teacher answer 长度、超长与截断比例；数学默认长度按第 3.4 节设置，必须完成真实长序列烟测。采样不把 ground truth 发给教师；教师正确率仅作质量统计，不作为默认筛选条件。
10. `sample` 使用独立依赖和模块命名，不导入 CARD 的 `verl`，不要求 CARD 目录在 PYTHONPATH；迁移的必要 helper 保留许可和来源说明。

转换后的训练/验证数据核心字段：

| 字段 | 约定 |
| --- | --- |
| `content` | `[{role, content}, ...]` 消息列表；对应原版 `data.prompt_key=content` |
| `teacher_response` | 非空教师文本；不保存教师 tokenizer token ids |
| `data_source` | 数据来源标签 |
| `extra_info.index` | 稳定样本 id |
| `reward_model.ground_truth` | 必需的数学标准答案字符串，只用于评测/质量统计，不进入模型 prompt 或 GAD reward |
| `reward_model.style` / `ability` / 其他元数据 | 保留来源信息并验证三个分支兼容性；数学 scorer 显式路由，不盲目复制 CARD 的 style 字符串 |

已有 GAD 格式数据直接校验后使用；已有 CARD `prompt` parquet + 索引通过转换器使用。首次 `run_sample.sh` 完成数据准备，之后训练只需一次 `run_all.sh`。可额外提供顶层串联入口，但离线数据生产与三阶段训练保持可单独重跑。

## 6. 实施顺序和验收

| 顺序 | 工作 | 完成标准 |
| --- | --- | --- |
| P0 | 固定官方分支、建立独立 backend、环境预检和配置 | driver/worker 均导入预期分支；dry-run 能展示三阶段完整依赖关系 |
| P1 | DAPO 去重/划分、采样、CARD 索引导入和长推理数据校验 | 三阶段读取同一批题和教师解答；ground truth 不泄漏；重跑不重复采样 |
| P1b | 数学模板、三个分支验证适配和独立评测入口 | 初始学生可获得数学指标；scorer 对照样例通过；评分不进入训练更新 |
| P2 | 单阶段脚本与 checkpoint 导出 | SeqKD、Warmup 独立运行；actor/critic 均可正确导出与重新加载 |
| P3 | 三阶段一键串联 | 顺序正确；Warmup 使用初始学生；GAD 使用 Warmup actor+critic；任一失败停止 |
| P4 | 同阶段恢复补丁 | 补丁有明确开关和回归证据；不修改冻结的算法逻辑 |
| P5 | 完整小规模验收与正式数学配置 | 覆盖长推理、actor 更新、阶段交接、数学验证和保存；统一比较基线与阶段产物 |

必要验证：

- 数学适配：同 id 不同题/答案冲突、重复包装提示、ground truth 泄漏、teacher/student token 长度差异、thinking 模板一致性和真实长序列资源检查。
- 数学评测：答案解析与 CARD 对照、accuracy 与 ±1 score 的区别、pass@1 与 pass@n 的区别；评测不进入训练 reward/advantage 路径。
- 数据契约：正常记录、缺失/空教师回答、重复 uid、prompt 不匹配、指纹变化、截断和 train/val 泄漏。
- 算法保留：对三个 pinned commit 做差异审查；判别器公式用固定 tensor 核对，Warmup/GAD 对固定输入比较 loss 与梯度，适配后允许合理浮点误差。
- 小规模 GPU 烟测：SeqKD/Warmup 越过原版第 10 步门控，验证 actor 确实更新；Warmup 判别器也更新；GAD 执行真实 GRPO 更新且 loss/梯度有限。
- 阶段交接：GAD 首步前的 actor/critic 权重与 Warmup 导出产物一致，判别头不被重置；S→S+K 步数、dataloader 和 reference 来源正确。
- 失败路径：采样失败、缺模型、错误 backend、checkpoint 缺分片、导出失败时返回非零；下一阶段不启动。
- 恢复：同阶段中断后恢复，检查 optimizer/scheduler、reference 锚点、global step 和数据顺序，区分权重续训与完整状态续训。
- 正式验收前不以“loss 能下降”替代算法一致性验证；ROUGE-L 仅作为原版诊断，不等同数学正确率。

训练数据源已确定为 DAPO-Math-17k。教师/学生模型路径、数据具体文件或 revision、GPU 数量/显存和正式训练步数尚未指定，示例配置使用必填项并在预检中报错。训练/评测长度是待实测的数学配置起点，不代表现有硬件已验证支持。本次仅修订计划，不安装依赖、不下载数据、不调用教师服务、不启动 GPU 训练。
