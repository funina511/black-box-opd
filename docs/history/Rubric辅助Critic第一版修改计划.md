# Rubric 辅助 Critic 第一版修改计划

日期：2026-10-11  
状态：2026-10-11 第一阶段独立离线模块与 24 题工程闭环完成。B0–B3、保存/重载及真实 CUDA optimizer/RNG 恢复验证通过，17 项 CPU 测试通过。标注 JSON 有效率为 91.4%，低于 98% 门槛，人工审阅待完成，暂不扩大至 512 题。原 GAD 后端、训练配置、源数据和源 checkpoint 保持原状。

实施记录：[rubric/SMOKE_RESULTS.md](../rubric/SMOKE_RESULTS.md)。实际默认冒烟配置为 24 题、16/4/4 划分、1 epoch、单卡 critic；后文较大规模草案不代表已开展正式效果实验。

## 1. 第一版的目标与边界

目标：在现有 GAD critic 上增加固定 rubric 的辅助监督，先通过独立、离线、小规模实验回答以下问题：

1. 数学过程的 rubric 标签是否足够一致、可靠，值得用于训练？
2. 原 critic 的共享骨干加辅助评分头，能否在未参与辅助训练的题目上预测这些标签？
3. 联合训练能否改善原判别头对数学质量的排序，或至少不造成明显退化？
4. 过程 rubric 的价值是否超过“只增加最终答案正确性监督”？
5. 数据标注、长序列训练和模型导出是否在现有资源内可承受？

第一版完成标准是离线验证报告、可复现的代码/数据产物及明确的继续或停止结论。训练失败或没有收益也是有效实验结论，不能为了“通过”而只报告最好的子集。

### 1.1 本版实施范围

- 固定三个评价维度：最终答案正确性、推导有效性、条件完整性。
- 同时标注 teacher 与 student 回答，不默认 teacher 正确。
- 保留现有 teacher > student 的成对判别损失，增加辅助监督。
- 共享 critic 骨干；保留原判别输出，增加三个 rubric 输出。
- 固定轨迹池，离线训练 critic；实验中不更新 actor，也不边采样边调用 judge。
- 保留所有原项目入口；新增独立 `LMOps/rubric/` 模块及统一脚本。
- 独立保存数据、配置、实验与 checkpoint，不覆盖正式 GAD 结果。

### 1.2 明确不包含的工作

- 不用 rubric 替换原判别器或 teacher > student 标签。
- 不改变 SeqKD、Warmup 的 actor 更新和 GAD 的 GRPO/PPO 更新公式。
- 不把 rubric 分数混入 actor reward；不开展完整在线 RL 训练。
- 不生成题目级动态 rubric，不做 evo-rubric，不增加文本生成式 critic。
- 不做逐 token / 逐步骤奖励，不新增过程切分及信用分配算法。
- 不同时修改采样数、响应长度、模型规模等变量来制造表面收益。

“保留原逻辑”在本版的准确含义：原成对损失项保留；新增辅助损失会改变共享骨干的梯度，因此属于算法扩展。关闭功能时必须回到原实现。离线固定轨迹训练也不等价于原在线 GAD 分布。

## 2. 当前代码核查与设计约束

以下来自本次本地核查，不沿用九月旧配置作为当前事实。

| 位置 | 当前情况 | 对本版的影响 |
| --- | --- | --- |
| `LMOps/train.yaml` | 新实验默认 Qwen3-0.6B、DAPO-Math-17k、GAD n=2、Warmup n=1、batch=64、长度 2048/8192 | 小规模实验优先用 0.6B；已有 run 必须读取自身保存的配置 |
| `gad/tools/pipeline.py` | 启动参数指定 `algorithm.adv_estimator=grpo` | 后续在线接入需考虑组内标准化，本版暂不修改 |
| `.backends/gad/verl/trainer/ppo/core_algos.py::compute_discriminator_loss` | `-logsigmoid(teacher_reward-student_reward).mean()` | 原标签体现来源偏好，不保证数学质量偏好 |
| `.backends/gad/verl/workers/critic/dp_critic.py` | 保留最后有效 response token 的分数 | 延续整段评分接口；不误称为过程奖励模型 |
| `.backends/gad/verl/workers/fsdp_workers.py` | `critic_model_config.num_labels = 1` | 多头模型不能直接走现有加载路径 |
| `gad/tools/prepare_gad_checkpoint.py::fix_critic_config` | 交接逻辑把 `num_labels` 固定为 1，并检查原评分头 | 本版必须独立处理多头保存/导出，不原样复用该修复函数 |
| `gad/tools/backend_patches.py` | 现有补丁约定不修改算法，应用到固定阶段后端 | 本版先在独立模块实现，不能把实验算法偷偷加入通用工程补丁 |
| `sample/build_gad_data.py` | 数据已有题目、teacher_response、ground_truth、样本索引 | 复用输入数据契约，标签另存 sidecar |
| `sample/teacher_index.py` | 使用 uid + prompt hash，保存 teacher 采样指纹 | 延续同样的来源追踪方式，回答还须增加 response hash |
| `sample/math_scorer.py` | CARD 对齐的答案评分器；`acc` 是 0/1，`score` 是 ±1 | 复用评价口径；辅助正确性标签不能直接使用 ±1 |

代码和实验目录已有用户修改，本版实施时只做明确的增量改动，不清理或回滚这些修改。实施前重新检查代码版本和目标 checkpoint，记录实际哈希。

## 3. 固定 rubric v1 的定义

所有题目使用同一个版本的维度、描述和分档规则。本版 critic 输入为“题目 + 待评价回答”，不拼接 rubric 文本；固定准则由监督标签体现。未来动态准则的显式输入属于后续版本。

| ID | 维度 | 标签 | 规则 |
| --- | --- | --- | --- |
| `answer_correctness` | 最终答案正确性 | 0 / 1 | 采用现有数学答案评分器的 `acc`，保存提取与解析状态 |
| `reasoning_validity` | 推导有效性 | 0 / 0.5 / 1 | 0：存在实质错误或无依据的关键跳跃；0.5：有可核查的有效进展，但关键论证不完整；1：主要推导有效，足以支撑所给结论 |
| `condition_coverage` | 条件完整性 | 0 / 0.5 / 1 | 0：忽略关键条件且影响结论；0.5：处理部分条件但有实质缺漏；1：必要条件、分支和边界均得到充分处理 |

详细定义写入 `rubric/specs/math_v1.yaml`，并配正例、反例、边界例。

必须规定：

- “部分正确”与“评审无法判断”不同。无法判断记 `null`、`valid=false`，不记 0.5。
- 不需要额外分类讨论的简单题，只要已满足题目条件，可得完整性 1；不能因答案简洁扣分。
- 只有最终答案、缺少题目所需推理的回答，过程分按缺失证据处理；不能凭答案正确推定过程正确。
- 必要的常规代数省略可以接受，不要求逐行复述教师解法。
- 不奖励篇幅、特定格式词、教师措辞相似度或“看上去很有推理”的表达。
- 最终答案错但包含有效推导，可以有过程部分分；这不意味着它应优于正确回答。
- 不设拍脑袋的质量总分权重。第一版默认逐维报告，正确性是硬评价依据；过程比较优先在相同正确性层内进行。
- v1 使用完整回答做主要过程标注；截断回答单列，不当作完整证明标注。无法可靠判断的过程维度 mask 掉。

### 3.1 自动正确性标签的边界

现有评分器涉及答案提取格式和窗口限制。标签 0 可能表示答案确实错，也可能是未提取到合法答案。必须保存 `pred`、`parse_status`、`scorer_version`，区分：

- 可解析且匹配：正确；
- 可解析但不匹配：评分器判错，抽查等价表达；
- 没有最终答案：任务输出不满足要求，保留原因；
- 提取失败/表达歧义：保留原始评测结果，但是否进入辅助监督由审计状态决定。

正式评测仍使用原评分器。人工修正辅助标签时独立记录原值、修正值和理由，不能悄悄修改原评测口径。

## 4. 离线数据设计

### 4.1 来源与规模

建议分两级，不直接全量标注 17k：

| 阶段 | 题目数 | 每题候选 | 用途 |
| --- | --- | --- | --- |
| 流程冒烟 | 24 | 已有 teacher 1 条 + student 2 条 | 检查采样、标注、训练、导出闭环，不做效果结论 |
| 离线试验 | 512 | teacher 1 条 + student 2 条 | 约 1536 条唯一回答、1024 个 teacher/student 对；去重/失败后数量另报 |

- teacher 首选当前训练 parquet 中实际使用的回答，和 teacher index 做 hash 校验，不为了得分更高换成别的参考答案。
- student 首选与目标 critic 配对的同一个完整 Warmup checkpoint 的 actor。必须显式固定 checkpoint 路径与步数，不能每次自动取“最新”。
- 若没有对应 0.6B checkpoint，只输出缺失报告；选择其他模型或从头训练是另一项明确实验，不静默替换。
- 采样保持来源 run 的 tokenizer/chat template、thinking 设置、温度与长度上限。512 题默认 student n=2；若来源实验 n 不同，明确记作离线采样设置，不冒称原实验复现。
- 使用 vLLM 独立采样进程；完成后关闭引擎再进行 judge 或 critic 训练，避免三种模型争用显存。
- 不修改共享正式 teacher 数据。离线采样、标签和过滤决定写在新目录。

### 4.2 划分与泄漏控制

- 从训练数据源抽样，按题目分组划分：train 320 / dev 96 / test 96；同题的 teacher 和所有 student 回答必须同分区。
- 固定抽样种子和划分文件；rubric 的措辞先在冒烟集定稿，再锁定 test。
- 题目 uid、规范化 prompt hash 双重检查，重复题/可识别近重复题合并到同一分区。
- 正式数学 dev、quick_dev 及 AIME/MATH500 等外部评测题不进入辅助训练。
- 如果沿用已在全部 DAPO train 上训练过的 Warmup critic，离线 test 只能称为“未参与本次辅助训练的题目”，不能声称模型从未见过。
- 严格未见题泛化需要另设独立数据，或重新训练排除这些题的基线；不纳入本轮最低交付。
- 全部实验组共用同一划分、轨迹和标签。test 只在选择完成后使用一次，不能据此调 lambda 或改 rubric。

### 4.3 保留困难样本，避免挑选容易成功的数据

主集合不根据 teacher 正确性筛选，记录：双方正确、教师对学生错、教师错学生对、双方错误。教师错学生对正是原损失与质量监督可能冲突的样本，不能为了通过实验删除。

若随机集合缺少关键分层，可另建 challenge 集补充这些样本；独立报告，不混入代表性集合的总体指标。过程维度无标签的回答仍可保留原判别训练，其辅助维度使用 mask。

### 4.4 数据文件与字段

```text
LMOps/data/rubric_math_v1/<dataset_id>/
  manifest.json
  splits.json
  trajectories.jsonl
  annotations.raw.jsonl
  annotations.reviewed.jsonl
  pairs.jsonl
  train.parquet
  dev.parquet
  test.parquet
  audit/
```

每条 trajectory 最少包含：

- `schema_version`、`uid`、`prompt_hash`、`response_id`、`response_hash`；
- 原始 prompt/messages、完整 response、ground_truth；
- `source_role`（teacher/student，只用于溯源与组对，不提供给 judge/critic）；
- 模型/checkpoint 指纹、采样 seed/参数、tokenizer 与模板哈希；
- `finish_reason`、`truncated`、原始 token 数与实际模型可见 token 数；
- teacher index 与来源 parquet 的指纹。

每条 annotation 最少包含：

```json
{
  "response_id": "uid:response_hash",
  "rubric_version": "math_v1",
  "rubric_hash": "...",
  "labels": {
    "answer_correctness": {"value": 1, "valid": true, "source": "math_scorer"},
    "reasoning_validity": {"value": 0.5, "valid": true, "source": "judge"},
    "condition_coverage": {"value": null, "valid": false, "source": "judge"}
  },
  "evidence": {"reasoning_validity": "指出对应推导及具体问题"},
  "judge_fingerprint": "...",
  "review_status": "pending"
}
```

`response_id` 实际应使用题目和回答哈希组合生成，示例字符串不作为真实 schema 约束。标签缓存键至少包含题目、回答、ground_truth、rubric、judge、提示词的哈希；不能仅按 uid 复用 student 标签。

`pairs.jsonl` 只存 teacher/student 回答 ID、题目 ID 与权重；标签按唯一回答保存，避免同一 teacher 因 student n=2 被标注两次。

## 5. 标注流程与质量门槛

### 5.1 judge 的职责

- 最终答案正确性主要来自规则评分器；judge 评价两个过程维度。
- judge 看到题目、匿名回答、固定 rubric，必要时看到标准答案；第一版不把完整 teacher 解法作为 student 的唯一评分模板。
- teacher/student 分别评价，使用相同提示词，不把来源、模型名、判别器原分数提供给 judge。
- 返回严格 JSON，包括标签、无法判断标志和简短证据。回答文本作为待评内容处理，不能让其中的指令改变评分规则。
- 同一批标注固定 judge checkpoint/版本、生成参数、上下文预算；禁止静默截断待评价回答。
- 上下文不足则显式记录失败，不能只读前半段仍声称完成整段评价。
- 本版不使用待训练 critic 自己生产监督，避免缺少外部依据的循环标注。

### 5.2 实现与调用方式

提供三种统一输入方式：已有人工/外部标签导入、单独启动的本地 vLLM judge、显式配置的 API judge。默认不填远端地址、不发送任何数据；实现阶段应先核查本地模型资源。

用户已确认 judge 使用本地 Qwen3-30B-A3B-Instruct-2507，GPU 1,2,3,4，TP=4；采样与 critic 训练串行使用 GPU 1。模型权重记录完整内容哈希。先通过下面的小样本审计再决定是否扩大标注。

保留原始输出、解析错误、重试次数及耗时。失败重试有上限，耗尽后记录 unknown，绝不自动补零。标签导入必须验证版本与哈希。

### 5.3 审计安排

1. 24 题冒烟集检查准则歧义、JSON 格式与答案解析。
2. 正式集合中分层抽查至少 100 条回答，覆盖正误、teacher/student、长短和截断类别；人工检查两个过程维度及证据是否成立。
3. 对约 10% 回答重复评审，报告逐维完全一致率、相差一个档/两个档的比例与覆盖率。
4. 初始工程门槛：结构化输出有效率至少 98%，人工认可的过程标签比例至少 85%，人工认为严重反向评分的比例不超过 5%。这些是本项目建议值，应在正式 test 前锁定，不能视为通用学术标准。
5. 若没有足够人工复核，自动一致性只能作为初筛，报告标明“标签质量未获人工验证”，不宣布质量门槛通过。
6. 缺失率、标签分布严重失衡、过程标签与答案正确性近乎完全重复时先修订标注，而不是继续扩大训练。

rubric 提示词/定义一旦变化，生成新版本与新标签缓存；不能覆盖旧实验的标签。

## 6. Critic 模型设计

### 6.1 保留原判别通道，新增三个辅助通道

```text
题目 + 回答
     ↓
共享 critic Transformer
     ↓ 最后一个有效 response token
评分投影：1 个原判别 logit + 3 个 rubric logits
     ├─ adv：原 teacher/student 排序损失
     ├─ answer_correctness
     ├─ reasoning_validity
     └─ condition_coverage
```

工程上优先在已核实的 Qwen TokenClassification 分支，将原输出投影的 1 行扩为 4 行；语义上是原判别头加三个辅助头，不需要四套骨干。首版不承诺支持所有 TRL/value-head/multimodal 分支，不支持时显式报错。

加载流程：

1. 先按原结构严格加载有效的单头 critic，包括已训练的 `score.weight`。
2. 新建四通道投影；第 0 行精确复制原 weight/bias；新增三行按固定 seed 初始化。
3. 更新 `num_labels`、`id2label`、`label2id`、head layout 与 rubric 元数据。
4. 在构建 optimizer/FSDP 之前完成变换，让所有新增参数正确注册。
5. 用相同输入、eval 模式验证扩展前后第 0 通道一致。

禁止直接使用 `ignore_mismatched_sizes=True` 加载四通道模型来代替上述迁移，它可能重置整块评分头。

### 6.2 输出与输入契约

- `adv_score` 保持原来不受限的标量，不加 sigmoid，不重新定标。
- 辅助 logits 使用 sigmoid 映射到 [0,1]；这里只是分档目标的预测值，未经校准不能称为数学正确的可靠概率。
- 输入不含来源角色、标准答案、judge 评语或 rubric 标签；普通数学题题干内自然出现的数值不属于泄漏。
- 保持原 chat template、非 thinking/thinking 设置、EOS 行为及 token 切片规则。
- 末 token 位置由实际 response mask 决定；覆盖右 padding、左 padding、EOS、空响应，不能仅假设 `长度-1` 总是正确。
- 空响应不可访问 -1 位置当作有效 token；数据阶段隔离或按明确规则处理。
- 长度不足以容纳完整回答时显式报错/隔离，不静默把整段标签附到截断输入。
- 先支持普通 padding 路径，再验证 remove-padding 的数值与梯度一致性；正式实验各组用相同的已验证路径。
- 不开启所有层 `output_hidden_states=True` 来获取最后层，避免保存无用大张量；优先复用 TokenClassification 输出投影。

## 7. 辅助损失、权重和重复样本处理

### 7.1 原判别损失保持不变

对每个原 teacher/student 对：

\[
L_{adv}=-\frac{1}{|P|}\sum_{(t,s)\in P}\log\sigma(d_t-d_s)
\]

teacher 错、student 对时仍保留原标签，同时记录该样本属于监督冲突组。本版不反转标签、不按质量过滤成对损失。

### 7.2 Rubric 损失

对三个维度使用带缺失 mask 的 `BCEWithLogitsLoss`，过程维度的 0.5 作为软标签：

\[
L_k=\frac{\sum_{i\in U}m_{ik}\,\mathrm{BCEWithLogits}(z_{ik},y_{ik})}
{\sum_{i\in U}m_{ik}},\qquad
L_{rubric}=\frac{1}{|K_{valid}|}\sum_{k\in K_{valid}}L_k
\]

- U 是当前逻辑优化批内的唯一回答，包含 teacher 和 student。
- 某维度没有有效标签时，该维度不参加平均；所有辅助维度无标签则辅助损失为零，并记录覆盖率。
- 第一版等权平均有效维度，不做类别重加权、不加置信度权重，避免同时引入太多变量。
- 这是一种有界分数拟合方案，不声称 0/0.5/1 是三分类概率分布；若未来改成有序分类，单独做消融。

联合目标：

\[
L=L_{adv}+\lambda L_{rubric}
\]

首轮候选 `lambda=0.1`；dev 显示信号不足时再试 `0.3`。两者均为试验起点，不是已验证最优值。记录两类损失与梯度规模，不能只根据损失数值相近决定权重。

### 7.3 Teacher 重复和梯度累积

每题有两个 student 时，同一 teacher 在原判别对中参与两次，符合成对平均；但 rubric 不应被当作两个不同 teacher 样本。

训练加载器以完整题目组构建逻辑 batch：每题 teacher 1 条、student 2 条，计算全部原始成对损失，辅助监督按唯一回答计数。若采用按 pair 的 forward 复用实现，teacher 辅助项必须按在本逻辑 batch 内的出现次数取倒数加权；分子与分母都使用相同去重权重。

关键约束：

- 归一化分母在整个 optimizer step 的逻辑 batch 上计算，不能对每个 microbatch 单独平均再简单平均。
- 不按 token 数归一化 rubric；不同长度回答仍是整段标签。
- 一个题目组跨 microbatch 可以，但不能无说明地跨逻辑优化 batch；不足一个完整组时不得静默改变权重。
- 分布式方案在所有 rank 汇总计数，并考虑 DDP/FSDP 梯度平均因子；验证与单进程整批结果一致，不能只 all-reduce 日志分母。
- `lambda=0` 基线仍复用相同轨迹、pair 列表、shuffle 和批划分。

原判别偏好与质量监督可能产生冲突，增加第二个头并不会自动消除冲突。需分别评估原判别通道与 rubric 通道；如果只有辅助头变好，不宣称原 GAD 奖励已经改善。

## 8. 离线训练流程与资源预算

### 8.1 执行顺序

`preflight → collect → annotate → audit → pack → train → evaluate → report`

- `preflight`：检查来源 run/checkpoint、模型头、数据指纹、judge、路径、磁盘、GPU 配置，生成计划 JSON。
- `collect`：复用 teacher，采样 student，保存固定轨迹及完整元数据。
- `annotate`：规则评分 + 固定 judge 或导入标签；只处理未完成且哈希匹配的回答。
- `audit`：生成抽查表和汇总，不把自动检查伪装成人工确认。
- `pack`：锁定 split，验证标签/轨迹对应，输出离线训练集。
- `train`：只训练 critic，actor 保持固定，不运行 vLLM。
- `evaluate`：dev 选型后，在 test 评价原判别头和辅助头。
- `report`：汇总组间结果、置信区间、失败样例、耗时、限制及是否值得进入后续实验。

### 8.2 初始超参数建议

- 模型：优先已有 0.6B Warmup critic；模型来源明确固定。
- optimizer：尽量沿用原 critic 的 optimizer 类型、betas、weight decay 和 grad clip，保存 resolved 配置；初始 lr 可从 1e-6 起。
- 初始化新离线 optimizer，不恢复原 Warmup 的 optimizer/scheduler/global step；这是派生实验，不是同阶段断点续训。
- microbatch 初始 1；通过梯度累积形成逻辑 batch，例如 16 题组。这个离线 batch 不等于正式 GAD batch64，所有对照组一致即可。
- epoch 候选 1–3，根据 dev 与过拟合趋势选择，不承诺固定步数即可收敛。
- response 长度保持来源上限 8192；记录实际长度、峰值显存、回答吞吐、每步耗时。
- 优先做单卡 0.6B 冒烟，结合 BF16 与 gradient checkpointing；若长样本/优化器状态超显存，明确升级为多卡 FSDP 或 offload，不静默缩短回答。
- 不预计固定分钟数：先测 5–10 个包含最长样本的优化步，再用实际 token 分布估算预算。
- 不默认占用所有空闲 GPU；在配置中明确资源，与正在运行的正式任务隔离。

### 8.3 断点与重复运行

采样/标注按哈希缓存，失败可重试；训练 checkpoint 保存 optimizer、scheduler、RNG、数据顺序与步数。训练状态恢复与“从已有 critic 开新实验”使用不同参数，禁止混淆。

`--stage all` 可以串行执行机器步骤，但若人工审计门槛尚未满足，只完成审计材料并记录 pending，不伪造审核通过。也可运行标为 exploratory 的自动标注实验，结论必须注明未通过人工标注验证。

## 9. 实验分组与评价

### 9.1 必做实验

| 组 | 初始化 | 训练目标 | 目的 |
| --- | --- | --- | --- |
| B0 | 原 Warmup critic | 不再训练 | 原始 checkpoint 参照 |
| B1 | 同一 critic，统一扩头 | 仅 L_adv，lambda=0 | 排除新增离线训练本身带来的收益 |
| B2 | 同一 critic，统一扩头 | L_adv + 正确性辅助损失 | 只增加答案监督的对照 |
| B3 | 同一 critic，统一扩头 | L_adv + 三维 rubric 辅助损失 | 第一版主体 |

B1 未训练的辅助头分数不作为有意义的 baseline。可选增加“冻结骨干、只训练 rubric 头”的线性探针，用于区分标签可读性和共享表征变化；不纳入首轮必跑。

B2、B3 的辅助项均使用均值约定；报告有效标签数量和各项梯度规模，不能假设相同 lambda 就等价于相同优化强度。

首轮固定一个 seed，B3 从 lambda=0.1 开始；只有 dev 支持继续时再扩大少量候选。最终选定设置对 B1/B2/B3 使用至少三个训练 seed 做确认，轨迹与标签保持固定。这只能估计训练随机性，不能覆盖采样与 judge 的全部不确定性。

### 9.2 标签与评分指标

1. 原判别任务：teacher > student 的 pair accuracy、原判别损失、分数分布。
2. 原判别头的数学质量：同题正确/错误回答排序准确率、正确性 AUROC（两类都存在时）、教师错学生对的特殊分层。
3. 辅助头：逐维 MAE、正确性 AUROC/Brier、过程标签分档一致性；秩相关只在有足够变化的子集中解释。
4. 过程增益：相同最终正确性条件下，能否区分过程优劣；避免仅学到“答案对则过程全部满分”。
5. Best-of-2：在两条固定 student 回答中，用原判别头选出的答案准确率。Rubric 头选择只作为额外诊断，分开列名，不能冒称正式 GAD reward。
6. 固定采样池的随机选择/首条选择与 oracle 上限，识别是否根本没有候选提升空间。
7. 统计按题目 bootstrap，不能把同题多个回答当作独立题目。报告分母、缺失率和 95% 区间；小样本宽区间必须如实呈现。
8. 长度、teacher/student 来源、正误、截断与难度分层，检查是否只学到长度或来源特征。

过程参考标签由同一 judge 生成时，模型与其一致只代表拟合 judge；人工审计子集必须单列。teacher/student 风格可成为来源捷径，不能仅依据 d_acc 提升判断数学能力提升。

### 9.3 验收门槛

工程门槛（必须满足）：

- 功能关闭不影响原训练；原单头权重精确迁移，多头保存后可一致重载。
- 无 train/dev/test 按题目泄漏、标签错位、失败当零、teacher 辅助重复计权。
- 缺失维度、microbatch、不同 rank 标签覆盖下的 loss/梯度缩放通过测试。
- 最长样本可以完成前向、反向、保存与恢复，无 NaN/Inf。

研究门槛（用于是否继续，不承诺必过）：

- 标注质量达到第 5 节预注册门槛，或者明确标为探索性结果。
- B3 在过程质量预测上有可解释的增益，并对 B2 有额外价值；不能只优于完全未训练的辅助头。
- 原判别头正确性排序或 best-of-2 至少有一致改善趋势；若跨 seed/题目区间不明确，结论为证据不足，应扩大独立验证，而非宣布成功。
- 可先预注册原判别任务准确率下降超过 3 个百分点作为退化调查阈值；阈值是诊断线，不要求保留错误的 teacher 优先行为。不以提升来源识别率替代质量提升。
- 如果辅助头提高、原判别头不变/退化，结论为“rubric 可学习，但辅助训练尚未改善原 GAD 评分”，后续再研究奖励组合或替换，符合版本边界。
- 不以 critic 离线指标推断 actor 数学准确率已经提高；后者需要后续独立在线训练实验。

## 10. 目录与逐文件修改范围

建议新建独立模块，避免把标注、模型训练和数据生产混在 sample 目录。

```text
LMOps/
  rubric/
    README.md
    configs/offline_v1.yaml
    specs/math_v1.yaml
    prompts/judge_math_v1.txt
    schema.py
    data.py
    collect.py
    annotate.py
    audit.py
    model.py
    losses.py
    train_offline.py
    evaluate.py
    pipeline.py
    tests/
  scripts/rubric.sh
  data/rubric_math_v1/<dataset_id>/
  runs/rubric/<run_id>/
  docs/Rubric辅助Critic第一版修改计划.md
```

| 文件/模块 | 计划内容 | 验证重点 |
| --- | --- | --- |
| `rubric/schema.py` | 轨迹、标签、pair、manifest 契约 | hash、枚举、null、重复和缺失检查 |
| `rubric/data.py` | 分组划分、配对、mask、逻辑批 | 题目隔离、teacher 去重权重 |
| `rubric/collect.py` | 读取原数据/teacher index，采样固定 student | 模板、checkpoint、长度与来源一致 |
| `rubric/annotate.py` | 数学评分、judge 适配、标签导入、缓存 | 解析失败与零分区分，匿名输入 |
| `rubric/audit.py` | 分层抽样、人工审阅导入、质量报告 | 不生成伪人工通过状态 |
| `rubric/model.py` | 单头严格加载、四通道迁移、末 token 取分 | 原权重保留与空/长响应 |
| `rubric/losses.py` | 原成对公式与辅助 BCE、逻辑批分母 | 与原公式/整批梯度对齐 |
| `rubric/train_offline.py` | 固定轨迹训练、恢复、资源日志 | 不更新 actor、不在线打分 |
| `rubric/evaluate.py` | B0/B1/B2/B3、分层、bootstrap | 质量与来源指标分离 |
| `rubric/pipeline.py`、`scripts/rubric.sh` | preflight、stage、dry-run、缓存与总结 | 任意 CWD 可启动、失败即停 |
| `LMOps/README.md` | 新增实验入口链接 | 标为独立离线实验 |
| 现有 `sample` 工具 | 尽量调用其引擎和数据契约；必要时仅提取可复用公共函数 | 原采样测试保持通过 |
| 现有 `gad` 训练后端 | 第一版不直接改动；必要模型导出兼容用新模块适配 | 不往运行中的 `.backends` 热打补丁 |

第一版主配置放在 `rubric/configs/offline_v1.yaml`。不为了离线实验在全局 `train.yaml` 中塞入未接通的在线 rubric 开关，也不让 `scripts/train.sh` 接收参数却实际不生效。

## 11. 配置草案与拟定命令

以下保留原设计草案。实际入口已经实现，完整可运行参数及默认 24 题、一 epoch 配置见 rubric/README.md。

```yaml
schema_version: rubric.offline.v1
run:
  name: auto
  seed: 42
  output_root: ../../runs/rubric
source:
  run_config: null                # 显式指定来源 run 的 train.yaml
  actor_checkpoint: null          # 同一 Warmup 步的 actor
  critic_checkpoint: null         # 同一 Warmup 步的 critic；缺少原评分头则失败
  dataset_manifest: null
rubric:
  spec: ../specs/math_v1.yaml
  dimensions: [answer_correctness, reasoning_validity, condition_coverage]
  input_mode: implicit_fixed
  unknown_policy: mask
sampling:
  prompt_count: 512
  teacher_n: 1
  student_n: 2
  inherit_source_generation_settings: true
split:
  unit: prompt
  train: 320
  dev: 96
  test: 96
judge:
  backend: import                # import / vllm / api；默认不调用外部服务
  annotations_path: null
  model: null
  prompt: ../prompts/judge_math_v1.txt
  temperature: 0
  max_retries: 2
model:
  layout: [adv, answer_correctness, reasoning_validity, condition_coverage]
  init: preserve_adv_expand_head
training:
  mode: joint                    # adv_only / correctness_aux / joint
  rubric_lambda: 0.1
  logical_batch_prompts: 16
  micro_batch_responses: 1
  lr: 1.0e-6
  max_epochs: 3
  update_actor: false
  reward_mode: original_adv       # 仅声明边界，不会运行在线 actor 更新
resources:
  gpus: null                     # 实施时明确指定
  strategy: single_gpu           # 不足时显式选择已验证的 fsdp 方案
```

- 相对路径统一相对当前配置文件解析，保存绝对路径 resolved 快照。
- 必填 null 在相应阶段 preflight 中报出，不自动寻找任意 checkpoint/judge。
- 每个 run 保存 `config.resolved.yaml`、数据/代码/模型/rubric 哈希与 head layout。
- 效果实验的配置只能覆盖离线参数，不修改来源 run 配置。

拟定命令：

```bash
cd /mnt/jinbo/RLRM/LMOps
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage preflight --dry-run
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage collect
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage annotate
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage audit
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage pack
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage train --variant adv_only
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage train --variant correctness_aux
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage train --variant joint
bash scripts/rubric.sh --config rubric/configs/offline_v1.yaml --stage report
```

训练命令创建独立 variant 子目录；report 在相应评价已完成时汇总，否则明确指出缺失阶段。执行器另提供独立 evaluate 与 all 阶段，不在 report 时隐式启动 GPU 工作。

## 12. Checkpoint 与后续 GAD 的边界

离线 checkpoint 保存：骨干、四通道评分头、HF 配置、rubric 版本、optimizer/scheduler/RNG、数据指纹、step 与 seed。保证关闭训练进程后能够独立重新加载评分，不依赖内存中的猴子补丁。

源 Warmup 的原评分头必须存在且形状正确；不能把缺少评分头的学生 CausalLM 当成已训练 critic。扩头是新实验初始化，原 optimizer 的一维评分头状态不能直接恢复到四维参数。

现有 `prepare_gad_checkpoint.py` 与 worker 固定单头；第一版多头 checkpoint 禁止直接送入原 GAD 流程。

### 12.1 用户确认的后续方向（仅记录，不制定正式接入计划）

2026-10-11 用户确认：当前先完成离线小规模验证，正式 GAD 的改造方案等验证结束后再讨论。这里不预先指定 Warmup/GAD 的具体接入、更新顺序、损失混合或 buffer 参数。

- 正式接入时必须保留在线标注选项，评价当前学生新生成的回答。
- Replay 方向考虑随训练更新的历史轨迹池，保存过去若干步骤中题目匹配的 teacher/student 回答，用于后续 critic 更新。
- 该历史池不同于本版预先构建并固定的离线验证数据集，也不把“固定离线标注池长期回放”作为默认正式方案。
- 在线标注决定监督从哪里来，历史 replay 决定哪些轨迹参与 critic 更新，两者可以组合；具体设置留待离线验证后确定。
- 历史回答若参与 rubric 辅助损失，仍需要与该回答及 rubric 版本匹配的有效监督；不能把旧回答标签按 uid 套在新回答上。标签保存/复用/重评策略暂不设计。

参考用户提供的 [Persistent Negatives for Adversarial Black-Box On-Policy Distillation](https://arxiv.org/html/2609.30864v1)：论文使用有界、持续刷新的题目匹配 teacher/student 比较池，以历史比较替换部分当前判别训练样本；actor 的 GRPO 更新仍使用当前采样。这里借鉴的是历史轨迹训练 critic 的思路，不把论文描述成 rubric 方法。

本版 critic 在离线训练中更新，评测时不更新；这不意味着后续正式训练将永久冻结 critic。奖励组合、替换判别器、历史 replay 实现和 evo-rubric 均不纳入当前第一步交付。

## 13. 测试计划

### 13.1 CPU 测试

- 数据：同题跨 split 被拒绝；prompt/response/rubric 哈希变化使缓存失效；teacher/student 匿名化；重复 teacher 标签去重。
- 标签：unknown 不转零；0.5 为有效软标签；格式失败、评分器 ±1 与 acc0/1 正确区分；伪造 evidence 不自动视为人工审阅。
- 损失：原判别公式对齐；缺失维度；全缺失 batch；teacher 重复次数；尾批；microbatch1/2 与整批梯度相同。
- 头迁移：第0行权重与初始输出保留；新增参数进入 optimizer；lambda0 不向辅助头产生损失梯度；不要把 AdamW weight decay 的变化误判为损失梯度。
- checkpoint：四通道保存重载；head layout/rubric 版本冲突拒绝；缺失原评分头拒绝；错误地加载为 num_labels1 被拒绝。
- 末 token：左/右 padding、EOS、空 response、不同长度；padding token 不参与评价。
- CLI：dry-run 不初始化 GPU/调用 judge；路径相对配置解析；缓存重复执行不损坏；失败传播到退出码。

### 13.2 GPU 验证

- 24 题冒烟闭环，包括长回答前向/反向和实际模型保存重载。
- 单头扩头前后的原分数一致性；普通 padding 与 remove-padding 路径一致性。
- BF16 条件下记录允许的数值误差，不要求所有随机算子逐位一致。
- 若使用分布式，增加不同 rank 有效标签数量不同/某 rank 全缺失的梯度测试。
- 一个进程结束后新进程恢复训练，确认 optimizer、scheduler 和数据进度正常。

### 13.3 原项目回归

如只新增独立模块，运行新测试及所复用的模型/评分器/数据契约测试。若改动共享采样、导出或后端准备工具，再运行对应现有回归套件，包括 `test_algorithm_parity.py`、`test_handoff_export.py` 及原采样测试；不要仅凭新脚本跑通就宣称原流程兼容。

本次只撰写文档，以上测试尚未执行。

## 14. 实施顺序、阶段产物和停止条件

| 阶段 | 工作 | 交付 | 进入下一阶段的条件 |
| --- | --- | --- | --- |
| P0 | 锁定来源 run/checkpoint、rubric 定义和数据契约 | schema、spec、preflight、配置 | 模型头与数据来源完整 |
| P1 | 构建24题轨迹、标注与审计工具 | 冒烟数据、原始 judge 输出、审核表 | 规则明确，标签可解析且证据可信 |
| P2 | 实现多头迁移、损失、离线训练、保存恢复 | CPU测试、GPU闭环报告 | 原头保留，梯度/权重/恢复正确 |
| P3 | 固定512题数据与划分，完成标签质量检查 | 带版本的数据与审计报告 | 达到标注门槛或标明探索性限制 |
| P4 | B0/B1/B2/B3 单 seed + dev 选型 | 分层指标、失败案例、耗时 | 有必要再确认的信号，且无工程错误 |
| P5 | 固定设置多 seed，最终 test，一次性总结 | 完整报告、模型、复现命令 | 给出继续/调整/停止的明确结论 |

发现格式偏差先修数据；发现原头丢失先修迁移；发现损失冲突先分析分层和 lambda；发现只学到答案正确性则不宣称过程 rubric 有效。不能用删掉困难样本、换 test 或改最终评测标准来“补救”结果。

最终报告必须包含：

- 源码/配置/数据/rubric/judge/模型指纹；
- 标签审计与缺失覆盖率；
- B0/B1/B2/B3 的逐维、排序、best-of-2 与分层结果；
- 正误和 teacher/student 冲突样例；
- 不同 seed 与题目区间、实际资源开销；
- 原 GAD 奖励是否有改善证据；
- 当前限制：离线固定分布、judge 标签误差、上游题目暴露、尚未验证 actor 提升；
- 后续建议：维持辅助训练、研究奖励组合/替换，或暂缓 evo-rubric。

第一版要交付的是可信的验证基础，而不是提前证明 rubric 一定优于原判别器。
