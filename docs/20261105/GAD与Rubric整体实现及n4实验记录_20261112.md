# GAD 与 Rubric 整体实现及 n=4 实验记录

记录日期：2026-11-12。用途：说明目前的方法、代码实际行为和实验进展，方便汇报、讨论及后续迁移。

本文依据当前源码、启动配置、冻结运行配置及已有报告整理。**实验进度快照为北京时间 2026-11-12 10:50 左右**，运行中的步数之后会继续变化。本文不修改训练代码、配置或已有实验，不启动任何测试。

## 1. 我们现在到底在研究什么

原版 GAD 希望让学生生成的回答越来越接近 Teacher 的回答分布。为此，它训练一个 Critic，区分 Teacher 与 Student 的回答，再让 Actor 学会生成 Critic 更认可的回答。

问题是：“像 Teacher”与“数学上做得好”不完全相同。Teacher 也可能算错，学生也可能偶尔比 Teacher 做得对；只看来源，Critic 还可能学到长度、表达方式等容易区分的特征。

我们的第一步是保留原来的对抗训练，同时教 Critic 额外判断三个方面：**最终答案是否正确、推理是否有实际进展、是否存在未解决的错误。** 第二步才是让这些预测分数直接参与 Actor 的奖励。

当前有三种主要方法，不能混为一谈：

| 方法 | Critic 学什么 | Actor 用什么奖励 |
|---|---|---|
| GAD baseline | 原来的 Teacher/Student 对抗排序 | 对抗分 `adv` |
| naive-rubric | 对抗排序 + 三维 rubric 辅助监督 | 仍然只用 `adv` |
| actor-rubric | 与 naive 相同的 Critic 训练目标 | `adv` 与 Critic 预测的 rubric 分数结合 |

**naive 不代表 rubric 没有影响 Actor。** 辅助训练会改变共享的 Critic，进而可能改变 `adv`；只是 Actor 的奖励公式没有直接加上另外三个头的分数。actor-rubric 则增加了直接使用三个预测分数的路径。

目前使用的是固定数学 rubric，独立 Judge 负责标注。尚未让 Critic 自己生成、演化 rubric，也没有让 Judge 在训练中被更新。

## 2. 整体流程：数据、三个训练阶段和评测

### 2.1 数据从哪里来

当前 n=4 使用 `data/dapo_math_17k_gad_v2/`，训练文件记录为17,405条，开发集512条。这里不仅有 DAPO 数学题，还包含此前生成、选定并整理好的 Teacher 回答，以及答案、索引和数据来源信息。

Teacher 在正式 GAD 训练中主要以已有回答数据的形式出现，不是每一步重新部署一个 Teacher 模型生成答案。Student 则由当前 Actor 在线采样。

`sample/` 负责准备题目、调用采样后端、保存 Teacher 回答和构建训练数据；`sample/teacher_engines.py` 等实现本地引擎及 API 采样路径。新造的数据是新的数据版本，不能直接当成现有实验相同的 Teacher 数据。

当前正式设置使用完整17k版本，**没有采用“只保留正确且未截断回答”的筛选版本**。训练输出长度上限保持8192。Teacher 原始生成是否截断、进入训练时是否又被裁剪，需要分别记录；它们不是同一件事。

### 2.2 SeqKD、Warmup、GAD 各做什么

| 阶段 | Actor | Critic | rubric 的位置 |
|---|---|---|---|
| SeqKD | 学习已有 Teacher 回答，作为独立蒸馏基线 | 不承担本文的联合 rubric 训练目标 | 不使用本文 rubric 辅助训练 |
| Warmup | 按原 Warmup 逻辑学习 Teacher 回答；保留原前期更新门槛 | 适应 Teacher/Student 分布，学习对抗排序 | rubric 方法从这里开始训练三个辅助头 |
| GAD | 用当前采样回答和 GRPO/PPO 风格目标更新策略 | 持续进行对抗训练，rubric 方法同时训练辅助头 | naive 间接影响奖励；actor-rubric 还直接混合预测分数 |

Warmup 从初始学生开始，**不接 SeqKD 的训练结果**。GAD 接 Warmup 的 Actor 与 Critic 权重。初次阶段交接和同阶段恢复不同：前者通过已有 HF 导出交接、重新初始化 GAD 优化器；后者恢复 GAD 自己保存的模型、优化器、调度器及随机状态等。

原实现前期存在暂不更新 Actor 的门槛，不能用刚启动的两三步断言 Actor 已更新。SeqKD/Warmup 的配置采样数为1，GAD n=4；“SeqKD 是监督学习”也不等于其工程路径完全没有生成操作。

### 2.3 评测与训练标签分开

正式对外比较使用统一八任务：AIME24、AIME25、AMC23、MATH-500、Minerva、Olympiad-Bench、GPQA、MMLU-Pro-Math。GPQA 是跨领域补充，不是纯数学任务。

目前沿用的正式比较口径为16k输出预算：`max_tokens=16384`、`max_model_len=18432`；AIME24/AIME25/AMC23/GPQA 每题8次，MATH-500/Olympiad-Bench/MMLU-Pro-Math 每题2次，Minerva 每题4次。实际比较以各 run 冻结的评测配置为准。

训练内 dev 验证不等于上述完整八项评测。`COMPLETED.json` 中的 `evaluation_status=complete` 也不能单凭字段名解释为八项测试全部完成，要看它记录的是哪一种验证。

ROUGE-L 已关闭，不作为本文 rubric 标签或 Actor 奖励。数学正确性来自固定的答案提取与数学评分器；评测侧的未决状态处理与训练标签映射要分别看待，不混用成两个不同的“数学真值”。

## 3. Rubric 评分：三个指标，两个标签来源

### 3.1 指标含义

| 符号 | 代码字段 | 标签来源 | 含义 |
|---|---|---|---|
| `c` | `answer_correctness` | 本地数学 scorer | 答案正确为1，明确错误或没有答案为0；无法可靠评分时无效 |
| `p` | `verified_progress` | Qwen3-8B Judge | 0：无可确认实质进展；0.5：有独立正确中间结果；1：可见解答已充分解决题目 |
| `e` | `unresolved_error_severity` | Qwen3-8B Judge | 0：未发现明确未解决错误；0.5：局部错误；1：破坏关键路线或结论的错误 |

`p`、`e` 可以为 `null`，表示无法判断。**`null` 不等于0。** 前者不参与该维监督；后者是一个确切标签。

`e` 越大代表错误越严重，因此转成“越高越好”的奖励时要减去它。Critic 学习 `e` 时仍应原样预测错误程度，不把标签先反转。

三个维度独立：答案对不代表过程全对；答案错也不代表没有正确进展。对 Teacher 回答同样进行评分，不因其身份就给 `c=1` 或 `p=1`。

### 3.2 当前 Judge 的输入输出

当前协议为 `direct_v1`，数学判据位于：

- [固定 rubric spec](../../rubric/specs/math_process_direct_v1.yaml)
- [Judge 提示词](../../rubric/prompts/judge_math_process_direct_v1.txt)
- [协议解析](../../rubric/core/process_direct.py)

每次请求评价**一条**可见回答，包含题目、回答、参考最终答案和可见性信息。Judge 不需要把 Teacher 与 Student 两条长轨迹一起比较，也不生成新的 rubric。

Judge 开启 thinking，在内部检查数学内容，最终只返回：

```json
{"verified_progress": 0.5, "unresolved_error_severity": 0}
```

没有长证据列表、步骤编号、引用片段等必填输出，也不沿用旧事件协议的语义复核链。保留格式和取值检查：两个字段只能为0、0.5、1或null。这样减少输出与解析负担，但**格式合法不等于数学判断正确**。

当前配置中 Judge 为 Qwen3-8B、TP=2、并发8、上下文预算32768、输出预算8192。这里的输出预算包括 Judge 自己的思考开销，不是 Student 的8192回答上限；短 JSON 也不代表请求很快。

`max_retries=0` 不表示整个系统永远只发一次请求：还存在请求账本、恢复与 annotation attempt 上限；例如预取失败后的主流程补尝试受共同上限约束。记录成本应看实际发送数，而不是简单用选中回答数代替。

### 3.3 截断回答如何处理

现在的原则是评价**已经看得到的数学内容**，不因未完成就清空所有过程监督。

例如，学生已正确推出一个关键中间结果，但8192长度到顶，尚未写出答案，可以得到 `c=0, p=0.5, e=0`。如果前面有独立成立的进展，后面又引入严重错误，可以得到 `p=0.5, e=1`；二者并不冲突。

纯复述题目、没有实际数学工作时，通常是 `p=0, e=null`。Judge 不能把自己补出的推理算到学生的进展里，也不能凭自己修复了错误就判学生已经修复。

正确性标签遵循 `scoring/math.py::correctness_label`：

- 题目本身被截断或不可用：正确性监督屏蔽。
- 没有提取到答案：有效标签0。
- grader 明确判对/判错：有效标签1/0。
- 解析或判分未决：屏蔽，不强行归零。

回答因长度结束不直接覆盖上述答案判断；日志里的 T 类也只是终止类型，不保证其中每条在 scorer 上都是0。

### 3.4 并不是每条回答都调用 Judge

为控制成本，当前配置按全局逻辑 minibatch 最多选择32个不同的监督身份做过程标注。GAD 满额且角色无重合时，Teacher11、Student21；Warmup 按其既有角色规则分配，32名额时通常各16。

候选按 seed、阶段、绝对步数和回答身份做确定性选择。角色重合、候选不足和去重会影响实际数量。**这不是按正确/错误/截断类别强制分层，也不是保证32次 API 请求。** 已缓存标签可以复用；请求失败或 `null` 又会减少有效监督。

在 n=4、batch64 下，一步有256条 Student 回答，但通常只有约21条被选中做新的过程监督；正确性标签则不受这个过程名额限制。必须分别报告：全体覆盖率、选中回答的有效率、请求成功率，不能只给一个“覆盖率”。

少量过程标签能训练共享 Critic，Critic 随后可以为其他回答预测分数。这是一种泛化假设，不是未标注回答已经得到真实 Judge 标签。

## 4. Critic：四个输出头与联合损失

### 4.1 四头模型的含义

Critic 共享一套 Transformer 主干，末端输出四个通道：

```text
同一条题目与回答 → 共享主干 → [adv, c_logit, p_logit, e_logit]
```

取回答最后一个有效 token 位置的四个值。`adv` 使用原始对抗分；另外三个 logit 经 sigmoid 后得到 `[0,1]` 区间的预测值 `ĉ、p̂、ê`。

这里的 Critic 是判别/评分模型，不是通常 actor-critic 教科书里的状态价值回归器。当前也不是逐个推理步骤、逐 token 都给一个监督标签：**是用包含过程信息的回答级标签，训练回答末端的评分头。**

扩为四头时保留已有 adv 行与主干，初始化额外三个头。辅助损失能更新共享主干，所以会影响对抗头后来学到的表示。

### 4.2 原始 BT loss 保留

对同题一对 Teacher/Student 回答，设对抗分为 `a_T、a_S`：

\[
L_{BT}=\operatorname{mean}_{pairs}\{-\log\sigma(a_T-a_S)\}.
\]

通俗地说：Teacher 分比 Student 高时，损失较小；反过来则损失较大。训练继续推动 Teacher 排在 Student 前面。

这项偏好基于来源，**没有用正确性标签自动交换 Teacher/Student 的位置**。Teacher错、Student对时，对抗排序与正确性辅助监督可能产生不同方向的压力；当前方法让不同头承担不同目标，而不是宣称这种张力已经消失。

### 4.3 三维 rubric 使用 BCE-with-logits

对任一维，模型输出 logit `z`，目标 `y` 可为0、0.5或1：

\[
\ell(z,y)=-y\log\sigma(z)-(1-y)\log(1-\sigma(z)).
\]

虽然名字叫二元交叉熵，BCE 可以接受0.5这类软目标。例如进展标签0.5会推动该头输出接近0.5，而不是把它变成一个单独的三分类类别。`p̂、ê` 因而可理解为标量评分预测，不宜都称为严格校准的事件概率。

当前模式是 `fixed_dimension_gated`，`rubric.loss_coef=0.1`，计数门槛 `count_gate_tau=4`。对维度 `d∈{c,p,e}`：

\[
N_d=\sum_i w_i m_{i,d},\quad
g_d=\min(N_d/4,1),
\]
\[
L_{rubric}=\frac13\sum_{d\in\{c,p,e\}}
g_d\frac{\sum_i w_i m_{i,d}\ell(z_{i,d},y_{i,d})}{\max(N_d,1)},
\]
\[
\boxed{L_{critic}=L_{BT}+0.1L_{rubric}}.
\]

这里 `i` 同时覆盖 Teacher 和 Student，`m` 表示标签有效与否，`w` 是重复监督身份的倒数权重。没有标签的维度贡献0。

三个设计分别解决不同问题：

1. **重复去权重。** n=4 时同一 Teacher 常被配成四对，辅助监督不应被重复算四倍；相同可见监督跨角色、跨卡和梯度累积统一计数。BT 仍按原配对目标训练，不套用同一去重方式。
2. **固定除以3。** 缺失某一维时不把其余维度突然放大。correctness-only 若也使用这套设置，就是只保留正确性那一项，仍除以3；标签足够时其系数相当于 `0.1/3`，不是把全部0.1重新分给它。
3. **小样本降权。** 某一维只有1个有效监督时门控为0.25，达到4个才为1，避免极少数标签支配更新。这个门槛不是 Actor 奖励中的标准差下限，二者不要混淆。

实现按全局逻辑 minibatch 计算分母，再拆分到卡和 microbatch；worker 处理分布式梯度平均的补偿。不能在每个 microbatch 内随意重新平均三个维度，否则换卡数或微批大小可能改变实际辅助权重。

**损失相加是正确的。** 标签“大代表更好”描述的是目标分数方向；loss 描述预测偏差，越小越好。错误头也是用正的 BCE 学习错误程度，只有进入 Actor 奖励时才减去其预测。

## 5. Naive-rubric：先让 Critic 学会更有意义的判断

naive-rubric 的关键是分开“训练 Critic 的监督”和“给 Actor 的奖励”：

```text
scorer 产生 c；Judge 产生 p/e
             ↓
      BT + rubric 辅助损失
             ↓
         更新四头 Critic
             ↓
Actor 的奖励公式仍只读取 adv
```

Warmup 和 GAD 都更新 rubric Critic，不是先离线训练完再永久冻结。Judge 则作为固定标注器，不参与反向传播。

naive 的配置为 `reward_source=adv_only`。n=4 版本额外开启 `actor_reward_shadow=true`：同一次 Critic 前向保存 `adv、ĉ、p̂、ê`，用于之后离线分析“如果加 rubric，会怎样改变奖励”；**影子记录不改变本次 Actor 优势或训练目标**。

因此 naive 回答的是：“增加数学监督之后，原本只依靠对抗分的 GAD 是否会变好？”它尚不能直接证明三个预测分数拿来做 Actor 奖励就一定可靠。

## 6. Actor-rubric：让预测分数直接影响策略更新

### 6.1 Actor 用的是 Critic 预测，不是当步 Judge 原始标签

把三个预测合成一个“越高越好”的分数：

\[
r_i=w_c\hat c_i+w_p\hat p_i-w_e\hat e_i.
\]

完整分量设置是三个权重都为1，`c-only` Actor 消融是 `(1,0,0)`。这里的 `c-only` **仍可使用完整三维训练的 Critic**，只是 Actor 只读取正确性分量；它不同于“Critic 也只学正确性”的 correctness-only 实验。

对所有新生成回答，Critic 一次四头前向就能产生分数，不为每个 Actor 回答额外请求 Judge。Judge 成本主要来自 Critic 的标注任务，而不是奖励相加本身。

### 6.2 已完成的旧 n=2 版：直接相加

旧正式 Actor-rubric 用：

\[
s_i=a_i+\beta(\hat c_i+\hat p_i-\hat e_i),\qquad \beta=0.05.
\]

再由原 GRPO 在同题回答间标准化。n=2 时若两条分数不相等、差值远大于数值 epsilon，优势大致就是 `+0.707、-0.707`。仅改变分差、没有改变排序时，策略获得的相对信号可能变化很小。

旧实验有17,344个同题回答对，严格翻转27对、打破原平局28对，合计约0.32%。这说明直接排序干预稀疏；**不是两次完整训练99.68%相同**，因为随后采样、Critic 和策略都可能逐步分化。

另外，旧纯 adv 路径和混合路径存在 BF16/FP32 奖励精度差异。旧结果不应解释为完全只改变了 rubric 奖励的严格消融。

### 6.3 当前 n=4 版：组内归一化后混合

当前实现先在同一道题的4条回答内处理两种分数：

\[
z_{a,i}=\frac{a_i-\bar a}{\operatorname{std}(a)+10^{-6}},\qquad
z_{r,i}=\frac{r_i-\bar r}{\max(\operatorname{std}(r),\tau_r)},
\]
\[
s_i=z_{a,i}+wz_{r,i}.
\]

随后仍调用原 GRPO：

\[
A_i=\frac{s_i-\bar s}{\operatorname{std}(s)+10^{-6}}.
\]

代码使用样本标准差。`w` 控制 rubric 的相对强度；`τ_r` 防止很小的 rubric 差异被无限放大。若 `w=0` 或该组 rubric 分数完全相同，直接旁路为原始 FP32 adv；若 adv 无差异而 rubric 有差异，rubric 可能成为该组主要学习信号。

这不是简单“把旧 β 调大”。一方面 n 从2增加到4，另一方面奖励混合形式也变了；不能把后续结果全部归功于采样数量。

组内运算在 driver 收齐全部同题回答后进行，不在各卡的 microbatch 内分别计算。同一题的4条回答必须属于同一组，不能因切卡或微批2而拆成两组2条。

三组 n=4 都开启 `actor_reward.fp32=true`，统一奖励/优势计算的入口精度；这不代表整个模型训练都改成FP32。

### 6.4 Actor 的损失仍沿用原策略更新

Actor 使用上面得到的组内优势，通过原有带 clipping 的策略梯度目标更新，并保留 KL loss。可概括为：

\[
L_{actor}\approx-\mathbb E\big[\min(\rho A,\operatorname{clip}(\rho)A)\big]
+\lambda_{KL}L_{KL},
\]

其中 `ρ` 是新旧策略对所采样 token 的概率比；具体 token mask、聚合和 clipping 以固定后端为准。当前 KL 系数0.001，熵系数0。groupnorm 路径关闭的是“把 KL 扣进 reward”，**并没有关闭 Actor 自身的 KL loss**。

序列奖励放在回答最后一个有效位置，GRPO 将回答级优势用于该回答的有效 token。rubric BCE 不直接加到 Actor loss，也不通过奖励反向传播到 Critic。

## 7. 一步 GAD 实际按什么顺序运行

当前顺序如下；部分工作并发，但更新依赖不变：

1. 取64个问题及对应 Teacher 回答；可提前为选中的 Teacher 提交 Judge 请求。
2. Actor 为每题生成4条 Student 回答，共256条，构成256对 Teacher/Student 数据。
3. 抽取实际可见的回答，确定过程标注名额；本地 scorer 在主线程计算，Student Judge 请求可提前在后台提交。
4. 计算 old/ref log-prob，并用**本步 Critic 更新前**的模型计算所有新 Student 回答的分数。
5. 生成奖励和 GRPO 优势。naive 只用 adv；Actor 组使用混合分数。
6. 等待本步所需标注结果，构造 Critic 批次，执行 BT 与辅助损失更新。
7. Actor 使用第5步已计算好的优势更新；不会因为第6步 Critic 更新完又重新打分。
8. 成功完成后提交扩展状态、日志和可选 replay 插入，按保存频率写 checkpoint。

因此“先更新 Critic、再更新 Actor”不意味着 Actor 用的是 Critic 刚学完本步 Judge 标签后的分数。新监督先改变 Critic，影响后续批次的评分；本步优势在 Critic 更新前已经确定。

Teacher 预取和 Student 提前标注的目的是把等待时间与生成、log-prob 和评分重叠，没有改变标签定义，也没有采用“Actor 先更新、Critic 后更新”。所有必需标注在优化器更新前汇合。

单条过程失败可屏蔽对应维度，正确性仍独立处理；但服务不可用、所需标注全部技术失败或学生连续失去有效过程监督等情况会触发已有保护，不能无限悄悄退化为 correctness-only。当前连续空监督阈值为3步。

## 8. Replay buffer：历史回答池，不是 Actor 的旧经验训练池

### 8.1 保存什么、给谁用

当前代码已实现有界 CPU replay，单位是同题的 Teacher/Student 回答对，包含实际 token、可见文本、题目与答案、来源步骤、回答身份、截断信息，以及已有标签等。

它的目的，是让 Critic 除了适应最新学生分布，也能看见近期过去的回答。**Actor 的 batch 仍是当前策略新生成的回答，replay 不替换 Actor 的 rollout 或优势。**

标签缓存与 replay 是不同功能：缓存避免同一可见回答和同一评分协议重复调用 Judge；replay 决定哪些历史回答进入 Critic 训练。历史 pair 可以有标签，也可以待标注；选中后沿当前缓存与标注路径处理，不能认为存入池子就永久有可靠标签。

### 8.2 混合与容量

配置中保留的参数为：

| 参数 | 值/含义 |
|---|---|
| `replay.enabled` | **当前三组 n=4 均为 false** |
| `mix_ratio` | 0.25，历史数据目标占比 |
| `capacity_pairs` | 最多1024对 |
| `max_age_steps` | 超过8个阶段步的条目淘汰 |
| `max_bytes` | 536,870,912字节，即512 MiB序列化内容预算 |
| `sampling` | 无放回均匀采样 |
| `eviction` | 优先淘汰最旧条目 |
| `seed` | 独立 RNG，seed43 |
| `stages` | 当前预设仅 GAD |

若本步256对、历史充足、比例0.25，则 Critic 训练批次由192对当前数据加64对历史数据组成，总量仍是256对，**不是额外再加64对变成320对**。池子不足时历史数量相应减少。

成功步骤提交后，将本步 fresh pairs 放入池；即使其中一部分本步没用于 Critic，也保留为未来候选。容量、年龄和字节上限共同约束，512 MiB并非整个 Python 进程 RSS 的严格上限。

### 8.3 恢复与当前证据

checkpoint 保存池内容、采样 RNG、已提交步和内容校验信息，避免恢复后重放顺序无故改变或重复插入同一步。

当前实现及恢复路径有 CPU 回归覆盖，但本轮 n=4 GPU 短测和正式运行都关闭 replay。**没有证据可以把本轮收益归因于 replay，也没有完成“开启 replay 后的正式性能消融”。** 迁移时继续保留其源码与测试，后续若启用，要作为独立实验变量。

## 9. 为什么开展 n=4 三组对照

旧 n=2 完整评测的八项等权平均正确率下界如下，作为研究动机，不是当前 n=4 结果：

| 旧实验 | 八项平均 |
|---|---:|
| SeqKD | 23.15% |
| GAD baseline | 24.92% |
| correctness-only | 25.77% |
| naive-rubric | 26.70% |
| 旧直接相加 Actor-rubric | 25.47% |

这些数字来自 [上一份两版设计与实验汇报](../Rubric两版设计与近两周实验汇报_20261105.md)。naive 比旧 GAD 高约1.78个百分点，旧 Actor-rubric 比 naive 低约1.23个百分点；历史训练 scorer、奖励精度和单 seed 等限制仍须保留，不能写成已证明的普遍因果结论。下界来自评分未决口径，不是置信区间。

n=4 的考虑是：每题提供更多候选，增加正确、错误和部分进展共存的机会；同时通过统一FP32及新的奖励混合，检验 rubric 是否能提供更有用的相对优势。

三组保持 batch64，因此 n=2→4 后每步学生回答从128增到256，计算成本和样本量也增加。三组 n=4 内部是同一采样数量的比较；与旧 n=2 比较不能称为等计算量实验。

## 10. 当前 n=4 实验配置与真实进度

### 10.1 共同设置

当前源服务器的配置主要为：Qwen3-0.6B 学生/评分主干、完整17k数据、seed42、batch64、GAD n=4、训练输出8192、non-thinking Actor、GAD Actor/Critic 学习率各 `1e-6`。Judge 为独立 Qwen3-8B thinking。

四张3090训练，另外两张3090运行 Judge。Actor/Critic 参数和优化器 offload 关闭，梯度检查点打开，Critic microbatch2，Actor 动态 token 预算10240，rollout utilization0.55、`max_num_seqs=128`、`enforce_eager=false`。128是引擎同时调度上限，不是整步最多只能处理128条回答。

三组均从各自指定的 Warmup271 开始，新增271步 GAD，到 global step542。baseline 用原 baseline Warmup；naive 与 Actor 组按方案共用原 rubric Warmup。**这是本机当前实验安排，新 H200 完整重跑的 Warmup 来源不同，见第12节。**

### 10.2 进度快照

| 实验/验证 | 核查到的状态 | 能说明什么 |
|---|---|---|
| `baseline_n4_06b_17k_v1` | 正式 GAD 已完成到542，完成标记与检查点存在 | baseline 训练完成；阶段 dev 完成不等于八项结果齐全 |
| `rubric_naive_n4_06b_17k_v1` | 已提交272、273两步；报告说明用户已停止 | 属于中止尝试，不当成完整 naive 结果 |
| `speed_naive_20261110_eos` | 2步训练 + 1步恢复，完成到274 | 验证新调度和恢复路径，不是正式质量结果 |
| `rubric_naive_n4_06b_17k_v2` | **仍在运行；最后完整扩展日志407，stage_step136；最近保存400** | 新增271步中约50.2%已提交，尚未训练完成 |
| 正式 Actor n=4 | 预定 `rubric_actor_n4_06b_17k_v1` 目录未发现；现有证据为工程短测 | 需等待完整 naive 记录选参，不能写成已有正式结果 |

进度依据是读取冻结配置、`extension_metrics.jsonl`、检查点指针、完成标记与相关进程。**日志到407与检查点到400不矛盾**：每步提交记录，检查点按保存频率落盘；不能称407已经具备完整磁盘恢复状态。

407步快照中：Student过程选中21条，两维都有效20条，可见标注等待约354.4秒，replay比例0。这只是单步观察，不代表全程平均标签质量或耗时。

11月10日报告写的是“正式 naive v2 未启动”，那是报告生成时的状态；本次运行目录与进程显示它后来已经启动。当前记录以本次快照为准。

baseline 使用后端 `0192d4849cc7`，naive v2 使用提速修复后的 `f0ce22a97d40`。后者涉及采样器恢复、标注调度及 EOS 等执行修复；已有短测核对题序和奖励口径，但不能因此声称两次正式实验的部署代码逐字相同。

### 10.3 工程短测和提速结果

最初三组都做过2+1步检查。Actor 临时参数下，每步384个同题两两配对中严格翻转59、61、75对，三步合计195/1152，约16.9%。这证明奖励可以改变排序，**不证明翻转方向正确，更不证明正式评测提高**。

11月10日的提速测试保留了原 offload/KV 配置和 Judge 并发8：offload加大KV候选仅改善约1%–3%，Judge并发16在固定32条样本上快约5.11%，未达到预设采纳门槛。

Teacher预取和Student提前标注已合入。最终短测每步 Teacher 命中11/11，96次发送94次可解析，失败过程标签屏蔽；三步平均约1174.8秒。与旧 naive 前两步相比，同口径短测下降约5.26%，但历史对照未同日重跑，不能把全部差异归因于调度，也不能承诺整轮同幅度提速。

该短测首两步显存连续采样有缺口，续训步补齐；没有 OOM 不等于三步完整峰值证据齐全。测试报告中的CPU447通过/5可选跳过等，是该次执行记录，不是本文重新跑过的测试。

## 11. 完整 naive 结束后，怎样确定 Actor 奖励

`gad/tools/select_actor_reward.py` 从完整 naive 的影子日志分析候选，不用测试集来挑参数，也不会自动开始训练。

主要规则：

1. 按稳定 question_id 的 SHA256 分成选择集和复核集，同题不跨两个集合；另外单独看 step450–542 的后期表现。
2. 比较 full `(1,1,1)` 与 Actor c-only `(1,0,0)`，各测试 `w∈{0.25,0.5,1,2}`，共8个主候选。
3. 每种分量的 `τ_r` 在观察候选效果前固定为选择集 rubric 组内标准差的25分位数；0和中位数仅做敏感性分析，不临时扩成新的主选参网格。
4. 检查修正错误排序的比例、净修正数、C/W优势间距、T/W偏向及总体排序干预比例。不能只追求翻转更多。

具体约束包括：有定义时修正精度至少80%；C/W净修正不为负；T/W翻转不更多偏向T；原非平局配对的改变比例不超过25%（含变成平局）；总净修正非负且同题C−W优势间距改善。无分母输出N/A，不伪装成0或100%。

C表示正常结束且正确，W表示正常结束且错误，T表示长度截断，U表示其他/未决。统计中“正确与错误”的有效标签配对和“C/W类别配对”并非完全同一集合，读报告时要看分母。

工具在满足选择集规则的候选中推荐，复核集和后期窗口单独报告，不自动把占位参数变成正式参数。当前 Actor 配置的 `w=1、τ_r=0.05` 是工程候选，`actor_reward_selection_confirmed=false` 保留，待完整 naive 分析和用户确认。

离线影子分析只回答“在已有回答上，奖励会怎样变化”，不能模拟 Actor 训练后产生的新分布；最后仍必须通过独立 Actor 正式训练和统一评测判断收益。

## 12. 已确定的边界与迁移衔接

目前可以确认的是：rubric 已从标签进入 Critic 辅助目标；Actor 混合路径已用短测验证接通；当前 naive n=4 正式训练正在推进。尚不能确认的是：完整 n=4 Actor-rubric 是否优于 naive、replay 是否带来额外收益、单卡H200是否提速。

下一阶段 H200 迁移已决定：代码走 GitHub，数据和大依赖经个人电脑中转；模型在目标机下载，重新跑公共 SeqKD、baseline Warmup、rubric Warmup及三组GAD。新机 naive 与 Actor 组共用新 rubric Warmup，不搬本机旧权重。当前本机实验与新机未来实验应分别命名、记录，不能把新机完整重跑写成当前 run 的续训。

具体交付边界见 [H200迁移计划](GAD_n4单卡H200迁移计划.md)。本记录不实施迁移，不改变仍在运行的 naive 实验。

## 13. 代码与证据索引

| 想确认的问题 | 主要入口 |
|---|---|
| 阶段预算、初始化与交接 | [pipeline.py](../../gad/tools/pipeline.py)、[prepare_gad_checkpoint.py](../../gad/tools/prepare_gad_checkpoint.py) |
| 四头初始化 | [prepare_rubric_critic.py](../../gad/tools/prepare_rubric_critic.py)、[heads.py](../../rubric/core/heads.py) |
| BT、BCE、门控公式 | [losses.py](../../rubric/core/losses.py)、[worker.py](../../gad/extensions/worker.py) |
| 回答身份、去重与 Critic 批次 | [critic_batch.py](../../gad/extensions/critic_batch.py) |
| 过程抽样、有效率与标注 | [process_sampling.py](../../gad/extensions/process_sampling.py)、[online_v2.py](../../gad/extensions/online_v2.py)、[annotation_work.py](../../gad/extensions/annotation_work.py) |
| 缓存身份 | [online_rubric.py](../../gad/extensions/online_rubric.py)、[label_cache.py](../../gad/extensions/label_cache.py) |
| Actor 奖励、影子统计 | [actor_reward.py](../../gad/extensions/actor_reward.py) |
| Replay 与步骤提交 | [replay_buffer.py](../../gad/extensions/replay_buffer.py)、[runtime.py](../../gad/extensions/runtime.py) |
| 数学正确性 | [math.py](../../scoring/math.py) |
| 正式选参 | [select_actor_reward.py](../../gad/tools/select_actor_reward.py) |
| 三组配置 | [baseline](../../configs/launch_baseline_n4_06b_17k.yaml)、[naive](../../configs/launch_rubric_naive_n4_06b_17k.yaml)、[Actor](../../configs/launch_rubric_actor_n4_06b_17k.yaml) |
| n=4 实施与短测 | [实施报告](../../reports/gad_n4_20261105/README.md) |
| 调度、续训和提速证据 | [提速报告](../../reports/gad_speed_20261110/README.md) |
| baseline 完成状态 | [完成记录](../../runs/baseline_n4_06b_17k_v1/gad/COMPLETED.json) |
| 正在运行的 naive v2 | [冻结配置](../../runs/rubric_naive_n4_06b_17k_v2/reports/resolved_configs/resolved_config_20261110T101604311686Z.yaml)、[检查点指针](../../runs/rubric_naive_n4_06b_17k_v2/gad/latest_checkpointed_iteration.txt) |

报告和 run 链接是本机证据位置，未必全部随 GitHub 发布；对外交流时本文的快照说明可独立阅读，迁移时按交付清单保留必要来源摘要。


## 2026-11-14 更新：naive完成，开始第三组选参与迁移

此前的运行进度是11月12日快照。naive v2现已正常完成271→542，耗时79.52小时。完整影子日志的固定网格选择只允许Actor-c-only（w=0.25，τ=0.045992180705070496）通过，复核和后期也通过；完整c+p−e未通过。这里仅限制Actor接收的分量，Critic仍学习完整c/p/e辅助目标。80%的方向比例刚好达标，且收益主要涉及截断配对，不能替代正式Actor训练验证。参数尚未写入正式launch，也未开始第三组训练。

代码与中转资产已在独立目录准备，未推送或连接H200。取舍、验证和命令见 [11月14日交付记录](n4选参与迁移交付记录_20261114.md) 与 [完整报告](../../reports/gad_n4_selection_migration_20261114/README.md)。H200仍计划从头训练，使用新Warmup和新naive选参。
