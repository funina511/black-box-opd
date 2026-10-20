# Rubric 过程监督 V2：设计与参考实现计划

> **已由 [GAD_Rubric_V2统一执行计划.md](GAD_Rubric_V2统一执行计划.md) 合并替代。** 本文保留历史设计供追溯；执行时只以合并文档为准。正式 baseline/rubric 重跑由用户负责，执行模型仅负责修复和有上限的小规模验证。

日期：2026-10-14。状态：设计已选定，待实现与校准。

本文是《Rubric评分修复与实验重做计划.md》的过程监督补充方案。用户已选择：采用 SRaR 的正向步骤／错误事件结构，结合数学 RRM 的具体数学判据和不同解法兼容原则。此次仅更新计划，没有修改训练实现。

评分口径已统一：answer_correctness、训练验证、外部评估共用主计划定义的 `math_unified_v1` 公共数学 scorer；正式结果只使用一种正确性标准。旧 CARD/OPRD 指标只用于历史审计，不作为新实验并列评分。

## 1. 目标与算法边界

在 8192-token 回答预算下，即使轨迹没有完成，也识别并保留已经成立的数学进展和已经发生的明确错误。未完成、错误、没有进展、无法判断不能共用一个含义模糊的标签。

本轮仍为 **rubric 辅助训练 critic**：Warmup 和 GAD 均启用；保留原 BT、Warmup actor 目标和 GAD actor 更新；actor 奖励仍只用 adv。第一轮 replay=false。

从参考工作借鉴评价结构，不直接复现 SRaR 的步骤级 actor 奖励或优势估计器。本轮 judge 提供有位置依据的事件，critic 学习回答/前缀级辅助目标；这不是完整的 token-level PRM，也不能以论文中的收益作为本实现有效性的保证。

## 2. 必读参考：论文、仓库、具体入口

以下链接于 2026-10-14 查阅。网页和 main 分支可能变化，执行模型应记录实际参考的论文版本、仓库 commit 和文件哈希；不要把第三方讲解代替论文及官方实现。只需先阅读相关文件，不需要安装或启动参考仓库。

### 2.1 主参考：SRaR

- 标题：**Step-wise Rubric Rewards for LLM Reasoning**。
- 论文：https://arxiv.org/abs/2605.17291
- PDF：https://arxiv.org/pdf/2605.17291
- 官方仓库：https://github.com/akarinmoe/SRaR
- 实际 judge 模板及奖励管理器：https://github.com/akarinmoe/SRaR/blob/main/recipe/SRaR/reward_manager.py
- 原始文本：https://raw.githubusercontent.com/akarinmoe/SRaR/main/recipe/SRaR/reward_manager.py
- 查阅 PDF 的附录 B.2（Rubric Distillation Prompt）、B.4（LLM Judge Prompt）、B.5（Step-Level Judge Prompt）；PDF 第 14–16 页。HTML 版本可能使用不同章节编号，应按标题查找。

**借鉴内容：**

- SUGGEST 描述可检查的正确步骤；PITFALL 描述具体错误事件，二者不是同一分数的两端。
- 每项判断关联到可见推理位置，保留发生了什么及发生在哪里。
- rubric 条目要具体、与数学任务有关，避免格式和篇幅奖励。

**不直接照搬：**

- SRaR 还包含 BONUS、ANSWER、步骤级 reward 与优势计算；本轮不加入额外“优雅解法”奖励，也不改变 actor 的 credit assignment。
- PITFALL 的 satisfied=true 表示犯错，不能误当成正向满足。
- 论文步骤检查 prompt 的“假设此前步骤正确”适合其局部检查任务；我们不能据此把建立在错误前提上的后续计算当作有效解题进展。
- 原实现依赖 `### Step N:` 等边界。此次不能为适配 judge 强制改变 actor 输出格式，优先对已有文本定位。
- 当前公开 `call_judge_async` 会把 response 裁至 12,000 字符；这是实现行为，不是截断监督的理论规则。本项目禁止静默裁剪后仍宣称评分覆盖了完整可见轨迹。
- 训练回答上限 8192 不等于论文已验证了本文的前缀评分规则。

### 2.2 主参考：数学 RRM

- 标题：**Curing “Miracle Steps” in LLM Mathematical Reasoning with Rubric Rewards**。
- arXiv：https://arxiv.org/abs/2510.07774
- ACL 正式版本：https://aclanthology.org/2026.acl-long.844/
- PDF：https://aclanthology.org/2026.acl-long.844.pdf
- 官方仓库：https://github.com/YouliangYuan/rrm-cure-miracle-steps
- 提示词入口：附录 A，Prompt 1（错误分析）、Prompt 2（Rubric Generation）、Prompt 3（Scoring Data Generation）、Prompt 4（RRM 使用提示）；重点阅读 Prompt 2–3，PDF 第 15–16 页。

**借鉴内容：**

- 将“推理严谨”展开为数学操作是否合法、条件是否满足、结论是否有依据等可观察判据。
- 检查不合法约分、遗漏定义域、少量试例推出普遍结论、未经证明的结构假设、无依据跳到正确答案等错误。
- 接受不同的有效解法，按数学上的充分性判断，不强求参考路径、术语、顺序或篇幅。
- 保留部分进展，最终答案正确也不能洗掉无效推导。

**不直接照搬：**

- RRM 的整段 0–10 分、按题生成评分表及其独立奖励模型训练流程，不直接对应当前四通道 critic。
- 完整解答 rubric 中“未证明充分条件”与截断前缀“尚未处理到条件”不同；必须按本文第 5 节处理。
- 数学条件已由上下文隐含保证时，不因没有再写一遍条件而判错；不能把形式展开程度当成严谨性。

### 2.3 补充参考：PRM800K

- 论文：**Let's Verify Step by Step**，https://arxiv.org/abs/2305.20050
- 官方仓库及标注说明：https://github.com/openai/prm800k
- 阅读 README 的数据 schema 以及 `prm800k/instructions/` 中原始标注说明（实际路径以固定 commit 为准）。

借鉴“有效步骤／错误步骤／中性步骤”之间的区别。仅复述题目、表示准备尝试、重复已完成的运算不能自动算实质进展。该项目不是 rubric 方法，也不作为我们截断策略已被验证的证据。

### 2.4 后续参考：ROPD，本轮不实现动态 rubric

- 论文：**Rubric-based On-policy Distillation**，https://arxiv.org/abs/2605.07396
- PDF 附录 D：https://arxiv.org/pdf/2605.07396
- 官方仓库：https://github.com/Peregrine123/ROPD_official
- rubric 生成模板：https://github.com/Peregrine123/ROPD_official/blob/main/prompts/rubricator.txt
- verifier 模板：https://github.com/Peregrine123/ROPD_official/blob/main/prompts/verifier.txt
- 已查阅的 verifier 原文：https://raw.githubusercontent.com/Peregrine123/ROPD_official/main/prompts/verifier.txt

参考其 teacher 也可能出错、不同解法安全、禁止模仿风格等原则。其当前 verifier 按项 true/false，不能直接作为“未发生／未知／错误”的统一标签规则。本轮不实现 teacher/student 对比生成 rubric；后续 evo-rubric 再单独设计。

## 3. 第一版结构：固定判据 + 当前题目的数学证据

保留固定版本的通用判据，由 judge 针对当前题目识别具体证据。第一轮不增加独立 rubric 生成服务，不依赖唯一参考解法，不把 teacher 回答作为绝对真值。

与 SRaR 的按题生成条目相比，这是有意简化：采用其“正确进展／错误事件”组织方式，而非完整重现其按题 rubric 数据集。若此版本不足以稳定判断进展，先报告校准失败，不能通过堆更多事件或无条件降低 unknown 来制造覆盖率。

judge 输入包括题目、可见回答、可选标准最终答案、可靠的截断元信息及固定准则版本。teacher/student 使用相同匿名输入协议；不告诉 judge 回答的模型身份。最终答案只用于核查，不能由答案正确反推过程正确，更不能替回答补推导。

## 4. 事件定义与证据要求

| 事件 | 含义 | 标签原则 |
|---|---|---|
| verified_progress | 已成立、与解题目标相关的数学进展 | 需要具体中间结论和成立依据 |
| explicit_error | 已观察到的不合法操作、计算或逻辑错误 | 指出错误位置、类型和影响 |
| corrected_error | 先前错误已被明确识别并正确修复 | 保留原错误和修复的引用关系 |
| neutral | 复述、重复、空泛计划等 | 不当成错误，也不增加进展 |
| unjudgeable | 当前材料/能力不足以可靠判定 | 给出具体原因，不能用中间分替代 |

第一轮不要求列出所有机械步骤，而是提取关键进展和实质错误。程序给原文提供稳定段落编号或字符范围，judge 返回原文短引用与对应编号；程序核验引用存在。不要依赖模型精确数字符。无法定位的事件不能作为高置信标签证据。

“引用存在”只保证证据来自原文，不保证数学判断正确，仍须人工校准。事件数量有限时，记录是否因输出预算遗漏分析；没有完成必要检查时不能输出“没有错误”的确定标签。

### 4.1 具体正向判据

- 正确建立变量、方程或数学模型，且模型忠实表达题意。
- 推导出有用的中间等式、不等式、不变量或约束。
- 正确完成与目标有关的分支分析或排除不可能情况。
- 证明当前采用的方法需要的关键前提。
- 有依据地修复此前错误，并重新建立可靠推导链。

选择不同正确方法、简洁但足够的常规变形均可计入。重复同一结论只算一次；无关但正确的计算不算有效进展；从已知错误前提推出的“局部计算正确”不能算全局有效进展。

### 4.2 具体错误判据

- 算术、符号、代数变形错误。
- 在上下文未保证条件时进行除法、开方、取对数等不合法变换。
- 漏掉已被当前结论排除的合法分支；凭有限枚举宣称穷尽。
- 使用未证明且非题设给定的关键假设。
- 结论与前述结果矛盾，或从无效路径直接跳到答案。

错误是否严重依据其对当前推导的影响，而非回答长短或错误数量。不能让大量简单正确步骤抵消一个决定性错误。

## 5. 截断、未完成与自我修正

### 5.1 截断轨迹不再整体 mask 过程维度

- 已完整表达的推导或明确错误照常评价。
- 末尾半截句子/公式单独标为不完整，不据此判断其尚未表达的结论。
- 缺少最终答案不抹掉前面的有效进展，也不因此自动判定前面有错。
- 已到达某个结论却缺少必要依据，与“尚未推进到该结论”不同。
- 保留 response source truncation、训练 token 截断和 judge 输入溢出三个不同字段。
- judge 输入溢出时不静默截头/截尾；按明确策略重试或 mask。第一版不引入有信息损失的自动摘要。

### 5.2 关键区别

“尚未讨论 x=0”本身不必是错误；“已经除以 x 并宣称获得全部解，而题设不保证 x≠0”可构成明确问题。条件是否必须在该处处理，应结合当前声称的结论判断。

“没有发现已发生错误”不等于“证明已经完成”。两者必须分开记录。

### 5.3 自我修正

仅说“等等，错了”不算修复。需要指出失效内容并给出正确替代，且后续不再依赖错误结果。修复后的有效结果可以提供进展，历史错误仍保留在事件日志。不得按修正次数加分，避免通过重复犯错/修错获得奖励。

## 6. Critic 辅助目标与 loss

### 6.1 建议四通道布局

```text
[adv, answer_correctness, verified_progress, unresolved_error_severity]
```

- adv：原 BT 目标，仍是 actor 唯一奖励来源。
- answer_correctness：调用主计划的公共 math_unified_v1 scorer，与训练验证、外部评估使用同一解析/数学判等实现，不由 judge 凭过程打分。
- verified_progress：有效数学进展的粗粒度目标。
- unresolved_error_severity：仍影响当前推导、未有效修复的错误严重程度；值越大错误越严重，不是正向奖励。

旧 reasoning_validity / condition_coverage 不保留为额外训练目标；相关检查并入具体事件。四通道数量虽相同，语义已经变化，不能把旧四通道模型或标签直接加载为 V2。

建议标签锚点（需第 9 节校准后定稿）：

| 目标 | 0 | 0.5 | 1 |
|---|---|---|---|
| verified_progress | 没有可验证的实质进展 | 有正确且相关的中间进展，尚未形成充分解答 | 当前可见推导已经充分解决目标，允许常规省略 |
| unresolved_error_severity | 已检查范围内未发现未修复错误 | 有局部可修复错误，尚未推翻核心路线 | 有使关键结论失效的实质错误 |

重要规则：

- judge 无法可靠判定时使用 null + mask，不能用 0.5 表示不确定。
- 没有任何可评价数学内容时，progress 可为 0；error 维度通常 mask，不能把“空回答没有错误”当成高质量过程信号。
- 正确前缀可能是 progress=0.5、error=0；错误轨迹也可保留错误前已经独立成立的 progress=0.5，同时 error=1。
- 某一处明确严重错误足以支持 error=1，即使轨迹另有无法判断的片段；反过来 error=0 需要足够的检查覆盖。
- progress=1 与 error=1 是否存在不一致要结合证据校验；不得将有实质未修复错误的同一主推导认定为充分解答。
- 过程的“解决目标”和最终答案的可解析、数学等价正确性不同，不强制二者完全一致。

### 6.2 训练方式

沿用 masked BCEWithLogits 的辅助训练框架和原有全局 minibatch 去重归一化：

```text
Lcritic = LBT + lambda_aux * mean_valid_dimensions(Lcorrectness, Lprogress, Lerror)
```

这是有序等级的软目标近似，并非已校准的概率。error 通道以严重程度为目标正常最小化损失，不对 loss 取负，也不直接从 actor reward 中扣除 error。

第一轮保持原 lambda_aux=0.1 作为可比较起点，不宣称这是最优权重。实际标签频率和梯度诊断不合理时，应在后续消融中调整，不通过语义不明的临时权重补救。

judge 先给事件证据，再给按锚点得到的汇总标签；程序负责 schema、证据引用、mask 和基本一致性校验，不把事件条数简单平均成质量分数。

## 7. Prompt 与 schema 实施要求

建议新增 `rubric/specs/math_process_v2.yaml`、`rubric/prompts/judge_math_process_v2.txt` 和版本化 contracts。

Prompt 包含以下顺序：任务边界 → 题目/回答/截断信息 → 可验证进展与错误判据 → 不同解法兼容 → 截断/自我修正规则 → 标签锚点 → JSON 输出合同。

- 模板使用结构占位符说明字段，不用唯一的“progress=0.5、error=unknown”示例暗示固定答案。
- 提供有区别的短例子：正确完整、正确前缀、明确错误、错误已修复、纯复述、末尾半截公式、替代有效解法。
- 示例用于解释边界，不能照搬被评题的答案或奖励某种文风。
- 正确性由公共数学 scorer 填充；judge 不覆盖它。没有最终答案可形成 0 分正确性标签，但不抹掉已有过程进展；解析歧义或判等工具异常按主计划 mask 正确性维度，不生成伪负例。
- 输出包括 rubric/schema 版本、completion 状态、关键事件、两个过程标签及理由、无法判断原因。
- 约束 valid=false 时 value=null；错误信息与数学 unknown 分开。
- 完成状态和事件仅作辅助证据；本轮不额外训练 completion head，不增加 actor 长度奖励。
- 标注预算通过真实长回答校准，不能因输出 JSON 更复杂就直接任意增大 max_tokens。记录输入/输出 token、失败率与标注时间。

## 8. 文件级改动与版本迁移

| 位置 | 必要工作 |
|---|---|
| `rubric/specs/`, `rubric/prompts/` | 新增 V2 判据和提示词，保留 V1 以便审计 |
| `rubric/core/contracts.py`, `heads.py` | 版本化维度、schema、模型布局；去掉默认 math_v1 的隐式假设 |
| `gad/tools/prepare_rubric_critic.py` | 从相同初始单通道 critic 扩展 V2，保留 adv 初始参数 |
| `gad/extensions/online_rubric.py` | 截断前缀评分、事件证据和逐维度 mask |
| judge client/server | 新 JSON schema、输出校验、上下文预算与异常处理 |
| `gad/extensions/critic_batch.py`, `worker.py` | V2 维度透传和指标命名，保持去重/DP 归一化 |
| 配置、cache、checkpoint | 引用全局 scoring.math；rubric/prompt/scorer/schema/head_layout 的身份校验 |
| 审计与验证脚本 | 新旧标签对照、人工样本、事件引用、目标分布 |

重新开展训练，使用新运行目录和新缓存。版本冲突直接拒绝，不靠“同为 4 通道”认定兼容。prompt/spec/schema/scorer 中任何影响标签语义的变更都应改变缓存身份。

## 9. 验收：先确保标签可用，再重跑训练

### 9.1 固定校准集

从已有轨迹中分层抽取 100–200 条作为建议起点，覆盖 teacher/student、正确/错误、完整/自然截断、自我修正、不同解法；另加入可控截断的同一回答前缀。校准题目与最终报告效果的测试题分开。

人工逐例检查：事件是否存在、数学判断是否成立、是否真正推进目标、错误是否已修复、unknown 是否必要。记录过程标签的一致性以及严重错误的漏检/误报，不能只看 JSON 合法率和非空覆盖率。

没有足够人工依据时应明确“尚未校准”，不能以 judge 自评替代人工通过。

### 9.2 成对测试

- 同一正确轨迹在不同完整步骤处截断：前面已成立的证据不能被清空，也不能自动获得完整解答分。
- 正确前缀后加入明确错误：已成立的独立进展保留，错误目标上升。
- 在错误之后只说“已修正”：不能当成真实修复；给出有效修复才改变未修复错误状态。
- 加入重复正确步骤：进展分不应仅因长度增长而提高。
- 改写为另一种正确解法：不因偏离示例路径扣分。
- 最后半截公式：局部 unknown，不全段 mask。
- 调换 teacher/student 来源身份但保持文本：标签一致。
- 缺失 judge 结果/上下文溢出：不得变成“无错误”。

上述测试需要真实 judge 校准与程序合同测试各自完成。简单 mock 测试只能证明代码传递正确，不能证明数学标签质量。

### 9.3 新训练组

按主计划重做 A：BT baseline；B：BT + 修复后的 correctness；C：BT + correctness + V2 progress/error。种子、数据、初始模型及训练预算对齐，replay=false。

先通过少步 GPU 验证与 5k 对照，再决定 17k。持续报告按 teacher/student 和截断状态分层的有效监督比例、标签分布、adv 质量诊断和 actor 外部数学评估。

三组正式验证/评估均使用 math_unified_v1 的同一 fingerprint；使用一份 metrics.json，主指标为按题平均 pass_at_1。生成设置、工具失败和分母处理遵循主计划第 4、5、10 节，不另外引入 V2 专用答案评分器。过程 judge 的评分不是第二套最终答案正确率。

## 10. 执行模型的完成清单

1. 阅读第 2 节原始论文提示词及代码，并记录固定版本；核对本文摘述与所用版本一致。
2. 完成主计划的公共数学 scorer、三端一致性和 seed 等前置修复。
3. 实现 V2 schema、提示词、头布局和逐维度截断监督；保留原 BT 与 adv-only reward。
4. 交付可审阅的 prompt/spec、JSON 合同、例子及人工校准结果；如规则不稳定，先修规则再训练。
5. 完成 CPU、少步 GPU、Warmup→GAD 传递和恢复合同验证。
6. 按新配置重跑 baseline / correctness-only / V2 joint，并报告实际收益及局限。

本文的前缀汇总标签、未修复错误严重程度以及 GAD 接入方式属于本项目的改编设计，不声称为 SRaR/RRM 原论文已验证的方案。
