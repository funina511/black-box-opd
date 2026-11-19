# GAD Rubric V2 统一执行计划

初版日期：2026-10-14；本轮计划更新：2026-10-15。状态：基础改造已实现，judge 校准未通过，真实 V2 Warmup → GAD 与保存恢复尚未验证。**本文件为唯一执行入口，取代此前两份计划的执行安排。** 下方已合并必要技术规格及论文/仓库链接，不需要按两份旧文档分别实施一遍。

**当前执行入口：附录 E「Thinking Judge 监督能力验证与 8B 候选对照」。** 附录 D 的协议优化已执行，报告见 `reports/rubric_v2/judge_protocol_v21_20261015/report.md`：容量与结构改善，但真实回答过程监督仍为零。下一轮优先验证 judge 的数学监督能力，速度优化随后进行。执行顺序、thinking 设置及本轮验证预算以附录 E 为准；附录 D 保留为已执行工作的规格与历史记录，原 BT、actor 更新、公共答案 scorer 和用户负责正式重跑的分工不变。此次文档更新只制定计划，不启动代码修改或验证任务。

## 1. 分工与完成边界

### 执行模型负责

- 修复公共数学评分、seed、数据元信息和相关评估入口。
- 实现 Rubric V2、截断轨迹过程监督、critic 辅助目标与必要日志。
- 完成 CPU 合同测试、有限的真实 judge 标签检查和少量 GPU 训练验证。
- 准备正式 baseline / rubric V2 配置、启动命令和 dry-run 核对结果，交付用户执行。

### 用户负责

- 决定何时启动并执行正式 baseline / rubric V2 训练及完整外部评估。
- 决定是否增加 correctness-only 消融、多个 seed、5k/17k 扩展。
- 审阅标注样例，完成需要人类数学判断的标签校准；执行模型不能把自身检查称为人工验收。

**执行模型不得自行启动 5k/17k 正式重跑、完整 epoch、多 seed 扫描或全量外部生成评估。** 小规模验证的目标是检查训练链路正确，不是证明 actor 性能提升；无需等数学准确率上涨才结束。

当前只授权计划所列实现和有上限的验证，不授权停止其他训练任务。运行验证前检查可用资源，使用独立 run/端口/缓存；若资源不足，完成其余工作并准确交付未验证项，不占用其他实验已使用的 GPU。

## 2. 固定设计

1. 正确性统一使用 `math_unified_v1`：基于 OPRD 数学判等实现修复答案提取；rubric 标签、训练验证、外部评估共用同一入口和身份。
2. 正式评估只产出一份 `metrics.json`。旧 CARD/OPRD 差异仅用于有界的历史审计，不继续发布双评分主表。
3. 过程监督采用 SRaR 的正确进展/错误事件组织方式，加上数学 RRM 的具体判据与不同解法兼容原则。
4. Critic 布局为 `[adv, answer_correctness, verified_progress, unresolved_error_severity]`。错误严重度越大表示越差，但它只是辅助预测目标，不直接进入 actor reward。
5. 原 BT、Warmup actor 目标、GAD actor 更新顺序保持不变；Warmup 和 GAD 都训练 rubric，actor 仍只使用 adv。
6. 截断轨迹的已完成步骤仍可评分；末尾不完整片段局部 unknown，不整体屏蔽过程监督。
7. 新头/标签语义使用新版本、新配置和新缓存，不接续旧 rubric 四通道 checkpoint。复用相同初始单通道 critic 的 adv 参数。
8. 本轮 replay=false；仅保证元信息、scorer 身份和保存恢复合同不被改坏，不做 replay 性能实验。

## 3. 严格执行顺序

### 第一步：阅读和固定接口，不启动模型

阅读当前实现、适用 AGENTS.md 和本文参考资料。记录代码版本与数据身份；确认公共 scorer、V2 schema/head layout、配置字段和缓存身份。论文仅阅读必要提示词/代码，不安装参考仓库，也不运行其启动脚本。

交付：简短修改清单及最终配置接口。可直接继续实现，不需要每完成一个子步骤都请求用户确认。

### 第二步：基础修复与 CPU 测试

依次完成：公共答案 scorer → 三端接入/单份指标输出 → 显式 rollout seed → teacher 选中候选与截断元信息 → cache/checkpoint 身份。

以固定测试样例验证解析/判等，再用两组旧 5k 的已有回答做小样本复核。默认历史审计最多 128 条回答，分层包含已确认的跨行 Answer 错误；不得重新生成旧回答。准备完整审计脚本即可，全量历史审计不作为本轮完成条件。

交付：修复代码、CPU 测试结果、带分歧原因的小样本审计。新 scorer 的三处入口必须返回一致公共结果；mask 与评估计零的消费策略按技术规格分别执行。

### 第三步：实现 V2 并完成程序级合同测试

完成固定判据、prompt、事件 schema、证据定位、两个过程标签、四通道模型准备和联合 loss 接入。参考附录 B，不新增按题 rubric 生成服务或步骤级 actor reward。

完成 CPU 测试：截断不全段 mask、局部 unknown、自我修正、证据不存在、标签语义/缓存版本、总损失、重复 teacher 权重与 DP/microbatch 归一化。直接复用有意义的现有测试，不为每个实现细节增加镜像测试。

### 第四步：有限真实 judge 检查

选取最多 24 条已存在或人工构造的明确轨迹，覆盖完整正确、正确前缀、明确错误、错误修复、纯复述、重复、半截公式与替代解法；来自训练/诊断题，不能拿最终测试题调 prompt。

第一轮每条标注一次。只有发现明确问题并修复后，才重测受影响样例一次；本步骤最多 48 次应用层标注请求，禁止循环调 prompt 直至指标好看。网络重试有独立有限上限，不能绕开总时间预算。

输出样例、证据、标签和失败原因。可从其中选择 8–12 条交给用户人工核查。执行模型的数学检查记为“模型辅助审查”，人工未审部分明确 pending；这不阻止安全的工程 smoke，但不能宣称标注质量已通过人工校准。

### 第五步：只做一条主要 GPU 训练 smoke

以下是工作量上限，不要求耗尽预算。以原项目的小模型和最小合法 batch 验证；建议 8–16 个短题组成 fixture、train batch=4 或满足 DP 整除的最小值、critic microbatch=1，Warmup n=1、GAD n=2。使用真实短 teacher 回答，不将裁坏的 teacher 文本冒充完整答案。

- baseline：最多 1 个全局训练步，确认 rubric=false 的路径正常。
- V2 joint Warmup：最多 2 个全局训练步，配置原有 gate 使其中至少一步包含 actor 更新，并检查四通道与共享主干梯度。
- 从该 smoke Warmup 导出并进入 GAD：最多 2 个全局训练步，检查交接、adv-only 奖励及 actor/critic 更新。
- 从 smoke GAD checkpoint 恢复：最多再 1 个全局训练步，检查状态、scorer/cache 身份与计数。

**正常路径合计最多 6 个全局训练步，不是每组各跑一个 epoch。** 固定微型 fixture，不能因 batch 变小而意外执行更多步；训练步到达上限必须真正结束。

GPU rollout 默认 cap 为 512，必要时最多 1024 token；明确它仅验证工程链路。baseline / correctness-only / lambda=0 的数学合同尽量用同一预生成 batch 的 CPU 测试验证，不为每个组合单独启动整套 trainer。correctness-only 正式训练不在模型职责内。

关闭 val_before_train、周期全量验证、最终全量验证及 ROUGE 等耗时路径。仅在交接/恢复必须的位置保存 checkpoint。若现有短跑开关仍会触发 final full validation，应先修正 smoke 的停止/验证控制并测试，不能只靠很大的 test_freq。

评分汇总正确性用已有回答完成，默认不额外启动外部 vLLM 生成任务。若原项目要求多卡合同，使用最小合法 DP；如只验证了 2 卡，则说明正式 4 卡未覆盖，不能声称拓扑等价已验证。

### 第六步：交付并停止训练工作

提交修改、必要测试结果、样例审查及风险说明，准备用户的正式启动命令和 dry-run 结果。完成后停止新增训练任务；正式训练效果留给用户检验。

## 4. 时间预算与失败停止条件

- 真实 judge 与 GPU 验证合计默认最多 **30 分钟墙钟时间**，包含服务启动/权重加载、重试和恢复验证；记录各段耗时。该上限是执行范围约束，不保证硬件能在此时间完成所有项目。
- 默认 6 个全局训练步上限；失败后仅可重跑受影响的最小阶段一次，额外最多 2 个全局步，且仍受同一个 30 分钟总预算约束。
- 不自动升级为 8192-token 长 rollout 验证或完整 5k 实验。长序列显存压力另列为未覆盖风险，交付可选命令，不自行执行。
- 遇到 OOM、重复 schema/标签错误、身份不一致、非有限 loss/梯度，停止当前 smoke，定位问题；不得用静默 mask 所有标签、降低数学判定标准或跳过校验换取“通过”。
- 预算耗尽即停止本次自建 smoke 作业并保留证据；不能终止其他进程，也不能继续后台跑验证。CPU 修复和静态检查仍可完成。未完成验证列为未完成，不伪造通过，也不进入正式重跑。
- 正常通过后不额外重复测试，不做超参调优，不以“再跑一组更稳妥”为由扩大任务。

## 5. 执行模型交付清单

- 文件级修改摘要、公共 scorer 协议、V2 prompt/spec/schema 与参考版本。
- 三端 scorer 一致性、seed 下发、teacher 元信息、DP/microbatch 辅助损失等 CPU 证据。
- 有界历史评分审计与真实 judge 样例；明确人工审阅状态。
- smoke 实际全局步数、耗时、使用 GPU/拓扑、关键 loss/梯度、Warmup→GAD 与恢复结果；未覆盖项单独列出。
- 新 baseline / rubric V2 正式配置及一键启动命令；可附 correctness-only 配置，但不得自动执行。
- 正式外部评估命令，统一 scorer/seed/采样协议，已通过 dry-run。
- 生成简洁交接报告，例如 `LMOps/reports/rubric_v2_handoff.md`，不要留下未说明的验证服务或后台训练。

## 6. 用户后续执行事项（不是模型的待办任务）

用户在审阅交付后启动同条件 baseline 和 rubric V2 正式实验、完整外部评估，并决定是否开展 correctness-only、多 seed 或 17k 扩展。建议先做 5k，但数据量、启动时机及正式实验预算由用户决定。

执行模型无需等待正式实验完成，也无需证明 V2 优于 baseline 才能交付。本轮完成标准是修复和工程验证的真实证据，而非研究效果。

## 7. 技术规格的适用方式

以下附录合并两份旧计划中仍适用的技术要求。**工作分工、样本数、时间和训练步数以本文件第 1–6 节为准。** 规格中的“正式评估”用于定义实现必须支持的行为，不是要求执行模型启动完整评估。

## 附录 A：公共评分、实验身份与数据修复技术规格

### 公共数学 scorer
#### 单一正式入口与版本

建议新增 `LMOps/scoring/math.py`，正式版本名 `math_unified_v1`。它基于现有 OPRD 评分链：`OPRD/sft/quality.py` 的答案提取/调用关系，以及其使用的 `grade_answer_mathd`、`grade_answer_sympy`，修复并明确提取合同后成为唯一正式入口。

OPRD 源码参考：`OPRD/sft/quality.py`、`OPRD/sft/score_math.py`、`OPRD/verl/verl/utils/reward_score/ttrl_math/math_utils.py`。实现时阅读相关目录的 AGENTS.md，优先在 LMOps 下建立轻量、固定版本的共享模块，不改动 OPRD 的训练工程。

固定所复用的数学判等源码、依赖和许可证，可将必要的纯 CPU 工具纳入共享模块。避免依赖运行时 PYTHONPATH 恰好指向哪个 `verl`；不能在不同调用端分别维护一份评分逻辑。该 scorer 不加载 LLM，也不启动独立评分服务。

保留旧 `LMOps/sample/math_scorer.py` 为 legacy 审计实现。取消新增 `answer_line_v2` 字符串匹配评分器的要求；新正式运行不再并列使用 CARD 和 OPRD 两个 scorer。

#### 答案提取合同

提取只看回答本身，不使用 ground truth 挑选候选。对全部可见回答定位显式答案标记，不再继承旧 CARD 的末尾 300 字符限制，也不从推理正文搜索“碰巧等于标准答案”的数字。

- 支持完整 `\\boxed{...}`，正确处理嵌套花括号；支持大小写不敏感的 `Answer:` / `Final Answer:` 及明确的 Markdown 包装。
- 按原文位置选择最后一个显式答案候选，而非无条件让早先的 boxed 覆盖后面的最终 Answer 行。Answer 行包含 boxed 时视为同一个候选，提取其表达式。
- 空 `Final Answer:` 标题后跟 `Answer: 42` 时继续解析后者，不把前缀作为答案内容。
- `Answer:\n42` 使用明确的后继非空行规则；若下一行也是答案标记，先解析标记，不能用跨行 `\s*` 吞入该前缀。
- 最后一个显式答案标记未闭合、为空或内容歧义时记录失败，不为了判对回退到更早的中间答案。
- 同一最终答案区域互相矛盾、无法确定唯一候选时标 ambiguous。合法的集合/区间/多根表达式作为一个数学答案处理，不能拆开后择一匹配 GT。
- 题目要求的对象类型应保留，不能简单剥掉所有单位、集合括号、顺序或定义域限制来制造等价。
- 明确的错误答案仍应判错。用真实回答和反例测试固定上述优先级，修改规则必须升级 scorer 版本。

#### 数学判等与返回接口

合法候选经过明确归一化后，采用固定版本的 MathD / SymPy 判等。支持其可靠覆盖的数字、分数及表达式等价，例如 142.0 与 142、0.5 与 1/2；不自行无限扩展数学证明能力。集合、区间、符号条件等需有明确测试，不能以宽泛数值近似替代语义。

有确定结论返回 correct/incorrect；依赖缺失、工具异常或超时保留独立状态，不能被工具内部的宽泛 except 静默折叠为“数学上错误”。判等预算和依赖身份记录在 scorer 协议中，所有调用端一致。

建议公共结构：

```text
scorer_version: math_unified_v1
scorer_fingerprint: 实现、提取规则、判等配置和依赖的身份
pred: str | null
correct: bool | null
parse_status: ok / no_answer / malformed / ambiguous
parse_method: boxed / answer_line / answer_next_line / none
grade_status: correct / incorrect / not_attempted / timeout / error
candidate_count: int
error_reason: str | null
```

`correct=null` 表示没有可靠的数学判定；由下表定义场景处理，不能直接 `bool(null)` 生成训练负例。若旧验证接口需要 `acc` 或 ±1 score，由薄适配器显式转换，不能在适配器重写提取/判等。

| 状态 | 正式评估/训练验证 | correctness 辅助标签 |
|---|---|---|
| 成功提取且判对/错 | 记 1/0 | 有效 1/0 |
| 明确没有最终答案 | 记 0，保留在分母 | 有效 0；过程维度仍独立评价 |
| 有答案标记但 malformed/ambiguous | 记 0，同时报告解析失败 | mask，并记录原因 |
| 判等工具 timeout/error | 保留样本；仅可给带错误计数的临时统计，正式报告需重试或复核解决 | mask；不能当作有效 0 |
| 缺失/非法 ground truth | 数据 preflight 失败，不能静默剔除样本后报告正式准确率 | 数据准备失败 |

工具异常未解决时 `evaluation_complete=false`，不把临时数字当完整实验成绩。合法但不符合约定输出格式的回答可按解析失败计错，但必须保留失败率供审阅。题目被截断时训练正确性标签仍 mask；响应被截断不自动 mask，已有最终答案可以判等，没有最终答案按上表处理。

#### 单一配置来源与三处一致性

新增全局 `scoring.math` 配置块，至少声明 version、入口与判等预算；rubric、训练验证和外部评估都引用该配置。不得长期保留三套可独立漂移的 scorer 路径。

- 取消此前新增独立 `rubric.scorer_path` 的方案。旧 `validation.scorer_path` / `GAD_MATH_SCORER` 如需适配，只可由统一配置生成；外部覆盖冲突时 preflight 报错。
- baseline、correctness-only、joint 的验证与评估使用相同 fingerprint；是否启用 rubric 不改变 scorer。
- 同一个 `(response, ground_truth)` 经三处入口必须获得相同公共解析/判等结果；仅上表列出的消费策略不同。
- 新模型导出/评估读取运行快照中的 scorer 身份，不跟随全局模板的后续修改。
- 身份进入标签 cache key、实验配置、checkpoint manifest 和恢复校验，包括共享依赖，不能仅哈希入口文件。
- 新正式实验使用新缓存，不将旧 CARD 标签直接迁移为新标签。
- 新数据准备若使用正确性过滤，也调用公共 scorer 并记录版本；现有 5k 题目和 teacher 回答保持固定，不因换 scorer 静默重新筛选而改变对照数据。

#### 单份正式结果与历史隔离

新评估在新 profile 下只产出一份正式 `scores/metrics.json`，以及对应的逐样本 `scored_samples.jsonl`，统一标注 `math_unified_v1` 和 fingerprint。不再自动生成并列的 `card_metrics.json`、`answer_line_v2_metrics.json`。

统一提取协议相对旧 OPRD 也可能改变部分结果，因此不能把新 scorer 称为与旧 OPRD 逐条等价。旧 `metrics.json` 和 `card_metrics.json` 均保持不变；旧回答的新评分写到审计目录或新评分 profile，绝不覆盖。

### 指标汇总要求
扩展 `evaluate/summarize_eval.py`：

- 新默认表仅显示统一 scorer 的结果及身份；旧双评分表只在显式 legacy/audit 模式展示。
- 主指标 `pass_at_1`：对每题的采样正确比例取平均。`sample_accuracy`：所有回答的正确比例；每题 n 相同且样本齐全时二者相同，不作为两个不同 scorer 的成绩展示。
- `first_rollout_accuracy` 只看每题第一个回答；`any_success_at_n` 表示至少一次成功的题目比例，明确 n，不声称有模型选出了最佳答案。
- 若兼容旧字段 `best_of_n_accuracy`、`sample_accuracy_avg_at_n`，必须标注其映射，不能默默改变分母。主显示用上述统一命名。
- `valid_rate` 如保留仅为格式诊断，不筛选 accuracy 的分母。另列 parse_failure_rate、no_answer_rate、grader_error_rate、evaluation_complete。
- 正式比较先核对预期题目与 rollout 索引，样本缺失或重复时不能静默缩小分母；补齐/解决后再生成正式结果。
- `score_mean_pm1` 不标成 accuracy。
- profile 名称不能替代真实协议：当前目录名含 n8/16k，而实际 MATH-500 为 n=2、max_new_tokens=8192。汇总从 manifest/sampling_config 读取并展示真实设置。
- 比较模型差异时按题目聚合；可做问题级 paired bootstrap，不把同题多次 rollout 当完全独立样本。单训练种子的区间不覆盖训练随机性。

### 种子与实验身份
涉及 `gad/tools/pipeline.py`、`extension_config.py`、`config.py`、新实验配置模板及 backend 补丁。

- 新正式配置要求显式 `rollout.seed`，建议初始对照为 42。
- baseline、correctness-only、joint 的 Warmup 和 GAD 都通过同一路径下发该值。
- 删除“开启 rubric/replay 才回退 rubric.seed”的新实验行为，避免开关功能顺便改变 rollout。
- 分别记录训练初始化/数据顺序种子、rollout 引擎种子、请求 SamplingParams seed、judge seed、replay seed、评估 seed。缺失项标缺失，不能统一写作一个 experiment seed。
- 实际 worker 日志与 stage manifest 记录最终生效值；若引擎与请求种子都使用同一值，明确记录，不仅保存命令行意图。
- 明确请求级 seed 的行为并在所有组保持一致。检查重复 seed 是否影响同题多次采样的多样性；同 seed 不意味着相同输出，更不保证不同 GPU 拓扑下逐位一致。
- 实验 preflight 比较 data/model/tokenizer 哈希、模型初始化、训练步数、batch、n、温度、长度、optimizer、验证设置和 backend 版本；列出允许不同的 rubric 项。
- 增加扩展代码路径下 `loss_coef=0` 的小规模控制，检查只有开关/重建 batch/随机数消耗引入的差异。不要只用旧 backend 与 disabled 的等价测试推断 enabled 路径等价。

正式比较中训练 GPU 数、评估 GPU 拓扑、版本和采样参数也尽量相同。重新从同一初始 actor/critic 各自跑 Warmup；不能让 baseline 与 rubric 共用已受 rubric 训练的 Warmup critic。

### 教师元信息
涉及 `sample/build_gad_data.py`、数据 contract、dataset backend patch、`gad/extensions/runtime.py`、`critic_batch.py`。

- 数据构建在选取答案时同时返回其原始索引，不再用 `.index(answer)` 从文本反推，以支持重复文本、不同元信息的候选。
- `extra_info` 保存 selected_teacher_index、选中回答文本哈希、source_finish_reason、source_truncated 和元信息版本。
- 截断状态分开记录：采样端 length 截断、训练 tokenization 截断；二者用于 judge 的可见性元信息。V2 不因此整体 mask 过程维度，按附录 B逐维度判断。
- runtime 优先读取绑定到选中回答的元信息，不固定读取 teacher_index 的 `[0]`。
- 旧数据迁移时从 selection sidecar/teacher index 和原文本交叉验证。多答案且无法唯一确认时停止数据准备，不默认为第 0 条。
- 新 replay pair 一并保存这些元信息，重放时保持一致。

验收：构造第 0 条截断、第 1 条完整且选第 1 条的案例；完整回答不得被错误 mask。反向案例、重复答案文本案例也应通过。

### 监督与联合训练诊断
#### 覆盖率和标签分布

涉及 `extensions/online_rubric.py`、`critic_batch.py`，并提供旧日志离线汇总脚本。

- 每个维度分别记录 teacher/student 的 0、0.5、1、unknown；给出 unique 数和实际辅助权重总量。
- 区分 pair 次数与唯一回答数；截断分子和分母必须同口径。
- 当前 batch 的 unknown 原因包括缓存命中的标签；另列本步新增标注失败计数，不混用。
- 拆分 prompt_truncated、response_truncated、judge_unknown、格式错误、context overflow、规则答案解析异常等原因。
- 记录未截断样本条件下的有效覆盖率，区分“因截断未评”与“已请求但 judge 不判断”。
- 统计有效标签单值比例、各维度有效样本数，避免以很低的辅助 loss 推断监督有效。
- 本节只描述诊断；prompt 与截断策略按附录 B修改，不把 unknown 强行转低分。

#### 损失和 adv

- 新增 `critic/loss_total = critic/loss_adv + rubric/loss_aux_weighted`，保留 `critic/d_loss` 的历史 BT 含义。
- 记录 optimizer step、microbatch 累积和全局归一化口径；不能再对已全局归一化的 aux 重复除累积次数。
- 总 loss 日志不等于梯度比例。梯度范数/余弦相似度如需分析，应在小型诊断任务中做，不默认给正式训练增加多次 backward。
- 固定一组独立诊断题目和回答，在无 dropout 的评估模式下保存四通道分数；不参与训练。
- 分析 adv 对正确/错误回答的排序、同题质量偏好，单独列 teacher 错而 student 对的情况；正确性指标标明 scorer 版本。
- 过程维度的 adv 相关性需经过人工抽查的标签，不能把未经审计的 judge 输出视为真值。

#### Replay 的处理边界

- 本轮主实验 replay=false，避免把回放和 rubric 变化混在一起。
- 保持轨迹池仅用于 critic 的既有设计；修复 teacher 元信息时覆盖 replay 保存/恢复测试。
- `max_bytes` 当前是 JSON 序列化字节预算，不是 Python RAM 硬上限。修正文档/指标名说明，另可记录进程 RSS 作为诊断，但不承诺严格内存限额。
- scorer 身份变化不能沿用旧标签恢复池。新正式实验使用空 replay 池；未来 replay 消融另行开始。

### 必须的 CPU 合同测试
1. 统一答案提取：普通行、标题后 Answer 行、空行、CRLF、Markdown、嵌套 boxed、多个候选及优先级、空标题、未闭合表达式、超出旧 300 字符窗口的答案、裸换行答案；legacy 审计样例保持原行为。
2. 明确的错误答案仍为错误；不能用“尽量提高正确率”的启发式从正文或 GT 猜答案。
3. boxed-only、数字/分数/表达式等价及明确不等价的反例，检查不会通过忽略结构误判集合、区间、多根或题目要求的对象类型。
4. rubric、验证、外部评估三处公共结果一致；覆盖有效 0/1、无答案、解析失败、判等超时/异常、prompt 截断及缓存身份变化。检查错误没有被静默变为训练负例。
5. 多题多样本 sample/first/any-success、按题 pass@1 和固定分母测试；缺失 GT、缺失 rollout、重复样本及工具错误导致不完整结果的处理。
6. baseline/rubric/replay 开关组合均传入相同显式 rollout seed，漏配新正式实验报错。
7. teacher 选中索引与截断状态、重复文本、replay round trip。
8. cache hit 和 miss 下当前 batch 覆盖率/原因一致；重复 teacher 时 unique 比例不超过 1。
9. 保留现有联合 loss、DP/microbatch 归一化测试，并核对 hook 测试检查的是本次生成的 backend，而非固定旧目录。

### 正式评估协议（实现并交付用户，不在本轮运行）
- 主指标：math_unified_v1 下的外部 pass_at_1（每题采样正确率平均）。每题 n 固定时等于 sample accuracy。三组使用完全相同 scorer 与依赖身份。
- 次指标：同一 scorer 下的 any_success_at_n、first_rollout_accuracy，以及解析失败率、无答案率、工具错误率、截断率和长度；不再并列发布另一套答案正确率。
- 保持完整题目/样本分母，缺失 GT、未完成生成或未解决工具错误时不发布正式完整成绩。
- 同时展示 DAPO 验证集与外部测试，不根据单一验证集上涨宣布有效。
- MATH-500 n=2，其他现有测试 n=8，max_new_tokens=8192；如决定改变，所有比较组一起改变并使用新 profile。
- 外部评估固定 non-thinking、temperature=1.0、top_p=0.95、top_k=-1、repetition_penalty=1.0、seed=42、max_model_len=10240，并对齐推理拓扑和依赖。各 benchmark 单独报告；如给总平均，明确采用 benchmark 宏平均，不能按回答条数混合不同 n。
- 训练验证继续可用 n=1、greedy 的低成本设置，但与外部采样评估分别命名。共用 scorer 不代表生成协议相同，不能直接用两个数值的高低判断域外泛化。
- 为新结果使用新 profile（如 math_unified_v1_n8_math500n2_8k），并记录实际参数；旧实验必须在同一新 scorer 下重评分后才参与新口径比较。
- 按题比较每组差异，报告样本量、训练重复次数与不确定性；不能只看 BT 判别准确率或 rubric loss。
- 不预设必须提升的数值门槛。进入 17k 的依据是有重复支持的收益或清晰、值得验证的机制假设。

## 附录 B：Rubric V2、论文与官方仓库

### 必读参考：论文、仓库、具体入口
以下链接于 2026-10-14 查阅。网页和 main 分支可能变化，执行模型应记录实际参考的论文版本、仓库 commit 和文件哈希；不要把第三方讲解代替论文及官方实现。只需先阅读相关文件，不需要安装或启动参考仓库。

#### 主参考：SRaR

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

#### 主参考：数学 RRM

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
- 完整解答 rubric 中“未证明充分条件”与截断前缀“尚未处理到条件”不同；必须按下文“截断、未完成与自我修正”处理。
- 数学条件已由上下文隐含保证时，不因没有再写一遍条件而判错；不能把形式展开程度当成严谨性。

#### 补充参考：PRM800K

- 论文：**Let's Verify Step by Step**，https://arxiv.org/abs/2305.20050
- 官方仓库及标注说明：https://github.com/openai/prm800k
- 阅读 README 的数据 schema 以及 `prm800k/instructions/` 中原始标注说明（实际路径以固定 commit 为准）。

借鉴“有效步骤／错误步骤／中性步骤”之间的区别。仅复述题目、表示准备尝试、重复已完成的运算不能自动算实质进展。该项目不是 rubric 方法，也不作为我们截断策略已被验证的证据。

#### 后续参考：ROPD，本轮不实现动态 rubric

- 论文：**Rubric-based On-policy Distillation**，https://arxiv.org/abs/2605.07396
- PDF 附录 D：https://arxiv.org/pdf/2605.07396
- 官方仓库：https://github.com/Peregrine123/ROPD_official
- rubric 生成模板：https://github.com/Peregrine123/ROPD_official/blob/main/prompts/rubricator.txt
- verifier 模板：https://github.com/Peregrine123/ROPD_official/blob/main/prompts/verifier.txt
- 已查阅的 verifier 原文：https://raw.githubusercontent.com/Peregrine123/ROPD_official/main/prompts/verifier.txt

参考其 teacher 也可能出错、不同解法安全、禁止模仿风格等原则。其当前 verifier 按项 true/false，不能直接作为“未发生／未知／错误”的统一标签规则。本轮不实现 teacher/student 对比生成 rubric；后续 evo-rubric 再单独设计。

### 3. 第一版结构：固定判据 + 当前题目的数学证据

保留固定版本的通用判据，由 judge 针对当前题目识别具体证据。第一轮不增加独立 rubric 生成服务，不依赖唯一参考解法，不把 teacher 回答作为绝对真值。

与 SRaR 的按题生成条目相比，这是有意简化：采用其“正确进展／错误事件”组织方式，而非完整重现其按题 rubric 数据集。若此版本不足以稳定判断进展，先报告校准失败，不能通过堆更多事件或无条件降低 unknown 来制造覆盖率。

judge 输入包括题目、可见回答、可选标准最终答案、可靠的截断元信息及固定准则版本。teacher/student 使用相同匿名输入协议；不告诉 judge 回答的模型身份。最终答案只用于核查，不能由答案正确反推过程正确，更不能替回答补推导。

### 4. 事件定义与证据要求

| 事件 | 含义 | 标签原则 |
|---|---|---|
| verified_progress | 已成立、与解题目标相关的数学进展 | 需要具体中间结论和成立依据 |
| explicit_error | 已观察到的不合法操作、计算或逻辑错误 | 指出错误位置、类型和影响 |
| corrected_error | 先前错误已被明确识别并正确修复 | 保留原错误和修复的引用关系 |
| neutral | 复述、重复、空泛计划等 | 不当成错误，也不增加进展 |
| unjudgeable | 当前材料/能力不足以可靠判定 | 给出具体原因，不能用中间分替代 |

第一轮不要求列出所有机械步骤，而是提取关键进展和实质错误。程序给原文提供稳定段落编号或字符范围，judge 返回原文短引用与对应编号；程序核验引用存在。不要依赖模型精确数字符。无法定位的事件不能作为高置信标签证据。

“引用存在”只保证证据来自原文，不保证数学判断正确，仍须人工校准。事件数量有限时，记录是否因输出预算遗漏分析；没有完成必要检查时不能输出“没有错误”的确定标签。

#### 具体正向判据

- 正确建立变量、方程或数学模型，且模型忠实表达题意。
- 推导出有用的中间等式、不等式、不变量或约束。
- 正确完成与目标有关的分支分析或排除不可能情况。
- 证明当前采用的方法需要的关键前提。
- 有依据地修复此前错误，并重新建立可靠推导链。

选择不同正确方法、简洁但足够的常规变形均可计入。重复同一结论只算一次；无关但正确的计算不算有效进展；从已知错误前提推出的“局部计算正确”不能算全局有效进展。

#### 具体错误判据

- 算术、符号、代数变形错误。
- 在上下文未保证条件时进行除法、开方、取对数等不合法变换。
- 漏掉已被当前结论排除的合法分支；凭有限枚举宣称穷尽。
- 使用未证明且非题设给定的关键假设。
- 结论与前述结果矛盾，或从无效路径直接跳到答案。

错误是否严重依据其对当前推导的影响，而非回答长短或错误数量。不能让大量简单正确步骤抵消一个决定性错误。

### 5. 截断、未完成与自我修正

#### 截断轨迹不再整体 mask 过程维度

- 已完整表达的推导或明确错误照常评价。
- 末尾半截句子/公式单独标为不完整，不据此判断其尚未表达的结论。
- 缺少最终答案不抹掉前面的有效进展，也不因此自动判定前面有错。
- 已到达某个结论却缺少必要依据，与“尚未推进到该结论”不同。
- 保留 response source truncation、训练 token 截断和 judge 输入溢出三个不同字段。
- judge 输入溢出时不静默截头/截尾；按明确策略重试或 mask。第一版不引入有信息损失的自动摘要。

#### 关键区别

“尚未讨论 x=0”本身不必是错误；“已经除以 x 并宣称获得全部解，而题设不保证 x≠0”可构成明确问题。条件是否必须在该处处理，应结合当前声称的结论判断。

“没有发现已发生错误”不等于“证明已经完成”。两者必须分开记录。

#### 自我修正

仅说“等等，错了”不算修复。需要指出失效内容并给出正确替代，且后续不再依赖错误结果。修复后的有效结果可以提供进展，历史错误仍保留在事件日志。不得按修正次数加分，避免通过重复犯错/修错获得奖励。

### 6. Critic 辅助目标与 loss

#### 建议四通道布局

```text
[adv, answer_correctness, verified_progress, unresolved_error_severity]
```

- adv：原 BT 目标，仍是 actor 唯一奖励来源。
- answer_correctness：调用本文的公共 math_unified_v1 scorer，与训练验证、外部评估使用同一解析/数学判等实现，不由 judge 凭过程打分。
- verified_progress：有效数学进展的粗粒度目标。
- unresolved_error_severity：仍影响当前推导、未有效修复的错误严重程度；值越大错误越严重，不是正向奖励。

旧 reasoning_validity / condition_coverage 不保留为额外训练目标；相关检查并入具体事件。四通道数量虽相同，语义已经变化，不能把旧四通道模型或标签直接加载为 V2。

建议标签锚点（需完成执行顺序第四步的样例检查，并标明人工校准状态）：

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

#### 训练方式

沿用 masked BCEWithLogits 的辅助训练框架和原有全局 minibatch 去重归一化：

```text
Lcritic = LBT + lambda_aux * mean_valid_dimensions(Lcorrectness, Lprogress, Lerror)
```

这是有序等级的软目标近似，并非已校准的概率。error 通道以严重程度为目标正常最小化损失，不对 loss 取负，也不直接从 actor reward 中扣除 error。

第一轮保持原 lambda_aux=0.1 作为可比较起点，不宣称这是最优权重。实际标签频率和梯度诊断不合理时，应在后续消融中调整，不通过语义不明的临时权重补救。

judge 先给事件证据，再给按锚点得到的汇总标签；程序负责 schema、证据引用、mask 和基本一致性校验，不把事件条数简单平均成质量分数。

### 7. Prompt 与 schema 实施要求

建议新增 `rubric/specs/math_process_v2.yaml`、`rubric/prompts/judge_math_process_v2.txt` 和版本化 contracts。

Prompt 包含以下顺序：任务边界 → 题目/回答/截断信息 → 可验证进展与错误判据 → 不同解法兼容 → 截断/自我修正规则 → 标签锚点 → JSON 输出合同。

- 模板使用结构占位符说明字段，不用唯一的“progress=0.5、error=unknown”示例暗示固定答案。
- 提供有区别的短例子：正确完整、正确前缀、明确错误、错误已修复、纯复述、末尾半截公式、替代有效解法。
- 示例用于解释边界，不能照搬被评题的答案或奖励某种文风。
- 正确性由公共数学 scorer 填充；judge 不覆盖它。没有最终答案可形成 0 分正确性标签，但不抹掉已有过程进展；解析歧义或判等工具异常按本文公共评分规格 mask 正确性维度，不生成伪负例。
- 输出包括 rubric/schema 版本、completion 状态、关键事件、两个过程标签及理由、无法判断原因。
- 约束 valid=false 时 value=null；错误信息与数学 unknown 分开。
- 完成状态和事件仅作辅助证据；本轮不额外训练 completion head，不增加 actor 长度奖励。
- 标注预算通过真实长回答校准，不能因输出 JSON 更复杂就直接任意增大 max_tokens。记录输入/输出 token、失败率与标注时间。

### 8. 文件级改动与版本迁移

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

### V2 必须覆盖的成对语义测试
- 同一正确轨迹在不同完整步骤处截断：前面已成立的证据不能被清空，也不能自动获得完整解答分。
- 正确前缀后加入明确错误：已成立的独立进展保留，错误目标上升。
- 在错误之后只说“已修正”：不能当成真实修复；给出有效修复才改变未修复错误状态。
- 加入重复正确步骤：进展分不应仅因长度增长而提高。
- 改写为另一种正确解法：不因偏离示例路径扣分。
- 最后半截公式：局部 unknown，不全段 mask。
- 调换 teacher/student 来源身份但保持文本：标签一致。
- 缺失 judge 结果/上下文溢出：不得变成“无错误”。

上述测试需要真实 judge 校准与程序合同测试各自完成。简单 mock 测试只能证明代码传递正确，不能证明数学标签质量。

## 附录 C：文件级基础改动清单

| 文件/目录 | 修改内容 |
|---|---|
| `sample/math_scorer.py` | 仅保留 legacy 审计行为和身份 |
| `scoring/math.py` 及必要依赖（新增） | 唯一正式数学 scorer、提取协议、MathD/SymPy 判等和状态合同 |
| `scoring/tests/`、`sample/tests/` | 公共评分/三端一致性及元信息回归测试 |
| `sample/build_gad_data.py` | 选中候选索引与来源信息绑定 |
| `gad/tools/config.py`, `extension_config.py` | 全局 scoring.math、三端统一引用、显式 seed、身份校验 |
| `gad/tools/pipeline.py`, backend patches | 公共验证 scorer 接入、seed 下发、实际配置记录、数据字段透传 |
| `gad/extensions/online_rubric.py` | 新 scorer 接口、解析状态、缓存和指标 |
| `gad/extensions/runtime.py`, `critic_batch.py` | teacher provenance、监督计权诊断 |
| `gad/extensions/worker.py` | 新增联合总损失日志；不改变 BT/actor 目标 |
| `gad/extensions/checkpoint_state.py` | scorer 版本/身份恢复检查（按现有接口最小扩展） |
| `evaluate/rescore.py`（新增或等价入口） | 无模型纯 CPU 重评分与分歧审计 |
| `evaluate/evaluate.py`, `summarize_eval.py` | 使用公共 scorer，单份正式 metrics.json、legacy 审计隔离、真实协议展示 |
| `LMOps/train.yaml` 与新实验快照 | 显式 seed、scorer、A/B/C 实验配置 |
| `LMOps/reports/rubric_audit/`（新增） | 离线审计、可复核样本和比较报告 |

## 附录 D：Judge 协议与有效监督修订计划（2026-10-15）

### D1. 本轮目的、证据与范围

目标：让真实长数学回答能够以可接受的开销获得可核查的过程标签，避免 joint 模式在学生没有过程监督时仍被误认为正常运行；同时保留可靠的局部监督。

已确认的依据：

- `reports/rubric_v2/qwen3_14b_128_20261015/`：96 条真实回答中，30 条上下文超限、50 条达到 768 输出 token 上限、14 条证据引用失败、1 条修复链接失败、1 条 JSON 非法控制字符；有效过程标签为零。当前解析器重新读取保存输出得到相同结果。
- `reports/rubric_v2/qwen3_14b_20261015/`：短样例仍有“仅声称修复被当成实际修复”和“合法未完成分支被当成错误”。结构校验通过不能证明数学判断正确。
- 当前 `JudgeClient.annotate()` 只拦截整批没有 parsed 的情况；全 unknown 的 parsed 结果仍能通过。teacher 成功也可能掩盖 student 全部失败。
- 当前 `parse_events()` 对任一事件的坏引用抛出整条异常，即使错误事件未参与一个证据完整的进展标签，也会一起丢掉该标签。

执行范围：修订输入输出协议、固定说明、证据校验、有效监督保护、日志和相关版本身份；有限 judge 验证；条件允许时完成剩余工程 smoke。原 BT、Warmup actor 目标、GAD actor 更新和 adv-only reward 保持原合同；公共答案 scorer 不改语义。replay 继续关闭，不新增 replay 实验，也不实现动态 rubric、额外 verifier 服务或步骤级 actor reward。

正式 baseline / rubric V2 重跑与全量外部评估仍由用户执行。不得停止、重启或修改其他实验使用的服务、配置快照、缓存和进程。

### D2. 三个长度限制与固定说明的口径

1. `data.max_response_length=8192` 约束训练可见回答长度；学生 rollout 的生成上限按实际下发配置核对。teacher 原始回答还可能在进入训练时被裁剪。不是 teacher 与 student 合计 8192。
2. teacher 和 student 分开标注，每次 judge 只看到一条回答、题目、参考最终答案、固定说明和可见性信息。
3. `judge.max_context_tokens` 是 judge 输入与输出的总容量；客户端按实际 judge tokenizer/chat template 检查 `input_tokens + max_output_tokens <= max_context_tokens`。它不是两条回答长度相加。
4. `judge.max_output_tokens` 是一次评分 JSON 的全部新生成 token 上限，包含事件、引用、解释和标签。当前 768 已被真实数据证明偏小。
5. “固定说明”指每次请求复用的任务说明、数学判据、标签锚点、关键反例和输出要求。当前 prompt 与 spec 有重复。此次精简是消除重复表达，不是削减数学要求，也不是让 judge 为每道题生成新 rubric。

先记录当前与修改后的输入 token 构成：固定说明、题目/参考答案、原始回答、定位包装、完整 chat 请求。各部分 token 数不必严格可加，完整 chat 请求长度才是容量检查依据。

### D3. 第一步：冻结对照与输入输出协议，先不启动模型

复用现有原始轨迹和 API 输出，不生成新 actor 回答。记录旧 prompt/spec/schema/parser 与模型、tokenizer、配置身份。保存新的请求构造样例及前后 token 长度对比。

#### 输入：紧凑定位，完整保留可见内容

- 修改 `rubric/core/process_v2.py` 的分段表示及 `rubric/core/contracts.py` 的请求构造，替换当前“包括空行在内，每行一个 JSON 对象”的高开销表示。
- 优先实现带稳定编号的紧凑文本块，例如短行前缀加原文；明确编号标记与原文本身的区分。程序维护编号到原文字符区间的映射，空行可保留为空行而不产生独立编号对象。
- 不要求按每个数学步骤切分，不引入另一个模型做分段。长行仍保留全文；跨行数学表达允许通过明确的编号范围定位，不把只能引用单行变成新的失败来源。
- 输出引用应在指定原文区间内逐字存在。表示转换必须可恢复实际可见原文，测试换行、空行、反斜杠、LaTeX、Unicode 和类似编号的原文内容。
- 不对数学表达做会改变语义的正规化，不通过模糊匹配补救幻觉引用，不裁掉开头或结尾来迎合上下文预算。区分传输包装与原文本身。
- 超限时记录实际输入 token、预留输出、上限和原因；不得把未送入 judge 的内容当作已检查。在线与校准必须共用同一构造和解析入口。

#### 输出：关键证据，减少重复文本

- 继续保留 completion、analysis_complete、关键事件、两个过程标签及必要证据关系；不新增训练 head。
- 只输出支持判定的关键事件，不逐步复述整个推导，不为每一句正确计算建立事件。当前最多 8 个事件可保留，但不要求凑满。
- 每个事件保留类型、稳定 ID、原文位置、短而足够定位的 quote、简短数学理由，以及需要时的错误/修复关联。标签层理由不重复事件中的长篇解释。
- 预算不足以可靠判断时允许 unknown。不能为满足 token 数而截断生成出的 JSON，也不能删除关键错误来减少事件数。
- 使用当前 vLLM/xgrammar 实际支持的 schema 特性；不要重新引入此前被拒绝的 `maxItems` 等约束。需要由 parser 执行的限制应有明确失败类型。
- wire schema 如有改动，使用明确的新 schema 修订及输入协议版本；训练侧仍消费现有两个过程维度。不得在旧 schema 名下静默改变字段含义。

交付：一套实际选定的紧凑协议及无模型 token 审计，不实现多个长期并存的协议分支。

### D4. 第二步：整理固定说明，保留数学规则并加强两个反例

现有文件职责调整为：

- `rubric/specs/math_process_v2.yaml`：维度含义、0/0.5/1/unknown 锚点、数学判据与边界规则的权威来源。
- `rubric/prompts/judge_math_process_v2.txt`：任务说明、如何检查证据、少量成对反例、如何输出的程序性说明。
- `judge_messages()`：确定性构造一次完整说明；同一规则不再在 prompt 和附加 spec 中长篇重复。实际发送的说明及其 hash 可审计。

实施前列出“旧规则 → 新位置”清单，至少覆盖：目标相关的正确进展、错误前独立进展、错误前提污染、关键与局部错误、定义域/运算合法性、分支与穷尽性、不同有效方法兼容、修复、截断、局部 unknown、空回答与复述、参考答案不能反推过程正确。

重点加入以下成对短例，替代部分重复叙述，而非无限加长 prompt：

1. **合法未完成分支 vs 非法声称穷尽**：明确假设 `x≠0` 后在该分支求解、尚未处理 `x=0`，可以是有效进展且无已发生错误；未经条件直接除以 x 并声称找到全部解，是明确错误。
2. **声称修复 vs 实际修复**：仅写“已经改正”不解除此前错误；可见原文给出正确替代并恢复相关推理，才支持 corrected_error。
3. **完整检查可见前缀 vs 已完成整题**：正确前缀可 `completion=prefix` 且 `analysis_complete=true`。后者表示充分检查当前可见材料，不表示整题已经解完。

只允许 judge 引用原文中实际发生的修复；reason 中由 judge 自己算出的正确答案不能充当轨迹的修复证据。不要用关键词正则宣称已解决数学真实性问题；这是需要真实 judge 与人工样例检查的语义要求。

本轮不改变 SRaR / 数学 RRM 的参考方向。论文与仓库仍见附录 B，无需再创建一套研究计划。

### D5. 第三步：按事件依赖保留可靠监督

修改 `parse_events()`，将“整条不可读”与“局部事件不可信”分开：

- JSON 不完整、根 schema/版本不匹配、无法建立可靠事件身份等整体错误：整条过程标注失败，保留原始输出与错误原因。重复事件 ID 不能悄悄覆盖。
- 单个引用位置/文本不匹配或事件链接不合法：该事件无效；记录具体原因，传播到依赖该事件的修复关系和标签。
- 证据完整、与坏事件无关的独立进展可以保留。不能靠删除坏事件直接提升标签值，也不能把原本的 unknown 自动改为有分数。
- `progress=1` 属于完整解答判断，不能仅因还剩一个正确中间步骤而保留；依赖完整性或关键证据受损时必须 mask。
- `error=0` 属于覆盖整个可见数学内容的否定结论。若丢失事件可能包含错误、修复或必要检查信息，保守 mask；不能先删除疑似错误再宣称无错误。首版可对任何事件校验失败统一 mask 此零错误结论。
- 明确、证据完整的未修复严重错误可在其他局部 unknown 时保留；但如果其关联修复事件不可信，且可能影响“仍未修复”的判断，相关错误标签也应 mask，不能把“修复无法验证”直接等同“数学上未修复”。
- 非零分数必须有与其语义一致的有效证据。尤其不得仅凭一个剩余普通进展事件保住原始满分。

统一返回：原始输出、原始标签、事件有效性与原因、最终有效标签、逐维 mask 原因。校准报告同时展示 raw 与 effective；unknown/mask 不能计入“判断正确”。

### D6. 第四步：训练保护按最终有效学生标签执行

保护应放在 `online_v2.annotate()` 已合并新结果与缓存、生成最终 masks 之后，优化器更新之前；不能只依赖 `JudgeClient` 是否返回 parsed。

规则：

1. 保留现有服务不可用及整批技术失败的终止保护。任何终止前保存已得到的结果、token 使用和失败明细；当前全失败异常中的 results 也必须由在线路径落盘，不能仅校准脚本保存。
2. 分 teacher/student、分进展/错误维度统计去重后的有效数量和覆盖率；单独记录“任一过程维有效”和“两维均有效”。答案正确性标签不计入过程监督覆盖。
3. 同时展示全部所选学生样本与具备完整题目的可标注样本数量，不能靠改变分母美化覆盖率。replay 开启时额外区分当前与历史学生；本轮不启动 replay 实验。
4. 在 joint 模式下，若存在可标注学生样本而其过程标注全部因上下文、JSON、证据合同等技术原因不可用，则本步停止，不能由 teacher 成功或 correctness 有效掩盖。
5. 对合法的数学 unknown，不强行生成标签：首版建议连续 3 个 critic 步骤没有任何有效学生过程标签时停止；前两个空覆盖步骤明确告警并记录。计数取最终标签，包括缓存；有任一有效学生过程维才重置。没有可标注学生的异常批次另记原因，也不能永久绕过空覆盖检测。
6. 一维有监督、另一维 unknown 不等于整批失败，保留可靠一维。持续单维退化必须在日志中可见，但本轮不任意设置高覆盖率阈值或强迫困难题得到确定标签。
7. `correctness_only` 与 rubric 关闭模式跳过过程覆盖保护；不得为了让 joint 跑起来自动切到 correctness_only。

配置应最小化：增加连续空学生过程监督的上限配置即可，默认建议 3；服务失败和全技术失败继续明确停止。不增加自动降级、自动换模型或自动改变阈值机制。

计数器纳入同阶段 checkpoint 保存恢复；Warmup → GAD 阶段初始化策略明确记录，建议新阶段从零计数。保存恢复后不能绕过保护。现有 V1 `empty_student_steps` 不应未经语义区分直接复用为 V2 新计数。

### D7. 第五步：选择预算并用固定样本有界验证

> 历史执行配置：本节 non-thinking/1536 设置已完成本轮协议实验，不再作为下一轮候选。下一轮使用附录 E 的 thinking 验证安排。

#### 先做无模型容量审计

- 使用现有冻结 128 条输入，按新构造方式和实际 judge tokenizer 计算完整输入长度；报告 p50/p90/p95/max、超限数量，以及与旧协议的差异。
- 候选起点：14B、TP=2、non-thinking、temperature=0、seed=42；context=32768、output=1536。先检查服务加载与少量长请求，不能将此配置写成已经通过的正式默认值。
- 初始并发 1；仅资源验证允许后使用 2。当前正式默认并发 8 未在这套长上下文配置下验证，不直接沿用，也不直接用并发 2 的吞吐估计并发 8。
- 如精简后仍有明确输出截断证据，可在唯一一次受影响重测中改为 output=2048。不是同时跑 1536/2048/更多组合的网格；增加输出会减少输入余量，应同步重算。
- 32k 是否足够以实际输入为准，且不能证明未来所有 on-policy 轨迹均适配。线上每次继续做容量检查。显存不足时停止该候选、保存证据；不裁短文本、不量化换模型、不抢占其他 GPU 以换取通过。

#### 真实 judge 小样本：最多 24 条首轮、总计 48 次应用请求

从冻结集合预先选定 16 条真实回答与 8 条边界样例。真实回答包含 teacher/student、长短、完整/截断，必须包含输入最长或旧协议超限的样本；8 条边界涵盖正确前缀、实际修复、假修复、合法未完成分支、非法声称穷尽、半截公式、复述及完整替代解法。记录选择规则和 IDs，不根据新输出重新挑题。

首轮各一次；只有定位明确问题并修复后，重测受影响样例一次，总计最多 48 次应用请求。V2 格式/证据失败不做无变化重试；传输重试至多一次，单独统计 HTTP 尝试数且受同一时间预算约束。8B 不再自动平行重测，不扩展到全 128 条重新标注。

报告至少包括：

- 实际发送数、本地超限、输出截断、整条 JSON/事件身份失败、局部证据失败；不要仅给“JSON 成功率”。
- 原始/最终标签分布，分角色、长度组、截断状态的逐维覆盖率；unknown 与程序丢弃分开。
- prompt/output token、单请求与整批耗时、GPU 峰值显存、缓存命中、所有尝试与停止原因。
- 假修复与合法分支是否还出现错误的确定标签；结构通过、预设锚点一致、人工数学审查分别报告。真实标签的人工审查仍可 pending，不能输出伪人工准确率。

此小样本只验证协议可行性与已知反例，不足以证明总体数学准确率。不得把 zero coverage、持续长输出截断、已知关键语义反例未修复的情况交付为“正式 joint 已就绪”。CPU/工程交付可完成，同时明确标注质量或长序列容量尚未通过；禁止循环调 prompt 直到样例全对。

### D8. 第六步：工程验证与标签质量验收分开记录

- 先跑必要 CPU 回归，不必等待真实 judge 完美校准。原先通过的 scorer 和基础修复仅做受影响回归，不重做历史审计。
- 剩余真实训练 smoke 沿用前文边界：V2 Warmup 最多 2 步 → GAD 最多 2 步 → 同阶段恢复最多 1 步。已完成 baseline 一步不重复。实际 actor/critic 更新次数、handoff、恢复计数和损失应从产物核对，不能硬编码“已完成 6 步”。
- judge 尚未就绪时，不放宽生产保护启动真实 joint。可先用纯 CPU 的受控标签/合成 batch 验证计权、mask、梯度和状态序列化。
- 如必须用受控标签执行 GPU 接线 smoke，入口必须显式标记 engineering-only，标签与本次精确 token/response ID 对应，使用独立缓存、checkpoint 和 run 身份；训练恢复不得把它当成真实 judge 数据，产物不可用于正式初始化。没有必要则不新增此 GPU 分支，只交付 CPU 通过和真实 smoke 未完成。
- 正式 judge 部署、原始 API 捕获和在线调用路径仍需真实检查；受控标签不能替代它们，也不能用于宣称 rubric 有效。

本轮后续执行的真实 judge 与 GPU 训练验证合计仍以 **30 分钟墙钟时间** 为上限，包含服务加载、传输重试和恢复；起止时间与每段耗时落盘。已有报告不覆盖、不重置以伪装本轮开销。正常训练上限为剩余 5 个全局步，失败后最多额外 2 步，仅重试受影响阶段一次且不增加时间预算。预算不足则停止本轮自建任务并交付未验证项，不要求为满足清单启动额外实验。

### D9. 文件级修改与有意义的验收测试

| 文件/位置 | 本轮改动 |
|---|---|
| `rubric/core/process_v2.py` | 紧凑定位、schema 修订、事件有效性及标签依赖校验 |
| `rubric/core/contracts.py` | 单一请求构造、说明渲染、输入协议身份 |
| `rubric/prompts/judge_math_process_v2.txt`、`rubric/specs/math_process_v2.yaml` | 规则去重、关键反例、覆盖与完成状态区分 |
| `gad/extensions/judge_client.py` | 容量/输出截断分类、失败产物、有限重试；保留真实服务身份校验 |
| `gad/extensions/online_v2.py` | 最终 masks 后的学生覆盖保护、缓存与新标注共同统计 |
| `gad/extensions/online_rubric.py`、`checkpoint_state.py` | 新协议/cache 身份、空覆盖计数器与恢复兼容性 |
| `gad/tools/config.py`、`extension_config.py`、配置覆盖文件 | 候选预算、连续空覆盖上限、参数校验和身份下发 |
| `gad/tools/calibrate_rubric_judge.py`、`validate_rubric_v2.py` | 共用新协议、原始/有效标签分开统计、有界请求和真实步数统计 |
| `gad/tests/`、`scoring/tests/` 中相关测试 | 补足失败传播、覆盖保护、版本与恢复；复用现有数学 scorer/归一化测试 |
| 新的 `reports/rubric_v2/` 子目录 | 冻结样本、前后 token 审计、协议、原始结果、审阅状态和交接记录 |

必须覆盖的程序行为：

1. 新表示精确保留原文；跨行引用与空行可定位；包装中的伪编号不篡改事件归属。
2. 输入+预留输出边界正确；输出截断不能产生半个可用 JSON；所有失败可追溯。
3. 无关坏事件不清空独立有效进展；坏修复/关键证据影响依赖标签；不能因删除事件把错误标签变好。
4. JSON 可解析但学生两维全 unknown、teacher 成功而 student 全技术失败、部分有效维度、缓存命中和混合新旧结果，均走正确保护路径。
5. correctness-only / rubric=false 不受过程覆盖规则误伤；计数器保存恢复前后行为一致。
6. 不同 protocol/prompt/spec/parser/judge 预算身份不能命中旧标签；旧失败或旧未知缓存不能阻止新协议重新标注。禁止删除用户旧缓存，以新 namespace/身份隔离。
7. 保持原 BT 与 adv-only、原辅助归一化合同。事件序号/引用表示变化不需要改变 critic 头布局或引入新的 actor reward。

版本策略：输入、schema、prompt、spec、parser、模型/tokenizer、解码预算均进入可追溯身份；程序解析含义变化也必须使旧缓存失效。协议变更前后的训练 checkpoint 默认不允许无提示地同阶段续跑；正式新实验使用新 run。工程 smoke 和已有实验目录保持隔离。

### D10. 交付与执行顺序摘要

按以下单一路径执行，不再重跑初版的所有步骤：

1. 冻结现有证据和样本，做 token 审计。
2. 紧凑输入/输出协议 → 固定说明去重与反例 → 事件依赖校验。
3. 最终学生覆盖保护 → 失败日志 → 缓存/checkpoint 身份 → CPU 测试。
4. 选定一套候选资源预算，进行最多 24 条首轮真实 judge 验证及一次有界受影响重测。
5. 在剩余预算和质量条件允许时完成真实 Warmup → GAD → 恢复；否则明确停在工程或标签质量哪一层。
6. 更新交接报告，给用户准备新配置与 dry-run 命令，停止新增实验。正式 baseline / rubric V2 由用户重跑。

交接报告分别列出「代码已修复」「CPU 已验证」「真实 judge 协议已验证」「数学标签人工审查状态」「真实训练/恢复已验证」「未覆盖」。不能把其中一项的通过替代另一项；无需为证明 actor 提升追加训练。

## 附录 E：Thinking Judge 监督能力验证与 8B 候选对照（2026-10-15）

### E1. 优先级和角色边界

用户确认：速度确实重要，但首先要验证 judge 是否能提供有效监督。先以 14B thinking 检查数学审核能力，再考虑 8B thinking 是否能保留所需质量。不能为缩短延迟而默认关闭 judge thinking，也不能把 14B non-thinking 的失败解释为 14B 的能力上限。

- Actor 保持用户当前 non-thinking rollout 配置。
- Critic 继续对轨迹作前向评分，使用 BT 与辅助头训练；当前不生成 CoT，无需给它增加生成型 thinking 流程。
- Judge 独立设置 `rubric.judge.enable_thinking=true`，不继承 `rollout.enable_thinking`。允许推理后输出最终评分 JSON。
- Judge 推理文本仅作诊断，不输入 critic、不充当被评轨迹的数学证据，不进入 actor reward。
- 本轮不修改 BT、actor 目标、辅助维度及计权，也不启动正式训练或 replay 实验。

### E2. 已知证据和本轮要回答的问题

现有紧凑协议在冻结 96 条真实回答中最大输入为 12197 token；最终 24 条均完整输出 JSON，真实回答最长输出 1081 token。继续加大 non-thinking 输出预算不是当前优先事项。

不过 16 条真实回答的两维有效监督仍为零，且已发现数学误判：

1. `real_900e7927fa39787bac81`：只许插入一对括号，却认可改变运算符等违规推导，将答案 721 判为完整正确。
2. `real_0440d331986484988ada`：将错误上游推导出的 x=4、y=6 当成正确，反而因参考答案为 40，把正确算式 4×6=24 判错。
3. `boundary_false_repair_0`：把仅声称修复标成实际修复。
4. `boundary_partial_formula_0`：把尚未写完的 x=12/ 判为正确进展，并替原文补出了除数；最终标签恰好符合锚点，事件语义却错。
5. 复述当成数学进展、错误前提下的后续运算当成可靠进展等已有反例。

需要分别回答：开启 thinking 后，约束检查、错误定位、修复辨识及局部未知判断是否改善；证据定位是否可靠；最终标签能否通过现有保守校验进入训练。不能只报告 JSON 成功率或与参考最终答案的一致性。

### E3. 先实现独立 thinking 路径，不同时大改 rubric

先检查 `gad/extensions/judge_client.py`、`gad/tools/rubric_judge.py`、`rubric_judge_server.py`、`extension_config.py`、配置校验与校准脚本。

1. 将请求与本地 token 计算里写死的 `enable_thinking=False` 改为读取 judge 专属配置；二者采用同一模型模板和开关。
2. 服务端按当前实际 vLLM 版本配置 reasoning parser；核查 reasoning 与 guided JSON 的兼容路径，验证 JSON 约束仅作用于最终答案，而不会从首 token 起阻止正常 thinking。
3. API 返回的推理与最终 JSON 分开记录；parser 只消费最终 `content`。先用少量固定请求确认真实产生了推理，并完成推理到最终 JSON 的转换，不能仅检查配置值为 true。
4. 当前 prompt 中“无额外思维链／仅输出 JSON”的表述应明确约束最终答案区域，避免与 thinking 开关冲突。首轮保持数学判据、标签锚点、输入表示、事件 schema 和原文引用校验不变；只作模式适配所必需的说明修订，记录差异。
5. thinking、reasoning parser、服务版本、采样参数、总生成预算进入服务及缓存身份；新建配置覆盖文件、端口、缓存和报告目录。旧 non-thinking 产物保留。
6. 若当前 vLLM 0.8.5 无法可靠支持该组合，明确定位并交付兼容性阻塞；不得静默退回 non-thinking 或在仍运行训练的共享环境中升级依赖。需要新环境或替代输出路径时另列实施差异，保持现有实验运行。

参考官方说明：

- [Qwen thinking 开关和推荐采样参数](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html)。
- [Qwen 的 vLLM 部署说明](https://qwen.readthedocs.io/en/stable/deployment/vllm.html)。
- [vLLM 0.8.5 reasoning outputs 与最终内容分离](https://docs.vllm.ai/en/v0.8.5/features/reasoning_outputs.html)。

文档有版本差异；执行模型须核对本机实现，不将当前文档中的新参数直接当成 0.8.5 支持的参数。

### E4. 14B thinking 候选配置与预算

- 首个候选仍为已下载的 Qwen3-14B、BF16、TP=2；thinking=true；seed=42；并发先为 1。
- 采样采用 Qwen thinking 推荐起点：temperature=0.6、top_p=0.95、top_k=20、min_p=0。不要继续写死原有 top_p=1、top_k=-1 或 greedy temperature=0。
- context=32768 为候选；总生成预算先用 8192 token，容纳推理与最终 JSON。它是实验起点，不是已证实足够或最优的值。每条请求继续以实际 thinking 模板计算输入与预留输出容量。
- API 的总 `max_tokens` 不等于“独立推理额度 + 独立 JSON 额度”。除非后端实际支持并验证了分阶段预算，否则不得声称已经保证给 JSON 留下 1536 token。推理耗尽预算、未生成最终 JSON时应记录为预算失败。
- 单独记录推理、最终 JSON、总输出的长度及实际 token 计数来源；后端不提供分项 token 用量时，注明本地重分词估计，不伪称服务精确计费值。
- 单请求超时根据 thinking 路径调整为显式候选值，避免沿用 non-thinking 的短超时把正常推理中止。首个候选可用 300 秒，仍受本轮总墙钟预算约束；超时不判为数学错误。
- 不以“thinking 输出更长”为质量提升证据；关注错误定位与标签变化。固定 seed 不代表与旧 greedy 实验随机过程相同，此对照是两种配置的比较，不能把所有变化严格归因于单独的 CoT 开关。

### E5. 首轮固定样本、审查与停止条件

复用 `judge_protocol_v21_20261015/selected24.jsonl` 中的 16 条真实回答和 8 条边界题，不生成新轨迹。优先处理 E2 的关键反例，再处理其余冻结样本，记录固定顺序。少量接线探针也从该集合选取并计入总请求预算。

在查看新输出前整理语义审查表：题目约束、关键正确进展、明确错误及其位置、是否实际修复、允许的标签/unknown、证据片段。人工未审时标为模型辅助，不把旧 judge 输出或最终答案 scorer 当作过程真值。

每条至少审查以下三层，分别报告：

1. **事件语义**：没有虚构原文步骤、错误类型/位置正确、修复发生在原文而非 judge 自己的推理中。
2. **汇总标签**：progress/error 与事件、可见完成状态及错误影响一致。最终标签相同但理由/事件错误不能算通过。
3. **可训练性**：经过真实证据与依赖校验后，哪些维度有效、哪些被 mask。数学 unknown、技术失败和确定误判分开。

对 critical 反例不能靠 unknown 算“判对”，但可靠弃判与错误的确定标签应区分；前者可能适用于选择性监督，后者有污染 critic 的风险。误判严重程度依其是否影响当前关键结论，而不是修改起来是否容易。

若思考后数学判断改善但逐字引用仍造成零覆盖，应明确记录“数学审核改善、生产标签仍不可用”。此时再安排下一阶段证据协议调整：可考虑由 judge 选择稳定片段 ID、程序自动提取原文，但需审查被选片段是否支持事件，不能用存在合法 ID 替代数学证据。首轮不同时更换这套协议，以免无法解释变化。

已知关键数学误判持续出现时，停止扩大标注与真实 joint；先分析推理和事件依据。不得放宽 parser、缩小困难样本比例或自动换成正确性监督来获得“通过”。

### E6. 8B thinking 是后续候选，不以参数量或与 14B 一致性决定

可以换回 Qwen3-8B，但前提是它本身能给出可用的监督。14B thinking 若显示足够的数学审核能力，8B thinking 才进入后续候选对照；14B 还未解决核心误判时，先定位问题，不自动开启多模型搜索。

- 用同一冻结集合、题目/原文、数学判据、输入输出协议、thinking 采样参数和总生成上限；只切换模型及必要的 tokenizer/服务身份。初次对照保持 BF16、TP=2、并发 1，减少额外部署变量。
- 真值依据是独立审查的数学事件与标签，14B 输出不是金标准。两者一致也可能同时错；必须审查分歧及共同错误。
- 按角色、长短、完整/截断分组比较：确定错误标签数量与类型、正确进展及严重错误识别、真实/假修复、错误前提污染、逐维有效覆盖、弃判与技术失败。
- 小集合只用于候选筛选，不足以宣称 8B 与 14B 等价。8B 若重新出现关键误判，不能用更高覆盖率或更快速度抵消。人工尚未审查时，结论限定为模型辅助初筛。
- 质量相近且达到所需监督标准后，再比较每条真实回答的输入/推理/JSON token、延迟、显存，以及每单位时间获得的有效且经审查可靠的标签数。8B 可能生成更长推理或重试更多，因此不能预设一定更快。
- TP=1、提高并发、缩短推理预算等部署优化属于后续阶段，先不与模型能力比较混在一起。正式在线吞吐还要考虑冷缓存 teacher+student 标注量，不能仅用短边界题延迟估算。
- 更换 judge 使用新实验与标签身份，不静默混用 8B/14B 标签池或接续改变监督源的正式 run。

### E7. 工作量与交付边界

本次用户只要求写入计划；当前不执行代码修改、模型启动或实验。

后续执行本轮计划时，默认仍保留一次有界验证窗口：真实 judge 最多 48 次应用层调用，接线探针和重测全部计入。优先完成最多 24 条 14B thinking；若通过能力初筛且预算允许，剩余额度用于同集合的 8B thinking。若修复重测占用了额度或时间不足，8B 留待后续，不为凑齐对照超额。非传输错误不做无变化重试；HTTP 重试另记且受同一时间窗口约束。

初始 GPU 窗口仍以 30 分钟墙钟时间作为有界停止点，包含加载与退出。**这不是要求在 30 分钟内证明 judge 有效，也不是据此缩减 thinking 额度或跳过困难样例。** 到点保存已完成结果并列出未完成项；后续预算由用户决定，不自动连续重启窗口。实际模型推理时间可能使 24 条无法完成，应如实交付。若输出预算或服务兼容性不足，报告阻塞而不是宣称模型数学能力失败。

本轮 GPU 预算优先用于 judge 能力，不为了完成旧清单挤入 Warmup → GAD → 恢复。真实训练验证继续列为待完成，待监督质量和标注可用性成立后再安排；用户负责正式重跑的分工不变。

交付必须包含：

- 独立 judge thinking 开关、实际服务参数与模式验证证据；actor 设置保持原值。
- 原始推理/最终 JSON、事件与汇总标签审查、raw/effective 覆盖、失败类型、预算与模型身份。
- 与现有 non-thinking 结果的逐例差异，说明采样配置也变化，避免不成立的因果结论。
- 14B 是否值得进入下一步；8B 是否已测、是否仅候选、哪些数学能力或吞吐尚未验证。
- CPU 接线/身份/预算测试结果；不以测试通过替代真实语义校准。

最终顺序为：**独立 thinking 接线 → 14B thinking 数学监督检查 → 必要时另行修订证据协议 → 条件满足后 8B thinking 同题对照 → 质量优先选模型 → 在线吞吐与剩余训练验证。** 不重复附录 D 已完成的全套工程改造，也不启动正式训练。

## 附录 F：片段证据简化与 8B Thinking 实测（2026-10-15）

### F1. 本轮授权与问题定位

用户在14B thinking冻结24条补完后明确要求：先简化过程监督协议、将修改计入文档，并开展Qwen3-8B测试；正式实验由用户启动。
本授权允许直接开展8B候选测试，不再等待E6的14B初筛通过条件。E3/E5的首轮协议冻结要求已经履行，
本附录属于明确记录的新协议阶段；保留旧阶段产物，不把模型或协议变化包装为单独thinking开关的因果对照。

补完中真实16条两维有效覆盖均为0，其中5条模型辅助检查正确的简单回答原始评分合理，却因手抄公式、
行定位或失效传播被mask。该结果支持减少标注负担；它不证明模型数学审核全可靠，也不是actor无收益的证据。

### F2. 简化内容与数学边界

1. 输入升级为`rubric.input.v2.2`，输出为`rubric.events.v2.3`；rubric维度与`math_process_v2`不变。
2. 程序按原文段落提供`s1`等片段ID；通常在空行断开，超过12物理行或1200字符时在行边界再切分。
   单行超过1200字符时保留整行，避免改写公式。空白、换行、LaTeX和Unicode全部保留；输入带长度和内容hash。
3. Judge事件只选择`span_id`，不输出quote或loc；程序按ID回填原文、字符偏移及物理行范围用于审计。
   片段存在仅证明引用可定位，不代表片段支持该数学判断。
4. 最多4个事件槽位，通常1至3个关键事件；不要求逐句列举。仍检查完整可见回答，不得隐去关键错误。
5. 局部progress=.5在部分引用失效时可保留独立有效支持，明确记录过滤的依赖；依赖坏事件的进展继续mask。
   不自动把缺证据的满进展降为.5，不猜测数学分值。
6. 全局progress=1或error=0不再因无关neutral/unjudgeable事件失败而整体清空。
   满进展仍要求completion=complete与analysis_complete；被引用的数学证据失效、未知错误或未知修复仍阻止全局认证。
   error=0仍要求可见数学内容检查充分；已观察到但未修复的错误不能靠省略标签引用来绕过。
7. 保留错误前提污染、合法未完成分支、真实/假修复、半截尾部、空数学unknown等原有数学标准。
   ID、JSON、局部依赖校验不替代数学正确性审核。
8. 不修改BT、辅助头布局/损失计权、actor目标或actor non-thinking；保留零学生过程覆盖保护。

### F3. 接入、隔离与验证

活动parser、prompt、spec、judge消息构造与在线缓存同时更新；缓存使用新wire版本namespace，
服务/标签/checkpoint身份包含协议和代码hash。旧协议结果只作历史证据，不静默用于新协议同阶段恢复。
新backend由现有代码指纹生成，不修改已使用的backend；新正式实验应新建run。
在线标签保存程序回填的source_evidence、原始judge事件和逐维mask，便于追查。

CPU验证覆盖无损恢复/同长度篡改、未知片段、原始公式回填、局部独立支持、错误依赖传播、
无关辅助事件局部失效、全局关键错误保护、客户端thinking/生成schema、缓存/恢复隔离和在线targets/masks。
这些验证不替代真实GPU训练或人工数学标签验收。

### F4. 8B测试与工作量边界

- 使用本地Qwen3-8B、BF16、TP2、GPU6/7、独立8021端口、thinking=true、并发1。
  temperature=.6、top_p=.95、top_k=20、min_p=0、seed42；context32768、总生成8192、单请求300秒。
  推理与最终JSON分开记录，不以推理非空冒充数学正确。
- 复用原冻结24条，16条真实与8条边界，保留困难题、截断题及5条简单正确对照，不重新生成actor轨迹。
- 最多24次新应用/评分HTTP尝试，校准关闭重试；一个30分钟GPU窗口包含加载、请求与退出。
  单条模式/预算失败保留并继续固定集合；请求拒绝或连续4次传输失败停止。到点保存partial，不自动重置预算或换模型。
- 先保存模型辅助审查清单，再看新输出。既报告原始/有效覆盖，也复核错误前提、修复关系和关键反例；
  unknown/mask不算判对，不把合法片段ID当作数学合格。人工未审核保持pending，不发布人工准确率。
- 与旧14B协议结果比较时明确模型、输入/输出协议和校验逻辑均发生变化，不能据此宣称8B与14B等价或更强。
- 本轮不启动正式训练、actor评测或replay。交付配置、小规模judge结果和可执行dry-run接口，正式训练由用户启动。

候选覆盖文件：`configs/overrides/rubric_v23_8b_thinking.yaml`。
实施与测试记录：`reports/rubric_v2/simplified_8b_20261015/`；完成后以该目录报告记录实际数量、预算和未覆盖项。

### F5. 本轮中止交付（2026-10-15）

用户要求暂时停止实验后，已停止自有judge并释放GPU6/7；不追加请求或训练。
片段协议及局部失效简化已接入，48项CPU验证及训练入口dry-run通过。
实际9次请求：8条完整thinking/JSON（真实student 2条、边界6条），第9条随停止中断，15条未运行。
总窗口约8分1秒；8条17个片段引用均可准确回填，但存在片段与数学理由不匹配。
真实两维覆盖各1/2，唯一有效的真实回答仍把原文721冒充已修复为77，不能将覆盖增加视为可靠监督增加。
5条简单正确对照、全部teacher尚未实测。人工pending，正式训练0步，actor效果及真实联合训练/恢复未验证。
详见`reports/rubric_v2/simplified_8b_20261015/report.md`；当前停在此交付点，正式实验仍由用户启动。

## 附录 G：代码核查修复交付（2026-10-15）

依据用户新增的`docs/核查问题修复计划.md`核查修复，不恢复此前停止的实验。
活动协议升为input v2.3/wire v2.4，非空物理行片段使错误/替代可分开选择；修复事件使用必填error_event_id枚举，
同片段不能自证修复，未知数学不能支持全局满分/无错，普通neutral保留局部处理。
parser仅保证结构、定位与依赖一致，返回semantic_entailment=not_checked，不声称已验证数学蕴含。

答案抽取、invalid评估口径、legacy scorer选择、配置类型、阶段n、验证完成状态/计数/禁用、epoch导出和teacher截断已修复。
scorer指纹变化，旧标签及旧unified run不能静默续用；新实验使用新身份，历史记录不覆盖。
26套件290项CPU验证通过，无skip；同一1800条冻结回答已知正确493→497，仍66条invalid，不发布完整正式准确率。
5个未使用、无引用backend先备份后清理，3个有历史引用的保留；关键源码已暂存，未提交/推送。

详细确认、例外与证据见`reports/code_audit_fixes_20261015/report.md`。
新协议真实8B生成、真实联合训练/恢复、actor收益及人工准确率仍未覆盖，正式实验继续由用户启动。
