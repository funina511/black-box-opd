# Rubric 评分修复与实验重做计划

日期：2026-10-14。状态：待实现；本文件不代表已完成修复或已启动实验。

后续决策（2026-10-14）：用户已选择 SRaR 正向步骤／错误事件 + 数学 RRM 具体判据与不同解法兼容原则。过程监督修改以配套文档 [Rubric过程监督V2设计与参考实现计划.md](Rubric过程监督V2设计与参考实现计划.md) 为准；其中提供论文、官方仓库、提示词位置、改编边界和验收要求。

评分决策（2026-10-14）：新实验统一使用一个以 OPRD 数学评分实现为基础的公共 scorer，供 rubric 正确性标签、训练验证和外部评估共同调用。旧 CARD 仅用于历史审计。本决策取代此前“新增 answer_line_v2 字符串评分，并与 OPRD 长期并列”的方案。

## 1. 目标与范围

修复已确认的评分、实验可比性和数据元信息问题，建立可追溯的诊断指标，再重新开展 baseline / rubric 对照实验。

按用户最新要求，不以接续当前 17k 实验、复用其 checkpoint 或兼容其训练状态为实施约束。正式实验使用新的运行目录、配置快照和标注缓存，从相同初始模型重新开始。已有实验保留作历史证据，不覆盖结果。本计划不包含停止任何运行进程的操作。

本轮保持以下算法约定：

- 原有 teacher > student BT 目标不变。
- Warmup actor 的原有训练目标、GAD actor 更新方式和更新顺序不变。
- rubric 从 Warmup 开始并延续到 GAD；四通道 critic，actor 只使用 adv 通道。
- 本轮主要对照关闭 replay，不同时研究 replay 的算法收益。
- rubric prompt、固定准则、截断过程监督及辅助目标语义按配套 V2 计划修改；judge 模型第一轮保持不变，以便隔离变量。
- 不在本计划中加入“按 rubric 改写 BT 排序”“rubric 直接进入 actor reward”或 evo-rubric。

## 2. 已确认事实与尚待验证的假设

### 2.1 已确认的问题

1. **训练 rollout 种子未对齐。** 旧 5k 实验的 Warmup/GAD Hydra 快照中没有 `actor_rollout_ref.rollout.seed`，vLLM 引擎使用 `config.get("seed", 0)`；rubric 5k 显式传入 42。旧配置的 `reward_model.arena.seed=42` 不等于 actor rollout seed。两组外部评估 seed 都是 42。
2. **CARD 最终答案正则会跨行误提取。** `Answer\s*:\s*([^\n]+)` 中冒号后的 `\s*` 可跨过换行，遇到 `Final Answer:\nAnswer: 42` 会提取 `Answer: 42`，归一化后错判。
3. **正确性辅助标签直接复用上述评分器。** 错误提取不是 `[INVALID]` 时，当前逻辑可能将其作为有效的 0 分标签训练 critic。外部评估中已确认该模式，训练池中受影响的数量必须另行审计。
4. **评分依赖配置不明确。** `extension_config.py` 用 `validation.scorer_path` 隐式设置 rubric scorer。新方案应显式声明一个全局数学 scorer，三处引用同一身份，避免修改局部路径时无意改变训练监督或造成口径分叉。
5. **两种评估口径差异容易被误读。** `metrics.json` 是 OPRD 数学评分；`card_metrics.json` 是旧 CARD 兼容评分。同一批回答，答案提取和判等规则不同。
6. **教师来源截断信息固定取第 0 个回答。** `runtime.py` 读取采样元信息的 `[0]`，但数据构建支持选择后续回答。当前 17k 训练数据全部选第 0 个，不代表此实现对其他选择策略正确。
7. **部分诊断指标口径不一致。** `unknown_reason` 主要统计本步新标注，缓存命中时跳过；截断计数按 pair 出现次数，`unique_*` 按唯一回答统计，GAD 的重复 teacher 会使两者分母不一致。
8. **critic/d_loss 不是联合损失。** 当前只记录 BT，容易把 rubric 的作用看漏。需要新增总损失，而非更改历史字段含义。

### 2.2 已观察到的现象，不应直接写成因果结论

- 5k rubric 在 OPRD 外部评估上 MATH-500 为 39.7%，旧基线 44.3%；DAPO 验证集反而上升。不能据此断言 rubric 全面有效或全面无效。
- 5k 缓存中 19,917 条标注的 condition_coverage：unknown 14,459；1 分 5,312；0.5 分 93；0 分 53。有效标签中约 97.3% 为满分。缓存条目分布不等于优化时的有效权重分布。
- 过程标签覆盖率偏低、标签与 evidence 可能不一致、输出示例可能产生提示偏置，仍需人工抽查和后续 prompt 专项实验。
- 辅助损失下降不证明 adv 更好；共享主干上的 BT / rubric 目标冲突目前是待测假设。

## 3. 优先级与交付顺序

| 优先级 | 工作 | 完成后能回答的问题 |
|---|---|---|
| P0-A | 公共数学 scorer、答案解析修复与统一配置 | 正确性标签和评估是否基于同一可靠实现 |
| P0-B | 旧回答与标签离线审计、新口径复算 | 已有下降中哪些是评分现象，哪些仍然存在 |
| P0-C | 统一实际 rollout seed 和实验身份记录 | baseline / rubric 是否只有预期变量不同 |
| P1-A | 教师选择元信息修复 | rubric mask 是否作用于正确的 teacher 回答 |
| P1-B | 监督、联合损失与 adv 诊断 | 辅助任务实际学到了什么、是否传递到奖励 |
| P2 | CPU 合同测试、少量 GPU 验证、新对照实验 | 实现正确且实验具有可解释性 |

先完成 P0/P1 和配套 V2 的标签校准，再启动新的正式训练。prompt、截断语义和四通道布局必须一起版本化，不能在旧语义上仅替换提示词。

## 4. P0-A：统一公共数学 scorer

### 4.1 单一正式入口与版本

建议新增 `LMOps/scoring/math.py`，正式版本名 `math_unified_v1`。它基于现有 OPRD 评分链：`OPRD/sft/quality.py` 的答案提取/调用关系，以及其使用的 `grade_answer_mathd`、`grade_answer_sympy`，修复并明确提取合同后成为唯一正式入口。

OPRD 源码参考：`OPRD/sft/quality.py`、`OPRD/sft/score_math.py`、`OPRD/verl/verl/utils/reward_score/ttrl_math/math_utils.py`。实现时阅读相关目录的 AGENTS.md，优先在 LMOps 下建立轻量、固定版本的共享模块，不改动 OPRD 的训练工程。

固定所复用的数学判等源码、依赖和许可证，可将必要的纯 CPU 工具纳入共享模块。避免依赖运行时 PYTHONPATH 恰好指向哪个 `verl`；不能在不同调用端分别维护一份评分逻辑。该 scorer 不加载 LLM，也不启动独立评分服务。

保留旧 `LMOps/sample/math_scorer.py` 为 legacy 审计实现。取消新增 `answer_line_v2` 字符串匹配评分器的要求；新正式运行不再并列使用 CARD 和 OPRD 两个 scorer。

### 4.2 答案提取合同

提取只看回答本身，不使用 ground truth 挑选候选。对全部可见回答定位显式答案标记，不再继承旧 CARD 的末尾 300 字符限制，也不从推理正文搜索“碰巧等于标准答案”的数字。

- 支持完整 `\\boxed{...}`，正确处理嵌套花括号；支持大小写不敏感的 `Answer:` / `Final Answer:` 及明确的 Markdown 包装。
- 按原文位置选择最后一个显式答案候选，而非无条件让早先的 boxed 覆盖后面的最终 Answer 行。Answer 行包含 boxed 时视为同一个候选，提取其表达式。
- 空 `Final Answer:` 标题后跟 `Answer: 42` 时继续解析后者，不把前缀作为答案内容。
- `Answer:\n42` 使用明确的后继非空行规则；若下一行也是答案标记，先解析标记，不能用跨行 `\s*` 吞入该前缀。
- 最后一个显式答案标记未闭合、为空或内容歧义时记录失败，不为了判对回退到更早的中间答案。
- 同一最终答案区域互相矛盾、无法确定唯一候选时标 ambiguous。合法的集合/区间/多根表达式作为一个数学答案处理，不能拆开后择一匹配 GT。
- 题目要求的对象类型应保留，不能简单剥掉所有单位、集合括号、顺序或定义域限制来制造等价。
- 明确的错误答案仍应判错。用真实回答和反例测试固定上述优先级，修改规则必须升级 scorer 版本。

### 4.3 数学判等与返回接口

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

### 4.4 单一配置来源与三处一致性

新增全局 `scoring.math` 配置块，至少声明 version、入口与判等预算；rubric、训练验证和外部评估都引用该配置。不得长期保留三套可独立漂移的 scorer 路径。

- 取消此前新增独立 `rubric.scorer_path` 的方案。旧 `validation.scorer_path` / `GAD_MATH_SCORER` 如需适配，只可由统一配置生成；外部覆盖冲突时 preflight 报错。
- baseline、correctness-only、joint 的验证与评估使用相同 fingerprint；是否启用 rubric 不改变 scorer。
- 同一个 `(response, ground_truth)` 经三处入口必须获得相同公共解析/判等结果；仅上表列出的消费策略不同。
- 新模型导出/评估读取运行快照中的 scorer 身份，不跟随全局模板的后续修改。
- 身份进入标签 cache key、实验配置、checkpoint manifest 和恢复校验，包括共享依赖，不能仅哈希入口文件。
- 新正式实验使用新缓存，不将旧 CARD 标签直接迁移为新标签。
- 新数据准备若使用正确性过滤，也调用公共 scorer 并记录版本；现有 5k 题目和 teacher 回答保持固定，不因换 scorer 静默重新筛选而改变对照数据。

### 4.5 单份正式结果与历史隔离

新评估在新 profile 下只产出一份正式 `scores/metrics.json`，以及对应的逐样本 `scored_samples.jsonl`，统一标注 `math_unified_v1` 和 fingerprint。不再自动生成并列的 `card_metrics.json`、`answer_line_v2_metrics.json`。

统一提取协议相对旧 OPRD 也可能改变部分结果，因此不能把新 scorer 称为与旧 OPRD 逐条等价。旧 `metrics.json` 和 `card_metrics.json` 均保持不变；旧回答的新评分写到审计目录或新评分 profile，绝不覆盖。

## 5. P0-B：离线审计与评估工具完善

### 5.1 增加纯重评分入口

在 `LMOps/evaluate/` 新增或扩展 `rescore.py`，与 `evaluate.py` 的生成/模型导出路径分开：只读既有 raw/scored JSONL，不加载模型、不启动 judge、不生成回答。

输入支持运行目录、stage、task 和 scorer 名称；输出放到新的 `LMOps/reports/rubric_audit/<audit_id>/` 或独立评分版本目录。对旧文件默认禁止覆盖。

先审计两组已完成 5k：

- baseline：`gad_qwen3_0.6b_5k/qwen3-0.6b-5k-seed42-n2`。
- rubric：`qwen3_0.6b_dapo_math_5k_rubric_0.6b_5k_20261012-215342-839723`。

按相同 `(global_index, rollout_index)` 对齐；检查输入重复、GT 缺失、题目数/样本数和输入哈希。不要把两种评分器的所有分歧都算成旧 CARD bug：解析规则、boxed 优先级、数学等价判断都可能导致分歧。

### 5.2 审计输出

- 仅在历史审计中对照旧 OPRD、旧 CARD 与 math_unified_v1。优先读取已有旧指标/逐样本分数；需复算时固定原实现版本。新正式训练报告只展示统一 scorer。
- 每条回答的各版 prediction、correct、解析状态和 disagreement 原因。
- 已确认跨行重复 Answer 前缀、无 Answer 但有 boxed、尾部窗口遗漏、归一化/等价判定差异、其他待复核等分类；不能确定则标待复核。
- 分阶段、数据集的混淆计数和固定数量实例。
- 对已有 rubric 缓存审计 correctness 标签翻转：旧 0→新 1、旧 1→新 0、mask→有效、有效→mask。
- 缓存不一定保存完整 response/GT；先检查 schema，再从明确关联的历史轨迹、calibration 或 rollout 日志获取。仅有 response hash 时不能恢复文本，也不能声称已审计全部标签。
- 记录可审计覆盖率、缺失原因；teacher/student、截断/完整、step 分层。首批 calibration 不是随机代表性样本，不外推全池错误率。

### 5.3 指标汇总改进

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

## 6. P0-C：统一种子与实验身份

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

## 7. P1-A：教师回答元信息绑定

涉及 `sample/build_gad_data.py`、数据 contract、dataset backend patch、`gad/extensions/runtime.py`、`critic_batch.py`。

- 数据构建在选取答案时同时返回其原始索引，不再用 `.index(answer)` 从文本反推，以支持重复文本、不同元信息的候选。
- `extra_info` 保存 selected_teacher_index、选中回答文本哈希、source_finish_reason、source_truncated 和元信息版本。
- 截断状态分开记录：采样端 length 截断、训练 tokenization 截断；二者用于 judge 的可见性元信息。V2 不因此整体 mask 过程维度，按配套计划逐维度判断。
- runtime 优先读取绑定到选中回答的元信息，不固定读取 teacher_index 的 `[0]`。
- 旧数据迁移时从 selection sidecar/teacher index 和原文本交叉验证。多答案且无法唯一确认时停止数据准备，不默认为第 0 条。
- 新 replay pair 一并保存这些元信息，重放时保持一致。

验收：构造第 0 条截断、第 1 条完整且选第 1 条的案例；完整回答不得被错误 mask。反向案例、重复答案文本案例也应通过。

## 8. P1-B：监督覆盖与联合训练诊断

### 8.1 覆盖率和标签分布

涉及 `extensions/online_rubric.py`、`critic_batch.py`，并提供旧日志离线汇总脚本。

- 每个维度分别记录 teacher/student 的 0、0.5、1、unknown；给出 unique 数和实际辅助权重总量。
- 区分 pair 次数与唯一回答数；截断分子和分母必须同口径。
- 当前 batch 的 unknown 原因包括缓存命中的标签；另列本步新增标注失败计数，不混用。
- 拆分 prompt_truncated、response_truncated、judge_unknown、格式错误、context overflow、规则答案解析异常等原因。
- 记录未截断样本条件下的有效覆盖率，区分“因截断未评”与“已请求但 judge 不判断”。
- 统计有效标签单值比例、各维度有效样本数，避免以很低的辅助 loss 推断监督有效。
- 本节只描述诊断；prompt 与截断策略按配套 V2 计划修改，不把 unknown 强行转低分。

### 8.2 损失和 adv

- 新增 `critic/loss_total = critic/loss_adv + rubric/loss_aux_weighted`，保留 `critic/d_loss` 的历史 BT 含义。
- 记录 optimizer step、microbatch 累积和全局归一化口径；不能再对已全局归一化的 aux 重复除累积次数。
- 总 loss 日志不等于梯度比例。梯度范数/余弦相似度如需分析，应在小型诊断任务中做，不默认给正式训练增加多次 backward。
- 固定一组独立诊断题目和回答，在无 dropout 的评估模式下保存四通道分数；不参与训练。
- 分析 adv 对正确/错误回答的排序、同题质量偏好，单独列 teacher 错而 student 对的情况；正确性指标标明 scorer 版本。
- 过程维度的 adv 相关性需经过人工抽查的标签，不能把未经审计的 judge 输出视为真值。

### 8.3 Replay 的处理边界

- 本轮主实验 replay=false，避免把回放和 rubric 变化混在一起。
- 保持轨迹池仅用于 critic 的既有设计；修复 teacher 元信息时覆盖 replay 保存/恢复测试。
- `max_bytes` 当前是 JSON 序列化字节预算，不是 Python RAM 硬上限。修正文档/指标名说明，另可记录进程 RSS 作为诊断，但不承诺严格内存限额。
- scorer 身份变化不能沿用旧标签恢复池。新正式实验使用空 replay 池；未来 replay 消融另行开始。

## 9. 测试与验收

### 9.1 必须的 CPU 测试

1. 统一答案提取：普通行、标题后 Answer 行、空行、CRLF、Markdown、嵌套 boxed、多个候选及优先级、空标题、未闭合表达式、超出旧 300 字符窗口的答案、裸换行答案；legacy 审计样例保持原行为。
2. 明确的错误答案仍为错误；不能用“尽量提高正确率”的启发式从正文或 GT 猜答案。
3. boxed-only、数字/分数/表达式等价及明确不等价的反例，检查不会通过忽略结构误判集合、区间、多根或题目要求的对象类型。
4. rubric、验证、外部评估三处公共结果一致；覆盖有效 0/1、无答案、解析失败、判等超时/异常、prompt 截断及缓存身份变化。检查错误没有被静默变为训练负例。
5. 多题多样本 sample/first/any-success、按题 pass@1 和固定分母测试；缺失 GT、缺失 rollout、重复样本及工具错误导致不完整结果的处理。
6. baseline/rubric/replay 开关组合均传入相同显式 rollout seed，漏配新正式实验报错。
7. teacher 选中索引与截断状态、重复文本、replay round trip。
8. cache hit 和 miss 下当前 batch 覆盖率/原因一致；重复 teacher 时 unique 比例不超过 1。
9. 保留现有联合 loss、DP/microbatch 归一化测试，并核对 hook 测试检查的是本次生成的 backend，而非固定旧目录。

### 9.2 少量 GPU 验证

正式训练前仅运行必要小规模验证：

- baseline、enabled+lambda=0、correctness-only、joint 各跑少数步骤，检查实际配置和参数更新。
- 所有组使用相同预生成 critic batch 验证 BT 与 aux；线上生成轨迹不要求逐位相同。
- Warmup→GAD 四通道权重传递、scorer 身份记录、保存恢复、缓存校验通过。
- adv-only reward 合同不变；原 actor 更新未被 rubric scorer 或 replay 直接改写。
- 新 scorer 与 cache 身份不一致时明确拒绝继续，不静默回退。

## 10. 实验重做方案

### 10.1 先做 5k，而非直接扩大到 17k

正式比较的共同条件：同一数据内容与顺序控制、同一初始 actor/critic、同一 Warmup/GAD 步数、显式 rollout seed、相同外部评估协议，replay=false。

| 组 | BT | 正确性辅助 | 过程辅助 | 用途 |
|---|---|---|---|---|
| A | 开 | 关 | 关 | 重做 baseline |
| B | 开 | math_unified_v1 | 关 | 检查可靠正确性监督是否有收益 |
| C | 开 | math_unified_v1 | V2 progress/error | 诊断新过程监督的增量影响 |

当前实现已有 correctness_only / joint；A/B/C 不要求任意逐维度组合开关，但需按配套 V2 更新 joint 的维度与头布局。C 必须在 V2 标签校准和合同验收通过后启动。旧 rubric 如保留，明确记为 legacy 诊断组，不能与 V2 混称为同一实验。

enabled+lambda=0 主要作小规模工程控制；若发现其与 A 存在明显差异，再扩大该控制实验，而非直接把差异归因于辅助损失。

第一轮使用明确的同一 seed 筛查；决定扩展到 17k 前，对候选设置与 baseline 做多个训练种子的配对重复，建议至少 3 个。以固定题目级外部指标及运行间差异判断，不只挑最好一次。

### 10.2 评估口径

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

## 11. 文件级实施清单

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

完成后应交付：可复核的旧结果审计、修复代码与测试证据、新配置 dry-run 差异报告、GPU 小规模验证记录，以及正式重跑清单。prompt 与新过程监督的具体方案见配套 [Rubric过程监督V2设计与参考实现计划.md](Rubric过程监督V2设计与参考实现计划.md)。
