# Rubric 监督 V2.4：实现与验收说明

更新：2026-10-16。依据：[修订方案](Rubric监督v2.4修改方案.md)。

**当前状态：代码与 CPU 验证完成；用户要求继续后，已完成一轮固定 8B Thinking Judge 审查材料生成。标签质量、真实 FSDP 训练及 Actor 收益尚未验收；正式实验由用户启动。**

> 附录A后续改造：当前首轮输出为 `rubric.events.v2.6`，复核最终为 `rubric.targeted_review.v3`（初始固定对照使用v2），采用紧凑区间及共用有界调度。下面主体保留首轮v2.4实施历史；最新代码、固定对照、CPU/GPU和回退状态见[附录A交付](../reports/rubric_v24_appendix_a_20261016/README.md)。人工数学验收仍为pending。

## 1. 本次实现

| 部分 | 实现与限制 |
| --- | --- |
| 证据协议 | 最多四个事件；`supporting_span_ids` 支持多个原文片段；`trajectory_claim` 忠实转述原文，`judge_verification` 说明 Judge 自行核查。原文由程序无损回填。 |
| 判据 | 独立正确中间结论支持 progress=0.5；不要求整题完成或逐步引用。error 按数学后果判严重性。正确截断前缀仍可 error=0；progress=1 需要整体解决目标。 |
| 归属与修复 | 修复显式关联原错误，支持片段必须包含原错误证据以外的替代位置；允许同时引用错误上下文。位置存在仍不证明数学蕴含。 |
| 元数据 | 分开记录 Judge 自报范围、程序结构状态、模型语义复核状态、训练 eligibility；默认人工 pending。保留 `analysis_complete/completion`，程序将它们整理为 `judge_reported_scope`，避免让 Judge 重复输出。 |
| 定向复核 | 独立可关闭的步骤。优先检查正向进展、error=0、修复和结构拒绝的标签；一次请求同时核查指定断言并允许一次证据重定位，随后重新运行结构检查。没有重新生成整套 Rubric。 |
| 复核失败 | 上下文超限、预算耗尽、缺失、反证、未解决分歧均 mask 受影响维度；不改成 0，不重试到肯定。模型通过记录为 `model_supported`，不是人工真值。 |
| 请求边界 | 每条轨迹最多一个额外定向请求，定向请求无 transport/format retry；共享线程安全请求计数。校准工具默认总上限 24 次生成请求、30 分钟，首轮/重试/定向均计入。 |
| Loss | `legacy_active`、`fixed_dimension`、`fixed_dimension_gated` 三种模式；固定分母为 3，门控先选 tau=4；沿用全局逻辑 minibatch、两来源 response ID 去重和 DP 梯度平均语义。 |
| 梯度诊断 | 可选周期性固定小 batch，BT 与总辅助项、各维辅助项的共享 backbone 范数、比值、余弦；排除 score head；不执行 optimizer update。遵守原训练 microbatch 上限。诊断 batch 随 checkpoint 保存和恢复；FSDP 实卡接线待验证。 |
| 在线监控 | 使用已有标签/缓存，记录来源覆盖率差异与比率、50 个百分点覆盖突变、至少四个有效标签同分告警，以及“规则判错且 progress=1”冲突计数；告警阈值不是语义验收标准。 |
| 缓存隔离 | 新协议命名空间和身份；独立工具将人工明确确认的坏缓存复制到新目录下的 quarantine，保留原目录与审计记录，不覆盖运行中缓存。 |
| 研究边界 | 四个 head、规则 correctness、BT、Warmup 原 Actor SFT 与 GAD 原 adv Actor 更新接口保留；未启用 Replay，未新增评分 head。 |

研究方案名称是 **V2.4**，但修改前 wire 已占用 `rubric.events.v2.4`。
因此实际升级为 **wire `rubric.events.v2.5` / input `rubric.input.v2.4`**。
旧输出按其冻结版本审计，不拿新 Parser 重新解释旧 JSON；旧 cache/checkpoint 不静默续用。

全局默认仍为旧 loss 和关闭定向复核；候选配置显式开启新选项。已有保存的实验配置没有被改写。

## 2. P0 固定资料

位置：`reports/rubric_supervision_v24_20261016/`。

- `freeze_manifest.json`：在新 Prompt 修改前冻结的抽样、排除清单和内容哈希。
- `holdout.jsonl`：从已有 **train** 轨迹随机选 12 道题，每题一个 Teacher/Student 回答，共 24 条；没有新增 Actor 采样，也没有取正式评估集。
- 排除此前 128/24/12 诊断输入的同题 ID、归一化文本和字符 shingle 近重复；剩余候选 272 题。不能证明没有所有语义近重复，人工仍需检查。
- 包含规则判对、判错和截断回答；难度尚未人工标注，不能把长度当成数学难度。
- `development_audit.jsonl`：旧 8 条完整输出的 16 个维度记录；原始分数、有效分数、mask 原因、问题描述、审查来源均保留。人工结论仍 pending。
- `development_audit_assisted.jsonl`：补充按维度的模型辅助判断及分类，与人工确认字段分开。
- `review/human_review_blind.jsonl`、`human_review_key.json`：新留出集盲审表及映射。先独立给数学标签，再查看 Judge 输出检查证据归属。
- 旧 frozen24 及此前 128 是**开发诊断集**。留出集如用于改规则，必须改记开发集。

旧 8 条的主要问题仍成立：括号题原文最终是 721，Judge 却将自己得到的 77 归给原文并给 1/0；篮球题确有原文正确求和，但后续仍使用错误总分，不能把整体视作已修复；假修复的严重性被低估。部分正确前缀和合法分支的原始判断/证据问题需分开审查。本次没有重新调用 Judge 来宣称这些问题已解决。

## 3. 建议验收设置：待用户确认

| 优先级 | 指标 | 建议门槛 |
| --- | --- | --- |
| P0 | 关键误接受 | 人工审查中 **0 条**：错误最终答案被 correctness 判对；明显未解决或错误解答 progress=1；关键未修复错误 error=0；把 Judge 自算或虚构修复归给原文。 |
| P1 | 原文支持 | 分数与证据分开审查；分数碰巧正确但证据不支持，不能计为可靠监督。已知关键归属反例不应被接受。 |
| P2 | 其他分数误标 | 暂建议每维最多 **1/20** 个经人工审查的保留标签；分母不足 20 保持证据不足，不自动扩样或降低门槛。 |
| P3 | 过程区分能力 | 单列同题、相同最终正确性下的可靠过程差异；报告独立题数。人工先约定前缀→关键错误→真实修复的预期变化。不能仅靠来源/长度/最终答案区分。 |
| P4 | 覆盖率 | 先描述，不设靠提高覆盖率即可通过的门槛；开发/留出、真实/反例分别报告。 |
| P5 | 成本 | 固定 8B Thinking；每窗口最多 24 次生成请求和 30 分钟 GPU 总耗时，包括重试和定向核查；不自动续跑。 |

**人工审查建议起点：旧 8 条 + 新留出 24 条，共 32 条、64 个维度。**
这不是要求新增 32 次 Judge 调用：旧 8 条复用结果；新 24 条是冻结池。
如果每条都需要定向复核，一个 24 次调用窗口通常最多完成 12 条。
人工可先盲审整个冻结池，实际生成多少就审查多少输出的证据；未生成项明确 pending。

“回答错却判对”须区分两个概念：correctness 由规则提供，Judge 不能覆盖；错误答案仍可包含独立正确的局部进展而得到 progress=0.5。
**progress=0.5 不表示答案正确。** progress=1 或把关键错误视为已修复，才属于需要重点排查的误接受。

0 个观测关键错误不证明总体风险为 0。12 道留出题也不足以建立很小的部署误差上界。
建议门槛、关键错误定义和审查规模详见 `acceptance_proposal.json`；`acceptance.json` 仍保持待确认。

## 4. 配置和阶段

- `configs/overrides/rubric_v24_8b_thinking.yaml`：阶段 A，固定 8B Thinking，开启一次定向复核，**legacy_active** loss；用于隔离 Judge 协议变化。
- `configs/overrides/rubric_v24_loss.yaml`：阶段 B/正式 B、C 共用 `fixed_dimension_gated`、tau=4、lambda=0.1；每 100 次 critic optimizer update 做一次固定 batch 梯度诊断，每 rank 最多四对，不增加 Judge 调用。可设 interval=0 关闭诊断。
- `train.yaml` 公开所有开关，默认不切换旧实验算法；全局默认的 non-thinking Judge 也没有被悄悄改成候选。
- 报告目录的 A/B/C 完整配置和 dry-run 日志仅供审查，没有启动训练。

数值含义：Correctness-only 使用固定分母时也是除以 3，辅助项是旧口径的 1/3；N=1 且 tau=4 时再乘 1/4。
B/C 必须共用 mode/lambda/tau，不能拿旧 legacy B 和新 fixed C 直接解释过程收益。
增加过程项仍可能提高总辅助梯度规模，需结合梯度诊断解释。

## 5. 后续命令

下面均在 `/mnt/jinbo/RLRM/LMOps` 执行。**已执行 CPU 测试和 dry-run；随后按用户“继续未完成的实验/验证”请求启动独立 Judge 窗口，详情见报告 `reports/rubric_v24_holdout_window1_20261016/`。以下正式训练命令未执行。**

### 5.1 CPU 回归

```bash
bash gad/tests/run_all.sh /mnt/jinbo/miniconda3/envs/gad/bin/python
```

### 5.2 人工确认后的小规模 Judge 验证

已生成 `reports/rubric_supervision_v24_20261016/judge_phase_A_config.yaml`，不必手写完整 YAML。
确认审查标准后，再启动一个独立窗口（GPU 6、7，固定 8B；端口 8021 不得被其他服务占用）：

```bash
/mnt/jinbo/miniconda3/envs/gad/bin/python gad/tools/calibrate_rubric_judge.py \
  --config reports/rubric_supervision_v24_20261016/judge_phase_A_config.yaml \
  --samples reports/rubric_supervision_v24_20261016/holdout.jsonl \
  --output reports/rubric_v24_holdout_window1 \
  --max-requests 24 --wall-seconds 1800
```

相同输出目录不会重新启动一个窗口；不足项保留未运行状态。不要自动重复窗口。
此轮若要与旧协议做同数据对照，应另用已冻结旧代码/服务身份，不能把旧 8 条和新留出集当成配对实验。
目前尚无新旧同数据质量对照结果。

生成逐维审查包与汇总（不调用模型）：

```bash
/mnt/jinbo/miniconda3/envs/gad/bin/python gad/tools/summarize_rubric_v24.py \
  --samples reports/rubric_v24_holdout_window1/samples.jsonl --cohort holdout \
  --output reports/rubric_v24_holdout_window1/audit.pending.jsonl
# 复制 pending 文件，人工填写 score/evidence/raw 判定、reviewer 和 review_status=completed。
/mnt/jinbo/miniconda3/envs/gad/bin/python gad/tools/summarize_rubric_v24.py \
  --audit reports/rubric_v24_holdout_window1/audit.reviewed.jsonl \
  --output reports/rubric_v24_holdout_window1/audit.summary.json
```

汇总单列人工和模型复核，不将 pending 计入可靠性分母；分组报告 raw→final、误杀、误接受、分数碰巧正确但证据错误，以及同题同答案正确性的比较。

### 5.3 正式 A/B/C 的准备命令（先 dry-run）

```bash
# A：BT baseline
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_5k --name v24_A \
  --overrides configs/overrides/rubric_v2_baseline.yaml \
  --overrides configs/overrides/rubric_v24_loss.yaml --dry-run

# B：BT + 规则 correctness
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_5k --name v24_B \
  --overrides configs/overrides/rubric_v2_correctness.yaml \
  --overrides configs/overrides/rubric_v24_loss.yaml --dry-run

# C：BT + 三维 Rubric；标签质量与实卡链路通过后再正式启动
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_5k --name v24_C \
  --overrides configs/overrides/rubric_v2_joint.yaml \
  --overrides configs/overrides/rubric_v24_8b_thinking.yaml \
  --overrides configs/overrides/rubric_v24_loss.yaml --dry-run
```

正式运行由用户去掉 `--dry-run`；C 还需先用相同覆盖配置启动独立 Judge 服务，见 `scripts/rubric_judge.sh --help`。
A/B/C 初始化、数据、seed、采样和 scorer 对齐。最终 Actor 性能仍需正式训练后评估。
真实标签 Warmup/GAD 各一次更新和恢复后的下一次更新，按方案在质量关卡后另做，不以 CPU 小网络测试替代。

## 6. 不应提前得出的结论

- 多片段协议和定向复核能否改善真实标签质量：**待新结果及人工审查**。
- 同一模型两次同意：仅模型一致性，不是高可信真值。
- 固定分母/门控使数值符合预期：**不代表错误标签已经消失**。
- CPU 小网络的参数更新一致：不等于 Qwen/FSDP/Warmup→GAD 实卡恢复已经通过。
- Rubric 能否提高 Actor：由用户后续完整 A/B/C 和统一评估判断。

## 7. 继续验证（2026-10-16）

- 完整 CPU 回归：27 套件、309 项，无 skip；最后缓存技术失败修正后，44 项相关检查通过。
- 请求停止逻辑补充：为服务清理预留 45 秒；优雅停止超时后仍清理自己拥有且进程出生标识匹配的进程组，避免停止子命令超时而遗留 GPU 服务。新增检查与 Thinking 模块共 35 项通过。
- 真实两 rank CPU Gloo：生产 `update_joint`，3 种 loss × 2 种 microbatch × 3 种标签缺失模式，各 rank 18 组；含跨来源、跨 rank 重复回答。与独立单批参考的参数最大绝对差为 1.87e-9，loss 差为 0。此结果不代替 FSDP/GPU 验证。
- Judge 使用独立目录、固定 8B Thinking、GPU 6/7；总请求数和 30 分钟墙钟预算已在启动前冻结。人工阈值仍待确认，不将本轮输出宣布为质量验收通过。

### 本轮实测结果

15 次请求、约 29 分 11 秒，处理 4 道题的 8 条真实回答；7 条首轮评分可解析，最终保留 5 个 progress、2 个 error，2 条两维同时保留。Student 仅保留一个 progress、没有 error；另有输出预算耗尽和截止附近复核超时。人工完成 0，不宣布质量关卡通过。GPU 服务已停止，卡已释放。

完整结果及可读盲审材料见 [本轮真实验证报告](../reports/rubric_v24_holdout_window1_20261016/report.md)。Python 3.10 future 超时归类和诊断模式/RNG 问题已据实修正，历史账本保留。

### 最终工程验收

所有本轮工程修复合入后，完整 CPU 回归 **28 套件、316 项通过，零跳过**；真实双 rank CPU 数值对照和 A/B/C dry-run 通过。完整交付、身份与源码快照见 [实现交付报告](../reports/rubric_supervision_v24_20261016/report.md)。该结论不代替数学标签质量、真实 Qwen/FSDP 更新恢复或 Actor 性能验收。


## 2026-10-17 附录 B/C 后续交付

已完成有限槽、交接状态解耦、默认跳过ROUGE、一次8B固定4条检查和真实Qwen0.6B worker更新/恢复；本节替代旧版必须全面人工验收后才可探索性训练的执行限制。人工质量认证及Actor收益仍未完成。冻结配置、全部原始记录、误判及成本风险见 [B/C报告](../reports/rubric_v24_bc_20261017/README.md)。正式实验未由执行助手启动。
