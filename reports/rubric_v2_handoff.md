# GAD Rubric V2 交接

2026-10-14。执行依据：`docs/GAD_Rubric_V2统一执行计划.md`。

## 2026-10-16：研究 V2.4 实现与有界验证

实际协议升级为 input v2.4/wire v2.5，支持多片段证据、原文主张/自算核查分离、一次可关闭的定向复核。
三种 loss 模式、全局去重门控、质量告警和固定 batch 梯度诊断已实现；正式 B/C 使用相同 mode/lambda/tau。
用户要求继续后，固定 8B Thinking 在 GPU6/7 执行共享 24次/30分钟窗口：实际15次、约29分11秒，
4题8回答中7条首轮评分可解析，最终保留progress5个、error2个、两维同时2条；Student仅保留一个progress。
人工0条、可靠性/过程区分/真实标签训练与恢复/Actor收益仍未验收；服务已停止、两卡释放。
Python3.10 future超时归类、停止清理及诊断模式/RNG问题已修正，历史失败账本保留。
最终28套件316项CPU回归通过，无skip；真实双rank数值对照及A/B/C dry-run通过。
详见[实现与验收说明](../docs/Rubric监督v2.4实现与验收说明.md)、[本轮实测报告](rubric_v24_holdout_window1_20261016/report.md)。
以下各节均为其冻结版本的历史记录。

## 2026-10-15：核查问题已修复，实验保持停止

活动协议为input v2.3/wire v2.4：修复事件必填原错误事件ID且使用不同物理行片段，未知数学阻止全局满分/无错，
合同拒绝与服务故障分开。parser不证明片段的数学蕴含关系。
答案抽取、invalid计分、评估完成状态、legacy scorer选择、数值配置、阶段n、验证计数/禁用、epoch导出及teacher截断已修复。
scorer指纹变化，旧标签及旧unified run不静默续训。26套件290项CPU验证通过，无skip。
1800条冻结回答已知正确493→497，仍66条invalid，不发布完整准确率；历史原始文件不变。
5个未使用backend先备份后清理，3个有历史引用的保留。未启动模型/训练，未提交或推送。
详见[逐项修复报告](code_audit_fixes_20261015/report.md)。下述为各自冻结版本的历史记录。

## 2026-10-15：片段协议与8B测试已按用户要求中止

当前活动协议为input v2.2/wire v2.3：程序原文片段ID替代手抄quote/loc、事件最多4个、局部独立支持保留，
协议/缓存/恢复身份更新。计划记于附录F；48项CPU验证及训练dry-run通过，没有启动正式训练。
8B thinking在GPU6/7实际请求9条，8条完整返回，第9条主动停止时中断，15条未运行，约8分1秒。
已返回17个片段ID均存在，但两条真实student中唯一两维有效的标签有明确数学/证据匹配错误，
因此不能把新增覆盖解释为可靠监督提升。全部teacher和5条正确对照未运行；人工pending，actor效果未测。
judge已关闭，两卡释放，实验暂停；详见[中止总结](rubric_v2/simplified_8b_20261015/report.md)。以下为历史结果。

## 2026-10-15：Thinking Judge 24条补完结果

按用户要求复用原6条，新增18次单次请求，冻结24条已全部处理，耗时23分5秒。
23条返回完整最终JSON，21条具有非空推理且正式解析；另外2条空推理、1条超时保留失败。
真实16条的两维有效覆盖仍为0；5条简单正确对照原始标签均合理，但全部因引用/定位问题mask，
说明证据接口瓶颈并非仅由student回答质量造成。边界8条中progress有效8条、error有效6条，
非法除法的错误进展仍能通过parser，合法未完成分支还出现原始error误判。

部分数学判断改善值得保留；当前未测得可训练覆盖提升或actor收益。
新增请求无重试，原6条/旧账本/模型prompt与parser身份不变；未启动训练或8B，GPU6/7已释放。
24条盲审材料齐备，人工完成0条；下一步建议先用已有输出验证稳定片段提取与数学支持关系。
详见[24条补完报告](rubric_v2/thinking_judge_complete24_20261015/report.md)。以下为历史阶段记录。

## 2026-10-15：附录 E Thinking Judge 当前结果

已实现judge独立thinking开关、采样和推理/最终JSON分离。14B thinking在GPU6、7真实返回了6条，
约束检查、假修复、半截公式有改善；但错误前提下的步骤仍被标成正确进展，真实2条的两维标签仍被mask，
且非法除法边界中的错误进展能通过parser。**不是14B能力上限的否定，也未达到正式joint监督标准。**

初次v0/Outlines约束准备超时经一次生成提示兼容修复处理；完整wire/parser/数学判据未放宽。
同一共享窗口21分5秒，共8次评分请求，0训练步。两卡已释放，其它实验未修改。
47项受影响CPU回归、修复后thinking模块11项、客户端7项定向回归均通过（批次有重叠）。
人工审阅pending；8B仅准备配置/CPU审计，没有运行模型对照。

详见[Thinking Judge结果与实施差异](rubric_v2/thinking_judge_20261015/report.md)。
候选为 `configs/overrides/rubric_thinking_14b_candidate.yaml`、`rubric_thinking_8b_candidate.yaml`，
均与actor non-thinking配置独立。以下附录D及早期命令作为历史接口保留。

## 2026-10-15：附录 D 当前状态

已改为无损紧凑输入 `rubric.input.v2.1` 与八个可选事件槽位的最终 wire
`rubric.events.v2.2`，加入局部证据依赖、最终学生 masks 后的监督保护及严格缓存/checkpoint 身份。
全技术失败立即停止，合法 unknown 默认连续三步空覆盖停止；同阶段恢复保留计数，新阶段从零开始。
48 项 CPU 回归、2 项部署接线检查及 pipeline dry-run 通过。

GPU6、7 共享约 19 分 44 秒、29 次评分请求完成一次有界修订。最终 24 条均为完整 JSON，
无输入超限或输出截断；但 **16 条真实回答的两维有效覆盖仍为零**，主要是引用改写、行号错误
及坏证据依赖。8 条边界中 3 条仍偏离预设锚点，假修复仍不可靠；人工审阅 pending。
因此 **正式 joint 仍未就绪**。本轮真实训练 0 步，Warmup→GAD→恢复仍未验证；不自动降级保护。
judge 已关闭，两卡已释放，已有 baseline 保持运行。

详见[附录 D 改造报告](rubric_v2/judge_protocol_v21_20261015/report.md)。
候选覆盖为 `configs/overrides/rubric_v21_candidate.yaml`（14B/32k/1536/并发1），没有改成已认证全局默认。
旧 V2 wire/cache/checkpoint 不可静默用于新协议同阶段恢复；后续新协议实验使用新 run。
以下早期结果及命令保留为历史记录，joint 命令继续只作待验证接口。

**实现已接入；真实 judge 校准未通过。** 四卡 baseline 的1步已完成；V2联合Warmup、
GAD交接/更新和新进程恢复未执行，不能使用上一版GPU结果代替V2验收。
本轮没有启动5k/17k正式训练、完整epoch、多seed或外部生成评测。

## 修改与固定接口

| 位置 | 已完成内容 |
| --- | --- |
| `scoring/math.py`、`scoring/vendor/` | 唯一正式 `math_unified_v1`；末显式答案、跨行/Markdown/嵌套boxed提取，保留数学对象结构，固定OPRD MathD/SymPy CPU子集、来源和Apache许可 |
| `train.yaml`、配置/启动/preflight | `scoring.math.{version,entrypoint,timeout_seconds,fingerprint}`统一三端；显式rollout/data/driver seed；旧评分仅作legacy审计；自定义模板的preflight使用CLI覆盖后的快照 |
| `sample/build_gad_data.py`、迁移工具、runtime | 候选原索引与文本hash/finish reason/截断绑定，重复文本不反推索引；保留来源截断与训练token截断 |
| `rubric/specs/math_process_v2.yaml`、prompt、版本化core | `[adv,answer_correctness,verified_progress,unresolved_error_severity]`；事件引用、局部unknown、修复关系、严格schema/证据校验 |
| `gad/extensions/online_v2.py`、worker | 当前回答在线监督、逐维mask、label分布/unique覆盖率/辅助权重；BT＋masked BCE及`critic/loss_total`，actor仍只取adv |
| cache/checkpoint | rubric/prompt/schema/处理合同/scorer及依赖身份参与缓存或恢复；不把旧V1四头当作V2加载 |
| `evaluate/`、兼容评测入口 | 一个正式scorer、一份每benchmark `scores/metrics.json`与逐回答文件；按题pass_at_1、严格样本分母、真实采样配置、工具错误导致不完整；旧双评分表显式`--legacy` |

默认replay=false。其`max_bytes`及`replay/serialized_bytes`是JSON序列化预算，
不是Python进程RAM硬限额。本轮没有replay性能实验。

公共结果包含版本/fingerprint、pred、parse_status/method、correct、grade_status、
candidate_count和error_reason。无最终答案：评估计0、correctness有效0；
malformed/ambiguous或grader错误：correctness mask；工具错误不发布完整评估成绩。
题目截断mask正确性，回答截断不会整体清空过程监督。

## 数据与历史审计

- 派生数据：`data/dapo_math_5k_gad_v2/gad_train.parquet`，原5000题、teacher文本和GT不变，
  不重新按新scorer筛选。原数据和历史实验结果未覆盖。
- 原GAD parquet SHA256：`497505cbcaad09bd60ec5f36eda7dbfe38b6c851bd49a0a777ed4227468bb5cc`。
- 派生SHA256：`f59b5f75de41c3f676de359dee5f7774bf048983c5e586501795097512f5acb6`。
- `reports/rubric_audit/v2_bounded/`：两组旧5k各64条，合计128条已有回答。
  50条临时计分分歧：16条跨行重复Answer前缀、28条最后显式标记优先级、6条数学归一化。
  分歧不能全算旧bug；4条工具不支持/异常仍需复核，不发布该审计集的完整新准确率。
- 完整纯CPU重评分和有界审计入口为`evaluate/rescore.py`；不需要重新生成回答。

## 验证证据与停止原因

- 公共评分/V2事件合同6项、V2配置/候选/cache/lambda=0合同4项通过。
  原rubric17项通过；sample66项通过（2个环境相关跳过）。
  通用pipeline66项、冻结run配置11项通过。最终完整gad CPU回归见
  `reports/rubric_v2/cpu_regression_final.log`。
- 最终生成backend上的Warmup/GAD更新顺序/失败屏障测试通过；原DP/microbatch去重
  梯度合同保留。仅为CPU证据，不宣称V2四卡联合梯度已验证。
- CPU单头→V2四头：adv行逐值相同，固定输入adv最大数值差`9.54e-7`；
  产物`runs/_initialization/qwen3_0.6b_math_v2_four_fixture/initialization_manifest.json`。
- `runs/rubric_v2_smoke_baseline/`：GPU1,2,3,4，Qwen3-0.6B，batch4，512/512，
  真实短teacher；仅1个全局步，actor/critic optimizer各1次，SFT loss约0.520，
  critic BT约1.469，grad norm约196.445，峰值显存15.50GiB。未触发数学验证或ROUGE。
- Judge用GPU6,7、独立8012端口、Qwen3-8B BF16 TP2。总36次校准应用请求；
  第一次API拒绝，第二次发现xgrammar忽略anyOf旁的required/properties，
  第三次产生12条完整JSON。修复了API不支持maxItems、anyOf完整对象分支和格式错误自动重试。
  没有更改数学标准来换取通过。
- 原12条输出复核：语法/原文引用均可解析，7个过程标签存在合同不一致；
  逐维mask这些标签，保留其它可靠证据。模型还把合法的非零分支当作严重错误，
  错误修复的事件类型与覆盖/完成状态不可靠，故**数学校准未通过**。
- 样例：`reports/rubric_v2/human_review_12.jsonl`；`human_review=pending`。
  模型辅助检查不代表人工验收。`judge_review_summary.json`列明问题。
- 模型验证从首次服务启动至baseline结束共约**22.8分钟**，实际训练**1步**。
  API/格式问题修复后仍沿用同一个30分钟截止时间，没有重置预算。
  细目见`validation_budget.json`。Judge及训练均已结束，GPU已释放。

**未覆盖：** V2真实联合Warmup、四头Warmup→GAD交接、GAD更新/断点恢复、
8192-token显存压力、actor效果、人工标签质量。长序列/正式效果由后续实验验证。
V1运行保留历史身份，新代码不将其冒充V2续训。

## 用户正式命令（本轮仅dry-run）

### 2026-10-15：追加 Qwen3-14B 测试

随后用户授权扩大至 96 条真实训练回答 + 32 条新边界样例，已在 GPU6、7 完成，
共约21分钟。真实回答30条上下文超限、50条输出截断、16条证据/事件/JSON失败，
有效过程标签0/96；边界32条可解析，11条存在标签合同问题，原始15条偏离模型辅助锚点。
这揭示了当前16k/768预算和证据接口的限制，不能把失败率当作数学错误率。
48条人工盲审材料已准备，人工仍pending；服务已关闭、两卡释放，baseline继续运行。
详见[128条扩大评测报告](rubric_v2/qwen3_14b_128_20261015/report.md)。没有更改全局配置或开展补测。

用户授权使用下载的 Qwen3-14B，在 GPU6、7 独立重测同一 12 条轨迹。
BF16/TP2/non-thinking 成功部署，过程标签合同不一致由 7 条降为 3 条，
但仍把“仅声称修复”当作有效修复、误判合法未完成分支，**数学校准仍未通过**。
此次共约 142 秒，12 次请求，无训练；两卡峰值各 20,564 MiB。服务已停止，
用户的 baseline 在 GPU1–4 正常继续。详见
[14B 校准报告](rubric_v2/qwen3_14b_20261015/report.md)。本次未更改全局 judge 默认配置。

公共单头初始化已经准备好，baseline/joint共用它；不需要重建大配置。
下面默认Warmup78步、GAD再78步、non-thinking、同样的种子/数据/预算。
`--run-id`决定目录名；新目录分别为`runs/baseline_math_v2_06b5k/`和`runs/rubric_math_v2_06b5k/`。
**joint命令是待执行接口，先审阅12条校准样例并解决judge问题；它未通过真实联合训练验收。**

```bash
cd /mnt/jinbo/RLRM/LMOps
# A：公共评分与显式seed的baseline。
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_5k \
  --overrides configs/overrides/rubric_v2_baseline.yaml \
  --run-id baseline_math_v2_06b5k --stages warmup,gad --dry-run

# B：Rubric V2配置；校准问题解决后再决定实际运行。
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_5k \
  --overrides configs/overrides/rubric_v2_joint.yaml \
  --run-id rubric_math_v2_06b5k --stages warmup,gad --dry-run
```

正式执行由用户去掉`--dry-run`。joint前需用相同配置启动judge并检查health：

```bash
bash scripts/rubric_judge.sh --action start
bash scripts/rubric_judge.sh --action health
# 使用结束后
bash scripts/rubric_judge.sh --action stop
```

恢复已存在的完整状态：

```bash
bash scripts/train.sh --run rubric_math_v2_06b5k --resume --stages warmup,gad
```

配置对照证据：`formal_comparison.json`，仅允许rubric项差异；
weights/tokenizer/data/scorer/seed/训练预算等均核对相同。
correctness-only另有`configs/overrides/rubric_v2_correctness.yaml`，本轮未运行正式消融。
17k旧数据尚需先用`sample/migrate_teacher_metadata.py`派生元信息，再配置新数据路径。

## 正式外部评估（由用户执行）

```bash
bash evaluate/run_eval.sh --run baseline_math_v2_06b5k \
  --methods base,warmup,gad --profile math_unified_v1_n8_math500n2_8k --dry-run
bash evaluate/run_eval.sh --run rubric_math_v2_06b5k \
  --methods base,warmup,gad --profile math_unified_v1_n8_math500n2_8k --dry-run
```

协议：MATH-500 n=2，其余n=8，max_new_tokens8192、max_model_len10240，
temperature1、top_p0.95、top_k-1、repetition_penalty1、seed42、non-thinking。
同scorer不等于同生成协议；DAPO greedy验证与外部采样评估分别报告。
已有5k模型的该协议dry-run已通过，日志`evaluation_dry_run.txt`。

## 参考版本

仅阅读官方论文/代码，未安装或运行参考项目。版本与文件hash在
`reports/rubric_v2/references/versions.json`；OPRD纯CPU工具来源/许可证在`scoring/vendor/`。
本实现借鉴[SRaR的进展/错误事件组织](https://arxiv.org/abs/2605.17291v1)、
[数学RRM的可观察数学判据与解法兼容](https://aclanthology.org/2026.acl-long.844/)，
并参考[PRM800K的中性步骤定义](https://github.com/openai/prm800k)。
没有复制步骤级actor reward、按题rubric生成或动态rubric流程。
