> 迁移发布说明（2026-11-14）：本目录仅携带历史报告摘要；文中原始日志、冻结副本和附件保留在源服务器，未随Git交付。当前发布与验证状态见 `reports/gad_n4_selection_migration_20261114/README.md`。

# Rubric Actor 奖励接入：实现与验证

日期：2026-11-03。状态：接入与有界验证完成，正式 Actor 性能尚待用户实验。执行依据：[实施计划及附录 A](../../docs/20261024/Rubric信号接入Actor与复用Warmup实施计划.md)。完整正式实验由用户启动。

## 实现

- 归并 `judge_fix/direct_v1_20261025/integration.patch` 的 14 项必要差分，保留主目录已有的评测、API 采样、W&B 和跨源码配置读取修复。
- GAD 可选 `adv_plus_rubric`：`adv + β(w_c sigmoid(z_c) + w_p sigmoid(z_p) − w_e sigmoid(z_e))`。与 Critic 的辅助 BCE 系数 λ 独立；BT、三维辅助 BCE、KL、GRPO 均保持原定义。
- Worker 在同一次无梯度四头前向中提取 `[B,4]` 分量，经真实 Critic compute_values → FSDP worker → DataProto 返回 driver。混合计算、传输和 GRPO 输入均为 FP32；只在最后一个有效回答 token 上放奖励，截断无需 EOS。
- `adv_only` / `β=0` 直接保留旧奖励值与 dtype，不返回额外分量；关闭模式使用旧返回接口。新增零值默认字段不纳入旧实验身份，不向历史 worker 的 Hydra schema 塞入不存在的字段。
- Warmup 的有效奖励模式固定为 `adv_only`，保留 Teacher SFT；SeqKD 不接入本扩展。
- 新 run 通过现有 handoff 从 Warmup271 的 Actor、四头 Critic 和 `data.pt` 初始化，GAD 新建优化器/调度器。同阶段恢复才载入完整 worker/扩展状态。奖励配置进入既有恢复身份；不能以恢复方式静默改 β 或权重。
- dry-run 读取显式来源 Warmup 的四头元信息，避免将尚未导出的目标 HF 误判为需要随机扩头。真实启动仍执行原交接工具。

## 日志口径

`actor_reward/*` 是本步指标，`actor_reward/window/*` 是最近最多 10 个已提交步骤。窗口不足 10 步时记录实际 `steps`；恢复时从现有 `extension_metrics.jsonl` 按 step 去重，并忽略 checkpoint 之后未提交的记录。没有新增独立监控 checkpoint。

- adv/c/p/e、rubric、贡献、总奖励的均值和标准差；真实长度、截断数和异常结束数。
- 使用实际 GRPO uid 配对，分别比较 adv→c、c→full、adv→full：严格翻转、平局、打破/产生平局、近 epsilon 分差、有效分母；TW 单列两个翻转方向。
- 将混合奖励送入原 GRPO 后的真实优势与预期值比较，报告非零优势组和优势变化；奖励均值上涨不作为新训练信号的证据。
- c 与现有有效正确性标签比较均值、偏差、MSE、AUC，按截断分层。仅有单类时 `auc_defined=0`，AUC 为 N/A，不停止训练。`no_answer→有效标签0` 保持原规则。
- p/e 仅使用本轮被选中且 mask 有效的 Student 标签，记录偏差/MAE和样本数，接受 0.5 软标签。通过明确的 pair_id/response_id 对应 `prepare` 张量；缓存顺序、未选中的缓存标签和 Teacher 标签均不充当 Student 监控标签。
- `actor_reward_monitor` 是既有 JSONL 日志中的紧凑行记录；W&B 仅接收数值指标。这些是与训练标签的一致性，不是独立数学真值或概率校准证明。

## 公平对照与保护

详见 [source_identity.json](source_identity.json)：原始完整 `dapo_math_17k_gad_v2` 的训练/开发文件、spec、prompt、scorer 内容一致；有效 tokenizer 模板及 non-thinking token probe 一致。HF 导出把模板另存为 `chat_template.jinja`，因此不以模板文件是否单独存在判定语义差异。

有意变化仅为 Actor 奖励模式/β/权重。实现目录和端口改到主目录，新 run 显式指定 Warmup271、GAD 新增271。维持 batch64、n=2、seed42、8192、Judge8B Thinking/concurrency8/TP2/每逻辑 minibatch 最多16条、λ=.1、τ=4、replay=false、学习率和 KL；保留用户的所有 offload=false、Actor/Critic gradient checkpointing=true、Critic microbatch=2、CUDA graph 配置。

[before.json](before.json) 与 `before/` 保存实施前的小文件副本和 hash；源 run 和已使用 backend 的 54,089 个文件元信息另有记录。源实验、隔离源码、旧 backend、标签缓存和评测结果不覆盖。

## 验证进度

- CPU：45套件完整回归已通过，最终奖励/传输测试7项通过；保留初次失败和修复记录。工程生成器曾在 Warmup 较深的 if/else 中产生缩进错误，已改为单行表达式替换；新测试的 fake Critic 漏设 enabled 也已修正。
- 固定开发样本：固定打乱顺序，首批128题×2，Warmup Actor271、原训练模板、non-thinking、temperature=.8、top_p=1、seed42、max8192，保存真实 token IDs。旧 calibration/单回答贪心验证和测试集排序样本均不满足复用条件。
- 首次单卡32题并发导致 KV 反复抢占，0条完成后主动中断，原日志保留。改为每批4题、1–4卡各自独立采样；不调整训练参数、不覆盖完成样本。四头推理每条回答只执行一次，离线比较 β=0/.05/.1/.3 和 adv/c-only/full。
- 有界训练：已完成独立 smoke，总计划预算仍271新步；2步→273保存退出，完整恢复后1步→274。没有重跑 Warmup 或开展正式评测。

固定样本、真实训练结果及正式启动命令如下。

## 已完成的固定样本结果

[完整分量及分析](fixed_dev/analysis_128.json)、[同题分差](fixed_dev/pair_deltas_128.jsonl)、[系数选择](beta_choice.json)。128题/256条中：C21、W65、T169、U1；T 内167条正确性标签0、2条 unknown，没有直接把截断当作错误标签。另1条非截断 unknown。有效正确性253条、其中21条标签1。

四头没有近乎恒定：c/p/e 标准差分别为 .350/.248/.243。adv 同题绝对分差中位数1.599；β=.05的 rubric 贡献分差中位数 .02097。c 对有效数学标签的 AUC=.940，但均值偏差 +.231、MSE=.155，不能把它当作已校准概率。

| β | adv→c 严格翻转 | c→full 严格翻转 | adv→full 严格翻转 | adv 平局→full 非平局 | 一对一错排序正确数 | TW 严格翻转 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0/128 | 0/1 | 11/13 | 0/32 |
| .05 | 0 | 3 | 2/128 | 1/1 | 11/13 | 0/32 |
| .1 | 0 | 4 | 3/128 | 1/1 | 11/13 | 0/32 |
| .3 | 3 | 3 | 5/128 | 1/1 | 11/13 | 0/32 |

首版冻结较小候选 β=.05。两次严格翻转都是 TT，另有一对 CC 打破平局；没有观察到正确性排序收益。过程项能改变偏好，与其已具备可靠过程语义或能提升 Actor 性能是不同结论。样本规模有限，未为提高翻转率扩大网格或继续采样；step542 的历史量级检查为可选项，本轮未运行。

定向抽查6个有变化的回答对和4个未翻转对照，共20条，使用相同 Qwen3-8B Thinking / direct_v1。19条结构有效，1条耗尽 Thinking 预算；总20次生成，prompt158,671、completion49,006 tokens、批耗时400.88秒。未启用复核，不改 prompt，没有把这20条当作全体过程标签准确率。[参考结果](fixed_dev/judge_references.jsonl)与[选择依据](fixed_dev/judge_selection.json)保留完整内容。

有限案例中，两条 TT 偏好变化可解释为倾向较有进展/较少混乱的前缀，但两侧仍有长篇重复和未完成推理；有的 Judge error=0 也不能据此认证整段数学无误。一对 CC 的过程偏好与 c-only 偏好相反，但双方答案正确，不能据此判定哪条过程更好。没有强制截断惩罚，也未删除 error 项。

离线生成复用 `VllmOfflineEngine`，本次显式使用 v0；正式 hybrid 训练沿用原 v1。模板、n/seed/temperature/top_p/top_k、输出上限一致，不宣称其生成与并行在线训练逐 token 一致。保留实际 prompt/completion token IDs，保证本批 Critic 打分无需文本再编码。

CPU 最终全套回归：45套件，422通过/5个可选真实 tokenizer 测试跳过。随后补充 TW 分差和显式平局比例，再在最终新 backend 上重跑7个奖励/传输/恢复测试，全部通过：[全套日志](cpu_final.log)、[最终新增测试](cpu_actor_final_generated.log)。

## 正式启动命令

在 `/mnt/jinbo/RLRM/LMOps` 执行。完整配置已包含来源 Warmup271、GAD 新增271和冻结的 β=.05；不要使用本报告的 `smoke.yaml` 启动正式训练。

```bash
cd /mnt/jinbo/RLRM/LMOps
export GAD_PYTHON=/mnt/jinbo/miniconda3/envs/gad/bin/python

# 先启动 Judge（6、7卡）。
bash scripts/rubric_judge.sh --config configs/launch_rubric_actor_06b_17k.yaml --action start
# 模型加载完成后检查；需返回 healthy=true 再启动训练。
bash scripts/rubric_judge.sh --config configs/launch_rubric_actor_06b_17k.yaml --action health

# 首次运行；使用1、2、3、4卡，不需要重跑SeqKD/Warmup。
bash scripts/train.sh \
  --config configs/launch_rubric_actor_06b_17k.yaml \
  --run-id rubric_actor_06b_17k_v1 --stages gad
```

可先在训练命令末尾加 `--dry-run`。不带 smoke 限制，目标 step542，正常保留验证，仍按每50步保存、保留最近5次完整保存（交接/导出目录另行保护）。新 run 文件夹为 `runs/rubric_actor_06b_17k_v1`。

第一次保存后需要同阶段恢复：先确保同配置 Judge 正常，再使用新 run 自己的配置，不重新指定初始化来源或改变奖励权重：

```bash
bash scripts/train.sh --run runs/rubric_actor_06b_17k_v1 --stages gad --resume-gad
```

只向 Actor 接入正确性预测的消融也使用同一 Judge 和三维联合 Critic，β与正确性权重相同，只有 Actor 的 p/e 权重为0：

```bash
bash scripts/train.sh \
  --config configs/launch_rubric_actor_c_only_06b_17k.yaml \
  --run-id rubric_actor_c_only_06b_17k_v1 --stages gad
```

[完整模式 dry-run](formal_dryrun_final.log)与[c-only dry-run](c_only_dryrun_final.log)已通过，显示同一 Warmup271、新增271、最终542和正确奖励模式；c-only 配置的 `rubric.mode` 仍为 `joint`。本轮没有运行该消融训练。

正式训练完成后，按现有独立评测入口统一做8项/16384长度的比较；当前两脚本分别覆盖原4项与新增4项，使用相同新 scorer 及任务各自的采样数。新 run 只训练GAD，评测显式选 `gad`，避免默认要求不存在的 SeqKD/Warmup 阶段：

```bash
bash evaluate/run_eval.sh --run runs/rubric_actor_06b_17k_v1 --methods gad
bash evaluate/run_extended_eval.sh --run runs/rubric_actor_06b_17k_v1 --methods gad
```

本轮未运行上述正式评测或修改 `evaluate/`。训练 YAML 中旧 `eval.*` 是停用的内部评测兼容设置；独立评测采样长度由上述脚本显式设置为16384。对照旧 naive run 时使用同一套入口/协议，或复用已确认相同协议的既有结果。


## 3步真实训练与恢复结果

[完整验收](smoke_verification.json)、[执行状态与清理](smoke_status.json)、[第一轮状态](first2_state_check.json)。独立 run 为 `runs/rubric_actor_smoke_20261103`，新 backend 为 `gad/.backends_gad_extensions_v1_67effa75d631/gad`。

- 第一次调用使用 Warmup271 的 Actor/四头 Critic/data.pt，新 GAD 优化器/调度器；只执行272、273两步，保存273并退出。第二次从这个新 run 完整恢复，执行274并保存退出。
- 两次调用的总计划均是271新步、最终542；工程限制分别2/1，不把总训练预算缩成3。三步均完成真实 Actor/Critic 更新，梯度非零，原始 step 指标未发现 NaN/Inf，无 OOM。
- 每步128条 Student /64组，真实 GRPO 优势与同题混合奖励重算结果的最大误差均为0。三步严格翻转、打破平局均为0/64：当前 β 在这些批次没有改变组内偏好，优势仅有 epsilon/数值量级差异。不能把本次工程通过解读成 Actor 性能提升；固定开发集2/128翻转与这192组0翻转一起说明直接影响可能较稀少。本轮没有因此追加GPU矩阵或调大β。
- 每步16次新 Judge 调用，总48/48预算；47条结构有效，1条在274耗尽 Thinking 输出预算，过程 mask 为0。无额外复核/格式重试。Student 有效过程标签数为11、11、10，监控排除 Teacher 标签和本轮未选中的缓存项。
- 正确性有效 Student 数为127、126、127，合计380/384；4条 unknown 参与模型预测奖励，但不计入误差/AUC。`no_answer` 仍按原规则给有效标签0。
- 最近10步窗口在第二次调用从既有已提交 JSONL 重建，实际窗口由2步扩展为3步，384条/192组，不重置、不混入 Teacher。

| step | Actor grad norm | Critic grad norm | rollout s | annotation s | Actor update s | 全步 s |
|---|---:|---:|---:|---:|---:|---:|
| 272 | 0.211 | 13.144 | 287.1 | 257.8 | 59.2 | 738.4 |
| 273 | 0.232 | 12.344 | 254.6 | 353.9 | 54.9 | 798.7 |
| 274 | 0.234 | 27.328 | 266.2 | 352.7 | 55.9 | 812.9 |

共耗时 45.4 分钟，含两次启动、交接、导出和清理。单步约12.3–13.5分钟；3步不是充分的性能消融，Judge耗时和轨迹长度会变化。这次保持了原训练提速参数，新增奖励复用已有前向，不额外调用 Judge。

恢复证据：

- Actor/Critic 各4个 rank 的优化器计数与 scheduler.last_epoch 均2→3，CPU/CUDA/NumPy/Python RNG 项齐备且实际加载日志存在。Warmup 原计数不是新 GAD 的优化器起点。
- `data.pt` 初始 SHA 与来源一致；恢复后 sampler 从128→192条、已取 minibatch 从2→3。扩展 stage_step 2→3，cache引用32→47，Judge累计32→48，counter 初始化标记为 `same_stage_checkpoint_restore`。
- 四个 Critic 头相对初始BF16载入基准的1024个权重均发生改变；273→274各头1023或1024个权重继续改变。比较时先对初始值作同样BF16转换，避免把交接量化误记为训练更新。
- Actor 第0层 q_proj 的4×32采样块相对初始128/128改变；恢复更新后127/128继续改变。这是采样块证据，优化器/调度器则检查了两角色全部4个rank。
- 首步 calibration 的32个混合Teacher/Student response_id 与旧 naive 实验一致，支持起点/游标/模板对齐；不把这个子集宣称为完整 batch 的逐token一致性证明。

[保护检查](protection_final.json)：来源实验/旧backend共54,089个文件大小、mtime、inode均未改变；主目录机器预设、evaluate、sample文件hash保持；新backend源hash验证通过。每3秒GPU采样的训练峰值约23.0GiB、Judge约22.6GiB，并非精确瞬时峰值。全部本次训练与Judge进程已退出，最终8张卡均无模型作业占用。

本轮没有执行正式271步实验、c-only完整消融或8项测试集评测；它们使用上面的完整配置和独立评测入口，由用户安排。

## 审阅差分

[本轮相对实施前工作区的25文件差分](implementation.patch)、[文件前后hash](implementation_files.json)。该差分包含 direct_v1 归并和本次奖励接入，使用反向 `git apply --check` 验证通过；没有混入用户此前已存在的评测、采样或W&B改动。

主要位置：配置/校验 `gad/tools/extension_config.py`；无梯度四头奖励/分量 `gad/extensions/worker.py`；配对和窗口诊断 `gad/extensions/actor_reward.py`、`runtime.py`；DataProto/Hydra连接 `gad/tools/algorithm_patches.py`；外部 Warmup dry-run 与运行身份 `gad/tools/pipeline.py`；回归 `gad/tests/test_actor_reward.py`。固定样本工具为 `gad/tools/diagnose_actor_reward.py`，本轮执行/验收脚本留在本报告目录，不加入训练调度框架。
