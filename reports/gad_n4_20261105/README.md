> 迁移发布说明（2026-11-14）：本目录仅携带历史报告摘要；文中原始日志、冻结副本和附件保留在源服务器，未随Git交付。当前发布与验证状态见 `reports/gad_n4_selection_migration_20261114/README.md`。

# GAD n=4 三组对照实施与验证

2026-11-07 实施并完成工程验证：全量CPU回归、三份dry-run、三组各2+1步完整恢复、部署哈希和历史保护检查均通过。共9次真实更新，训练卡1–4、Judge卡6–7均已释放。正式训练和正式评测均由用户启动；第3组正式参数仍待完整naive日志选定并由用户确认。

## 实现

- 公共开关采用 `actor_reward.fp32`（默认 false），位于 rubric 之外。三组均开启，在 driver、原 GRPO 之前把纯对抗分转成 FP32。Critic forward/优化器精度不变。
- `adv_plus_rubric_groupnorm` 在 driver 汇总完整 uid 后计算内层标准化，最后有效 token 写入 s，再进入原 GRPO；显式关闭 reward KL，Actor 原 KL loss 保留。worker 只提取四头，不做组内运算。
- 训练、影子日志重算、离线选择都调用 `actor_reward.reward_scores` / `groupnorm_scores`。旧 additive 模式保留。
- w=0 或单组 rubric 零方差时直接传原 FP32 adv；τ=0 不计算 0/0。对抗分无区分力时允许 rubric 决定方向，单独统计。
- naive 用 `adv_only` + `rubric.actor_reward_shadow=true`，一次四头 forward 同时记录，奖励与优势不变。
- baseline 的 `actor_reward.scorer_monitor=true` 使用同一 `math_unified_v1` 及训练标签映射，只观察 Student；原 Critic batch 原对象返回，不插入监督标签、不重建 Critic 输入。无四头预测时日志 parts 为 `[adv,null,null,null]`。
- 每条影子记录增加跨 step 稳定的 `question_id`（有效 prompt token/text/template 哈希）。保留 step/uid/response_id/parts/target/mask/finish_reason/tokens/category/实际优势；保留最近 10 步窗口与恢复去重。
- 增加组构成、C/W/T 平均优势、同组 C−W 优势差、rubric std 分位数、下限组翻转占比、直通组、弱对抗分组及优势变化等指标；既有所有组内配对统计继续保留。
- baseline 的旧 Warmup 数据路径与 v2 不同：必须先逐行比较题目、Teacher、reward_model、data_source、ability、extra_info.index 和顺序，同时核对 batch、seed、长度、模板开关。初始模型路径迁移也核对全部权重/配置/tokenizer 文件 SHA256。检查写入新 run 的 `warmup_source_compatibility.json`。单头直接交接，不扩头。
- 新默认值不进入历史身份。历史 baseline、naive、additive actor-rubric 的 Hydra 参数与 stage signature 均与修改前一致，见 `legacy_identity_check.json`。
- 原 offload=false、Actor/Critic gradient checkpointing=true、Critic microbatch=2、Actor token budget=10240、CUDA graph、学习率/KL/grad clip 等保持模板设置。

## 三份正式配置

|组|配置|Warmup|Actor 奖励|过程抽样名额|
|---|---|---|---|---|
|baseline|configs/launch_baseline_n4_06b_17k.yaml|baseline_06b_17k/271|FP32 adv|无|
|naive|configs/launch_rubric_naive_n4_06b_17k.yaml|rubric_direct_06b_17k_20261026/271|FP32 adv，四头影子记录|32|
|actor-rubric|configs/launch_rubric_actor_n4_06b_17k.yaml|同 naive|分别归一化后加权|32|

共用：64 题×4=256 条学生回答/步；271→542；non-thinking，seed42，temperature0.8，8192 tokens；数据原完整 dapo_math_17k_gad_v2（17405 行），scorer fingerprint `f3549b873f06e227cdadb6fe44eff97f5d0278d414618dd6b8d39eaee814ef8e`；50 步保存、最近 5 次（已有 handoff/export pin 规则保留）；step400 quick、step542 full。

第三组当前 full/(w=1,τ=0.05) 仅为方案规定的工程短测占位参数，`rubric.actor_reward_selection_confirmed=false`。正式启动会在创建 run 之前明确停止，直到 naive 完整日志选参并由用户确认、写回配置。dry-run 和有限步 engineering smoke 可验证路径。

## 启动命令

在 LMOps 目录执行，三组按顺序使用训练卡 1,2,3,4；Judge 只用 6,7，独立端口 8152。

```bash
cd /mnt/jinbo/RLRM/LMOps
export GAD_PYTHON=/mnt/jinbo/miniconda3/envs/gad/bin/python

# 1. baseline，无需 Judge
bash scripts/train.sh --config configs/launch_baseline_n4_06b_17k.yaml \
  --run-id baseline_n4_06b_17k_v1 --stages gad

# 2. naive：先启动 Judge，模型加载后确认 healthy=true
bash scripts/rubric_judge.sh --config configs/launch_rubric_naive_n4_06b_17k.yaml --action start
bash scripts/rubric_judge.sh --config configs/launch_rubric_naive_n4_06b_17k.yaml --action health
bash scripts/train.sh --config configs/launch_rubric_naive_n4_06b_17k.yaml \
  --run-id rubric_naive_n4_06b_17k_v1 --stages gad
```

naive 完成后执行离线选择（CPU，无 Judge/模型推理；只输出建议，不改配置）：

```bash
"$GAD_PYTHON" gad/tools/select_actor_reward.py --run runs/rubric_naive_n4_06b_17k_v1
```

该工具按 question_id 的 SHA256 分半；在看任一 w 的结果前固定两种分量各自的选择集 std 第25分位 τ；只从 full/c-only × w={0.25,0.5,1,2} 的 8 个候选选择。τ=0/中位数的16组只作敏感性参考。复核集与 step450–542 各自报告约束。零分母输出 null/N/A；无解不扩大网格、不修改规则。排序干预上限包含“原非平局变成平局”，严格修正/破坏与 TW 方向只统计严格翻转，原有 tie resolved/created 指标另报。

用户确认后，更新第三组配置的 weights、group_weight、rubric_std_floor，并将 `rubric.actor_reward_selection_confirmed` 设为 true，再运行：

```bash
# 3. 确认参数后启动；沿用第二组同一 Judge 服务，停过则先 start/health
bash scripts/train.sh --config configs/launch_rubric_actor_n4_06b_17k.yaml \
  --run-id rubric_actor_n4_06b_17k_v1 --stages gad
```

续训统一使用已保存 run：`bash scripts/train.sh --run runs/<run-id> --stages gad --resume-gad`。不重新执行 SeqKD/Warmup。

## 验证证据

- `cpu_after_isolation_fix.log`：46 个套件全部通过，439 个 unittest 用例，其中 5 个可选 tokenizer 用例跳过；另含两个 scorer self-test。见 `cpu_verification.json`。
- `cpu_n4_final_backend.log`：实际最终 backend 的组内奖励/GRPO/跨分片/影子/监控/选择定向测试通过。
- `dryrun_{baseline,rubric_naive,rubric_actor}.log`：三份正式配置 dry-run；最终 backend 身份见 `backend_identity.json`。
- `smokes_status.json`、`*_first2_training.log`、`*_resume1_training.log`、`gpu_samples.jsonl`：每组独立目录，真实 2 步 + 完整恢复 1 步；工程配置关闭 dev、无正式评测，Judge 每组持久请求上限96，不重置续训计数。
- `gpu_verification.json`、`{baseline,rubric_naive,rubric_actor}_verification.json`：三组完整验收均通过；含各rank优化器/调度器/RNG、数据游标、头部权重更新、GRPO重算、Judge有效标签和计时。
- `before.json` / `before/`、`protection_before.json`：结束后核对706885个历史文件/链接对象的lstat inode/size/mtime，均未变化；113个非目标源文件哈希不变，见 `protection_final.json` / `source_scope_final.json`。这不是对所有历史权重重新计算SHA256。
- `backend_final.json`：最终backend的patched文件、扩展、rubric协议和scorer Python模块均与源文件/部署身份匹配。

正式 step542 统一评测继续使用既有 `all8_dapo_v2_af1_taskn_16k` 协议和 `evaluate/summarize_runs.py`，本次未修改 evaluate。第1/2组按完整方法比较，第2/3组共享 Warmup；单 seed 不能估计训练随机性。

首次 GPU 检查在首步 rollout 后、Critic/Actor 更新前发现 baseline 新监控的 Ray 隔离导入问题（`scoring` 不在 sys.path）。失败尝试没有执行优化器更新，原目录及日志保留在 `runs/baseline_n4_smoke_20261107` / `attempt1/`。修复采用与 OnlineRubric 一致的显式 scorer entrypoint 加载并校验同一 fingerprint，补充 `python -I`、仅 backend 路径、临时 cwd 的 CPU 子进程测试；最终新 backend 定向 12 项测试通过。修复后的 baseline 使用独立 `_v2` 短测目录，旧 backend 不重写。

## 工程验证结论（2026-11-07）

全量CPU回归46套件通过，439用例中434通过、5个可选tokenizer用例跳过；另有两个scorer self-test。三份dry-run通过，均指向新backend `0192d4849cc7`。GPU每组先训练2步，保存step273，再完整恢复训练1步到274，共9次成功更新，没有启动正式训练或正式评测。

三组每步均为256条Student/64组，奖励FP32，实际GRPO与公式重算最大误差均为0；无OOM或非有限训练指标，Actor/Critic梯度均非零。四个rank的optimizer和scheduler从2连续到3，数据游标从128到192题，监控窗口从1到2到3。单头baseline直接交接；两组rubric四个头以及Actor抽查参数均在恢复后继续更新。

baseline的只记录监控与naive影子记录均保持优势逐值不变。actor-rubric临时参数下，三步相对纯adv的平均绝对优势变化分别为0.3551/0.3868/0.4039；每步384对组内配对中严格排序翻转59/61/75对。这证明混合信号实际进入Actor更新，不证明方向正确或Actor性能提高，也不据此冻结正式参数。

### Judge与有效标签

两组每步均选中Teacher11/Student21，真实请求均为32次，续训前后持久计数64→96，没有重置或额外重试。

|组|实际请求/解析成功|Teacher Progress / Error有效|Student Progress / Error有效|
|---|---|---|---|
|naive|96 / 96|33/33、33/33|63/63、62/63|
|actor-rubric|96 / 93|33/33、33/33|60/63、59/63|

actor-rubric的3次失败均为8192 token thinking预算耗尽（step273一条、step274两条）。逐条确认其Progress/Error标签为无效并屏蔽，correctness仍来自独立 `math_unified_v1`；训练和恢复正常继续。完整证据见 `rubric_actor_verification.json` 的 `failed_annotations_correctly_masked`。解析成功和“有效”只表示协议/掩码通过，尚未人工判断过程评分的语义可靠性。

### 单步成本与271步粗估

|组|三步总耗时（秒/步）|中位数（分钟/步）|按三步均值估计271步（小时）|
|---|---|---|---|
|baseline|902.574 / 858.250 / 956.834|15.04|68.2|
|naive|1296.054 / 1302.311 / 1261.623|21.60|96.9|
|actor-rubric|1361.146 / 1349.803 / 1407.112|22.69|103.3|

|组|平均rollout（秒）|平均标注（秒）|Critic扣除标注（秒）|平均Actor更新（秒）|
|---|---|---|---|---|
|baseline|512.7|0|185.8|116.4|
|naive|498.7|404.1|180.7|112.9|
|actor-rubric|500.5|485.9|180.6|114.5|

`timing_s/update_critic`包含标注，上表已减去独立标注计时；纯Critic栏仍含其余准备/监控开销。总步时还含log-prob、ref、reward、诊断和保存，不能只把上表四栏当作完整总和。271步估计仅由三步均值线性外推：两条样本含短测边界保存，未计启动、正式dev验证、正式评测、后续轨迹/缓存变化与重试成本；不作为长期性能基准。

### 显存、保护与清理

`nvidia-smi`每3秒采样，观测到的训练卡单卡峰值：baseline23.47GiB、naive23.54GiB、actor-rubric22.99GiB；Judge峰值分别为22.55GiB和23.27GiB。采样可能错过更短瞬时峰值；baseline原调度中断后有一段采样缺口，见 `gpu_summary.json`，不能声称覆盖了baseline全部时刻。0/5卡没有模型作业，只有短暂驱动上下文。

706885个历史文件/链接对象与原快照一致，113个非目标源文件哈希不变；旧run/backend/标签缓存/评测产物保留，历史三类配置的Hydra参数和stage signature未变。最终保护脚本最初混用stat/lstat误报33个W&B日志链接；已按原快照的lstat口径修正，逐项确认链接本身未变，证据见 `protection_symlink_audit.json`，原误报留在 `protection_stat_mismatch_20261107.json`，没有为消除报错而改动历史文件。

原调度退出后，`resume_smokes.py`从已完成的baseline step273恢复调度，跳过已完成两步；源状态和脚本另存 `*.before_resume_20261107`。最终所有本轮训练进程与拥有的Judge服务均清理，GPU1–4、6–7已释放，无后台训练继续运行。

本轮工程验证已完成。正式实验按上方命令由用户启动；第3组仍需完整naive542日志离线选参及用户确认，`rubric.actor_reward_selection_confirmed=false`保持不变。Actor性能收益需完整正式训练与统一评测验证。
