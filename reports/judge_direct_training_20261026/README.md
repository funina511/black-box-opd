> 迁移发布说明（2026-11-14）：本目录仅携带历史报告摘要；文中原始日志、冻结副本和附件保留在源服务器，未随Git交付。当前发布与验证状态见 `reports/gad_n4_selection_migration_20261114/README.md`。

# direct_v1 真实训练与恢复验证（2026-10-26）

**工程验证通过。Warmup12步→GAD新增4步已完成，GAD在累计14步保存退出后，完整恢复到16步。** GPU训练探针墙钟约77.5分钟（含模型加载、交接、恢复及原停服清理）；未运行完整benchmark。训练GPU1–4，Judge GPU6/7，现已释放全部本轮进程。

本轮验证了监督进入训练和状态恢复，**没有解决上轮已知标签误判，也没有验证Actor性能收益。** 当前仍是隔离候选，未把新协议合入正式训练入口。

## 实际配置

- 原始`dapo_math_17k_v2`：17405条，batch64，prompt2048/response8192，Actor non-thinking。
- Qwen3-8B Thinking，TP2、并发8，上下文32768、输出预算8192；每条仅一次请求，无语义复核、自动重试或标签修复。
- 保留用户加速设置：Actor/Critic/reference关闭offload，Actor/Critic开启梯度检查点，Critic静态microbatch2，rollout.enforce_eager=false。
- 每个全局逻辑optimizer minibatch抽样16条唯一轨迹；实测每个driver step只有一个逻辑minibatch。Warmup8 Teacher+8 Student，GAD5 Teacher+11 Student；全部16步均满足。
- BT保持全量，correctness覆盖原本可评分轨迹；三维辅助损失lambda0.1、tau4、固定分母3，Actor奖励仍只取adv。Replay关闭。
- 工程探针关闭评估，未修改正式评估默认值；保留每50步保存、最近5次设置，并通过已有工程步数接口在12/14/16强制保存。

## 监督与更新证据

Warmup更新12次Critic、3次Actor（10/11/12）；GAD更新4次Critic和4次Actor。所有已提交step连续且唯一，损失和梯度日志均为有限值。

| 梯度诊断 | Progress共享主干梯度范数 | Error共享主干梯度范数 |
|---|---:|---:|
| Warmup step1 | 7.844 | 7.339 |
| GAD step13 | 3.968 | 6.819 |

每个阶段只在首次optimizer更新前做一次有限梯度诊断，恢复后没有重新从零触发诊断。诊断显式排除score头，以上确属共享主干梯度，不是只看到head梯度。

HF交接保留`adv/answer_correctness/verified_progress/unresolved_error_severity`四头。初始score为FP32、交接导出为BF16；控制加载精度转换后，两个过程头仍分别有284/273个元素改变；主干`q_proj.weight`的一个4×32切片也有23个元素改变。原始变化量可能包含精度转换，不能直接当作纯训练增量；独立梯度诊断与BF16控制共同支持实际更新。详见`head_transition.json`。

## 保存与恢复

- Warmup12、GAD14和GAD16均通过原有checkpoint完整性、扩展manifest及标签缓存hash检查。
- 四个rank的Actor和Critic优化器step、scheduler last_epoch均由2延续到4；不是重新初始化优化器再跑两步。各rank保存cpu/cuda/numpy/random RNG，恢复日志记录实际加载。
- 扩展状态由GAD stage_step2/global14延续到stage_step4/global16；学生空监督计数状态以`same_stage_checkpoint_restore`恢复。
- GAD累计Judge局部计数32→64，跨阶段总账192+64=256；成功请求ID没有重复，Warmup和GAD已完成步骤未重做。
- 未选中轨迹的过程mask保持0。第6/11步各1条Thinking耗尽，两个过程mask为0、correctness独立保留，技术失败未永久缓存。

详见`optimizer_resume_checks.json`、`extension_resume_checks.json`、`resume_evidence.log`和`analysis.json`。

## 真实成本

| 项目 | 实测 |
|---|---:|
| Judge真实发送 / 预算 | 256 / 260 |
| 可解析 / 两个过程维有效标签 | 254 / 各254 |
| Thinking预算耗尽 | 2（Warmup6/11） |
| 格式失败 / 重试 / 二次复核 / OOM | 0 / 0 / 0 / 0 |
| HTTP峰值并发 | 8 |
| 单次API耗时中位数 / P95最近秩 | 38.1s / 144.0s |
| prompt / completion tokens | 828,038 / 297,249 |

可解析或程序有效不等于数学正确。254条均获得两个非null过程标签；没有据此宣称总体标注准确率。

下面单位为秒。Judge列来自请求调度墙钟；Critic列扣除了包含Judge和本地评分的整个annotation时间。两个诊断步的Critic列仍包含额外梯度诊断，不能用于稳态速度比较。rollout列为gen总计，不再重复加其内部generate/reshard；表中没有逐列列尽ref、logprob等小段。

| 阶段 | step | 总计 | rollout | Judge | Critic扣标注 | Actor | 保存 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| warmup | 1 | 358.8 | 13.6 | 190.6 | 150.6 | 0.0 | 0.0 | 一次梯度诊断 |
| warmup | 2 | 128.8 | 13.1 | 92.2 | 20.4 | 0.0 | 0.0 | 普通步 |
| warmup | 3 | 285.5 | 15.2 | 248.6 | 19.0 | 0.0 | 0.0 | 普通步 |
| warmup | 4 | 214.5 | 20.7 | 171.2 | 19.6 | 0.0 | 0.0 | 普通步 |
| warmup | 5 | 164.6 | 19.1 | 123.2 | 19.2 | 0.0 | 0.0 | 普通步 |
| warmup | 6 | 377.6 | 14.4 | 338.1 | 22.3 | 0.0 | 0.0 | 普通步 |
| warmup | 7 | 229.1 | 18.2 | 188.7 | 19.1 | 0.0 | 0.0 | 普通步 |
| warmup | 8 | 154.7 | 14.6 | 117.0 | 20.1 | 0.0 | 0.0 | 普通步 |
| warmup | 9 | 363.9 | 14.6 | 327.2 | 19.3 | 0.0 | 0.0 | 普通步 |
| warmup | 10 | 202.7 | 19.2 | 143.0 | 21.3 | 15.9 | 0.0 | Actor开始更新 |
| warmup | 11 | 364.8 | 13.9 | 310.9 | 19.2 | 17.7 | 0.0 | 普通步 |
| warmup | 12 | 189.5 | 13.2 | 129.9 | 19.8 | 16.3 | 7.3 | 保存 |
| gad | 13 | 344.0 | 21.5 | 131.7 | 175.9 | 6.8 | 0.0 | 一次梯度诊断 |
| gad | 14 | 195.2 | 30.0 | 107.9 | 37.7 | 6.0 | 7.4 | 保存 |
| gad | 15 | 251.0 | 25.9 | 164.5 | 48.4 | 5.3 | 0.0 | 恢复后首步 |
| gad | 16 | 260.0 | 15.4 | 181.3 | 45.8 | 4.9 | 7.4 | 保存 |

Warmup非诊断/非保存步中位总耗时221.8s，其中Judge约180.0s；这里多数步处于Actor尚未更新的critic warmup区间。GAD唯一非诊断/非保存观测是恢复后step15：总251.0s，Judge164.5s，整个annotation165.2s，扣标注后其余约85.8s。

**额外标注不超过训练耗时50%的目标，在这次早期探针中没有达到。** GAD15按内部时间拆分约为165.2/85.8=192%；这是同一步内部估算，不是另跑baseline得到的因果对照。只有4个GAD更新且包含初始化、保存和恢复，不能外推成熟Actor整轮耗时。后续若降成本，可按计划只比较一次8条抽样候选；本轮没有修改16条设置或追加新一轮GPU实验。

训练卡3秒采样峰值最大24124MiB（约23.6GiB），Judge为22660MiB（约22.1GiB）。这些值包含初始化、CUDA Graph和恢复，并非普通更新稳态显存，也不是精确瞬时峰值。完整采样保存在`gpu_samples.jsonl`。

## 收尾与保护

原停服命令漏继承GAD_PYTHON，默认Python缺yaml，软停止返回1；已有按PID/birth归属的fallback随后清理本轮Judge组，GPU已释放。只修复独立探针驱动的解释器继承，未改训练算法。指定解释器的相同停服命令已返回0并清理已退出进程的残留owner；没有为此重启Judge或重跑训练。原错误和fallback结果保留在`status.json`，复查见`cleanup_interpreter_recheck.log`。修复后的运行中服务软停止没有另外复测。

190个受保护文件中186个与启动前一致。`evaluate/rescore.py`、`evaluate/test_extended.py`、`evaluate/summarize_eval.py`、`evaluate/evaluate.py`在探针期间发生变化，本轮工具没有写入或回滚这些文件。用户随后确认：这4处改动只影响新增评测数据集，与本轮训练验证无关，无需为此补测，也不构成后续合入的阻碍。原始hash差异保留记录，不能把190项全称为hash一致；本轮相关训练/scorer源码保护检查通过，详见`protection_final.json`。

## 仍未回答的问题

- 上轮已知的符号改写被算作进展、严重递推错误被低估等质量问题仍存在，本轮未继续修改prompt或扩大模型搜索。标签未经人工验收。
- Actor最终准确率收益需用户运行同条件完整baseline/correctness/direct joint；本探针不作收益结论。
- 历史baseline/correctness训练scorer为`2a2b8a02…`，本候选为修复后的`f3549b87…`。数据相同不能消除这一混杂，正式比较需匹配scorer或明确局限。
- 新实现仍在`judge_fix/direct_v1_20261025/source/LMOps/`，尚未合入正式入口；已有正式run、backend和默认协议未被替换。

## 复现入口

- 配置：`../../judge_fix/direct_v1_20261025/configs/training_probe_20261026.yaml`。
- 驱动：`../../judge_fix/direct_v1_20261025/tools/run_training_probe.py`；已运行的目录禁止直接重新启动，以免重置历史。脚本的停服环境继承修复发生在本次训练完成后，原日志保留。
- 汇总：`../../judge_fix/direct_v1_20261025/tools/analyze_training_probe.py`，仅读取证据，不启动推理或训练。
- 完整执行证据：`warmup_*`、`gad_first2_*`、`gad_resume2_*`、`server.log`、`generation_budget.json`及实际run目录：`/mnt/jinbo/RLRM/LMOps/judge_fix/direct_v1_20261025/source/LMOps/runs/direct_v1_probe_06b17k_20261026`。
