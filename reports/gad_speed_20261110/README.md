> 迁移发布说明（2026-11-14）：本目录仅携带历史报告摘要；文中原始日志、冻结副本和附件保留在源服务器，未随Git交付。当前发布与验证状态见 `reports/gad_n4_selection_migration_20261114/README.md`。

# GAD n=4 提速测试（2026-11-10）

状态：阶段 0–3 的功能验证已完成，各阶段显存已在续训步补测；首两步的连续显存采样有缺口，不能声称三步峰值记录完整。正式 naive v2 和正式评测均未启动。

对应计划：[GAD_n4提速测试与naive重跑计划](../../docs/20261105/GAD_n4提速测试与naive重跑计划.md)。

## 执行边界与可行性

- 开始时 GPU 1–4、6–7 均空闲，未发现残留训练/Ray/Judge 服务；原 naive v1 已由用户停止。历史目录保留，不删除、不续训。
- 所有 GPU 训练短测均为独立 `speed_*` run，每次最多 2 步；合并验证为独立 naive 2 步、完整状态续训 1 步。训练用 1–4，Judge 用 6–7，0/5 不运行模型。
- **本轮 GPU 实验始终 `replay.enabled: false`**。Replay 仅在 CPU 回归用例中启用，源码和测试保留。
- B1 可行：保存 epoch 起点的私有 sampler RNG，并恢复后快进；不改变不中断时的 Torch 题序。缺少状态的旧检查点不允许同阶段精确续训。旧 Warmup271 仅作为显式阶段交接导入，保留原交接题序，原 `data.pt` 不改写。
- B2/B3 可行：Teacher 预取、Student 提前提交复用原标签构造、缓存键、失败屏蔽、请求账本和重试上限。Correctness 仍在主线程计算，后台只处理 Judge 请求和结果；任何优化器更新前等待标注结果。
- 可选 B4 本轮不采用，保持先 Critic 后 Actor 的更新顺序。

## 已确认的准备验证

- CPU 旧/新交接对照：272、273、274 步各 64/64 个 `question_id` 与正式 baseline 一致，见 [sampler_identity_precheck.json](sampler_identity_precheck.json)。实际 GPU 的 272–274 步各 64/64 身份核对也已通过。
- 独立临时候选后端最初 12 项针对性 CPU 测试全部通过，见 [candidate_backend_verification.json](candidate_backend_verification.json)。后续增加 EOS 回归：修复前缓存命中 0/3，修复后 3/3，见 [修复前](eos_regression_before_fix.log)、[修复后](eos_and_speed_after_fix.log)。
- 最终完整 CPU：452 项用例，447 通过、5 项可选跳过，另有 2 项 scorer 自测通过，见 [cpu_verification.json](cpu_verification.json)。最终生成后端上额外执行 replay、resume patch、algorithm parity、process sampling 四套回归，合计 27 项通过。
- 32 条已有真实轨迹的 correctness 主/后台评分完全一致。主线程 0.262 秒，后台 4.484 秒；因此保留主线程评分，见 [scorer_thread_parity.json](scorer_thread_parity.json)。该检查没有新增 Judge 请求。
- 计划禁止修改的 Judge 客户端、服务端、语义模块、direct 协议、spec 和 prompt 哈希均保持原值。

## 阶段 1：A1/A2 已完成

| 档位 | Offload | KV utilization | 两步均值（秒） | 相对同日 K0 |
|---|---|---:|---:|---:|
| K0 | 全关 | 0.55 | 889.818 | 对照 |
| K2 | 全开 | 0.85 | 877.460 | 快 1.39% |
| K3 | 全开 | 0.90 | 865.861 | 快 2.69% |

三档各两次真实更新均无 OOM，题目身份与 baseline 对应步一致，GRPO 重算差异及优势变化均为 0。未达到 10% 门槛，**保留 K0**；不改正式配置的 offload、KV 和梯度检查点。K1 的回退条件未触发，可选关闭梯度检查点测试的采纳前提也未满足。较大 KV 池作为主要瓶颈的判断未获本轮证实。详见 [a1_verification.json](a1_verification.json)。

固定 32 条 Judge 对照：并发 8 为 362.257 秒、31/32 解析成功；并发 16 为 343.748 秒、31/32 解析成功。两档无 OOM，各有一次 Thinking 预算耗尽。并发 16 只快 5.11%，低于 15% 门槛，保留并发 8。两档标签逐项一致性仅报告、不作采纳门槛。详见 [a2_verification.json](a2_verification.json)。

## 阶段 2/3：实现与功能验证完成

实际代码已合入；naive、Actor-rubric 两个 n=4 配置只增加 Teacher 预取和提前标注执行开关。baseline 配置、3090 四卡预设、训练数据、奖励/损失和抽样规则不改。

完整 CPU 初次有一个旧 hook 夹具缺少新增计时接口，补齐接口后全部通过；额外部署副本回归的启动器也已重置 unittest 的 argv。原日志保留，没有删改断言或跳过失败项。

第一份合并 GPU 尝试 `speed_naive_20261110` 在任何 Critic/Actor 更新前主动停止：发现 worker 从模型 generation config 使用多 EOS，预取需要读取同一配置。旧尝试、旧后端和预留的 11 次请求保留，见 [中止证据](stage3_attempt1_stopped_before_updates.json)。新增 EOS 回归后完整 CPU 再次通过。最终短测为独立 `speed_naive_20261110_eos`，新后端 `f0ce22a97d40`，仍为 2+1；旧 11 次预留从阶段 3 的 192 次上限中扣除，新尝试上限为 181 次。

最终独立 run 为 `speed_naive_20261110_eos`，真实完成 272、273 两步后，从完整状态续训 274；没有重复已提交的更新。外层调度器和 GPU 采样器提前退出后，训练子进程仍正常保存了 273（stage returncode=0），本轮只补跑剩余一步。恢复记录见 [stage3_controller_recovery.json](stage3_controller_recovery.json)。

- 三步各 64/64 个题目身份与正式 baseline 对应步一致；GRPO 实际重算误差、优势变化均为 0。
- 每步 32 个标注名额（Teacher 11、Student 21），Teacher 预取命中均为 11/11；账本齐全，成功短测新增 96 次发送，加上旧尝试预留 11 次，共 107/192。解析 94/96，最终失败 2 条的两个过程标签均按无效屏蔽，correctness 继续独立由 math scorer 产生。
- Actor、Critic 四卡所有 optimizer step 和 scheduler 从 2 恢复至 3；RNG 状态存在，数据游标推进 64 题/1 batch，原 Warmup `data.pt` 哈希未变。四个 Critic 头及 Actor 抽查权重在续训后都有更新。
- Replay 始终关闭；所有三步都完成 rollout、Critic、Actor，最终保存到 274，未出现 OOM。验证不涉及正式 Actor 性能。

详见 [功能与显存补测结果](stage3_phase_probe_verification.json)、[原始模型/优化器检查](rubric_naive_verification.json)。

### 耗时与分布

| step | rollout 秒 | 可见标注等待秒 | Critic 计算秒（扣等待） | Actor 秒 | 整步秒 |
|---|---:|---:|---:|---:|---:|
| 272 | 538.0 | 271.4 | 181.1 | 114.6 | 1192.7 |
| 273 | 499.3 | 275.2 | 176.8 | 113.4 | 1159.3 |
| 274 | 455.8 | 346.1 | 173.4 | 102.3 | 1172.6 |

三步均值中，old-log-prob 29.2 秒、ref 27.0 秒、reward 24.7 秒。上述计算与后台 Student 标注重叠，因此完整标注时长不能再次与可见等待相加。所有原生计时键、逐步值和均值均保存在耗时 JSON 中。

前两步同口径：旧 naive 均值 1241.237 秒，新 naive 1175.984 秒，降幅 **5.26%**。三步均值 1174.840 秒，271 步线性外推约 **88.4 小时**；旧 naive 前两步外推约 93.4 小时。Teacher 预取和提前提交确实与训练计算重叠，但长尾 Student 请求仍留下等待，当前没有实现计划中 53–68 小时的乐观推算。

这些是早期短测；历史对照没有同日重跑，Judge 输出和后续轨迹有随机波动，边界保存也计入两步，正式 dev 验证、长跑波动和额外重试均未计入。不能把上述外推作为完整训练承诺，或将全部差异归因于调度。逐环节原值见 [timing_comparison.json](timing_comparison.json)。

272 步新/旧 naive 均为 C=20、W=48、T=188，截断率 0.734375、平均 6931.625 token；正式 baseline 为 C=25、W=41、T=189、U=1，截断率 0.73828125、平均 6965.680 token。只报告分布，不作质量门槛，见 [step272_distribution.json](step272_distribution.json)。

### 显存证据的范围

首两步的采样器提前停止，无法恢复其更新阶段的真实显存峰值。**完整逐步采样检查仍记为未通过**，见 [strict_gpu_sampling_check.json](strict_gpu_sampling_check.json)；没有回填或插值，也没有把缺失样本当作显存为零。原严格分析入口保留，另设带缺口报告的分析入口。

续训 274 步用每 3 秒采样完整覆盖 rollout、Critic、Actor、checkpoint 四个阶段。下表只表示该续训步观察到的各卡物理显存峰值（MiB），不冒称是三步全程峰值：

| 阶段 | 训练卡 1 / 2 / 3 / 4 | Judge 卡 6 / 7 |
|---|---|---|
| gen | 15374 / 15374 / 15374 / 15374 | 23202 / 23202 |
| update_critic | 13452 / 13534 / 13424 / 13638 | 23202 / 23202 |
| update_actor | 22892 / 22992 / 23848 / 23054 | 23202 / 23202 |
| save_checkpoint | 22892 / 22992 / 23848 / 23054 | 23202 / 23202 |

三秒采样可能遗漏更短的瞬时峰值。三步成功退出与无 OOM 日志独立保留。

## 历史保护说明

开始前保存了 708,544 个历史 run/后端文件对象的 `lstat` 元数据快照。前后核查已通过：changed=0、missing=0；它不是所有历史权重的完整内容哈希。

发现原 pipeline 会在每次启动时 `touch` 已存在的 `.gad_backend_used` 空标记。现已改为仅首次创建；本次 K2 启动产生的标记 mtime 变化已明确回滚，见 [backend_marker_touch_rollback.json](backend_marker_touch_rollback.json)。没有修改模型、检查点或历史后端代码。

当前原始进度见 [stage1_status.json](stage1_status.json)、[stages23_status.json](stages23_status.json)。

## 后续运行

本轮自有训练/Ray/Judge 进程已清理，正式训练和评测没有启动。用户后续启动 naive v2、50 步诊断和选参命令见 [commands.md](commands.md)。第 3 组仍需完成 naive 后的选参与用户确认；冻结扩展和补丁生成器，不自动放开占位参数。
