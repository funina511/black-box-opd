# Warmup 与 GAD 训练提速修改计划

日期：2026-09-25。状态：性能统计与隔离短跑入口已实现；KV 0.50 试验在 step 104、critic micro=2 试验在 step 102 按用户要求中断。后续性能短跑暂停，由用户直接调整参数并启动正式实验；正式训练参数尚未改动。

执行说明：基线完成 7 个新增步骤；KV 0.50 和 critic micro=2 均未到最终保存步，因此只作为部分观察，不作为完整验证。短跑参数与正式配置分离。

## 1. 目标与约束

优先降低在线生成及判别器更新耗时，再决定 GAD 是否使用更多 GPU。所有收益必须由短跑数据确认。

第一轮保留以下设置：

- 原版判别器损失、Warmup SFT、GAD PPO/GRPO/KL、更新顺序及优化器步数口径。
- train_batch_size=64、max_prompt_length=2048、max_response_length=8192。
- n_warmup=1、GAD n=8。当前 Warmup n=1 是已采用的调优设置，不宣称与原始 n=8 相同。
- 学习率、训练数据、教师回答、温度、top_p、thinking 模式与验证指标。
- actor token 预算 10240、参数/优化器 offload、enforce_eager=true。
- 已落地的每 200 步 quick 验证、阶段末 full 验证和每 50 步保存。

不把降低候选数、截短回答、跳过判别器更新或缓存跨步骤判别器输出混入“纯性能优化”。这些都会影响训练内容或目标，另行讨论。

## 2. 实测基线

来源：LMOps/runs/gad_qwen3_1.7b/qwen3-1.7b-4x3090/logs/warmup.log，核查时完成约 132/271 步，最近 20 步约：

| 环节 | 秒/步 | 占比 |
|---|---:|---:|
| 生成及对应阶段开销 timing_s/gen | 278 | 64% |
| 判别器更新 | 86 | 20% |
| actor 更新 | 36 | 8% |
| old_log_prob、ref、reward 等 | 34 | 8% |
| 总计 | 434 | 100% |

- 最近回答均长约 6000 tokens，约 55% 达到 8192 上限。
- 前 9 步回答均长约 1434 tokens、每步约 194 秒，因此不能用训练早期的速度衡量当前阶段。
- SeqKD 训练时不生成学生文本；Warmup 每步必须生成负例并训练判别器，两者耗时不能直接横比。
- 当前 Warmup 尚未到第 200 步验证，慢的主因不是验证。SeqKD 的新 quick 验证实测约 12.5–13.9 分钟，最终 full 约 50.4 分钟。
- 日志没有发现 OOM、Traceback 或 NCCL error，训练在持续前进。

GAD 当前没有运行实测。按 batch=64、n=8、4 卡计算，每步 512 个回答、每卡 128 对判别器输入；Warmup 为每步 64 个回答、每卡 16 对。GAD 的 actor 使用所有候选，不能按 Warmup 去重后的 SFT 工作量估算。简单乘 8 的约一小时/步仅作为预算风险提示，不作为性能承诺。

## 3. 第一阶段：建立可比较的测量与安全短跑入口

### 3.1 性能统计

新增 gad/tools/analyze_training_perf.py，读取现有日志和 validation_math_metrics.jsonl，输出 JSON 与 Markdown 摘要：

- 区分 stage、正常训练步、保存步、验证步及启动预热。
- 汇总 step/gen/generate_sequences/reshard/update_critic/update_actor/ref/reward/old_log_prob 的均值、中位数和尾部耗时。
- 同时输出回答长度、截断率、prompt 数、候选数和估算生成 tokens/s，避免把“生成更短”误认为引擎更快。
- 单独统计 quick/full 验证及生成、ROUGE、数学评分耗时。
- 保存实际配置快照、代码/补丁指纹、GPU 数量、所用 checkpoint。
- timing_s/gen 包含 generate_sequences 等子项，不将两者相加重复计算；分布式日志聚合值和单卡峰值显存分开标注。

输出到对应 run 的 reports/performance/。不覆盖历史基线，更新实验 PERFORMANCE.md 的结果索引。

### 3.2 短跑隔离

新增顶层 scripts/benchmark_train.sh，复用现有 pipeline，不创建第二套训练器。

- 正式训练已安全退出后，在维护窗口顺序执行候选试验；不与正式训练争用 GPU 或大量 CPU offload 带宽。
- 以同一完整 checkpoint 的副本为起点，为每个候选创建独立 run_id 和输出目录。
- Warmup 同阶段恢复模型、优化器、scheduler、RNG、dataloader，保持四卡；不能只加载 HF 权重就当作同起点比较。
- 如果恢复点为 S、测量目标为 K 步，正确设置绝对目标 S+K；显式处理 stage_epochs 优先级，防止一次性能试验意外跑完整 epoch。
- 每次至少执行真实 actor 与 critic 更新，建议前 2 步预热、后 5 步测量；出现波动时再扩展样本，不无条件延长。
- 关闭短跑内的重复验证；短跑不写正式阶段的 COMPLETED 标记，也不冒充完整训练阶段。
- 固定 prompt 顺序、起点与随机种子。不同引擎调度仍可能生成不同文本，报告实际 token 数并结合 tokens/s 比较。
- GPU 短跑之前完成配置解析、Hydra composition 和路径检查。

## 4. 第二阶段：优先改善在线生成

### 候选 A：增加 KV cache 可用空间

```yaml
rollout:
  gpu_memory_utilization: 0.50  # 基线 0.40
```

先测试 0.50；仅当显存与吞吐结果支持、且仍有明显收益空间时再测试 0.55。不要同时改变 enforce_eager、候选数、回答上限或 TP。

实施位置：experiments/qwen3_1.7b 的候选配置；现有 pipeline 已传递该参数，无需改生成算法。

记录：

- 引擎初始化时实际 KV 容量、并发容量；不能只使用预检的估算值。
- generate_sequences 与整个 gen 阶段耗时、真实生成 tokens/s。
- 调度抢占/重计算信息（若本地 vLLM 提供）、各卡显存峰值。
- 首次生成、进入反向传播、保存 checkpoint 是否成功。

vLLM sleep 释放 KV 不代表可以无限提高预算；仍需覆盖引擎初始化、唤醒、权重同步及与其他模型共存时的显存峰值。

保留条件：相近生成长度下吞吐与总步时有稳定改善，且完整训练步无 OOM。改善不足或初始化/训练不稳定则回退到 0.40。不得仅凭 GPU 利用率升高宣布成功。

## 5. 第三阶段：减少判别器微批开销

### 候选 B：保持静态分批，micro=1 → 2

```yaml
optim:
  critic_dynamic_bsz: false
  critic_micro_batch_size_per_gpu: 2
```

- 先独立对比基线，再与通过验证的候选 A 组合。
- 维持原始成对判别损失、样本顺序与一次 optimizer.step 的样本总量，不修改判别器更新频率。
- 核对各阶段归一化后的 critic minibatch 可被 micro 整除。
- 当前 Warmup 每卡 16 对样本，micro=2 时从 16 个微批降到 8 个；当前 GAD 每卡 128 对，从 128 个降到 64 个。
- 当前 forward_micro_batch_size_per_gpu 由 critic 的 micro 配置插值，改为 2 同时影响 compute_values 前向；验收同时测 reward 和 update_critic。
- 在长教师/长学生样本上测试峰值显存，不能只用短回答证明可行。

验证：使用固定输入、关闭测试模型中的随机 dropout，对 micro=1/2 的成对损失与累积梯度做容差比较；真实训练保持原 dropout 设置，不承诺浮点或随机轨迹逐位一致。不同于 actor 的 token-mean 分批，判别器是样本对均值，应保留对应样本数缩放。

先只试 micro=2；micro=4 不进入首轮。若不稳定或收益不明显，回退到 1，不连带撤销已经验证有效的 KV 配置。

## 6. 第四阶段：在 GAD 开始前验证规模效应

Warmup 完成后，从最终 actor/critic 合并产物建立独立的 GAD 性能试验：

1. 四卡、n=8，基线配置执行真实生成、GRPO、critic 和 actor 更新。
2. 测试已在 Warmup 通过的 A/B 组合，但不假定其在 GAD 自动适用。
3. 保持额外步数语义：GAD 从 Warmup step S 开始，短跑目标为 S+K，正式目标仍由原 epoch 预算决定。
4. 原始交接默认只使用合并权重并继承 step/dataloader；不要错误启用“读取 Warmup 优化器作为 GAD 全状态恢复”。同阶段 GAD 重启才使用 --resume-gad。
5. 根据实测步时估算 813 个新增步骤与剩余验证时间，报告范围，不再用 Warmup 乘 8 代替实测。

如四卡的实际总时长仍不可接受，再进入下一阶段，不直接把正式流水线放任运行数周。

## 7. 可选第五阶段：GAD 使用八卡

前提是 GPU 资源可用，且用户选择该资源方案。此阶段保持全局 batch=64、n=8 和训练预算。

```yaml
resources:
  n_gpus_per_node: 8
  gpus: "0,1,2,3,4,5,6,7"
```

必须在 GAD 初始化边界执行：

- 完成四卡 Warmup → 合并 actor/critic → 八卡重新初始化 GAD 的 FSDP 参数分片。
- 不能用 --resume-gad 或 Warmup 同阶段恢复直接读取四卡 optimizer/model 分片作为八卡状态；当前 checkpoint 文件名含 world_size，原地跨卡数恢复不受支持。
- 检查 dataloader 状态与 global step 正确继承，HF critic 判别头完整，新的八卡 checkpoint 能在八卡恢复。
- 独立 run_id 保存八卡分支，明确配置和数据来源；不触发 SeqKD/Warmup 重训，不覆盖四卡产物。
- 测试单次更新和保存/恢复后，再估算收益。每卡样本数下降不意味着吞吐必然翻倍，PCIe/FSDP 通信可能抵消部分收益。

实施时可增加阶段级资源覆盖；如果只需要一次八卡 GAD，优先使用独立 GAD 配置与 --from-stage gad，而不是扩展大量配置机制。

## 8. 非首轮代码优化与日志修正

### 8.1 修正 Warmup 微批警告的含义

当前 Warmup actor 用师生长度之和分批，而其 SFT 前向只使用教师文本。预算 10240 下出现“single sample 16xxx tokens”不等于一次前向超过模型上下文，也不等于 OOM。

先增强日志，区分 actor/critic、学生/教师实际长度、组合预算与单样本例外。若要改为按教师长度分批，必须作为独立优化实验：原 actor token-mean 按微批样本数累积并非对任意分组等价，不能仅改 compute_teacher=True 就称为无损。

### 8.2 Warmup 附加计算

old_log_prob、ref、reward/adv 路径是否存在仅供诊断的可省计算，逐项追踪依赖后再决定。当前这些环节占比远小于生成，不优先修改原更新路径，更不直接删除 GAD 所需的 old_log_prob/ref/KL。

### 8.3 暂缓事项

暂不升级 vLLM/FlashInfer、不重新开启 CUDA graph、不改训练回答上限、不改 GAD n、不异步更新或减少判别器更新频率。若 A/B 与八卡方案仍不足，再单独列出这些方案的训练语义与环境兼容性影响。

## 9. 配置、文件与交付物

| 文件/目录 | 计划修改 |
|---|---|
| experiments/qwen3_1.7b/train.yaml | 只在性能验收后更新选中的参数 |
| experiments/qwen3_1.7b/perf/ | 基线与候选配置、试验说明；不混用正式 run_id |
| gad/tools/analyze_training_perf.py | 新增日志分析与吞吐统计 |
| scripts/benchmark_train.sh | 新增隔离短跑入口，复用 pipeline |
| gad/tools/config.py | 补充 critic micro 整除检查及必要的短跑参数校验 |
| gad/tools/pipeline.py | 短跑隔离、恢复预算和产物标识；避免更改性能配置触发已完成阶段重训 |
| gad/tests/ | 性能解析、恢复预算、配置传递、判别器梯度累积测试 |
| experiments/qwen3_1.7b/PERFORMANCE.md | 更新实测表与选型依据 |
| runs/.../reports/performance/ | 各试验配置、数据统计、GPU 记录与结果报告 |

正式参数只维护一份；候选配置明确继承/覆盖规则或由工具生成，不能引入多份互相矛盾的运行配置。

## 10. 当前任务与回退

- 先完成只读统计与 CPU 侧开发，保持活动训练代码不变。
- 比较收益和重启成本：Warmup 已进行过半，如果基准试验耗时超过可能节省的剩余时间，可让其按当前配置完成，优先在 GAD 阶段验证优化。
- 需要切换时先等待最新完整 checkpoint。当前更新 step 与 checkpoint step 不同；不要在未保存时为改参数强行中止。
- 修改文件不会热更新到运行中的 Python/Ray 进程；切换必须通过明确的恢复启动。
- 保留当前四卡 world_size、模型与数据路径、原配置快照；回退恢复的是最近完整状态，不是只有模型权重。
- 任何失败候选保留日志，不覆盖正式 COMPLETED、handoff 或 latest marker。

## 11. 验收与顺序

实施顺序：日志分析工具 → 安全短跑入口 → 候选 A → 候选 B → A+B 组合 → GAD 四卡实测 → 可选八卡 → 正式配置与文档。

验收必须包含：

1. 原版核心损失和更新顺序未被修改；样本数与 optimizer step 口径可核对。
2. 短跑只执行设定的新增步骤，不污染正式 checkpoint 和阶段完成标记。
3. 微批改动通过固定输入的损失/梯度测试；真实 GPU 完整执行生成、两路判别器训练和 actor 反向。
4. 报告实际 tokens/s、正常步耗时、峰值显存、长回答比例和验证耗时，不用不同长度的单步耗时直接比较。
5. Warmup 的有效配置在 GAD n=8 下再次测试；八卡方案必须通过保存与恢复。
6. 以实测结果决定是否保留每项改动。没有明显收益的候选不合入正式配置。
