> LMOps / RUBRIC-GAD / FORMAL 17K

# 附录 A 实现修正与附录 B/C 计划收敛提纲

*—— 17k 正式实验后续执行草案*

> 版本 v1.0 2026-10-20 状态：待确认
> 只读核查后的执行提纲；代码尚未按本提纲修改

| 项目 | 内容 |
|---|---|
| 文件性质 | 执行提纲（待确认）；不是完成报告，也不是新的实验计划 |
| 核查范围 | 附录 A 的实现正确性；附录 B/C 的后续计划合理性 |
| 核查依据 | 当前源码、40 套件 / 386 项 CPU 回归、现有数据统计与验证记录 |
| 保护边界 | 未修改代码、数据、缓存、已部署 backend 或运行中的 baseline_06b_17k |

> **阅读提示**
>
> 本提纲先固定“已确认问题—修复动作—验收条件—B/C 计划调整—执行顺序”。确认后再进入代码修改；在此之前，现有实验、数据、缓存和运行中的 baseline 保持不动。

## 1 目标与边界
本提纲把已确认的问题、对应修复动作、验收条件、附录 B/C 的计划调整和执行顺序收敛为一份可审阅的执行草案。它不替代正式实验计划，也不把“CPU 测试通过”写成“实现已全部正确”。

### 1.1 目标

- 修正附录 A 中已确认的实现缺口，优先处理恢复路径和标签选择这类会改变实验结论的问题。
- 把附录 B 从方向性补验计划收敛为可执行的评估身份修复与性能窗口安排。
- 把附录 C 从目录设想收敛为可验证的依赖内置、资产复制和打包验收方案。
- 在正式 joint 冻结前，形成一份清晰的“已修复 / 未完成 / 待用户决定”清单。

### 1.2 不做的事

- 不改变 Judge 数学判据、模型、采样参数、loss 归一化、BT/Adv 目标或 Actor 更新路径。
- 不启动正式 17k 长跑、GPU 性能窗口或真实 Judge 请求。
- 不修改 baseline_06b_17k、历史数据、缓存、已部署 backend 或现有 checkpoint。
- 不把 B/C 的迁移动作与算法/性能改动混成一个大补丁。

## 2 核查结论摘要
当前 A 的主体实现已经落地，CPU 全量回归为 40 个套件、386 项 unittest、0 失败、0 skip。但“全绿”只说明当前 CPU 路径没有回归；仍有恢复路径、配置校验和测试可移植性方面的缺口。

*表 1 核查结论与优先级*

| 优先级 | 事项 | 当前状态 | 结论 |
|---|---|---|---|
| P1 | N3：GAD resume 绕过 extension state | 未完全实现 | handoff-step 的 partial checkpoint 可被当成 Warmup handoff |
| P2 | S1：correct-nontruncated 判断顺序 | 有缺陷 | 正确但截断的候选被误判为 unresolved |
| P2 | N4：spec/version 一致性 | 部分实现 | math_v1 + v2.7 spec 组合可通过校验 |
| P2 | N10：测试可移植性 | 未完成 | 仍依赖 /mnt/jinbo 下的 8B/14B 模型 |
| P3 | J1：重试上限边界 | 需明确 | 当前 ledger 按 step 隔离，是否要求全 run 累计待定 |
| 策略 | A.6 完全相同重复记录 | 需明确 | 当前也标记 incomplete，是否允许去重后完整待定 |

## 3 附录 A 的实现修正项
以下项目按“先失败复现、再修复、最后回归”的顺序执行。修复不改变数学判据；只修正恢复边界、候选选择顺序、启动前校验和测试夹具。

### 3.1 N3：GAD resume 与 extension checkpoint P1
**现状：** resolve_gad_resume() 的早期 handoff 分支只检查 actor/critic/data.pt，没有传入 extensions_required=True，也没有要求 handoff_initialization.json。

**影响：** 一个位于 handoff step、但缺少 training_extensions/ 的 GAD checkpoint 会被当成 Warmup handoff，以 fresh optimizer 重新起步；这违反 N3 的完成/恢复边界。

**动作：** required=True 时，只有存在 handoff_initialization.json 才允许走 handoff 分支；否则必须走 last_complete_checkpoint(..., extensions_required=True)，并在缺 extension state、marker 或 manifest 引用文件时明确失败。

**验收：** 伪造 handoff-step partial checkpoint 无 extension state，不得返回 mode='handoff'；合法 Warmup handoff 和合法 extension checkpoint 仍正常；baseline 无 extension 不受影响。

### 3.2 S1：correct-nontruncated 的候选顺序 P2
**现状：** pick_teacher_candidate() 先检查 unresolved，再检查“correct but truncated”；因此 unresolved 在前、正确但截断在后时会抛 UnresolvedCandidates。

**影响：** 一条明确的“正确但被截断”候选被升级成整批 unresolved，改变数据构建结果和报告原因。

**动作：** 先检查 first_ok(require_correct=True, require_complete=False, track_unknown=False)，确认是否存在正确但截断候选；再决定是否抛 unresolved。保留 correct 与 correct-nontruncated 各自语义。

**验收：** unresolved + correct truncated → correct_but_truncated；只有 unresolved → 结构化 unresolved；后续正确且未截断候选 → 正常选择；select=all 不变。

### 3.3 N4：spec/config/prompt 版本一致性 P2
**现状：** extension_config.validate() 只在 rubric.version == math_process_v2 时读取 spec；rubric.version=math_v1 + 当前 v2.7 spec + legacy scorer 可以通过 validate_config()。

**影响：** 启动前不报错，运行时才发现协议不匹配；N4 的“启动前失败”目标没有完全实现。

**动作：** 启用 rubric 时校验 spec['version'] == cfg['rubric.version']；v2.7 继续校验 schema_version、input_protocol；prompt 显式声明继续校验。拒绝 math_v1 + v2.7 spec 混搭，保留 math_v1 + 旧 spec 的兼容路径。

**验收：** 当前 v2.7 正常；旧 v2.6 spec 失败；schema 匹配但 input protocol 不匹配失败；config version 与 spec version 不一致失败；所有失败发生在 Judge 请求之前。

### 3.4 N10：测试可移植性与失败回归 P2
**现状：** run_all.sh 已纳入 rubric/tests；但 test_appendix_bc.py 和 test_thinking_judge.py 仍从 /mnt/jinbo 加载 8B/14B 模型，且不是 optional/skip。

**影响：** 本机全绿只因为模型恰好在；迁移到干净 LMOps 目录后测试会失败，和 N10 的“不依赖 /mnt/jinbo 绝对模型路径”冲突。

**动作：** 用临时 tokenizer fixture 替换绝对模型路径；必须加载真实 tokenizer 的测试改为显式环境变量控制并 skipUnless；补 S1、N3 resume、J1 step 边界、N4 config version 的失败回归。

**验收：** 干净目录下 40/386 仍全通过；无 /mnt/jinbo 绝对模型依赖；失败用例能证明修复前失败、修复后通过。

### 3.5 J1：重试上限的语义边界 P3
**现状：** AnnotationJournal 的目录包含 step_<step>；同一请求在 step 1 失败后，step 2 会得到新的 journal 目录，recover() 返回空，重新从 attempt 1 开始。

**动作：** 明确“累计”是 per-step 还是全 run。若要求全 run 累计，把 attempt ledger 提升到 stage/identity 级；若只要求同 step 重启累计，在 A.2 文档中明确写出。

**验收：** 同 step 跨进程累计；若采用 run 级，同请求在 step 1/2 连续失败仍受同一上限约束；成功结果仍可直接复用。

### 3.6 A.6：完全相同重复记录的处理策略
**现状：** rescore 对完全相同重复记录也计入 duplicate_samples，并把结果标记为 incomplete；A.6 只明确要求冲突内容默认隔离并标记不完整。

**动作：** 明确“完全相同重复”是去重后允许完整，还是和冲突重复一样标记不完整。若允许去重后完整，调整 incomplete 条件；若保持保守，在验收中写明。

**验收：** 相同重复、冲突重复、缺失、越界分别有明确状态；不通过 clamp 或缩小分母提高指标。

## 4 测试与验收矩阵
每个失败回归先在修复前复现，再在修复后通过；不把“测试数量增加”当成“路径已覆盖”。

*表 2 修复与验收矩阵*

| 项目 | 必测场景 | 通过标准 |
|---|---|---|
| N3 | handoff-step partial checkpoint 无 extension；合法 handoff；合法 extension | 该失败的失败，该通过的通过，不进入错误 handoff 分支 |
| S1 | unresolved + correct truncated；仅 unresolved；后续正确候选 | 原因正确、选择顺序正确、select=all 不变 |
| N4 | v2.7；v2.6；version 混搭；坏 spec；缺 prompt 声明 | 启动前正确接受/拒绝，不产生 Judge 请求 |
| N10 | 干净目录、无绝对模型路径 | 全部 suite 可运行，无 skip 冒充通过 |
| J1 | 同 step / 跨 step 的确认失败与未知提交 | 按选定语义断言，不无限重试 |
| A.6 | 相同重复、冲突重复、缺失、越界 | 分母固定，状态明确 |
| A.3/E2 | 重复执行、任务中断、raw 完整只评分 | 零重复生成、只补缺失工作 |
| B.2/E4 | 只改 HF 权重、tokenizer、template | 能检测；相同文件重复命令零生成 |

## 5 附录 B 的计划修订
总体判断：B 的边界和顺序合理；B.2 是唯一实质代码修复，必须在评估 identity 冻结前完成。

### 5.1 B.1：保留现有证据边界

- 保留 40/386、四卡 correctness-only 链路和迁移证据。
- 不把“测试通过”写成“性能已改善”；不把用户中断写成 OOM。

### 5.2 B.2：E4 的 post-export identity

- 区分“源 checkpoint 身份”和“导出后 HF 内容身份”；首次未导出时允许后者待定。
- export_stage() 成功后更新 identity，再写 plan.json；第一次与第二次相同命令不得因新增 HF 字段误判配置变化。
- HF shard/index、tokenizer、template 均纳入；不要只依赖 actor 顶层 hash。
- CPU 回归：只改 HF 权重、tokenizer/template 能检测；相同文件重复命令零生成；raw 完整、评分缺失仍只评分。

### 5.3 B.3：C2/C3 口径修正

- 把执行说明中的 1.85/1.23 小时修正为 3.70/2.47 小时。
- 明确“引用回写修正”不等于“有效标签已经增加”；保留旧缓存不混用的要求。

### 5.4 B.5/B.6：性能补验的可执行化

- 固定 --steps 4 --warmup-steps 1，对应 1 预热 + 2 正常 + 1 保存退出。
- 明确源 checkpoint 为 global_step_500，501 是首个观测 step，避免 off-by-one。
- B0/B1/B2 单变量对照；180 分钟窗口内不扩展排列组合；失败保留退出码和诊断。
执行顺序建议：B.2（E4 + 回归）→ B.3（文档修正）→ B.5/B.6（性能补验）→ 机器配置发布。

## 6 附录 C 的计划修订
总体判断：C 的方向合理，但目前更像迁移目标，不是可直接执行的验收方案。以下四项必须先补。

*表 3 C 计划相关的当前观察*

| 观察项 | 当前事实 |
|---|---|
| verl.bundle | 约 1.2 MB，可进入普通 Git |
| 四份评估 Parquet | 合计 124,409 字节，可进入普通 Git |
| 初始化资产 | 约 2.3 GB，应走完整发布目录 |
| gad/verl 跟踪文件 | 约 683 个，存在重复源码树 |
| 历史 reports 跟踪 | 当前 11 个文件被跟踪 |

### 6.1 C.2/C.3：bundle 的 refs 与重建

- 当前 bundle 只有 HEAD ref；应生成 seqkd/warmup/gad 三个显式 ref。
- 空目录 clone 后逐个 cat-file 验证三个 commit，再运行 prepare_backends。
- SOURCE.json 记录来源、commit、bundle hash、许可证和重建说明。

### 6.2 C.5：资产必须是真实文件，不是绝对软链接

- assets/models、assets/datasets、assets/initialization 使用真实文件或受控复制。
- 逐文件 hash；大资产不进入普通 Git；发布验收拒绝 /mnt/jinbo 绝对软链接。

### 6.3 C.7：打包工具重构

- package_runtime.py 读取 third_party/、assets/，不再把 OPRD 和数据直接塞进 runtime。
- 发布清单明确未提交变更、未跟踪文件、bundle、测试数据和环境说明。

### 6.4 C.6/C.8：删除顺序与干净目录验收

- 先生成并验证发布包，再按清单删除旧 backend；不先清理。
- 干净目录 CPU 重建、preflight、三阶段 dry-run、evaluate dry-run。
- 在 N10 绝对模型路径修好之前，干净目录测试会失败，不能把 C.8 标成通过。

## 7 执行顺序与冻结门禁
按“恢复边界 → 数据选择 → 配置校验 → 测试可移植性 → 评估身份 → 迁移工具 → 性能”的顺序推进。

1. **[A-P1]** 修 N3 resume 分支，补 handoff-step 无 extension 的失败回归。
1. **[A-P2]** 修 S1 候选顺序，补 unresolved + correct truncated 的夹具。
1. **[A-P2]** 补 N4 的 rubric.version 与 spec.version 一致性检查。
1. **[A-P2]** 处理 N10 的绝对模型路径，改为 fixture 或 optional skip。
1. **[A-P3]** 明确 J1 累计边界和 A.6 重复策略；补对应回归。
1. **[CPU]** 在干净目录重跑 40/386；确认无 skip 冒充通过。
1. **[B.2]** 实现 E4 post-export identity + 重复执行回归。
1. **[B.3]** 修正 C2/C3 文档口径；保留历史原始证据。
1. **[C]** 完成 bundle refs、资产复制、打包工具重构和干净目录验收。
1. **[B.6]** 在 180 分钟窗口内做 B0/B1/B2 单变量补验。
1. **[FREEZE]** 按门禁冻结：joint 需 R1–R6 + N1/N4/N5/N7 + N3/N10；评估需 E4；恢复需 N2/N3；数据需 N6。
冻结判断必须区分“能跑”“测试通过”“恢复正确”“标签质量已证明”和“性能已改善”。本提纲只处理前四项中的工程边界；最后一项仍由有限性能窗口回答。

## 8 交付物

- A 修复清单：文件、问题、修复、回归结果、未完成项及理由。
- 干净目录 CPU 回归日志：40 套件 / 386 用例，0 skip。
- N3、S1、N4、N10、J1、A.6 的失败前/修复后对照。
- B.2 E4 的 CPU 证据：首次导出、重复执行、HF 权重变化。
- C 的 bundle/资产/打包清单及干净目录重建记录。
- 一页“仍未解决 / 需用户决定”清单，避免把待定项写成已完成。

## 9 需要用户确认的点

*表 4 需要用户确认的策略项*

| 决策项 | 可选方案 | 影响 |
|---|---|---|
| J1 累计口径 | per-step / 全 run | 决定 ledger 放 step 还是 stage/identity 级 |
| A.6 相同重复 | 去重后允许完整 / 任何重复都不完整 | 决定完整性与指标发布边界 |
| N5 语义 | progress/error 两维独立 / error unknown 阻断 | 决定标签覆盖率与冻结口径 |
| C 完整包 | 携带按需 Teacher/Judge / 仅列待补资产 | 决定发布包体积与自包含程度 |

未确认前，建议保持当前默认：N5 两维独立；A.6 保守标记不完整；J1 先按 step 级实现并在文档中写明；C 的发布包先保证代码/依赖/评估数据自包含，模型资产按清单补齐。

## 附：核查依据（只读）

- [17k正式实验提速与迁移执行计划](17k正式实验提速与迁移执行计划.md)（附录 A/A.10/B/C）
- [verification.json](../reports/formal17k_20261020/verification.json)（40 套件 / 386 用例）
- [pipeline.py](../gad/tools/pipeline.py)（GAD resume、stage signature、checkpoint verification）
- [build_gad_data.py](../sample/build_gad_data.py)（correct-nontruncated 选择）
- [extension_config.py](../gad/tools/extension_config.py)（spec/config 协议校验）
- [test_appendix_bc.py](../gad/tests/test_appendix_bc.py)、[test_thinking_judge.py](../gad/tests/test_thinking_judge.py)（绝对模型路径）
- [evaluate.py](../evaluate/evaluate.py)、[rescore.py](../evaluate/rescore.py)（E2/E3/E4 相关路径）
- [package_runtime.py](../gad/tools/package_runtime.py)（bundle 与打包边界）

本文件为提纲，不代表任何代码、数据或实验状态已经改变。
