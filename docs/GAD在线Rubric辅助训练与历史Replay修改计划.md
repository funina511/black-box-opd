# GAD 在线 Rubric 辅助训练与历史 Replay 修改计划

日期：2026-10-12  
状态：实现交接文档，2026-10-12 修订；本次只修改计划，不修改训练代码、配置、数据或运行状态。

**最新统一口径：启用 rubric 的主方案从 Warmup 第一轮 critic 更新就开始加入 rubric 辅助监督，并在 GAD 中继续。Warmup 的 critic 同时适应当前 teacher/student 分布与固定 rubric 评分标准；仅 GAD 开启属于阶段消融，不是主方案。**

本计划依据用户最新决定，取代上一版“暂缓正式 GAD 接入”的范围约定。[离线第一版计划](Rubric辅助Critic第一版修改计划.md)保留为历史设计与已实现模块的参考，不再要求其全部研究门槛通过后才能开展在线实验。

## 0. 给实现者的执行摘要

用户决定将研究重心转向正式 GAD：最终通过 actor 数学能力提升验证 rubric 的价值，而不是把离线 critic 标签拟合当作最终目标。

按以下顺序交付，每一阶段都必须可单独验证：

1. **首先实现从 Warmup 开始的在线 rubric 辅助 critic 训练。** 各主实验组从相同初始学生模型及判别器初始化条件开始，Warmup 保留原 teacher SFT 和原成对判别损失，同时用独立 verifier 提供的固定 rubric 标签训练 critic；GAD 接续本组 Warmup 的 actor 和全部 critic 评分头，继续联合更新。
2. **其次实现独立可选的历史轨迹 replay buffer。** 池子在正式训练中逐步建立，保存完整、题目匹配的 teacher/student 历史比较，供后续 critic 更新；不是预先构建后长期不变的离线标注集。
3. 提供两个相互独立、默认关闭的功能开关，形成原版、rubric-only、replay-only、rubric+replay 四种组合。
4. 先验证 rubric-only 的 Warmup→GAD 闭环和短程 actor 对照，不以完成 replay 或获得离线显著增益为前提；Warmup 的联合训练与完整交接是首版必做项。
5. 原版开关全关时保持原行为；所有算法扩展走显式分支并记入配置与代码指纹。

**首轮研究假设：Warmup 中让 critic 同时学习 teacher/student 分布判别和 rubric 质量评价，使进入 GAD 的 critic 具备两类能力，再通过持续联合训练改善原判别奖励，进而改善 actor 学习。** 本版不将辅助评分直接混入 actor reward，不更换原判别标签，不同时引入 evo-rubric。若辅助头学好而 actor 不提升，应如实报告该假设未获支持。

## 1. 已确认的需求与非目标

### 1.1 必须满足

- 保留原版 teacher > student 的成对判别损失。
- actor 的 Warmup SFT 与 GAD GRPO/PPO 更新公式不变。
- Warmup 和 GAD 都支持在线标注；独立 verifier（代码中的 judge）评价当前回答，不按题目 ID 套用离线旧回答标签。它是冻结的监督标注器；critic 是持续学习的判别与质量评分模型，两者不是两套并列的 actor 奖励。
- critic 继续更新，不因 rubric 引入而永久冻结。
- replay 仅改变 critic 更新用的数据，不把历史轨迹带入 actor 的 policy loss。
- actor 最终数学评测是主结果；critic 标签预测/排序只是辅助诊断。
- 源 Warmup、SeqKD 结果和已有训练目录不覆盖；所有实验采用新 run。
- 复用 `LMOps/rubric` 已验证的 schema、评分器、损失和评分头迁移能力，不推倒重写。

### 1.2 第一轮范围

主方案的 `rubric.stages` 为 `[warmup, gad]`，总开关仍默认关闭。用户显式开启 rubric 后，必须从 Warmup 的第一轮 critic 更新加入辅助损失，再将训练好的全部评分头交接到 GAD。

- Warmup actor 从初始学生模型开始，按原版 teacher SFT 更新；SeqKD 仍是独立基线，不用其结果初始化 Warmup。
- Warmup critic 同时优化原成对判别损失和 rubric 辅助损失；没有独立的“先做完原 Warmup，再追加 rubric 预热”阶段。
- 原 `critic_warmup` 若暂时跳过 actor 更新，critic 仍执行原判别＋辅助更新，不等待 actor 解冻后才开始 rubric 训练。
- GAD 使用本实验自己的 Warmup actor 与四通道 critic；辅助头不得重置。
- `[gad]` 作为阶段消融，允许从共同的原版 Warmup checkpoint 开始，仅在 GAD 起点扩头。
- replay 首版仍只作用于 GAD；SeqKD 不启用任何 rubric/replay critic 功能。

### 1.2.1 固定 rubric、独立 verifier、可学习 critic 的职责

固定 rubric 是共享评分标准，不在本版由模型生成。独立 verifier（judge）按该标准提供过程评分标签，答案正确性由现有规则评分器提供；critic 学习这些评分，同时保持原判别任务。实际 actor reward 仍取 critic 的 adv 通道。

独立 verifier 可以由可调用的 teacher 模型承担，也可使用单独的已校准模型；“角色独立”不要求模型型号不同。首版仍按独立服务部署，以隔离生成式标注与 FSDP 训练。

critic 未来生成、演化 rubric 及直接使用 rubric reward 的研究不在本版；当前辅助头训练不会自动产生 rubric 文本生成能力。

### 1.3 暂不实现

- rubric 文本进化、题目级规则生成、生成式 critic。
- rubric 替换原标签、排序反转、teacher 错误样本自动剔除。
- 辅助分数直接混入 actor reward、规则正确性奖励混入 reward。
- step-level/process reward、修改 GRPO 标准化、改变 critic/actor 更新顺序。
- judge 与训练模型在同一组 GPU 上自动交替装卸的复杂调度。
- replay 的重要性采样、优先经验回放、teacher 最佳回答重选。

## 2. 本次代码核查：必须据此接入

| 当前位置 | 事实 | 实现要求 |
| --- | --- | --- |
| `gad/tools/pipeline.py::build_overrides` | 映射全局 YAML 到各阶段 Hydra 参数；指定 GRPO | 扩展参数必须真正传入后端并能在 resolved 配置中核实 |
| `gad/tools/prepare_backends.py` | SeqKD/Warmup/GAD 来自不同固定 worktree | 算法扩展必须同时覆盖实际 warmup 与 gad worktree，不修改 SeqKD 算法 |
| `gad/tools/backend_patches.py` | 当前工程补丁约定不改算法 | 新建显式算法扩展补丁组，不偷偷塞进原工程补丁 |
| `.backends/gad/verl/trainer/ppo/ray_trainer.py::fit` | 新采样→原 critic 评分→优势→critic 更新→actor 更新 | 保持该顺序，新辅助梯度对下一批奖励起作用 |
| 同上 | `uid` 是运行时 UUID，按 n 重复 | 区分 GRPO 组 ID 与持久化题目/回答 ID |
| `.backends/warmup/verl/workers/actor/dp_actor.py` | teacher response 上的 SFT loss；不是 rubric policy reward 更新 | rubric 只修改 critic 联合目标，不把 GAD 的 actor 损失复制进 Warmup |
| Warmup `ray_trainer.py` | 先更新 critic，再按 `critic_warmup` 门槛决定是否更新 actor | critic 辅助任务从阶段首步生效，原 actor 门槛保持不变 |
| `.backends/gad/verl/workers/critic/dp_critic.py` | `_forward_micro_batch` 最终只保留末有效 response token 分数 | 多头必须在保留该语义的基础上扩展 |
| 同上 | `update_critic` 使用固定字段白名单 | 新 label/mask/weight 字段必须显式通过 select 与微批切分 |
| `fsdp_workers.py` | num_labels 固定 1，mini batch 乘 n 后按 DP 大小归一化 | 多头加载及分布式归一化必须一起改 |
| `prepare_gad_checkpoint.py` | config 修复强制 num_labels=1，检查单头形状 | 需要认识 1/4 两种 schema，不能把多头改回单头 |
| 原 `_save_checkpoint` | 保存模型与 dataloader 后写 latest tracker | replay/扩展状态必须在发布“完整 checkpoint”前写完 |
| `rubric/train_offline.py` | 单卡、固定回答；actor 不更新 | 复用纯函数，不直接把离线主循环塞进 Ray worker |
| `rubric/inference.py` | 当前本地 judge 进程直接初始化 vLLM | 不能在每个训练 rank 或每步重新创建 judge 引擎 |

当前全局默认是 0.6B、GAD n=2、Warmup n=1、batch64、2048/8192。已有 run 的保存配置优先；实现时再次检查，不根据默认值重写历史实验。

### 2.1 离线结果的使用方式

24 题实验已完成工程闭环但每组只更新一步；512 题实验尚无训练结果。512 题中 762/1536 回答截断，两个过程标签有效率不足一半；judge JSON 最终有效率约 92.1%，人工审阅为零。

这些事实提示需要改进格式约束与监督覆盖诊断，但**不要求继续完成大规模离线训练才允许进入 Warmup→GAD 小步测试**。先用初始学生的当前轨迹做必要的 judge 校准；Warmup 末及 GAD 初复查覆盖率，不能只依据旧 Warmup 模型上的标签质量。

## 3. 功能矩阵与关闭语义

下表描述 GAD 阶段；主方案的 Warmup 使用相同 rubric 开关和联合损失，但 replay 不启用，actor 仍以 teacher response 做 SFT。

| rubric.enabled | replay.enabled | 模型/监督 | actor 数据 |
| --- | --- | --- | --- |
| false | false | 原单头、原成对损失 | 当前 rollout |
| true | false | 四通道、当前回答在线标签、联合损失 | 当前 rollout |
| false | true | 原单头、当前/历史成对比较混合 | 当前 rollout |
| true | true | 四通道、当前/历史比较及各自匹配标签 | 当前 rollout |

必须保证：

- 两开关关闭时，不加载 rubric/judge 的重依赖、不启动服务、不扩头、不建 buffer、不增加 RPC、不消耗额外采样 RNG。
- replay-only 无需安装/启动 judge。
- rubric 的 lambda=0 可作为诊断模式，但显式关闭是 `enabled=false`；不承诺 enabled=true 的扩头、请求耗时与 RNG 等同于原版。
- 未实现的开关组合、阶段、分布式模式必须在 preflight 报错，不能接受参数后默默忽略。
- 第一里程碑允许 `replay.enabled=true` 明确报“尚未实现”；实现完成前不得标称支持四种组合。

## 4. 配置与 CLI 契约

以现有 `LMOps/train.yaml` 和 `--set` 为唯一正式训练配置入口；不再建立一套平行训练主配置。新增默认值均不改变当前实验。

### 4.1 建议配置

```yaml
rubric:
  enabled: false
  stages: [warmup, gad]               # 主方案；[gad] 仅作阶段消融
  mode: joint                         # joint / correctness_only
  loss_coef: 0.1
  spec: rubric/specs/math_v1.yaml
  prompt: rubric/prompts/judge_math_v1.txt
  head_layout: [adv, answer_correctness, reasoning_validity, condition_coverage]
  reward_source: adv_only              # 首版只允许这个值
  label_source: online
  label_scope: selected_critic_batch   # replay 开启后只标注本步选中训练的比较
  teacher_cache: true
  student_cache: true                  # 按回答精确匹配；不是按 uid
  truncated_process_policy: mask
  on_annotation_error: mask
  on_service_unavailable: abort_step
  judge:
    backend: openai_compatible         # 首版连接显式配置的本地独立服务
    base_url: null
    model: null
    model_revision: null
    tokenizer_path: null
    api_key_env: null                  # 本地无鉴权可为空；绝不写明文密钥
    temperature: 0
    max_output_tokens: 768
    max_context_tokens: 16384
    structured_output: required
    concurrency: 8
    request_timeout_seconds: 120
    step_timeout_seconds: 600
    max_retries: 2
  seed: 42

replay:
  enabled: false
  stages: [gad]
  mix_ratio: 0.25
  max_age_steps: 8
  capacity_pairs: 1024
  max_bytes: 536870912
  sampling: uniform_without_replacement
  eviction: oldest_first
  insert_after_step: true
  store_labels: true
  seed: 43

resources:
  # 原 n_gpus_per_node、gpus 等继续控制训练
  judge_gpus: null                     # 只供可选本地 judge 启动脚本使用，须显式指定
```

以上 ratio、容量、超时、lambda 是保守试验起点，不是实验结论。replay 中任一容量/年龄限制达到就先淘汰最老条目；容量以比较对计，不以题目计。

`correctness_only` 仅使用规则标签，不请求过程 judge，但使用同一四通道布局以便消融。纯 joint 模式正常需要 judge；不得服务失败后自动退化为 correctness_only 并继续声称在做 rubric 实验。

正式 YAML 路径继续相对该 YAML 解析；judge tokenizer/spec/prompt 和代码指纹加入 run 签名。凭据通过环境变量传递，日志脱敏。worker 只需要模型头与辅助损失参数，judge URL/凭据应留在 driver/client，不广播给每个 rank。

### 4.2 参数映射

- `rubric.*` 的 driver 侧字段 → Hydra `trainer.rubric.*`。
- 启用状态、loss_coef、mode、layout、spec hash → `critic.rubric.*`。
- `replay.*` → `trainer.critic_replay.*`。
- judge 启动资源不计入训练的 Ray resource pool，也不计为 actor 数据并行卡数。
- stage 不在列表中时 runtime enabled=false；保留声明配置及“本阶段未启用”的记录。
- preflight 比较 driver 与 worker 的 layout/hash/coef，避免两端配置失配。

### 4.3 主方案的新实验与预算

主方案使用新 run，执行 `--stages warmup,gad`，从同一初始学生模型和同一判别器初始化条件开始。若执行完整 `seqkd,warmup,gad`，SeqKD 保持独立，不作为 Warmup 的初始化模型。

不能复用已经完成、未接受 rubric 监督的旧 Warmup，并把它称为“从 Warmup 开始启用”的主方案。旧结果可保留为参考或供仅 GAD 开启的阶段消融使用；主方案要重新运行相应 Warmup。

建议提供 `--gad-new-steps <K>`：当 Warmup 完成于绝对步 S，GAD 终点为 S+K。若 S 在启动时还未确定，计划先记录符号终点，在 Warmup 完整保存后解析，并更新 GAD 阶段配置快照。不要在训练启动前猜测不存在的 handoff checkpoint。

主实验各组固定相同 Warmup epoch/样本预算及 GAD 新步数。该参数必须处理 `stage_epochs` 对 steps 的优先级，dry-run 展示 Warmup 预算、预计/最终 S、额外 K 和绝对终点；例如 S=271、K=80 时，GAD 终点为351。

工程冒烟可显式将 Warmup 限制为少量步、随后交接少量 GAD 步；必须使用独立 smoke run，并标明提前结束，不能覆盖完整 Warmup 的完成记录。效果实验使用预先固定的正式 Warmup 预算。

### 4.4 仅 GAD 开启的消融入口

建议保留以下待实现接口，仅供消融或显式复用已完成的兼容 Warmup：

```text
--init-from-warmup-run <run path or alias>
--init-step <explicit warmup step>
--gad-new-steps <K>
```

`rubric.stages=[gad]` 的消融读取指定原 Warmup actor/critic，在新 run 中派生四头 critic，并只运行 GAD。输入若是已完成 rubric Warmup 的四头模型，则不得再次扩头，需严格验证 layout/hash。

两种情况都不覆盖源 run，也不冒称同阶段完整恢复。`--resume` 只允许相同阶段及相同算法配置的完整状态恢复；不能用它把关闭 rubric 的历史 run 改成开启状态。

## 5. 正式训练数据流与更新顺序

### 5.1 Warmup 数据流

```text
当前学生生成 student 回答，与离线 teacher 回答配对
    ↓
固定 rubric → 独立 verifier / 规则评分器 → 两侧辅助标签
    ↓
critic：原 teacher > student 损失 + rubric 辅助损失
    ↓
按原 critic_warmup 门槛决定是否执行 actor 的 teacher SFT 更新
    ↓
继续下一步；阶段末保存 actor 与完整四通道 critic
```

保留 Warmup 后端原有计算/更新顺序，即使其框架中仍计算 values/advantages，也不删除这些路径或让 rubric 分数成为 SFT 权重。Warmup actor 的训练目标、选取 teacher 数据的规则与优化器保持原样。

标签来自该步实际送入 critic 的 teacher/student 回答；不固定使用旧 Warmup 结束后的学生轨迹。teacher 可按精确回答缓存，student 逐步按新回答匹配标签。Warmup 默认 n=1，不能套用 GAD n=2 的重复数或 batch 分母。

### 5.2 GAD 数据流

以进入本步时的 critic 为 D_t，actor 为 π_t：

```text
π_t 生成当前 student 回答，与原 teacher 回答组成 F_t
  ├─ actor_batch：始终保持当前 rollout 的整批数据
  │    └─ D_t 评分 → 原优势计算 → 等待 actor 更新
  └─ critic_batch：F_t，或从 F_t 与历史池 P_t 选出的固定数量比较
       └─ 精确回答标签缓存/在线 judge → 联合损失更新 D_t → D_(t+1)
原 actor_batch + 原 advantages → 更新 π_t → π_(t+1)
成功完成本步后，将 F_t 写入历史池，得到 P_(t+1)
```

具体接入步骤：

1. 完成 rollout 的 repeat/union，形成正式 teacher/student tensor。
2. 建立稳定题目/回答引用，保留原运行时 GRPO uid；不改变当前 row 顺序。
3. 保持原 `compute_values(actor_batch)`，只返回原 adv 通道的末 token 分数。
4. 保持原 reward/KL/advantage 计算。
5. 独立构建 `critic_batch`：replay 关闭时就是本步当前比较；开启时执行第 9 节的混合采样。
6. 根据实际 critic 输入内容准备标签，driver 批量向 judge 发请求，检查和写缓存。
7. 将 label/mask/去重权重加入 `critic_batch`，调用一次原 critic worker 更新入口。
8. 用原 `actor_batch` 调用 actor update；绝不覆盖它的 ids、uid、old/ref logprob、advantages。
9. actor/critic 更新完成后提交 buffer 变更，保存扩展状态与指标。

第一版采用同步标注屏障，方便检查每步监督与版本。可以把请求与无依赖计算重叠作为后续性能优化；不能跨步使用不对应当前回答的标签。

本地代码是“评分→更新 critic→更新 actor”；参考论文采用其他次序的部分不照搬。仅 GAD 开启且共用原 Warmup 起点的消融，第一步 actor 分数应与对应 baseline 相同或在迁移数值误差内。主方案使用各自已完成联合训练的 Warmup critic，GAD 第一步奖励就可能不同，这是预期效果，不能套用前述同分断言。

## 6. 真实轨迹、ID 与 DataProto 契约

### 6.1 三类 ID 必须分开

- `group_uid`：当前 GRPO 组 UUID，保持原逻辑，只服务本步优势分组。
- `prompt_id`：原样本索引 + 规范化题目 hash，用于长期匹配。
- `response_id`：prompt hash + 实际回答 token/text hash + tokenizer/template/version，用于标签缓存。
- replay 另有 `pair_id = hash(prompt_id, teacher_response_id, student_response_id, producer_step, occurrence_id)`，区分不同历史比较条目。

dataset 的 extra_info.index 不能当成回答 ID；GRPO 随机 UUID 也不能当永久 cache key。

### 6.2 用实际模型可见内容标注

- student 文本从实际生成 token IDs 解码；不能把 decode 后再 encode 的近似序列替换回训练数据。
- teacher 使用实际传给 critic 的 token 序列及其可见文本。teacher index 的完整原文只用于溯源；若训练截断，不拿原始完整解答标签套在前缀上。
- 记录 prompt 是否因长度被截断；缺失题目关键信息时不能让 judge 看完整原题、critic 看残缺题目却不标注差异。
- `finish_reason`、生成达到上限、teacher 原采样截断与本次 token 截断分别保存。当前 rollout 若未输出这些字段，应在 rollout/DataProto 输出新增元数据，而不是把所有长度达上限都当成正常结束。
- EOS 与 PAD 即使 token ID 相同，也要使用实际有效 mask 判断。
- unknown、不支持的题目类型、judge 失败、上下文超限和截断各有独立原因码。

### 6.3 critic_batch 的新增 tensor

建议在原 teacher/student 字段旁增加：

```text
student_rubric_targets       float32 [B,K]
student_rubric_mask          float32/bool [B,K]
teacher_rubric_targets       float32 [B,K]
teacher_rubric_mask          float32/bool [B,K]
student_aux_weight          float32 [B]
teacher_aux_weight          float32 [B]
```

K 固定为 3；correctness_only 通过 mask 关闭其他维度。targets 中 unknown 填有限占位 0，但 mask=0；禁止 NaN 乘零后传播。

non_tensor metadata 保留 prompt/response/pair IDs、source_step、是否 replay、label version，必须与 row 重排同步。`select_keys`、DataProto 分片、Ray 序列化和微批切分都要覆盖新增 tensor。

global 辅助维度计数、有效 pair 数和 logical optimizer minibatch 标识放在 meta_info 或等价显式结构。不能靠 worker 每个 microbatch 自己猜分母。

## 7. 在线 Judge：服务与标签策略

### 7.1 首版部署

采用独立、长期存活的本地 vLLM 服务，通过显式 OpenAI-compatible 地址访问；训练 driver 负责请求和缓存，所有训练 rank 不加载 judge。

新增 `scripts/rubric_judge.sh`，支持 health、start、stop（仅本脚本拥有的 PID）、日志和模型指纹。命令参数须以本机已安装 vLLM 版本核实，不能直接假设最新 structured-output CLI 可用。

- judge GPU 与训练 GPU 必须明确且不重叠。原离线实验用 judge 1,2,3,4 的配置不能直接套到同时占用这四张卡的正式训练。
- 可示例使用训练 1,2,3,4、judge 0,5,6,7，但只是配置示例，启动前检查实际资源，不能默认这些卡空闲。
- judge 模型、tokenizer、revision、精度、模板和输出 schema 固定并入指纹。
- 网络端点是本地或显式用户配置的服务；不自行选择外部付费 API。dry-run 不发送题目、不启动模型。
- 服务已启动时核对模型和版本；只检查 HTTP 200 不够。
- 不每步重新加载 judge，不在 Ray worker 里临时创建另一个 vLLM 引擎。

首版若独立显存不足，应报告所缺资源；选择较小 judge 是新配置/新实验，须重新校准，不能静默替换。GPU 同卡分时调度暂不支持。

### 7.2 在线标注内容

保持固定 math_v1 三个维度：

- 正确性：现有 CARD 对齐规则评分器；acc0/1，不把 ±1 score 当 BCE target。
- 推导有效性、条件完整性：匿名题目+单条回答+固定 rubric，judge 输出0/0.5/1或unknown及证据。
- judge 不知道 teacher/student 身份，不接收原 critic 分数，不要求复现 teacher 解法。
- GT 可作为评分依据，但不进入 actor/critic 的正常输入。

继承已有 label schema 和审计记录。使用 JSON schema/约束解码（经本机版本验证）；输出解析错误不能通过默认高分或简单正则猜值来修复。

### 7.3 批量、缓存和失败处理

- 对本步实际进入 critic 更新的回答去重，一条 teacher 不因 n 个 student 请求 n 次。
- cache key 包含实际可见 prompt/response、GT、rubric/spec、judge/template/decoding hash。
- 成功确定标签缓存；最终答案规则标签另记 scorer hash。
- 稳定语义 unknown（如不可判断）可以缓存；网络超时/服务失败只记录失败，不永久固化为该回答 unknown。
- retry 有上限，按 request_id 返回结果，不能依赖网络返回顺序对应 batch 行。
- 单条失败默认 mask 对应过程维度；正确性标签如可得仍保留。整步服务不可用/超时则在任何 optimizer update 前终止本步，不默默运行成无 rubric 训练。
- first implementation 只在启动及 optimizer 边界做请求，不容许一半 rank 更新、一半 rank 等 judge。
- 截断的过程维度首版保持 mask，记录 teacher/student 各自覆盖率；之后若研究“前缀评分”需新版本和独立实验。

### 7.4 标注校准与覆盖诊断

无需再完成 512 题离线 critic 实验。先用当前 actor 的约32条回答检查格式、证据、正误和长度分层，至少包含 student 错误与截断案例。

- 结构化成功率、teacher/student 各维有效率、全同分率、进程耗时必须可见。
- 运行时持续记录缺失原因；unknown 多不等于 judge 工程失败，但意味着辅助监督覆盖有限。
- 连续若干步 student 过程监督接近零时，输出明确警报/在诊断边界结束，不把完整训练结果描述成“充分验证过程 rubric”。
- 质量人工抽查是必要诊断，不机械要求完成全部512题离线审计才允许GPU冒烟。未复核数据明确标为探索性。
- 不为了提高覆盖率把截断回答直接判0或只筛选短正确回答。

## 8. 多头 Critic 与联合损失

### 8.1 模型结构及迁移

复用 `rubric/model.py` 的扩头语义：共享骨干，输出顺序 `[adv, correctness, reasoning, conditions]`。只支持已核实的 Qwen2/Qwen3 TokenClassification 分支；其他架构先拒绝。

主方案在 **Warmup 启动前** 确定四通道结构，并生成/记录可复现的初始化产物；不等到 GAD 交接时才第一次扩头。

必须区分初始模型类型：若从原学生 CausalLM 初始化 critic，原评分头本来需要按原版规则随机初始化，允许这一明确的头初始化过程；若从已训练的 critic checkpoint 加载，则 score.weight/bias 必须严格存在，不能把缺失评分头误当正常初始化。原骨干权重加载缺失仍然是错误。

1. 按原版初始化规则建立/严格加载单头 critic，记录模型来源、score 初始化 seed 和形状。基线与 rubric 组的初始骨干及 adv 行必须一致；不能让新增随机数消耗改变 actor 初始化或 rollout RNG。
2. 扩头时第0行精确复制，其他行固定 seed 初始化，隔离 RNG 以免改变 actor 抽样。
3. 保存新配置/权重及初始化/迁移 manifest，源模型与 checkpoint 不变；同一原版初始化条件可用确定性 fixture 核对，不要求从已有 Warmup 取 adv 行。
4. FSDP worker 根据 num_labels/layout 严格加载四通道，移除无条件强制1的逻辑。
5. 在 FSDP/optimizer 建立前完成结构确定。meta tensor/rank0 广播路径必须通过真实多卡测试，不能只单卡检查。

也可实现 worker 内扩头，但必须证明每rank一致且未对 meta tensor 做无效复制；两种方法择一，禁止重复扩头。

Warmup→GAD 是四头到四头：严格加载已训练的完整评分头，不调用单头扩头函数、不重置辅助行。仅 GAD 消融才在原单头 Warmup 后首次扩头。

### 8.2 forward 和 reward 接口

建议新增内部函数返回所有通道 logits，原评分 wrapper 仍取第0通道，输出 `[B,response_length]`，仅末有效 token 非零。

`compute_values(actor_batch)` 的外部返回值不改变。辅助头只在 critic update 或抽样诊断中使用，不作为 baseline、reward、advantage 注入 actor。

remove-padding 分支目前有 `squeeze(-1)` 等单头假设；要检查 unpack、pad_input、gather、response slicing 的 `[tokens,4]` 维度。不能临时关闭 remove-padding 又不在各实验组保持一致。

### 8.3 损失定义

原判别损失保留：

\[
L_{adv}=\frac{1}{B}\sum_{i=1}^B-\log\sigma(d_{T_i}-d_{S_i}).
\]

对当前逻辑优化 minibatch 中的唯一可见回答 U，维度k的有效 mask为m，标签y：

\[
L_k=\frac{\sum_{u\in U}m_{uk}\operatorname{BCEWithLogits}(z_{uk},y_{uk})}
{\sum_{u\in U}m_{uk}},\quad
L_{aux}=\frac{1}{|K_{valid}|}\sum_{k\in K_{valid}}L_k,
\quad L=L_{adv}+\lambda L_{aux}.
\]

correctness_only 只激活第一个辅助维度。unknown mask掉；某维全无标签不进入平均；全部无标签时辅助梯度为零并记录。

teacher 仍可能错、student 仍可能对；本版保留原成对偏好，只记录冲突分层。辅助头与判别头可以冲突，不能在实现时悄悄改标签缓解冲突。

### 8.4 去重、微批和多卡归一化

teacher 回答在多个 pair 中重复时，原 pair loss每对保留；辅助损失按唯一回答计一次。student 若文本完全重复，按同一可见输入ID去重辅助监督，pair multiplicity仍保留。

推荐 driver 对每个逻辑优化 minibatch 计算回答出现次数，给每次出现附 `1/multiplicity` 权重；同一回答即使出现在不同 rank，也只能合计一次辅助权重。

设数据并行梯度平均的组大小为 W，维度全局分母 C_k，rank r 的辅助加权分子为 N_rk，则该 rank 在 backward 前应采用 `W * N_rk / C_k`（再按有效维度平均），使平均后的梯度等于全局均值。若实际后端的 reduce 语义不同，应按真实规则推导并测试，不套用错误的总 GPU 数；SP/FSDP 组不能混为一谈。

要求：

- 分母覆盖整个 optimizer minibatch 和全部 rank，不按microbatch局部平均。
- 原 adv 的静态梯度累积缩放保持原公式；aux 自己累积完整全局归一化贡献，不再重复除一次 accumulation。
- 首版约束 `critic.use_dynamic_bsz=false`、SP=1、critic ppo_epochs=1，且本地 batch 可整除 microbatch，符合当前默认；其他组合在支持前 fail-fast。
- 配置允许多个 optimizer minibatch 时，每个 minibatch独立分母/去重，不能按整个 rollout 分母缩放每次 optimizer.step。
- 各rank微批数量一致，无有效标签的rank也走相同计算/collective顺序，避免FSDP死锁。
- 统计只取局部mean再平均不足以验证loss正确；必须对照单进程整批梯度。

## 9. 历史轨迹 Replay Buffer

### 9.1 目的与参考边界

参考 [Persistent Negatives for Adversarial Black-Box On-Policy Distillation](https://arxiv.org/html/2609.30864v1)：在固定大小的判别训练批中加入历史、题目匹配的teacher/student比较，同时让actor使用当前rollout。

这里采用“保持当前总比较对数”的设计，避免把单纯增加critic更新量当成replay效果。不声称严格复现论文全部超参数或更新次序。

### 9.2 存储内容

池位于driver CPU，存储完整比较对象：

- prompt_id、pair_id、teacher/student response_id；
- 实际可见prompt/token IDs、teacher/student有效token IDs、位置/掩码重建所需边界；
- 可解码文本、GT及scorer/schema版本；
- producer_stage、global_step、actor checkpoint/policy版本或step；
- finish/truncation信息、插入顺序、已存在的rubric标签和judge指纹。

不存GPU Tensor、梯度图、旧advantages、旧policy logprob、旧critic score作为新reward目标。压缩掉padding后保存，不能每条复制整块2048+8192的无效padding。

可以将teacher/prompt对象按hash共享以节省内存；缓存淘汰时正确处理引用，不删除仍被checkpoint引用的数据。

### 9.3 采样与淘汰

F_t 为本步新比较，B=|F_t|。先按最大年龄与容量清理 P_t，随后：

\[
k=\min(\lfloor \rho B\rfloor, |P_t|),\qquad
C_t=\mathrm{sample}(F_t,B-k)\cup\mathrm{sample}(P_t,k).
\]

- 无放回均匀抽样，专属 RNG；池不足时用当前数据填满，总数始终B。
- C_t构造完成后重新打包、分片；禁止把current batch某些tensor换掉而漏换teacher/label/id。
- 本步F_t只在本步critic和actor都更新成功后入池，不允许同一步刚入池又被当成历史抽中。
- 保留最近条目，年龄按本阶段成功更新步计算，GAD从Warmup继承global step时另记stage起点。
- ratio限制 `0 <= rho < 1`。首个replay实验建议0.25；rho0不抽历史。
- 受capacity_pairs、max_age_steps、max_bytes共同限制；单条超内存预算就不入池并计数，不截断其文本。
- 池内pair不需要与本步F_t的题目相同，但每条历史pair内部teacher/student必须属于同一题。
- 默认每阶段空池，不跨Warmup/GAD自动继承历史；同阶段resume必须恢复池。

### 9.4 与 rubric 组合

replay不依赖rubric；无标签历史pair依然可用于原判别损失。

rubric+replay时，先选出C_t，再对其中实际要训练的回答补齐有效版本标签。历史命中标签可以复用；未标注或缓存版本不匹配的回答在线请求，不按旧模型分数充当标签。

本步没有被选中训练的fresh比较也可以入池，带`labels_pending`；以后抽中时再标注。这样无需为了未来可能不用的数据提前支付judge成本。

若rubric版本/judge版本在同run内改变，应拒绝resume；需要新实验，不在训练中逐条偷偷切换评价标准。

### 9.5 对齐计算预算与性能

B和critic optimizer step数保持与无replay一致，但历史回答的token长度可能不同，因此不能称GPU计算量完全相同。记录current/replay token量、耗时、batch年龄、有效ratio与字节数。

critic_batch重新计算 `global_token_num`，不能沿用actor_batch的token元数据。如原MFU统计只统计student字段，应标明口径，并新增teacher/student实际token计数。

首版重新pad到与原backend一致的字段宽度，position_ids从保存的实际token边界正确生成；不得重新tokenize历史文本导致序列变化。后续更紧凑padding只能作为单独性能改动。

## 10. 状态保存、恢复与失败一致性

### 10.1 完整扩展 checkpoint

```text
runs/<run>/<warmup|gad>/global_step_<S>/
  actor/...
  critic/...
  data.pt
  training_extensions/
    manifest.json
    replay_state.pt                  # replay开启才存在
    rubric_state.json                # rubric开启才存在
    sampling_rng.pt
  checkpoint_complete.json
```

扩展manifest至少记录：版本、两个开关、head layout、rubric/judge指纹、source handoff、stage步数、配置/代码hash、buffer大小/内容hash、RNG状态引用、label cache快照/索引。

- 先保存各rank模型/optimizer/scheduler和dataloader，再保存扩展状态，校验完成后原子发布complete标志与latest tracker。
- 同步修改恢复查找、checkpoint完整性检查与清理工具；不能只写一个标志而原`find_latest_ckpt_path`仍挑中未完成目录。
- 两开关关闭时兼容历史无extension checkpoint；开启时从同阶段恢复缺失扩展状态必须失败。
- 新建 Warmup：按原模型初始化并扩头，建立新 optimizer；rubric 扩展状态从首步开始存在。
- 主方案 Warmup→GAD：加载全部四头模型，按原跨阶段约定建立 GAD optimizer/scheduler并继承正确的step/dataloader语义；不强行带入Warmup optimizer状态。replay池在GAD为空，标签缓存仅在完整指纹一致时复用。
- 仅 GAD 消融从原单头 Warmup 初始化时允许一次扩头；必须记录消融模式，不能当成“缺扩展状态的同阶段resume”。
- label cache若外置，checkpoint必须固定依赖条目的内容hash/快照；不能只存易被覆盖的SQLite路径。resume后不要求对所有缓存逐一重hash，但被引用条目须可验证。
- 缓存可保留超前完成的judge结果，因为纯标签缓存不等于已提交训练步；buffer和训练进度只恢复到最后完整checkpoint。
- 步中途失败不允许仅从失败位置继续：内存可能已更新critic但未更新actor，应丢弃未提交进度，从最后完整checkpoint恢复。

### 10.2 模型与优化器约束

- 原单头→四头是新run初始化，不恢复旧一行head的optimizer状态到四行参数。
- Warmup 或 GAD 的同阶段四头resume均严格恢复四头权重、optimizer、scheduler、dataloader与RNG；不得每次重新随机初始化辅助头。跨阶段交接与同阶段resume分开实现。
- replay-only仍是单头。
- 导出工具区分1/4输出，并保存id2label/layout/hash；禁止强制num_labels=1。
- 关闭rubric后直接resume四头run应报错；需要显式派生只保留adv的模型，不在首版自动做。
- 首版不支持改变world size的完整状态恢复；不得用HF初始化冒称exact resume。
- 检查tiny lr在当前FSDP精度/optimizer状态下产生真实参数更新，沿用原精度时报告验证结果；不把离线FP32主参数实现未经评估搬进所有正式worker。

## 11. 后端扩展组织与导入边界

建议新建 `gad/tools/algorithm_patches.py`，与既有 `backend_patches.py` 分开：

- 工程patch保持原约束；算法patch显式命名、幂等、anchor验证、编译检查、hash记录。
- 为实验建立独立 backend root，例如 `gad/.backends_gad_extensions_v1`，避免热修改现有任务使用的worktree。
- `prepare_backends.py` 接受明确的profile；默认 profile仍是原版。
- 扩展profile预装关闭时无副作用的rubric/replay分支；配置决定运行是否启用。
- preflight分别确认目标Warmup和GAD worktree的patch marker与版本，并用两阶段真实worker验证；只支持GAD不能标记主方案完成。未完成replay实现时拒绝开启。

算法模块建议部署在目标backend的 `verl/experimental/gad_extensions/`，源码由本仓库可追踪模块生成/复制，并校验hash；不能只改生成副本。

现有 `rubric` 文件存在裸导入及 `common import *` 等离线假设。提取可复用核心为明确的相对导入模块（例如 `rubric/core/`），在线worker只import loss/model/schema等轻量部分，不连带导入vLLM和离线pipeline。

无论选择显式安装包还是把模块同步进backend，都必须用Ray worker真实进程验证导入来源和hash；不能只验证driver里能import。沿用原backend guard，防止导入CARD或环境中另一份verl。

## 12. 逐文件实现清单

以下文件名是建议职责划分；实现者可以合并轻量模块，但不得省略职责和测试。

| 文件 | 修改/新增内容 |
| --- | --- |
| `LMOps/train.yaml` | 两开关默认false、配置注释；不改原训练默认值 |
| `gad/tools/config.py` | defaults、路径解析、阶段支持、数值范围、judge必填项与失配检查 |
| `gad/tools/pipeline.py` | 两阶段Hydra映射、初始模型→rubric Warmup→四头GAD交接、消融初始化入口、额外步数与签名 |
| `gad/tools/preflight.py` | 服务能力/模型指纹、GPU不重叠、head维度、backend功能支持、完整恢复检查 |
| `gad/tools/prepare_backends.py` | 独立算法profile、生成目标root、记录manifest |
| `gad/tools/algorithm_patches.py` | driver/critic/model/checkpoint/配置的显式扩展patch |
| `gad/extensions/online_rubric.py` | 当前实际轨迹适配、去重、标签join及mask |
| `gad/extensions/judge_client.py` | 批量请求、schema、timeout/retry、health、匿名输入、响应ID对齐 |
| `gad/extensions/label_cache.py` | 内容寻址、版本、瞬时错误、快照/恢复 |
| `gad/extensions/replay_buffer.py` | 完整比较对象、采样淘汰、容量、RNG、序列化 |
| `gad/extensions/critic_batch.py` | actor/critic批隔离、历史重建、跨rank去重权重、逻辑批计数 |
| `gad/extensions/checkpoint_state.py` | 扩展状态写入/读取、原子完整性、兼容检查 |
| `rubric/core/`及现有rubric文件 | 复用schema/loss/head迁移；保留原离线CLI可用 |
| Warmup/GAD backend `ray_trainer.py` | 分别保持原顺序，两阶段critic标注屏障与扩展checkpoint；GAD额外支持pool提交 |
| backend `dp_critic.py` | 多通道forward、第0通道reward、联合loss、新字段保留和正确缩放 |
| backend `fsdp_workers.py` | 根据schema加载1/4头、传入rubric配置、scheduler更新次数保持原样 |
| backend `ppo_trainer.yaml` | 显式扩展字段及默认关闭值，不滥用任意`+`绕过配置验证 |
| backend dataset/rollout | 保留稳定prompt ID和准确的teacher/student截断/结束元数据 |
| `gad/tools/prepare_gad_checkpoint.py` | 主方案四头Warmup→四头GAD严格交接；单头消融允许一次扩头；保留原版1头路径 |
| `gad/tools/merge_model2hf.py` / `export_model.py` | 检查多头config/权重一致，不能导出丢辅助头的模型 |
| `gad/tools/prune_checkpoints.py` / 原完整性判断 | extension完整性、缓存引用和受保护checkpoint |
| `scripts/rubric_judge.sh` | 独立服务启动/检查/停止，不杀其他用户/实验进程 |
| `gad/tests/test_online_rubric*.py` | 数据、标签、数值、恢复与调用顺序测试 |
| `gad/tests/test_replay*.py` | 历史池、计算批预算、actor隔离与恢复测试 |
| `docs/`、`README.md`、`gad/PIPELINE.md` | 可执行命令、功能状态、结果边界及恢复说明 |

文档和日志统一术语：固定离线数据集用于已完成预实验；历史replay池用于正式训练过去步骤的轨迹；标签缓存只是避免重复标注的存储，不等于replay策略。

## 13. Warmup 联合训练与 GAD 交接（首版必做）

### 13.1 Warmup 的两个 critic 学习目标

1. **分布判别适应**：原成对损失使 critic 学习当前 teacher/student 回答的相对分数。
2. **Rubric 评分适应**：辅助损失使共享骨干和三个辅助通道学习固定质量标准，涵盖答案正确性、推导有效性和条件完整性。

二者从第一轮 critic 更新同时发生，默认固定相同 lambda 贯穿 Warmup/GAD，不新增单独 rubric 预热阶段。后续如需分阶段 lambda，作为明确消融，不在首版悄悄调权重。

Warmup actor 仍按原 teacher SFT 更新；rubric 不改变其 loss、样本权重或teacher选择。训练循环原有 `critic_warmup` 门槛也保留：actor被跳过的步，critic的两项损失仍正常更新。不要把“actor跳过”误判成整步失败。

### 13.2 主方案交接契约

阶段末必须保存并转移：

- 本实验 Warmup 的 actor 权重；
- critic 骨干、adv 行和三个训练后的辅助行；
- layout、rubric/spec/judge/scorer指纹、源初始化信息；
- 原框架要求的step/dataloader等跨阶段交接信息。

GAD延续本组Warmup的模型状态；按原跨阶段规则建立optimizer/scheduler与reference，记录实际reference来源，不为rubric另改KL参考模型选择。不能混用A组actor和C组critic再称为主实验。

交接验收使用固定输入，比较Warmup导出前、HF中间产物与GAD加载后的四个通道，允许明确的dtype误差，不能只检查文件存在或adv一行。再验证GAD真实反向后辅助参数继续变化。

label cache可以在rubric/judge/GT/实际可见输入全部一致时复用；Warmup学生旧标签绝不按题目ID套用到GAD新回答。replay默认只在GAD建立空池，不自动带入Warmup历史。

### 13.3 阶段消融与不支持组合

- 主方案：`rubric.enabled=true`、`stages=[warmup,gad]`。
- 全关基线：`rubric.enabled=false`，两阶段走原逻辑。
- 仅GAD消融：`stages=[gad]`，可复用原Warmup；在GAD初始化时扩头。
- 仅Warmup开启、GAD关闭暂不作为首版必做项；输入四头而GAD按单头运行时必须明确报错，不能自动丢头。如后续实现，需独立定义adv导出及辅助冻结语义。
- SeqKD、Warmup replay首版不支持；阶段校验须按开关及实际执行阶段给出明确错误。

主方案以“Warmup联合训练→完整四头交接→GAD联合训练”整体通过作为功能完成条件，不能把Warmup支持推迟到replay之后。

## 14. 以 Actor 为中心的实验设计

### 14.1 第一轮主实验：两阶段启用 rubric

| 组 | Warmup critic | GAD critic | 要回答的问题 |
| --- | --- | --- | --- |
| A | 原判别损失 | 原判别损失 | 两阶段原版基线 |
| B | 原判别＋正确性辅助 | 原判别＋正确性辅助 | 仅答案监督能否改善actor |
| C | 原判别＋完整rubric辅助 | 原判别＋完整rubric辅助 | 过程rubric是否有额外actor收益 |

A/B/C均从相同初始学生模型、原判别头初始化条件及训练数据开始，各自执行完整Warmup，再进入本组GAD。B/C的辅助行初始化seed一致；A保持原单头路径。SeqKD仍独立，不作为Warmup起点。

固定两阶段的训练题数/epoch、样本顺序、rollout n、长度、温度、优化器、actor更新门槛与GAD新步数；reference遵循原框架相同构建规则。新增judge/head/cache使用独立RNG，尽量不扰动原actor随机数。

Warmup actor目标未改变，理论上不直接受critic奖励影响；数值执行和随机性可能导致差异，不声称各组actor必然逐位相同。必须在Warmup末分别评价actor并报告起点差异，同时记录critic原判别与rubric指标。

最终主比较是相同Warmup+GAD训练预算后的actor能力，代表整个方法的效果，不能全部归因于GAD阶段。另报告各组从自身Warmup末到GAD末的变化，但不能仅凭相对增量忽略不同起点。

### 14.2 阶段消融与分级预算

在主方案工程闭环通过后，可追加C-GAD：Warmup关闭rubric，只有GAD开启joint。与A共用原版Warmup来源以隔离GAD效应；C与C-GAD比较用于评估从Warmup开始适应rubric的价值。该消融不替代主方案、不作为主方案实施前置要求。

三级预算：

1. **工程冒烟**：独立smoke run中运行2–5个Warmup步，验证actor门槛前critic确实联合更新；另外覆盖门槛后的至少一次teacher SFT更新，再完成Warmup→GAD交接并运行2–5个GAD步。可用fixture验证门槛逻辑，但真实GPU闭环需确实观察两种actor更新。若原门槛为10，仅跑5步不能声称Warmup actor更新已验证。
2. **短程效果试验**：A/B/C使用相同正式Warmup预算，再各运行建议80个GAD新步。具体预算据实测耗时固定；不能将很短的smoke Warmup当成正式预热。额外报告共同初始模型、各组Warmup末和GAD末三个评价点。
3. **效果确认**：固定两阶段预算，至少3个seed并扩展评测。短程无显著差异不自动判方法无效；解释监督覆盖、reward通道变化和计算成本。

开发时可用扩头lambda0 fixture做数值回归，不必额外跑一套完整训练。无需先把离线指标调到显著改善；必要的judge格式/覆盖检查通过后，尽快运行两阶段actor对照。

### 14.3 主指标与统计

主指标：actor在独立数学题上的固定解码准确率，比较相同新增训练步后的结果及相对共同起点的变化。

- Warmup与GAD中quick-dev仅用于进度和预先声明的模型选择；记录共同初始、各组Warmup末、GAD末三个评价点，最终保留独立完整dev/外部数学测试评价。
- 固定greedy或相同采样协议，不能将rubric组更大评测n带来的收益算作训练收益。
- 同时报告输出长度、截断率、有效最终答案率、KL/熵、训练耗时、judge GPU-hours/请求token和总GPU-hours。
- 至少报告“相同Warmup预算＋相同GAD新增actor更新步数”的结果；judge成本包含两个阶段，不能漏算Warmup标注。额外补充相同总计算/时间预算的效率比较。
- 配对题目bootstrap与跨seed波动分开报告；不对多seed和多回答当作独立题目膨胀显著性。
- 明确主benchmark和训练预算，不在test上反复选择lambda或挑有利checkpoint。

辅助诊断：原adv对当前学生正确/错误回答排序、GRPO组内有效差异比例、rubric逐维覆盖与预测、教师错学生对冲突、梯度比例。

特别是当前n=2且按组标准差归一化，actor信号主要依赖两条新回答的相对排序；仅改善teacher-vs-student判别准确率或rubric MAE不必然改变actor更新。

### 14.4 之后再加入 replay

replay实现完成后追加：D在A的原版Warmup来源上只于GAD启用replay；E在C的rubric Warmup来源上于GAD启用rubric+replay。分别与同来源、同步数的A/C比较，避免用不同Warmup critic造成混淆。首轮不同时改变rubric与replay。

buffer改善critic稳定性不等于actor最终必然提升；同样以actor评价作为主结论。

## 15. 日志和诊断指标

在已有timing和训练指标之外新增：

```text
rubric/judge_requests, cache_hits_teacher, cache_hits_student
rubric/judge_seconds, queue_seconds, retry_count, json_valid_rate
rubric/valid_{dimension}_{teacher|student}
rubric/unknown_reason/*, truncated_teacher, truncated_student
rubric/loss_{dimension}, loss_aux, loss_aux_weighted
rubric/grad_norm_aux_sample, grad_norm_adv_sample   # 明确仅抽样诊断，避免每步额外全图求导
critic/loss_adv, teacher_wrong_student_right_count
replay/size_pairs, bytes, sampled_pairs, realized_ratio
replay/age_mean, age_max, evicted_age, evicted_capacity
replay/current_tokens, historical_tokens
training/actor_updated, critic_updated, stage_step, absolute_global_step
```

diagnostic值必须有分母和解释。actor reward日志只记录adv；rubric logits不能被误标成actor reward。原 d_acc 不使用全batch未mask的无关token求和；如发现历史诊断口径缺陷，保持loss原公式并单独修正/说明指标。

只保存必要的抽样文本和judge原始错误输出，完整轨迹/缓存有独立保留策略；不要每step复制所有长文本到多份日志导致磁盘迅速膨胀。

## 16. 测试与验收清单

### 16.1 默认关闭回归

- 原loss、advantage、Warmup SFT、更新顺序及actor tensor字段不变。
- 两开关关闭不启动judge、不创建buffer、不扩head；旧配置/旧checkpoint继续可用。
- 原backend和扩展backend开关关闭时，小型确定性fixture输出/梯度一致；真实GPU记录数值容差。
- CLI新增参数不可破坏run alias、stage、skip_completed和resume签名。

### 16.2 在线 rubric

- teacher去重、同题不同回答不共享错误缓存、hash/version变化失效。
- 网络乱序返回与batch行正确匹配；JSON错误/超时/上下文不足不变成零分。
- 实际token与judge文本一致，超过上限/原题被截断有明确状态。
- 原初始化→Warmup扩头时adv值一致；Warmup→GAD四个通道都一致；多头reload不重置；GAD奖励只有adv通道，Warmup actor仍执行teacher SFT。
- joint/correctness_only的mask与loss正确；全缺失或某rank缺失无NaN。
- DDP/FSDP多rank+microbatch1/2梯度对齐全局整批参考，含重复teacher跨rank的情况。
- 实际更新调用顺序测试覆盖Warmup与GAD：保留原compute_values/advantages→critic→按原门槛actor的顺序，失败标注在两次update之前退出。门槛前actor正常跳过不算失败，critic辅助损失仍生效。
- raw padding与remove-padding多头路径测试；最长响应真实反向。

### 16.3 Replay

- 空池、池未满、rho0、不同容量/年龄限制、无放回抽样、专属RNG恢复。
- 本步样本不被当历史抽到；历史pair的prompt/teacher/student永不拆错。
- 总比较对数B与optimizer step数不增加。
- actor_batch逐字段不变、无历史response_id进入policy loss。
- 不存旧advantages/logprob；实际历史token重建一致。
- label版本检查、pending历史标签补齐、瞬时错误允许重试。
- rollout产生后失败不提交新buffer；恢复选择最后完整step。

### 16.4 保存恢复

- 真正新进程恢复actor/critic/optimizer/scheduler/dataloader、buffer、label索引和RNG。
- 缺少任何必要扩展文件时拒绝resume或跳过未完整checkpoint，不静默清空buffer。
- checkpoint轮换保留被引用的cache/object文件；不清理源Warmup。
- 1/4头导出重载与源分数一致；四头配置不被旧fix函数改成1。

### 16.5 真实GPU验收

- 单卡或小模型fixture不能替代正式4卡FSDP验证；Warmup和GAD各自后端都要覆盖。
- 按14.2完成Warmup门槛前/后、完整四头交接及2–5步GAD，含长样本、在线服务和两阶段新进程恢复；确认Warmup actor执行SFT、GAD actor执行原策略更新、critic两阶段都有辅助梯度。
- 两阶段adv/aux变化符合时序；主方案GAD首步允许因Warmup critic不同而奖励不同；judge服务按阶段复用且GPU分配不冲突。
- 不跑全量训练直到工程冒烟和资源预算清楚；一旦通过就进入短程actor对照，避免持续堆砌离线检查。

## 17. 实施里程碑与交接要求

### M1：两阶段 Rubric 配置、数据、服务与模型

交付同时支持Warmup/GAD的独立backend profile、双开关配置（replay明确未支持）、可用judge client/cache、真实token适配、Warmup启动前单头→四头初始化及默认关闭回归。仅GAD扩头是消融路径。

### M2：Warmup 联合训练、交接与 GAD 延续

交付Warmup首步辅助更新、原SFT和actor门槛保留、多卡归一化、Warmup同阶段恢复、完整四头交接、GAD持续联合更新与恢复。按照14.2完成真实GPU闭环，不只跑GAD的2–5步。

**M2完成即可进行A/B/C两阶段actor实验，不等待replay。Warmup支持是M2的必做内容。**

### M3：Actor 主实验与阶段消融

交付初始/Warmup末/GAD末三点评测、准确率/长度/截断/成本比较和限制。按需要增加仅GAD开启的阶段消融。若只有辅助头改善而actor不改善，应明确报告；不把监督不足等同于整个rubric方向无效。

### M4：Replay 独立功能

交付GAD专用bounded历史比较池、固定大小critic batch、状态恢复和replay-only测试。此时才能正式标记四种开关组合均可用；Warmup replay不在本版范围。

### M5：组合实验

进行D/E对照，匹配各自Warmup来源。Warmup rubric已在M2实现，不再作为此阶段可选项；evo-rubric及判别器替换不在本计划内。

每个里程碑实现者必须提供：

- 修改文件、实际支持的阶段/开关组合、未支持且会报错的模式；
- 精确启动、dry-run、两阶段resume、交接和评测命令；
- 代码/配置/模型/数据/judge指纹及初始adv头一致性证据；
- CPU测试、Warmup→GAD多卡GPU冒烟证据，不能把计划中的测试当成已通过；
- 指标分母、Warmup与GAD各自预算、完整结果目录及已知限制。

## 18. 拟定启动示例（实现完成后替换为实测命令）

以下是接口约定，不是当前已经可运行的命令。各主实验使用同一初始模型/数据/初始化seed，执行各自Warmup；示例跳过独立SeqKD基线，不从SeqKD加载权重。

```bash
cd /mnt/jinbo/RLRM/LMOps

# 开启rubric前，按实际资源配置独立verifier并检查模型指纹。
bash scripts/rubric_judge.sh --config train.yaml --action health

# A：两阶段原版基线。
bash scripts/train.sh --stages warmup,gad --name gad_baseline \
  --gad-new-steps 80 \
  --set rubric.enabled=false --set replay.enabled=false --dry-run

# B：从Warmup开始，仅正确性辅助；无需过程judge请求。
bash scripts/train.sh --stages warmup,gad --name gad_correctness \
  --gad-new-steps 80 \
  --set rubric.enabled=true --set 'rubric.stages=[warmup,gad]' \
  --set rubric.mode=correctness_only --set replay.enabled=false --dry-run

# C：主方案，Warmup与GAD持续进行完整rubric辅助训练。
bash scripts/train.sh --stages warmup,gad --name gad_rubric \
  --gad-new-steps 80 \
  --set rubric.enabled=true --set 'rubric.stages=[warmup,gad]' \
  --set rubric.mode=joint --set replay.enabled=false --dry-run

# C-GAD：阶段消融，复用原版Warmup，仅在GAD起点扩头。
bash scripts/train.sh --stages gad --name gad_rubric_gad_only \
  --init-from-warmup-run qwen3_0.6b_17k --init-step 271 --gad-new-steps 80 \
  --set rubric.enabled=true --set 'rubric.stages=[gad]' \
  --set rubric.mode=joint --set replay.enabled=false --dry-run

# E：Replay实现后，rubric覆盖两阶段，replay只覆盖GAD。
bash scripts/train.sh --stages warmup,gad --name gad_rubric_replay \
  --gad-new-steps 80 \
  --set rubric.enabled=true --set 'rubric.stages=[warmup,gad]' \
  --set rubric.mode=joint --set replay.enabled=true \
  --set 'replay.stages=[gad]' --dry-run
```

主实验A/B/C由配置中相同Warmup预算控制，dry-run应明确列出Warmup会重新训练、初始模型来源、critic初始化、Warmup结束后GAD终点的解析方式。不要把以上命令自动转成复用旧Warmup。

消融示例中的旧run只有在初始模型、数据、Warmup预算等与A可比时才能用于正式比较，否则只用于工程检查。D/E应按14.4匹配来源，必要时复用对应已完成Warmup，禁止混淆实验组。

judge配置缺失在dry-run列出，不自动选择模型/端点；baseline和correctness_only无需过程judge。所有命令在真正实现后补充实测日志与结果目录。

最终研究结论围绕actor：**在相同Warmup和GAD预算下，从Warmup起让critic同时适应teacher/student分布与rubric评分，是否带来可重复的数学准确率改善，以及需要多少额外成本。** 离线代码继续用于标签与实现诊断，不再作为上线实验前必须拿到显著增益的关卡。
