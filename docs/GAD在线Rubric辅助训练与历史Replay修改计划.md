# GAD 在线 Rubric 辅助训练与历史 Replay 修改计划

日期：2026-10-12  
状态：实现交接文档；本次只新增计划，不修改训练代码、配置、数据或运行状态。

本计划依据用户最新决定，取代上一版“暂缓正式 GAD 接入”的范围约定。[离线第一版计划](Rubric辅助Critic第一版修改计划.md)保留为历史设计与已实现模块的参考，不再要求其全部研究门槛通过后才能开展在线实验。

## 0. 给实现者的执行摘要

用户决定将研究重心转向正式 GAD：最终通过 actor 数学能力提升验证 rubric 的价值，而不是把离线 critic 标签拟合当作最终目标。

按以下顺序交付，每一阶段都必须可单独验证：

1. **首先实现在线 rubric 辅助 critic 训练。** 从同一个已有 Warmup checkpoint 开始 GAD，对当前生成的 teacher/student 配对数据进行标注，保留原判别损失，增加辅助损失，critic 和 actor 均继续训练。
2. **其次实现独立可选的历史轨迹 replay buffer。** 池子在正式训练中逐步建立，保存完整、题目匹配的 teacher/student 历史比较，供后续 critic 更新；不是预先构建后长期不变的离线标注集。
3. 提供两个相互独立、默认关闭的功能开关，形成原版、rubric-only、replay-only、rubric+replay 四种组合。
4. 先验证 rubric-only 的短程 actor 训练，不以完成 replay 或获得离线显著增益为前提。
5. 原版开关全关时保持原行为；所有算法扩展走显式分支并记入配置与代码指纹。

**首轮研究假设：固定 rubric 辅助监督通过共享 critic 骨干改善原判别通道，进而改善 actor 学习。** 本版不将辅助评分直接混入 actor reward，不更换原判别标签，不同时引入 evo-rubric。若辅助头学好而 actor 不提升，应如实报告该假设未获支持。

## 1. 已确认的需求与非目标

### 1.1 必须满足

- 保留原版 teacher > student 的成对判别损失。
- actor 的 Warmup SFT 与 GAD GRPO/PPO 更新公式不变。
- 正式训练支持在线标注；judge 评价当前回答，不按题目 ID 套用离线旧回答标签。
- critic 继续更新，不因 rubric 引入而永久冻结。
- replay 仅改变 critic 更新用的数据，不把历史轨迹带入 actor 的 policy loss。
- actor 最终数学评测是主结果；critic 标签预测/排序只是辅助诊断。
- 源 Warmup、SeqKD 结果和已有训练目录不覆盖；所有实验采用新 run。
- 复用 `LMOps/rubric` 已验证的 schema、评分器、损失和评分头迁移能力，不推倒重写。

### 1.2 第一轮范围

第一轮只在 **GAD 阶段** 开启 rubric，从同一个原 Warmup checkpoint 初始化各组。这样避免把 Warmup 差异混入 GAD 效果。

Warmup 接入设计预留在第 13 节，待 GAD 功能通过后再做；不作为第一轮验收前提。SeqKD 永远不启用这两项 critic 功能。

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
| `gad/tools/prepare_backends.py` | SeqKD/Warmup/GAD 来自不同固定 worktree | 不只修改 `gad/verl` 而忽略实际 `.backends/gad` |
| `gad/tools/backend_patches.py` | 当前工程补丁约定不改算法 | 新建显式算法扩展补丁组，不偷偷塞进原工程补丁 |
| `.backends/gad/verl/trainer/ppo/ray_trainer.py::fit` | 新采样→原 critic 评分→优势→critic 更新→actor 更新 | 保持该顺序，新辅助梯度对下一批奖励起作用 |
| 同上 | `uid` 是运行时 UUID，按 n 重复 | 区分 GRPO 组 ID 与持久化题目/回答 ID |
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

这些事实提示需要改进格式约束与监督覆盖诊断，但**不要求继续完成大规模离线训练才允许进入正式 GAD 小步测试**。首轮仅做必要的在线 judge 校准和工程冒烟。

## 3. 功能矩阵与关闭语义

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
  stages: [gad]
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

### 4.3 新实验与原 Warmup 交接

复用旧 Warmup 必须新建 run，不能对已完成 run 使用 `--set` 强行改变算法再 resume。

建议新增两个明确参数：

```text
--init-from-warmup-run <run path or alias>
--init-step <explicit warmup step>
--gad-new-steps <K>
```

这些是待实现接口。语义：读取源 run 配置/模型/tokenizer 和完整 Warmup checkpoint，生成新目标 run 的派生 handoff，保留原 actor/critic/reference 初始化约定；不复制为同阶段完整恢复。原 Warmup 文件只读。

如果源 Warmup 步数 S=271，`--gad-new-steps 80` 的 GAD 绝对终点为 351。该参数必须处理 `stage_epochs` 对 steps 的优先级，dry-run 展示 S、额外 K、绝对终点及循环上限；禁止设置 `gad_steps=80` 后实际仍跑完原 epoch。

若沿用已有等价机制，也必须实现上述明确语义，并在最终 README 给出确切可运行命令。不能让用户自行猜测配置组合。

## 5. 正式训练数据流与更新顺序

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

本地代码是“评分→更新 critic→更新 actor”；参考论文采用其他次序的部分不照搬。rubric-only 第一步 actor 使用的分数与 baseline 应相同或在迁移数值误差内，差异从后续 critic 更新后逐步体现。

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

优先在新 run 的 handoff/export 阶段生成四通道 HF 初始化产物：

1. 严格加载源单头 critic，验证其 score.weight 与可选 bias。
2. 扩头时第0行精确复制，其他行固定 seed 初始化，隔离 RNG 以免改变 actor 抽样。
3. 保存新配置/权重及迁移 manifest，源 checkpoint 不变。
4. FSDP worker 根据 num_labels/layout 严格加载四通道，移除无条件强制1的逻辑。
5. 在 FSDP/optimizer 建立前完成结构确定。meta tensor/rank0 广播路径必须通过真实多卡测试，不能只单卡检查。

也可实现 worker 内扩头，但必须证明每rank一致且未对 meta tensor 做无效复制；两种方法择一，禁止重复扩头。

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
runs/<run>/gad/global_step_<S>/
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
- 派生Warmup初始化是新阶段：允许单头迁移、fresh optimizer和空buffer，并明确记录，不能当成“缺扩展状态的resume”。
- label cache若外置，checkpoint必须固定依赖条目的内容hash/快照；不能只存易被覆盖的SQLite路径。resume后不要求对所有缓存逐一重hash，但被引用条目须可验证。
- 缓存可保留超前完成的judge结果，因为纯标签缓存不等于已提交训练步；buffer和训练进度只恢复到最后完整checkpoint。
- 步中途失败不允许仅从失败位置继续：内存可能已更新critic但未更新actor，应丢弃未提交进度，从最后完整checkpoint恢复。

### 10.2 模型与优化器约束

- 原单头→四头是新run初始化，不恢复旧一行head的optimizer状态到四行参数。
- 同阶段四头resume严格恢复四头权重、optimizer、scheduler与RNG；不得每次重新随机初始化辅助头。
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
- preflight确认目标GAD worktree存在所需patch marker和具体版本；未完成replay实现时拒绝开启。

算法模块建议部署在目标backend的 `verl/experimental/gad_extensions/`，源码由本仓库可追踪模块生成/复制，并校验hash；不能只改生成副本。

现有 `rubric` 文件存在裸导入及 `common import *` 等离线假设。提取可复用核心为明确的相对导入模块（例如 `rubric/core/`），在线worker只import loss/model/schema等轻量部分，不连带导入vLLM和离线pipeline。

无论选择显式安装包还是把模块同步进backend，都必须用Ray worker真实进程验证导入来源和hash；不能只验证driver里能import。沿用原backend guard，防止导入CARD或环境中另一份verl。

## 12. 逐文件实现清单

以下文件名是建议职责划分；实现者可以合并轻量模块，但不得省略职责和测试。

| 文件 | 修改/新增内容 |
| --- | --- |
| `LMOps/train.yaml` | 两开关默认false、配置注释；不改原训练默认值 |
| `gad/tools/config.py` | defaults、路径解析、阶段支持、数值范围、judge必填项与失配检查 |
| `gad/tools/pipeline.py` | Hydra映射、新run Warmup初始化、额外步数语义、run签名与dry-run |
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
| backend `ray_trainer.py` | 保持原顺序，构建critic专用批、标注屏障、pool提交、扩展checkpoint |
| backend `dp_critic.py` | 多通道forward、第0通道reward、联合loss、新字段保留和正确缩放 |
| backend `fsdp_workers.py` | 根据schema加载1/4头、传入rubric配置、scheduler更新次数保持原样 |
| backend `ppo_trainer.yaml` | 显式扩展字段及默认关闭值，不滥用任意`+`绕过配置验证 |
| backend dataset/rollout | 保留稳定prompt ID和准确的teacher/student截断/结束元数据 |
| `gad/tools/prepare_gad_checkpoint.py` | 1/4通道兼容或派生rubric初始化专用出口 |
| `gad/tools/merge_model2hf.py` / `export_model.py` | 检查多头config/权重一致，不能导出丢辅助头的模型 |
| `gad/tools/prune_checkpoints.py` / 原完整性判断 | extension完整性、缓存引用和受保护checkpoint |
| `scripts/rubric_judge.sh` | 独立服务启动/检查/停止，不杀其他用户/实验进程 |
| `gad/tests/test_online_rubric*.py` | 数据、标签、数值、恢复与调用顺序测试 |
| `gad/tests/test_replay*.py` | 历史池、计算批预算、actor隔离与恢复测试 |
| `docs/`、`README.md`、`gad/PIPELINE.md` | 可执行命令、功能状态、结果边界及恢复说明 |

文档和日志统一术语：固定离线数据集用于已完成预实验；历史replay池用于正式训练过去步骤的轨迹；标签缓存只是避免重复标注的存储，不等于replay策略。

## 13. Warmup 支持策略

### 13.1 首轮不改 Warmup

A/B/C组共用同一个现有Warmup actor/critic初始化，在GAD起点扩头。这样结果可以归因于GAD阶段辅助训练。

单独开启GAD rubric无需重跑SeqKD和Warmup。judge也不应在SeqKD阶段占GPU。

### 13.2 后续可选支持

GAD功能通过后，再允许 `rubric.stages: [warmup, gad]`：

- Warmup actor仍按原teacher SFT更新，原critic_warmup门槛不变。
- Warmup critic使用同样原判别+辅助损失，非冻结。
- 四头Warmup向四头GAD交接保留全部头；不得重新初始化辅助头。
- GAD开关为false但输入四头Warmup时，要求显式选择兼容导出，不能悄悄删头。
- replay默认只作用GAD；Warmup replay另行验证，不能因为配置列表可写就宣称支持。

先实现GAD时如收到warmup stage配置应在preflight直接说明未支持。不要先引入不工作的参数占位。

## 14. 以 Actor 为中心的实验设计

### 14.1 第一轮只做 rubric

| 组 | 设置 | 要回答的问题 |
| --- | --- | --- |
| A | 两开关关闭，原GAD | 正常基线 |
| B | rubric开启、correctness_only、replay关闭 | 仅答案监督的作用 |
| C | rubric开启、joint、replay关闭 | 过程rubric是否有额外actor收益 |

A/B/C均从同一个原Warmup S开始；B/C使用相同扩头初始化seed。A走原单头路径。开发时额外做扩头lambda0的数值/调用检查，但不把这个额外组变成大规模研究前置条件。

共同固定：actor/critic/reference初始权重、题目顺序seed、n、长度、温度、KL、优化器、actor/critic更新次数、训练题数和评测设置。

策略更新后输出自然会分化，不能强求各组后续trajectory完全相同；可固定初始seed和调度口径，不宣称整个RL运行严格逐样本配对。

### 14.2 分三级预算

1. **工程冒烟：2–5个GAD新步。** 验证真实在线标签、FSDP反向、actor更新、完整保存和新进程恢复，不看效果结论。
2. **短程试验：建议80个新步作为起点。** 起点和终点统一评测，中途仅少量固定检查点诊断；实际预算依测得耗时确定。20步只够调试，不作为无效结论。
3. **效果确认：若有信号或仍有合理疑问，固定训练预算、至少3个seed并扩展评测。** 短程无显著差异不自动判方法无效。

不要求先把离线指标调到显著改善。judge格式/监督覆盖工程检查要通过，但不能用大规模离线优化替代在线主实验。

### 14.3 主指标与统计

主指标：actor在独立数学题上的固定解码准确率，比较相同新增训练步后的结果及相对共同起点的变化。

- 训练中quick-dev仅用于进度和预先声明的模型选择；最终保留独立完整dev/外部数学测试评价。
- 固定greedy或相同采样协议，不能将rubric组更大评测n带来的收益算作训练收益。
- 同时报告输出长度、截断率、有效最终答案率、KL/熵、训练耗时、judge GPU-hours/请求token和总GPU-hours。
- 至少报告“相同actor更新步数”的结果；当judge开销明显时，再补充“相同总计算/时间预算”的效率比较。
- 配对题目bootstrap与跨seed波动分开报告；不对多seed和多回答当作独立题目膨胀显著性。
- 明确主benchmark和训练预算，不在test上反复选择lambda或挑有利checkpoint。

辅助诊断：原adv对当前学生正确/错误回答排序、GRPO组内有效差异比例、rubric逐维覆盖与预测、教师错学生对冲突、梯度比例。

特别是当前n=2且按组标准差归一化，actor信号主要依赖两条新回答的相对排序；仅改善teacher-vs-student判别准确率或rubric MAE不必然改变actor更新。

### 14.4 之后再加入 replay

replay实现完成后追加：D=replay-only、E=rubric+replay，并与相同步数的A/C对比。首轮不要同时改变rubric与replay，避免无法归因。

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
- 原head迁移前后值一致；多头reload不重置；score通道只有adv进入actor。
- joint/correctness_only的mask与loss正确；全缺失或某rank缺失无NaN。
- DDP/FSDP多rank+microbatch1/2梯度对齐全局整批参考，含重复teacher跨rank的情况。
- 实际更新调用顺序测试：compute_values→advantages→critic→actor，失败标注在两次update之前退出。
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

- 单卡或小模型fixture不能替代正式4卡FSDP验证。
- 运行2–5步含长样本、在线服务和恢复；确认actor与critic参数都有实际更新。
- 首步后续步的adv/aux变化符合时序，judge进程只存在一份且GPU分配不冲突。
- 不跑全量训练直到工程冒烟和资源预算清楚；一旦通过就进入短程actor对照，避免持续堆砌离线检查。

## 17. 实施里程碑与交接要求

### M1：Rubric 配置、数据、服务与模型

交付独立backend profile、双开关配置（replay明确未支持）、可用judge client/cache、真实token适配、单头→四头初始化、默认关闭回归。

### M2：Rubric 在线联合更新与恢复

交付正式GAD数据流、FSDP正确归一化、adv-only reward接口、多头保存导出、同阶段完整恢复；运行2–5步工程冒烟。

**M2完成即可进行A/B/C短程actor实验，不等待replay。**

### M3：Actor 短程对照

交付起点/终点评测、准确率/长度/截断/成本比较和局限性。若只有辅助头改善，则明确指出尚未改善actor；若未知/错误标签太多，则定位监督问题，不直接否定rubric方法。

### M4：Replay 独立功能

交付bounded历史比较池、固定大小critic batch、状态恢复和replay-only测试。此时才能正式标记四种开关组合均可用。

### M5：组合实验与可选Warmup支持

进行D/E对照；按需要实现Warmup范围扩展，单独报告初始化差异。evo-rubric及判别器替换不在本计划内。

每个里程碑实现者必须提供：

- 修改文件、实际支持的开关组合、未支持且会报错的模式；
- 精确启动、dry-run、resume和评测命令；
- 代码/配置/模型/数据/judge指纹；
- CPU测试与GPU冒烟证据，不能把计划中的测试当成已通过；
- 指标分母、训练新步数、完整结果目录及已知限制。

## 18. 拟定启动示例（实现完成后替换为实测命令）

以下是接口约定，不是当前已经可运行的命令：

```bash
cd /mnt/jinbo/RLRM/LMOps

# 启动前按实际空闲资源填写judge配置；服务单独启动并检查模型指纹。
bash scripts/rubric_judge.sh --config train.yaml --action health

# 原版对照：同一个Warmup来源，新run，不改原run。
bash scripts/train.sh --stages gad --name gad_baseline \
  --init-from-warmup-run qwen3_0.6b_17k --init-step 271 --gad-new-steps 80 \
  --set rubric.enabled=false --set replay.enabled=false --dry-run

# Rubric主实验：先dry-run确认judge地址、资源、head和终点。
bash scripts/train.sh --stages gad --name gad_rubric \
  --init-from-warmup-run qwen3_0.6b_17k --init-step 271 --gad-new-steps 80 \
  --set rubric.enabled=true --set rubric.mode=joint \
  --set replay.enabled=false --dry-run

# Replay完成后才支持：
bash scripts/train.sh --stages gad --name gad_rubric_replay \
  --init-from-warmup-run qwen3_0.6b_17k --init-step 271 --gad-new-steps 80 \
  --set rubric.enabled=true --set replay.enabled=true --dry-run
```

judge配置缺失应在dry-run中列出缺项，不自动选择模型/端点。baseline若不需要judge，应允许独立启动而不检查judge配置。

最终研究结论围绕actor：**相同训练设置下，rubric辅助训练是否带来可重复的数学准确率改善，以及改善需要多少额外成本。** 离线代码继续用于标签与实现诊断，不再作为上线实验前必须拿到显著增益的关卡。
