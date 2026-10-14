> Qwen3-1.7B 当前实验请从 `LMOps/scripts/train.sh`、`LMOps/scripts/preflight.sh` 和
> `LMOps/train.yaml` 进入；运行产物在 `LMOps/runs/`。
> 下面的 `gad/` 命令仍是通用实现说明。

# GAD 三阶段流水线（SeqKD → Warmup → GAD）实施说明

本文件说明按 `LMOps/GAD修改计划.md` 对 `LMOps/gad` 所做的修改：新增了什么、怎么运行、
已验证到什么程度、以及还需要在 GPU 机器上完成哪些验收。

默认关闭扩展时，算法本身没有被改动：判别器结构、判别器损失、Warmup actor 的 SFT 更新、GAD 的 GRPO
更新顺序与 KL 设置全部保持官方实现；新增的是**外围工程**（独立 backend、数据、模板兼容、
checkpoint 交接、评测与编排）。

显式启用在线 rubric/replay 的算法扩展使用独立后端，配置、启动、恢复及工程证据见
[ONLINE_RUBRIC.md](ONLINE_RUBRIC.md)。原 actor 目标与 adv 奖励接口保留。

---

## 1. 新增文件一览

```text
LMOps/gad/
  PIPELINE.md                      # 本文件
  configs/pipeline.example.yaml    # 三阶段统一配置（必填项 + 注释）
  configs/smoke.example.yaml       # GPU 烟测配置（越过 critic_warmup 门控）
  configs/math_eval.example.yaml   # 数学评测配置
  scripts/pipeline/_common.sh      # 入口脚本公共部分
  scripts/pipeline/run_all.sh      # 一键三阶段
  scripts/pipeline/run_seqkd.sh    # 单阶段
  scripts/pipeline/run_warmup.sh
  scripts/pipeline/run_gad.sh
  scripts/pipeline/run_math_eval.sh
  tools/common.py                  # 日志/哈希/manifest/子进程
  tools/config.py                  # 配置加载与校验
  tools/backend_patches.py         # 五个最小后端补丁（可校验、幂等）
  tools/prepare_backends.py        # 三个 detached worktree + 补丁 + manifest
  tools/prepare_gad_checkpoint.py  # Warmup → GAD 交接 / GAD 最终模型导出（--mode export-final）
  tools/pipeline.py                # 三阶段编排、门控、续训判定、dry-run
  tools/preflight.py               # 训练前预检（含 math scorer 探针）
  tools/math_metrics.py            # CARD 对齐评分 + 统计 + 对照自测
  tools/evaluate_math.py           # 外部/开发集数学评测
  tools/check_ray_guard.py         # 在真实 Ray worker 中验证 backend 导入守卫
  tests/test_pipeline_cpu.py       # 无需 GPU 的流水线测试
  tests/test_resume_patch.py       # 同阶段续训补丁的行为回归
  tests/test_algorithm_parity.py   # 冻结损失/优势函数对照
  tests/test_math_validation.py    # 验证阶段数学指标（补丁插入代码 + 逐行 diff）
  tests/test_handoff_export.py     # 真实 DTensor 分片：交接导出 + GAD 最终模型导出
  tests/run_all.sh                 # 一键跑全部 CPU 套件

LMOps/sample/                      # 离线数据生产（独立依赖，不依赖 CARD / verl）
  README.md  requirements.txt
  task_template.py                 # 统一数学模板、prompt 规范化与哈希、教师指纹
  teacher_index.py                 # 索引 schema / 严格读写 / CARD 索引导入
  math_scorer.py                   # vendored CARD math_dapo 评分（Apache-2.0，含出处）
  prepare_prompts.py               # DAPO 去重、冲突检查、固定 train/dev 划分
  sample_teacher.py                # 教师采样 CLI（断点续采、失败清单、--log-file）
  teacher_engines.py               # 两个 backend：OpenAIEngine / VllmOfflineEngine
  build_gad_data.py                # 生成 content + teacher_response parquet
  validate_data.py                 # 数据契约校验（泄漏、空答案、长度预算）
  prepare_eval_data.py             # AIME/AMC/MATH500 评测集（缺失显式列出）
  make_smoke_data.py               # 合成烟测数据（明确标记 SYNTHETIC）
  configs/…  scripts/…  tests/test_data_contract.py
  tests/test_teacher_sampling.py   # 采样器 × 本地 mock 端点（断点/指纹/失败路径）
  tests/test_eval_data.py          # 评测集构建（缺失基准、重复题、泄漏）
  tests/test_split_no_leak.py      # 按题面分组的划分 + run_sample.sh 阶段选择
```

原有 `scripts/train/*.sh`、`tools/merge_model2hf.py`、`verl/` 源码基线均保留未改；
`tools/merge_model2hf.py` 被交接流程直接调用。

---

## 2. 运行方式

```bash
# 0) 数据：DAPO 去重/划分 → 教师采样 → GAD 数据 → 校验（CPU）
cp LMOps/sample/configs/sample.example.yaml /abs/path/sample.yaml   # 填路径
# 教师有两种 backend（teacher.engine）：
#   vllm_offline：教师在同一进程内加载（一条命令、无端口、无 sudo、日志进文件，推荐）
#   openai      ：教师由 serve_teacher.sh 单独提供（远程/他人服务时用）
bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml --dry-run
setsid nohup bash LMOps/sample/scripts/run_sample.sh \
  --config /abs/path/sample.yaml --stage sample > /dev/null 2>&1 < /dev/null &
tail -f /abs/path/data/logs/sample.log                      # 断开终端也不会中断
bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml --stage convert
bash LMOps/sample/scripts/run_sample.sh --config /abs/path/sample.yaml --stage validate

# 1) 训练：预检 → dry-run → 正式跑
cp LMOps/gad/configs/pipeline.example.yaml /abs/path/run.yaml       # 填路径
python LMOps/gad/tools/preflight.py --config /abs/path/run.yaml
bash LMOps/gad/scripts/pipeline/run_all.sh --config /abs/path/run.yaml --dry-run
bash LMOps/gad/scripts/pipeline/run_all.sh --config /abs/path/run.yaml
```

分段执行与恢复：

```bash
bash LMOps/gad/scripts/pipeline/run_all.sh --config /abs/path/run.yaml --from-stage warmup
bash LMOps/gad/scripts/pipeline/run_gad.sh  --config /abs/path/run.yaml --resume-gad   # 同阶段完整续训
```

最终模型导出（评测前必须做，否则评的是 Warmup 导出）：

```bash
python LMOps/gad/tools/prepare_gad_checkpoint.py --mode export-final \
  --gad-dir /abs/path/runs/<run-id>/gad --manifest /abs/path/runs/<run-id>/final_model_manifest.json
# -> actor_hf = <gad-dir>/global_step_{S+K}/actor/huggingface
```

评测：

```bash
python LMOps/gad/tools/evaluate_math.py --help
bash LMOps/gad/scripts/pipeline/run_math_eval.sh \
  --model /abs/path/merged_hf --parquet /abs/path/math_eval/test.parquet \
  --out-dir /abs/path/eval/gad --label gad --n 16 --math500-n 2 \
  --temperature 1.0 --top-p 0.95 --max-tokens 32768
```

---

## 3. 关键设计

### 3.1 三个独立 backend（P0）

`tools/prepare_backends.py` 从固定的三个 commit 创建 detached worktree：

| 阶段 | 分支 | commit | 目录 |
| --- | --- | --- | --- |
| SeqKD | `seqkd` | `c8b3a62f…` | `gad/.backends/seqkd` |
| Warmup | `warmup` | `b6e88286…` | `gad/.backends/warmup` |
| GAD | `gad` | `707757b9…` | `gad/.backends/gad` |

* 不再 `git checkout` 切换分支，源码基线 `gad/verl` 保持不动；
* 每个阶段用自己的 `PYTHONPATH`，并通过自动生成的 `sitecustomize.py` 守卫：
  driver 与继承该环境的 Ray worker 只要导入了**其它路径**的 `verl`，进程立即报错退出；
  这一点在本机已复现价值——环境里确实存在一个指向别的项目的 `verl`；
  driver 侧已在 CPU 测试中验证（`GuardTests`），worker 侧用
  `tools/check_ray_guard.py` 在训练机上一条命令验证（需要能启动 Ray）；
* 每个 backend 的补丁 diff 会导出为 `.backends/<stage>.patch` 并记录 SHA-256。

### 3.2 五个最小后端补丁（仅工程适配）

`tools/backend_patches.py`，锚点校验 + 幂等，可随时 `verify`。**冻结的三个算法文件
（`core_algos.py` / `dp_actor.py` / `dp_critic.py`）依然逐字节等于源仓库**，下面 4、5 只改
分批与引擎传参，不碰任何损失函数：

1. `gad-chat-template-kwargs`：让 `RLHFDataset` 把 `data.apply_chat_template_kwargs`
   （例如 `enable_thinking: false`）真正传给 4 处 `apply_chat_template` 调用，并在
   `ppo_trainer.yaml` 中新增该配置键。原版没有转发该参数，只加 Hydra 配置会被静默忽略，
   导致教师数据与学生训练模板不一致。
2. `gad-resume-worker-state`：可选开关 `trainer.resume_load_worker_state`，开启后在
   `_load_checkpoint` 中恢复 actor/critic 的模型、optimizer、scheduler 与 RNG。默认
   **关闭**，保持原版"权重来自 `model.path` + 恢复步数与 dataloader"的语义。
3. `gad-math-validation`：在三个分支的 `_validate()` 里，对**已经生成**的 student 文本追加
   CARD 对齐的数学评分（无额外生成、无梯度），ROUGE-L 原样保留：

   * 评分器由环境变量 `GAD_MATH_SCORER`（pipeline 对每个阶段注入，来自
     `validation.scorer_path`，默认 `LMOps/sample/math_scorer.py`）指定，与离线评测**同一份**
     实现；配置了却加载失败时**直接报错**，不会静默退化成"只有 ROUGE-L"；
   * 指标：`val/math/acc`（`acc` 的均值，才是正确率）、`val/math/score_mean`（±1 score 均值，
     明确不叫正确率）、`val/math/answer_parse_failure_rate`、`val/math/response_len_mean`、
     `val/math/truncated_rate`；同时把 `acc` 交给原版 per-data-source 聚合，使其进入
     `val-core/acc/mean@1` 而不是让 ROUGE-L 占据主指标位；
   * 截断定义：`response tokens 到 EOS 为止 >= rollout.response_length`，长度取自 rollout
     自己的 attention mask，与该分支构造 response mask 的方式一致；
   * 每次验证追加一行 `<stage_out>/validation_math_metrics.jsonl`（含 global_step、样本数、
     scorer 路径、`val_kwargs.n/do_sample/temperature`），因此**不依赖 logger 后端**即可核对；
   * pipeline 在没有该产物时会**拒绝把阶段标记为完成**（见 3.7）。

   `tests/test_math_validation.py` 直接 import 打过补丁的 `ray_trainer`，把插入的三段代码
   从源码里抽出来对假 batch 执行，并对 `ray_trainer.py` 与 pinned 版本做逐行 diff：
   **只允许新增、且每个新增块都必须落在 `PATCH(gad:…)` 标记内**，reward/advantage 代码逐字节不变。
4. `gad-teacher-aware-micro-batch`：`rearrange_micro_batches` 原来只用**学生**侧的
   attention mask 估算"这一微批多少 token"，但同一微批里 Warmup actor 跑的是**教师**序列
   （SFT），判别器**同时**前向学生与教师。长数学回答下教师侧更长，于是微批实际前向的 token
   数远超预算。补丁做两件事：

   * **预算覆盖这一微批真正会前向的所有 mask**：`compute_teacher=True` 仍表示只用教师侧
     （SeqKD）；两侧都在时按**学生+教师之和**；
   * **硬上限**：`ceildiv(total, budget)` 只约束*平均*负载，Karmarkar-Karp 均衡器仍可能给出
     超过预算的单批（实测 4×(1024+8192)、预算 12288 时 pinned 把四条样本塞进**一个 36864
     token** 的微批；另一组实测里 pinned 连学生侧都切出了 20480 > 12288 的单批）。现在 split
     数会一直增长到**每一批都 ≤ 预算**为止；`all_reduce(MAX)` 与 `all_reduce(MIN)` 在每一轮里
     都所有 rank 一起参与，所以 DP 各 rank 的微批数始终一致、不会各走各的。
     单条样本自身就超过预算时无法再拆（拆分的最小单位是样本），它保持为 singleton 并打印一条
     明确警告（每条的前向本身仍满足 `max_seq_len <= max_token_len`）。

   **这不是"数学上等价"的改动，补丁也不再这样声明。** `use_dynamic_bsz=True` 下 trainer 按
   **样本数**缩放每个微批的 token 归一化损失（`dp_actor.py`：
   `loss = policy_loss * (len(data) / ppo_mini_batch_size)`），所以累加目标是
   `Σ_g (|g|/B) · tokenmean(g)`，等价于给每条样本权重
   `w_i = (|g(i)|/B) · (n_i / Σ_{j∈g(i)} n_j)`：

   * 同一微批内**长度相同**的样本 ⇒ 完全不变（可逐位复现，见测试）；
   * 长度不同 ⇒ 重新分组会改变每条样本的权重，更新与 pinned **不逐位相同**。

   用 pinned 的 `compute_sft_loss` + pinned 的缩放规则实测（`tests/test_microbatch_equivalence.py`）：
   两条样本、1 token@loss1.0 与 3 token@loss3.0，合批累加目标 **2.5**、拆批 **2.0**，梯度不同；
   权重与闭式解逐值一致，且 `Σ w_i = 1`（所以差异是"重新分配"，不是量级漂移）。
   方向也是确定的：补丁给出的微批更小 ⇒ 每条样本更接近单独成批 ⇒ 权重更接近均匀的 `1/B`
   （即削弱了 pinned 里"长样本在同批内占比更高"的隐性加权）。

   **需要与 pinned 逐位一致时**：`GAD_MICRO_BATCH_BUDGET=upstream` 环境变量会恢复 pinned 的
   预算模型与切分（不覆盖教师侧、不设上限），实测与 pinned 的 partition 完全相同；
   此时若单批超预算，补丁会打印一条提示（不静默）。
   `GAD_MICRO_BATCH_BUDGET=teacher-aware`（默认）即上面的行为。

   实测（8 条样本、学生/教师各 8192 token、预算 12288）：

   | | 微批数 | 单批最大 学生+教师 token |
   |---|---|---|
   | 修前（源仓库/pinned） | 6 | **32768**（预算的 2.7 倍） |
   | 修后（打过补丁） | 8 | 16384（= 单条样本两侧之和，已不可再分） |

   **判别器不受这个补丁影响**（也就没有上面的权重偏差）：R5 让 `critic.use_dynamic_bsz=False`
   + `critic.ppo_micro_batch_size_per_gpu=1`（原版 `_dapo_common.sh` 的默认），静态分批走
   `batch.split(micro)`，**根本不调用** `rearrange_micro_batches`。动态路径里：
   Warmup actor 的 batch 同时带学生与教师张量（`compute_teacher=False`），补丁把预算改成
   两侧之和 ⇒ 与 pinned 的切分不同（这是 OOM 的根因，也是上面权重偏差的来源）；
   GAD actor 与 log-prob 路径的 batch 只有学生张量 ⇒ **选边规则与 pinned 相同**，
   但"每批 ≤ 预算"这条上限仍可能改变切分 —— 实测 pinned 在学生侧也会切出 20480 > 12288 的
   单批（均衡器只保证平均），此时补丁的切分与 pinned 不同，换来的就是不 OOM。
   ```bash
   # 想验证判别器确实没走动态分批（日志里不应出现 rearrange 相关的超预算警告）
   grep -c "teacher-aware-micro-batch:cap" runs/<run-id>/logs/gad.log
   ```

5. `gad-vllm-engine-limits`：把 `rollout.max_num_seqs` 真正传给 `LLM(...)`。该键在 Hydra 配置里
   存在但从未进引擎，调小它对 vLLM 并发（以及 KV 显存压力）没有任何效果。


`tests/test_pipeline_cpu.py::test_frozen_algorithm_files_are_untouched` 断言
`core_algos.py`、`dp_critic.py`、`dp_actor.py` 与 pinned 后端逐字节一致。

### 3.3 阶段预算与门控（P3）

* 预算有**两种口径**，同一阶段同时给了就按 epoch 口径（原版口径）：
  * **epoch 口径** `budget.stage_epochs.{seqkd,warmup,gad}`：原版脚本用
    `trainer.total_epochs` 定预算（`scripts/reference/gpt5-chat-filtered-7b-*.sh`：
    seqkd=4、warmup=2、gad=4；官方 README 的 Warmup 写 1 epoch，本计划 §3.2 取 README 的 1）。
    `steps = epochs × (train_rows // train_batch_size)`，换池子/batch 会自动跟着变。
    SeqKD/Warmup 从 step 0 起，`epochs` 就是各自数据的遍历次数；**GAD 的
    `stage_epochs.gad` 是"全局绝对目标"**（原版把 Warmup 的 `global_step_S` 连
    `latest_checkpointed_iteration.txt` 一起搬进 GAD 目录再 `total_epochs=4`），
    新增步数 = `epochs × steps_per_epoch − S`，必须 > 0；
  * **步数口径** `seqkd_steps / warmup_steps / gad_steps`：各阶段**新增步数**，
    仅当该阶段没有 epoch 预算时生效（冒烟/快速试跑用）。
* SeqKD 是独立基线，不参与 Warmup 初始化；Warmup 从初始学生开始；
* GAD 沿用 Warmup 的 actor 与 critic，并恢复 Warmup 的 global step S，
  `trainer.total_training_steps = S + 新增步数`，因此预算不会被误读成
  "再训练整个 epoch 数"；
* epoch 循环长度由 `ceil(target_steps / steps_per_epoch)`（GAD 额外 +1）自动推导，保证
  epoch 循环足以到达目标步数；`budget.epochs_override.<stage>` 只在该推导之上抬高循环
  次数，不能用来下调预算；
* 原版门控保留：SeqKD/Warmup `critic_warmup=10`，GAD `0`；`smoke.example.yaml` 用
  `warmup_steps: 14` 保证真正越过门控、actor 确实被更新；
* 原版超参保留：SeqKD actor lr 5e-6，Warmup/GAD actor 与 critic lr 1e-6，
  `grad_clip=0.2`，rollout `temperature=0.8`，GAD actor `kl_loss_coef=0.001`。
* **rollout.n 按阶段拆开**（原版 `STAGE_N_SEQKD:=1` / `STAGE_N_WARMUP:=1` / GAD `ROLLOUT_N`）：
  SeqKD/Warmup 的 loss 是教师 SFT，候选不进优势估计，而且 actor 更新前
  `dp_actor.deduplicate_by_uid()` 只保留每个 uid 的**第一条** —— 所以那两个阶段 n=8 会让每步
  512 条序列里的 7/8 在 actor 更新时被丢掉；判别器仍可能看到其余候选。
  因此 n=1 是本实验的吞吐调优，不等同于严格的 n=8 判别器候选分布。
  现在 `rollout.n_seqkd`/`rollout.n_warmup` 默认 1，
  `rollout.n`（GAD 组大小）保持 8，并且校验会像原版 `dapo_check_gad_rollout_n` 一样
  **拒绝 GAD n<2**（GRPO 在 uid 组内归一化优势，n=1 没有优势信号）。
* **R5：判别器分批与 actor 解耦，静态 micro=1**（原版 `_dapo_common.sh`
  `CRITIC_DYNAMIC_BSZ:=False` / `CRITIC_MICRO:=1`，计划 §129 的首版决定）。pipeline 对
  Warmup/GAD 传 `critic.use_dynamic_bsz=False` + `critic.ppo_micro_batch_size_per_gpu=1`
  （配置键 `optim.critic_dynamic_bsz` / `optim.critic_micro_batch_size_per_gpu`）。
  为什么是"口径"而不是"调优"：判别器同时前向学生与教师，静态 micro=1 意味着
  **逐条样本**前向（每条两侧各 ≤ 10240 padded token），既从根上避免了补丁 4 里的超预算微批，
  又因为静态路径走 `batch.split(micro)`、根本不调用 `rearrange_micro_batches`，
  使相同输入下的判别器分批规则与 pinned 一致；n_warmup=1 与 n=8 的输入分布仍不同。
  原版还有 `DYNAMIC_BSZ:=True`（actor/ref/rollout 动态）——本 pipeline 的
  `actor_rollout_ref.actor.use_dynamic_bsz=True` 与之一致。
  `optim.critic_dynamic_bsz: true` 可退回动态分批（更快，但判别器分批规则被补丁 4 改变，
  属于偏离）；配置校验会拦下 `critic_micro_batch_size_per_gpu < 1`。

### 3.4 Warmup → GAD 交接（P2/P3）

`tools/prepare_gad_checkpoint.py`：

1. 按**步号数值**扫描 `global_step_*`，从新到旧寻找第一个**完整** checkpoint
   （actor/critic 的 model+optimizer 分片、`config.json`、tokenizer、`data.pt` 齐全）；
2. 若 `latest_checkpointed_iteration.txt` 与所选步号不一致，**先报错、后操作**（除非显式
   `--allow-marker-mismatch`），避免把"没跑完的 Warmup"当成正常结果；
3. 复制 actor/critic/`data.pt` 到独立的 GAD 目录并写 `latest_checkpointed_iteration.txt`，
   Warmup 目录只读、不被覆盖；
4. 调用原版 `tools/merge_model2hf.py` 分别导出 actor 与 critic；
5. 校验判别器仍是**单标量 value head**：`score.weight` 形状为 `[1, hidden]`、
   基础权重不缺；随后把合并后的 `config.json` 的 `architectures` 修正为
   `…ForTokenClassification` 并写回 `num_labels: 1`。
   （原因：`merge_model2hf.py` 按 `config.architectures` 选类，而 critic checkpoint 的
   `config.json` 仍是学生的 `…ForCausalLM`；value head 权重会保留，但元数据必须显式纠正，
   否则后续任何按 architectures 路由的加载器都会误判。）
6. 输出 `handoff_manifest.json`：来源、步号 S、目标 S+K、文件校验值、判别头校验结果。

`--mode export-final`（第二轮修订新增）用于**导出 GAD 真正训练出来的模型**：按同样规则找最后一个
完整 checkpoint，只合并 actor，并拒绝三类错误产物——缺权重（与来源分片逐键比对）、
含判别器参数（`score.*`）、架构是 `…ForTokenClassification`。输出
`final_model_manifest.json`。**为什么必须这样**：`handoff_manifest.actor_hf` 是 GAD 的
**初始**模型（Warmup 导出），用它做最终评测等于报告"训练前"的数字；pipeline 现在还会把
导出的 step 与 GAD 阶段校验过的 `last_step` 对齐，不一致直接报错
（`refusing to evaluate the wrong checkpoint`）。

**`--resume-gad` 的语义（第二轮修订修正）**：交接工具拒绝写入非空 GAD 目录，因此"中断后用同一
run 恢复"必须区分两种情形，`pipeline.resolve_gad_resume()` 现在显式判定：

| 情形 | 判定 | 行为 |
| --- | --- | --- |
| 无完整 GAD checkpoint | `fresh` | 走正常 Warmup→GAD 交接并从头开始 |
| 有完整 checkpoint 且 `< S+K` | `resume` | **不重复交接**，复用已记录的 handoff（`handoff_manifest.json`，缺失时回退 `stage_gad.json`）作为模型路径，交给 trainer 原地恢复 |
| 已有 `>= S+K` | `resume-complete` | 不再启动 trainer（否则会多跑一步到 S+K+1），只做产物校验并补写完成标记 |
| 有完整 checkpoint 但**没加** `--resume-gad` | `fresh` + `conflict` | 真跑时直接报错并提示"用 `--resume-gad` 原地继续，或换 `--run-id` 重开"，不再让交接工具抛 `refusing to overwrite`；`--dry-run` 只告警并照常打印计划 |

若既无 handoff 记录也无 `stage_gad.json`，直接报错说明缺什么，而不是猜路径。

**语义边界（不夸大）**：正常无中断的跨阶段衔接只承接权重、global step 与 dataloader，
optimizer 重新初始化；完整 optimizer/scheduler/RNG 续训需要显式 `--resume-gad`
（即后端补丁 2），两者在 manifest 与 `run_summary.json` 中分别记录。

**reference 锚点**：GAD 阶段传入的 `actor_rollout_ref.model.path` 是 Warmup 合并后的 actor，
而原版 worker 正是用同一个 `config.model.path` 构建 reference policy（`fsdp_workers.py` 中
`self.config.model.path` → `ref_module_fsdp`）。因此 KL 锚点就是 Warmup actor，
不会被额外固定成"最初的学生模型"——这是原版行为，未做修改。

**vLLM 引擎上限（必须显式传）**：`rollout.max_model_len` 与
`rollout.max_num_batched_tokens` 默认都取 `data.max_prompt_length + data.max_response_length`。
vLLM 自己的默认是 `max_num_batched_tokens=8192`、`max_model_len=prompt+response`，而
`enable_chunked_prefill` 默认 True —— 一旦长度预算超过 8192（本配置 2048+8192=10240），
`vllm_rollout_spmd.py:128` 会在**引擎构造时**直接 raise：

```
ValueError: Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len
```

所以 pipeline 无条件传这两个值（原版 harness 也是传 `TOTAL_LEN`），`validate_config`
也会在预检阶段就报出"显式值 < 长度预算"的配置错误，而不是等 40 秒后在 vLLM 里炸。
`rollout.max_num_seqs` 跟原版一样是 128，并且由补丁 5 真正传给引擎。

**24GB 卡上的显存账**：4×3090 跑 1.7B + 10240 长度预算时，单步峰值由以下几项叠加：

| 项 | 量级 | 说明 |
|---|---|---|
| vLLM 权重 | ~3.4GB（生成期） | 更新期不占：`sleep(level=1)` 把权重卸到 CPU（`fsdp_vllm.py:217`） |
| vLLM profiling/激活开销 | ~2.7GB（生成期） | 实测：`u=0.30` 时 `0.30×23.5 − 3.4 − 0.95(剩余KV) ≈ 2.7GB` |
| vLLM KV cache | init 定死，见下 | 生成期使用；更新期被丢弃 |
| CUDA graph pool | 仅 `enforce_eager=false` 时 | graph pool 不在 CuMemAllocator 的 sleep tag 里 → **常驻训练期**，所以要关 |
| actor 参数 + 梯度 | ~2–3GB/卡 | FSDP 分片后；`models.param_offload` 可再挪到 CPU |
| Adam 状态 | ~3.4GB/卡 | `optimizer_offload` 只在更新之外放 CPU，`load_fsdp_optimizer` 更新时会**搬回 GPU** |
| LM head logits | `tokens × vocab × 2B` | 12288 token × 151936 vocab ≈ **3.7GB**，本配置最大的单项 |
| 激活（checkpointing 后） | ~2–4GB | 与 `ppo_max_token_len_per_gpu` 成正比 |

**vLLM 的 KV cache 在引擎构造时一次性定尺寸，而且必须能装下一条满长序列**，否则引擎直接拒绝
启动（`v1/engine/core.py:_initialize_kv_caches` → `kv_cache_utils.py:check_enough_kv_cache_memory`）。
`wake_up` 只是把 allocator 重新挂上（`v1/worker/gpu_worker.py:100`），**不会**重算池子大小，
所以 init 那一刻的取值决定了整轮 run 的 KV 上限。Qwen3-1.7B 的算式：

```
KV/token = 2(K,V) × 8(kv heads) × 128(head_dim) × 28(layers) × 2B = 112KiB
一条 10240-token 序列 = 1.09GiB
可用 KV ≈ gpu_memory_utilization × 总显存 − 权重 − ~2.7GiB 开销
```

实测：`u=0.30 → 0.95GiB < 1.09GiB`，引擎报 `ValueError: ... KV cache is needed, which is larger
than the available KV cache memory`；`u=0.35 → 2.1GiB`、`u=0.40 → 3.3GiB`。
`tools/preflight.py` 现在用同一套算式（`common.vllm_kv_budget`）在**预检**里算一遍
（实测复现 vLLM 的 0.95GiB，误差 <0.01GiB），所以这类失败不会再烧 GPU 时间。
KV 池在每次 actor 更新前就被丢弃，所以给它留大一点是"显存换生成速度"，不吃训练空间。

实测现象（训练期）：`seqkd` 在第 11 步 OOM —— `critic_warmup=10` 让前 10 步不更新 actor，第一次
反传正好落在第 11 步，峰值是 Adam 状态回迁 + 12288-token 的 LM head logits。
**SeqKD 跑到第 10 步不代表任何阶段的分批或显存是安全的。**

阶梯（全局 `LMOps/train.yaml` 可设置这些字段，依次试，别一次全加）：

1. `rollout.enforce_eager: true`（已默认）—— 去掉常驻训练期的 CUDA graph pool；
   后端/原版 harness 的默认本来就是 `ENFORCE_EAGER:=True`。
2. `rollout.gpu_memory_utilization: 0.40` —— 下限约 0.33，再低 vLLM 起不来（见上）。
3. `optim.*_ppo_max_token_len_per_gpu 12288 → 10240` —— 卡到长度预算后最坏只有单条满长序列
   （logits ~3.1GB 而不是 3.7GB），但**不能低于 10240**（后端 `assert max_token_len >= 序列长度`，
   `validate_config` 已在预检阶段拦下）。
4. `models.param_offload: true` —— actor/critic 参数放 CPU。
5. `data.train_batch_size 64 → 32` —— 缩 rollout 批。

**`rollout.free_cache_engine` 在 pin 的 vLLM 0.8.5 上是空操作**（实测代码核对）：释放逻辑在
`vllm_rollout_spmd.py:269,445`，被 `if vllm_version in ("0.5.4", "0.6.3")` 挡住；它对 vLLM 的
唯一作用是同文件 90 行的断言 `assert not (not enforce_eager and free_cache_engine)`，即强制
`enforce_eager=true`。原版 harness 阶梯里的 "OOM → `FREE_CACHE_ENGINE=True`" 这一档在本项目
pin 的 vLLM 上**不成立**，别指望它省显存 —— KV 的丢弃和权重的 offload 已由 sharding manager
每步自动完成（`fsdp_vllm.py:217`）。

**不要用 `expandable_segments:True`**（实测踩过）：PyTorch 的 OOM 提示会建议它，但本流程的
rollout 引擎是 `enable_sleep_mode=True`，vLLM 因此启用 `CuMemAllocator`，而它在
`vllm/device_allocator/cumem.py:146` 直接断言拒绝：

```
AssertionError: Expandable segments are not compatible with memory pool.
```

`run.pytorch_cuda_alloc_conf` 默认留空；`validate_config` 会拦下含 `expandable_segments`
的取值，想压碎片只能用兼容项（例如 `max_split_size_mb:128`）。

### 3.5 数据与数学适配（P1/P1b）

* DAPO 去重后唯一题数 17,917（实测 1,791,700 行 → 17,917 题，0 个同 id 冲突，
  495 组"同题面不同 id"），固定 `seed=42` 留出 512 题 → **17,405 train / 512 dev**，
  与计划预期一致；`--expected-unique` 会在上游文件变化时报错；
* **划分按题面哈希分组，不按 uid 抽签**（第二轮修订修正）：DAPO 里同一道题会有两个
  `extra_info.index`，按 uid 随机划分会把这些孪生题分到两侧——旧产物实测有 **27 道 dev 题与
  训练集题面相同**（`validate_data.py` 会因此拒绝数据；若绕过校验则验证集被污染）。现在
  以 prompt-hash 组为最小单位整体抽取，`split_manifest.json` / `stats.json` 记录
  `unit=prompt_hash_group`、组数、dev 实际大小与 `dev_size_overshoot`、`leaked_prompt_hashes`
  （内部划分必须为空，否则直接报错）；`--card-full-train` + `--dev-file` 时也会检测外部
  dev 集与训练集的题面重叠，默认报错，需显式 `--allow-dev-leak`（`validate_data.py` 侧对应
  `--allow-val-leak`）才会记录并继续；
* 训练数据默认取教师**第一个有效回答**（确定性、不做正确性筛选），三阶段共用同一份；
  `--select correct` 是显式实验开关（会丢题），默认关闭；
* ground truth 只保存在 `reward_model.ground_truth`，不进入 prompt、不进入 GAD reward；
* 教师文本只经文本接口获得（黑盒），索引按 `(uid, raw_prompt_hash)` 建键，并与采样指纹
  摘要绑定；指纹变化时采样器**拒绝写入**旧索引，也不会删除用户已有文件；
* 数学评分对齐 CARD：末次 `Answer:`、只看末尾 300 字符、`acc` 才是正确率。
  `tools/math_metrics.py --self-test` 用 11 组对照样例固定这些行为，其中明确包含
  "仅 `\boxed{}` 不算对""`142.0` 与 `142` 不等价"等已知限制。

### 3.6 一键脚本的保证

`run_all.sh` 及其分段版本都通过 `tools/pipeline.py`：

* `set -euo pipefail` + 阶段失败立即停止，失败阶段写进 `run_summary.json`；
* `--dry-run` 输出最终配置、backend、完整命令、checkpoint 输入输出与阶段预算，不占 GPU；
* 只有产物校验通过才写 `COMPLETED.json`；跳过已完成阶段需要配置与输入签名一致；
* 每个 run 独立目录，记录 resolved config、源码 commit、补丁 diff、日志与退出码；
* W&B 只是可选 logger，凭据只从环境读取，不写入配置或日志。

### 3.7 阶段完成的判据（第二轮修订加强）

`verify_stage_artifacts()` 除了原来的 checkpoint 完整性、`latest_checkpointed_iteration.txt`、
目标步号之外，现在还会在"该跑验证"时要求数学指标产物存在且可解析：

* `validation_metrics_expected(cfg)` 为真（配置了 scorer、`validation.require_math_metrics`
  未被关闭、且 `val_before_train` 或 `test_freq>0` 至少有一个会触发验证）；
* 此时 `<stage_out>/validation_math_metrics.jsonl` 缺失、为空、无法解析或最后一条没有
  `acc`/`count`，阶段一律**不写完成标记**并报错，错误信息直接指向
  `GAD_MATH_SCORER` 与 `gad-math-validation` 补丁；
* 校验通过后，`stage_<stage>.json`、`COMPLETED.json`、`run_summary.json` 与日志里都会带上
  `math_metrics`（记录每个验证步的 `acc`、截断率、解析失败率、样本数与 scorer 路径），
  因此"开发集正确率"是否真的产出是可核查的，而不是靠"loss 下降"推断。

### 3.8 最终评测用的是哪个模型（第二轮修订修正）

`run.eval_after_train` 现在先调用 `prepare_gad_checkpoint.py --mode export-final` 合并
**GAD 最后一个完整 checkpoint 的 actor**，再把它交给 `evaluate_math.py`；导出 step 必须等于
GAD 阶段校验过的 `last_step`，否则报错。`run_summary.json` 里记录
`final_model.actor_hf` 与 `final_model.manifest`，评测日志首行也打印模型路径与步号。
（修订前这里取的是 `handoff_manifest.actor_hf`，即 Warmup 导出，等于把"训练前"当成了 GAD 结果。）

---

## 4. 已验证 / 待验证

### 已完成（CPU，可复现）

一条命令跑完全部 CPU 验证：`bash LMOps/gad/tests/run_all.sh <python_bin>`
（本次结果：**12 个套件、176 个用例 + 评分自测全部 PASSED**）。

| 项目 | 命令 | 本次结果 |
| --- | --- | --- |
| backend 建立 + 5 个补丁 + 幂等 + 冻结文件未改 | `tests/test_pipeline_cpu.py` | 63 passed |
| **vLLM 引擎 KV 预算可预测**：112KiB/token 的算式与引擎一致、0.30 失败/0.40 通过、缺维度时只告警不崩、随仓库的 run 配置能过自己这道门（`enforce_eager=true`、无 `expandable_segments`） | 同上（`VllmKvBudgetTests`） | 6 passed |
| 守卫拒绝导入外部 `verl`（driver 侧） | 同上（`GuardTests`） | passed |
| Ray 守卫异常摘要：标记在 400 字符之后也能被识别（第二轮修订） | 同上（`RayGuardToolTests`） | passed |
| 三阶段 dry-run 依赖链与步数 | 同上（`DryRunTests`） | passed |
| 微批预算覆盖学生+教师两侧 | 同上（`MicroBatchBudgetTests`） | 2 passed |
| **R5 判别器静态 micro=1**：dry-run 只传 `critic.use_dynamic_bsz=False`+`critic.ppo_micro_batch_size_per_gpu=1`（不传动态预算）、可显式切回动态、`critic_micro<1` 被配置校验拦下、真实 Hydra compose 后 `critic.use_dynamic_bsz=False`；**rollout.n 按阶段拆开**：dry-run 里 seqkd/warmup 传 1、GAD 传组大小，GAD n<2 被拦下（原版 `dapo_check_gad_rollout_n`） | 同上（`DryRunTests`/`ConfigTests`/`HydraComposeTests`） | 7 passed |
| `--from-stage/--to-stage` 与签名跳过 | 同上（`StageSelectionTests`） | passed |
| 交接：最后完整 checkpoint、marker 校验、拒绝覆盖、dataloader 位置读取 | 同上（`HandoffTests`） | passed |
| **同阶段恢复**（fresh / resume / resume-complete / 无 flag 时的冲突提示）与"恢复时不再重跑交接"、缺 handoff 记录报错、忽略半成品新 checkpoint | 同上（`GadResumeTests`） | 9 passed |
| **阶段完成必须有数学指标产物**（缺失/空/坏行/无 acc 全部判失败） | 同上（`StageArtifactTests`） | passed |
| **最终评测模型来源**：dry-run 指向 S+K 而非 S；导出 step 与 GAD 校验步号不一致即报错 | 同上（`FinalModelEvalTests`） | passed |
| **真实分片合并导出**：DTensor 分片 → `merge_model2hf` → actor 权重逐键一致、critic 保持单标量判别头（score 权重/偏置逐值一致）、config 被纠正、Warmup 目录未被写入 | `tests/test_handoff_export.py` | 6 passed |
| **GAD 最终模型导出**：选中最后一个完整 step、导出权重与训练分片一致（含哨兵值证明不是初始导出）、拒绝判别器架构、跳过更新的半成品 step | 同上（`FinalModelExportTests`） | 4 passed |
| **tie_word_embeddings 学生**（Qwen3-1.7B 真实情况）：合并导出保留 `lm_head`、重载后仍与 `embed_tokens` 绑定、绑定的头与训练分片逐值一致；tied 豁免不会掩盖真实丢权重 | 同上（`TiedEmbeddingExportTests`） | 6 passed |
| **三阶段 Hydra 配置解析**：用真实 `ppo_trainer.yaml` compose 每个阶段的全部 override（含 24GB 的 offload/batch/tp 设置） | 同上（`HydraComposeTests`） | 2 passed |
| 同阶段续训补丁行为（默认不改语义 / 开启后加载 actor+critic / 缺 checkpoint 报错） | `tests/test_resume_patch.py` | 4 passed |
| 算法冻结：判别器损失逐元素对齐、mask 梯度为零、GRPO 分组归一化、SFT 损失无 KL | `tests/test_algorithm_parity.py` | 5 passed |
| **微批上限与梯度一致性**（本轮修订）：pinned 切分确实超预算（36864 vs 12288）、补丁后每批 ≤ 预算、单条超预算=singleton、`GAD_MICRO_BATCH_BUDGET=upstream` 与 pinned 逐位相同、教师侧路径在装得下时不变；等长样本合批/拆批**逐位相同**、不等长样本复现 2.5 vs 2.0 且梯度不同、权重符合闭式 `(\|g\|/B)·(n_i/Σn_j)` 且 `Σw=1`；补丁源码里不再出现"等价"字样 | `tests/test_microbatch_equivalence.py` | 12 passed |
| **验证阶段数学指标**（第二轮修订）：acc ≠ score 均值、截断判定、解析失败率、JSONL 产物追加、未配置 scorer 时保持 ROUGE-only、配置错误即报错；并对 `ray_trainer.py` 与 pinned 版本逐行 diff（只允许标记内新增） | `tests/test_math_validation.py` | 10 passed |
| 数据契约（空答案/重复 uid/prompt 不匹配/指纹/泄漏/`--allow-val-leak`） | `sample/tests/test_data_contract.py` | 16 passed |
| **划分不泄漏**（第二轮修订）：同题面双 id 必须同侧（多 seed）、dev 超抽如实上报、确定性、外部 dev 泄漏拦截；端到端跑通 `prepare_prompts → build_gad_data → validate_data` 且 `shared_prompt_hashes=0` | `sample/tests/test_split_no_leak.py` | 7 passed |
| **`run_sample.sh` 阶段名**（第二轮修订）：`--stage convert` = `convert_train`+`convert_dev`、未知名退出码 2、单步可选 | 同上（`RunSampleCliTests`） | 3 passed |
| 教师采样：断点续采不重写、指纹变化拒绝、服务失败非零退出、泄漏拒发、截断记录 | `sample/tests/test_teacher_sampling.py` | 7 passed |
| 评测集构建：缺失基准显式列出、`--strict` 失败、重复题与训练集泄漏上报 | `sample/tests/test_eval_data.py` | 5 passed |
| 真实 DAPO 去重与划分（重新生成，泄漏为 0） | `prepare_prompts.py --expected-unique 17917` | 17,405 / 512，`leaked_prompt_hashes=[]`，dev 496 组（含 15 个多 id 组） |
| CARD 评分对照（11 组样例 + acc/score 区分） | `tools/math_metrics.py --self-test` | passed |
| 预检（含"发现外部 verl"告警 + scorer 真实探针 + vLLM KV 预算 + 教师池截断口径） | `tools/preflight.py --config …` | passed（本沙箱无 CUDA，KV 预算一项如实降级为 warning；另报 train 12.6% / val 10.7% 截断） |
| 评测链路（CPU/HF 后端 2 题烟测） | `evaluate_math.py --backend hf --limit 2` | predictions/report 正常输出 |

> 合并导出测试顺带纠正了一处校验逻辑错误：值头模型（`…ForTokenClassification`）本身就
> **没有** `lm_head`，早期"critic 缺少基座权重"的判定会误报。现在改为把合并结果与**各自来源分片**
> 逐键比对，并把 `lm_head` 缺失作为预期差异记录在 `handoff_manifest.json` 的
> `critic_vs_actor.absent_head_keys` 中。

**未能在此环境验证的一项**：Ray worker 侧的导入守卫。Ray 在本沙箱无法启动
（`The current node timed out during startup`），因此 `tools/check_ray_guard.py` 如实返回
"INCONCLUSIVE"（退出码 2）而不是谎报通过。请在 GPU 机器上执行一次：

```bash
python LMOps/gad/tools/check_ray_guard.py --backend LMOps/gad/.backends/gad
# 期望：positive 用例中 worker 导入 .backends/gad，negative 用例被 gad-guard 拒绝
```

### 待 GPU 机器上完成（本次环境 `torch.cuda.is_available() == False`）

1. **worker 守卫**：`python LMOps/gad/tools/check_ray_guard.py --backend .backends/gad`（本沙箱 Ray 无法启动）；
2. **烟测**：用 `configs/smoke.example.yaml` 跑完三阶段，确认 SeqKD/Warmup 越过第 10 步门控后
   actor 确实更新、Warmup 判别器也有更新、GAD 执行真实 GRPO 更新且 loss/grad 有限；
   同时确认每个阶段的 `validation_math_metrics.jsonl` 里出现真实 `acc`（CPU 侧只验证了插入代码与
   产物判据，没有真实 GPU 验证过一遍）；
3. **长序列**：把 `max_prompt_length/max_response_length` 提到 2048/8192，用真实长样本重跑，
   并记录教师与学生长度分布、截断率（`validate_data.py --student-tokenizer`），
   同时观察 `val/math/truncated_rate`；
4. **交接一致性**：比较 GAD 首步前的 actor/critic 权重与 Warmup 导出产物，确认判别头未被重置；
5. **数据顺序**：确认 `data.pt` 恢复后 dataloader 位置与 S→S+K 目标正确
   （handoff manifest 已记录 `dataloader_state._num_yielded`，与 `steps_per_epoch` 对照即可）；
6. **中断恢复**：`--resume-gad` 下核对 optimizer/scheduler/RNG/global step（判定逻辑与"不重跑交接"
   已在 CPU 侧回归，GPU 侧需确认真实分片能被加载）；另外可故意在训练中途 kill，验证
   `resume` 分支与 `resume-complete` 分支各自的行为；
7. **外部评测**：准备好 AIME24/25、AMC23/24、MATH500 本地路径后跑
   `run_math_eval.sh`，统一比较初始学生、SeqKD、Warmup、GAD 四个模型；
   `run.eval_after_train` 串联时确认 `run_summary.json` 的 `final_model.actor_hf` 指向
   `global_step_{S+K}` 而不是 `global_step_S`。

在这些完成之前，不应把本目录的产物描述为"已完成算法验证"。

---

## 5. 已知限制（如实记录）

* **环境**：本机 `nvidia-smi`/NVML 不可用（`CUDA initialization ... Error 304`），因此没有执行
  任何 GPU 训练；上面第 4 节列出的项目全部仍是待办。Ray 也无法在本沙箱启动，
  故 worker 侧守卫检查记为"未定论"而非"通过"。
* **代理环境变量**：若宿主机的 `NO_PROXY` 写成 `...::1,[::1]` 这种形式，部分 httpx 版本会在构造
  OpenAI 客户端时抛 `InvalidURL: Invalid port`。这是环境问题，不是采样器缺陷；
  在导师服务机上把 `NO_PROXY` 设为 `localhost,127.0.0.1` 即可（`sample_teacher.py` 的测试就是这么做的）。
* **epoch 语义**：GAD 恢复 Warmup 的 global step，`total_epochs=4` 不再等价于"再训 4 个 epoch"，
  所以配置默认按步数推导 epoch；想复现原脚本 epoch 数请用 `budget.epochs_override`。
* **teacher 长度**：DAPO-17k 只有题面与标准答案，没有教师轨迹；真实教师解答必须由
  `sample_teacher.py` 采样得到，`make_smoke_data.py` 的合成教师文本**不能**用于正式实验。
* **CARD 评分**：只做字符串规范化，不做数值/符号等价；小数与分数形式可能被判错，
  报告里同时给出 `answer_parse_failure_rate` 供人工判断。
* **多机**：`nnodes` 参数存在但未验证；第一版验收范围为单机多卡。
* **判别器可作弊点**：教师与学生长度差异可能被判别器利用（截断差异），  计划要求同时记录长度分布与判别分数，这一步在 GPU 烟测中完成。
* **Warmup actor 的分批不是逐位等价**（如实记录）：补丁 4 让 Warmup actor 的 token 预算覆盖
  教师侧并强制上限，而 `use_dynamic_bsz` 的累加按微批样本数缩放，**重新分组会改变每条样本的
  权重**（`w_i = (|g|/B)·(n_i/Σ_{j∈g}n_j)`，实测 2.5 vs 2.0）。这是 24GB 上"不 OOM"与
  "逐位复现 pinned"之间无法同时满足的地方：pinned 的切分本身就会在此配置下超预算
  （实测 36864/12288）。想逐位一致就设 `GAD_MICRO_BATCH_BUDGET=upstream` 并接受显存风险；
  判别器与 GAD actor 不受影响（R5 静态 micro=1 / GAD batch 无 teacher 张量）。
* **数学验证指标**：训练中的 `val/math/acc` 与外部评测共用同一份 CARD 对齐 scorer，
  但开发集（DAPO 留出 512 题）与外部基准（AIME/AMC/MATH500）不是同一批题；
  两者不可互相替代。CPU 侧只验证了插入代码、产物判据与 scorer 对齐样例，
  **没有**在 GPU 上真实跑过一次验证。
* **`--resume-gad` 的边界**：`resume` 分支复用已记录的 handoff 作为模型路径，
  依赖 `handoff_manifest.json`（或 `stage_gad.json`）；两者都不存在时会直接报错，
  此时需要人工指定初始模型路径，pipeline 不会猜。
* **`validation.scorer_path` 默认值**：默认指向本目录的 `LMOps/sample/math_scorer.py`；
  若把 `LMOps/gad` 单独拷走，需要显式配置该路径（预检会报错并指出）。
