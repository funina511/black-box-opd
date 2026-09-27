# sample 采样设计修改计划：单进程离线 vLLM 引擎（vllm-offline backend）

> 目标读者：本仓库维护者。状态：**已实施**（2026-09-21）。
> 实施结果：`teacher_engines.py`（`OpenAIEngine`/`VllmOfflineEngine`）、`sample_teacher.py` 拆出
> `run_engine()` 数据面、CLI 新增 `--backend/--model-path/--log-file/--progress/--engine-factory`，
> 指纹新增 `engine` 字段（HTTP 路径 digest 保持不变）；CPU 侧 21 个新用例（含 fake vLLM 的引擎内部路径）
> + 现有 7 个 HTTP 用例全绿；旧 HTTP 部分索引已重命名为 `teacher_index.http_partial.jsonl`。
> 关联问题：① 采样需要「教师服务 + 采样器」两个进程；② tmux/终端断开后作业暂停（pty 输出背压）；
> ③ 目标机器没有 sudo，无法用 `loginctl enable-linger` / 系统级 systemd 单元。

---

## 1. 目标与非目标

**目标**

1. 新增 `vllm_offline` 采样后端：**一个进程**内加载教师权重（`vllm.LLM`）并直接采样，不需要
   `serve_teacher.sh`、不需要端口、不需要 HTTP。
2. 数据面**完全不变**：索引 schema、header/指纹机制、按 `(uid, raw_prompt_hash, digest)` 断点续采、
   逐条落盘、失败清单、`finish_reason`/截断记录、学生 tokenizer 长度统计、`--limit/--dry-run/--summary-out`。
3. 让「没有 sudo」也能长期稳定跑：作业不必依赖登录会话，也不依赖任何终端（日志写文件），
   即使用户 manager 随会话结束也不影响已落盘的数据与续采。
4. 保留现有 `openai`（HTTP）后端：教师由别人/别的机器提供、或继续用 CARD 现有索引时仍然可用。

**非目标**

* 不改 `build_gad_data.py` / `validate_data.py` / 训练侧任何契约。
* 不实现「多机教师」「教师 logits 蒸馏」（GAD 是黑盒文本蒸馏，取文本接口即可）。
* 不引入新的索引 schema 版本（仍 `gad.teacher_index.v1`）。

---

## 2. 现状与问题定位

| 现状 | 位置 |
| --- | --- |
| HTTP 采样器（`openai` SDK + 线程池） | `sample_teacher.py`：`main()` 中 `OpenAI(...)`、`sample_one()`、`ThreadPoolExecutor` 循环（L318-400） |
| 教师服务 | `scripts/serve_teacher.sh` → `vllm.entrypoints.openai.api_server` |
| 指纹构造（硬编码 `provider="openai_compatible"`） | `sample_teacher.py:129-141` |
| 索引读写/续采/指纹一致性 | `teacher_index.py`（`prepare_index`、`check_resume_consistency`、`read_done_uids`、`append_record`） |

**已确认的故障链（本机实测）**：教师进程 stdout 是 tmux pane 的 pty → 断开 VS Code SSH 后
`session-1.scope` 被 teardown、tmux 服务器不再读 pane → pty 64KB 缓冲写满 → vLLM **阻塞在 `write(2)`** →
生成停止 → 采样器只是「等不到响应」。断开期间进程仍存活（所以重连后继续），不是 SIGHUP 被杀。
`Linger=no` 使这个问题必然发生；`sudo` 不可用使 `enable-linger` 方案不可行。

**结论**：把「教师」放进采样器进程内，并且让采样器**只往文件写日志**，两条同时做，问题从设计上消失。

---

## 3. 设计

### 3.1 引擎抽象（唯一的重构接缝）

现有 `sample_one(row) -> {"answers","finish_reasons","truncated","attempts"}` 已经是天然的引擎接口。
抽成显式接口，写盘/续采逻辑与引擎无关：

```python
# LMOps/sample/teacher_engines.py（新增）
class TeacherEngine:
    name: str                                  # "openai" | "vllm_offline"
    def describe(self) -> dict: ...            # 记录进 header notes 与 summary 的引擎事实
    def sample(self, rows, on_result) -> None: # on_result(row, payload|None, error|None)
        ...

class OpenAIEngine(TeacherEngine):     # 现 sample_teacher.py 的 HTTP+线程池逻辑原样搬入
class VllmOfflineEngine(TeacherEngine):# 新增
```

`VllmOfflineEngine` 关键实现点：

1. **懒加载**：`import vllm` 只在该后端构造时执行（CPU 测试环境、数据契约套件不需要 vllm）。
2. 构造：`LLM(model=model_path, tokenizer=model_path, tensor_parallel_size=TP,
   max_model_len=…, gpu_memory_utilization=…, dtype=…, enforce_eager=…, trust_remote_code=True,
   disable_log_stats=…, seed=…)`。
3. **模板必须与 HTTP 路径一致**：用 `llm.get_tokenizer()` 对 `row["prompt"]`（DAPO 原始消息列表，
   不重新套数学指令）调用
   `apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **chat_template_kwargs)`，
   其中 `chat_template_kwargs = {"enable_thinking": False}` 当 `--enable-thinking false`
   （与 HTTP 的 `extra_body.chat_template_kwargs` 完全对应）。
4. 生成：按 `--batch-size` 分块调用
   `llm.generate(prompts, SamplingParams(n=m, temperature=…, top_p=…, max_tokens=…, seed=…), use_tqdm=False)`；
   每个 `RequestOutput.outputs[i]` → `text`/`finish_reason=="length"`，语义与 HTTP 的
   `choice.message.content`/`choice.finish_reason` 一一对应（空串按现有规则丢弃）。
5. 分块内部异常 → 该块逐条记为 failure（等价于现有 `--retries` 已耗尽），不打断整轮；
   块大小默认 128，逐条回调写盘（沿用 `append_record`），**引擎崩溃最多丢一块**。
6. `--concurrency/--timeout/--retries` 对离线后端无意义：忽略并在启动时打印一行说明
   （HTTP 后端行为不变）。

### 3.2 CLI / 配置接口

新增（全部有默认值，`openai` 为默认，**不破坏现有调用**）：

```
--backend {openai,vllm-offline}        默认 openai
--model-path PATH                      vllm-offline 必填（教师权重目录）
--tensor-parallel-size N              默认 1
--max-model-len N                     默认 12288
--gpu-memory-utilization F            默认 0.90
--dtype STR                           默认 bfloat16
--enforce-eager                        可选
--gpus "1,2,3,4"                       进程级 CUDA_VISIBLE_DEVICES（离线后端）
--batch-size N                         默认 128（分块）
--engine-kwargs-json '{}'              逃生口，合并进 LLM(...) kwargs
--progress {log,tqdm,quiet}            默认 log（每 --log-every 条一行）
--log-every N                          默认 100
--log-file PATH                        把本进程 stdout/stderr 重定向到文件（默认 None）
--base-url / --api-key                 仅 openai 后端必填（校验前置）
```

配置文件（`sample.example.yaml` 与用户配置）：

```yaml
teacher:
  engine: openai              # openai | vllm_offline
  base_url: ...               # openai
  api_key_env: ...            # openai
  model: <served name>        # openai；同时是指纹里的 model id
  model_path: /path/teacher   # vllm_offline
  tensor_parallel_size: 4
  max_model_len: 12288
  gpu_memory_utilization: 0.90
  dtype: bfloat16
  gpus: "1,2,3,4"
  batch_size: 128
  log_file: <out_dir>/logs/sample.log
```

### 3.3 指纹与索引兼容（重要）

* `build_teacher_fingerprint_payload(...)` 增加可选参数 `extra: dict|None`，并入 payload → 参与 digest。
* 两个后端的 `provider` 不同（`openai_compatible` vs `vllm_offline`），且离线侧额外记录
  `{backend, model_path, vllm_version, dtype, tensor_parallel_size, max_model_len}`。
* **后果（有意为之）**：HTTP 产生的部分索引不能被离线后端续采——`check_resume_consistency`
  会因 header digest 不一致而**拒绝且不修改文件**。
  迁移办法（二选一，需用户决定）：
  1. 把 `teacher_index.jsonl` 重命名为 `teacher_index.http_partial.jsonl`（或删除），离线从头采；
     代价 = 已采的 116 条（约十几分钟 GPU）；
  2. 先用 `--backend openai` 把 HTTP 那轮跑完，再切离线（仅当教师服务仍可用时才划算）。
* `m/temperature/top_p/max_tokens/enable_thinking/seed/base_url|model` 仍参与指纹；
  `batch_size/concurrency/gpu_memory_utilization` **不**参与（不影响采样分布）。

### 3.4 「断连不中断」的设计保证（无 sudo）

1. `--log-file` 让采样器**不往终端写**：即使读端消失也不可能阻塞在 `write(2)`。
2. `--progress log --log-every 100` 把输出量压到 ~180 行（17,917 条），彻底消除背压诱因。
3. 文档给出无 sudo 的标准起法：
   `setsid nohup <cmd> --log-file <out>/logs/sample.log > /dev/null 2>&1 < /dev/null &`
   （本机 `KillUserProcesses` 为默认 `no`，脱离终端 + 忽略 SIGHUP 即可长期存活；
   不需要 `loginctl enable-linger`。）
4. 由于教师就在同一进程内，**不再存在"教师被杀/被冻结"这一半问题**。

### 3.5 脚本与文档改动

| 文件 | 改动 |
| --- | --- |
| `scripts/run_sample.sh` | 按 `teacher.engine` 组装参数；离线时导出 `CUDA_VISIBLE_DEVICES=$(teacher.gpus)`、传 `--log-file`；`sample` 阶段不再提示需要教师服务 |
| `scripts/serve_teacher.sh` | 保留（远程教师/HTTP 模式仍用），文档标注「离线模式不需要」 |
| `scripts/run_sample_teacher.sh` | 当配置 `engine: vllm_offline` 时跳过起服务逻辑，直接跑阶段（也可考虑改名为 `run_sample_detached.sh`） |
| `configs/sample.example.yaml` | 增加 `teacher.engine` 与离线键，默认 `openai` |
| `sample/README.md`、`gad/PIPELINE.md` §2 | 两种起法并列；离线为推荐单进程模式；补「无 sudo + 断连」说明 |
| `requirements.txt` | 注明 `vllm` 仅 `vllm_offline` 后端需要（可选依赖，不写死版本，沿用训练环境同一版本） |
| `runs/configs/sample_qwen3_1.7b.yaml`（用户配置） | 切到 `engine: vllm_offline`，`gpus: "1,2,3,4"`, `tensor_parallel_size: 4` |

---

## 4. 实施顺序

1. **重构（无行为变化）**：抽出 `teacher_engines.py`，`OpenAIEngine` 承接现有逻辑；`sample_teacher.py`
   只剩 CLI + 数据面（`load_prompts`/`prepare_index`/`check_resume_consistency`/`append_record`/summary）。
   跑通现有 7 个 HTTP mock 用例（必须逐条不变）。
2. **新增 `VllmOfflineEngine`** + CLI/配置键 + `--log-file/--progress` + 参数校验。
3. **测试**（见 §5）。
4. **脚本/文档/示例配置**更新。
5. **用户配置迁移**：重命名旧 HTTP 部分索引 → `--limit 100` 测速（同时验证断连不中断）→ 全量。
6. 回归：`bash LMOps/gad/tests/run_all.sh <python>` 全绿。

---

## 5. 测试计划

**CPU（必须，可在本沙箱跑）**

| 用例 | 断言 |
| --- | --- |
| `FakeEngine` 注入（Python API） | 分块→回调→写盘顺序；resume 跳过已完成 uid；`--limit` 生效 |
| 失败路径 | 引擎对某 uid 抛错 → 进 `<index>.failures.jsonl`、退出码 1（`--allow-incomplete` 时 0） |
| 指纹稳定性 | 同参数两次运行 digest 相同；改 `m/temperature/max_tokens/enable_thinking/model` digest 变化；改 `batch_size/concurrency` **不**变化 |
| provider 隔离 | 用 HTTP 写一个索引，再用 `vllm-offline` 续采 → 非零退出且**文件未被修改** |
| 懒加载 | `--backend openai` 的子进程里 `"vllm" not in sys.modules`（防止 CPU 环境被迫安装 vllm） |
| 模板一致性 | 用本地 Qwen3-1.7B tokenizer 渲染 `enable_thinking=false/true`，断言渲染文本与 HTTP `chat_template_kwargs` 语义一致（含 `add_generation_prompt`） |
| CLI 校验 | 离线缺 `--model-path` → exit 2；HTTP 缺 `--base-url` → exit 2 |
| 进度/日志 | `--log-file` 后进程 stdout 不是 pty（写文件）；`--progress log --log-every 100` 行数上界可断言 |

**GPU（你的机器，一条命令，小模型）**

```bash
$PY LMOps/sample/sample_teacher.py --backend vllm-offline \
  --model-path /mnt/jinbo/RLRM/model/Qwen/Qwen3-0.6B \
  --prompt-parquet <out>/train.parquet --out /tmp/offline_probe/teacher_index.jsonl \
  --m 1 --max-tokens 64 --limit 2 --progress log --log-file /tmp/offline_probe/run.log
# 期望：2 条记录 + header；再跑一次 → "already sampled"，新增 0 条（续采）
```

**断连验收（你机器，2 分钟）**：`wc -l` 断开前后对比，断开期间条数应**继续增长**
（这正是本次修改要拿到的新行为）。

**教师质量/长度统计**：全量首 100 条后比较 `stats.json`（学生 tokenizer 的 p50/p95/p99、截断率、
`truncated_before_final_answer`）与 HTTP 路径是否量级一致（应一致：同权重、同模板、同采样参数）。

---

## 6. 验收标准

1. 单进程一条命令完成 17,917 条采样；无端口、无 `serve_teacher.sh`、无 sudo。
2. 断开 VS Code SSH ≥ 2 分钟期间，索引条数持续增长；重连后无异常。
3. 新索引可被 `build_gad_data.py` / `validate_data.py` 直接消费（契约零改动）；
   `sample_teacher.py --summary-out` 里的 `teacher_fingerprint_digest` 能填进训练配置并通过预检。
4. 现有 `openai` 路径、7 个 HTTP 用例、全套 CPU 套件（当前 10 套件 111 用例）保持全绿。

---

## 7. 风险与回滚

| 风险 | 缓解 |
| --- | --- |
| 离线引擎与 HTTP 引擎输出分布略有差异（vLLM 版本/批调度数值差异） | 指纹隔离 provider + 记录 vllm 版本；同一轮不要混用两个后端；报告里写清用的是哪个 |
| 进程内 vLLM 崩溃 → 教师重载（~4 分钟） | 分块 + 逐条落盘；重跑即续采；`--log-file` 保留崩溃现场 |
| 采样器进程与训练共用 GPU 竞争 | 文档明确：采样完再训练；`--gpus` 显式限定卡片 |
| 模板 kwargs 传法不一致（例如教师模板不认识 `enable_thinking`） | 模板一致性单测 + 渲染失败时降级并**显式记录**（不静默改语义） |
| `m>1` 语义与 HTTP 不同（一次调用返回 m 条） | 断言 `num_answers`、`finish_reasons`、`truncated` 与 HTTP 一致；`select` 语义不变 |
| 回滚 | 保留 `--backend openai` 与 `serve_teacher.sh`，默认值仍是 `openai`；本次改动不改索引 schema，可随时切回 |

---

## 8. 工作量估计

| 步骤 | 估计 |
| --- | --- |
| 引擎抽象重构 + HTTP 用例回归 | 0.5 天 |
| `VllmOfflineEngine` + CLI/配置 + 进度/日志 | 0.5 天 |
| CPU 测试（7 类）+ GPU 小模型冒烟脚本 | 0.5 天 |
| 脚本/文档/示例配置 + 用户配置迁移 | 0.5 天 |

---

## 9. 需要你确认的三点

1. **旧的部分索引**（HTTP，已 116 条）：重命名丢弃后离线重采（推荐，代价约十几分钟 GPU），
   还是先跑完 HTTP 再切离线？
2. **示例配置默认值**：`configs/sample.example.yaml` 保持 `engine: openai`（兼容），
   只有你自己的 `runs/configs/sample_qwen3_1.7b.yaml` 切到 `vllm_offline` —— 可以吗？
3. **`--log-file` 默认行为**：由 `run_sample.sh` 默认写入 `<out_dir>/logs/sample.log`
   （即使你忘记重定向也不会被断连卡住），是否同意？

---

## 10. 不等实施的临时绕过（今天就能用，无 sudo）

```bash
# 采样器（教师仍在 tmux 里也行）：日志进文件 + 完全脱离终端
cd /mnt/jinbo/RLRM
setsid nohup env NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
  bash LMOps/sample/scripts/run_sample.sh \
    --config runs/configs/sample_qwen3_1.7b.yaml \
    --python /mnt/jinbo/miniconda3/envs/gad/bin/python --stage sample \
  > /mnt/jinbo/RLRM/dataset/dapo_math_17k_gad/logs/sample.log 2>&1 < /dev/null &
tail -f /mnt/jinbo/RLRM/dataset/dapo_math_17k_gad/logs/sample.log
```

对**教师**同样有效：把它的 stdout 从 pty 换成文件（`serve_teacher.sh ... > teacher.log 2>&1`），
它就不会再因为 pane 不被读取而阻塞；否则即使采样器脱离了终端，教师仍会被卡住。
