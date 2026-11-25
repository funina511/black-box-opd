# H200 前期适配与安装顺序

2026-11-14。本轮按用户安排完成**环境安装与 baseline 短测**。用户已确认本轮可使用 `/tmp` 并在实验完成后复制重要产物，因此正式 baseline 输出已设为 `/tmp/bloxopd_h200_runs`；存储清理和启动计划检查已完成，训练尚未启动。baseline 复用现有 Qwen3-0.6B，并已生成共同单头 Critic；**不需要下载 8B 或启动 Judge**。8B 与 Judge 属于后续 rubric 实验的独立准备步骤。

用户已明确授权清理，四个剩余短测 checkpoint 已删除，释放约53 GiB；清理后 `/tmp` 约余79–80 GiB。短测日志、配置及小状态已归档为 `reports/h200_deployment/smoke_metadata_20261114.tar.gz`，该归档不含模型/优化器权重，旧短测不能继续恢复或导出。删除记录见 `reports/h200_deployment/formal_storage_preparation.json`。正式训练从保留的原始学生和共同 Critic 初始化开始，不依赖短测权重。当前余量高于此前约65.5 GiB的baseline保存峰值估算，但低于100 GiB的宽裕建议，运行期间需保留这部分空间。

目标环境已安装并通过 `pip check`、CUDA BF16 和 FlashAttention 前后向检查，无需重复安装。baseline 已完成 SeqKD2、Warmup2、GAD2→3 保存后恢复至5、导出模型的两题外评，以及正式 batch64/n4/8192 形状的单步提速对照。采用后的参数与边界见下文；机器可读汇总为 `reports/h200_deployment/deployment_summary.json`。原始方法与短测步骤见[交接文档](../docs/20261105/H200新服务器部署与实验交接_20261114.md)。

## 本轮实测：baseline 提速参数

两组均从同一个工程 Warmup2 出发，完整 Teacher 池、batch64、n4、8192回答上限、seed42和学习率不变；各执行一次真实 GAD 更新。工程 run 设置 `critic_warmup=0`，正式配置仍保留10。验证仅用原dev前4题，保存/验证均每步执行。

| 参数或实测项 | 原参照 | 已采用候选 |
|---|---:|---:|
| rollout.gpu_memory_utilization | 0.30 | **0.40** |
| rollout.max_num_seqs | 64 | **128** |
| Critic microbatch | 2 | **4** |
| Actor token预算 | 10240 | 10240 |
| 生成耗时 | 44.662 s | 42.671 s |
| Critic更新耗时 | 86.617 s | 60.830 s |
| Actor更新耗时 | 9.914 s | 9.815 s |
| 单步总耗时（含验证/保存） | 174.741 s | 146.893 s |
| 平均回答token | 776.117 | 804.418 |
| 最长回答token | 8192 | 8192 |
| Torch峰值 allocated / reserved | 58.915 / 67.605 GiB | 72.623 / 81.221 GiB |

本次单步总时间减少约15.9%，主要来自Critic更新时间减少约29.8%；候选回答平均更长约3.6%。两组均完成256条生成、Actor/Critic更新、checkpoint与4题数学验证，未出现CUDA OOM。只有单步、组合参数对照，不能把收益归因于单独提高utilization，也不能承诺整轮固定提速或最坏长度分布下无OOM；初始化、HF导出和首次图捕获不包含在step计时内。完整token/阶段/显存采样和比较见 `reports/h200_deployment/baseline_throughput_comparison.json`。

`gpu_memory_utilization`控制vLLM实例显存规划，增加预算主要扩大可用KV空间，不是GPU计算利用率；实际训练后端在生成结束后sleep释放KV，所以训练或保存时显存下降正常。并发128仍生成同样64题×4条回答。[vLLM 0.8.5 参数说明](https://docs.vllm.ai/en/v0.8.5/serving/engine_args.html#cacheconfig)列出了显存预算和调度参数的含义。

Actor动态token预算保留10240：当前实现先按每个微批计算token均值，再按样本数加权，改变动态分桶可能改变有效损失权重。Critic固定等大小微批2→4仍覆盖相同逻辑minibatch并累积到一次optimizer更新。代码依据和限制见 `reports/h200_deployment/baseline_throughput_notes.md`。这次只调整baseline；naive/Actor仍保留待共卡实测的0.30/64/micro2，不能直接套用无Judge结果。

## 机器与资产：上一轮检查快照

- 单张 H200 NVL，143771 MiB 显存；容器实际 CPU 配额 15 核、内存 250 GiB。不要按宿主机可见的 128 CPU / 2 TiB 规划并发。
- 项目所在盘检查时仅剩约 24 GiB。长期模型和 run 的存储由用户后续安排；当前安装及短测仍需预留实际写入峰值，正式长训练暂缓。
- 已有 `gad` / Python 3.10.22 环境，上一轮检查时核心训练依赖尚未安装；shell 的 base 和 `/opt/conda` 都不是目标训练环境。
- 已有学生模型 `../model/Qwen/Qwen3-0.6B`；本地 HF 下载 metadata 记录 revision `c1899de289a04d12100db370d81485cdf75e47ca`，上一轮仅核对文件。新配置直接引用它，真实加载由本轮初始化及短测验证。
- Teacher v2 数据、八任务评测数据和 `third_party/verl/verl.bundle` 已在预期目录。训练/dev/quick 的 manifest 行数为 17405/512/128；实际 Parquet 读取留给安装后的预检。
- 本轮准备共同单头 Critic 初始化和固定后端。Judge 模型 `assets/models/Qwen3-8B` 留到后续 rubric 阶段，不阻塞 baseline。

## 环境安装：按顺序执行

下面按实际依赖顺序列出安装命令；执行状态以本轮部署报告为准。先确认现有空间能容纳环境安装和短测保存峰值，长期存储落实后再启动正式训练。`--no-cache-dir` 只减少 pip 下载缓存，不能减少已安装包和 checkpoint 的必要占用。

### 1. 激活已经创建的环境，生成安装清单

```bash
cd /root/siton-tmp/home/jinbo/bloxopd/LMOps
source /root/siton-tmp/home/jinbo/miniconda3/etc/profile.d/conda.sh
conda activate gad
export GAD_PYTHON="$(command -v python)"
python --version
mkdir -p reports/h200_deployment

python - <<'PY'
from pathlib import Path
source = Path('environments/requirements.lock.txt')
lines = [line for line in source.read_text().splitlines()
         if not line.lower().startswith(('verl==', 'flash-attn=='))]
Path('reports/h200_deployment/requirements.install.txt').write_text('\n'.join(lines) + '\n')
PY
```

该安装清单只排除两个需要单独处理的包：verl 由固定 bundle 提供，FlashAttention 先匹配二进制 wheel。其余仍使用源环境锁定版本。本机已完成依赖求解和安装，`pip check` 通过；没有用 `--no-deps` 跳过依赖。

本次先用 pip dry-run 验证依赖，再用临时安装的 uv 加速实际安装，并复用已下载的 CUDA wheel。Torch 使用官方 cu124 来源，FlashAttention wheel 的 SHA256 与 `environment.json` 记录一致；其余包按上述安装清单安装。uv 只作为 `/tmp` 内的安装工具，不是新增项目依赖。实际命令与版本分别保存在 `reports/h200_deployment/torch_install_command.json`、`training_stack_install.log` 和 `pip-freeze.actual.txt`；下述 pip 命令保留为标准安装顺序。

### 2. 安装 Torch 2.6 / CUDA 12.4 运行时

```bash
python -m pip install --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

命令来自 [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v260)。驱动显示的 CUDA 能力、系统 nvcc 的 12.8 toolkit、Torch wheel 自带的 12.4 运行时是不同层次；不据此升级整个训练栈。

### 3. 安装 vLLM 0.8.5

```bash
python -m pip install --no-cache-dir vllm==0.8.5 \
  -c reports/h200_deployment/requirements.install.txt
```

[vLLM 0.8.5 官方安装说明](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)说明默认二进制使用 CUDA 12.4，并提醒 Torch/CUDA 构建的兼容性。这里的约束文件用于保持项目依赖版本；发生冲突时先查看具体包，不静默解除约束升级。

### 4. 匹配 ABI，安装 FlashAttention 2.7.3

先检查刚安装的 Torch：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)'
```

`environment.json` 记录的原 wheel 是 Linux x86_64、Python3.10、Torch2.6、CUDA12、CXX11 ABI FALSE。以下命令仅在这些条件匹配时安装它；不匹配时停止并选择同版本的对应构建。若最终需要编译，再检查匹配的 toolkit 和编译依赖。

```bash
python - <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path
import torch

compatible = (sys.version_info[:2] == (3, 10)
              and platform.system() == 'Linux' and platform.machine() == 'x86_64'
              and torch.__version__.split('+')[0] == '2.6.0'
              and torch.version.cuda == '12.4'
              and not torch._C._GLIBCXX_USE_CXX11_ABI)
if not compatible:
    raise SystemExit('当前 Python/Torch/CUDA/ABI 与记录的 wheel 不匹配，请先选择匹配构建。')
record = json.loads(Path('environments/environment.json').read_text())
url = record['installed_origins']['flash-attn']['url']
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                       '-c', 'reports/h200_deployment/requirements.install.txt', url])
PY
```

### 5. 安装其余锁定依赖，检查实际环境

```bash
python -m pip install --no-cache-dir -r reports/h200_deployment/requirements.install.txt
python -m pip check
python -m pip freeze > reports/h200_deployment/pip-freeze.actual.txt
python - <<'PY'
import torch, vllm, transformers, flash_attn, xgrammar, ray, pyarrow
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('vllm:', vllm.__version__, 'transformers:', transformers.__version__)
print('flash_attn:', flash_attn.__version__, 'ray:', ray.__version__)
print('gpu:', torch.cuda.get_device_name(0))
x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
print('bf16 matmul:', (x @ x.T).shape)
torch.cuda.synchronize()
PY
```

最后一段会实际使用 GPU，安装完成后在空闲卡执行并记录结果。版本、导入、BF16 矩阵乘法通过之后，本轮通过 baseline 短测继续验证真实 FlashAttention、vLLM 和单卡 FSDP；Judge 结构化请求留到后续 rubric 阶段。

## 本轮模型准备：现有 0.6B 与共同单头初始化

学生模型使用现有目录，以下直接从它生成 baseline 所需的共同单头 Critic。该工具名称中的 `rubric` 不意味着需要 Judge；此命令不下载或加载 8B。

```bash
python gad/tools/prepare_rubric_critic.py \
  --source ../model/Qwen/Qwen3-0.6B \
  --target assets/initialization/qwen3_0.6b_math_v2_adv_seed42 \
  --seed 42 --single-head --spec rubric/specs/math_process_direct_v1.yaml
```

baseline 使用该共同单头；以后执行 rubric Warmup 时，pipeline 才在它自己的 run 内派生四头并检查 adv 行不变。SeqKD 独立从学生模型训练，不输入 Warmup。新 run 的 scorer 指纹在完整环境中生成，保留真实身份检查。现有统一路径预检会检查共同 Critic，因此即使首先只测 SeqKD，也先完成此初始化。

## 三份 H200 配置

| 配置 | 初始化与用途 |
|---|---|
| [h200_baseline_n4.yaml](../configs/h200_baseline_n4.yaml) | 初始学生 + 共同单头；公共 SeqKD、baseline Warmup、baseline GAD |
| [h200_naive_n4.yaml](../configs/h200_naive_n4.yaml) | 初始学生 + 由共同单头派生的四头；rubric Warmup、naive GAD、影子记录 |
| [h200_actor_n4.yaml](../configs/h200_actor_n4.yaml) | 从 `runs/h200_naive_06b_17k_n4` 的 Warmup271 另起 GAD；正式选参确认仍为 false |

三份均使用真实单卡 GPU `"0"`、训练 TP1、Judge TP1；GPU ID 用字符串，避免数值 0 被当成未设置。baseline 不启用 Judge；naive/Actor 显式开启 `resources.allow_judge_gpu_sharing`。默认 false 的旧配置仍拒绝 GPU 重叠；开启后仍检查服务实际 GPU 与声明一致，并校验模型/协议身份。

| 执行参数 | 前期配置 | 说明 |
|---|---|---|
| rollout / Judge 显存比例 | baseline 0.40 / 不启用；rubric 0.30 / 0.30 | baseline单步已测，rubric共卡仍待测 |
| rollout max_num_seqs | baseline 128；rubric 64 | 限制同时调度数；每步仍是64题×4条回答 |
| Actor/Critic offload | false | 三组一致；后续按真实峰值决定是否开启 |
| gradient checkpointing | true | 保留长回答训练的节省显存路径 |
| Critic microbatch | baseline 4；rubric 2 | baseline单步已测，rubric保留源配置起点 |
| 保存频率 | 50 step | 保留恢复机会，阶段结束也保存 |
| keep_last_checkpoints | 1 | 每阶段最新完整保存，加受保护的交接/导出 |

训练方法仍为完整 v2 Teacher 数据、batch64、GAD n4、2048/8192 长度、seed42、FP32 reward；rubric direct_v1、thinking Judge 32768/8192、过程配额32、replay=false。quick128/final512 验证保持启用。显存比例和调度数只是工程起点，不能据此承诺无 OOM 或预计提速倍数。

### 保存一份的准确含义

沿用现有 `RetentionMonitor`，在新 checkpoint 完整后清理旧保存。两个后端分角色保留参数保持 null，避免绕过交接/导出保护。

- SeqKD、Warmup、GAD 分目录保留，不会因为 GAD 前进而删除源 Warmup271。
- Warmup→GAD 会在 GAD 内建立 `global_step_271` 交接副本并导出 HF，该目录受保护；另有最新 GAD checkpoint。因此整个 run 并非只剩一个 step 目录。
- 含 HF 导出权重的其他 checkpoint 也受保护；若手动导出中间模型，它们会额外占用空间。
- 新保存写入时，新旧文件会并存，清理最长通常等待一个30秒轮询周期。仍需为保存峰值留空间。

本机实测：SeqKD 完整 checkpoint 约 **7.26 GiB**；Actor+Critic 完整 checkpoint 约 **13.93 GiB**；GAD 交接副本含 HF 导出约 **16.47 GiB**。公共 SeqKD、源 Warmup、GAD 交接副本和新旧两份 GAD 保存合计约 **65.5 GiB**，还未含其他导出、日志和临时余量。建议正式 baseline 输出目录至少预留 **100 GiB 可用空间**，不是只按模型权重大小预留。安装环境与共同 Critic 后，项目盘本轮约余13 GiB，不能承担正式 run。

小规模续训验收后，已清理其 `/tmp` GAD step2/5 checkpoint；原正式形状参照验收后，也清理了它的 GAD step2/3，每次释放约30.4 GiB，以供下一个工程对照使用。随后经用户明确同意，又删除剩余的 Warmup2、SeqKD2 和加速候选 GAD2/3，为正式实验释放约53 GiB。所有旧短测权重均已清理，日志、配置、小状态快照和验收证据已保留或归档。历史完成记录描述清理前实际通过的训练；清理记录为 `temporary_smoke_cleanup.json`、`temporary_formalshape_cleanup.json` 和 `formal_storage_preparation.json`。

本轮沿用原交接、恢复和清理流程；导出工具新增单卡 NO_SHARD Tensor 支持，见下文。自动保留策略只清理本次 run 内已被新保存替代的旧 checkpoint。后续安置存储时，在首次启动前修改三份配置的 `run.output_root`，并同步 Actor 的 `budget.init_from_warmup_run`；模型搬迁则更新对应路径。已有 run 续训使用冻结配置，不靠改新配置来迁移历史 run。

## 本轮安装完成后的 baseline 预检和短测

先准备 baseline 后端并检查计划。使用本项目 bundle，不能另行 `pip install verl`。这两条命令使用关闭 rubric 的 baseline 配置，不要求 8B 路径存在或 Judge 服务在线：

```bash
bash scripts/preflight.sh --config configs/h200_baseline_n4.yaml \
  --stages seqkd,warmup,gad --prepare-backends \
  --report reports/h200_deployment/preflight_baseline.json
bash scripts/train.sh --config configs/h200_baseline_n4.yaml \
  --run-id h200_baseline_06b_17k_n4 --stages seqkd,warmup,gad --dry-run
```

接着按交接文档第8节建立独立 `h200_smoke_*` run，本轮覆盖 SeqKD、baseline Warmup→GAD、保存恢复、数学验证和小规模外评。1–2步短测显式设置 `optim.critic_warmup=0` 并核查 Actor 真更新；正式配置保留原门槛。baseline 的正式 batch64/n4/8192 上限验证记录实际回答长度、显存峰值及生成/训练/保存耗时；小输出逻辑短测与正式形状验证分别报告。

上一轮安装前的 CPU 定向测试与静态配置快照见 `reports/h200_deployment/preparation_checks.json`。本轮预检、真实更新、恢复与外评分别记录在同目录的 `preflight_baseline_smoke.json`、`seqkd_checkpoint_verification.json`、`warmup_checkpoint_verification.json`、`gad_resume_verification.json`、`external_eval_smoke_results.json`。其中恢复核对了优化器/scheduler从1→3、数据游标12→20样本、采样器与扩展状态连续，Actor/Critic参数继续变化；四题dev与两题外评只用于工程验收，不用于质量结论。

### 本机 Ray 启动与信号栈兼容

首次 SeqKD 短测在 `ray.init()` 内启用 `faulthandler` 时出现 `OSError: [Errno 12] Cannot allocate memory`，当时尚未发生模型更新。独立 CPU 复现表明：导入完整 `verl.trainer.main_ppo` 后再启用会失败，但进程启动时设置 `PYTHONFAULTHANDLER=1`，导入后仍保持启用且可再次调用；两次的 `RLIMIT_AS` 和 `RLIMIT_DATA` 均无限制，未发现 scorer 或导入守卫修改这些限制。

`pipeline.stage_env` 现在为训练进程设置 `PYTHONFAULTHANDLER=1`，让解释器在导入原生库之前安装诊断信号栈，后续 Python 子进程继承该环境。该改动保留故障诊断。相关 CPU 回归验证三个阶段生成的环境均能在 Python 启动时启用它；修复后的 SeqKD、Warmup、GAD 和断点恢复短测均通过。

这一现象符合 Linux 动态 XSTATE/AMX 与备用信号栈的约束：已允许的特性需要更大的信号帧时，过小的 `sigaltstack` 会返回 ENOMEM；反过来，提前安装较小的信号栈也可能使之后的 CPU AMX 权限申请被拒绝。因此提前启用会影响此进程的 CPU AMX 可用性，不能据此声称与此前 CPU 指令选择完全一致。当前训练主体使用 H200 CUDA；后续若研究 CPU AMX 性能，应单独验证匹配的信号栈方案。机制依据见 [Linux 内核 XSTATE 文档](https://docs.kernel.org/arch/x86/xstate.html)。

### 单卡 FSDP checkpoint 导出

单张 H200 的 FSDP 使用 NO_SHARD，真实 checkpoint 保存普通 Tensor；原 `gad/tools/merge_model2hf.py` 要求 DTensor，会阻塞 Warmup→GAD 和外评导出。现已为唯一 `world_size=1/rank0`、全普通 Tensor 的保存增加严格支持，保留 BF16 导出口径、Actor tied embeddings 和 Critic score 头；普通 Tensor 的多卡分片、额外 rank 或混合非 Tensor 数据仍拒绝导出。原 DTensor 合并路径继续保留。

真实 tiny Qwen Actor/Critic 导出后逐项重载对比通过，原 CPU DTensor 交接回归也通过。检查记录为 `reports/h200_deployment/single_rank_export_checks.json`。真实 0.6B 交接和外评仍由标准工具执行完整性校验。

## 后续 rubric 阶段：单独准备 8B 与 Judge

这一节在开展 naive/Actor 实验时执行，本轮 baseline 环境与短测无需执行。仅在缺失时下载 Judge，并记录解析到的 revision；已有完整 8B 时复用实际来源记录。

```bash
python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'Qwen/Qwen3-8B'
revision = HfApi().model_info(repo).sha
target = Path('assets/models/Qwen3-8B')
snapshot_download(repo_id=repo, revision=revision, local_dir=str(target))
record = {'repo_id': repo, 'revision': revision, 'path': str(target.resolve())}
Path('reports/h200_deployment/judge_model.json').write_text(json.dumps(record, indent=2) + '\n')
PY
```

模型准备后，在空闲卡先启动 Judge，等待加载完成并通过 health，再运行 rubric 预检。现有启动器保留忙卡检查；Judge 不注册为第二张 Ray GPU。

```bash
bash scripts/rubric_judge.sh --config configs/h200_naive_n4.yaml --action start
bash scripts/rubric_judge.sh --config configs/h200_naive_n4.yaml --action logs
# 等待日志确认服务就绪，再执行：
bash scripts/rubric_judge.sh --config configs/h200_naive_n4.yaml --action health
bash scripts/preflight.sh --config configs/h200_naive_n4.yaml \
  --stages warmup,gad --prepare-backends \
  --report reports/h200_deployment/preflight_naive.json
```

再按交接文档第8节补齐 rubric Warmup→GAD、恢复和 Actor 占位奖励通路短测，以及至少一次正式 batch64/n4/8192 上限的共卡更新。检查有效 c/p/e、mask、真实结构化请求，记录生成/标注/训练/保存耗时和峰值显存；baseline 短测通过不代替 rubric 共卡验收。

## 存储落实且验收通过后才执行的正式顺序

以下命令用于后续安排，本轮只做环境与短测，正式长训练等待用户落实存储。公共 SeqKD + baseline 可独立运行；完成上述 rubric 独立准备后，再启动 Judge 并等待 health，运行 rubric Warmup + naive：

```bash
bash scripts/train.sh --config configs/h200_baseline_n4.yaml \
  --run-id h200_baseline_06b_17k_n4 --stages seqkd,warmup,gad

bash scripts/rubric_judge.sh --config configs/h200_naive_n4.yaml --action start
# 等待加载完成，health 通过后再训练。
bash scripts/rubric_judge.sh --config configs/h200_naive_n4.yaml --action health
bash scripts/train.sh --config configs/h200_naive_n4.yaml \
  --run-id h200_naive_06b_17k_n4 --stages warmup,gad

python gad/tools/select_actor_reward.py \
  --run runs/h200_naive_06b_17k_n4 --output reports/h200_actor_selection
```

预计公共 SeqKD542、两类 Warmup各271、每组 GAD新增271步（271→542）。naive 必须保留272–542完整影子日志和最终checkpoint，供选参使用。

Actor 的分量/w/tau 需由新 naive 结果确定并经用户确认，之后修改 Actor 配置并置 `rubric.actor_reward_selection_confirmed=true`。确认读取的是新 rubric Warmup271 后，再执行：

```bash
bash scripts/train.sh --config configs/h200_actor_n4.yaml \
  --run-id h200_actor_06b_17k_n4 --stages gad --dry-run
# 检查计划及参数无误后，去掉 --dry-run 执行。
```

正式任务在 tmux 等已有终端托管方式中运行，阶段日志位于 `runs/<run_id>/logs/`。八任务16k评测按交接文档第10节显式调用现有 core4/ext4 两个入口；三份配置均关闭旧自动评测入口。
