# Qwen3-0.6B / 5k 探索配置

这是一条独立的超参探索实验，不续接 1.7B 的任何 checkpoint，也不写入其 run 目录。当前 1.7B 训练继续使用原配置。

## 数据

从现有 `data/dapo_math_17k_gad` 的 17,405 条训练 UID 中，用 Python `random.Random(42).sample` 无放回抽取 5,000 条。原始 `train.parquet` 和带教师回答的 `gad_train.parquet` 使用相同 UID、相同顺序；保留原来的 512 条独立 dev 和 128 条 quick dev。抽样结果位于 `data/dapo_math_5k_gad/`，`manifest.json` 记录源文件、输出文件和 UID 列表的 SHA-256。教师回答直接复用现有数据；本地 Qwen3-0.6B 与 Qwen3-1.7B 的 tokenizer 文件哈希一致。

如需在新的空目录重建同一子集：

```bash
cd /mnt/jinbo/RLRM/LMOps
/mnt/jinbo/miniconda3/envs/gad/bin/python sample/make_train_subset.py \
  --raw-train data/dapo_math_17k_gad/train.parquet \
  --gad-train data/dapo_math_17k_gad/gad_train.parquet \
  --dev data/dapo_math_17k_gad/gad_dev.parquet \
  --out-dir data/dapo_math_5k_gad --size 5000 --seed 42
```

工具不会覆盖已有的非空输出目录；重建时先选新目录并核对哈希。不要从 dev 抽样。

## 训练预算

配置见 [train.yaml](train.yaml)，模型为本机的 `Qwen3-0.6B`，GAD 每题 `rollout.n=2`（与当前 1.7B GAD 运行值一致），batch=64。每 epoch 有 `5000 // 64 = 78` 个更新步，末尾 8 条因 `drop_last` 不进入该 epoch：

| 阶段 | 预算 | 目标步数 |
|---|---:|---:|
| SeqKD | 2 epoch | 156 |
| Warmup | 1 epoch | 78 |
| GAD | 从 Warmup step 78 续接，绝对目标 2 epoch | 156，即新增 78 步 |

过程验证频率沿用 200 步，因此这些短阶段只做阶段末完整 dev 验证。首次 GPU 训练尚未运行；0.6B 的实际步时和显存仍待验证。探索结果用于筛选超参，不能直接当作 1.7B / 17k 实验的最终结论。后续改参数时请更换 `run.run_id`，保留固定的训练子集和 dev。

等当前 1.7B 任务释放 GPU 后，可使用：

```bash
cd /mnt/jinbo/RLRM/LMOps
/mnt/jinbo/miniconda3/envs/gad/bin/python gad/tools/preflight.py \
  --config experiments/qwen3_0.6b_5k/train.yaml --stages seqkd,warmup,gad
bash gad/scripts/pipeline/run_all.sh \
  --config experiments/qwen3_0.6b_5k/train.yaml --dry-run
bash gad/scripts/pipeline/run_all.sh \
  --config experiments/qwen3_0.6b_5k/train.yaml
```
