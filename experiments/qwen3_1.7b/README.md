# Qwen3-1.7B GAD 实验

训练数据为原有 17,405 条 `select=all`，dev 为原有 512 条；教师索引及 UID 划分均保留。SeqKD 从学生初始模型开始，Warmup 也从初始学生开始，GAD 续接 Warmup。迁移清单在运行目录 `reports/migration_manifest.json`；旧数据和 checkpoint 路径仍有链接。

```bash
ROOT=/mnt/jinbo/RLRM/LMOps
bash "$ROOT/scripts/preflight.sh" --stages seqkd
bash "$ROOT/scripts/train.sh" --stages seqkd --dry-run
bash "$ROOT/scripts/train.sh" --stages seqkd
bash "$ROOT/scripts/train.sh" --from-stage warmup
```

SeqKD 的最新完整 checkpoint 是 `global_step_600`。恢复会读取同阶段模型、优化器和 dataloader 状态；第 600 步曾运行验证不代表有 checkpoint，以 latest marker 和完整文件为准。不要并发启动同一个 run_id。实际恢复前先看预检报告；恢复后检查日志中的起始 step、actor 更新和下一个 checkpoint。

验证配置：保存频率 50 步，验证频率 200 步；过程验证固定 128 题，阶段末全量 512 题。`quick_dev.manifest.json` 记录父文件和子集哈希。`validation_math_metrics.jsonl` 记录 scope、数量、哈希及耗时，日志键按 `quick/` 和 `full/` 分开。初始基线由历史验证记录提供，当前续跑不重复做 step 0 验证。回答上限仍为 8192 tokens。`val_batch_size=64` 需要真实 GPU 验证，如显存不足，只将此项退回 16。

独立 SeqKD 基线的训练设置与 Warmup/GAD 协议分别记录。`n_warmup=1` 是本实验调优设置；原版更严格的 n=8 对判别器候选分布不同，不能声称两者无损等价。`select=all` 也是明确实验选择，迁移没有改动。

历史验证耗时与复测表见 [PERFORMANCE.md](PERFORMANCE.md)。

回退：停止进程，将 `validation.quick_enabled=false`、`data.val_batch_size=16`、`resources.test_freq=50` 写入新配置并预检。旧路径链接继续可用；历史配置在 `RLRM/runs/configs/*.historical`，历史 resolved config 保持原样。不要删除迁移清单、UID 子集或已写 checkpoint。
