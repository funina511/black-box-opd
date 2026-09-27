# 验证提速基线与复测

## 调整前实测

来源：迁移保留的 `runs/gad_qwen3_1.7b/qwen3-1.7b-4x3090/logs/seqkd.log`，提取 `timing_s/testing`。在 SeqKD step 50、100、…、600 共 12 次完整 512 题验证，耗时范围 105.1–117.2 分钟，平均 109.9 分钟，合计 21.99 小时。训练设置：每 50 步验证、`val_batch_size=16`、回答上限 8192 tokens。

## 调整后实测（SeqKD）

新配置每 200 步做固定 128 题 quick 验证，阶段末做 512 题 full 验证；保存频率仍为 50，`val_batch_size=64`。SeqKD 的 quick 验证在 step 800、1000 分别耗时 750.1、835.6 秒（平均 13.2 分钟），其中生成平均 575.1 秒、ROUGE-L 平均 216.1 秒；step 1084 的 full 验证耗时 3023.4 秒（50.4 分钟）。来源为该 run 的 `seqkd/validation_math_metrics.jsonl`。quick 与 full 的样本数不同，准确率曲线应分别标注。Warmup/GAD 的验证耗时仍需各阶段运行后另行记录。

## Warmup 训练基线（2026-09-25）

原日志最近 20 个正常训练步（114–133）：平均总步时 433.9 秒，`gen` 278.3 秒，`generate_sequences` 251.3 秒，`update_critic` 85.7 秒，`update_actor` 35.9 秒；平均回答长度 5918 tokens。`gen` 包含 `generate_sequences`，两者不能相加。`perf/max_memory_allocated_gb` 是分布式日志聚合值，不是单卡显存峰值。完整 JSON/Markdown 报告在运行目录 `reports/performance/warmup_baseline_*.{json,md}`。GPU 候选试验使用[隔离短跑入口](perf/README.md)；正式 `train.yaml` 暂保持 KV 0.40、critic micro=1。
