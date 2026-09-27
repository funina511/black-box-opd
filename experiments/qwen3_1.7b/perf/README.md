# Warmup / GAD 性能试验

候选覆盖只定义在 [candidates.yaml](candidates.yaml)。正式配置仍是 `../train.yaml`，未经实测不改其 KV 比例或判别器 micro。

本轮以 Warmup 的完整 checkpoint step 100 为同一起点。每个试验都把它复制到独立 `runs/performance_benchmarks/<run_id>/warmup`，恢复 actor/critic 模型、优化器、scheduler、RNG 与 dataloader。试验只产生自己的 checkpoint、日志和 `benchmark_result.json`，不写正式 run 的阶段完成标记。

基线已按原计划运行 7 个新增步骤：前 2 步预热，随后 4 个普通测量步和 1 个最终保存步。KV 0.50 在完成 step 104 后、critic micro=2 在完成 step 102 后按用户要求中断，均未保存最终 checkpoint。两组结果只保留为部分观察。用户决定暂停后续 GPU 短跑，直接修改正式配置并启动实验。

```bash
ROOT=/mnt/jinbo/RLRM/LMOps
bash "$ROOT/scripts/benchmark_train.sh" --candidate baseline               # 仅配置与 Hydra 检查
bash "$ROOT/scripts/train.sh" --stages warmup
```

GPU 短跑目前暂停。正式配置参数位于 `../train.yaml`：`rollout.gpu_memory_utilization` 控制生成阶段 KV 池，`optim.critic_micro_batch_size_per_gpu` 控制判别器微批。每组短跑的 `reports/gpu_samples.jsonl` 记录显存，`reports/benchmark_result.json` 记录已完成步数和阶段耗时；只有完成目标的结果才写入正式 run 的 `reports/performance/`。

使用 `python gad/tools/compare_training_perf.py <baseline.json> <candidate.json> ... --output <comparison.md>` 对照。工具会校验同一 checkpoint、后端指纹及仅限 A/B 的配置差异，并按共有的普通训练步比较；只有 2 个普通步的短筛选结果必须在正式采用前复测。生成 token 数可能因 vLLM 调度改变，应同时比较回答长度、截断率、估算 tokens/s 与总步时。`timing_s/gen` 已含 `generate_sequences`，不可相加。

正式续跑时，SeqKD 的已完成标记按阶段相关训练输入校验；调整 Warmup/GAD 的 KV 与 critic micro 不会让 SeqKD 重训。Warmup 完成并产生最终 handoff 后再建立 GAD n=8 的同类短跑，不能把 Warmup 的收益直接套用到 GAD。
