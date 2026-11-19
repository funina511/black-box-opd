# H200 baseline 吞吐分析与提速候选

2026-11-14。依据已完成的SeqKD v2两步、baseline Warmup两步、首次GAD step3、正式形状参照及提速候选日志，以及本机实际backend/vLLM源码。分析任务只读日志；GPU执行由部署任务完成。

结论：正式batch64/n4/8192形状的两组真实更新均通过。`.40 / maxseq128 / Critic micro4`组合将这一次step从174.741秒降至146.893秒（耗时减少15.94%，吞吐倍数1.190）；主要改善来自Critic阶段。该组合已作为baseline默认，Actor token预算保持10240。三项同时变化且每组只有一步，不能把收益单独归因于gpu_memory_utilization，也不能承诺整轮固定提速。

## 已有短测的时间分解

单位秒，均为日志 `timing_s/*`；表中gen已包含generate_sequences，不重复相加。启动、模型加载、编译/图捕获和pipeline导出不在这些step计时内。

| 阶段/step | gen | Critic更新 | Actor更新 | 验证 | 保存 | step总计 | 保存+验证占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SeqKD 1 | 0.582 | — | 2.103 | 8.140 | 8.269 | 19.096 | 85.9% |
| SeqKD 2 | 0.594 | — | 0.573 | 8.090 | 8.233 | 17.491 | 93.3% |
| Warmup 1 | 7.746 | 1.166 | 0.990 | 8.080 | 14.513 | 35.015 | 64.5% |
| Warmup 2 | 7.863 | 1.138 | 0.625 | 7.876 | 14.558 | 32.401 | 69.2% |
| GAD 3 | 7.498 | 4.512 | 1.124 | 7.814 | 14.477 | 37.753 | 59.0% |

GAD step3为4题×4条回答，平均404.812 tokens，最高512，50%到达长度上限；实际约6477个生成token。保存和验证合计22.291秒，剔除这两项后step约15.462秒。GPU低负载的重要原因是小工作量和频繁CPU/I/O阶段，不是仅由utilization参数解释。SeqKD的gen不能与GAD生成直接类比。

短测设置每步保存和验证；正式配置已是save_freq50/test_freq200，且使用quick128/final512。正式训练的验证成本也更大，因此不能用短测四题验证成本直接推算整轮时间。`keep_last_checkpoints=1`减少历史磁盘占用，不减少每次序列化模型和optimizer的成本。

证据：`/tmp/bloxopd_h200_smoke/runs/h200_smoke_seqkd_20261114_v2/logs/seqkd.log`、`/tmp/bloxopd_h200_smoke/runs/h200_smoke_baseline_20261114_v1/logs/warmup.log`、本目录`gad_first_training.log:674`。

## 显存预算在实际后端中的作用

- 本次训练日志指向 `vllm.v1.worker.gpu_worker.Worker`；不要把独立Judge固定的V0设置套到训练rollout。安装版本均为vLLM0.8.5。
- 已安装 `vllm/v1/worker/gpu_worker.py:165–212` 的计算为：`KV可用字节 = 总GPU显存 × gpu_memory_utilization − profiling峰值`，峰值还计入检测到的非Torch分配。该参数控制rollout实例能规划的KV容量，不是Actor/Critic训练显存上限，也不是GPU算力利用率百分比。
- backend `vllm_rollout_spmd.py:151–166`实际传入sleep、utilization和max_num_seqs；`fsdp_vllm.py:185–217`在生成前唤醒、同步权重，生成后sleep(level=1)。vLLM V1 `gpu_worker.py:78–100`将weights迁出并释放其他受管理内存。训练/保存时低显存与生成时峰值可不同。
- max_num_seqs限制同时调度的回答数：正式64题×4共256条，maxseq64或128都不会改变这256条的方法预算。更多KV与更高并发只有在队列/缓存/调度限制生成时才可能带来收益；当前日志没有足够KV利用率、排队/抢占数据支持“已被KV卡住”的结论。
- 首轮smoke是eager，日志明确提示未使用CUDA Graph；正式配置`enforce_eager=false`。图捕获增加启动成本和显存，不能把首次捕获时间算成稳态生成退化。
- 当前固定版本的`free_cache_engine`旧分支只适用于0.5.4/0.6.3；不能用它代替实际sleep路径来推断本版本显存行为。

一秒采样的nvidia-smi均值包含启动/等待/保存，且会漏瞬时峰值；例如首次GAD采样最大25165MiB，而Torch记录峰值allocated25.925GiB、reserved32.664GiB，不能混为一种统计。日志`perf/cpu_memory_used_gb`来自`psutil.virtual_memory().used`（`fsdp_workers.py:626`），也不能当作本进程或容器250GiB配额的使用量。

## 已采纳的候选边界

| 参数 | 正式形状参照 | 提速候选 |
|---|---:|---:|
| rollout utilization | 0.30 | 0.40 |
| rollout max_num_seqs | 64 | 128 |
| Critic microbatch | 2 | 4 |
| Actor动态token预算 | 10240 | **仍为10240** |
| batch / n / response | 64 / 4 / 8192 | 不变 |
| 学习率、KL、逻辑minibatch、优化器步数 | 原方法 | 不变 |

两组从相同工程Warmup2、相同Teacher池、seed42另起独立工程run，各做一次真实GAD更新；工程`critic_warmup=0`显式保留。候选只是执行吞吐比较，不是正式质量对照。三个执行参数同时变化只能证明组合是否值得采用，不能单独归因某一个参数。

Critic micro2→4的代码依据：`fsdp_workers.py:868–878`将逻辑minibatch归一化为64×4/1=256；`dp_critic.py:266–267`按固定大小切分并计算累积次数，`core_algos.py:832–836`的BT loss为每对均值，`dp_critic.py:305`除以累积次数，最后318行只做一次optimizer step。因此128个micro2或64个micro4覆盖同一256对、同一归一化目标；可能有数值舍入差异，micro4可能提高训练激活峰值；本次真实峰值和阶段采样见下表。

**未采用Actor token预算20480的理由：** 日志实际`loss_agg_mode=token-mean`；`dp_actor.py:423–425`按token预算重新分桶，`core_algos.py:540–541`先做每个微批的token均值，`dp_actor.py:507–509`再乘微批样本数/逻辑minibatch。响应长度不等时，改变分桶可改变有效token权重，不保证仅有浮点舍入差异。例如长度1与3的两条样本、各自token loss为0与1，分开时按样本加权为0.5，合并token均值为0.75。因而本次保留Actor10240，不修改原损失实现。

上述backend代码均来自 `gad/.backends_gad_extensions_v1_f0ce22a97d40/gad/verl/`；installed vLLM来自目标gad环境的`lib/python3.10/site-packages/vllm/`。

## 正式形状与组合候选的实测结果

两组均从工程Warmup2更新到GAD3，64题×4回答、上限8192、seed42、Actor token10240，4题真实dev验证。冻结配置的全部差异只有run ID和表中三个执行参数。参照run为`h200_smoke_baseline_formalshape_20261114_v1`，候选为`h200_smoke_baseline_fast_20261114_v1`。

| 指标 | 参照 .30 / 64 / micro2 | 候选 .40 / 128 / micro4 |
|---|---:|---:|
| gen / 秒 | 44.662 | 42.671 |
| reward / 秒 | 8.605 | 4.727 |
| Critic更新 / 秒 | 86.617 | 60.830 |
| Actor更新 / 秒 | 9.914 | 9.815 |
| 四题验证 / 秒 | 4.319 | 6.926 |
| 保存 / 秒 | 14.033 | 14.685 |
| step总计 / 秒 | 174.741 | 146.893 |
| 实际回答token数 | 198686 | 205931 |
| 回答平均长度 | 776.117 | 804.418 |
| 达到8192上限 | 1/256（0.39%） | 2/256（0.78%） |
| 回答tokens / gen秒 | 4448.659 | 4826.018 |
| 回答tokens / step秒 | 1137.031 | 1401.912 |
| 1秒NVML采样最大 / GiB | 51.956 | 65.655 |
| gen阶段采样最大 / GiB | 42.097 | 56.247 |
| Critic阶段采样最大 / GiB | 18.935 | 18.878 |
| Actor阶段采样最大 / GiB | 34.450 | 34.284 |
| Torch allocated高水位 / GiB | 58.915 | 72.623 |
| Torch reserved高水位 / GiB | 67.605 | 81.221 |

候选生成token多3.65%，gen token吞吐高8.48%，整step的回答token吞吐高23.30%；Critic耗时少29.77%。这里的Critic也处理Teacher配对，不能只按学生回答token精确归一化其全部工作。两个run的题目顺序一致，但218/256条回答长度发生变化，验证平均长度也从534.25变为698.25，因此生成/验证耗时变化不是完全相同输出上的计时。单次reward计时变化也不应直接归因执行参数。

两组Actor/Critic更新标记均为1，梯度有限且非零，阶段验证通过；四题评分均为1/4，无未决评分。未观察到CUDA OOM或preemption日志。完成后出现的DataLoader killed traceback均已与Ray主动清理对应PID和成功退出记录闭环，不属于训练OOM。原始异常保留在报告中。

NVML按1秒采样，再按`training_phases.jsonl`的UTC epoch与Asia/Shanghai时间对齐，可能漏短峰值；Torch数值是`update_actor`返回时读取的allocator高水位，没有该方法内按phase重置的证据，不能称为Actor单阶段峰值。全程采样峰值可发生在没有独立phase标记的验证段，不能用训练阶段采样最大代替全run最大。

本次仅支持将候选用于baseline的工程默认；没有加载8B Judge，naive/Actor共卡仍需各自验证。平均回答约800 tokens、仅1–2条触顶，并未覆盖256条都接近8192的最坏批次。完整原始参数、阶段秒数、tokens、GPU样本与退出证据见[formalshape_performance.json](formalshape_performance.json)和[baseline_throughput_comparison.json](baseline_throughput_comparison.json)。

## 比较时记录什么

1. 先拿到正式形状参照的真实Actor/Critic更新、loss/梯度有限、生成token数和峰值；若参照OOM，先处理执行参数，不能降低batch/n/长度伪装完成正式形状。
2. 对候选分别比较`generate_sequences/gen`、`old_log_prob/ref/reward`、`update_critic/update_actor`，并单列保存、验证、启动/图捕获/导出。以实际token量辅助比较，不仅比较总秒数。
3. 保持相同prompt UID与初始化；改变vLLM调度后即使seed相同，回答及长度也可能变化，因此记录分布，不能把更短回答全部当成提速。
4. 单步A/B只能作为工程筛选证据。选择候选应同时看执行正确性、长回答覆盖、峰值余量及实测改善，不宣称整轮固定倍数提升。
5. 本次是无Judge的baseline；有Judge时必须重新核对共卡峰值和竞争。新增8B的常驻及推理开销不能由baseline结果代替。
6. 临时run清理前保留配置、阶段报告、timing/token/显存记录及更新验证证据，保留作为两组共同来源的Warmup2；不把HF保护目录的额外占用漏算。
