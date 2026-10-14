# 在线 Rubric / Replay 工程验证记录

日期：2026-10-12。完整 actor 效果实验由用户执行，本次未运行数学基准评测。
启动说明见 [ONLINE_RUBRIC.md](ONLINE_RUBRIC.md)。

## 1. 实际环境与范围

- 学生与 critic backbone：本地 Qwen3-0.6B；训练 GPU 1,2,3,4，四卡 FSDP。
- 在线 judge：本地 Qwen3-8B，BF16、TP=2、GPU 6,7，non-thinking，guided JSON / xgrammar。
- Python 环境：`/mnt/jinbo/miniconda3/envs/gad/bin/python`。
- 工程训练使用独立16题、batch8、512/512长度，Warmup n=1，GAD n=2。
  为缩短工程运行，数据选自有完整短 teacher 的题目；覆盖率不能推广到完整17k。
- 最长响应另用真实0.6B、四卡、2048 prompt＋8192 response验证反向；标签为合成工程标签。
- 所有训练和 judge 进程已结束，检查时8张GPU均已释放。

## 2. Warmup → GAD 与两阶段恢复

结果目录：`runs/online_wg_smoke_v1/`。

| 阶段/绝对步 | actor optimizer 次数 | critic optimizer 次数 | 历史比较数 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Warmup 1 | 0 | 1 | 0 | 门槛2之前仅 critic 联合更新 |
| Warmup 2 | 1 | 2 | 0 | 新进程恢复，执行原 teacher SFT |
| Warmup 3 | 2 | 3 | 0 | 再次恢复，完成 Warmup |
| GAD 4 | 1 | 1 | 0 | 四头交接，按原约定新建 optimizer |
| GAD 5 | 2 | 2 | 4/16 | 新进程恢复，25%历史比较，原 actor 策略更新 |

actor 与 critic 的 model、optimizer、scheduler、worker RNG、dataloader均有实际恢复日志。
缓存恢复后，Warmup第3步命中8个teacher与2个student回答；GAD第5步恢复16对历史池，
更新后扩至32对。历史只进入独立 critic batch；actor batch保留当前 rollout。

四个评分头从Warmup第1步起都有实际FP32参数更新，首步每行最大变化约`1.00e-6`；
GAD第4→5步每行也变化约`1.00e-6`。
Warmup末的四头权重完整导出为BF16；固定输入下，源模型、HF导出和四卡FSDP加载后的
四通道分数完全一致。adv reward仍只在末有效response token返回。

证据：

- `runs/online_wg_smoke_v1/{logs,stage_warmup.json,stage_gad.json,handoff_manifest.json}`。
- 各阶段 `extension_metrics.jsonl`、`global_step_*/checkpoint_complete.json`。
- `runs/_online_contracts/{optimizer_steps,head_updates,model_handoff}.json`。

## 3. 独立分支与恢复负例

- `runs/online_replay_smoke_v2/`：judge已停止时完成单头Warmup2步＋replay-only GAD2步。
  GAD比较数始终16，第二步4对历史，pool由16增至32；没有rubric缓存或judge请求。
- `runs/online_correctness_smoke_v2/`：judge已停止时，从上述四头Warmup3只读派生，
  完成correctness-only GAD2步；replay关闭、judge请求0、两个过程维度mask/count/loss均为0。
  此派生仅验证分支，不作为正式correctness-only主实验。
- 真实checkpoint临时镜像分别缺失`data.pt`、critic config、actor tokenizer、replay状态、
  worker extra_state时，都拒绝该保存点并回退至上一完整点；源文件保持原样。
- 新进程恢复真实replay文件后，32对池及专属RNG恢复，下一次历史选择与参考完全一致。
- 执行实际已部署Warmup/GAD更新块的CPU测试确认：标注失败发生于两次update之前；
  actor失败不提交replay；门槛前正常跳过actor仍提交成功的critic步。

证据：`runs/_online_contracts/restore_negative.json`及两个独立run的配置、日志、阶段manifest。

## 4. 分布式、长响应与关闭回归

- 四卡FSDP：raw padding/remove-padding × microbatch1/2，含跨rank重复teacher、
  两个rank无辅助监督；与全局去重参考的梯度最大差`4.58e-5`。
- rank0真实加载、其余rank构建四头meta模型，再广播加载；四通道分数一致。
- 在真实0.6B模型上，原backend与扩展关闭路径的成对损失及梯度最大差均为0。
- 真实最长输入10240 tokens完成联合loss、反向和`lr=1e-6`更新，
  参数最大变化`1.91e-6`、grad norm约94.01，纯critic进程峰值显存7.79GiB。
  该数值不包含actor、reference和rollout服务，不代表正式整套训练显存。

证据：`runs/_online_contracts/{gpu_contract,model_handoff}.json`。
复跑GPU契约使用`gad/tests/gpu_online_contract.py`及`gpu_model_handoff.py`，须显式四卡torchrun。

## 5. CPU验证与修复

最终`gad/tests/test_*.py`：**153项通过，146.5秒**。
共享core提取后原`rubric/tests`的17项也已通过。
日志保存至`runs/_online_contracts/gad_cpu_tests.log`。

本轮必要验证修复了：

1. 完整checkpoint选择补查dataloader、模型/tokenizer元数据；新保存点记录这些文件的hash，
   缺失时不发布complete marker，内容变化也拒绝恢复。
2. judge整步deadline包含健康检查，避免重复健康检查在整步预算之外消耗时间。
3. CLI支持`--set 'rubric.stages=[gad]'`等YAML/JSON列表；原布尔、数值、GPU字符串与路径语义保留。
4. 新实验自动按代码/补丁指纹创建独立backend目录；已使用的backend不热修改。

联合冒烟v1代码hash为`7a4994dbe34fa67772a82ee56f9210dfef3420412ca14258deea087ca5170125`。
恢复加固后v2代码hash为`6a7b078aaf0d419cf87872d5390cc569b4e005d13dda5fd1b1f4d9d8738f8f77`，
backend为`gad/.backends_gad_extensions_v1_bd141ca6ed74/`。
v1已完成，仅保留为历史工程证据；源码升级后不能把它当作当前版本同阶段续训。
judge身份指纹为`9576a3d01476f76f76c7c81529a61120ac6a2ec57afe9c491cc85ece1c5e1306`。
其余配置/数据/模型指纹见`runs/_online_contracts/verification_summary.json`与各run manifest。

## 6. 标签问题与后续效果实验

联合冒烟最终请求中有效JSON为**57/58（98.3%）**，重试次数另计，不计入该分母。
唯一最终格式错误是unknown标签未给null，已记录原文并mask，没有伪造0分或永久缓存失败。

| 阶段/步 | 最终JSON有效/请求 | student reasoning有效/唯一回答 | student condition有效/唯一回答 |
| --- | ---: | ---: | ---: |
| Warmup 1 | 16/16 | 8/8 | 3/8 |
| Warmup 2 | 15/15 | 7/8 | 3/8 |
| Warmup 3 | 6/6 | 8/8 | 1/8 |
| GAD 4 | 12/13 | 14/16 | 1/16 |
| GAD 5 | 8/8 | 16/16 | 5/16 |

condition coverage仍偏低，可能限制辅助监督量；格式有效不等于语义标注正确。
每阶段32条校准记录保留待人工审阅，未宣称通过人工标签质量验收。
`rubric/judge_seconds`当前统计整个标签准备屏障，含规则评分与缓存处理；
无judge模式也可能非零，应结合judge请求数和token数解释成本。

Ray退出时存在DataLoader worker被终止的清理消息；各运行均成功保存、exit0且释放GPU。

**这些结果确认训练与恢复工程闭环，尚不能说明actor数学性能提高。**
完整A/B/C应各自从同一adv初始化重新训练Warmup、匹配GAD预算，
用相同non-thinking评测设置比较base/Warmup末/GAD末；命令见启动说明。
