# 核查问题确认与修复结果

2026-10-15。依据`docs/核查问题修复计划.md`完成核查修复。**未恢复judge实验，模型请求0次、训练0步。**
历史模型、训练配置、回答和旧评分均未覆盖。

## 1. 逐项处理

| 条目 | 实际确认与处理 |
| --- | --- |
| A1 | 确认Markdown列表/引用/序号答案变成可信负标签；支持这些前缀及嵌套引用，并通过OnlineRubric实际targets/masks验证正确答案为1/True |
| A2 | 确认空白粘连数字；禁止数字组/小数跨空白合并，保留单位及2 x/2x合法等价 |
| A3 | 确认加粗值残留单侧星号；统一清理标签/值边缘Markdown，pred正确恢复 |
| A4 | 确认末尾占位符可制造负标签；模板不覆盖前面实际答案，普通散文/畸形末答案保持invalid。文档第二个内联Answer示例原代码实际已正确，未复现所称抛错；修复后两个示例均正确 |
| A5 | 确认坏box遮蔽后续box；只有完整box覆盖内部候选，多最终box仍保持ambiguous |
| A6 | 确认invalid评估计0而训练mask；evaluation_value/compute_score传播None和evaluation_valid，不完整评估不发布正式准确率，不通过丢弃unknown抬高分数 |
| A7 | 单位/角度/iπ探针确认；归一化单位结构，保留指数分组及π邻接乘法，记录vendor本地改动；未引入近似容差 |
| B1 | 确认错误片段能冒充其自身修复；改为非空物理行片段，修复与错误必须是不同片段，真实多行替代/代入检验回归通过。同一行混写无法结构确证时保守mask |
| B2 | 确认有效unjudgeable仍可获满分/无错；未知数学阻断progress=1/error=0，普通neutral不阻断，保留局部独立进展 |
| B3 | 确认721可被虚假reason冒充77；属于语义审核边界。spec及返回validation_scope明确semantic_entailment=not_checked，不用正则冒充数学判断，没有追加第二个judge |
| B4 | 确认合同拒绝被当technical、mask重复及原因覆盖；分离contract_rejected统计/停止原因，保留首个因果原因，细分未知数学/未知错误/未完成。critic头数由layout派生；4个事件槽位是独立预算，不是头数 |
| B5 | 确认空修复链接可生成；采用完整anyOf分支，修复使用必填error_event_id枚举e1–e4。没有引入vLLM0.8.5不支持的minItems，实际xgrammar CPU编译通过 |
| C1 | 确认默认统一scorer却跑CARD断言；self-test按实际版本选择预期和说明，两版本均通过 |
| C2 | 确认legacy配置被送给统一scorer；显式传scorer路径，仅统一版本传统一配置，生成前检查版本/指纹，显式路径不存在直接失败 |
| C3 | 确认禁用验证仍要求产物且legacy开关无效；不再要求产物，同时关闭val_before_train/test_freq，两种profile均生效 |
| C4 | 确认不完整评估能通过验收；拒绝evaluation_complete=false、grader_error_rate或unscorable_rate非零，并保留诊断字段 |
| C5 | 确认legacy异常成为完整0分；异常标记grader_error/unscorable，最终acc为None，临时指标不能使阶段验收通过 |
| C6–C8 | 确认整数串、阶段n覆盖和算术类型问题；加载时正规化整数串，拒绝bool/浮点/非数字，算术前检查类型；stage-local n生效，显式stage专用键优先 |
| C9四项 | 均确认修复：完整验证计数乘val_n；扩展未设seed使用上游默认0；导出dry-run按epoch解析目标，日志使用实际交接差值；parse_failure与no_answer分开 |
| D1 | 确认套件漏测；发现全部CPU测试，生成隔离backend，训练屏障不再因backend缺失跳过；明确环境依赖门和可写Outlines临时缓存，本机最终无skip |
| D2 | 文档“8目录均无引用”不成立，3个仍被历史run引用而保留；另5个未使用且无引用，先完整压缩备份，再注销/清理 |
| D3 | 关键源码确实未跟踪；评分器、协议、配置及必要框架/测试已纳入Git暂存供审阅，未提交/推送，大规模原始输出不批量加入Git |
| D4 | 确认cap之后比较恒False；在teacher tokenizer之后、postprocess_data之前记录，并执行生成backend实际代码检查 |

原计划明确保留的BT、辅助loss归一化、DP缩放、actor目标及去重合同未调整。

## 2. 协议与兼容性

活动协议为 **input `rubric.input.v2.3` / wire `rubric.events.v2.4`**。
普通事件保留span_id/reason/related_event_ids；修复事件改用必填error_event_id，且错误/修复片段不同。
不同片段只排除最直接的同证据伪修复，**不证明数学修复确实发生**。

现有8B覆盖仍名为`configs/overrides/rubric_v23_8b_thinking.yaml`，只包含模型/服务参数；
新run实际协议由当前spec/schema/parser身份决定，文件名不表示复用旧标签。
scorer fingerprint由`e3aaaac272846f564ce6ca6af969b395b1382ec7740a3884d7c689591ab3cb1c`
变为`2a2b8a02925ecd3b6a5ff99f32a7cd04df03c8b962278c42a1099eb4380467b7`。
旧缓存与旧unified run不能静默使用新监督源，没有批量改写旧配置指纹。

无最终答案仍按已有定义计正确性0；malformed/ambiguous/grader异常等为invalid。
训练和评估共用是否可计分的判定，invalid只进入明确的临时诊断，不发布完整准确率。

## 3. 同一冻结评估集的位移

冻结已有baseline GAD的4组共1800条回答，修复前保存全部评分，修复后重评分。
原始4个文件SHA256均不变，没有重新生成或覆盖旧scores。

| 组别 | 回答数 | 修复前已知正确 | 修复后已知正确 | 固定分母位移 |
| --- | ---: | ---: | ---: | ---: |
| AIME24 | 240 | 8 | 8 | 0 pp |
| AIME25 | 240 | 4 | 4 | 0 pp |
| AMC23 | 320 | 65 | 67 | +0.625 pp |
| MATH-500 | 1000 | 416 | 418 | +0.2 pp |
| 合计 | 1800 | 493 | 497 | **+0.2222 pp** |

79条抽取/状态/标签记录变化，48条训练标签(value/valid)变化。
可计分1712→1734；grader error 82→27；修复后另有39条malformed，共66条invalid。
上表是固定1800分母的已知正确下界，**不是完整正式准确率**。
历史其它run数字不能直接与新口径比较；本审计不替代其它回答重评分或生成完整性检查。
见[score_shift.json](score_shift.json)、[逐条变化](score_changes.jsonl)及本目录before/after_scored.jsonl。

## 4. 验证

```bash
bash /mnt/jinbo/RLRM/LMOps/gad/tests/run_all.sh /mnt/jinbo/miniconda3/envs/gad/bin/python
```

**26套件、290项CPU测试通过，0跳过，末行ALL CPU SUITES PASSED。**
其中核查回归19项；训练hook4项执行当前生成backend的更新顺序、judge/actor失败屏障、teacher截断及seed默认值。
首次全套有2项旧口径断言失败（no_answer不再归parse_failure、grader异常不再是可信0分），
同步更新意图后完整重跑，没有删除测试。见[最终日志](all_cpu_tests.log)。

9份当前run配置中，8份legacy通过校验与相应阶段resume dry-run；
online_correctness_smoke_v2需用已有约定的`--stages gad`，默认全阶段命令被正确拒绝。
旧unified run baseline_math_v2_06b5k因指纹变化继续拒绝恢复，这项保护保留。
新8B joint配置校验通过，没有启动训练。
见[配置审计](run_config_audit.json)、[resume审计](resume_dry_run_audit.json)及resume_dry_runs/。
stage-local n修复后按声明生效，历史实际被忽略的n不能冒充新n条件下的对照。

## 5. 归档与未覆盖

5个未使用backend先完整压缩，备份约10.7MB，见retired_backends/；
备份hash/commit及清理记录见[backend_cleanup.json](backend_cleanup.json)，3个历史引用目录保留。
CPU验证使用临时独立worktree，结束后清理，没有改写历史backend。

修复前后源码与hash见[source_snapshots.tar.gz](source_snapshots.tar.gz)，
机器验收见[verification.json](verification.json)。关键源码已暂存供用户审阅，未创建提交或推送。

**未覆盖：** 新协议真实8B生成、真实联合Warmup→GAD→恢复、actor收益与人工标签准确率。
此前停止的实验保持停止；B3数学语义问题仍需后续处理。
