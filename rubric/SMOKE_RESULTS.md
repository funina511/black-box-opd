# 第一阶段离线冒烟记录

日期：2026-10-11。工程闭环完成；标签质量门槛未通过。

## 实际设置

- 固定 0.6B/17k Warmup global_step_271 的配对 actor/critic；
  继承 non-thinking、temperature=0.8、top_p=1、2048/8192。
- 24 题，train/dev/test = 16/4/4；每题已有 teacher 1 条、student 2 条。
  72 条唯一回答、48 个原始判别对。每个训练组 1 epoch，
  逻辑 batch 16 题、microbatch 1 回答；lr=1e-6、lambda=0.1。
- judge：Qwen3-30B-A3B-Instruct-2507，GPU 1,2,3,4、TP=4。
  critic 使用 GPU 1；FP32 主参数/AdamW 状态、BF16 autocast。
- run：math_v1_24q_seed42_392af33fea81。
  首次采样记录在 math_v1_24q_seed42_68a0703899db；
  来源、生成参数、题目与划分一致后复用到一 epoch run，没有重复采样。
- 初期 judge 请求按单条执行，保留已完成缓存后切换为批量请求。
  阶段代码快照、实际 batch 信息、原始输出和重试均留存。

## 工程验收

- 17 项 CPU 合同测试通过：哈希、unknown/mask、匿名 judge、
  原始 token/EOS/末响应位置、题目隔离、辅助去重与整批梯度、
  评分头严格迁移/重载、完整训练状态恢复、批量缓存与 checkpoint 轮换。
- 三个训练组最长输入均为 8311 token，扩头前后 adv 分数差为 0。
  原 weight/bias 第 0 行精确复制，不使用忽略形状差异的加载。
- B1/B2/B3 各完成一个 optimizer step，耗时分别约 78.4/78.8/78.8 秒；
  峰值 allocated 显存均约 11.17 GiB。
- 在额外新进程恢复 B3 的 312 个参数 optimizer 状态、scheduler 和
  CPU/CUDA RNG；最长题组完成反向及一次内存中的更新，参数实际变化
  约 1e-6，无 NaN/Inf。峰值约 11.71 GiB。
  该工程验证不改写实验 checkpoint 或 test 结果。
- 所有模型在独立评价进程中严格重载；各阶段完成后释放 GPU。
- student 使用实际生成 token，而非重新编码后的近似序列。
  一条原始 8192 token 的回答重编码为 8197，修复后无需裁短；
  最终 24 个完整题组均保留。

恢复证据：[engineering_verify.json](../runs/rubric/math_v1_24q_seed42_392af33fea81/engineering_verify.json)。

## 标签审计

- 原数学评分器：teacher 正确 18/24，student 正确 3/48。
- teacher 对/student 错 35 对，teacher 错/student 对 2 对，
  双方正确 1 对，双方错误 10 对；均保留原 teacher > student 标签。
- teacher 1 条、student 36 条截断，共 37/72。
  截断回答过程维度使用 unknown/mask。
- 35 条完整回答中，32 条最终得到合法 judge JSON：
  **91.4% < 98% 门槛**。按全部实际输出尝试计算有效率为 71.4%。
  解析错误和重试原文保留，失败不补零。
- 有效标签：正确性 68 条、推导有效性 32 条、条件完整性 29 条。
  推导有效性只有 1 和 0.5，尚无 0 标签，需要人工检查分布。
- 重复评审 4 条；两个过程维度各有 3 条双方可判断，3/3 标签相同。
  样本太少，不能据此证明 judge 可靠。
- 人工复核 0 条，审计状态保持 pending；本轮 pack/train/report
  明确标为 exploratory。

审阅材料：[review_requests.jsonl](../data/rubric_math_v1/math_v1_24q_seed42_392af33fea81/audit/review_requests.jsonl)；
统计：[audit/summary.json](../data/rubric_math_v1/math_v1_24q_seed42_392af33fea81/audit/summary.json)。

## 四组结果

以下只有 4 道 test 题，不能作为效果结论。

| 组 | 原判别 pair accuracy | 数学正确性排序 | 固定 student Best-of-2 | Oracle |
| --- | --- | --- | --- | --- |
| B0 原 critic | 100% | 66.7% | 25% | 25% |
| B1 原损失 | 100% | 66.7% | 25% | 25% |
| B2 正确性辅助 | 100% | 66.7% | 25% | 25% |
| B3 三维辅助 | 100% | 66.7% | 25% | 25% |

各组在这些离散指标上相同，B3 的额外收益尚无证据。
配对 bootstrap 差值区间为零来自四题的排序/选择完全一致，
不表示真实效应已精确估计为零。辅助逐维误差、分层、分母、
bootstrap 区间及冲突样例见
[完整报告](../runs/rubric/math_v1_24q_seed42_392af33fea81/report.json)。

## 下一步

先审阅标签，检查 JSON 格式约束/转义和标签分布；
judge 提示词或解码方式变化须使用新版本/哈希和新标签缓存。
不要直接把这一套未通过审计的设置扩到 512 题。
同时需要处理 student 大量截断所造成的过程监督缺失；
更换来源 actor 或采样设置属于新实验，不能混入当前对照。

源 Warmup 已见过 DAPO train，test 只表示未参加本次辅助训练。
本轮没有 actor 更新，不能从离线 critic 指标推断 actor 准确率提高。
