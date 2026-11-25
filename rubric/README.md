# Rubric 辅助 Critic：独立离线实验

统一入口为 LMOps/scripts/rubric.sh。配置在 configs/offline_v1.yaml；
相对路径按 YAML 所在目录解析，脚本可以从任意工作目录启动。

当前默认：24 题，train/dev/test = 16/4/4；复用 0.6B/17k 的
Warmup step 271 teacher 数据和配对 actor/critic，student 每题采样 2 条。
继承该 run 的 non-thinking、temperature=0.8、top_p=1、2048/8192 长度。
judge 为本地 Qwen3-30B-A3B-Instruct-2507，GPU 1,2,3,4、TP=4；
采样和训练单独使用 GPU 1，各阶段串行启动独立进程。

本轮 24 题已经跑完：[冒烟记录](SMOKE_RESULTS.md)。
工程门槛通过；judge JSON 有效率 91.4%，人工审阅 pending，
暂不直接扩大至 512 题。

## 启动

~~~bash
cd /mnt/jinbo/RLRM/LMOps
bash scripts/rubric.sh --stage all --dry-run
bash scripts/rubric.sh --stage all
~~~

默认 all 完成采样、标注和审计材料后，在人工审核尚未满足时停止。
冒烟闭环可显式使用下面的探索性模式；报告保留“人工未验证”状态。

~~~bash
bash scripts/rubric.sh --stage all --exploratory
~~~

每个 run 保存 config.resolved.yaml。通过该快照继续已有实验：

~~~bash
bash scripts/rubric.sh --config runs/rubric/<run_id>/config.resolved.yaml --stage report
~~~

默认自动生成 run_id。改变参数会生成独立目录，不覆盖原实验。
默认冒烟只训练一个 epoch。下一档 512 题（320/96/96）可以选 1–3 epoch：

~~~bash
bash scripts/rubric.sh --size pilot --set training.max_epochs=3 --stage all --dry-run
bash scripts/rubric.sh --size pilot --set training.max_epochs=3 --stage all
~~~

## 分阶段与断点

stage 支持 preflight、prepare（仅 CPU 准备题目）、collect、annotate、
audit、pack、train、evaluate、report、all。
collect/annotate 根据哈希缓存已完成结果。pack 后标签和划分锁定。

~~~bash
bash scripts/rubric.sh --stage train --variant adv_only --exploratory
bash scripts/rubric.sh --stage train --variant correctness_aux --exploratory
bash scripts/rubric.sh --stage train --variant joint --exploratory
bash scripts/rubric.sh --stage evaluate --variant baseline --split dev
bash scripts/rubric.sh --stage evaluate --variant joint --split test
~~~

B0=baseline 不训练；B1=adv_only；B2=correctness_aux；B3=joint。
所有训练组以同一源 critic、扩头 seed、轨迹、shuffle、逻辑 batch 开始。
B1 的未训练辅助头不作为有意义的评价输出。

~~~bash
# 本次最多执行一个 optimizer step，保存完整恢复状态：
bash scripts/rubric.sh --stage train --variant joint --max-steps 1 --exploratory
# 单独的新进程恢复 optimizer、scheduler、RNG、epoch、batch offset：
bash scripts/rubric.sh --stage train --variant joint --resume auto --exploratory
~~~

离线训练保留最近五次完整保存，dev 选中的最佳 checkpoint 额外保护。
已完成的组会直接复用。all --resume auto 能继续已存在的未完成组，
尚未开始的组会从固定源 critic 开始。
训练使用 FP32 主参数/AdamW 状态、BF16 autocast、gradient checkpointing，
完整回答不被裁短。仅支持验证过的单卡普通 padding 路径，未接通 FSDP 或
remove-padding。资源不足会失败并保留缓存，不能静默改模型、长度或 loss。

dev 按预先固定的 adv loss + lambda × 有效辅助 MAE 均值选择 checkpoint；
B1 lambda=0。test 只在该组训练结束、dev 选择完成后执行。
同一模型/数据的评价重复命令读取已有结果；改变已锁定结果会报错。
先看 dev，再决定新 run 的设置，不能根据 test 改当前实验。

## 标签审阅

数据目录为 data/rubric_math_v1/<run_id>/。
audit/review_requests.jsonl 为分层抽查材料：冒烟集检查全部唯一回答，
512 题集合至少 100 条。把填写后的独立人工记录保存到
audit/reviews.jsonl，再执行 audit；原请求文件保持不变。

每条记录须有 response_id、cache_key、reviewer、accepted（布尔值）、
severe_reverse（布尔值）、reason 和 corrections。
corrections 的键为维度名，值为 value/valid 字段；如未知则
value=null、valid=false。接受原标签时 corrections={}。
拒绝标签但未给替代值时，两个过程维度会 mask。
正确性修正只影响辅助标签，原数学评分器 acc 始终单独保留。

~~~bash
bash scripts/rubric.sh --stage audit
bash scripts/rubric.sh --stage pack
~~~

人工通过状态、JSON 有效率、重复判断一致性、分布与缺失率在
audit/summary.json。只做自动一致性检查不能获得人工通过状态。
pack 后修改标签须建立新数据版本，不能覆盖已训练输入。

导入外部标签可设置 judge.backend=import 和 judge.annotations_path；
必须符合 schema.py 的完整注释格式、版本和缓存哈希。API 方式须显式
设置 URL、model、api_version，并通过配置指定的环境变量传递密钥。
本地模式不调用远端服务。judge 不接收角色、模型名或 teacher 参考解法。
上下文超限、输出无效、截断回答的过程标签均保留失败原因并 mask，
不以零分替代 unknown。

## 产物与解释

- data/rubric_math_v1/<run_id>/：固定题目、划分、完整轨迹、原始与审阅标签、
  原始 judge 输出、审计、配对与 train/dev/test parquet。
- runs/rubric/<run_id>/：配置、指纹、阶段日志、派生单头导出、
  variants 下四个组、HF 多头模型、训练恢复状态、评价与 report.md/json。
- 评分头顺序：[adv, answer_correctness, reasoning_validity, condition_coverage]。
  原 adv weight/bias 精确复制；训练前检查最长样本的分数迁移。
- 原 teacher > student 损失保留，teacher 辅助监督只按唯一回答计一次。
  缺失标签的分母按整个 optimizer step 计算，分布式训练未启用。
- 标签、judge、提示词、GT、响应、题目、源 checkpoint 和数据都有哈希校验。
  正式 dev、quick_dev 及已有外部数学测试集按规范化题目排除。

report 区分来源偏好、数学排序、辅助拟合、相同正确性下的过程排序、
固定 student Best-of-2、首条/随机/oracle，并给出按题目 bootstrap 区间。
24 题是工程验证，不能证明效果。源 Warmup 已见过 DAPO train；
test 仅表示未参加本次辅助训练。该离线实验没有更新 actor。
多头 checkpoint 是独立 TokenClassification 模型，不能直接交给原 GAD
的单头加载路径；本版本不修改现有 GAD 后端。

## CPU 验证

~~~bash
/mnt/jinbo/miniconda3/envs/gad/bin/python -m unittest discover -s rubric/tests -v
~~~
