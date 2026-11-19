# LMOps：数学推理 GAD / Rubric 实验

项目包含 Teacher 数据采样、SeqKD、Warmup、GAD 训练和数学评测。当前在线 rubric 使用独立 Judge 标注，Critic 学习 BT + rubric 辅助损失；Actor 可选择纯对抗奖励或 rubric 混合奖励。

## 从哪里开始

| 工作 | 正式入口 |
|---|---|
| 训练默认值、模型和数据选择 | [train.yaml](train.yaml) |
| 训练、续训 | [scripts/train.sh](scripts/train.sh) |
| 训练前检查 | [scripts/preflight.sh](scripts/preflight.sh) |
| Teacher 数据采样 | [scripts/sample.sh](scripts/sample.sh)、[sample/README.md](sample/README.md) |
| 评测与重评分 | [scripts/eval.sh](scripts/eval.sh)、[evaluate/README.md](evaluate/README.md) |
| 数学正确性评分 | [scoring/math.py](scoring/math.py) |
| Judge 启停 | [scripts/rubric_judge.sh](scripts/rubric_judge.sh) |
| 四头 Critic、奖励与 replay | [gad/extensions/](gad/extensions/)、[rubric/core/](rubric/core/) |
| 环境版本、固定后端来源 | [environments/README.md](environments/README.md)、[third_party/verl/](third_party/verl/) |

先激活目标 Python 环境，再在项目根目录运行：

```bash
cd /你的项目路径/bloxopd/LMOps
export GAD_PYTHON="$(command -v python)"
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_17k_v2 --name trial --dry-run
```

这是检查命令，不是 H200 已验证的训练配置。新机还需要下载模型、补齐数据、重建 Critic 初始化并设置机器资源。

## 当前 n=4 实验

- [baseline 配置](configs/launch_baseline_n4_06b_17k.yaml)
- [naive-rubric 配置](configs/launch_rubric_naive_n4_06b_17k.yaml)
- [actor-rubric 配置](configs/launch_rubric_actor_n4_06b_17k.yaml)
- [整体实现与实验记录](docs/20261105/GAD与Rubric整体实现及n4实验记录_20261112.md)
- [三组对照方案](docs/20261105/GAD_n4三组对照实施方案.md)

这三份配置仍描述源机从旧 Warmup271 开始的实验，保留用于历史复现。

H200 新实验使用 [baseline](configs/h200_baseline_n4.yaml)、[naive](configs/h200_naive_n4.yaml)、[Actor](configs/h200_actor_n4.yaml)。0.6B baseline 已通过环境安装、训练/恢复、导出外评和正式形状一步对照，采用 rollout 0.40/并发128/Critic microbatch4；正式输出已设为用户确认的 `/tmp/bloxopd_h200_runs`，短测权重已获授权清理，正式训练尚未启动。rubric 共卡尚未实测，Actor 正式参数仍需新 naive 选参确认。安装顺序、实测边界和保存策略见 [H200 适配说明](environments/H200_SETUP.md)。

## 配置路径与迁移

训练 YAML 的模型、数据、scorer、prompt/spec、输出及显式 Warmup 来源路径，均以**该 YAML 所在目录**解析；命令行路径按调用目录解析。解释器由 `GAD_PYTHON` 或当前环境选择，不写死旧服务器的 Conda 路径。

已有 run 内的冻结配置保留绝对路径，这是记录实际执行资产的依据；迁移新实验不批量改写历史配置。 新实验launch的scorer指纹自动取当前环境，历史run仍保留原值。迁移只需确认Git版本并校验两个中转包，不要求逐文件源码或模型哈希；详见 [r2精简记录](docs/20261105/迁移路径与校验精简记录_20261114.md)。详情见 [目录整理与路径说明](docs/20261105/目录整理与相对路径修正记录_20261112.md) 和 [H200迁移计划](docs/20261105/GAD_n4单卡H200迁移计划.md)。

## 历史与可选工具

已合并的 `eval_fix` 和旧 GPT5/7B shell 脚本已移入本机 [archive](archive/README.md)。正式评测无需再应用旧补丁。旧离线 rubric 预实验仍可从 [rubric/README.md](rubric/README.md) 查阅，区别于在线训练入口。

`judge_fix` 尚因旧实验路径引用保留，但不进入 Git 发布或新运行包。上游 LMOps 项目介绍单独保存在 [历史说明](docs/history/LMOps上游项目说明.md)。
