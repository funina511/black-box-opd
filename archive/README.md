# 本机历史归档

这些目录保存历史证据，不属于正式运行或迁移包。归档内容默认不进入 Git；本文件说明去向。

2026-11-12 已整理：

| 原路径 | 归档路径 | 正式替代入口 |
|---|---|---|
| `eval_fix/` | `archive/eval_fix/` | `evaluate/`、`scoring/math.py` |
| `gad/scripts/train/` | `archive/gad_upstream/scripts/train/` | `scripts/train.sh` |
| `gad/scripts/generate/` | `archive/gad_upstream/scripts/generate/` | `scripts/sample.sh`、`scripts/eval.sh` |

移动前后逐文件 SHA256 已核对，记录在本机 `cleanup_20261112.json`。没有重新应用旧 integration.patch，也没有改动正式数学评分器。旧脚本按历史原文保留，不承诺移动后可直接执行；需要复现时在独立副本恢复原目录关系。

`judge_fix/` 暂未移动：旧 rubric Warmup 的冻结配置仍引用它。它已排除 Git 发布，待不再需要旧路径恢复时再归档。正在使用的 `gad/verl/`、生成 backend、模型及 runs 保持原位。

`rubric/` 根目录的离线工具和相关测试暂保留；`rubric/core/`、`gad/extensions/` 和 replay 是正式实现，不属于可删除的旧副本。
