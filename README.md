# LMOps 八卡 H200 数学实验

本分支包含 Qwen3-0.6B 的 SeqKD、Base-GAD、正确性消融、RLVR–GRPO、Naive rubric 和 Actor-rubric。八卡专用入口的参数由以下命令列出。

```bash
python scripts/h200_8gpu.py --help
```

迁移指南和数据压缩包单独分发。先在仓库的父目录解包 Teacher 和评测数据，再安装 environments/requirements.lock.txt 指定的训练栈、下载模型并重建共同 Critic。运行目录与灰测参数由新服务器的实际资源确定。八任务评测入口为 evaluate/run_all_eval.sh，读取已有成绩使用 evaluate/summarize_runs.py。
