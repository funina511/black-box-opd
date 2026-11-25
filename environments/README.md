# 环境准备

H200 新机的实际路径、安装顺序、单卡配置和 checkpoint 保留策略见 [H200_SETUP.md](H200_SETUP.md)。目标 gad 环境已安装并通过依赖、CUDA BF16 和 FlashAttention 前后向检查；baseline GPU 短测的实际验收记录保存在 `reports/h200_deployment/`。

训练与Judge共用 gad / Python3.10 环境；版本见 environment.json 和 requirements.lock.txt。
1. 准备匹配GPU驱动、CUDA12.4及Python3.10，不升级训练栈。
2. 按记录安装torch/CUDA轮子，再安装vLLM及匹配FlashAttention、xgrammar构建，最后安装其余Python依赖。
3. pip-freeze.txt中的editable/file://原路径仅作来源审计，不能直接当安装命令；本项目backend从内置bundle重建，通过子进程PYTHONPATH选择，不额外pip install其它verl。
4. FlashAttention/vLLM实际安装版本和dist-info direct_url见environment.json；无法从原环境证明的wheel下载地址标为未知，应按记录ABI在目标机重新构建或携带原wheel。锁定版本不是离线wheelhouse。
5. export GAD_PYTHON=/your/env/bin/python 后运行预检。本机结果只证明记录中的环境；另一台服务器仍需重新验证驱动、CUDA 和真实 GPU 短测。
