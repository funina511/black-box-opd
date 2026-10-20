# LMOps

新实验使用 `math_unified_v1` 与显式 rollout seed。Rubric V2 的实现状态、
校准问题及正式配置入口见 [V2交接报告](reports/rubric_v2_handoff.md)。

Qwen3 的 SeqKD → Warmup → GAD 通用实验入口。新实验通过一份 [train.yaml](train.yaml) 选择模型、准备好的数据集和训练参数；启动时自动将完整配置保存到 `runs/<运行名>/train.yaml`。

```bash
cd /mnt/jinbo/RLRM/LMOps
# 先检查计划，再启动；两条命令使用同样的参数
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_17k --name n2 --dry-run
bash scripts/train.sh --model qwen3_0.6b --dataset dapo_math_17k --name n2
# 切换数据和模型，无需复制完整 YAML
bash scripts/train.sh --model qwen3_1.7b --dataset dapo_math_5k --name trial --dry-run
# 已有实验：按其保存配置恢复、评测、导出
bash scripts/train.sh --run qwen3_0.6b_17k --resume --dry-run
bash evaluate/run_eval.sh --run qwen3_0.6b_17k --dry-run
bash scripts/export_model.sh --run qwen3_0.6b_17k --stage gad --dry-run
```

| 位置 | 用途 |
| --- | --- |
| `train.yaml` | 全局训练默认值、模型/数据映射及历史 run 别名 |
| `scripts/train.sh` | 通用启动入口；顶部可设置模型、数据和覆盖参数 |
| `runs/<运行名>/train.yaml` | 自动保存的本次完整配置，恢复/评测/导出的依据 |
| `evaluate/run_eval.sh` | 设置目标 run、评测模型、数据集和采样参数 |
| `sample/configs/sample.yaml` | 共享教师数据的生产配置，入口为 `scripts/sample.sh` |
| `gad/configs/benchmark_candidates.yaml` | 性能候选设置，入口为 `scripts/benchmark_train.sh` |
| `data/`、`assets/` | 共享准备好数据、模型和原始数据链接 |
| `docs/` | [实验运行指南](docs/实验运行指南.md)及[本次改造计划](docs/全局训练配置与实验目录精简计划.md) |

初始全局默认采用 0.6B 探索设置：non-thinking、GAD n=2、batch=64、SeqKD/Warmup/GAD 为 2/1/2 epoch，GAD 的 epoch 是包含 Warmup 的绝对目标。每 50 step 及阶段末保存一次，每阶段轮换最近 5 次完整保存；交接、完成结果和已导出模型额外保留。

原 `experiments/` 已归档到各 run 的 `reports/legacy_experiment/`，旧 checkpoint 路径保留。`qwen3_0.6b_5k`、`qwen3_0.6b_17k`、`qwen3_1.7b` 是历史 run 别名；新实验使用自动生成的唯一目录名。全局配置进入 Git，运行配置随 `runs/` 一起备份。

## 原仓库说明

独立离线 rubric 实验入口：[rubric/README.md](rubric/README.md)；
在线 Warmup/GAD rubric 与历史 replay：[gad/ONLINE_RUBRIC.md](gad/ONLINE_RUBRIC.md)。
运行 bash scripts/rubric.sh --stage all --dry-run 查看固定轨迹、judge 与资源计划。

<!--# LMOps: Enabling AI w/ LLMs-->

# LMOps
LMOps is a research initiative on fundamental research and technology for building AI products w/ foundation models, especially on the general technology for enabling AI capabilities w/ LLMs and Generative AI models.

- Better Prompts: [Automatic Prompt Optimization](https://arxiv.org/abs/2305.03495), [Promptist](https://arxiv.org/abs/2212.09611), [Extensible prompts](https://arxiv.org/abs/2212.00616), [Universal prompt retrieval](https://arxiv.org/abs/2303.08518), [LLM Retriever](https://arxiv.org/abs/2307.07164), [In-Context Demonstration Selection](https://arxiv.org/abs/2305.14726)
- Longer Context: [Structured prompting](https://arxiv.org/abs/2212.06713), [Length-Extrapolatable Transformers](https://arxiv.org/abs/2212.10554)
- LLM Alignment: [Alignment via LLM feedback]()
- LLM Accelerator (Faster Inference): [Lossless Acceleration of LLMs](https://arxiv.org/abs/2304.04487)
- LLM Customization: [Adapt LLM to domains](https://arxiv.org/pdf/2309.09530.pdf)
- Fundamentals: [Understanding In-Context Learning](https://arxiv.org/abs/2212.10559)

## Links

- [microsoft/unilm](https://github.com/microsoft/unilm): Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities
- [microsoft/torchscale](https://github.com/microsoft/torchscale): Transformers at (any) Scale

## News
- [Paper Release] Nov, 2023: [In-Context Demonstration Selection with Cross Entropy Difference](https://arxiv.org/abs/2305.14726) (EMNLP 2023)
- [Paper Release] Oct, 2023: [Tuna: Instruction Tuning using Feedback from Large Language Models](https://arxiv.org/pdf/2310.13385.pdf) (EMNLP 2023)
- [Paper Release] Oct, 2023: [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) (EMNLP 2023)
- [Paper Release] Oct, 2023: [UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation](https://arxiv.org/abs/2303.08518) (EMNLP 2023)
- [Paper Release] July, 2023: [Learning to Retrieve In-Context Examples for Large Language Models](https://arxiv.org/abs/2307.07164)
- [Paper Release] April, 2023: [Inference with Reference: Lossless Acceleration of Large Language Models](https://arxiv.org/abs/2304.04487)
- [Paper Release] Dec, 2022: [Why Can GPT Learn In-Context? Language Models Secretly Perform Finetuning as Meta Optimizers](https://arxiv.org/abs/2212.10559)
- [Paper & Model & Demo Release] Dec, 2022: [Optimizing Prompts for Text-to-Image Generation](https://aka.ms/promptist)
- [Paper & Code Release] Dec, 2022: [Structured Prompting: Scaling In-Context Learning to 1,000 Examples](https://arxiv.org/abs/2212.06713)
- [Paper Release] Nov, 2022: [Extensible Prompts for Language Models](https://arxiv.org/abs/2212.00616)

## Prompt Intelligence

Advanced technologies facilitating prompting language models.

### Promptist: reinforcement learning for automatic prompt optimization

[Paper] [Optimizing Prompts for Text-to-Image Generation](https://arxiv.org/abs/2212.09611)

> - Language models serve as a prompt interface that optimizes user input into model-preferred prompts.

> - Learn a language model for automatic prompt optimization via reinforcement learning.

![image](https://user-images.githubusercontent.com/1070872/207856962-02f08d92-f2bf-441a-b1c3-efff1a4b6187.png)


### Structured Prompting: consume long-sequence prompts in an efficient way

[Paper] [Structured Prompting: Scaling In-Context Learning to 1,000 Examples](https://arxiv.org/abs/2212.06713)

- Example use cases:

> 1) Prepend (many) retrieved (long) documents as context in GPT.

> 2) Scale in-context learning to many demonstration examples.

![image](https://user-images.githubusercontent.com/1070872/207856629-2bb0c933-c27b-4177-9e10-e397622ae79b.png)


### X-Prompt: extensible prompts beyond NL for descriptive instructions

[Paper] [Extensible Prompts for Language Models](https://arxiv.org/abs/2212.00616)

> - Extensible interface allowing prompting LLMs beyond natural language for fine-grain specifications

> - Context-guided imaginary word learning for general usability

![Extensible Prompts for Language Models](https://user-images.githubusercontent.com/1070872/207856788-5409d04d-c406-4b29-ae7b-2732e727d4cc.png)


## LLMA: LLM Accelerators

### Accelerate LLM Inference with References

[Paper] [Inference with Reference: Lossless Acceleration of Large Language Models](https://arxiv.org/abs/2304.04487)

> - Outputs of LLMs often have significant overlaps with some references (e.g., retrieved documents).

> - LLMA losslessly accelerate the inference of LLMs by copying and verifying text spans from references into the LLM inputs.

> - Applicable to important LLM scenarios such as retrieval-augmented generation and multi-turn conversations.

> - Achieves 2~3 times speed-up without additional models.

![image](https://user-images.githubusercontent.com/6700539/231664563-aec35679-b4ab-4b6b-b6b4-b2b4ea1aab53.png)


## Fundamental Understanding of LLMs

### Understanding In-Context Learning

[Paper] [Why Can GPT Learn In-Context? Language Models Secretly Perform Finetuning as Meta Optimizers](https://arxiv.org/abs/2212.10559)

> - According to the demonstration examples, GPT produces meta gradients for In-Context Learning (ICL) through forward computation. ICL works by applying these meta gradients to the model through attention.

> - The meta optimization process of ICL shares a dual view with finetuning that explicitly updates the model parameters with back-propagated gradients.

> - We can translate optimization algorithms (such as SGD with Momentum) to their corresponding Transformer architectures.

![image](https://user-images.githubusercontent.com/1070872/208835096-54407f5f-d136-4747-9629-3219988df5d4.png)

## Hiring: [aka.ms/GeneralAI](https://aka.ms/GeneralAI)
We are hiring at all levels (including FTE researchers and interns)! If you are interested in working with us on Foundation Models (aka large-scale pre-trained models) and AGI, NLP, MT, Speech, Document AI and Multimodal AI, please send your resume to <a href="mailto:fuwei@microsoft.com" class="x-hidden-focus">fuwei@microsoft.com</a>.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

### Contact Information

For help or issues using the pre-trained models, please submit a GitHub issue.
For other communications, please contact [Furu Wei](http://gitnlp.org/) (`fuwei@microsoft.com`).
