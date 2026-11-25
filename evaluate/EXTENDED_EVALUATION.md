# 扩展评测说明

2026-10-24：数据和 CPU 接入验证已完成，尚未启动模型生成或 GPU 评估。同日修订：

- 提示词改为与训练一致的 DAPO 模板，选择题另加一句只输出选项字母的要求（数据版本 `extended_dapo_v2`）。
- 选择题评分升级为 `choice_exact_v2`。
- Minerva 增加评估侧单位规则。
- 按任务设置采样次数。

本轮加入四项任务，原有 AIME24、AIME25、AMC23、MATH-500 保留。所有文件位于 `LMOps/assets/datasets/math_eval/`，评估运行不再依赖外部 OPRD 数据目录。每项单独报告，原四任务平均不应与扩展任务混为同一个历史指标。

| 任务名 | 题数 | 数据范围 | 评分 | 每题采样 |
| --- | ---: | --- | --- | ---: |
| Minerva | 272 | 本地 OPRD 的 OCW 测试快照 | math_unified_v1 + minerva_units_v1 | 4 |
| Olympiad-Bench | 675 | 本地 OPRD 的 675 题快照，不代表完整上游基准 | math_unified_v1 | 2 |
| GPQA | 198 | 本地 source 字段为 gpqa_diamond | choice_exact_v2 | 8 |
| MMLU-Pro-Math | 1,351 | MMLU-Pro 中全部 category=math 的题目 | choice_exact_v2 | 2 |

## 数据版本与提示词

共有三个数据版本，题目、标准答案、选项顺序和 `extra_info.index` 完全相同，区别只在指令：

- `math_eval/<任务>/`：OPRD `test_data` 原样导入。题目末尾是 "Please reason step by step, and put your final answer within \boxed{}."。
- `math_eval/extended_dapo_v1/<任务>/`：去掉上面这句，在开头加上与 AIME/AMC/MATH-500 评测文件及训练数据相同的 DAPO 指令（"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer …"）。保留用于复现，未用于评测。
- `math_eval/extended_dapo_v2/<任务>/`：**正式口径**。在 v1 基础上，只给两个选择题任务在题目末尾另起一段加上："The last line must contain only the uppercase letter of the corresponding option in the question, for example: Answer: B"（即"最后一行仅输出题面对应的大写选项字母，例如 Answer: B"）。Minerva 和 Olympiad-Bench 与 v1 逐字节相同。

加这一句的原因：DAPO 指令只要求写出 "the answer to the problem"，没有说明选择题要写字母，模型可能直接写选项内容，而评分只认字母。示例里的字母 B 可能对选择有轻微的引导作用，可以通过各方法预测为 B 的比例与正确答案中 B 的比例（GPQA 54/198）对比来检查。

改用 DAPO 模板的原因：训练后的模型几乎只用 `Answer:` 行作答。在已有 MATH-500 评测中，GAD 正常结束的回答 763/768 以 `Answer:` 结尾，只有 17 条使用 `\boxed`；base 有 535/1000 条使用 `\boxed`。若扩展任务换一种指令，方法间比较会混入格式遵循的差异。

转换规则为 `DAPO 前缀 + 题目去掉 boxed 指令后 strip()`，与 OPRD `test_data_dapo` 的做法一致。OPRD 已有的两项逐行核对结果：GPQA 198/198、Olympiad-Bench 675/675 的提示词完全相同。Minerva 和 MMLU-Pro-Math 在 OPRD 中没有 DAPO 版本，按同一规则生成。

选择题行额外记录：

- `extra_info.options`：从题面解析出的选项文本。GPQA 198/198、MMLU-Pro 1350/1351 解析成功。MMLU-Pro-8182 的选项文字里含与选项标签同名的 "(I)"，切分不唯一，记为 `options_parse=ambiguous`、`options=None`。
- `extra_info.lowercase_labels_in_stem`：题干自带 a)、b) 这类小写编号（GPQA 3 题，MMLU-Pro 26 题）。

来源、哈希、解析统计和与 OPRD 的核对结果见 [extended_dapo_v2/manifest.json](../assets/datasets/math_eval/extended_dapo_v2/manifest.json)。v2 选择题与 OPRD 的比较不计末尾新增的这一句。v1 可由同一脚本加 `--dapo-version extended_dapo_v1` 逐字节重建。原样导入版本的清单见 [extended_manifest.json](../assets/datasets/math_eval/extended_manifest.json)。

## 选择题评分 choice_exact_v2

只从明确的最终作答中取选项，推理过程里出现的字母不用于猜测答案。可识别的写法：

- `\boxed{B}`，包括 `\boxed{(B)}`、`\boxed{\text{B}}`、`\boxed{\textbf{B}}`、`\boxed{\mathbf{B}}`。
- 行首冒号标签：`Answer: B`、`**Final Answer**: B`、`**Final answer:** B`。
- 只有标题的行，例如 `### Final Answer`、`**Final Answer**`：取其后第一个非空、非 `$$` 的行。
- "is" 型声明，出现在句中或行首都算：`The correct answer is B.`、`the answer is **B**`、`Final answer is B`。
  - 只有值以选项字母开头时才采用，所以 "the answer is not A"、"The answer is a bit tricky" 不会覆盖之前的作答。
  - 单个字母后面只跟理由（`B, since …`、`B because …`）时取该字母。
  - 值里出现多个选项（`A and B`、`A, B`、`A or B`、`(A) or (C)`）时，作为歧义的最终作答保留：不会只取其中一个，也不会回退到更早的答案。
- 字母后跟选项内容，例如 `\boxed{B. a}`、`Answer: (B) 7`：只有内容与该字母对应的选项文本一致时才接受，不一致按格式错误处理。没有 `options` 元信息时不接受这种写法。

按位置取最后一个作答。以下情况记为答错（格式失败）：

- 最后的作答列出多个选项，或同一行有冲突的 `\boxed`。
- 越界字母，或没有明确作答。
- 题干自带小写编号时用小写字母作答（例如 `\boxed{d}`）。

标准答案或选项元信息无效则记为未决。

**选项内容诊断：**回答直接写了选项内容而不是字母（例如 `\boxed{42}`），主指标仍记为格式失败。若规范化后恰好与一个选项文本相同，会在样本中记录 `content_option`，并在 `metrics.json` 的 `choice_content_diagnostic` 中汇总。其中 `accuracy_lower_if_content_counted` 只用于诊断，不进入主区间。

选择题评分只用于外部评估，未改训练数学评分器、BT 或 rubric 更新。

## Minerva 单位规则 minerva_units_v1

Minerva 标准答案都是不带单位的数字，题目则常写 "(in cm)"。`math_unified_v1` 保留单位，因此 "1.6 cm" 对 "1.6" 判错。`evaluate/minerva.py` 在统一评分器之后做一次评估侧处理：标准答案为标量、且最终答案恰好是"数字 + 单位"形式时，只用数字部分重新判分。

- 可识别的单位来自固定白名单：带 SI 前缀的常见单位、时间/角度/天文单位、`^\circ`、`\%` 等。
- `\text{…}`/`\mathrm{…}` 只有在内容本身是白名单单位组合时才算单位，例如 `\text{ cm}`、`\mathrm{~cm}`、`\text{ km s}^{-1}`、`\mathrm{erg}\,\mathrm{s}^{-1}`。
- 数字本身不对时仍然判错，例如 "1.58 cm"、"16 mm" 对 1.6。
- 不是单位的后缀不处理，例如 `5n`、`5 \pi`、`5/2`、`5 + 2 m`、`5 \text{or 6}`、`5 \text{ or } 6`、`5 \mathrm{n}`、`5 \text{+ 2}`、`5 \text{ approximately}`。
- 单位是否与题目要求一致无法检查，因为标准答案本身不带单位。

样本中记录 `unit_stripped` 和原始答案 `pred_with_unit`。`metrics.json` 记录 `eval_postprocess` 身份和 `unit_stripped_samples`/`unit_stripped_correct`。训练评分器和 `scoring/math.py` 均未改动。

## 数学评分覆盖率仍有限

2026-10-24 的标准答案静态检查发现，Minerva 有 28/272 题、Olympiad-Bench 有 72/675 题超出当前评分器支持范围，涉及方程、复杂符号、多组答案等。它们继续保留在固定分母中，以未决上下界报告，不能记成模型答错或从分母删除。

## 答案形式规则 answer_forms_v1（2026-10-26）

`evaluate/answer_forms.py` 在 `math_unified_v1`（Minerva 上再加 `minerva_units_v1`）之后，只对 Minerva 和 Olympiad-Bench 生效，训练评分器和 `scoring/math.py` 不变。规则：

- **答案列表：**顶层的 `,`、`;` 和单词 `and` 分隔各项，按无序集合匹配。`or` 不拆分，`1 or -4` 视为含糊作答。
- **多组数对：**`(a,b),(c,d)` 是有序元组的列表，元组内逐元素比较。
- **等式型标准答案：**`f(x)=2x`、`k=1`、`\frac{dx}{dt}=kx-a` 这类左边简单的等式比较右边。
  - 只写右边，或左边相同都接受；函数自变量换名（`f(m)=m^2`）也接受。
  - `MR=SRMC` 这类"词=词"按无序词对比较。
  - 左边复杂的（微分方程、`\angle BEA_1=90`）仍为未决。
- **前缀：**标准答案是值时，预测里简单的 `x = `、`f(t) = ` 前缀去掉再比较。
- **符号回退：**`\lfloor\rfloor`、`\lceil\rceil`、`\binom`、阶乘，以及 sympy 保留字母（N、S、E、I）、下标符号、花括号写法的 `\arcsin{}`/`\exp{}`、`\sin^{-1}`、`\log_{b}`，由本模块直接解析，再在固定探测点上比较数值，相对误差 1e-12。
  - 含取整、组合数、阶乘时用整数点，否则用有理数点。
  - `I(0)` 这类写法视为一个量的取值，不当作 `I*0`。
- **仍为未决：**含任意常数的标准答案（`C_1`、`\pi k`）、不等式（`k \geqslant 2`）、`t(0,4]`、带说明文字的答案。

安全约束：

- 对 `math_unified_v1` 原本就支持的标准答案，规则只会把答案改判为正确，不会把正确或未决改判为错误。
- 格式不合法的最终答案只可能被改判为正确。
- 只有原先判不了的标准答案，才会由规则直接判错。

覆盖率：标准答案可判的题，Minerva 从 244 增至 262/272，Olympiad-Bench 从 603 增至 670/675。

核对：在四个模型（base、SeqKD、GAD、correctness-only GAD）的已有样本上逐条比对，并用 OPRD 评分器交叉检查新判定的样本。

- 双方不一致的样本中，"本规则判对、OPRD 判错"的 158 条人工确认均正确。
- 反向的只有 2 条 `E_1 or E_2`，按含糊作答判错。
- 比对脚本和输出位于 `reports/eval_ext4_20261026/`。

新 profile 为 `ext4_dapo_v2_af1_taskn_8k`，旧 `ext4_dapo_v2_taskn_8k` 保持不变。规则是否启用由 rescore 配置中的 `answer_forms_fingerprint` 决定，规则加入之前已规划的评估仍按旧方式评分。已有生成可直接重新评分，不需要 GPU：

```bash
bash evaluate/run_extended_eval.sh --run RUN --methods seqkd,gad --allow-new-eval-scorer \
  --reuse-generation-from ext4_dapo_v2_taskn_8k
```

这两项目前属于带评分覆盖限制的扩展评估；若需要确定的单一准确率，需要另行完善评估侧判分。该计数仅检查标准答案，模型输出仍可能产生额外未决。也不能把当前统一评分结果直接称为上游官方协议成绩。

GPQA 和 MMLU-Pro-Math 全部标准答案的选项范围及自匹配检查通过。这不代表模型已经进行评估。

## 运行

从 LMOps 目录启动前先检查计划：

```bash
export GAD_PYTHON=/mnt/jinbo/miniconda3/envs/gad/bin/python
bash evaluate/run_extended_eval.sh --run baseline_06b_17kcnt --dry-run
```

该入口有以下默认设置：

- 只评四个扩展任务，使用 `extended_dapo_v2` 数据，profile 为 `ext4_dapo_v2_af1_taskn_8k`（2026-10-26 前为 `ext4_dapo_v2_taskn_8k`，未启用答案形式规则）。
- 模型默认选 base、SeqKD、GAD；需要 Warmup 时显式加 `--methods base,seqkd,warmup,gad`。
- non-thinking，输出上限 8192，seed=42，其他采样设置与 `run_eval.sh` 相同。
- 每题采样次数按上表设置。题目多的任务，误差主要来自题目之间的差异，增加每题采样帮助不大，所以 MMLU-Pro-Math 和 Olympiad-Bench 只采 2 次。

每个模型约 6,724 条生成；若四项都按每题 8 次采样，则约 19,968 条。原四任务继续用原入口和已有 profile 评测，不在这里重复生成。

确认训练结束、模型就绪及 GPU 空闲后，去掉 `--dry-run` 才执行。

`--task-n TASK=N` 可以覆盖单个任务的采样次数，后给出的值生效。修改后应换一个新 profile，不能与默认协议的结果混用。只评选择题：

```bash
bash evaluate/run_extended_eval.sh --run baseline_06b_17kcnt \
  --tasks GPQA,MMLU-Pro-Math --profile ext2_choice_dapo_v2_taskn_8k --dry-run
```

`--max-prompts 2` 只用于流程冒烟检查，不作为正式统计结果。

旧训练快照若仍记录旧版数学评分器，首次评估需额外添加 `--allow-new-eval-scorer --generate-missing` 并使用新 profile。不修改旧训练配置。结果位于各 run 的 `evaluation/<PROFILE>/`，汇总可使用同一命令加 `--summary`。

profile 的身份包含模型集合、任务集合，以及是否复用旧生成。因此不能向已有 profile 追加方法或任务，否则会报 `evaluation settings changed`。`--reuse-generation-from` 与 `--generate-missing` 也不能同时使用。补评时应在 profile 名后加后缀，另建一个 profile，采样设置保持不变，`summarize_runs.py` 会把同长度的 profile 合并到一行。例如 `baseline_06b_17kcnt` 的 SeqKD 已用旧评分器评完 Minerva 和 Olympiad，按 af1 重新评分写入 `ext4_dapo_v2_af1_taskn_8k_seqkd_mo`；其余任务（SeqKD 的 GPQA、MMLU-Pro，GAD 的全部四项）需要另行生成：

```bash
bash evaluate/run_extended_eval.sh --run baseline_06b_17kcnt --methods seqkd --tasks GPQA,MMLU-Pro-Math \
  --profile ext4_dapo_v2_af1_taskn_8k_seqkd_gm --gpu-ids 0,1 --dry-run
bash evaluate/run_extended_eval.sh --run baseline_06b_17kcnt --methods gad \
  --profile ext4_dapo_v2_af1_taskn_8k_gad --gpu-ids 2,3 --dry-run
```

跨 run 汇总，包括原四任务和扩展任务的均值，见 `evaluate/summarize_runs.py`：

```bash
python evaluate/summarize_runs.py baseline_06b_17k correctness_06b_17k_20261021 --max-tokens 8192 --csv out.csv
python evaluate/summarize_runs.py --all
```

- 每行对应一个 run、方法和生成长度，同长度的多个 profile 合并到一行。同一任务出现在多个 profile 时取最新的 metrics，并打印 note。
- Avg 是该行已有任务的等权均值，GPQA 也计入。Core4/Ext4 只在该组任务齐全时给出。
- 有未决样本的任务用下界 C/N，并标 `*`。不比较评分器版本。

## 重建数据

从另一个 OPRD 数据副本重建原样导入版本，再生成 DAPO 版本：

```bash
python evaluate/prepare_extended_data.py --source-root /path/to/OPRD/datasets/test_data \
  --output-root /path/to/new/math_eval
python evaluate/prepare_extended_data.py --dapo --source-root /path/to/new/math_eval \
  --oprd-dapo-root /path/to/OPRD/datasets/test_data_dapo
```

脚本不覆盖已有任务文件。迁移包构建器已递归收集 `assets/datasets/math_eval/`，会包含这些新增数据及清单。

CPU 测试：`python -m unittest evaluate.test_extended`。
