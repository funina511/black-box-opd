# GAD n=4 单卡 H200 迁移计划

状态：2026-11-14 r2 更新（路径与校验精简）。**本轮只在当前服务器准备代码、数据和迁移说明，代码通过 GitHub 交付，数据和部分依赖经个人电脑中转到另一台 H200 服务器；目标机部署和实验另行开展。** naive n=4 已完成，本机已建立独立发布副本及两个中转包；尚未推送、传输或验证 H200 可运行。实际交付位置、验证结果和命令见 [11月14日交付记录](n4选参与迁移交付记录_20261114.md) 及其链接的报告。下文旧日期命令为准备流程示例，不要对已生成目录重复执行。

本机前置整理已于2026-11-12执行：eval_fix与旧训练/生成shell归档，五份launch配置相对路径化，并完成配置与异目录核查。详细范围见 [目录整理与相对路径修正记录](目录整理与相对路径修正记录_20261112.md)。11月14日已继续完成本机打包准备；GitHub发布、中转与目标机部署仍待执行。

用户已同意将涉及运行中实验的进一步整理留到本次实验结束后统一处理，具体范围见 **3.5节**。11月14日按其条件核查：旧Warmup依赖仍服务于本机后续Actor实验，原路径继续保留；发布副本排除候选树和运行产物。其他涉及导入、身份或重复配置的较大改动暂不开展，逐项理由见交付记录第4节；不把所有拟精简项当成发布前必须重构的要求。

目标实际目录为 `<目标机绝对父路径>/bloxopd/LMOps`。用户通过 VS Code Remote-SSH 分别连接两台服务器。默认不要求服务器之间互通：代码走 GitHub，补充资产走“源服务器 → 个人电脑 → H200”。两台服务器不共享磁盘、Conda 环境或 SSH 配置。

最近已验证的 GAD 扩展后端为 `f0ce22a97d40`，含续训修复、Teacher 预取和提前标注。打包时核对实际代码版本；如果其后已有改动，记录差异，不把旧报告当作新代码已验证的证明。方法定义见 [三组对照方案](GAD_n4三组对照实施方案.md)，执行背景见 [提速报告](../../reports/gad_speed_20261110/README.md)。其中迁移旧 Warmup、仅重跑 GAD 的安排由本文替代。

## 1. 本轮范围与新机实验口径

新机从头训练，不迁移本机模型权重、Warmup checkpoint、优化器或 Judge 缓存。Qwen3-0.6B、Qwen3-8B 在新机下载，Critic 初始化在新机重建。

| 新机后续实验 | 起点与复用关系 |
|---|---|
| 公共 SeqKD | 从初始学生训练一次，作为独立基线 |
| baseline | 初始学生与共同单头 Critic → 新 baseline Warmup → baseline GAD |
| naive-rubric | 初始学生与从共同单头扩出的四头 Critic → 新 rubric Warmup → naive GAD |
| actor-rubric | 复用上述新 rubric Warmup 的 Actor 和 Critic，另起 GAD；参数按新 naive 记录选择 |

SeqKD 不作为 Warmup 的初始化来源；actor-rubric 不从 naive 最终 GAD 权重继续训练。完整重跑只需一次公共 SeqKD、两份 Warmup、三组 GAD。三组在同一台 H200 上评测，旧3090结果作为历史参考。

本轮边界：

- 只整理迁移交付物，不启动训练、Judge 或 GPU 测试，不连接目标机执行部署。
- 本机历史 run、报告、冻结配置和运行环境保持原状；涉及在用依赖的整理按3.5节延后，先在独立副本验证，再形成发布版本。
- 保留 replay 的代码、配置与测试，即使当前实验关闭 replay。
- 不改变标签、奖励、损失、Judge 协议或预算，不重写训练阶段交接。
- 本机不提前实现 H200 同卡适配、不猜测显存比例、不跑提速搜索，只记录部署待办。
- 复用现有打包工具与 Git、tar、rsync，不新增迁移框架、旧路径兼容层或大量防御分支。

## 2. 需要迁移和不需要迁移的内容

| 内容 | 本轮处理 |
|---|---|
| 正式训练、sample、evaluate、scoring、rubric 源码及相关测试 | 纳入代码交付，保留完整功能 |
| 配置、文档、环境版本记录 | 纳入；旧机器绝对路径作为来源记录，新机部署时生成实际配置 |
| `third_party/verl/` 及评估依赖 | 小型源码、模板和来源清单入 Git；`verl.bundle` 经电脑中转，保持原相对路径，不依赖旧机器相邻项目 |
| `assets/datasets/math_eval/` | 数据文件统一放补充依赖包，经电脑中转；任务映射和评测代码入 Git |
| `data/dapo_math_17k_gad_v2/` | 单独数据包，保留 train/dev、Teacher 回答、索引、manifest、quick_dev，约0.25 GiB，以实测为准 |
| 模型来源、revision及Critic初始化说明 | 只带必要元信息，不携带模型时不扫描权重 |
| 关键报告及本计划引用的文件 | 选取必要文件随包交付，不搬全部 reports |
| Qwen 模型、共同 Critic 初始化权重、旧训练权重 | 不迁移，新机下载或生成 |
| 旧 `runs/`、优化器、Judge 缓存、生成的 `gad/.backends_*` | 不迁移 |
| 整个 Conda 环境、旧日志与临时缓存 | 默认不迁移；带版本记录，新机安装 |

**必须迁移现有 Teacher 回答数据。** 下载 DAPO 原题不能还原此前的 Teacher 回答、划分与预处理。本轮从头训练仍使用相同完整17k数据，不重新采样，不恢复 correct-nontruncated 筛选；无需下载原 Teacher 模型。

公开模型在目标机下载并记录实际使用的 revision；旧 revision 不明确就如实注明，不要求新机与旧模型逐文件哈希一致，也不编造 commit。共同单头 Critic 不是下载模型后的现成文件，须保留 `gad/tools/prepare_rubric_critic.py`、seed42和生成说明；新机只生成一份共同初始化，再为 rubric 扩头。

**固定采用两条交付通道：GitHub 放代码、配置、文档、环境记录和小型依赖源码；个人电脑中转训练数据、评测数据与 verl.bundle。** 新机 clone 后再补齐这些资产。各包保持原相对路径，清单标明所属通道；不把大归档提交到 Git，也不把 clone 成功当成资产齐备。

## 3. 本机准备顺序

### 3.1 固定来源，建立独立发布目录

1. 记录当前 Git HEAD、工作区改动和相关配置，明确发布包含哪些未提交改动。
2. 在独立目录整理，不在当前训练目录批量删除、替换路径或清理 Git 历史。
3. 核对新文件是否纳入，尤其是 n4 配置、verl bundle、新增评测数据。`.gitignore` 不会移除已跟踪的历史文件，也不会自动纳入忽略的新依赖。
4. 发布前检查 API key、`.env`、SSH 私钥、私人连接配置及日志敏感字段；只保留环境变量名和占位符，不复制整份用户 SSH 配置。

### 3.2 准备 Git 发布副本和电脑中转包

可用现有工具导出一份供整理的源码快照，输出目录须尚不存在。`runtime.tar.gz` 仅作本机发布整理/备用，不是新机必须接收的代码包；正式代码从 GitHub 获取。训练数据和补充依赖分开打包：

```bash
cd /mnt/jinbo/RLRM/LMOps
export GAD_PYTHON=/mnt/jinbo/miniconda3/envs/gad/bin/python
export H200_RELEASE_DIR=/tmp/lmops_h200_release_20261112

"$GAD_PYTHON" gad/tools/package_runtime.py \
  --config configs/launch_rubric_naive_n4_06b_17k.yaml \
  --output "$H200_RELEASE_DIR"

tar -C /mnt/jinbo/RLRM -czf "$H200_RELEASE_DIR/training_data.tar.gz" \
  LMOps/data/dapo_math_17k_gad_v2

tar -C /mnt/jinbo/RLRM -czf "$H200_RELEASE_DIR/supplemental_assets.tar.gz" \
  LMOps/third_party/verl/verl.bundle \
  LMOps/assets/datasets/math_eval
```

这里用现机可解析配置记录来源，**不加 `--include-assets`**。r2工具在不携带资产时只记录模型/数据来源，不读取或哈希权重；源码由Git版本记录，不额外生成逐文件源码哈希。现有工具没有仅含数据的资产开关，直接单独 tar 即可。中转资产归档均采用 `LMOps/` 根路径，新机之后解到 `bloxopd/` 下即可保持层级。`/tmp` 仅作准备目录，交付后将归档留存到稳定位置，避免系统清理。

runtime 快照包含 bundle 和评测数据，整理 Git 发布副本时将这些资产排除，改由上述补充包交付，避免重复入库；无需修改打包工具。

工具默认不含 reports；将本计划及关键说明实际引用的少量报告补入独立发布目录或另附小包，不留下无法访问的必要依据。不要将源目录的全部 reports 直接装入包。

r2工具的 `REBUILD.md` 已改为通用迁移说明，使用第5节的新机部署待办，不将旧3090命令当成H200启动命令。包内 `assets/manifest.json` 及模型清单描述来源信息，不代表新机已有模型或旧初始化。不得把未携带模型时的 `assets_complete: false` 擅自改成 true。

### 3.3 核对交付物

本机只做迁移直接相关的核查：

- 接收后对两个压缩包各校验一次，再解包确认入口和必需资产存在；不再要求逐文件核验。软链接继续使用包内相对路径。
- 记录数据行数、版本和已有 fingerprint，不重生成数据、不重写原manifest、不新增整池哈希扫描。
- 源码以Git commit记录，环境记录版本；不另生成code_manifest逐文件哈希清单。
- 仅整理归档不重复跑训练或全套昂贵测试。发现缺文件就补清单；确需代码修复时说明原因并做对应 CPU 测试。
- 清单区分包内资产、新机下载项和重建项；本方案有意不生成包含模型的 `assets.tar`。

最终交付分为：

- **GitHub 发布版本**：代码、配置、文档、环境记录、小型依赖及必要报告摘要，记录明确 commit/tag。
- **电脑中转目录**：`training_data.tar.gz`、`supplemental_assets.tar.gz`、`TRANSFER.md`、`SHA256SUMS`，以及简短来源说明。

`TRANSFER.md` 记录对应 Git commit/tag、两个归档的恢复路径、数据版本、模型来源和第5节待办。工具生成的精简 `manifest.json` 可作为本机打包记录；不要求用户再核验一套源码或模型哈希。`runtime.tar.gz` 留在源机作备用，不放入默认中转目录。

将需中转的文件放入发布目录下的 `transfer/`，不混入 runtime 或 Git 工作目录。在该目录生成相对路径校验值，例如：

```bash
cd "$H200_RELEASE_DIR/transfer"
sha256sum training_data.tar.gz supplemental_assets.tar.gz > SHA256SUMS
# 在接收端运行一次：
sha256sum -c SHA256SUMS
```

生成清单后不再悄悄修改归档；若修改，重新生成并再次验证。

### 3.4 准备 GitHub 发布

在独立发布目录/分支检查 diff，确认代码和小型依赖齐全后再 commit/tag。归档可能包含未提交文件，不能仅凭原仓库 HEAD 声称与发布版本一致；记录最终发布 commit，核对它与归档代码的内容。

发布副本使用明确的 Git 纳入范围：保留小型依赖源码和来源说明，排除模型、训练数据、评测数据文件、bundle、中转归档、运行产物和凭据。对已跟踪资产需在发布分支单独整理；仅追加 ignore 规则不能解决。避免在源机工作目录直接 `git add .`。

准备好可审阅的 commit/tag 与 push 命令，由用户执行发布；当前仅修改计划，不实际推送。沿用用户现有 GitHub 仓库，不擅自创建仓库或更改可见性。代码 commit 与中转包版本一一对应；无需为迁移改写 Git 历史或清理源机实验。

### 3.5 本次实验结束后：统一整理在用依赖与重复实现（按需收尾）

这部分在源服务器完成，属于迁移前收尾，不是把H200部署提前搬到本机。目标是让发布版本只有清楚的正式入口，同时保留当前实验可追溯性；不趁机重写训练逻辑。

**开始条件与顺序**

1. 核对 `rubric_naive_n4_06b_17k_v2` 已正常完成目标训练，最终checkpoint可用，训练driver/worker已退出；异常退出、日志暂时不增长不算完成。若还需要恢复未完成训练，继续保持其依赖原位。
2. 保存本轮实际使用的代码版本、配置、scorer/Judge身份、最终checkpoint与必要日志清单，保留原Warmup来源记录。当前快照不等于最终发布版本。
3. 检查是否仍有评测、导出、Judge服务、后台任务或后续本机实验使用这些路径；被使用的文件延后移动。只处理本实验所属服务，不做全局进程清理。**不以数学评测未决数必须为0作为训练结束或目录整理的门槛。**
4. 先做历史副本归档和入口/配置整理，再处理需要改变导入或身份哈希的代码精简。独立副本通过对应验证后再形成新的发布版本，最后执行3.2–3.4的最终打包和发布。

**拟处理清单**

| 对象 | 收尾时的处理 | 必须保留的边界 |
|---|---|---|
| `judge_fix/` 的候选源码、快照、测试产物 | 逐项确认有用修复已在正式 `gad/`、`rubric/` 中；未合入内容只列差异，不整目录覆盖。保存必要报告和归档清单后，将不再被引用的候选树移出主目录 | 旧 rubric Warmup 冻结配置仍指向其中的backend、prompt/spec和scorer。只要还需按旧路径恢复/复现，就保留所需路径；不回写旧配置、不用软链接悄悄指向新版实现 |
| `gad/.backends_*` 与候选树里的生成backend | 区分本轮实际使用、历史复现所需及纯短测临时worktree；无引用的临时worktree可按Git worktree机制清理，正式发布仍由bundle重建 | 不直接递归删除源仓库或仍被登记/使用的worktree；不强制覆盖活动backend。当前 `gad/verl/` 不因“可重建”就先删掉 |
| 五份完整launch与多层overrides的重复设置 | 以现有 `train.yaml + methods + machines + overrides` 机制为基础，整理公共n=4设置和naive/Actor差异；在发布版本中给出一套明确推荐组合 | 不新增YAML继承框架。拆分前后比较最终解析配置、阶段预算和交接来源；不得把“复用旧Warmup”默默变成“从零训练”，H200新配置另行明确 |
| 旧离线rubric预实验入口 | 旧 `collect/annotate/train_offline` 等与其专属配置、测试成组归档，正式共享实现保留在 `rubric/core/` | 先区分旧专属测试和仍覆盖在线契约的测试；同步整理 `scripts/rubric.sh`、测试发现与打包白名单，不只删除被import的文件。不要为了归档把全部旧逻辑搬进core |
| 旧事件/证据复核协议与direct协议并存 | 在独立副本核对direct仍需要的导入和身份计算；能明确分离的旧入口归档，重复且语义完全一致的工具才合并。无法小范围分离的部分保留并注明原因 | `process_v2`、`semantic_v24` 等目前仍有导入/哈希关系。不是删掉旧分支就完成；保留direct的输入输出、null/mask、失败保护和请求账本语义，不改Judge准则 |
| `gad/deepscaler/` 及旧安装文件 | 检查固定backend中的实际可达导入和环境安装依赖；确认当前受支持路径不再需要后才归档 | 现有backend仍有deepscaler导入，不能凭主入口搜索不到就删除。不要为了删一个小目录大幅改vendor代码 |
| eval_fix归档中的独有回归用例 | 核对正式测试是否覆盖关键评分边界；确有缺口时迁入 `scoring/tests/` 或正式evaluate测试入口，删除新测试对旧候选scorer的依赖 | 正式评分只保留 `scoring/math.py`；不重新引入第二套评分实现，不批量恢复整套历史重评分工具 |

**共同约束与验收**

- `gad/extensions/core` 已是指向 `rubric/core` 的相对软链接，不是两份源码；继续共享即可。BT、rubric BCE、Actor奖励、replay、Teacher预取与训练阶段交接不借本轮整理改变。
- 单纯路径/配置拆分应保持解析后的参数和阶段签名一致；实际移动核心文件或删减旧协议可能改变代码哈希/Judge指纹，应记录为新发布身份，不能伪造原 `f0ce22a97d40`。新机从空标签缓存开始，旧实验身份与标签不回写。
- 归档保留原文件及内容清单，不删除模型、最终checkpoint、训练数据或尚需复现的证据。该计划授权的是整理，不能将“无须迁移”直接解释为“本机全部删除”。
- 按实际改动运行相关CPU测试，不因小范围配置或打包修正重复全套测试、GPU训练或全资产扫描。旧专属测试若退出正式套件，应说明去向，不删断言来掩盖失败。
- 在新的临时 `bloxopd/LMOps` 路径核查训练、sample、evaluate入口与配置解析，不依赖源机 `judge_fix`、旧绝对路径或工作目录巧合。真实单卡GPU验证仍留给H200，CPU通过不能写成训练已验证。
- 验收报告列出移动/合并/保留文件、测试结果、发布commit、后端/Judge身份变化和仍待目标机验证的事项。没有必要改动的兼容分支可以保留，**不把“删得最多”当目标**。

## 4. GitHub 获取代码，个人电脑中转补充资产

**注意终端所在机器。** VS Code 连接源服务器时，集成终端中的命令运行在源服务器；连接 H200 时则运行在 H200。下面标为“个人电脑”的命令应在电脑本地的 PowerShell、Terminal 等运行，不要在 Remote-SSH 窗口的远程终端执行。

### 4.1 源服务器：发布代码并准备下载目录

完成第3节审阅后，将发布分支/tag 推到用户现有 GitHub。将两个资产包、版本说明和校验清单放在源机的 `transfer/` 中。源码归档保留备用，不需要重复经电脑上传。

### 4.2 个人电脑：下载，再上传

下面的 `source-host` 和 `h200-host` 使用个人电脑上连接 VS Code 时采用的 SSH Host 别名；替换路径后执行。第一条将整个 transfer 目录下载到电脑当前目录，第二条准备 H200 接收目录，第三条上传，不要求两台服务器相互 SSH。

```bash
# 在个人电脑的本地终端执行，而非任一服务器的远程终端。
scp -r source-host:/tmp/lmops_h200_release_20261112/transfer ./

ssh h200-host 'mkdir -p /absolute/parent/bloxopd/incoming/lmops_20261112'

scp -r ./transfer h200-host:/absolute/parent/bloxopd/incoming/lmops_20261112/
```

若不用 SSH 别名，改成 `user@host`；非默认端口分别用 `scp -P 端口`、`ssh -p 端口`。电脑已有的 SSH 配置若包含跳板机，沿用该配置即可，不复制私钥到服务器。SCP 示例没有断点续传保证；较大资产需续传时可在支持 rsync 的电脑/WSL 中使用 rsync，或使用支持续传的 SFTP 工具。VS Code 的文件下载功能也可用于先取回归档。

### 4.3 H200：clone 固定代码版本，核对收到的资产

使用 VS Code 连接 H200 后，在该远程终端运行。仓库 URL、已发布 tag 和绝对路径均须替换：

```bash
mkdir -p /absolute/parent/bloxopd
cd /absolute/parent/bloxopd
git clone <你的GitHub仓库URL> LMOps
cd LMOps
git checkout --detach <本次发布tag或commit>

cd /absolute/parent/bloxopd/incoming/lmops_20261112/transfer
sha256sum -c SHA256SUMS
```

`LMOps` 已存在时不覆盖、不强制 reset，先核对其版本与本地改动。GitHub 认证由目标机独立配置，不能假定个人电脑或源机凭据会自动可用。

校验通过后，部署时将两个资产包解到 `/absolute/parent/bloxopd/`，它们自带 `LMOps/` 前缀，补齐 Git 中没有的数据与 bundle；不要解到 `LMOps/` 里形成重复的 `LMOps/LMOps/`。清单须保证资产包不会覆盖 Git 管理的源码或新机配置。

本轮交付验收止于：GitHub 发布内容明确、电脑中转包齐全，用户传输后 clone 的代码版本与说明一致、归档校验通过。环境安装、模型下载、同卡适配和训练留到下一阶段。11月14日已准备本机交付物，未 push、传文件或连接远端。

## 5. 留到迁移后处理的事项

这里只交代待办，不展开部署命令、显存预算、提速实验或正式训练命令；原第6节及后续详细方案不再保留。

| 目标机后续工作 | 必须保留的注意点 |
|---|---|
| 解包、安装环境、下载模型 | 根目录为 `bloxopd/LMOps`；按版本记录安装，实测 H200 上 Torch、FlashAttention、vLLM 可用；固定模型版本 |
| 重建共同 Critic 初始化 | 复用 `prepare_rubric_critic.py`，seed42、单头初始化；rubric 按现有机制扩头并保留 adv 行 |
| 单卡及 Judge 共卡适配 | 核对 `extension_config.py`、`rubric_judge.py`、`preflight.py` 三处重叠限制，使用默认关闭的显式选项，保留身份校验；先启动 Judge，再启动训练 |
| 生成新机配置 | 重定位路径、卡号和 TP，不覆盖3090配置。baseline/naive 清除 `budget.init_from_warmup_run`、`budget.init_step` 等旧来源；Actor 指向新 rubric Warmup |
| 核对完整训练预算 | 保留 SeqKD 2 epoch、Warmup 1 epoch、GAD新增271步、GAD n=4；SeqKD/Warmup 的 n 保持1。实际步数由 dry-run 核实，勿将总步数误作新增步数 |
| 小规模验证 | 覆盖 SeqKD、两类 Warmup、Warmup→GAD、保存与续训及共卡显存；不能仅验证旧 Warmup→GAD，不预先承诺提速 |
| 正式重跑与评测 | 按第1节复用关系执行，Actor 参数按新 naive 记录选择；统一八任务16k评测，训练输出上限仍8192。评测前释放训练/Judge占用 |

现有 n4 launch 含旧 Warmup 路径，**迁移后不能直接照搬启动**；这是部署待办，不要求本机为打包重写框架。`f0ce22a97d40` 是已有 GAD 扩展后端参考值，不要求 SeqKD/Warmup 后端目录同名。模型文件和 TP 变化后建立新 Judge 指纹，不复用旧机缓存。

同卡适配、显存比例、吞吐和完整重跑耗时尚未验收，留到目标机结合实际资源验证。新机部署完成前，本机原始代码、数据和历史实验继续保留。

修订记录：2026-11-12 收敛为本机迁移准备与 SSH 交付，取消旧权重迁移、旧 run 路径兼容和跨拓扑续训安排；删除原第6节以后的详细部署、短测、提速及训练命令，仅保留目标机待办和完整重跑口径。此前未完成的文档修订由本版统一替代。

补充修订：2026-11-12 按 VS Code Remote-SSH 的实际使用方式，确定“GitHub 发布/clone 代码 + 个人电脑中转数据与大依赖”为主流程；取消源服务器直连 H200 的前提，runtime 归档改为备用。

补充修订：2026-11-12 新增3.5节，将当前在用依赖、Judge候选树、重复launch、离线rubric与旧协议等收尾整理明确安排在本次实验结束后；当前仅写入计划，不执行这些改动。

补充修订：2026-11-14 naive正常完成后执行离线选参与本机迁移准备；使用独立发布副本，保留本机Actor实验仍需的旧Warmup依赖。仅修正打包遗漏与Git排除范围，具体取舍及验证见本次交付记录。


2026-11-14 r2补充：本次只修改独立发布副本，原Actor-c-n4实验与Judge保持原代码。新launch的scorer指纹改为自动生成，机器/采样示例用相对路径；Judge启动取消未被使用的全权重哈希字段，源码打包取消逐文件哈希及AST审计。默认中转只保留Git版本、两个归档及其一次性校验、简短来源说明；旧逐文件清单留作本机历史记录，不是部署门槛。新tag使用`h200-prep-20261114-r2`，旧tag不移动。

本次没有解除旧run跨路径续训限制，也没有提前放开H200共卡。新机从头训练时清除旧Warmup来源，使用新机生成的身份；不要求模型路径、inode或旧scorer指纹与源机一致。checkpoint完整性、标签mask和服务模型/协议的基本检查继续保留。详见 [路径与校验精简记录](迁移路径与校验精简记录_20261114.md)。
