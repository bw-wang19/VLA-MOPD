# 经典 MOPD 复现范围与 VLA 扩展契约

## 1. 复现定位

本仓库的第一阶段目标是复现 Ma et al. **MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training** 的核心算法，而不是宣称重建其 Qwen3-30B-A3B 或 MiMo-V2-Flash 数值结果。

当前可合理声称的范围是：

- 忠实实现经典论文 Eq. (1)–(5) 的 reverse-KL、sampled-token policy-gradient 和 corrected top-k 两条路径；
- 实现同源、多教师、按已知 domain label 硬路由的训练闭环；
- 用单元测试、解析分布或小模型实验验证公式、梯度、mask、路由和聚合语义；
- 定义可由 autoregressive action-token VLA 接入的状态、动作、teacher scoring 与指标契约。

下列内容不在“经典论文数值复现”的可声明范围内：原论文完整 SFT 数据混合与过滤流程、Stage-1 SFT checkpoint、Stage-2 domain RL teachers、内部训练基础设施、Qwen3-30B-A3B 表格数值，以及 MiMo-V2-Flash 的工业规模结果。

权威链接与论文关系见 [`ref/README.md`](../ref/README.md)。

## 2. 经典 MOPD 三阶段定义

令输入为 \(x\)，其已知领域标签为 \(d(x)\)，student 为 \(\pi_\theta\)，领域 teacher 为 \(\pi_{\phi_{d(x)}}\)。经典流程是：

1. **Stage 1 — General SFT**：在覆盖所有目标能力的语料上得到共同初始化 \(C\)。
2. **Stage 2 — Domain-specialized RL**：每个 teacher 都从 \(C\) 独立训练；各领域可以采用不同环境、reward 和 RL 配方。
3. **Stage 3 — MOPD**：student 重新从 \(C\) 初始化；student 生成自己的 rollout；每个样本按 \(d(x)\) 路由给一个冻结 teacher；teacher 在相同 student prefix 上评分；student 用 dense token-level reverse-KL 信号更新。

“同源”不是可省略的实现细节：主论文的消融显示，更强但分布差异更大的外部 teacher 会显著提高初始 KL，并可导致熵收缩或训练发散。

## 3. Eq. (1)–(5) 的精确复现范围

以下所有 token 分布均条件于相同的 \((x,y_{<t})\)。\(y\sim\pi_\theta\) 表示 trajectory 由当前 student 生成，\(\operatorname{sg}[\cdot]\) 表示 stop-gradient。

### 3.1 Eq. (1)：on-policy per-token reverse KL

\[
\mathcal L_{\mathrm{rev\text{-}KL}}
=\mathbb E_{x,\,y\sim\pi_\theta}
\left[
\frac{1}{|y|}\sum_t\sum_v
\pi_\theta(v)
\log\frac{\pi_\theta(v)}{\pi_{\phi_{d(x)}}(v)}
\right].
\]

必须满足：

- 外层状态/prefix 分布来自 student，而不是 teacher 的离线 completion；
- 每个位置的 teacher 与 student 接收完全相同的 prefix；
- teacher 参数冻结且以 `eval` 语义评分；
- \(|y|\) 只统计有效 response/action token，padding、prompt token 和无效 action 必须由 mask 排除。

### 3.2 Eq. (2)：sampled-token policy-gradient 梯度

\[
\nabla_\theta\mathcal L
=-\mathbb E_{x,y}
\left[
\frac{1}{|y|}\sum_t
\log\frac{\pi_{\phi_{d(x)}}(y_t)}{\pi_\theta(y_t)}
\nabla_\theta\log\pi_\theta(y_t)
\right].
\]

这是 Eq. (1) 梯度的 Monte Carlo policy-gradient 形式。实现测试应区分两条 student log-prob 路径：形成 advantage 的旧值必须 detach；乘在 loss 中的当前 log-prob 必须保留梯度。

### 3.3 Eq. (3)：token-level MOPD advantage

\[
\widehat A_{\mathrm{MOPD},t}
=\operatorname{sg}\!\left[
\log\pi_{\phi_{d(x)}}(y_t)
-\log\pi_\theta(y_t)
\right].
\]

正值表示 routed teacher 对 student 已采样 token/action 赋予更高概率；负值表示 teacher 相对不支持该选择。不能对不同 teacher 的 logits 先平均后再代入本式；经典方法是一条样本只路由到一个 teacher。

### 3.4 Eq. (4)：双侧裁剪后的最终 PG loss

\[
\widehat A^{\mathrm{clip}}_{\mathrm{MOPD},t}
=\operatorname{clip}
\left(\widehat A_{\mathrm{MOPD},t},-A_{\max},+A_{\max}\right),
\]

\[
\mathcal L^{\mathrm{PG}}_{\mathrm{MOPD}}(\theta)
=-\mathbb E_{x,y}
\left[
\frac{1}{|y|}\sum_t
\widehat A^{\mathrm{clip}}_{\mathrm{MOPD},t}
\log\pi_\theta(y_t)
\right].
\]

验收要求：advantage stop-gradient、双侧而非单侧 clip、先做每条 trajectory 的有效 token 平均，再按 batch/domain 配方聚合。直接对整个 padded batch 做全局 token mean 会改变论文中 \(1/|y|\) 的样本权重。

### 3.5 Eq. (5)：corrected teacher-top-k distillation

令

\[
\mathcal T_t^d
=\operatorname{TopK}_k
\left(\pi_{\phi_{d(x)}}(\cdot\mid x,y_{<t})\right),
\]

则

\[
\mathcal L^{\mathrm{TopK}}_{\mathrm{MOPD}}(\theta)
=\mathbb E_{x,y}
\left[
\frac{1}{|y|}\sum_t
\sum_{v\in\mathcal T_t^d}
\left(
\pi_\theta(v)\log\frac{\pi_\theta(v)}{\pi_{\phi_{d(x)}}(v)}
-\pi_\theta(v)+\pi_{\phi_{d(x)}}(v)
\right)
\right].
\]

复现时必须保留 \(-\pi_\theta(v)+\pi_{\phi_{d(x)}}(v)\) 校正项。它使截断目标在 top-k 支持上仍以 teacher/student 相等为极小点；“只在 top-k 上计算朴素 reverse-KL”不是论文 Eq. (5)。经典论文的集合是 **teacher top-k**，也不应静默替换成 student top-k、交集或并集。

### 3.6 本阶段暂不混入的扩展

最早的 MiMo-V2-Flash Technical Report 还包含：

- inference sampler \(\mu_\theta\) 与 training policy \(\pi_\theta\) 间的截断重要性采样；
- MOPD token advantage 与 outcome-reward / GRPO advantage 的加权组合。

这些是有价值的后续扩展，但不属于本仓库“经典论文 Eq. (1)–(5) 最小闭环”的默认语义。启用时必须使用独立配置并在实验名和报告中显式标注。

## 4. 主论文报告的 Stage-3 设置

| 项目 | 论文报告值或状态 |
| --- | --- |
| Student 初始化 | Stage-1 SFT checkpoint |
| Teachers | 从同一 Stage-1 checkpoint 做 domain RL；Stage 3 全部冻结 |
| 领域 | Math、Instruction Following、SWE |
| 每 batch 领域比例 | `0.35 : 0.35 : 0.30` |
| Batch size | `2048` |
| 每 prompt rollout 数 \(N\) | `1` |
| Dynamic Sampling | MOPD 不使用 |
| PG advantage clip \(A_{\max}\) | `5` |
| Top-k 路径 \(k\) | `64` |
| 各领域训练最大长度（Appendix A） | Math/IF `32,768`；SWE `65,536`，最多 50 个交互 turn |
| Teacher 计算 | 独立异步 prefill service；与 student sampling 重叠 |
| 论文评测解码 | temperature `1.0`，不做 top-p 或 top-k 截断 |
| MOPD learning rate | **未报告** |
| MOPD optimizer / scheduler | **未完整报告** |
| Stage-3 训练 rollout 的完整采样参数 | **未完整报告** |

最后三项不得从 Mix-RL、domain RL、Open-MOPD 或 NeMo 的配置中猜测并冒充原论文值。仓库提供的默认值只能标为“本实现默认值”，并在实验产物中完整落盘。

本仓库当前的 `domain_ratios` 在整个有限训练集上做确定性配额和随机交错，不是 distributed balanced-batch sampler；因此单个 optimizer update 的领域组成可波动，尤其在小 batch 下。论文的 `0.35:0.35:0.30` 是 per-batch 配方。训练日志会同时给出实际 trajectory/token share，但若要复现这一项工程配方，仍需增加跨 rank、跨 gradient-accumulation microbatch 协调的 balanced sampler。

断点续训限定为同一训练计划的恢复：checkpoint 契约会固定 `max_steps`（它决定 cosine scheduler 曲线）、world size、模型/数据/算法与优化语义，并保存轻量输入资产清单。小文件使用 SHA-256，大文件仅记录大小与 mtime，远程 locator 也不能证明 Hub 内容未变；因此这是一层 fail-fast 防线，不替代不可变 snapshot、数据版本控制和正式实验的外部 checksum。

## 5. 为什么不能声称原论文数值复现

经典论文公开了算法、部分数据来源、关键超参数和评测协议，但没有开放构成其受控实验所需的完整资产：

- 精确的 broad-coverage SFT 数据混合、清洗、蒸馏与 prompt 细节；
- 论文实际使用的 Stage-1 Qwen3-30B-A3B SFT checkpoint；
- Math、IF、SWE 三个 Stage-2 RL teacher checkpoint；
- 各领域完整 RL 数据、reward/verifier、SWE 环境快照与训练基础设施；
- 足以逐项重建训练的全部 optimizer、scheduler、rollout 和随机性配置。

MiMo-V2-Flash 仓库开放模型权重不等于开放经典 MOPD 受控实验的中间资产。Open-MOPD 虽然提供完整公开配方，但使用 SmolLM3-3B、不同数据与额外优化机制，是后续独立实验。

因此报告措辞必须遵守：

- 可以写“复现 Eq. (1)–(5)”“算法级复现”“缩小规模验证”；
- 只有实际运行并记录的结果才能写成本仓库结果；
- 论文表格只能写成“论文报告值”；
- 不得写“完整复现论文”或暗示已经复现 Qwen3/MiMo 的原始数值。

## 6. LLM → VLA 扩展契约

经典 LLM 公式中的 prefix/token，在 VLA 中对应 student 实际访问的 state 与实际采样/执行的 action。扩展不改变 on-policy 核心：**student 决定访问哪些状态，teacher 只在这些状态上评分，不替 student 执行动作。**

### 6.1 State 契约

每个 \(s_t\) 至少应保存：

- 与 student rollout 时一致的视觉观测，包含相机名称、顺序、时间戳和预处理元数据；
- language instruction / task description；
- 模型需要的 proprioception、机器人形态信息与必要历史；
- `episode_id`、`step_id`、`task_id`、`domain_id`、终止/截断标志；
- 能确保 teacher 与 student 对同一物理时刻评分的不可变观测快照。

禁止在 teacher scoring 时重新从环境读取一个已经变化的“当前观测”来替代 rollout 中的 \(s_t\)。

### 6.2 Action 契约

\(a_t\) 必须是 student 在 \(s_t\) 上真实采样并送给环境的 action，或其可逆的 action-token 表示。对于 action chunk，还需保存：

- chunk horizon、维度顺序与每个 token/维度的有效 mask；
- 执行了 chunk 中哪些步、何时因成功/失败/终止被截断；
- rollout 时的 student/behavior log-prob 与 policy version；
- teacher 对完全相同 action token、相同 state/history 的 log-prob。

teacher 生成的 argmax action 不能替换 student action 后再计算 Eq. (3)，否则训练数据不再是 student on-policy rollout。

### 6.3 Action tokenizer 与 normalization 契约

使用经典 sampled-token 或 top-k 目标前，student 与所有 teachers 必须共享：

- action vocabulary、token IDs、量化边界与特殊 token；
- action chunk 长度、展平顺序、token-to-joint/dimension 映射和 mask 语义；
- 坐标系、控制模式、物理单位、关节/末端顺序、gripper 开闭约定；
- action bounds、normalization statistics 和对应数据键（例如 OpenVLA-OFT 的 `unnorm_key`）。

加载模型时应 fail fast 校验这些 metadata，不能只因 tensor shape 相同就允许蒸馏。若 teacher/student 是连续 flow、diffusion 或其他没有可比逐 action log-density 的策略，Eq. (1)–(5) 不能直接套用；为它们设计 score/flow divergence 属于另一个明确标注的研究分支。

### 6.4 Same-origin 契约

默认实验要求 student 与所有 teachers：

1. 从同一个 VLA SFT checkpoint 分叉；
2. 使用同一模型族和 action 表示；
3. teacher 只通过对应领域/任务的 RL 或受控后训练形成专长；
4. Stage 3 teacher 冻结并关闭 dropout。

如研究异源 teacher，必须作为独立消融，并记录初始 per-domain reverse-KL、student entropy、梯度范数与 collapse 指标；不能与经典 same-origin 主结果混报。

### 6.5 Hard routing 契约

经典 MOPD 使用已知标签的 oracle hard routing：

\[
\texttt{domain\_id}\longrightarrow\texttt{teacher\_id}.
\]

- 每条 trajectory 在一次更新中只由一个 routed teacher 监督；
- 路由表必须一一可审计，未知或缺失 domain 应直接报错；
- 一个 trajectory 中不得悄然切换 teacher；
- teacher ensemble、均匀平均、学习型 router、token/step 级动态路由均属于后续方法，不是经典基线。

VLA 的 `domain_id` 可以表示 task suite、技能、机器人形态或受控环境族，但一项实验必须预先固定定义，不能根据评测结果事后重标。

### 6.6 On-policy 与异步执行契约

同步实现直接用当前 \(\pi_\theta\) rollout。若 rollout worker 与 learner 异步：

- 每条 action 保存 behavior policy version 和 behavior log-prob；
- 监控 train/behavior log-ratio 与策略陈旧度；
- 若使用 MiMo Technical Report 的截断重要性采样，必须显式记录上下界和被 gate 掉的 action 比例；
- 未做校正且策略已明显陈旧时，不得称为严格 on-policy。

### 6.7 聚合与指标契约

长 horizon 任务不应仅因 action token 更多就支配共享 student。训练与报告至少同时保留以下三个粒度：

1. **Token/action 粒度**：teacher/student sampled-action log-prob、advantage、reverse-KL 估计、student entropy、clip fraction、有效 action 数。
2. **Trajectory 粒度**：先在每条 trajectory 的有效 token/action 上求平均；记录 return、success、长度、终止原因与恢复动作统计。
3. **Domain 粒度**：分别记录样本数、trajectory 数、有效 action-token 数及其占比、loss、KL、entropy、success rate；最终总分以 domain macro-average 报告，而不是只给全局 micro-average。

推荐同时报告每个领域的 teacher-headroom recovery：

\[
\mathrm{Recovery}_d
=\frac{s_d^{\mathrm{student}}-s_d^{\mathrm{SFT}}}
{s_d^{\mathrm{teacher}}-s_d^{\mathrm{SFT}}},
\]

并单列分母接近零或为负的领域，避免产生无意义的比值。训练日志还应显示配置的 domain 采样比例与实际 trajectory/action-token 占比；二者不等价。

## 7. 最小实验矩阵

在宣称 VLA-MOPD 有效前，至少应比较：

- SFT initialization；
- 每领域独立 single-teacher VLA-OPD（能力上界/RouteOPD 参考）；
- naive hard-routed VLA-MOPD；
- off-policy teacher BC/SFT；
- mixed-domain sparse-reward RL（若环境 reward 可得）。

评测至少包括每领域 task success、macro-average、headroom recovery、student entropy、reverse-KL、训练样本/环境步数，以及 seen/unseen task 的遗忘情况。所有曲线必须标明 seed、模型 checkpoint、teacher 路由表、环境版本和 action metadata 摘要。
