# VLA-OPD 论文忠实性与实现边界

本文档规定本项目哪些行为直接来自论文、哪些是为形成可运行软件所做的选择，以及哪些内容由于作者代码和实验配置未公开而无法核验。规范来源是 [VLA-OPD arXiv:2603.26666v1](https://arxiv.org/html/2603.26666) 的 Algorithm 1、Eq. (5)–(7) 和实验章节。

## 1. 三阶段闭环

### Phase 1 — Student on-policy sampling

对 batch 中每个 prompt，当前 student policy 与环境闭环交互并生成 $G$ 条 trajectory：

\[
\tau_i=(s_{0,i},a_{0,i},s_{1,i},a_{1,i},\ldots,s_{T_i,i}).
\]

动作必须从 student 的 categorical action distribution 随机采样，并实际送入环境。状态因此来自 student 诱导的分布，而非离线 expert demonstration 或 teacher rollout。

### Phase 2 — Frozen-teacher labeling

teacher 不在环境里执行动作。对于 student 已访问的每个 $s_{t,i}$，teacher 对**同一个 student action** $a_{t,i}$ 计算

\[
\log\pi_{tea}(a_{t,i}\mid s_{t,i}).
\]

这一步是 forced scoring，不是生成 teacher trajectory，也不是把 student action 替换成 teacher argmax。teacher 参数冻结；观测必须是 rollout 时保存的不可变快照。

### Phase 3 — Mode-seeking student update

student 对收集到的相同 observation/action pair 做可微重打分，以 Eq. (6) 的 dense token reward 执行 Eq. (7) 的 policy-gradient 更新。完成更新后丢弃该 on-policy batch，再由新 student 收集下一批 trajectory。

## 2. Eq. (5)–(7)

### Eq. (5)：negative reverse KL

论文优化目标是

\[
\max_\theta\mathcal J(\theta)=
\mathbb E_{s\sim\pi_\theta}
\left[-D_{KL}\left(
\pi_\theta(\cdot\mid s)\,\|\,\pi_{tea}(\cdot\mid s)
\right)\right].
\]

reverse-KL 的 action 期望由 student 分布采样，而非从 teacher 分布采样。实现因此只要求 teacher 给 student 已选 token 的 log-probability；若 adapter 能返回完整 logits，可以额外记录完整 KL/entropy 指标，但这不是 sampled Eq. (7) 更新的必要输入。

### Eq. (6)：raw immediate OPD reward

对采样 action：

\[
r_t^{OPD}
=-\left(
\log\pi_\theta(a_t\mid s_t)-
\log\pi_{tea}(a_t\mid s_t)
\right)
=\log\pi_{tea}(a_t\mid s_t)-
\log\pi_\theta(a_t\mid s_t).
\]

论文明确要求 reward 中的 student log-probability 使用 `stop_gradient`。代码把 rollout 时保存的 `behavior_student_log_probs` 用于 reward：

```text
reward = (teacher_log_probs - behavior_student_log_probs).detach()
```

默认 `reward_clip: null`。论文没有为 VLA-OPD Eq. (6) 指定 reward clipping，所以任何裁剪都只能作为显式扩展，不能标为论文默认。

### Eq. (7)：group Monte Carlo average

论文写为

\[
\nabla_\theta\mathcal J(\theta)
\approx\frac{1}{G}\sum_{i=1}^{G}\sum_{t=0}^{T_i}
\nabla_\theta\log\pi_\theta(a_{t,i}\mid s_{t,i})
\,r_{t,i}^{OPD}.
\]

用梯度下降实现时：

\[
\mathcal L_{OPD}
=-\frac{1}{N_{traj}}\sum_i\sum_{t\in valid(i)}
\log\pi_\theta(a_{t,i}\mid s_{t,i})
\operatorname{sg}[r_{t,i}^{OPD}].
\]

`G` 的作用是对同一 prompt 下的独立 trajectory 做 Monte Carlo 平均、降低环境和采样方差。它不构造相对 advantage：没有减 group mean，也没有除 group standard deviation。

## 3. 时间聚合：论文依据与歧义

本实现默认 `reduction: trajectory_sum`，原因是 Eq. (7) 和 Algorithm 1 都明确写出 `sum_t`，仅对 group 和 prompt batch 求平均。于是长 trajectory 在默认目标中可能贡献更大的总权重。

公开论文仍留下两个无法由官方代码消除的歧义：

1. 论文没有说明实际训练代码是否在 padding/micro-batch loss 聚合时又除以有效 token 数；第三方实现采用 token mean 不能证明作者实现也如此。
2. Eq. (7) 直接把 $r_t^{OPD}$ 乘在同一时刻的 score function 上，没有写 reward-to-go、discounted return 或未来 reward 累积。对于动作会改变未来状态的 MDP，这与完整 episodic credit assignment 并不等价；本项目优先忠实于已发表公式，使用 raw immediate reward。

本实现提供两个显式、非论文默认的长度聚合选项：

- `trajectory_token_mean`：每条 trajectory 内只对有效 action token 求平均，再对 trajectories 求平均。它让不同有效长度的 trajectory 总权重相同，并与 RLinf 的长度归一化 OPD reduction 对齐。
- `global_token_mean`：对整个 batch 的全部有效 token 求平均。它只改变整体梯度尺度；长 trajectory 因 token 更多，仍有更大的总权重。

两者都必须在实验报告中标注为非论文默认。这里的 token mean 以 `token_mask=True` 为分母；当每个 step 的 action-token 数固定时，`trajectory_token_mean` 等价于按有效 action/step 数平均。

## 4. 这不是 GRPO/PPO

论文先介绍 GRPO 用作对比，然后在 Eq. (7) 后明确说明 VLA-OPD 直接使用 raw reverse-KL reward，不进行 outcome reward normalization。因此纯 OPD 默认不包含：

- group-relative mean/std normalization；
- 环境二值 success reward 或其他 sparse outcome reward；
- critic/value network；
- GAE、TD target 或 reward-to-go；
- PPO old/current importance ratio、clipped surrogate 或 KL-to-reference penalty；
- SimpleVLA-RL 的 dynamic sampling 筛组规则。

配置中的 `reward_clip` 也不是 PPO ratio clip；默认值为 `null`。论文报告的 “Distill + GRPO” 是先完成 distillation、再运行一个独立 GRPO fine-tuning 阶段。本项目的核心 trainer 只覆盖 “Distill”。

环境 adapter 仍可返回 task reward 供日志或评测使用，但核心 OPD trainer 不把它传入 loss。

## 5. On-policy 一致性

Eq. (6)–(7) 的记号使用同一个当前 policy。工程实现区分：

- `behavior_student_log_probs`：采样时的值，用于 detached reward；
- `current_student_log_probs`：更新前重新计算的可微值，用于 policy loss。

在没有 PPO importance correction 的前提下，两者必须来自同一 policy version，并在数值容差内一致。本项目通过 `algorithm.on_policy_logprob_tolerance` fail fast；真实 adapter 不应跨多个更新 replay 旧 trajectory，也不应在采样与打分之间改变 temperature、action mask 或 normalization。

## 6. VLA/action 接口约束

### Observation

每个 step 必须保留 teacher/student 可共同重放的完整输入：相机图像及顺序、instruction、proprioception、必要历史、预处理信息、episode/step ID，以及终止/截断信息。teacher scoring 时重新向环境请求“当前帧”会得到不同物理状态，违反论文条件。

### Action

每个 student sample 应同时包含：

- categorical `action_token_ids`；
- rollout 时每个 token 的 behavior log-probability；
- token/chunk valid mask；
- 实际传给环境的 decoded `env_action`；
- 若 chunk 提前终止，哪些 token 真正执行过。

student 与 teacher 必须共享 action vocabulary/support、token ID、量化 bins、chunk 展平顺序、控制坐标系、单位、action bounds、normalization statistics 和 `unnorm_key`。张量 shape 一样不足以证明可比较。

### 为什么不能直接接官方 continuous L1 OpenVLA-OFT

VLA-OPD 的实验 teacher 是 SimpleVLA-RL；论文算法要求 teacher/student action-token logits。SimpleVLA-RL 的[实验设置](https://arxiv.org/html/2509.09674)明确说明，其 OpenVLA-OFT 是修改版：保留 parallel decoding/action chunking，但使用 LLaMA2 categorical output head 生成 action tokens，并用 cross-entropy 训练；官方 OpenVLA-OFT 则是 MLP continuous action head 和 L1 regression，而且两者 checkpoint 不能直接互换。

因此，本项目的 faithful adapter 目标是 categorical action-token policy。continuous deterministic L1 输出没有 $\log\pi(a\mid s)$，不能代入 Eq. (6)。若以后接 Gaussian、flow 或 diffusion policy，必须先推导并实现其规范化 density/score divergence，作为单独算法分支。

## 7. 论文明确报告的实验设置

| 项目 | 论文公开内容 |
| --- | --- |
| Benchmarks | LIBERO 四个 suite；RoboTwin2.0 四个双臂任务 |
| Teacher | 冻结 SimpleVLA-RL performance oracle |
| Student backbone | 论文称 OpenVLA-OFT；其 action-logit 语义对应 SimpleVLA-RL 的 categorical 修改版 |
| LIBERO student init | 每 task 1 条 demonstration 的 SFT |
| RoboTwin2.0 student init | 每 task 1,000 条 trajectory 的单任务 SFT |
| Main batch size | `64` |
| Main group size | `G = 8` |
| Group-size ablation | 固定 batch `32`，比较 `G = 2, 4, 8` |
| Objective ablation | reverse-KL、forward-KL、hard-CE |
| 训练方式 | “Distill” 与后续独立的 “Distill + GRPO” |

Algorithm 1 以 $B\times G$ 聚合，因而把 batch size 解释为 prompt batch、总 trajectory 数为 $BG$ 是自然解释；但正文只写 “batch size 64”，且作者代码未公开，所以本项目的 `prompt_batch_size: 64` 是显式实现选择，不是假装已经核验的事实。

## 8. 未披露或不足以核验的内容

VLA-OPD 论文未给出完整的：

- OPD learning rate、optimizer、betas、weight decay 和 scheduler；
- distillation update 数、每个 rollout batch 的 optimizer epoch/复用次数；
- 训练时 student/teacher temperature、top-k/top-p 和精确 action mask；
- LIBERO/RoboTwin2.0 的所有 horizon、termination 和 truncation 细节；
- gradient clipping、precision、gradient accumulation 和分布式组件布局；
- teacher/student 的可下载精确 checkpoints、数据 split 与 preprocessing snapshot；
- 随机种子、评测 episode 清单和足以重建所有图表的日志。

其中部分值可在 SimpleVLA-RL 或第三方 RLinf 配置中找到，但它们不是 VLA-OPD 论文明确报告值，不能静默移植后声称数值复现。`configs/toy.yaml` 中除 `G=8` 外的数值都只服务于 CPU correctness demo。

## 9. 可声明范围

本项目可以声明“论文公式的算法级独立实现”和“toy 环境的闭环正确性验证”。只有在真实 adapter、原始兼容 checkpoint、完整 benchmark 配置和实际实验日志就绪后，才能报告本项目自己的 LIBERO/RoboTwin 结果；即便如此，缺少作者配置时也应称独立复现实验，而不是官方数值复现。

