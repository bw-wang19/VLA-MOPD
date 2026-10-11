# VLA-OPD

这是 [VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation](https://arxiv.org/abs/2603.26666) 的**独立算法级实现**。它位于 `VLA-MOPD/VLA-OPD/`，拥有自己的 Python 包、配置、脚本、测试和依赖，不导入上层 `vla_mopd`，也不与多教师 MOPD 的训练逻辑混用。

截至 2026-10-11，论文项目页的官方代码仍标为 **Coming Soon**；因此本项目根据论文 Eq. (5)–(7) 和 Algorithm 1 实现可测试的最小闭环，不声称是作者代码，也不声称已经复现论文的 LIBERO/RoboTwin2.0 数值。第三方 [RLinf OPD 实现](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/opd.html) 只作为工程交叉参考，不用于补写论文未披露的训练细节。

论文忠实性、公式语义和公开信息缺口见 [docs/paper_fidelity.md](docs/paper_fidelity.md)，一手资料索引见 [ref/README.md](ref/README.md)。

## 实现的算法

VLA-OPD 反复执行三个阶段：

1. **Student sampling**：student 在环境中执行自己采样的 action token；每个 prompt 独立采样 `G` 条 trajectory。
2. **Teacher labeling**：冻结 teacher 不控制环境，只在 student 实际访问的观测上，为 student 实际采样/执行的 action token 计算 log-probability。
3. **Student optimization**：以逐 token 的 negative reverse-KL 样本值作为 raw、immediate policy-gradient reward 更新 student。

论文 Eq. (5) 的目标为

\[
\max_\theta\;\mathcal J(\theta)
=\mathbb E_{s\sim\pi_\theta}
\left[-D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid s)\,\|\,\pi_{tea}(\cdot\mid s)
\right)\right].
\]

对 student 采样的 action，Eq. (6) 定义

\[
r_t^{\mathrm{OPD}}
=\log\pi_{tea}(a_t\mid s_t)-\log\pi_{beh}(a_t\mid s_t),
\]

其中形成 reward 的 log-probability 必须 `detach`。Eq. (7) 使用

\[
\frac{1}{G}\sum_{i=1}^{G}\sum_t
\nabla_\theta\log\pi_\theta(a_{t,i}\mid s_{t,i})
\,r_{t,i}^{\mathrm{OPD}}.
\]

本实现相应使用下降损失

\[
\mathcal L=-\operatorname{mean}_{trajectory}
\sum_{t\in valid}
\log\pi_\theta(a_t\mid s_t)\,
\operatorname{sg}[r_t^{\mathrm{OPD}}].
\]

`algorithm.reduction` 可以选择三种聚合方式：

- `trajectory_sum`（默认）：每条 trajectory 内对有效 action token 求和，再对 trajectories 求平均。它直接对应 Eq. (7) 的 `sum_t`，但长 trajectory 的总权重更大。
- `trajectory_token_mean`：每条 trajectory 内先对有效 action token 求平均，再对 trajectories 求平均。每条 trajectory 总权重相同，适合混合不同有效长度的样本，也是与 RLinf 长度归一化方式对齐的选项。
- `global_token_mean`：直接对整个 batch 的所有有效 action token 求平均。每个 token 等权，但长 trajectory 包含更多 token，因此其总权重仍然更大。

例如，希望长度为 100 和 10 的两条 trajectory 对 batch loss 具有相同总权重时，使用：

```yaml
algorithm:
  reduction: trajectory_token_mean
```

这里的 “action-token mean” 只统计 `token_mask=True` 的 categorical action token；padding、未执行的 action chunk token 和终止后的 token 不进入分子或分母。当每个环境 step 的有效 action-token 数固定时，它等价于按有效 action/step 数进行长度平均。默认仍保留 `trajectory_sum`，因为论文没有说明作者代码是否在 Eq. (7) 之外又做了长度归一化。

这里的 `G` 只是降低 Monte Carlo 方差的多 trajectory 平均。纯 VLA-OPD **没有** GRPO group-relative reward normalization、critic、GAE、PPO importance-ratio clipping，也不使用环境的稀疏成功奖励。论文的 “Distill + GRPO” 是 OPD 完成后的额外 RL 阶段，不属于本项目默认 loss。

## 项目结构

```text
VLA-OPD/
├── vla_opd/
│   ├── types.py          # prompt、观测、action、trajectory 与标签类型
│   ├── protocols.py      # environment/student/teacher 接口契约
│   ├── rollout.py        # student 的 G 路 on-policy 环境采样
│   ├── labeling.py       # teacher 对 student action 做 forced scoring
│   ├── losses.py         # Eq. (6)–(7) sampled reverse-KL PG loss
│   ├── trainer.py        # 三阶段训练循环
│   ├── config.py         # 严格 YAML 配置与论文默认检查
│   ├── runtime.py        # 真实 VLA/环境的 factory 边界
│   ├── toy.py            # 不下载权重的 CPU correctness demo
│   └── cli.py            # 配置检查和训练入口
├── configs/toy.yaml
├── scripts/
├── tests/
├── docs/paper_fidelity.md
└── ref/README.md
```

## 安装与验证

建议使用独立环境；只需要 PyTorch 和 PyYAML 即可运行核心实现：

```bash
cd /home/wbw/workspace/VLA-MOPD/VLA-OPD
python -m pip install -e ".[test]"
```

检查配置：

```bash
python -m vla_opd.cli --check-config configs/toy.yaml
```

运行测试：

```bash
bash scripts/check.sh
```

运行 CPU toy 闭环：

```bash
bash scripts/train_toy.sh
# 等价于：python -m vla_opd.cli configs/toy.yaml
```

`toy.yaml` 只验证 rollout → teacher labeling → reverse-KL update 的接口、mask、梯度和保存流程。它不是机器人 benchmark，也不能用其 loss 或 reward 数字与论文表格比较。配置中的 tiny batch/horizon、学习率、优化器设置和训练步数都是本地 smoke-test 值；只有 `group_size: 8` 与论文主实验公开值相同。

## 接入真实 VLA 与环境

把 `runtime.factory` 改为一个 `module.path:callable_name`。factory 接收完整 `config` 及 `runtime.kwargs`，返回 `vla_opd.runtime.ExperimentComponents`：

- `student`：同时是 `torch.nn.Module` 和 `StudentPolicy`；`sample_action` 负责单步随机采样，`score_actions` 对已保存 action token 做可微批量重打分。
- `teacher`：实现 `TeacherPolicy.score_actions`，冻结参数，并对相同 observation/history、相同 student action token 返回 log-probability。
- `env_factory(*, prompt, prompt_index, group_index)`：为每条 trajectory 创建独立环境；环境实现 `reset(prompt, seed=...)`、`step(env_action)` 和 `close()`。
- `prompts`：可被训练循环采样的 prompt/task 集合。

核心入口也可以单独使用：

```python
from vla_opd.labeling import label_with_teacher
from vla_opd.losses import sampled_reverse_kl_pg_loss
from vla_opd.rollout import GroupRolloutCollector
```

真实 adapter 必须满足以下约束：

- 保存 rollout 当时的不可变图像、语言指令、proprioception 和必要历史；teacher labeling 时不能重新读取已经变化的环境状态。
- 保存 student 实际执行的 categorical action token、对应 `env_action`、valid mask 和 behavior log-probability；teacher 不能用自己的 argmax action 替换它。
- student 与 teacher 必须共享可比较的 action vocabulary、token ID、量化边界、action-chunk 布局、坐标系、单位、normalization statistics 和 `unnorm_key`。
- 收集后必须在 policy 仍与 behavior policy 一致时更新。默认 tolerance 会检查 current/behavior log-probability；本实现没有 PPO ratio 来修正 stale rollout。
- `score_batch_size` 只控制 teacher/student forced-scoring 的工程 micro-batch，不改变 `B×G` 的算法聚合；真实大模型应按显存调整。
- action chunk 中未执行的 token、padding，以及 episode 终止后的 token 必须 mask 掉。

论文实验并非直接使用官方 OpenVLA-OFT 的 continuous MLP/L1 action head。它以 SimpleVLA-RL 为 teacher/实验基础；该实现将 OpenVLA-OFT 改为 LLaMA2 categorical output head，以 action token 和 cross-entropy 训练。Eq. (6) 需要 sampled action 的 log-probability，所以原版 continuous deterministic L1 head 不能直接接入；若要支持连续策略，必须先定义规范化且可计算的 action density，不能把 L1 error 冒充 log-probability。

## 公开实验信息与复现边界

论文主实验明确报告 `batch size = 64`、`G = 8`；消融固定 batch size 32 并比较 `G ∈ {2,4,8}`。论文没有完整披露足以数值复现的 optimizer、learning rate、scheduler、OPD 更新次数/每批复用次数、训练 sampling temperature、精确 horizon/truncation、分布式布局、checkpoint 和全部预处理细节。“batch size 64”是否严格表示 64 个 prompt、从而总计 `64 × 8` 条 trajectory，也没有官方代码可核验；本配置将其解释为 prompt batch size，并在文档中显式保留这一歧义。

因此本项目可以声称：

- 实现并测试论文 Eq. (5)–(7) 的算法语义；
- 提供可替换的 VLA、teacher 和 environment 接口；
- 运行 CPU toy correctness demo。

但不能声称：

- 已重建作者的未公开训练代码或超参数；
- 已复现 LIBERO/RoboTwin2.0 的成功率、收敛速度或遗忘曲线；
- toy 结果能够验证论文的经验结论。

