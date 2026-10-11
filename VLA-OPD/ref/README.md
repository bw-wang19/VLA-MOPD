# VLA-OPD 参考资料

本目录只维护来源索引，不提交论文 PDF、大模型权重、benchmark 数据或第三方代码副本。

## 一手资料

- VLA-OPD 论文：[arXiv abstract](https://arxiv.org/abs/2603.26666)、[HTML](https://arxiv.org/html/2603.26666)、[PDF](https://arxiv.org/pdf/2603.26666)
- VLA-OPD 项目页：[IRPN Lab project page](https://irpn-lab.github.io/VLA-OPD/)
- VLA-OPD 展示仓库：[IRPN-LAB/VLA-OPD](https://github.com/IRPN-LAB/VLA-OPD)
- SimpleVLA-RL 论文与实现：[arXiv](https://arxiv.org/abs/2509.09674)、[PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
- OpenVLA-OFT 论文与实现：[project page](https://openvla-oft.github.io/)、[arXiv](https://arxiv.org/abs/2502.19645)、[moojink/openvla-oft](https://github.com/moojink/openvla-oft)
- LIBERO：[Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- RoboTwin：[RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)

截至 2026-10-11，VLA-OPD 项目页的 Code 入口仍显示 **Coming Soon**；公开展示仓库没有论文训练实现、完整配置、checkpoint 或数据处理代码。因此本子项目以论文公式为规范，不能把伪代码之外的工程选择归因于作者。

## 第三方交叉参考

- RLinf OPD 文档：[OPD for OpenVLA-OFT on LIBERO](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/opd.html)
- RLinf 仓库：[RLinf/RLinf](https://github.com/RLinf/RLinf)

RLinf 是独立第三方框架，不是 VLA-OPD 作者公开的官方实现。它对于核对以下工程语义很有帮助：student 在环境中 rollout、teacher 只给保存的 student action token 打分、teacher/student 使用一致的 `unnorm_key`。但其公开 recipe 采用自己的配置与 token-mean loss 聚合；这些值不能用来填补 VLA-OPD 论文未披露的超参数，也不能证明作者内部代码采用相同行为。

## 阅读顺序

1. 先读 VLA-OPD §3 和 Algorithm 1，确认三阶段及 Eq. (5)–(7)。
2. 再读 VLA-OPD §4.1–4.4，区分主实验、`Distill + GRPO` 和消融。
3. 读 SimpleVLA-RL §4.1，确认实验中的 OpenVLA-OFT 是 categorical action-token 修改版，而不是官方 continuous MLP/L1 head。
4. 最后把 RLinf 当作第三方工程参考，不把其实现细节标成原论文事实。

