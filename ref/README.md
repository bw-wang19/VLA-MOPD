# MOPD / VLA-OPD 权威资料索引

本目录记录本项目采用的论文谱系与实现参考。项目中的“经典 MOPD”特指 **Multi-Teacher On-Policy Distillation**，而不是名称相近的 multi-rollout OPD。

## 结论与时间线

1. **最早公开提出 MOPD 范式**：小米 `MiMo-V2-Flash Technical Report` 在 2026 年 1 月公开了三阶段流程及训练—推理重要性采样、可选 outcome reward 的技术公式。
2. **经典、正式命名的 MOPD 论文**：Ma et al. 在 2026 年 6 月的 `MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training` 中系统化了方法、给出 Eq. (1)–(5)、受控对比实验与关键消融。本项目以该文的 Eq. (1)–(5) 作为算法复现主规范。
3. **首个端到端公开配方**：2026 年 8 月的 `Open-MOPD` 发布了代码、数据、模型与评测流程，但它是后续独立研究，不是经典论文的官方代码，也不是经典论文原实验资产的公开版本。
4. **VLA 直接前作**：`VLA-OPD` 将单教师 OPD 用于 VLA 的环境交互轨迹；它不是 multi-teacher。`VLA-MOPD` 的目标是把经典 MOPD 的多教师硬路由与 VLA-OPD 的 student-induced state rollout 结合起来。

## 一手来源

| 资料 | 定位 | 链接 |
| --- | --- | --- |
| **MiMo-V2-Flash Technical Report** | MOPD 最早公开来源；见 §4.1、§4.4 | [arXiv](https://arxiv.org/abs/2601.02780) · [HTML](https://arxiv.org/html/2601.02780) · [官方仓库](https://github.com/XiaomiMiMo/MiMo-V2-Flash) |
| **MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training** | 经典 MOPD 主论文；本项目 Eq. (1)–(5) 的规范来源 | [arXiv](https://arxiv.org/abs/2606.30406) · [HTML](https://arxiv.org/html/2606.30406) |
| **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes** | OPD 基础工作；解释 student rollout、teacher scoring 的基本范式 | [arXiv](https://arxiv.org/abs/2306.13649) |
| **MiniLLM: Knowledge Distillation of Large Language Models** | reverse-KL / policy-gradient 蒸馏基础 | [arXiv](https://arxiv.org/abs/2306.08543) · [官方代码](https://github.com/microsoft/LMOps/tree/main/minillm) |

## 可复现实现与后续配方

| 资料 | 可借鉴内容 | 使用限制 |
| --- | --- | --- |
| **Open-MOPD** | mixed-domain SFT → 三个 domain RL teacher → hard-routed M-OPD；提供 patched veRL、启动脚本、SmolLM3-3B 模型及数据 | 后续论文使用自己的 top-k 与优化预算平衡设计；不可称为经典 MOPD 的官方实现 |
|  | [论文](https://arxiv.org/abs/2608.19098) · [项目页](https://bytedtsinghua-sia.github.io/Open-MOPD/) · [GitHub](https://github.com/BytedTsinghua-SIA/Open-MOPD) · [模型与数据](https://huggingface.co/collections/BytedTsinghua-SIA/open-mopd-multi-teacher-on-policy-distillation) | |
| **NVIDIA NeMo RL — MOPD** | async GRPO、独立 teacher worker、sampled-token advantage 与 full-vocabulary reverse-KL 的工程实现 | NVIDIA 的后续实现；配置和扩展目标不等同于经典论文原实验 |
|  | [算法文档](https://docs.nvidia.com/nemo/rl/nightly/about/algorithms/mopd.html) · [GitHub 文档源码](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/about/algorithms/mopd.md) | |

## VLA 迁移依据

| 资料 | 定位 | 链接 |
| --- | --- | --- |
| **VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation** | 单教师 VLA OPD：student 在环境中访问状态，teacher 对同一状态下 student 已采样的 action token 给 dense log-prob；使用 reverse-KL policy gradient | [arXiv](https://arxiv.org/abs/2603.26666) · [HTML](https://arxiv.org/html/2603.26666) · [项目页](https://irpn-lab.github.io/VLA-OPD/) |
| **RLinf — OPD for OpenVLA-OFT on LIBERO** | 可运行的 OpenVLA-OFT + LIBERO 单教师 OPD 工程参考；包含 teacher scoring、OPD loss、配置及指标 | [使用文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/opd.html) · [GitHub](https://github.com/RLinf/RLinf) · [OPD loss](https://github.com/RLinf/RLinf/blob/main/rlinf/algorithms/losses.py) |

VLA-OPD 官方项目页当前仍标注代码为 “Coming Soon”。因此，本项目可借鉴其论文定义，但不能把 RLinf 或本仓库实现描述成 VLA-OPD 作者发布的官方代码。

## 本项目引用规则

- 方法定义、公式和经典超参数引用 Ma et al. 的 MOPD 主论文。
- 讨论 MOPD 的首次提出、重要性采样或 outcome reward 组合时引用 MiMo-V2-Flash Technical Report。
- 讨论完整公开训练配方时单独标注 Open-MOPD，不把它与经典论文原实验混同。
- 讨论机器人环境中的 student-induced states、action-token teacher scoring 时引用 VLA-OPD；讨论具体可运行框架时另行引用 RLinf。
- 任何结果都必须标明是“公式级复现”“缩小规模实验”还是“论文报告值”，不得使用“复现论文数值”概括未运行或无法重建的实验。

更精确的公式、超参数、不可复现项和 LLM→VLA 接口约束见 [`docs/reproduction_scope.md`](../docs/reproduction_scope.md)。
