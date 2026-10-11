# VLA-MOPD

这里先落地了经典 MOPD 的算法级复现：student 在自己的 rollout 上生成 token，样本按已知 domain label 硬路由到对应的冻结 teacher，teacher 对完全相同的 student prefix 做 forced scoring，再用逐 token reverse-KL 信号更新一个共享 student。

当前实现以 Ma et al. 的 [MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://arxiv.org/abs/2606.30406) Eq. (1)–(5) 为规范，提供：

- sampled-token policy-gradient 目标，`A_t = stopgrad(log p_teacher - log p_student)`，双侧 `[-5, 5]` 裁剪；
- corrected teacher-top-k 目标，保留论文 Eq. (5) 的 `-p_student + p_teacher` 校正项；
- `domain -> teacher checkpoint` 的 oracle hard routing；
- 每条 trajectory 先按有效 response token 求均值，再在 batch 中求均值；
- JSON/JSONL/Parquet/Hugging Face `save_to_disk` 数据加载与确定性 domain-ratio 混合；
- checkpoint 续训、W&B 可选记录、per-domain token share/KL proxy/entropy/clip 指标；
- 面向 autoregressive action-token VLA 的损失和路由接口。

这不是 Qwen3-30B-A3B 原表格的数值复现。经典论文没有公开完整 Stage-1 SFT checkpoint、三个 Stage-2 RL teacher、精确数据配方和全部 Stage-3 训练参数。可复现边界、公式验收条件及 VLA 扩展契约见 [docs/reproduction_scope.md](docs/reproduction_scope.md)。

## 项目结构

结构沿用 CPVM 的“代码—配置—脚本—参考资料—测试”分层，同时避开仓库名中连字符不能作为 Python 包名的问题：

```text
VLA-MOPD/
├── vla_mopd/code/          # MOPD loss、token 对齐、数据、teacher 路由、训练入口
├── config/                 # PG/top-k YAML 与 Accelerate 配置
├── scripts/                # 一键训练和完整检查
├── tests/                  # 不依赖 GPU/权重的公式与路由单测
├── docs/                   # 复现范围、VLA 接口约束
└── ref/                    # 论文和公开实现索引（不存大权重）
```

大模型、数据和输出应放在独立数据盘，不要放进仓库。

## 环境

建议使用 Python 3.11：

```bash
cd /home/wbw/workspace/VLA-MOPD
conda create -n vla-mopd python=3.11 -y
conda activate vla-mopd
python -m pip install -e .
```

若希望复用本次验收过的 Python 顶层版本（PyTorch 的 CUDA wheel source 仍需按机器选择）：

```bash
python -m pip install -e . -c constraints-tested.txt
```

需要 W&B 时安装：

```bash
python -m pip install -r requirements_wandb.txt
```

当前 CPU 单测也已在工作区现有的 `cpvm-cu132` 环境（PyTorch 2.12.1+cu132、Transformers 5.12.1、Accelerate 1.14.0）验证。`requirements.txt` 给出兼容范围，`constraints-tested.txt` 只锁本次验收的顶层 Python 包，二者都不是原论文环境；正式实验还应保存完整 `pip freeze` 和 CUDA/NCCL 版本。

## 数据格式

每个 domain 在 YAML 中对应一个独立文件；domain 标签由配置写入并覆盖文件内的旧标签，避免错误路由。每条记录必须至少提供 `prompt` 或 `messages` 之一；两者同时存在时以 `messages` 为准。

纯文本：

```json
{"prompt": "Solve x^2 - 5x + 6 = 0."}
```

Chat 格式：

```json
{"messages": [{"role": "system", "content": "You are a math assistant."}, {"role": "user", "content": "Solve x^2 - 5x + 6 = 0."}]}
```

支持 `.json`、`.jsonl`、`.ndjson`、`.parquet`，也支持 `datasets.Dataset.save_to_disk()` 目录。所有 teacher checkpoint 都必须包含可由 `AutoTokenizer.from_pretrained()` 加载的 tokenizer 文件。teacher 和 student 必须共享 tokenizer/action vocabulary；代码会校验完整 token→ID 映射、关键 special-token ID、chat template 和模型 vocab size，但 checkpoint 是否确实从同一个 Stage-1 分叉仍需由实验元数据保证。

为同时保证 raw-policy rollout 与可微打分严格一致、并在打分阶段启用 gradient checkpointing，student 不得含非零 dropout、BatchNorm 或 stochastic-depth 一类 train/eval 行为不同的层；加载时会 fail fast。Qwen/Llama 类后训练模型通常本来就使用零 dropout。

## 配置

先复制并修改：

- [config/mopd_pg.yaml](config/mopd_pg.yaml)：论文 Eq. (3)–(4)，推荐作为经典 baseline；
- [config/mopd_topk.yaml](config/mopd_topk.yaml)：论文 corrected top-k Eq. (5)，默认 `k=64`。

检查 YAML（不会加载数据或模型）：

```bash
python -m vla_mopd.code.train_mopd \
  --check-config config/mopd_pg.yaml
```

论文 Stage 3 报告的全局 batch size 是 `2048`、每 prompt rollout 数 `N=1`、Math:IF:SWE 比例为 `0.35:0.35:0.30`。本实现的有效全局 batch 为：

```text
per_device_train_batch_size × world_size × gradient_accumulation_steps
```

论文没有报告 Stage-3 learning rate 和完整 optimizer/scheduler，示例 YAML 中的这些值只是本实现默认值，最终 resolved config 会写入实验目录。

当前数据 mixer 对整个有限训练集执行确定性的最大余数配额，因此全数据窗口的 domain 数量严格接近配置比例；它**不保证每个 optimizer update** 都恰好是论文的 `0.35:0.35:0.30`。小 batch 时应从日志里的 per-domain `trajectory_share`/`token_share` 检查波动；若实验要求逐 update 固定配比，需要另接 distributed balanced-batch sampler。这个差异不影响 Eq. (1)–(5) 的公式复现，但属于论文训练配方尚未完全重建的部分。

## 训练

单进程：

```bash
bash scripts/train_mopd.sh config/mopd_pg.yaml
```

两进程：

```bash
NUM_PROCESSES=2 \
bash scripts/train_mopd.sh config/mopd_pg.yaml
```

也可以直接使用 Accelerate：

```bash
accelerate launch \
  --config_file config/accelerate_2gpu.yaml \
  --module vla_mopd.code.train_mopd \
  config/mopd_pg.yaml
```

续训时将 `resume_from_checkpoint` 设为具体 `checkpoint-N` 路径，或设为 `latest` 自动选择当前 run 目录中同时带有 `resume_contract.json` 和最终完成标记 `trainer_state.json` 的最大 step。已有产物的同一 run 只允许从其最新完整 checkpoint 恢复；从更早 checkpoint 分叉时必须使用新的 `run_name`。续训会根据 checkpoint 自身的契约校验模型、数据、算法、world size、optimizer/batch 配置及本地输入资产身份，保留首次运行元数据，并把续训事件追加到 `resume_history.jsonl`。

这是“同一训练计划的精确恢复”，不是任意 warm start：`max_steps` 参与 cosine schedule，续训时必须保持不变，不能靠修改它延长训练。日志频率、保存频率和上报方式可以调整；若要改变总步数或算法配置，应建立新实验并明确采用新的调度计划。身份清单对不超过 1 MiB 的本地文件记录 SHA-256，对大权重分片记录路径、大小与 mtime；多进程启动还会比较各 rank 的清单摘要。远程 Hub 名称只能记录 locator，因此正式实验应使用所有节点共享的不可变本地 snapshot，并额外保存完整权重/数据版本信息。

输出结构：

```text
<output_dir>/mopd/<pg|topk>/<run_name>/
├── resolved_config.yaml
├── run_metadata.json
├── resume_history.jsonl   # 仅发生续训时创建
├── checkpoint-N/
│   ├── resume_contract.json
│   ├── trainer_state.json  # 最后原子写入的完整性标记
│   └── ...                 # Accelerate optimizer/scheduler/RNG/model state
└── final/
```

当前 `LocalTeacherPool` 会在每个 worker 中常驻所有 teacher，适合小模型、公式验证和研究原型；经典论文的大模型训练使用独立异步 teacher prefill service。若模型无法同时驻留，先把 `teacher_device` 设为 `cpu` 做正确性 smoke test，论文规模则应实现/接入异步服务后端，而不是牺牲 hard-routing 或 token 对齐语义。

## 验证

运行配置检查和全部 CPU 单测：

```bash
bash scripts/check_reproduction.sh
```

再运行一次不下载模型、不使用 GPU 的端到端 smoke test（PG 与 corrected top-k 各一 step，并验证 PG checkpoint 续载）：

```bash
CUDA_VISIBLE_DEVICES="" python tests/smoke_train.py
```

测试覆盖：

- PG advantage 的符号、stop-gradient、双侧 clip 与长度归一化；
- corrected top-k 在 teacher=student 时 loss/gradient 为零；
- `k=vocab` 时 Eq. (5) 等于完整 reverse KL；
- 变长 response 的 causal-logit 对齐；
- 混合 batch 的 hard routing 与原顺序恢复；
- 配置 fail-fast、数据格式和确定性 domain ratio。

## 向 VLA 扩展

当前训练入口是 causal-LM 复现，不把普通多模态 VLM 冒充成机器人 VLA。接入 OpenVLA-OFT 一类 autoregressive action-token policy 时，需要替换三层工程适配，而核心损失无需改写：

1. student rollout 从文本 `generate()` 换成环境交互，保存不可变视觉/proprioception state 和实际执行的 action token；
2. teacher backend 在同一 student-induced state/history 上，对同一 action token forced-score；
3. 数据与模型加载器校验 action tokenizer、chunk layout、坐标系、normalization/`unnorm_key` 和有效 action mask。

VLA 中仍必须保持 same-origin teacher、trajectory 级 teacher 固定、hard routing、有效 action mask 和 per-domain success/KL/entropy/token-share 记录。连续 flow/diffusion policy 没有可直接比较的逐 token log-prob，不能未经推导直接套用这里的 Eq. (1)–(5)。完整契约见 [docs/reproduction_scope.md](docs/reproduction_scope.md)。

## 参考资料

论文谱系、Open-MOPD、NeMo RL、VLA-OPD 与 RLinf 的一手链接集中在 [ref/README.md](ref/README.md)。
