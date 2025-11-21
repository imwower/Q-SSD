# Q-SSD: Quantized State Space Dual Architecture (量子化状态空间对偶架构)

Q-SSD 是一种新型神经网络架构，旨在解决后摩尔定律时代 AI 计算的算力与存储瓶颈。它继承了脉冲神经网络（SNN）的线性时间复杂度和状态记忆特性，同时采用对现代 GPU/TPU 友好的密集张量运算与 1.58-bit 极值量化。项目目标是提供一种“超越神经形态计算的硅基融合范式”，通过抽取 SNN 的计算本质，实现 $O(N)$ 推理复杂度、$O(1)$ 显存占用，以及类突触的整数加法计算。

## 目录 (Table of Contents)
- [背景与动机 (Background)](#背景与动机-background)
- [核心特性 (Key Features)](#核心特性-key-features)
- [架构概览 (Architecture)](#架构概览-architecture)
- [性能对比 (Performance)](#性能对比-performance)
- [安装指南 (Installation)](#安装指南-installation)
- [使用说明 (Usage)](#使用说明-usage)
- [Roadmap](#roadmap)
- [引用 (References)](#引用-references)
- [许可证 (License)](#许可证-license)

## 背景与动机 (Background)
- Transformer 瓶颈：推理复杂度为 $O(N^2)$，KV Cache 使显存随序列长度线性增长。
- SNN 困境：稀疏、异步、条件分支（if/else）导致 Warp Divergence 与非合并内存访问，难以在 GPU 上高效运行。
- Q-SSD 方案：摒弃 LIF 神经元，采用选择性状态空间模型（Selective SSMs）并引入 BitNet b1.58，将 SNN 的“状态压缩”优势与 GPU Tensor Core 的密集运算效率结合。

## 核心特性 (Key Features)
- ⚡️ 线性推理复杂度：推理时间 $O(N)$，并行扫描算法使训练具备 $O(\log N)$ 并行效率。
- 💾 恒定显存占用：彻底消除 KV Cache，显存占用与序列长度无关，仅取决于模型参数。
- 🔋 1.58-bit 权重 (BitNet)：权重约束为 $\{-1, 0, +1\}$，矩阵乘法退化为浮点加减法，显著降低算术能耗。
- 👁️ Event2Vec 嵌入：针对 DVS 事件流的向量化嵌入层，缓解稀疏事件与密集计算单元的阻抗匹配问题。
- 🏗️ 硬件原生：为 SIMT 架构设计，无 Warp Divergence，无随机内存访问。

## 架构概览 (Architecture)
Q-SSD 由堆叠的 Quantized State Space Block 构成，每个 Block 包含如下组件：

### Quantized State Space Mixer
- 替代 Transformer Self-Attention，负责时间维度混合。
- BitLinear 投影：将输入 $x$ 投影为 $z, B, C$ 等分量，仅涉及整数加减。
- Short Conv：一维短卷积，保留离散化过程中的高频局部信息。
- SSM Core（FP16/BF16）：递归计算 $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$，为数值稳定性保留高精度。
- Gate & Output：SiLU 门控与 BitLinear 输出投影。

### Quantized Channel Mixer
- 替代 Transformer FFN。
- 结构：GLU 变体（SwiGLU/GeGLU）实现通道混合。
- 扩展与压缩：使用 1.58-bit BitLinear 将维度扩展至 $4d$ 再压缩回 $d$。

### 量子化策略 (Quantization Strategy)
- 权重：使用 Absmean 量化将权重映射至 $\{-1, 0, 1\}$。
- 激活：采用平滑梯度补偿、旋转变换（Rotation）与分布对齐，缓解 Mamba 架构中的激活值离群点。

### Event2Vec 输入层
- 面向异步事件流 $(x, y, t, p)$。
- 通过参数化空间嵌入与时间卷积嵌入，将事件流映射为密集向量序列 $E$，实现“稀疏输入 → 密集计算”。

## 性能对比 (Performance)
| 特性 | Transformer (LLM) | SNN (LIF) | Q-SSD (Proposed) |
| :--- | :---------------- | :-------- | :--------------- |
| 时间复杂度 | $O(N^2)$ | $O(1)$ | $O(1)$ |
| 显存增长 | 线性（KV Cache） | 恒定 | 恒定 |
| 计算范式 | FP16 乘累加 | 稀疏累加（GPU 效率低） | Int8/1.58-bit 加法（GPU 效率高） |
| 能效（相对） | ~1.1 pJ（FP16 Mult） | ~0.x pJ（理论值） | ~0.03 pJ（Int Add） |

## 安装指南 (Installation)
环境要求：
- Python 3.10+
- PyTorch 2.0+（推荐 CUDA 支持）
- Triton（用于优化 Kernel）
- mamba-ssm / causal-conv1d

```bash
# 克隆仓库
git clone https://github.com/yourusername/Q-SSD.git
cd Q-SSD

# 创建虚拟环境
conda create -n qssd python=3.10
conda activate qssd

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "causal-conv1d>=1.2.0"
pip install "mamba-ssm>=1.2.0"
pip install -r requirements.txt
```

## 使用说明 (Usage)

### 1. 模型定义 (Model Definition)
```python
from qssd.models import QSSDModel
from qssd.config import QSSDConfig

# 初始化配置（类似 Mamba + BitNet）
config = QSSDConfig(
    d_model=512,
    n_layer=12,
    vocab_size=10000,
    ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
    quantization_mode="1.58bit",
)

model = QSSDModel(config).cuda()

# 前向传播
x = torch.randint(0, 10000, (1, 1024)).cuda()
logits = model(x)
print(logits.shape)  # torch.Size([1, 1024, 10000])
```

### 2. 处理神经形态数据 (Event Processing)
```python
from qssd.layers import Event2Vec

# 模拟 DVS 事件流 (Batch, Time, H, W, Polarity)
events = torch.randn(1, 100, 64, 64, 2).cuda()

# 向量化嵌入
e2v = Event2Vec(resolution=(64, 64), dim=512)
embeddings = e2v(events)

# 传入主网络
output = model(embeddings)
```

## Roadmap
- [ ] Phase 1: 核心模块实现（BitLinear, Q-SSM Block）
- [ ] Phase 2: Event2Vec 嵌入层实现与 DVS 数据集适配
- [ ] Phase 3: 在 CUDA 上实现优化的 1.58-bit Kernel（Triton）
- [ ] Phase 4: 在 ImageNet/CIFAR 和 NLP 数据集上进行预训练验证

## 引用 (References)
- Mamba: Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. ArXiv.
- BitNet b1.58: Ma, S., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. ArXiv.
- Bi-Mamba: Towards Accurate 1-Bit State Space Models. ArXiv.
- Event2Vec: Processing Neuromorphic Events directly by Representations in Vector Space. ArXiv.

## 许可证 (License)
This project is licensed under the MIT License.
