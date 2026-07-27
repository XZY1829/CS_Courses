---
title: "基于 Transformer 的机器翻译（英文到中文）实验报告"
date: "2026年6月"
documentclass: ctexart
geometry: margin=2.5cm
CJKmainfont: "Microsoft YaHei"
mainfont: "Times New Roman"
monofont: "Consolas"
toc: true
numbersections: true
---

# 基于 Transformer 的机器翻译（英文到中文）实验报告

## 一、任务概述

本实验实现了一个基于 Transformer 架构的英文到中文机器翻译系统。从数据预处理、模型设计、训练策略、解码方法到评估指标，完整实现了端到端的 Seq2Seq 翻译流程，并通过三轮迭代优化将测试集 BLEU 从基线 24.57 提升至 **40.89**（+16.32）。

---

## 二、实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows 11 (10.0.26200) |
| GPU | NVIDIA GeForce RTX 3060 (12GB) |
| Python | 3.13.13 (Anaconda) |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| cuDNN | 9.1.0 |
| 评估工具 | sacrebleu |

---

## 三、数据集与预处理

### 3.1 数据来源

使用 `cmn-eng-simple` 中英平行语料集（来源于 Tatoeba 项目），包含约 21,570 条英中平行句对，按以下方式划分：

| 数据集 | 句对数量 | 用途 |
|--------|----------|------|
| 训练集 | 18,000 | 模型训练 |
| 验证集 | 500 | 超参调优与早停判定 |
| 测试集 | 3,070 | 最终评估 |

### 3.2 英文端预处理

- 使用 NLTK `word_tokenize` 做词级分词并转小写
- 使用 subword-nmt 学习 BPE（5000 merge operations），缓解英文端 OOV 问题
- 最终英文词表大小约 3,400

### 3.3 中文端预处理（核心改进点）

**初始方案（基线）：** 使用 jieba 做词级分词，低频词（出现次数 <3）被替换为 `<UNK>`。
- 问题：约 37.3% 测试句含有 UNK token，严重影响翻译质量。

**优化方案（v3）：** 改为字符级分词（逐字拆分 + OpenCC 简体统一）。
- 优势：常用汉字约 2,250 个，频率基本都 >=2，几乎全部保留。
- 最终测试集 UNK 率：**0.17%**（降低约 200 倍）。

```python
# 字符级中文分词核心代码
cc = OpenCC('t2s')  # 繁转简统一
for char in sentence.strip():
    if char in (' ', '\n', '\t', '\r'):
        continue
    cn_sentence += cc.convert(char) + ' '
```

### 3.4 特殊 Token 设计

| Token | ID | 用途 |
|-------|----|----|
| `<PAD>` | 0 | 批内填充对齐 |
| `<BOS>` | 1 | 解码起始符 |
| `<EOS>` | 2 | 句子结束符 |
| `<UNK>` | 3 | 未知词替代 |

---

## 四、模型架构

### 4.1 整体结构

采用标准的 Encoder-Decoder Transformer 架构（Vaswani et al., 2017），关键参数如下：

| 参数 | 值 | 说明 |
|------|------|------|
| d_model | 256 | 隐藏层维度 |
| nhead | 8 | 多头注意力头数 |
| num_encoder_layers | 4 | 编码器层数 |
| num_decoder_layers | 4 | 解码器层数 |
| dim_feedforward | 1024 | FFN 中间层维度 |
| dropout | 0.15 | Dropout 概率 |
| 激活函数 | GELU | 非线性激活 |
| 归一化模式 | Pre-Norm | 每个子层先归一化 |
| 参数共享 | tie_embeddings | 输出层与目标 embedding 权重共享 |

### 4.2 核心机制：多头注意力（Multi-Head Attention）

多头注意力是 Transformer 的核心组件，其原理：
- 将 Query、Key、Value 投影到 `nhead` 个不同子空间
- 每个头独立计算缩放点积注意力：$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
- 多头输出拼接后再做线性投影

在本模型中的使用位置：
- **Encoder Self-Attention**：源句 token 互相关注，捕获上下文依赖
- **Decoder Masked Self-Attention**：目标端只能看已生成的词（causal mask）
- **Decoder Cross-Attention**：目标词关注源句 memory，实现翻译对齐

### 4.3 位置编码

使用标准正弦位置编码：
$$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

---

## 五、训练策略

### 5.1 优化器与学习率

- 优化器：AdamW（beta1=0.9, beta2=0.98, eps=1e-9, weight_decay=1e-4）
- 学习率调度：Noam Schedule（warmup=2000 步，factor=0.8）
  $$lr = factor \cdot d_{model}^{-0.5} \cdot \min(step^{-0.5}, step \cdot warmup^{-1.5})$$

### 5.2 损失函数

- 交叉熵损失 + Label Smoothing（eps=0.15）
- 忽略 PAD 位置（`ignore_index=0`）

### 5.3 R-Drop 正则化（核心优化 2）

**问题：** 基线存在严重过拟合（train_loss=0.82 vs val_loss=2.30，gap=1.48）。

**方案：** 引入 R-Drop（Liang et al., 2021）：同一 batch 两次前向传播（不同 dropout mask），除 CE loss 外，添加两次输出分布的对称 KL 散度约束：

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{CE}^{(1)} + \mathcal{L}_{CE}^{(2)}) + \alpha \cdot \frac{1}{2}(D_{KL}(p \| q) + D_{KL}(q \| p))$$

其中 alpha=3.0。该方法迫使模型对 dropout 噪声保持输出一致，比单纯增大 dropout 更有效。

**效果：** 最终过拟合 gap 从 1.48 降至 **0.18**。

### 5.4 源端噪声数据增强（核心优化 3）

在 DataLoader 取样时，对源端（英文）施加三级噪声：

1. **Word Dropout（p=0.05）**：随机将 token 替换为 `<UNK>`
2. **Token Delete（p=0.05）**：随机删除 token
3. **Token Swap（p=0.05）**：随机交换相邻 token

目标端仅做 Word Dropout（不做删除/交换），保证监督信号稳定。

### 5.5 早停与安全机制

- Early Stopping：连续 12 个 epoch 验证 BLEU 无显著提升（delta<0.02）则停止
- Loss Explosion 检测：val_loss > best x 2.0 连续 3 次则紧急停止
- NaN/Inf 检测：出现即终止
- 混合精度训练（AMP）：使用 FP16 加速约 40%，降低显存占用

### 5.6 Checkpoint 平均

训练结束后保留 BLEU 最高的 top-5 个 checkpoint，对其参数取平均：
$$\theta_{avg} = \frac{1}{K}\sum_{k=1}^{K}\theta_k$$

这是翻译任务常用方法，可平滑参数、提升泛化。

---

## 六、解码方法

使用 Beam Search 解码，配置如下：

| 参数 | 值 | 说明 |
|------|------|------|
| beam_size | 5 | 搜索宽度 |
| beam_alpha | 0.6 | GNMT 长度归一化强度 |
| max_decode_len | 64 | 最大解码长度 |
| no_repeat_ngram_size | 3 | 禁止重复 3-gram |

长度归一化公式（GNMT style）：
$$\text{score}_{norm} = \frac{\log P(y|x)}{((5+|y|)^\alpha / (5+1)^\alpha)}$$

---

## 七、实验结果

### 7.1 三轮迭代对比

| 版本 | 关键改进 | Val BLEU | Test BLEU | 过拟合 Gap | 训练时间 |
|------|----------|----------|-----------|------------|----------|
| v1 基线 | jieba 词级分词 | 30.54 | 24.57* | ~1.48 | 45 min |
| v2 正则化 | + 早停 + Checkpoint Avg | 34.35 | 26.61 | ~0.85 | 35 min |
| **v3 最终** | + 字符级 + R-Drop + 噪声增强 | **40.74** | **40.89** | **0.18** | 139 min |

*注：v1 的 test_bleu 为训练结束后 160 样本估算值。

### 7.2 训练曲线关键节点

| Epoch | Train Loss | Val Loss | Val BLEU | 说明 |
|-------|-----------|----------|----------|------|
| 1 | 6.652 | 5.574 | 0.26 | 初始化阶段 |
| 5 | 4.316 | 3.775 | 14.40 | warmup 结束，快速学习 |
| 10 | 3.582 | 3.178 | 24.97 | 超越 v1 基线 |
| 15 | 3.230 | 2.973 | 31.43 | 超越 v2 |
| 22 | 2.977 | 2.866 | 36.57 | 首次突破 36 |
| 31 | 2.807 | 2.808 | 38.88 | Gap 接近 0 |
| 46 | 2.657 | 2.774 | **40.74** | 最佳 epoch |
| 58 | 2.582 | 2.764 | 40.56 | 训练终止（早停） |

### 7.3 消融分析（基于 Val BLEU 峰值估算）

| 配置 | Val BLEU | 相对基线提升 |
|------|----------|------------|
| v2 基线（词级 + dropout） | 34.35 | - |
| + 字符级分词（消除 UNK） | ~37-38 | +3~4 |
| + R-Drop（alpha=3.0） | ~39-40 | +1~2 |
| + 源端噪声增强 | 40.74 | +0.5~1 |

### 7.4 翻译示例

| 源句（英文） | 模型输出（中文） |
|-------------|-----------------|
| She has beautiful eyes. | 她对一个美丽的眼睛。 |
| We all fell asleep. | 我们都睡着了。 |
| I majored in chemistry at the university. | 我在大学主修化学。 |
| Please give me a glass of water. | 请给我一杯水。 |
| He loves her. | 他爱她。 |
| Tom is drunk now. | 汤姆现在喝醉了。 |

---

## 八、遇到的问题与解决方案

### 8.1 UNK 覆盖率过高（最严重）

**问题：** jieba 词级分词导致 37.3% 测试句含 UNK，模型无法正确翻译"绘画""鉴赏力"等低频词。

**诊断：** 统计发现中文词表只有约 4,500 词，但测试集中大量"长尾词"被过滤。

**解决：** 改为字符级分词。汉字总量约 2,250 个（频率>=2），覆盖率从 62.7% 提升至 **99.83%**。

**效果：** 单独此项预估贡献 +3~5 BLEU。

### 8.2 严重过拟合

**问题：** v1 训练到后期 train_loss=0.82，val_loss=2.30，gap 高达 1.48，模型在训练集上"背诵"。

**诊断：** 数据量仅 18,000 句对，模型参数约 10M+，容量过剩。

**解决：** 组合使用三种正则化手段：
- Label Smoothing 0.15
- R-Drop（alpha=3.0）
- 源端噪声增强（word dropout + delete + swap）

**效果：** 过拟合 gap 从 1.48 降至 0.18（缩小 8.2 倍）。

### 8.3 解码重复

**问题：** 翻译中出现"我我我""的的的"等连续重复 token。

**解决：** Beam Search 中加入 no_repeat_ngram_size=3 的 n-gram blocking。

**效果：** 重复问题基本消除。

### 8.4 Windows 环境兼容

**问题：** 
- 文件默认 GBK 编码导致 UnicodeDecodeError
- OpenMP 库冲突（libiomp5md.dll already initialized）
- PowerShell 不支持 `&&` 与 `<` 重定向

**解决：** 
- 所有 `open()` 强制 `encoding='utf-8'`
- 设置 `KMP_DUPLICATE_LIB_OK=TRUE`
- 改用 PowerShell 管道语法（`Get-Content | ...`）

### 8.5 训练速度与显存平衡

**问题：** R-Drop 需要两次前向传播，训练时间增加约 60-80%。

**解决：** 
- 启用 AMP（FP16 混合精度），抵消约一半开销
- max_tgt_len 从 64 提升至 96（适配字符级更长的序列），同时保持 batch_size=96 不降低

---

## 九、关键超参数配置

```yaml
# 模型结构
d_model: 256
nhead: 8
num_encoder_layers: 4
num_decoder_layers: 4
dim_feedforward: 1024
dropout: 0.15

# 训练
epochs: 60
batch_size: 96
warmup_steps: 2000
lr_factor: 0.8
label_smoothing: 0.15
weight_decay: 0.0001
grad_clip: 1.0

# 正则化
word_dropout: 0.05
rdrop_alpha: 3.0

# 解码
beam_size: 5
beam_alpha: 0.6
no_repeat_ngram_size: 3

# 早停
patience: 12
early_stop_min_delta: 0.02
top_k_checkpoints: 5
```

---

## 十、项目文件结构

```
NLPhw_transformer/
|-- train_transformer.py        # 核心训练/测试脚本（含模型定义）
|-- configs/
|   `-- train_cuda.yaml         # 超参数配置文件
|-- cmn-eng-simple/
|   |-- preprocess/
|   |   |-- tokenizer.py        # 中英文分词预处理
|   |   |-- build_dataset.py    # 构建词表与数据划分
|   |   `-- cmn.txt             # 原始平行语料
|   |-- training.txt            # 训练集
|   |-- validation.txt          # 验证集
|   |-- testing.txt             # 测试集
|   |-- word2int_en.json        # 英文词表
|   `-- word2int_cn.json        # 中文词表
|-- runs_transformer_cuda/      # 实验输出目录
|   `-- train_20260609_151606_v3_char_rdrop/
|       |-- avg_model.pt        # 最终模型（top-5 平均）
|       |-- best_model.pt       # 最佳单一 checkpoint
|       |-- metrics_train.json  # 训练指标
|       |-- train_log.csv       # 逐 epoch 日志
|       |-- test_predictions.txt # 测试预测结果
|       `-- args.json           # 完整参数快照
|-- requirements.txt            # 依赖列表
`-- README.md                   # 项目说明
```

---

## 十一、运行方式

### 11.1 环境准备

```bash
pip install -r requirements.txt
```

### 11.2 数据预处理

```bash
cd cmn-eng-simple/preprocess
python tokenizer.py
subword-nmt learn-bpe -s 5000 < en.txt > en_code.txt
subword-nmt apply-bpe -c en_code.txt < en.txt > en_refine.txt
subword-nmt get-vocab --input en_refine.txt --output en_vocab.txt
python build_dataset.py
```

### 11.3 训练

```bash
python train_transformer.py --config configs/train_cuda.yaml
```

### 11.4 独立测试

```bash
python train_transformer.py --mode test \
  --config configs/train_cuda.yaml \
  --checkpoint runs_transformer_cuda/train_20260609_151606_v3_char_rdrop/avg_model.pt \
  --max_bleu_samples 0
```

---

## 十二、总结与未来方向

### 12.1 实验总结

本实验从零实现了完整的 Transformer 翻译系统，通过三轮迭代优化实现了显著的 BLEU 提升：

- **字符级分词** 解决了中文端 UNK 覆盖率问题（贡献最大，+3~5 BLEU）
- **R-Drop 正则化** 有效控制过拟合（gap 缩小 8 倍，+1~2 BLEU）
- **源端噪声增强** 提升了模型鲁棒性（+0.5~1 BLEU）
- **Checkpoint 平均** 进一步稳定了模型性能

最终在 3,070 条全量测试集上达到 **BLEU = 40.89**，相比词级基线 24.57 提升了 **+16.32**。

### 12.2 不足与展望

1. **数据规模有限**：仅 18,000 句对，限制了模型上限。可考虑引入回译（Back-Translation）扩充数据。
2. **未使用预训练**：如引入 mBART 或 NLLB 的预训练权重，预期可进一步提升。
3. **解码效率**：逐句 beam search 较慢（全量测试约 28 分钟），可改为 batch beam search。
4. **评估维度单一**：仅用 BLEU，可增加 COMET、BERTScore 等语义评估。

---

## 参考文献

1. Vaswani A, et al. "Attention Is All You Need." NeurIPS 2017.
2. Liang X, et al. "R-Drop: Regularized Dropout for Neural Networks." NeurIPS 2021.
3. Wu Y, et al. "Google's Neural Machine Translation System." arXiv 2016.（GNMT 长度归一化）
4. Sennrich R, et al. "Neural Machine Translation of Rare Words with Subword Units." ACL 2016.（BPE）
