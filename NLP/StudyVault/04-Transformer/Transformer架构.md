---
source_pdf: NLP课件_jfyu_第八章_8-1_Transformer_release.pdf
part: 8.3
keywords: encoder, decoder, masked-attention, cross-attention, layer-norm, residual-connection, autoregressive, bleu
---

# Transformer 架构（★★★）

#nlp-deep-learning #transformer #encoder #decoder #masked-attention #cross-attention #layer-norm #residual-connection #autoregressive #non-autoregressive #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 整体结构 | **Encoder-Decoder**，基于 Self-Attention，无 RNN/CNN |
| Encoder Block | Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm |
| Decoder Block | Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm |
| Layer Norm | **x'ᵢ = (xᵢ − m) / σ**，对**每个样本的特征维度**归一化 |
| 残差连接 | **output = LayerNorm(x + SubLayer(x))**（Post-LN，原论文） |
| Cross-Attention | **K, V ← Encoder 输出**；**Q ← Decoder 状态** |
| 生成范式 | **AT** 逐步自回归；**NAT** 并行生成所有位置 |
| 评估 | 机器翻译常用 **BLEU score** |
| 原论文 | *Attention is All You Need* (Vaswani et al., NIPS 2017) |

## 整体架构：Encoder-Decoder

Transformer 采用经典的 **Seq2Seq Encoder-Decoder** 框架，但完全用注意力机制替代 RNN：

```
输入序列 ──→ [Encoder × N] ──→ 编码表示 H
                                      │
目标序列 ──→ [Decoder × N] ←──────────┘ (Cross-Attention)
                │
                ↓
            输出概率分布
```

| 组件 | 层数（Base） | 作用 |
|------|-------------|------|
| **Encoder** | N=6 层 | 双向上下文编码输入序列 |
| **Decoder** | N=6 层 | 自回归生成目标序列，并 attend 到 Encoder |
| **Embedding + Pos Encoding** | 共享/独立 | 词嵌入 + 位置编码 |
| **输出层** | Linear + Softmax | 预测下一个 token |

> [!important] 里程碑意义
> 原论文证明：**仅靠注意力机制**即可在机器翻译上达到 SOTA，开启 Transformer 时代。

---

## Encoder Block 结构

每个 Encoder 层包含两个子层，均带残差连接和层归一化：

```
输入 x
  │
  ├─→ Multi-Head Self-Attention ──→ (+) ──→ LayerNorm ──┐
  │                                    ↑                  │
  └────────────────────────────────────┘                  │
                                                            ↓
  ├─→ Feed-Forward Network (FFN) ──→ (+) ──→ LayerNorm ──→ 输出
  │                                    ↑
  └────────────────────────────────────┘
```

**FFN**（逐位置前馈网络）：

**FFN(x) = max(0, xW₁ + b₁)W₂ + b₂**

- 同一层内所有位置共享 FFN 参数
- 通常中间维度 = 4 × d_model（Base: 768 → 3072 → 768）

| 子层 | 注意力类型 | 可见范围 |
|------|-----------|----------|
| Self-Attention | Q=K=V 来自同一层输入 | **所有位置**（双向） |

---

## Layer Normalization

**Layer Normalization** 对**单个样本**在**特征维度**上计算均值和方差：

**x'ᵢ = (xᵢ − m) / σ**

其中 m、σ 为 x 在特征维度上的均值和标准差（与 Batch Norm 按 batch 维度不同）。

| 对比 | Batch Norm | Layer Norm |
|------|-----------|------------|
| 归一化维度 | batch × 序列 | **特征维度** |
| 适用场景 | CV 大 batch | **NLP 变长序列、小 batch** |
| Transformer 选用 | — | **Layer Norm**（序列长度可变） |

---

## 残差连接（Residual Connection）

Transformer 采用 **Add & Norm** 模式（原论文为 Post-LN）：

**output = LayerNorm(x + SubLayer(x))**

```
SubLayer(x) = 注意力或 FFN
残差路径: x 直接加到子层输出 → 再 LayerNorm
```

| 作用 | 说明 |
|------|------|
| **梯度畅通** | 恒等路径缓解深层网络梯度消失 |
| **稳定训练** | 子层只需学习**增量** Δ，而非完整映射 |
| **与 Pre-LN 变体** | 部分后续模型用 LayerNorm(x) 后再进子层（Pre-LN），训练更稳定 |

> [!warning] 原论文 vs 后续变体
> 课件与原始 Transformer 使用 **Post-LN**（先加后归一化）；BERT 等使用 **Pre-LN** 或 Post-LN 变体，考试需区分。

---

## Decoder Block 结构

Decoder 比 Encoder 多一个 **Cross-Attention** 子层，且 Self-Attention 带 **Mask**：

```
输入 y（已生成部分）
  │
  ├─→ Masked Multi-Head Self-Attention ──→ Add&Norm
  │
  ├─→ Multi-Head Cross-Attention ──→ Add&Norm
  │       Q ← Decoder    K,V ← Encoder 输出
  │
  └─→ FFN ──→ Add&Norm ──→ 输出
```

### Masked Self-Attention（掩码自注意力）

训练/推理时，位置 i **只能 attend 到 ≤ i 的位置**，不能看到"未来" token。

```
注意力矩阵（Mask 后）:
     t1  t2  t3  t4
t1 [  ✓   ✗   ✗   ✗ ]
t2 [  ✓   ✓   ✗   ✗ ]
t3 [  ✓   ✓   ✓   ✗ ]
t4 [  ✓   ✓   ✓   ✓ ]
```

> [!danger] 为何必须 Mask
> 自回归生成时，训练用**完整目标序列**做 teacher forcing；若不 mask，Decoder 可直接"偷看"未来答案，推理时却无法获得，造成**训练-推理不一致**。

### Cross-Attention（交叉注意力）

连接 Encoder 与 Decoder 的桥梁：

| 来源 | 角色 | 含义 |
|------|------|------|
| **Encoder 输出** | **K, V** | 源序列的全局编码信息 |
| **Decoder 当前状态** | **Q** | "我现在需要源序列的哪部分？" |

**CrossAttention(Q_dec, K_enc, V_enc) = softmax(K_encᵀ Q_dec / √dₖ) V_enc**

```
Encoder:  "我 爱 自然 语言 处理"  →  H (源表示)
                                              ↓ K, V
Decoder:  "I  love  ___"  →  Q  ──→  Cross-Attn  →  对齐源-目标
```

直觉：Decoder 每步生成时，通过 Cross-Attention **动态查询** Encoder 中与当前生成最相关的源语言片段（类似对齐）。

---

## 自回归 vs 非自回归生成

| 范式 | 缩写 | 生成方式 | 优点 | 缺点 |
|------|------|----------|------|------|
| **自回归** | **AT** | 逐步生成 y₁, y₂, …, yₙ，每步依赖已生成全部 token | 质量高，标准 Transformer Decoder | 串行，推理慢 |
| **非自回归** | **NAT** | 并行预测所有位置（如迭代/refinement） | 推理快 | 需独立假设或迭代修正，质量可能下降 |

```
AT:  y₁ → y₂ → y₃ → y₄   (每步输入含之前所有 token)
NAT: [?, ?, ?, ?] → 一次/迭代并行填充全部位置
```

原论文 Transformer 翻译模型采用 **AT** 解码：每步预测下一个 token，直到输出 `<EOS>`。

---

## 评估指标：BLEU Score

**BLEU (Bilingual Evaluation Understudy)** 是机器翻译最常用的自动评估指标：

- 基于 n-gram **精确率**（1-gram 到 4-gram）
- 含**简短惩罚（Brevity Penalty）**，防止过短译文刷分
- 范围 0~1（或 0~100），越高表示与参考译文越接近

> [!tip] 考试关联
> Transformer 原论文在 WMT 英德/英法翻译上 BLEU 显著提升，是证明其有效性的关键实验结果。

---

## 参考论文

**Attention is All You Need**  
Vaswani, Shazeer, Parmar, et al. — **NIPS 2017**

核心贡献：
- 提出纯注意力架构 Transformer
- Multi-Head Attention + Position Encoding + Encoder-Decoder
- 机器翻译 SOTA，训练可充分并行

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "Encoder 子层顺序" | Self-Attn → Add&Norm → FFN → Add&Norm |
| "Decoder 比 Encoder 多什么" | Masked Self-Attn + **Cross-Attention** |
| "Cross-Attention Q/K/V 来源" | Q←Decoder, K/V←Encoder 输出 |
| "为何 Masked Attention" | 防止看到未来 token，保证自回归一致性 |
| "Layer Norm 归一化维度" | **特征维度**（per sample） |
| "残差公式" | LayerNorm(x + SubLayer(x)) |
| "AT vs NAT" | AT 逐步串行；NAT 并行生成 |
| "翻译评估指标" | BLEU score |
| "原论文" | Attention is All You Need, NIPS 2017 |

## 相关笔记
- [[自注意力机制]]
- [[Transformer_练习题]]
- [[RNN基础与参数学习]]
