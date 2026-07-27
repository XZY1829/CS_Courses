---
source_pdf: NLP课件_jfyu_第八章_8-1_Transformer_release.pdf
part: 8.1-8.3
keywords: practice, self-attention, multi-head-attention, encoder, decoder, masked-attention, cross-attention
---

# Transformer 练习（10 题）

#practice #transformer

## 相关概念
- [[自注意力机制]]
- [[Transformer架构]]

> [!hint]- 核心模式（点击查看）
> | 关键词 | 答案 |
> |--------|------|
> | Q/K/V 公式 | $Q=W^q I$, $K=W^k I$, $V=W^v I$ |
> | 缩放点积 | **softmax(KᵀQ/√dₖ)V** |
> | 多头输出 | $W^O \cdot \text{concat}(\text{head}_1, \ldots, \text{head}_h)$ |
> | 位置编码原因 | SA **置换不变**，需注入顺序 |
> | Encoder Self-Attn | **双向**，所有位置可见 |
> | Masked Self-Attn | 位置 i **只看 ≤i** |
> | Cross-Attn | **Q←Dec, K/V←Enc** |
> | AT vs NAT | 逐步串行 vs **并行**生成 |

---

## 第 1 题 - Self-Attention 动机 [recall]
> 相比 RNN 和 CNN，Self-Attention 主要解决了哪两个关键问题？

> [!answer]- 查看答案
> 1. **并行化**：RNN 必须按时间步串行计算，Self-Attention 可矩阵并行
> 2. **全局依赖**：CNN 感受野局部，Self-Attention 单层即可让任意两位置直接交互
> 参见 [[自注意力机制]]。

---

## 第 2 题 - Q/K/V 计算 [recall]
> 给定输入表示矩阵 I 和投影矩阵 $W^q$, $W^k$, $W^v$，写出 Q、K、V 的计算公式，并说明各自直觉角色。

> [!answer]- 查看答案
> - $Q = W^q I$（Query：当前位置"查询"什么）
> - $K = W^k I$（Key：各位置"被匹配"的索引）
> - $V = W^v I$（Value：被加权聚合的实际内容）
> 注意力权重由 Q 与 K 相似度决定，输出为 V 的加权和。参见 [[自注意力机制]]。

---

## 第 3 题 - 缩放点积手算 [application]
> 设 dₖ=1，KᵀQ 得到标量分数 s=4。经 softmax 后权重为 1.0（单元素）。若 V 为列向量 [2, 5]ᵀ，求输出 O。若忘记除以 √dₖ 但 dₖ=1，结果是否改变？

> [!answer]- 查看答案
> O = V · 1.0 = **[2, 5]ᵀ**。
> 当 **dₖ=1** 时，√dₖ=1，缩放无效果，有无缩放结果相同。当 dₖ 较大时，不缩放会导致 softmax 过于尖锐。参见 [[自注意力机制]]。

---

## 第 4 题 - 多头注意力结构 [recall]
> 简述 Multi-Head Attention 的计算流程：从 Q/K/V 到最终输出 bᵢ。

> [!answer]- 查看答案
> 1. 用 h 组投影矩阵 $W^{q,j}$, $W^{k,j}$, $W^{v,j}$ 将 Q/K/V 映射到 h 个子空间
> 2. 每个头独立计算 **Attention(Qⱼ, Kⱼ, Vⱼ)**
> 3. 拼接各头输出：**concat(bᵢ,₁, …, bᵢ,ₕ)**
> 4. 线性投影：$b_i = W^O \cdot \text{concat}(\ldots)$ 
> 参见 [[自注意力机制]]。

---

## 第 5 题 - 位置编码作用 [recall]
> Self-Attention 为何必须加入位置编码？不加会怎样？

> [!answer]- 查看答案
> Self-Attention 对输入 token 做**集合式**处理，打乱顺序不改变各 token 被 attend 的方式（**置换不变**）。语言中词序承载语义（"狗咬人"≠"人咬狗"），必须加入 **eᵢ** 位置编码使模型感知顺序。参见 [[自注意力机制]]。

---

## 第 6 题 - Encoder vs Decoder [recall]
> 对比 Transformer Encoder 和 Decoder 中 Self-Attention 的可见范围差异。

> [!answer]- 查看答案
> | 组件 | Self-Attention 类型 | 可见范围 |
> |------|---------------------|----------|
> | **Encoder** | 标准 Self-Attention | **所有位置**（双向上下文） |
> | **Decoder** | **Masked** Self-Attention | 位置 i **仅可见 ≤ i**（不能看未来） |
> Decoder 额外还有 **Cross-Attention**  attend 到 Encoder 输出。参见 [[Transformer架构]]。

---

## 第 7 题 - Masked Attention 原因 [analysis]
> 训练 Transformer Decoder 时使用完整目标序列（teacher forcing）。若 Self-Attention 不做 Mask，会导致什么问题？推理时又如何体现？

> [!answer]- 查看答案
> 不 Mask 时，位置 i 可直接 attend 到位置 j>i 的**未来 token**，相当于训练时"偷看答案"。推理时只能逐步生成，无法获得未来信息，造成**训练-推理分布不一致**，生成质量严重下降。Mask 确保训练与自回归推理条件一致。参见 [[Transformer架构]]。

---

## 第 8 题 - Cross-Attention 机制 [recall]
> Cross-Attention 中 Q、K、V 分别来自哪里？其作用是什么？

> [!answer]- 查看答案
> - **Q ← Decoder** 当前状态（"我现在需要源序列什么信息？"）
> - **K, V ← Encoder** 输出（源序列编码表示）
> 作用：建立**源-目标对齐**，Decoder 生成每个目标 token 时动态查询 Encoder 中最相关的源语言片段。参见 [[Transformer架构]]。

---

## 第 9 题 - Layer Norm 与残差 [recall]
> 写出 Transformer 子层的 Add & Norm 公式，并说明 Layer Norm 与 Batch Norm 的归一化维度差异。

> [!answer]- 查看答案
> **output = LayerNorm(x + SubLayer(x))**
> - **Layer Norm**：对**单个样本的特征维度**归一化（适合变长序列）
> - **Batch Norm**：对 batch 维度归一化（CV 常用）
> 参见 [[Transformer架构]]。

---

## 第 10 题 - AT vs NAT [analysis]
> 比较自回归（AT）与非自回归（NAT）生成的优劣，并说明原论文 Transformer 翻译模型采用哪种方式。

> [!answer]- 查看答案
> | 方面 | AT（自回归） | NAT（非自回归） |
> |------|-------------|----------------|
> | 生成方式 | 逐步 y₁→y₂→…→yₙ | 并行预测所有位置 |
> | 质量 | 通常更高 | 可能因独立假设损失质量 |
> | 推理速度 | 慢（串行） | 快（并行） |
> | 代表 | 原始 Transformer Decoder | NAT、Mask-Predict 等 |
> 原论文 Transformer 机器翻译采用 **AT** 解码。参见 [[Transformer架构]]。

---

> [!summary]- 模式总结（点击查看）
> | 关键词 | 答案 |
> |--------|------|
> | SA 动机 | 并行 + 全局依赖 |
> | 缩放点积 | softmax(KᵀQ/√dₖ)V |
> | 多头融合 | concat → $W^O$ |
> | 位置编码 | 补偿 SA 无顺序感知 |
> | Encoder Attn | 双向全可见 |
> | Masked Attn | 防偷看未来 |
> | Cross-Attn | Q←Dec, K/V←Enc |
> | Add&Norm | LayerNorm(x+SubLayer(x)) |
> | AT/NAT | 串行高质量 vs 并行快速 |
> | 评估指标 | BLEU score |
