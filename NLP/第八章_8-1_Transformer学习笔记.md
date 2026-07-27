# 第八章 8.1 Transformer 网络架构

> 南京大学 虞剑飞《自然语言处理》课程笔记
>
> 核心论文：Vaswani et al. "Attention is All You Need" (NIPS 2017)

---

## 一、引言：为什么需要 Transformer？

### 1.1 循环神经网络（RNN）的致命缺陷

RNN 处理序列的方式是**逐步递推**的：

```
x1 → h1 → x2 → h2 → x3 → h3 → x4 → h4
```

- 计算 h2 **必须等** h1 完成，计算 h3 **必须等** h2 完成……
- 这意味着 **无法并行化**，训练速度极慢
- 序列越长，计算时间线性增长

### 1.2 两种替代方案

| 方案 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **方案1：CNN 替换 RNN** | 用卷积核在序列上滑动 | 可以并行计算 | **只能捕获局部信息**，远距离依赖需要很多层堆叠 |
| **方案2：自注意力机制** | 每个位置直接关注所有其他位置 | 可以并行 + 捕获全局信息 | 计算量随序列长度平方增长 |

**结论**：自注意力机制（Self-Attention）是更优方案，由此诞生了 Transformer。

---

## 二、自注意力机制（Self-Attention）

### 2.1 注意力机制回顾

在标准的注意力机制中：

1. 有一个 **query（查询向量）** q
2. 有一组 **key（键向量）** x1, x2, x3, ...
3. 计算 q 与每个 key 的**点积**得到注意力得分
4. 通过 **Softmax** 归一化得到注意力分布（概率分布）
5. 用注意力分布对 value 做**加权求和**

**计算示例**：

假设 q = [1, 1, 1, 1]，三个输入向量：

```
x1 = [1, 0, 1, 0]
x2 = [0, 2, 0, 2]
x3 = [1, 1, 1, 1]
```

点积计算：
- q · x1 = 1×1 + 1×0 + 1×1 + 1×0 = 2
- q · x2 = 1×0 + 1×2 + 1×0 + 1×2 = 4
- q · x3 = 1×1 + 1×1 + 1×1 + 1×1 = 4

Softmax([2, 4, 4]) ≈ [0.063, 0.468, 0.468]

### 2.2 自注意力的核心思想

**关键区别**：在自注意力中，query 不是外部给定的，而是**从输入自身产生的**。

对于输入序列 a1, a2, a3, a4，每个位置同时充当 query、key 和 value：

```
每个输入 ai 通过三个不同的线性变换，生成三种向量：

  qi = Wq · ai    （Query：我想找什么？）
  ki = Wk · ai    （Key：我能提供什么？）
  vi = Wv · ai    （Value：我的实际内容是什么？）
```

其中 Wq、Wk、Wv 是三个**可学习的参数矩阵**。

### 2.3 自注意力的完整计算流程

以计算位置 1 的输出 b1 为例：

**第一步：生成 Q、K、V**

```
q1 = Wq · a1
k1 = Wk · a1,  k2 = Wk · a2,  k3 = Wk · a3,  k4 = Wk · a4
v1 = Wv · a1,  v2 = Wv · a2,  v3 = Wv · a3,  v4 = Wv · a4
```

**第二步：计算注意力得分（Attention Scores）**

```
α(1,1) = q1 · k1    （a1 和自己的相关性）
α(1,2) = q1 · k2    （a1 和 a2 的相关性）
α(1,3) = q1 · k3    （a1 和 a3 的相关性）
α(1,4) = q1 · k4    （a1 和 a4 的相关性）
```

**第三步：Softmax 归一化**

```
[α'(1,1), α'(1,2), α'(1,3), α'(1,4)] = Softmax([α(1,1), α(1,2), α(1,3), α(1,4)])
```

**第四步：加权求和得到输出**

```
b1 = α'(1,1)·v1 + α'(1,2)·v2 + α'(1,3)·v3 + α'(1,4)·v4
```

对每个位置 i 重复以上过程，就得到所有输出 b1, b2, b3, b4。

### 2.4 矩阵形式（高效并行计算）

将所有输入拼成矩阵 I = [a1, a2, a3, a4]，整个过程可以用矩阵运算一步完成：

```
Q = Wq · I        ← 所有 query 向量
K = Wk · I        ← 所有 key 向量
V = Wv · I        ← 所有 value 向量

A = K^T · Q        ← 注意力得分矩阵（维度 n×n）
A' = Softmax(A)    ← 逐列做 Softmax

O = V · A'         ← 输出矩阵
```

**注意力得分矩阵 A 的含义**：A[i,j] 表示位置 j 的 query 对位置 i 的 key 的关注程度。

> **核心公式**：Attention(Q, K, V) = Softmax(K^T Q) · V
>
> 待学习参数：Wq、Wk、Wv 三个矩阵

### 2.5 课堂练习解析

给定输入矩阵和参数矩阵：

```
I = [[1, 0, 1],       Wq = [[1, 1, 0, 0],      Wk = [[0, 1, 0, 1],      Wv = [[0, 0, 1, 1],
     [0, 2, 1],              [0, 0, 0, 1],             [0, 1, 1, 1],             [2, 3, 0, 1],
     [1, 0, 1],              [1, 0, 1, 1]]             [1, 0, 0, 0]]             [0, 0, 3, 0]]
     [0, 2, 1]]
```

计算步骤：Q = Wq · I，K = Wk · I，V = Wv · I，然后 A = K^T · Q，Softmax，最后 O = V · A'。

### 2.6 多头自注意力（Multi-Head Self-Attention）

**动机**：单个注意力头只能捕获一种关系模式。多个头可以关注**不同类型的依赖关系**。

**以 2 个头为例**：

```
第一步：正常计算 Q、K、V
  qi = Wq · ai,  ki = Wk · ai,  vi = Wv · ai

第二步：分头（Split）
  qi,1 = Wq,1 · qi    ki,1 = Wk,1 · ki    vi,1 = Wv,1 · vi    ← head 1
  qi,2 = Wq,2 · qi    ki,2 = Wk,2 · ki    vi,2 = Wv,2 · vi    ← head 2

第三步：每个头独立做自注意力
  bi,1 = Attention(qi,1, K1, V1)    ← head 1 的输出
  bi,2 = Attention(qi,2, K2, V2)    ← head 2 的输出

第四步：拼接 + 线性变换
  bi = WO · [bi,1 ; bi,2]           ← 最终输出
```

**实际应用中**，通常不是额外增加参数，而是将 d_model 维度**均分**给各个头：
- 如果 d_model = 512，有 8 个头，则每个头处理 64 维
- 计算量和单头差不多，但表达能力更强

### 2.7 位置编码（Positional Encoding）

**问题**：自注意力机制是**位置无关**的——打乱输入顺序，输出（对应也打乱后）不变。但语言是有序的！

**解决方案**：在输入中加入位置信息。

```
输入 = 词嵌入 + 位置编码
ai_new = ai + ei
```

位置编码 ei 的两种生成方式：

**方式一：人工构造（Sinusoidal，原始 Transformer 使用）**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- pos 是位置，i 是维度索引
- 不同位置的编码是唯一的
- 理论上可以泛化到训练中没见过的更长序列

**方式二：参数学习**

- 把位置编码当作可训练参数，和模型一起学习
- BERT 等模型采用这种方式

---

## 三、Transformer 网络架构

### 3.1 整体架构：Encoder-Decoder

```
┌──────────┐     ┌──────────┐
│  Encoder │ ──> │  Decoder │
│          │     │          │
│ 输入序列  │     │ 输出序列  │
└──────────┘     └──────────┘
```

- **Encoder**：将输入序列编码为一组隐藏表示
- **Decoder**：基于 Encoder 的输出，自回归地生成目标序列

### 3.2 Encoder 详解

Encoder 由 **N 个相同的 Block 堆叠**而成（原始论文 N=6）。

每个 Block 包含两个子层：

```
┌─────────────────────────────────┐
│         输入 x1, x2, x3, x4     │
│              ↓                   │
│    Multi-Head Self-Attention     │
│              ↓                   │
│       Add & Layer Norm          │  ← 残差连接 + 层归一化
│              ↓                   │
│    Feed-Forward Network (FC)     │  ← 逐位置的全连接网络
│              ↓                   │
│       Add & Layer Norm          │  ← 残差连接 + 层归一化
│              ↓                   │
│         输出 h1, h2, h3, h4     │
└─────────────────────────────────┘
```

#### 3.2.1 Scaled Dot-Product Attention

在实际实现中，注意力公式增加了一个**缩放因子**：

```
Attention(Q, K, V) = Softmax(Q · K^T / √d_k) · V
```

为什么要除以 √d_k？
- 当 d_k（key 的维度）较大时，点积的结果数值会很大
- 导致 Softmax 输出趋向 one-hot（梯度极小）
- 除以 √d_k 让数值保持在合理范围内

#### 3.2.2 Add & Layer Norm

**Add（残差连接）**：

```
output = x + SubLayer(x)
```

- 将子层的输入直接加到子层的输出上
- 缓解深层网络的梯度消失问题
- 让信息可以跨层直接传递

**Layer Norm（层归一化）**：

对每个样本的特征维度做归一化（不同于 Batch Norm 对 batch 维度做归一化）：

```
x'_i = (x_i - m) / σ
```

其中 m 是该样本所有特征的均值，σ 是标准差。

参考论文：[Layer Normalization](https://arxiv.org/abs/1607.06450)

### 3.3 Decoder 详解

#### 3.3.1 自回归生成（Autoregressive）

Decoder 采用**自回归**方式逐步生成输出：

```
输入: "Machine Learning"（经过 Encoder 编码）

Step 1: START → Decoder → Softmax → "机"（最大概率）
Step 2: START, 机 → Decoder → Softmax → "器"
Step 3: START, 机, 器 → Decoder → Softmax → "学"
Step 4: START, 机, 器, 学 → Decoder → Softmax → "习"
Step 5: START, 机, 器, 学, 习 → Decoder → Softmax → END（结束）
```

每一步：
1. 将已生成的序列输入 Decoder
2. Decoder 输出一个 size=V 的概率分布（V 是词表大小）
3. 取概率最大的词作为输出
4. 将该词加入已生成序列，重复直到生成 END token

#### 3.3.2 Masked Self-Attention（掩码自注意力）

**核心问题**：在生成第 t 个词时，模型**不应该看到**第 t+1, t+2, ... 位置的信息（因为它们还没生成）。

**解决方案**：在 Decoder 的 Self-Attention 中加入 **Mask（掩码）**。

```
普通 Self-Attention：                Masked Self-Attention：

b1 关注 a1,a2,a3,a4                  b1 只关注 a1
b2 关注 a1,a2,a3,a4                  b2 只关注 a1,a2
b3 关注 a1,a2,a3,a4                  b3 只关注 a1,a2,a3
b4 关注 a1,a2,a3,a4                  b4 关注 a1,a2,a3,a4
```

实现方式：在注意力得分矩阵中，将未来位置的值设为 **-∞**，经过 Softmax 后变为 0。

#### 3.3.3 Cross-Attention（交叉注意力）

Cross-Attention 是 Decoder 和 Encoder 之间的桥梁：

```
- Query（Q）：来自 Decoder（当前要生成什么）
- Key（K）：来自 Encoder 的输出（源序列的信息）
- Value（V）：来自 Encoder 的输出（源序列的内容）
```

工作流程：

```
Encoder 输出：a1, a2, a3
  → k1 = Wk·a1,  k2 = Wk·a2,  k3 = Wk·a3
  → v1 = Wv·a1,  v2 = Wv·a2,  v3 = Wv·a3

Decoder 当前输入（经过 Masked Self-Attention 后）：
  → q = Wq · (Decoder中间表示)

计算注意力：
  α'1, α'2, α'3 = Softmax(q·k1, q·k2, q·k3)
  output = α'1·v1 + α'2·v2 + α'3·v3
```

**直觉理解**：Cross-Attention 让 Decoder 在生成每个词时，能够「查看」源序列的不同部分，决定重点关注哪些信息。

#### 3.3.4 Decoder Block 完整结构

```
┌──────────────────────────────────────┐
│           Decoder 输入                │
│               ↓                       │
│     Masked Multi-Head Self-Attention │
│               ↓                       │
│          Add & Layer Norm            │
│               ↓                       │
│   Multi-Head Cross-Attention         │ ← Q 来自 Decoder，K/V 来自 Encoder
│               ↓                       │
│          Add & Layer Norm            │
│               ↓                       │
│     Feed-Forward Network (FC)        │
│               ↓                       │
│          Add & Layer Norm            │
│               ↓                       │
│          Decoder 输出                 │
└──────────────────────────────────────┘
```

### 3.4 非自回归 Decoder（NAT）

自回归（AT）每次只生成一个词，速度慢。非自回归（NAT）尝试一次性生成所有词：

| 特性 | AT (Autoregressive) | NAT (Non-Autoregressive) |
|------|---------------------|--------------------------|
| 生成方式 | 逐个生成 | 并行生成所有词 |
| 输入 | START, w1, w2, ... | START, START, START, ... |
| 速度 | 慢（序列步骤） | 快（一步完成） |
| 质量 | 高 | 通常较低（多模态问题） |

NAT 的挑战：需要预先知道输出长度，且并行生成的词之间缺乏依赖关系。

---

## 四、关键知识点速查表

| 概念 | 核心要点 |
|------|---------|
| Self-Attention | 每个位置同时作为 Q、K、V 的来源 |
| Q、K、V | Q=查询, K=键, V=值，分别由 Wq, Wk, Wv 线性变换得到 |
| Scaled Attention | 除以 √d_k 防止点积过大导致梯度消失 |
| Multi-Head | 多个注意力头捕获不同模式的依赖关系 |
| Positional Encoding | 弥补自注意力忽略位置信息的缺陷 |
| Encoder Block | Self-Attention → Add&Norm → FFN → Add&Norm |
| Decoder Block | Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm |
| Masked Self-Attn | 防止 Decoder 看到未来位置的信息 |
| Cross-Attention | Q 来自 Decoder，K/V 来自 Encoder，建立源-目标连接 |
| Layer Norm | 对每个样本的特征做归一化，稳定训练 |
| 残差连接 | output = x + SubLayer(x)，缓解梯度消失 |

---

## 五、Transformer 的核心优势总结

1. **并行计算**：Self-Attention 中所有位置的计算可以同时进行（vs. RNN 必须顺序计算）
2. **全局依赖**：每个位置直接关注所有其他位置（vs. CNN 只看局部窗口）
3. **可解释性**：注意力权重矩阵直观地展示了哪些位置之间有强关联
4. **灵活架构**：Encoder-only、Decoder-only、Encoder-Decoder 三种变体适用于不同任务

---

## 六、课程作业：基于 Transformer 的机器翻译

- **任务**：英文 → 中文翻译（如 "tom is a student ." → "汤姆 是 个 学生 。"）
- **数据集**：训练集 18000 句，验证集 500 句，测试集 2636 句
- **评价指标**：BLEU Score

### BLEU Score 计算

```
BLEU = BP · exp(Σ wn · log(pn))    n = 1 到 N
```

- **BP**：长度过短句子的惩罚因子（防止模型只输出短句获得高分）
- **pn**：n-gram 精确率（候选译文中出现在参考译文中的 n-gram 比例）
- **wn = 1/N**：各 n-gram 的权重（通常 N=4）
- **BLEU 值范围**：0~1，越高越好

### 参考资料

- 《神经网络与深度学习》第 15.4.2 小节
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- Vaswani et al. "Attention is All You Need" (NIPS 2017)
- Wang et al. "Learning Deep Transformer Models for Machine Translation" (ACL 2019)
