---
source_pdf: NLP课件_jfyu_第七章_循环神经网络_release.pdf
part: 7.6-7.7
keywords: deep-rnn, bidirectional-rnn, recursive-nn, rvnn, tree-structure
---

# 深度RNN与递归神经网络（★★）

#nlp-deep-learning #rnn #bidirectional-rnn #recursive-nn #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 堆叠 RNN | 多层 RNN 叠加，底层捕获低级特征，高层捕获抽象语义 |
| 双向 RNN | 前向 + 后向，每个位置获得**完整上下文** |
| RvNN | 按**树结构**递归组合，适合短语/句法级表示 |
| RNN ⊂ RvNN | 链式结构是 RvNN 的退化特例 |
| 代表模型 | Socher RvNN、MV-RNN、RNTN；SST 情感树库实验 |

## 堆叠 RNN（Deep / Stacked RNN）

将多个 RNN 层**垂直堆叠**，第 l 层的输出作为第 l+1 层的输入：

```
堆叠 RNN (L=3):
xₜ → [RNN 层1] → [RNN 层2] → [RNN 层3] → yₜ
      hₜ⁽¹⁾       hₜ⁽²⁾       hₜ⁽³⁾
```

**设计要点**：
- 底层学习局部/低级模式（词级、字符级特征）
- 高层组合为更抽象的语义表示
- 每层可使用 SRNN、GRU 或 LSTM
- 深度增加表达能力，但也需更多数据和正则化

> [!tip] 与 CNN 的类比
> 堆叠 RNN 类似 CNN 的"小核大深度"思想——通过多层累积扩大**有效上下文感受野**。

---

## 双向 RNN（Bidirectional RNN）

标准 RNN 仅利用**过去**信息（左上下文）。**#bidirectional-rnn** 同时运行两个方向：

- **前向 RNN**：← 从左到右处理，捕获前缀上下文
- **后向 RNN**：→ 从右到左处理，捕获后缀上下文
- **合并**：拼接或求和 **[h⃗ₜ ; h⃖ₜ]** 作为最终表示

```
双向 RNN:
前向: x₁ → x₂ → x₃ → x₄    →  h⃗₁ h⃗₂ h⃗₃ h⃗₄
后向: x₁ ← x₂ ← x₃ ← x₄    →  h⃖₁ h⃖₂ h⃖₃ h⃖₄
合并:                      →  [h⃗ₜ ; h⃖ₜ]  每个位置含双向上下文
```

**典型应用**：
- 序列标注（NER、POS）：每个词需要左右上下文
- 机器阅读理解：理解代词指代、否定范围等

> [!important] 与 RNN 的区别
> 双向 RNN **不能**用于在线/流式预测（未来信息不可用），主要用于**完整序列已知的离线任务**。

---

## 递归神经网络（RvNN）

**Recursive Neural Network（#recursive-nn）** 按**树结构**（而非链）递归组合子表示，适合建模短语的**组合语义**。

### 基本思想

给定一棵二叉树（通常来自句法分析），从叶子节点（词向量）开始，逐层向上合并：

**h_parent = f(W·[h_left ; h_right] + b)**

```
RvNN 树结构组合:
        h_root
       /      \
    h₁₂       h₃₄
   /   \     /   \
  w₁   w₂   w₃   w₄
```

### 主要变体

| 模型 | 提出 | 核心特点 |
|------|------|----------|
| **标准 RvNN** | Socher et al., 2011 | 共享权重矩阵 W，按句法树递归合并 |
| **Syntactically-Untied RvNN** | — | 不同句法关系使用**不同**权重矩阵 |
| **MV-RNN** | Socher et al., 2012 | 词用**向量+矩阵**表示，矩阵编码修饰关系 |
| **RNTN** | Socher et al., 2013 | 引入**张量**运算捕获更复杂的组合交互 |

#### 标准 RvNN（Socher 2011）

- 所有节点共享同一组合函数 f 和权重 W
- 输入：句法分析树 + 预训练词向量
- 输出：根节点向量用于分类（如情感极性）

#### MV-RNN（Matrix-Vector RNN）

- 每个词表示为 **(词向量, 词矩阵)** 对
- 矩阵编码该词如何修饰/被修饰
- 组合时矩阵与向量交互，捕获"形容词-名词"等修饰语义

#### RNTN（Recursive Neural Tensor Network）

- 在组合函数中引入**三阶张量** V：
  **h = f([h_L ; h_R]ᵀ V [h_L ; h_R] + W[h_L ; h_R] + b)**
- 张量项捕获子节点间的**多向交互**，表达力更强
- 在 SST 上取得当时最优结果

---

## RvNN 与 RNN 的关系

> **RNN 是 RvNN 的特例**：当递归结构**退化为链**（每个节点只有一个子节点延续）时，RvNN 等价于 RNN。

```
RvNN（树）:          RNN（链）:
    root                h₄
   /    \               ↑
  h₁    h₂₃             h₃
 / \    / \              ↑
w₁ w₂  w₃ w₄            h₂ → h₁
                         ↑
                        x序列
```

| 维度 | RNN | RvNN |
|------|-----|------|
| 结构 | 链（线性序列） | 树（任意分支） |
| 组合方式 | 时间步递推 | 子树递归合并 |
| 词序建模 | 隐式（位置编码） | 显式（句法结构） |
| 典型输入 | 原始序列 | 句法分析树 + 词向量 |

---

## Stanford Sentiment TreeBank（SST）实验

**SST** 是 Stanford 情感树库，为短语和完整句子提供**细粒度情感标签**（5 级），并附带 **Stanford Parser** 生成的句法树。

**实验意义**：
- 验证 RvNN/RNTN 能否在**组合过程中**正确传播情感
- 例如："not very good" 中 "not" 应翻转 "good" 的极性
- RNTN 在 SST 上展示了**句法感知的组合语义**学习能力

```
SST 情感组合示例:
"good"        → 正面 (+)
"very good"   → 强正面 (++)
"not good"    → 负面 (-)    ← 否定翻转
"not very good" → 弱负面     ← 组合语义
```

> [!tip] 后续发展
> 纯 RvNN 依赖外部句法分析器；后续 Tree-LSTM、Gumbel-Tree 等尝试**端到端**学习组合结构。注意力机制 → 见 [[注意力机制]]。

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "双向 RNN 合并" | 拼接 [h⃗ₜ ; h⃖ₜ] 或 element-wise 求和 |
| "双向 RNN 局限" | 需完整序列，无法在线预测 |
| "RvNN 组合公式" | h = f(W·[h_L ; h_R] + b) |
| "RNN vs RvNN" | RNN 是链式结构，RvNN 的特例 |
| "RNTN 特点" | 引入三阶张量捕获子节点交互 |
| "MV-RNN 特点" | 词 = 向量 + 矩阵，矩阵编码修饰 |
| "SST 用途" | 短语级情感分析 + 句法树 |
| "堆叠 RNN 作用" | 多层提取层次化特征 |

## 相关笔记
- [[RNN基础与参数学习]]
- [[GRU与LSTM]]
- [[注意力机制]]
