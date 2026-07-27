# NLP 期末模拟试卷

> **说明**：本卷主体覆盖第 3–8 章（不含 8.4.4），另保留学长考后笔记提到的 NLP 定义热身题；总分 100 分。难度分布：easy 20% / medium 50% / hard 30%。  
> 建议用时：120 分钟。

---

## 试题部分

### 一、概念简答题（共 20 分）

**1.（5 分，easy，第 1 章）** 请给出自然语言处理（NLP）的定义，并列举至少三点"相对人工语言处理，自然语言更难"的原因。

---

**2.（5 分，easy，第 3 章）** 写出**多项式分布朴素贝叶斯**文本分类的核心假设与决策规则。说明为何该模型属于**生成式模型**。

---

**3.（5 分，medium，第 5 章）** 比较 Word2Vec 中 **CBOW** 与 **Skip-Gram** 的输入、输出及各自适用场景。One-hot 表示的两个主要缺陷是什么？

---

**4.（5 分，easy，第 8 章）** T5 模型如何将多种 NLP 任务统一为同一框架？其预训练任务 **Masked Span Prediction** 与 BERT 的 MLM 有何异同？请各举一例说明 T5 的 **prefix（任务前缀）** 用法。

---

### 二、计算题（共 40 分）

**5.（8 分，medium，第 4 章 · 课堂练习改编）** 给定训练语料（每行一句，已加起止符）：

```
<BOS> John read Moby Dick <EOS>
<BOS> Mary read a different book <EOS>
<BOS> She read a book by Cher <EOS>
```

（1）用**最大似然估计**的 **2-gram** 语言模型，计算句子 `John read a book` 的概率 $p(\text{John read a book})$。需列出各条件概率的分子、分母。

（2）说明句子 `Cher read a book` 在不平滑时概率为何为 0。

---

**6.（8 分，medium，第 4 章 · 课堂练习改编）** 在上题同一语料上，设词汇表大小 $|V|=13$，使用**加 1 平滑**（additive smoothing）：

$$p(w_i \mid w_{i-1}) = \frac{1 + c(w_{i-1}, w_i)}{|V| + \sum_{w_i} c(w_{i-1}, w_i)}$$

计算平滑后 $p(\text{Cher read a book})$ 的数值（保留 5 位有效数字）。

---

**7.（8 分，medium，第 6 章 · 课堂练习改编）** 对 $6 \times 6$ 灰度图像做**二维窄卷积**（$P=0$，$S=1$），滤波器为：

$$
W = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}
$$

输入图像为：

$$
X = \begin{bmatrix}
3 & 0 & 1 & 2 & 7 & 4 \\
1 & 5 & 8 & 9 & 3 & 1 \\
2 & 7 & 2 & 5 & 1 & 3 \\
0 & 1 & 3 & 1 & 7 & 8 \\
4 & 2 & 1 & 6 & 2 & 8 \\
2 & 4 & 5 & 2 & 3 & 9
\end{bmatrix}
$$

（1）求输出特征图的尺寸 $H_{\text{out}} \times W_{\text{out}}$。

（2）手算**左上角**和**右下角**两个位置的卷积输出值。

---

**8.（8 分，medium，第 8 章 · 课堂练习改编）** 给定 query $\mathbf{q} = [1, 1, 1, 1]^\top$，输入 $\mathbf{x}_1 = [1,0,1,0]^\top$，$\mathbf{x}_2 = [0,2,0,2]^\top$，$\mathbf{x}_3 = [1,1,1,1]^\top$，打分函数为**点积** $s(\mathbf{x}, \mathbf{q}) = \mathbf{x}^\top \mathbf{q}$。

（1）计算三个打分 $s_1, s_2, s_3$。

（2）对 $[s_1, s_2, s_3]$ 做 **softmax**，得注意力权重 $\alpha_1, \alpha_2, \alpha_3$（保留 3 位小数）。

（3）计算上下文向量 $\text{att} = \sum_{n=1}^{3} \alpha_n \mathbf{x}_n$。

---

**9.（8 分，medium，第 8 章 · 高频考点）** 机器翻译 BLEU 评估：

- **参考译文**（1 条）：`the cat is on the mat`（6 词）
- **候选译文**：`the cat on the mat`（5 词）

设 $N=4$，各阶权重 $w_n = 1/4$，仅考虑 1–4 gram 精确率 $p_n$，并计算**简短惩罚** $BP$：

$$BP = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \le r \end{cases}$$

其中 $c$ 为候选译文词数，$r$ 为参考译文词数。

（1）分别计算 $p_1, p_2, p_3, p_4$。

（2）计算 $BP$ 与最终 BLEU 值（不使用平滑；若某阶 $p_n=0$，需说明影响）。

---

### 三、模型分析题（共 25 分）

**10.（10 分，medium，第 3 章 · 课堂练习改编）** 给定如下精简训练集，特征集 Feature Set = [计算机, 排球, 运动会, 高校, 大学]：

| 类别 | 文档特征（词是否出现） |
|------|------------------------|
| 教育 | 大学, 计算机, 高校 |
| 教育 | 大学, 计算机 |
| 体育 | 大学, 运动会, 排球 |
| 体育 | 运动会, 排球 |

（1）不 smoothing，求先验 $P(\text{教育})$、$P(\text{体育})$。

（2）用**拉普拉斯平滑**（加 1 法，$M=5$ 个特征）估计 $P(w_i \mid c_j)$，写出公式并给出全部 10 个条件概率。

（3）对测试文档「**大学 计算机**」（仅含这两个特征），判断其类别。

---

**11.（8 分，hard，第 7 章 · 学长高频）** 简述 **RNN 的参数学习（训练）过程**：

（1）写出简单循环网络（SRNN）的隐状态更新公式。

（2）说明**时间反向传播（BPTT）**的基本思路。

（3）解释梯度消失/爆炸的成因，并各给出一种常用应对方法。

---

**12.（7 分，hard，第 6–7 章）** Kim (2014) **TextCNN** 与 **BiLSTM** 都常用于句子级文本分类。

（1）画出 TextCNN 的四个主要步骤。

（2）从**并行性、长程依赖、关键特征提取**三方面比较二者优劣。

（3）TextCNN 为何选用 **Global Max-pooling** 而非取 RNN 最后时刻隐状态？

---

### 四、开放式设计题（共 15 分）

**13.（10 分，hard，综合 · 学长高频）** 设计一个**自动对下联**系统：输入上联（如"春风得意马蹄疾"），输出语义与格律匹配的下联。

请说明：

（1）任务形式化（输入/输出表示）；

（2）模型架构选型及理由（至少对比两种方案）；

（3）训练数据与损失函数；

（4）推理时的约束策略（对仗、平仄、词性等至少提两点）。

---

**14.（5 分，hard，第 8 章）** 基于 **Transformer Encoder-Decoder** 设计英中机器翻译系统时：

（1）Decoder 中 **Masked Self-Attention** 的作用是什么？训练与推理阶段如何体现？

（2）**Cross-Attention** 中 Q、K、V 分别来自哪里？起什么作用？

（3）除 BLEU 外，再列举一种生成/翻译质量评估思路（一句话即可）。

---

---
---

## 参考答案与解析

### 一、概念简答题

#### 1. NLP 定义与自然语言难点（5 分）

**定义**：自然语言处理（NLP）是研究如何让计算机理解、分析、生成和处理人类自然语言的学科，目标是实现人机之间的自然语言交互。

**自然语言相对人工语言更难的原因**（答出任意 3 点即可，每点约 1 分）：

1. **歧义性**：一词多义、句法歧义、语义歧义普遍存在（如"咬死了猎人的狗"）。
2. **上下文依赖性**：同一表达在不同语境下含义不同，需要动态理解。
3. **非规范性**：口语省略、错别字、网络新词、方言等导致形式不固定。
4. **知识需求**：理解语言常需世界知识、常识与背景信息。
5. **隐含信息**：大量语义隐含于言外，需推理（如隐喻、讽刺）。
6. **演化性**：词汇与用法随时间快速变化。

---

#### 2. 多项式朴素贝叶斯（5 分）

**条件独立假设**（特征条件独立）：

$$P(\mathbf{x} \mid c_j) \approx \prod_{i=1}^{M} P(w_i \mid c_j)$$

其中 $\mathbf{x} = (w_1, \ldots, w_M)$ 为文档的特征向量。

**决策规则**：

$$c^* = \arg\max_{j} P(c_j) \prod_{i=1}^{M} P(w_i \mid c_j)$$

**为何是生成式模型**：模型显式学习类先验 $P(c_j)$ 与类条件分布 $P(w_i \mid c_j)$，通过联合概率 $P(\mathbf{x}, c_j) = P(c_j) P(\mathbf{x} \mid c_j)$ 做分类，属于对数据**生成过程**的建模；而非直接学习决策边界 $P(c_j \mid \mathbf{x})$（判别式）。

---

#### 3. CBOW vs Skip-Gram（5 分）

| 项目 | CBOW | Skip-Gram |
|------|------|-----------|
| 输入 | 上下文词向量（平均） | 中心词向量 |
| 输出 | 预测**中心词** | 预测**上下文词**（一对多） |
| 适用 | 高频词、大数据，训练快 | 低频词、小数据集效果更好 |

**One-hot 缺陷**：

1. **维度灾难**：向量维度 = 词表大小 $|V|$，极度稀疏。
2. **语义隔离**：不同词向量正交，内积恒为 0，无法表达语义相似性。

---

#### 4. T5 统一框架（5 分）

**统一框架**：T5 将所有 NLP 任务建模为 **Text-to-Text**——输入一段文本，输出一段文本；同一 Encoder-Decoder 架构 + 同一损失，仅通过**任务前缀（prefix）**区分任务。

**Masked Span Prediction vs BERT MLM**：

| | T5 MSP | BERT MLM |
|---|--------|----------|
| 架构 | Encoder-Decoder | 仅 Encoder |
| 遮盖 | 连续 **span** → sentinel token | 随机单 token → [MASK] |
| 预测 | Decoder **生成**被遮 span 文本 | 预测被遮 token |

**Prefix 示例**：

- 翻译：`translate English to German: That is good.` → `Das ist gut.`
- 分类：`sst2 sentence: This is great!` → `positive`

---

### 二、计算题

#### 5. 2-gram 概率（8 分）

**（1）条件概率统计**

语料 bi-gram 统计（历史 → 后继）：

| 条件 | 后继统计 | 概率 |
|------|----------|------|
| $\langle\text{BOS}\rangle$ | John:1, Mary:1, She:1 | $p(\text{John}\mid\langle\text{BOS}\rangle)=1/3$ |
| John | read:1 | $p(\text{read}\mid\text{John})=1/1=1$ |
| read | Moby:1, a:2 | $p(a\mid\text{read})=2/3$ |
| a | different:1, book:1 | $p(\text{book}\mid a)=1/2$ |
| book | $\langle\text{EOS}\rangle$:1, by:1 | $p(\langle\text{EOS}\rangle\mid\text{book})=1/2$ |

**联合概率**：

$$
\begin{aligned}
p(\text{John read a book})
&= p(\text{John}\mid\langle\text{BOS}\rangle) \cdot p(\text{read}\mid\text{John}) \cdot p(a\mid\text{read}) \cdot p(\text{book}\mid a) \cdot p(\langle\text{EOS}\rangle\mid\text{book}) \\
&= \frac{1}{3} \times 1 \times \frac{2}{3} \times \frac{1}{2} \times \frac{1}{2} \\
&= \frac{1}{18} \approx 0.0556
\end{aligned}
$$

**（2）** 语料中 **Cher** 从未作为 $\langle\text{BOS}\rangle$ 的后继出现，故 $p(\text{Cher}\mid\langle\text{BOS}\rangle)=0/3=0$，整条句子概率为 **0**（零概率问题）。

---

#### 6. 加 1 平滑（8 分）

平滑后各概率（课件 Ch4 p29–30）：

| 概率 | 计算 | 值 |
|------|------|-----|
| $p(\text{Cher}\mid\langle\text{BOS}\rangle)$ | $(0+1)/(13+3)$ | $1/16$ |
| $p(\text{read}\mid\text{Cher})$ | $(0+1)/(13+1)$ | $1/14$ |
| $p(a\mid\text{read})$ | $(2+1)/(13+3)$ | $3/16$ |
| $p(\text{book}\mid a)$ | $(1+1)/(13+2)$ | $2/15$ |
| $p(\langle\text{EOS}\rangle\mid\text{book})$ | $(1+1)/(13+2)$ | $2/15$ |

$$
p(\text{Cher read a book}) = \frac{1}{16} \times \frac{1}{14} \times \frac{3}{16} \times \frac{2}{15} \times \frac{2}{15} \approx 1.49 \times 10^{-5}
$$

（课件参考值 $\approx 0.00002$ 量级；精确值 **$1.49 \times 10^{-5}$**。课件提取文本中 `$p(\text{book}|a)$` 的结果疑似写错，按公式应为 $2/15$。）

---

#### 7. 二维卷积（8 分）

**（1）输出尺寸**

$$H_{\text{out}} = W_{\text{out}} = \lfloor (6 + 2 \times 0 - 3)/1 + 1 \rfloor = 4$$

输出为 **$4 \times 4$**。

**（2）卷积手算**

滤波器 $W$ 对窗口内元素逐位相乘再求和。

**左上角**（$X[0:3, 0:3]$）：

$$
\begin{aligned}
&3{\times}1 + 0{\times}0 + 1{\times}(-1) \\
+&1{\times}1 + 5{\times}0 + 8{\times}(-1) \\
+&2{\times}1 + 7{\times}0 + 2{\times}(-1) \\
=&\; 3 - 1 + 1 - 8 + 2 - 2 = \mathbf{-5}
\end{aligned}
$$

**右下角**（$X[3:6, 3:6]$）：

$$
\begin{aligned}
&1{\times}1 + 7{\times}0 + 8{\times}(-1) \\
+&6{\times}1 + 2{\times}0 + 8{\times}(-1) \\
+&2{\times}1 + 3{\times}0 + 9{\times}(-1) \\
=&\; 1 - 8 + 6 - 8 + 2 - 9 = \mathbf{-16}
\end{aligned}
$$

完整输出矩阵：

$$
\begin{bmatrix}
-5 & -4 & 0 & 8 \\
-10 & -2 & 2 & 3 \\
0 & -2 & -4 & -7 \\
-3 & -2 & -3 & -16
\end{bmatrix}
$$

---

#### 8. 注意力机制（8 分）

**（1）打分**

$$
\begin{aligned}
s_1 &= \mathbf{x}_1^\top \mathbf{q} = 1+0+1+0 = 2 \\
s_2 &= \mathbf{x}_2^\top \mathbf{q} = 0+2+0+2 = 4 \\
s_3 &= \mathbf{x}_3^\top \mathbf{q} = 1+1+1+1 = 4
\end{aligned}
$$

**（2）Softmax**（课件 Ch8 p12 答案）

$$
\alpha = \text{softmax}([2, 4, 4]) \approx [0.063,\; 0.468,\; 0.468]
$$

验算：$e^2 : e^4 : e^4 = 1 : 7.389 : 7.389$，归一化后 $\alpha_1 = 1/(1+7.389+7.389) \approx 0.063$。

**（3）上下文向量**

$$
\begin{aligned}
\text{att} &= 0.063 \cdot [1,0,1,0] + 0.468 \cdot [0,2,0,2] + 0.468 \cdot [1,1,1,1] \\
&\approx [0.532,\; 1.405,\; 0.532,\; 1.405]
\end{aligned}
$$

---

#### 9. BLEU 计算（8 分）

**候选**：`the cat on the mat`  
**参考**：`the cat is on the mat`

**（1）各阶 $p_n$**（clip 后精确率）

| 阶 | 候选 n-gram 数 | 匹配数 | $p_n$ |
|----|----------------|--------|-------|
| 1-gram | 5 | 5（the, cat, on, the, mat 均在参考中） | $5/5 = 1.000$ |
| 2-gram | 4 | 3（the cat ✓, cat on ✗, on the ✓, the mat ✓） | $3/4 = 0.750$ |
| 3-gram | 3 | 1（the cat on ✗, cat on the ✗, on the mat ✓） | $1/3 \approx 0.333$ |
| 4-gram | 2 | 0（the cat on the ✗, cat on the mat ✗） | $0/2 = 0$ |

**（2）BLEU**

$c=5$，$r=6$，因 $c < r$：

$$BP = e^{1 - r/c} = e^{1 - 6/5} = e^{-0.2} \approx 0.819$$

几何平均：

由于 $p_4=0$，$\log p_4$ 不可取，原始 BLEU 的几何平均为 0。

$$
\text{BLEU} = BP \times 0 = \mathbf{0}
$$

> 考试提示：如果题目明确要求使用平滑 BLEU，再对 $p_n$ 做加法平滑或其他指定平滑；本题按课件原始公式不平滑，所以 4-gram 为 0 时 BLEU 直接为 0。

---

### 三、模型分析题

#### 10. 朴素贝叶斯参数估计与分类（10 分）

**（1）先验**

$$P(\text{教育}) = 2/4 = 0.5,\quad P(\text{体育}) = 2/4 = 0.5$$

**（2）拉普拉斯平滑条件概率**

公式（$M=5$ 个特征）：

$$P(w_i \mid c_j) = \frac{1 + N(w_i, c_j)}{M + \sum_{i'=1}^{M} N(w_{i'}, c_j)}$$

**教育类**特征计数：大学 2、计算机 2、高校 1、排球 0、运动会 0，合计 5：

| 特征 | $P(w \mid \text{教育})$ |
|------|-------------------------|
| 大学 | $(1+2)/(5+5) = 3/10$ |
| 计算机 | $(1+2)/(5+5) = 3/10$ |
| 高校 | $(1+1)/(5+5) = 2/10$ |
| 排球 | $(1+0)/(5+5) = 1/10$ |
| 运动会 | $(1+0)/(5+5) = 1/10$ |

**体育类**特征计数：大学 1、排球 2、运动会 2、计算机 0、高校 0，合计 5：

| 特征 | $P(w \mid \text{体育})$ |
|------|-------------------------|
| 大学 | $(1+1)/(5+5) = 2/10$ |
| 排球 | $(1+2)/(5+5) = 3/10$ |
| 运动会 | $(1+2)/(5+5) = 3/10$ |
| 计算机 | $(1+0)/(5+5) = 1/10$ |
| 高校 | $(1+0)/(5+5) = 1/10$ |

**（3）分类测试文档「大学 计算机」**

$$
\begin{aligned}
P(\text{教育}) \prod P(w \mid \text{教育}) &\propto 0.5 \times \frac{3}{10} \times \frac{3}{10} = 0.045 \\
P(\text{体育}) \prod P(w \mid \text{体育}) &\propto 0.5 \times \frac{2}{10} \times \frac{1}{10} = 0.010
\end{aligned}
$$

**预测类别：教育**。

---

#### 11. RNN 训练过程（8 分）

**（1）SRNN 更新公式**

$$\mathbf{h}_t = f(\mathbf{U}\mathbf{h}_{t-1} + \mathbf{W}\mathbf{x}_t + \mathbf{b})$$

$\mathbf{U}$：隐状态递归权重；$\mathbf{W}$：输入到隐状态权重；$\mathbf{b}$：偏置；$f$ 为非线性激活（如 $\tanh$）。

**（2）BPTT 思路**

1. **前向**：按 $t=1,\ldots,T$ 递推计算 $\mathbf{h}_t$ 与输出、损失 $L$。
2. **展开**：将循环网络沿时间轴展开为深层前馈网络。
3. **反向**：从 $t=T$ 向 $t=1$ 反向传播，用链式法则求 $\partial L / \partial \mathbf{U}, \mathbf{W}, \mathbf{b}$。
4. 远距离梯度含连乘项 $\prod_{\tau=k+1}^{t} (\mathbf{U}^\top \text{diag}(f'(\mathbf{z}_\tau)))$。

**（3）梯度问题与应对**

| 问题 | 成因 | 应对 |
|------|------|------|
| **梯度消失** | $\|\gamma\| < 1$ 时连乘趋 0 | GRU/LSTM；残差/线性依赖结构 |
| **梯度爆炸** | $\|\gamma\| > 1$ 时连乘指数增长 | **梯度截断**（Gradient Clipping）；权重衰减 |

---

#### 12. TextCNN vs BiLSTM（7 分）

**（1）TextCNN 四步骤**（Kim, 2014）

1. 句子 → **词向量矩阵**；
2. 多种 **kernel_size** 的 Conv1d 提取 n-gram 特征；
3. **ReLU + Global Max-pooling**；
4. 拼接多尺度特征 → **全连接 + Softmax** 分类。

**（2）比较**

| 方面 | TextCNN | BiLSTM |
|------|---------|--------|
| 并行性 | 卷积可并行，训练快 | 时间步串行，难并行 |
| 长程依赖 | 感受野有限（靠多层/大核） | 理论上可建模长依赖 |
| 关键特征 | Max-pool 抓最显著 n-gram | 信息经多步传递可能被稀释 |

**（3）Global Max-pooling 理由**

情感等句子级任务常由**少数关键短语**（如 "not bad at all"）决定；Max-pooling 在全句范围选取**最强 n-gram 响应**，比 RNN 末态更能保留 salient 特征，且结构更简单高效。

---

### 四、开放式设计题

#### 13. 自动对下联系统（10 分）

**参考答案要点**（言之有理即可，以下为参考方案）：

**（1）形式化**

- 输入：上联字符序列 $\mathbf{s} = (c_1, \ldots, c_n)$。
- 输出：下联 $\mathbf{t} = (d_1, \ldots, d_n)$，长度与上联相同，满足对仗与格律约束。

**（2）模型选型**

| 方案 | 说明 | 优劣 |
|------|------|------|
| **Seq2Seq + Attention** | 上联作 Encoder 输入，Decoder 生成下联 | 实现成熟，可解释对齐；长联性能有限 |
| **Transformer Encoder-Decoder** | 与上类似，并行训练 | 长程依赖更好，需更多数据 |
| **预训练 + 微调**（如 T5/GPT） | `对下联：{上联}` → 下联 | 利用大模型语言知识，需领域微调 |

推荐：**Transformer Seq2Seq** 或 **T5 Text-to-Text 微调**。

**（3）数据与损失**

- 数据：大规模**平行对联语料**（上联–下联对），可爬取/人工标注。
- 损失：交叉熵（逐字生成）；可加**复制机制**处理专有名词。
- 可选辅助任务：词性对齐损失、押韵损失。

**（4）推理约束**

1. **字数对齐**：强制输出长度 = 输入长度。
2. **对仗约束**：名词对名词、动词对动词（可用词性模板或对齐词典）。
3. **平仄/押韵**：Beam Search 中对违反格律的候选减分。
4. **重复惩罚**：避免与上联重复字词。

---

#### 14. Transformer 机器翻译（5 分）

**（1）Masked Self-Attention**

- 位置 $i$ 只能 attend 到 $\leq i$ 的位置，**不能看未来 token**。
- 训练时用 Teacher Forcing + Mask，与自回归推理一致，避免"偷看答案"。
- 推理时逐步生成，每步仅依赖已生成前缀。

**（2）Cross-Attention**

- **Q ← Decoder** 当前状态；**K, V ← Encoder** 输出。
- 作用：建立**源–目标对齐**，生成每个目标词时动态查询源句最相关片段。

**（3）其他评估**

- **ROUGE**（摘要）、**BERTScore**（语义相似度）、**COMET**（神经指标）、人工 **MOS** 等。

---

## 附录：难度与章节覆盖对照

| 题号 | 分值 | 难度 | 章节 | 类型 |
|------|------|------|------|------|
| 1 | 5 | easy | Ch1 | 学长考点 |
| 2 | 5 | easy | Ch3 | 学长考点 |
| 3 | 5 | easy | Ch5 | 概念 |
| 4 | 5 | easy | Ch8 | 学长考点（T5） |
| 5 | 8 | medium | Ch4 | 课堂练习 |
| 6 | 8 | medium | Ch4 | 课堂练习 |
| 7 | 8 | medium | Ch6 | 课堂练习 |
| 8 | 8 | medium | Ch8 | 课堂练习 |
| 9 | 8 | medium | Ch8 | 学长考点（BLEU） |
| 10 | 10 | medium | Ch3 | 课堂练习 |
| 11 | 8 | hard | Ch7 | 学长考点（RNN） |
| 12 | 7 | hard | Ch6–7 | 学长考点（CNN） |
| 13 | 10 | hard | 综合 | 学长考点（设计） |
| 14 | 5 | hard | Ch8 | 开放式 |

**课堂练习改编计算题**：第 5、6、7、8、10 题（共 5 道）。  
**未考查**：8.4.4 节内容。
