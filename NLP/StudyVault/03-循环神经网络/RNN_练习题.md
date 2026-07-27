---
source_pdf: NLP课件_jfyu_第七章_循环神经网络_release.pdf
part: 7
keywords: practice, rnn, srnn, bptt, gru, lstm, attention, ner
---

# RNN 练习（12题）

#practice #rnn

## 相关概念
- [[RNN基础与参数学习]]
- [[GRU与LSTM]]
- [[深度RNN与递归神经网络]]
- [[注意力机制]]

> [!hint]- 核心模式（点击查看）
> | 关键词 | 答案 |
> |--------|------|
> | SRNN 公式 | hₜ = f(U·hₜ₋₁ + W·xₜ + b) |
> | BPTT 梯度 | δₜ,ₖ = δₜ,ₜ × Π(Uᵀf'(zτ)) |
> | 梯度消失 | $|\gamma|<1$ → $\gamma^{(t-k)}$ 趋 0 |
> | GRU 更新 | hₜ = z⊙hₜ₋₁ + (1-z)⊙h̃ₜ |
> | LSTM cell | cₜ = f⊙cₜ₋₁ + i⊙c̃ₜ |
> | 注意力 | α=softmax(s(·,q))；att=Σαₙxₙ |
> | NER F1 | 实体级边界+类型精确匹配 |
> | SQuAD | EM（完全匹配）+ Token F1 |

---

## 第 1 题 - SRNN 更新公式 [recall]
> 写出简单循环网络（SRNN）的隐状态更新公式，并说明 U、W、b 的含义。

> [!answer]- 查看答案
> **hₜ = f(U·hₜ₋₁ + W·xₜ + b)**
> - **U**：隐状态到隐状态的递归权重矩阵
> - **W**：输入到隐状态的权重矩阵
> - **b**：偏置向量
> - **f**：非线性激活函数（如 tanh）
> - 三者均在所有时间步**共享**。

---

## 第 2 题 - 注意力计算 [recall]
> 给定输入序列 X = {x₁, x₂, x₃} 和 query q，写出软性注意力的完整计算步骤（含打分函数 s(x, q) = xᵀq）。

> [!answer]- 查看答案
> 1. **打分**：s₁ = x₁ᵀq, s₂ = x₂ᵀq, s₃ = x₃ᵀq
> 2. **归一化**：α = softmax([s₁, s₂, s₃])，得 α₁ + α₂ + α₃ = 1
> 3. **加权求和**：**att(X, q) = α₁x₁ + α₂x₂ + α₃x₃**
> 结果 att 是上下文向量，供 Decoder 或分类器使用。

---

## 第 3 题 - BPTT 梯度公式 [recall]
> 写出 BPTT 中从时刻 t 回传到时刻 k（t > k）时，损失对 hₖ 的梯度表达式，并解释连乘项的物理含义。

> [!answer]- 查看答案
> $\delta_{t,k} = \frac{\partial L_t}{\partial h_k} = \delta_{t,t} \times \prod_{\tau=k+1}^{t} (U^T \cdot \mathrm{diag}(f'(z_\tau)))$
> 连乘项表示误差信号从 t 逐步回传到 k 时，每步经过 Uᵀ 和激活函数导数的缩放。多次连乘导致梯度随距离指数级衰减（消失）或增长（爆炸）。

---

## 第 4 题 - 梯度消失与爆炸 [recall]
> 解释 SRNN 中梯度消失和梯度爆炸的原因，并分别给出一种应对方法。

> [!answer]- 查看答案
> **原因**：BPTT 梯度含 $\gamma^{(t-k)}$ 因子。当 $|\gamma| < 1$ 时远距离梯度趋 0（#gradient-vanishing）；当 $|\gamma| > 1$ 时梯度指数增长（#gradient-exploding）。
> - **爆炸应对**：**梯度截断**（Gradient Clipping）或权重衰减
> - **消失应对**：改进模型结构（**GRU/LSTM**）或线性依赖改进 hₜ = hₜ₋₁ + g(xₜ, hₜ₋₁; θ)

---

## 第 5 题 - GRU 公式与门的作用 [recall]
> 写出 GRU 的四个核心公式，并说明更新门 zₜ 和重置门 rₜ 各自的作用。

> [!answer]- 查看答案
> - **zₜ = σ(Wz·xₜ + Uz·hₜ₋₁ + bz)**（更新门）
> - **rₜ = σ(Wr·xₜ + Ur·hₜ₋₁ + br)**（重置门）
> - **h̃ₜ = tanh(Wh·xₜ + Uh·(rₜ ⊙ hₜ₋₁) + bh)**（候选状态）
> - **hₜ = zₜ ⊙ hₜ₋₁ + (1 - zₜ) ⊙ h̃ₜ**（最终状态）
> - **更新门 zₜ**：控制保留旧状态 vs 接受新候选的比例（z→1 多保留，z→0 多更新）
> - **重置门 rₜ**：控制计算候选时是否"忽略"历史（r→0 仅用 xₜ，r→1 正常融合 hₜ₋₁）

---

## 第 6 题 - LSTM 公式 [recall]
> 写出 LSTM 中 cell state 和 hidden state 的更新公式，并说明遗忘门 fₜ 的作用。

> [!answer]- 查看答案
> - **cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ**（cell state 更新）
> - **hₜ = oₜ ⊙ tanh(cₜ)**（hidden state 输出）
> - **遗忘门 fₜ**：控制上一时刻 cell state **cₜ₋₁ 有多少被保留**。f→1 完全保留，f→0 完全遗忘。加法结构使梯度可沿 cell state 直接传播，缓解梯度消失。

---

## 第 7 题 - 双向 RNN [recall]
> 双向 RNN 如何合并前向和后向的隐状态？它适用于什么类型的任务，有什么局限？

> [!answer]- 查看答案
> **合并方式**：拼接 **[h⃗ₜ ; h⃖ₜ]** 或 element-wise 求和/平均。
> **适用任务**：需要**完整上下文**的离线任务，如 NER、POS 标注、机器阅读理解。
> **局限**：需要**完整输入序列**才能运行后向 RNN，**无法用于在线/流式预测**（未来信息不可得）。

---

## 第 8 题 - NER 标注与评估 [application]
> 对句子"张三在北京大学工作"进行 BIO 标注（PER=人名，ORG=机构）。若模型预测"张三"为 PER（正确）但"北京"为 ORG（漏掉"大学"），计算该句的实体级 Precision、Recall 和 F1。

> [!answer]- 查看答案
> **标准实体**：PER: "张三"；ORG: "北京大学"（共 2 个）
> **预测实体**：PER: "张三"（正确）；ORG: "北京"（边界错误，应为"北京大学"）
> - **TP = 1**（"张三"完全匹配）
> - **FP = 1**（"北京"预测错误）
> - **FN = 1**（"北京大学"未被正确预测）
> - **Precision = 1/2 = 0.5**
> - **Recall = 1/2 = 0.5**
> - **F1 = 2×0.5×0.5 / (0.5+0.5) = 0.5**

---

## 第 9 题 - SQuAD 评估 [application]
> 标准答案为 "the American Civil War"，模型分别预测 "American Civil War" 和 "Civil War"，求各自的 EM 和 F1 分数。

> [!answer]- 查看答案
> **预测 "American Civil War"**：
> - EM = 0（不完全匹配，缺少 "the"）
> - F1：标准 token {the, American, Civil, War}，预测 {American, Civil, War}，交集 3，P=3/3=1.0, R=3/4=0.75, **F1 = 2×1.0×0.75/(1.0+0.75) ≈ 0.857**
> **预测 "Civil War"**：
> - EM = 0
> - F1：交集 {Civil, War} = 2，P=2/2=1.0, R=2/4=0.5, **F1 = 2×1.0×0.5/(1.0+0.5) ≈ 0.667**

---

## 第 10 题 - 缩放点积注意力 [application]
> 向量维度 D=512 时，为什么点积注意力 s(x,q)=xᵀq 需要除以 √D？若不缩放会有什么后果？

> [!answer]- 查看答案
> 当 D 较大时，x 和 q 各分量独立且方差有限，xᵀq 的方差**随 D 线性增长**（≈ D·Var(xᵢ)·Var(qᵢ)）。D=512 时 dot product 值很大，softmax 进入**饱和区**（接近 one-hot），梯度接近 0。
> **缩放点积** s(x,q) = xᵀq/√D 将方差稳定在 O(1)，使 softmax 保持有效梯度。这是 Transformer 中 #scaled-dot-product 的标准做法。

---

## 第 11 题 - GRU vs LSTM 选型 [analysis]
> 在数据量有限（10 万句）、序列平均长度 30 的中等规模 NER 任务上，你会优先选择 GRU 还是 LSTM？请从参数量、训练效率和长程依赖能力三个角度分析。

> [!answer]- 查看答案
> 优先选择 **GRU**。
> - **参数量**：GRU 仅 2 门、无独立 cell state，参数约为 LSTM 的 3/4，10 万句数据下更不易过拟合
> - **训练效率**：GRU 计算更简单，收敛通常更快，适合资源有限场景
> - **长程依赖**：平均长度 30 属于中等序列，GRU 的更新门已足够捕获此类依赖；LSTM 的三门+cell state 优势在**更长序列**（100+）时更明显
> - **结论**：NER 还需 BiLSTM/BiGRU + CRF，GRU 在效率与性能间更平衡；若验证集上 GRU 明显欠拟合，再换 LSTM。

---

## 第 12 题 - RvNN 与 RNN 关系 [analysis]
> 为什么说 RNN 是 RvNN 的特例？RvNN 在情感分析任务上相比链式 RNN 有什么潜在优势？

> [!answer]- 查看答案
> **RNN 是 RvNN 特例**：当递归结构**退化为链**（每个内部节点只有一个子节点延续，形成线性链）时，RvNN 的自底向上合并等价于 RNN 的时间步递推。
> **RvNN 优势**：
> 1. **句法感知组合**：按句法树合并，"not very good" 中否定词可正确翻转子树情感
> 2. **短语级监督**：SST 等数据集提供短语级标签，RvNN 可在中间节点计算损失
> 3. **组合语义**：MV-RNN/RNTN 通过矩阵/张量捕获修饰关系，表达力超过纯链式结构
> **劣势**：依赖外部句法分析器，错误会传播；训练复杂度更高。

---

> [!summary]- 模式总结（点击查看）
> | 关键词 | 答案 |
> |--------|------|
> | SRNN | hₜ = f(Uhₜ₋₁ + Wxₜ + b) |
> | BPTT | 时间展开 + 反向传播；连乘致梯度问题 |
> | 梯度爆炸 | 截断 / 权重衰减 |
> | 梯度消失 | GRU/LSTM / 线性依赖 |
> | GRU 核心 | z 更新门 + r 重置门；凸组合 hₜ |
> | LSTM 核心 | f/i/o 三门 + cell state 加法更新 |
> | 双向 RNN | [h⃗ ; h⃖]；离线任务；不能流式 |
> | 注意力 | softmax(s(·,q)) → Σαₙxₙ |
> | 缩放点积 | xᵀq/√D |
> | NER 评估 | 实体级 P/R/F1，边界+类型精确匹配 |
> | SQuAD | EM + Token F1 |
> | RNN ⊂ RvNN | 链 = 树的退化结构 |
