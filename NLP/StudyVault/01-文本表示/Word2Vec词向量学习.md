---
source_pdf: NLP课件_jfyu_第五章_文本表示_V2.pdf
part: 5.3.1
keywords: word2vec, CBOW, skip-gram, word embedding, neural language model
---

# Word2Vec 词向量学习（★★★）

#text-representation #word2vec #cbow #skip-gram #word-embedding #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 核心思想 | "You shall know a word by the company it keeps" (Firth 1957) |
| Look-up Table | 词向量矩阵 $L \in \mathbb{R}^{D \times V}$，$\mathbf{x} = L\mathbf{e}$（e为one-hot） |
| V 确定方法 | 全部词/频率阈值/Top-V高频词 |
| D 确定方法 | 超参数，一般几十到几百维 |
| 学习方法 | 随机初始化 + 目标函数优化（如最大化语言模型似然度） |

## 基于语言模型的词向量学习 (Bengio 2003)

- **输入**：前 n-1 个词的词向量拼接
- **网络结构**：Table look-up → Hidden(tanh) → Softmax
- **目标**：最大化 P(wₜ|wₜ₋₁...wₜ₋ₙ₊₁)
- **公式**：y = U·tanh(Hx + d) + Wx + b
- **副产品**：训练好的 L 矩阵即为词向量

> [!warning] 计算瓶颈
> Softmax层需要对整个词表V计算归一化，当V很大时计算代价极高

## C&W 模型 (Collobert & Weston)

- **思想**：判别式学习，不直接预测词语，而是打分判断正确性
- **方法**：将正确的n-gram与随机替换中心词的n-gram对比
- **目标函数**：Hinge Loss
  - loss = Σ Σ max(0, 1 + score(w'ᵢ,C) - score(wᵢ,C))
- **优势**：避免了softmax的全词表计算

## CBOW 模型 (Continuous Bag-of-Words)

- **输入**：上下文词向量的**平均**（词序不影响）
- **预测目标**：中心词 wᵢ
- **隐层**：h = (1/2C) × Σ e(wₖ)
- **概率**：P(wᵢ|W_C) = exp{h · e(wᵢ)} / Σₖ exp{h · e(wₖ)}
- **目标函数**：L* = argmax_L Σ log P(wᵢ|W_C)

## Skip-Gram 模型

- **输入**：中心词 wᵢ 的词向量
- **预测目标**：所有上下文词 W_C = {wᵢ₋C, ..., wᵢ₋₁, wᵢ₊₁, ..., wᵢ₊C}
- **概率**：P(w_C|wᵢ) = exp{e(wᵢ)·e(w_C)} / Σₖ exp{e(wᵢ)·e(wₖ)}
- **目标函数**：L* = argmax_L Σ_{wᵢ∈V} Σ_{w_C∈W_C} log P(w_C|wᵢ)

```
CBOW vs Skip-Gram 对比:
┌─────────────┬──────────────────┬──────────────────┐
│             │     CBOW         │   Skip-Gram      │
├─────────────┼──────────────────┼──────────────────┤
│ 输入        │ 上下文词(多→1)   │ 中心词(1→多)     │
│ 预测        │ 中心词           │ 上下文词         │
│ 特点        │ 训练速度快       │ 低频词效果好     │
│ 词序        │ 忽略词序         │ 忽略词序         │
└─────────────┴──────────────────┴──────────────────┘
```

> [!important] CBOW/Skip-Gram共同缺陷
> 只使用**局部上下文信息**，未利用语料的整体分布（共现统计）信息

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "CBOW预测方向" | 上下文 → 中心词 |
| "Skip-Gram预测方向" | 中心词 → 上下文 |
| "C&W loss类型" | Hinge Loss (margin-based) |
| "Bengio模型计算瓶颈" | Softmax层的全词表归一化 |
| "词向量维度D如何确定" | 超参数，人工设定 |

## 相关笔记
- [[向量空间模型与离散表示]]
- [[GloVe与其他词表示方法]]
- [[短语与句子表示学习]]
