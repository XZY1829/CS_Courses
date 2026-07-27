---
source_pdf: NLP课件_jfyu_第五章_文本表示_V2.pdf
part: 5.3.1
keywords: GloVe, co-occurrence matrix, character-word hybrid
---

# GloVe 与其他词表示方法（★★）

#text-representation #glove #word-embedding #co-occurrence #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| GloVe全称 | Global Vectors for Word Representation |
| 核心动机 | 同时利用局部上下文和全局共现统计信息 |
| 核心思想 | 词向量内积应反映共现矩阵中的统计关系 |
| 权重函数 | f(Xᵢⱼ) 使不同共现频率有不同权重 |
| 字-词混合 | 结合词级和字级表示获得更优表示 |

## GloVe 模型

### 基本定义
- **Xᵢⱼ**：单词j出现在单词i上下文的次数
- **Xᵢ** = Σₖ Xᵢₖ：单词i上下文中所有词的总次数
- **Pᵢⱼ** = P(j|i) = Xᵢⱼ/Xᵢ：条件概率

### 推导过程

1. 目标：找到函数F使得 F(e(wᵢ), e(wⱼ), ẽ(wₖ)) = Pᵢₖ/Pⱼₖ
2. 线性化：F(e(wᵢ) - e(wⱼ), ẽ(wₖ)) = Pᵢₖ/Pⱼₖ
3. 内积形式：F((e(wᵢ) - e(wⱼ))ᵀ · ẽ(wₖ)) = Pᵢₖ/Pⱼₖ
4. 同态性要求：F = exp → e(wᵢ)ᵀẽ(wₖ) = log(Pᵢₖ)
5. 加入偏置项：**e(wᵢ)ᵀẽ(wₖ) + bᵢ + b̃ₖ = log(Xᵢₖ)**

### 目标函数
- J = Σᵢ,ⱼ f(Xᵢⱼ) × (e(wᵢ)ᵀẽ(wⱼ) + bᵢ + b̃ⱼ - log Xᵢⱼ)²
- 权重函数：f(x) = (x/x_max)^α if x < x_max, else 1
- 超参数：x_max = 100, α = 3/4

> [!tip] GloVe vs Word2Vec
> - Word2Vec（CBOW/Skip-Gram）：只用局部窗口信息
> - GloVe：同时利用全局共现统计 + 局部窗口
> - GloVe本质上是对共现矩阵的加权最小二乘分解

## 字-词混合表示学习

- **动机**：词语由字构成，可以从字的组合获得额外表示
- **方法1**：字向量平均/拼接后与词向量结合（AVE/CON方式）
- **方法2**：门控机制（Gating）控制字/词信息融合比例
- **代表**：FastText（利用子词n-gram信息）

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "GloVe与Word2Vec区别" | GloVe同时利用全局+局部信息 |
| "GloVe目标函数类型" | 加权最小二乘 |
| "权重函数作用" | 避免高频共现对(如 the-is)主导损失 |
| "共现矩阵Xᵢⱼ含义" | 单词j出现在单词i上下文窗口的次数 |

## 相关笔记
- [[Word2Vec词向量学习]]
- [[向量空间模型与离散表示]]
