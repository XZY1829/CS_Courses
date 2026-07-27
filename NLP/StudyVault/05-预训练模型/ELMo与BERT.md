---
source_pdf: NLP课件_jfyu_第八章_8-2_预训练模型_V2.pdf
part: 8.4.1
keywords: elmo, bert, mlm, nsp, self-supervised, contextualized-embedding, fine-tuning, glue
---

# ELMo 与 BERT（★★★）

#nlp-deep-learning #pretrained-model #elmo #bert #mlm #nsp #self-supervised #masked-lm #next-sentence #fine-tuning #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| 自监督学习 | 监督学习框架 + **无标注数据**自动构造训练样本 |
| 静态词向量缺陷 | Word2Vec/GloVe **一词一向量**，无法处理**一词多义** |
| ELMo | 双向 RNN LM；多层加权 **α₁h₁+α₂h₂**；**上下文相关**词向量 |
| BERT | Transformer **Encoder**；**MLM** + **NSP** 预训练 |
| SpanBERT | 预测**连续 span**（MLM 变体） |
| 微调范式 | **预训练 → 微调 → 下游任务** |
| BERT-Base | 12 层 / 768d / 12 头 / **110M** |
| BERT-Large | 24 层 / 1024d / 16 头 / **340M** |
| 评估 | **GLUE** benchmark 多任务 |

## 自监督学习（Self-Supervised Learning）

**自监督学习**是预训练语言模型的基础范式：

- 使用**监督学习**的训练方式（定义输入-标签、计算损失、反向传播）
- 但标签从**无标注语料**中**自动构造**，无需人工标注

```
原始语料:  "The cat sat on the mat"
自动任务:  遮盖 "cat" → 预测 "cat"  (MLM)
           或预测下一个词           (LM)
```

| 对比 | 监督学习 | 自监督学习 |
|------|----------|-----------|
| 标签来源 | 人工标注 | **从数据本身构造** |
| 数据规模 | 有限 | **海量无标注文本** |
| 代表任务 | 情感分类 | MLM、NSP、CLM |

> [!important] 核心思想
> 利用语言本身的结构作为监督信号，先在大规模语料上学习通用表示，再迁移到下游任务。

---

## 传统词向量的缺陷

Word2Vec、GloVe 等学习**静态词向量（Static Word Embedding）**：

| 问题 | 说明 | 示例 |
|------|------|------|
| **一词一向量** | 每个词只有一个固定表示 | "bank" 在"河岸"和"银行"中向量相同 |
| **无法一词多义** | 忽略上下文 | "He went to the **bank**" vs "a **bank** loan" |
| **无上下文感知** | 词向量查表，与句子无关 | 无法区分 polysemy |

这推动了 **上下文相关词向量（Contextualized Word Embedding）** 的发展。

---

## ELMo（Embeddings from Language Models）

**ELMo** — Peters et al., **NAACL 2018 Best Paper**

### 架构

- 基于**双向 RNN 语言模型**（前向 LM + 后向 LM）
- 每个词的最终表示来自**多层 RNN 隐状态的加权组合**

**ELMo 词向量 = α₁·h₁ + α₂·h₂ + … + α_L·h_L**

其中 **α** 权重由**下游任务学习**（非固定）。

```
前向 LM:  w₁ → w₂ → w₃ → ...  (预测 w_{t+1})
后向 LM:  ... ← w₃ ← w₂ ← w₁  (预测 w_{t-1})
                ↓
         多层 h 加权 → 上下文相关 embedding
```

### 关键特点

| 特点 | 说明 |
|------|------|
| **上下文相关** | 同一词在不同句子中向量不同 |
| **多层融合** | 浅层偏语法，深层偏语义（经验观察） |
| **特征型使用** | 作为**固定特征**输入下游模型（非端到端微调为主） |
| **双向** | 前向+后向 LM，但**非深度双向**（非同一层同时看左右） |

> [!warning] ELMo vs 真双向
> ELMo 是**两个独立单向 LM 的拼接**，不同于 BERT 在同一层内真正双向 attend。

---

## BERT（Bidirectional Encoder Representations from Transformers）

**BERT** — Devlin et al., **NAACL 2019 Best Paper**

### 架构

- 基于 **Transformer Encoder**（双向 Self-Attention）
- 同一层内每个 token **同时看到左右上下文**

### 预训练任务 1：MLM（Masked Language Model）

随机遮盖输入中 **15%** 的 token，预测被遮盖词：

| 15% 被选中 token 的处理 | 比例 |
|-------------------------|------|
| 替换为 **[MASK]** | 80% |
| 替换为**随机词** | 10% |
| **保持不变** | 10% |

> [!tip] 为何不全用 [MASK]
> 微调阶段无 [MASK] token，纯 [MASK] 会造成预训练-微调不一致；随机词和保持不变缓解此问题。

### 预训练任务 2：NSP（Next Sentence Prediction）

给定句子对 (A, B)，预测 B 是否为 A 的**下一句**（50% 真相邻，50% 随机句）。

- 学习**句子级关系**，对 QA、NLI 等任务有帮助
- 后续研究（如 RoBERTa）发现 NSP 收益有限，但 BERT 原论文包含此任务

### SpanBERT（MLM 变体）

- 不随机 mask 单个 token，而是 mask **连续 span**
- 更好地建模**短语级**语义和边界

---

## BERT 微调范式

```
阶段 1: 预训练（大规模无标注语料）
         MLM + NSP → 通用语言表示

阶段 2: 微调（下游标注数据）
         加任务头（分类/序列标注/QA span）
         端到端训练全部参数

阶段 3: 下游任务推理
```

| 下游任务 | 常用做法 |
|----------|----------|
| **文本分类** | 取 [CLS] token 表示 → 分类层 |
| **序列标注** | 每个 token 的 BERT 输出 → 标注层 |
| **问答 QA** | 预测答案 span 的起止位置 |
| **句对任务** | [CLS] 或 token 级表示 → 分类 |

> [!important] 范式转变
> BERT 确立了 **"预训练 + 微调"** 范式：同一预训练模型通过微调适配多种 NLP 任务。

---

## BERT 配置

| 版本 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| **BERT-Base** | 12 | 768 | 12 | **~110M** |
| **BERT-Large** | 24 | 1024 | 16 | **~340M** |

---

## GLUE Benchmark

**GLUE (General Language Understanding Evaluation)** 是评估自然语言**理解**能力的标准基准，包含多个子任务：

| 任务类型 | 示例任务 | 评估能力 |
|----------|----------|----------|
| 单句分类 | SST-2（情感） | 句子级理解 |
| 句对分类 | MRPC、QQP | 语义等价/相似 |
| 推理 | MNLI、RTE | 自然语言推理 |
| 语言学 | CoLA | 语法 acceptability |
| 评分 | STS-B | 语义相似度回归 |

BERT 在 GLUE 上大幅刷新 SOTA，证明了预训练+微调的有效性。

---

## ELMo vs BERT 对比

| 维度 | ELMo | BERT |
|------|------|------|
| 骨干网络 | 双向 RNN LM | Transformer Encoder |
| 双向性 | 两层独立单向 LM | 单层真正双向 Self-Attn |
| 预训练任务 | 语言建模（预测上下词） | MLM + NSP |
| 使用方式 | 主要作**特征** | **端到端微调** |
| 参数量 | ~94M | Base 110M / Large 340M |
| 发表 | NAACL 2018 | NAACL 2019 |

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "自监督学习定义" | 监督框架 + 无标注数据自动构造标签 |
| "静态词向量问题" | 一词一向量，无法一词多义 |
| "ELMo 表示公式" | α₁h₁ + α₂h₂ + …，权重下游学习 |
| "BERT 预训练任务" | MLM（15% mask）+ NSP |
| "MLM mask 策略" | 80% [MASK], 10% 随机, 10% 不变 |
| "BERT-Base 配置" | 12层/768d/12头/110M |
| "微调范式" | 预训练 → 微调 → 下游 |
| "GLUE 作用" | 多任务 NLU 评估基准 |

## 相关笔记
- [[T5_BART与GPT系列]]
- [[大语言模型与多模态]]
- [[预训练模型_练习题]]
- [[Transformer架构]]
