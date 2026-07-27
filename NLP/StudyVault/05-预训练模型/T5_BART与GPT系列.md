---
source_pdf: NLP课件_jfyu_第八章_8-2_预训练模型_V2.pdf
part: 8.4.2-8.4.3
keywords: t5, bart, gpt, chatgpt, rlhf, in-context-learning, span-prediction, fine-tuning
---

# T5、BART 与 GPT 系列（★★★）

#nlp-deep-learning #pretrained-model #t5 #bart #gpt #chatgpt #rlhf #in-context-learning #span-prediction #fine-tuning #concept

## 概览表（一目了然）
| 条目 | 要点 |
|------|------|
| T5 | **Text-to-Text** 统一框架；Seq2Seq；**Masked Span Prediction** |
| BART | **Denoising Seq2Seq**；文档损坏+重构 |
| GPT | Transformer **Decoder** + Masked SA + **CLM** |
| GPT-3 | **175B**；**In-context Learning**（Zero/One/Few-shot） |
| ChatGPT | **SFT + RLHF** |
| GPT-4 | **多模态** |
| 大模型三阶段 | **预训练 → SFT → RLHF** |
| 规模演进 | ELMo(94M) → BERT(340M) → GPT-2(1.5B) → GPT-3(175B) |

## T5（Text-to-Text Transfer Transformer）

**T5** — Raffel et al., 2020

### 核心思想：统一为 Text-to-Text

将所有 NLP 任务统一建模为**文本输入 → 文本输出**：

| 任务 | 输入文本 | 输出文本 |
|------|----------|----------|
| 翻译 | `translate English to German: That is good.` | `Das ist gut.` |
| 分类 | `sst2 sentence: This is great!` | `positive` |
| 摘要 | `summarize: ...长文档...` | `...摘要...` |

> [!important] 统一建模
> 同一 Seq2Seq 架构 + 同一损失函数，仅通过**任务前缀**区分任务类型。

### 架构与预训练

- **Encoder-Decoder**（标准 Transformer Seq2Seq）
- 预训练任务：**Masked Span Prediction**
  - 随机遮盖连续 span，替换为 sentinel token
  - 模型预测被遮盖的 span 文本

### 模型规模

| 版本 | 参数量 |
|------|--------|
| T5-Small | ~60M |
| T5-Base | ~220M |
| T5-Large | ~770M |
| T5-11B | **11B** |

---

## BART（Bidirectional and Auto-Regressive Transformers）

**BART** — Lewis et al., 2020

### 核心思想：Denoising Seq2Seq Pre-training

对文档施加多种**损坏（Corruption）**，训练模型**重构**原文：

| 损坏方式 | 说明 |
|----------|------|
| **Token Masking** | 随机 mask token |
| **Token Deletion** | 随机删除 token |
| **Text Infilling** | 遮盖 span（类似 T5） |
| **Sentence Permutation** | 打乱句子顺序 |
| **Document Rotation** | 旋转文档片段 |

```
损坏文档 ──→ [Encoder] ──→ 表示 ──→ [Decoder] ──→ 重构原文
```

### 架构

- **Encoder**：双向（类似 BERT）
- **Decoder**：自回归（类似 GPT）
- 预训练 = **去噪自编码**

### 下游任务适配

| 任务 | 做法 |
|------|------|
| **分类** | 取 Decoder **最后一个 token** 的表示 → 分类头 |
| **翻译** | 额外添加 **Language Identifier Embedding** 到 Encoder |
| **生成** | 直接使用 Decoder 自回归生成 |

---

## GPT 系列（Generative Pre-Training）

### GPT-1

- **Transformer Decoder** + **Masked Self-Attention**
- 预训练：**因果语言建模（CLM）**——预测下一个 token
- 微调：加任务头适配下游

```
GPT:  单向（只看左侧上下文）
BERT: 双向 Encoder（看全上下文）
```

### GPT-2（1.5B）

- 更大规模、更大数据
- 展示**零样本**迁移能力（无需微调即可执行部分任务）
- 强调 **LM 即多任务学习**

### GPT-3（175B）

- 规模跃升至 **1750 亿**参数
- 核心能力：**In-context Learning（上下文学习）**

| 模式 | 示例 | 说明 |
|------|------|------|
| **Zero-shot** | 仅任务描述，无示例 | 直接生成答案 |
| **One-shot** | 1 个输入-输出示例 | 从示例推断模式 |
| **Few-shot** | 少量示例（通常 k=10~100） | 上下文演示学习 |

> [!important] 与微调的区别
> In-context Learning **不更新模型参数**，仅通过 prompt 中的示例和指令引导生成。

### ChatGPT

在 GPT 基础上增加对齐训练：

1. **SFT（Supervised Fine-Tuning）**：人工标注对话数据微调
2. **RLHF（Reinforcement Learning from Human Feedback）**：人类偏好奖励模型 + 强化学习优化

### GPT-4

- 更大规模、更强推理
- **多模态**：支持图像输入（与文本联合理解）

---

## 大语言模型实现三阶段

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  预训练      │ →  │  SFT        │ →  │  RLHF       │
│  (Pretrain)  │    │  (指令微调)  │    │  (人类对齐)  │
└─────────────┘    └─────────────┘    └─────────────┘
  海量无标注文本      高质量指令-回复对     人类偏好排序+奖励模型
  CLM/MLM 等         监督学习             PPO 等 RL 优化
```

| 阶段 | 目标 | 数据 |
|------|------|------|
| **预训练** | 学习语言知识、世界知识 | 互联网规模文本 |
| **SFT** | 学会遵循指令、对话格式 | 人工标注指令数据 |
| **RLHF** | 输出更符合人类偏好（有用、安全、诚实） | 人类对比排序 + 奖励模型 |

---

## 预训练模型规模对比

```
ELMo (94M) ──→ BERT-Large (340M) ──→ GPT-2 (1.5B) ──→ GPT-3 (175B)
   │                  │                    │                  │
  RNN LM          Transformer Enc      Transformer Dec    ICL 涌现
```

| 模型 | 参数量 | 架构 | 预训练任务 | 使用范式 |
|------|--------|------|-----------|----------|
| ELMo | 94M | Bi-RNN LM | CLM | 特征提取 |
| BERT | 340M | Encoder | MLM+NSP | 微调 |
| GPT-2 | 1.5B | Decoder | CLM | 零样本/微调 |
| GPT-3 | 175B | Decoder | CLM | In-context Learning |
| T5 | 11B | Enc-Dec | Span Prediction | Text-to-Text 微调 |
| BART | 406M~ | Enc-Dec | Denoising | 微调 |

> [!tip] 规模效应
> 参数量增大不仅提升性能，还带来**涌现能力**（如 GPT-3 的 In-context Learning），小模型中不明显。

---

## 架构选型对比

| 模型 | 架构类型 | 注意力方向 | 适合任务 |
|------|----------|-----------|----------|
| **BERT** | Encoder-only | 双向 | 理解（分类、NER、QA） |
| **GPT** | Decoder-only | 单向（因果） | 生成（文本续写、对话） |
| **T5/BART** | Encoder-Decoder | Enc 双向 + Dec 因果 | 序列到序列（翻译、摘要） |

---

## 考试/测试常见模式
| 场景/关键词 | 答案 |
|-------------|------|
| "T5 统一框架" | 所有任务 → Text-to-Text + 任务前缀 |
| "T5 预训练任务" | Masked Span Prediction |
| "BART 预训练" | Denoising Seq2Seq（多种文档损坏） |
| "GPT vs BERT 架构" | Decoder-only 单向 vs Encoder-only 双向 |
| "GPT-3 核心能力" | In-context Learning（Zero/One/Few-shot） |
| "RLHF 三步骤" | 预训练 → SFT → RLHF |
| "ChatGPT 训练" | SFT + RLHF |
| "In-context vs 微调" | 不更新参数 vs 更新全部/部分参数 |
| "规模演进" | 94M → 340M → 1.5B → 175B |

## 相关笔记
- [[ELMo与BERT]]
- [[大语言模型与多模态]]
- [[预训练模型_练习题]]
- [[Transformer架构]]
