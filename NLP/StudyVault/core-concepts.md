# NLP 期末 90 分复习全稿

> 适用目标：6 小时从 0 开始复习，主攻期末大题与开放题。  
> 考试范围：第 3-8 章为主体，8.4.4 不考；NLP 定义与自然语言难点可能作为基础题出现。  
> 公式格式：行内公式使用 `$...$`，重要公式单独成行使用 `$$...$$`。  
> 复习原则：不要只背名词，要按“为什么引入 → 解决什么问题 → 怎么算/怎么答 → 有什么局限”来记。

---

## 0. 先把整门课串起来

NLP 这门课的主线其实很清楚：先把文本变成机器能处理的形式，再做分类和概率建模；离散表示不够表达语义，所以引入词向量；词向量只能表示词，句子/文档需要网络结构，于是有 CNN/RNN；RNN 难并行、长距离依赖困难，于是引入 Attention 和 Transformer；标注数据不够，于是进一步引入大规模预训练模型。

| 阶段 | 核心问题 | 方法 | 局限 | 引出 |
|------|----------|------|------|------|
| 第 3 章 文本分类 | 文档属于哪一类 | BOW、TF-IDF、NB、SVM、MaxEnt | 离散稀疏，语义弱 | 语言模型、词表示 |
| 第 4 章 语言模型 | 一个句子有多大概率 | N-gram、平滑、神经语言模型 | 数据稀疏，参数多 | 分布式词向量 |
| 第 5 章 文本表示 | 词/句子如何变成向量 | Word2Vec、GloVe、句向量 | 只靠词向量难建模结构 | CNN/RNN |
| 第 6 章 CNN | 如何捕获局部 n-gram 特征 | Conv、Pooling、TextCNN | 长距离依赖弱 | RNN/Attention |
| 第 7 章 RNN | 如何处理变长序列和历史信息 | SRNN、BPTT、LSTM、GRU、Attention | 串行、梯度问题 | Transformer |
| 第 8 章 Transformer | 如何并行建模全局依赖 | Self-Attention、Encoder/Decoder | 需要大数据大算力 | 预训练模型 |
| 第 8 章预训练 | 如何利用无标注语料 | BERT、T5、BART、GPT、ChatGPT | 推理、事实、可控性仍有限 | LLM 应用与开放题 |

### 0.1 6 小时冲刺安排

| 时间 | 目标 | 具体动作 |
|------|------|----------|
| 0:00-0:40 | 建立主线 | 读第 0 节和每章开头的“过渡” |
| 0:40-2:20 | 计算题 | 重点手算 Ch3 NB/IG、Ch4 N-gram/平滑、Ch6 卷积、Ch7/8 Attention、Ch8 BLEU |
| 2:20-3:40 | 模型题 | 背 TextCNN、RNN/BPTT、LSTM/GRU、Transformer Encoder/Decoder |
| 3:40-4:40 | 开放题 | 背机器翻译、生成式模型、自动对下联系统模板 |
| 4:40-5:40 | 做模拟卷 | 完整写一遍 `mock-exam.md` 的试题部分 |
| 5:40-6:00 | 最后压缩 | 只看“考前速记清单”和自己错的计算题 |

### 0.2 学长考后笔记给出的最高优先级

学长考后笔记提到真正后悔没有掌握的是：

1. **BLEU 公式**
2. **多项式朴素贝叶斯用于文本分类**
3. **T5 模型的预训练任务与 prefix**

同时出现过的其他题点包括：CNN 用于文本分类、RNN 训练过程、模型设计题“给定上联对出下联”、NLP 定义、自然语言相对人工语言更难的原因。

---

## 1. NLP 基础概念：定义与难点

### 1.1 NLP 是什么

自然语言处理（Natural Language Processing, NLP）研究如何让计算机处理、理解和生成自然语言。它既关注语言的形式处理，如分词、词性标注、句法分析，也关注语义理解和语言生成，如文本分类、机器翻译、问答、摘要、对话系统。

如果考试问“NLP 的定义”，可以这样答：

> NLP 是人工智能与语言学交叉的方向，目标是建立能够自动分析、理解、生成和应用人类自然语言的计算模型，使计算机能够完成人类语言相关任务，如文本分类、机器翻译、信息抽取、问答和对话。

### 1.2 自然语言为什么比人工语言难

人工语言如编程语言有严格语法和唯一解释，自然语言没有。自然语言的难点主要来自歧义、上下文依赖、知识依赖和开放性。

| 难点 | 含义 | 例子 |
|------|------|------|
| 词法歧义 | 一个词有多个含义 | “苹果”可以是水果，也可以是公司 |
| 句法歧义 | 同一句子可有多种结构 | “咬死了猎人的狗” |
| 语义歧义 | 结构相同但含义依赖上下文 | “他看见她拿着望远镜” |
| 语用歧义 | 真实意图不等于字面意思 | “你可真准时”可能是讽刺 |
| 新词与未知现象 | 语言不断变化 | 网络词、新术语、缩写 |
| 世界知识依赖 | 理解需要常识 | “杯子掉到地上碎了” |
| 跨语言差异 | 不同语言词序、语法、语义单元不完全对应 | 机器翻译中的省略、倒装、成语 |

### 1.3 NLP 方法论演进

| 阶段 | 方法 | 核心思想 | 局限 |
|------|------|----------|------|
| 规则方法 | 人工规则、词典、语法 | 专家总结语言规律 | 覆盖率低，维护成本高 |
| 统计方法 | 标注语料 + 概率模型 | 用数据估计语言规律 | 特征工程重，语义表示弱 |
| 深度学习 | 神经网络 + 表示学习 | 自动学习特征和表示 | 需要大数据与算力 |
| 预训练大模型 | 大规模自监督预训练 + 微调/提示 | 统一多任务，泛化更强 | 幻觉、推理、可控性仍是问题 |

---

## 2. 第 3 章：文本分类

### 2.1 为什么先讲文本分类

分词和规则方法只能做基础语言处理，真正的 NLP 应用往往要做决策，例如新闻分类、情感分析、垃圾邮件识别。文本分类就是把文本映射到预定义类别，是后续机器学习方法进入 NLP 的典型入口。

文本分类任务可以形式化为：

$$
f(d) \rightarrow c
$$

其中 $d$ 是文档或句子，$c$ 是类别标签。单标签分类中每个文本只属于一个类别，多标签分类中一个文本可以同时属于多个类别。

### 2.2 向量空间模型 VSM 与词袋模型 BOW

计算机不能直接处理自然语言文本，所以第一步是把文本表示成向量。

**向量空间模型（Vector Space Model, VSM）**：将每篇文档表示成一个向量，向量每一维对应一个特征词。

例如词表为 `[大学, 计算机, 排球, 运动会]`，文档“大学 计算机 计算机”可以表示为：

$$
[1,2,0,0]
$$

**词袋模型（Bag of Words, BOW）**：只统计词是否出现或出现次数，忽略词序。

| 表示方式 | 含义 | 优点 | 缺点 |
|----------|------|------|------|
| Boolean | 出现为 1，不出现为 0 | 简单 | 忽略频次 |
| Count | 出现几次记几次 | 保留词频 | 高频停用词干扰大 |
| TF-IDF | 词频乘逆文档频率 | 强调区分性强的词 | 仍忽略词序和语义 |

### 2.3 TF-IDF

只看词频会让“的、是、在”等常见词权重很高，但它们对分类帮助不大。因此引入 IDF 降低常见词权重。

$$
\text{tf-idf}_i = tf_i \cdot idf_i
$$

$$
idf_i = \log\frac{N}{df_i}
$$

| 符号 | 含义 |
|------|------|
| $tf_i$ | 词 $t_i$ 在当前文档中的出现次数 |
| $N$ | 文档总数 |
| $df_i$ | 包含词 $t_i$ 的文档数 |
| $idf_i$ | 逆文档频率，越常见越小 |

直觉：某个词在当前文档中频繁出现，且在其他文档中不常出现，那么它更可能是该文档的关键特征。

### 2.4 特征选择：DF、MI、IG

文本向量维度通常很高，所以需要特征选择。特征选择的目标是保留对分类有用的词，删除噪声词、低频词或无区分能力的词。

| 方法 | 核心思想 | 适合记忆方式 |
|------|----------|--------------|
| DF 文档频率 | 删除出现文档太少或太多的词 | 太稀有或太常见都不可靠 |
| MI 互信息 | 衡量词与类别的统计相关性 | 词出现后类别不确定性下降多少 |
| IG 信息增益 | 衡量知道某特征后类别熵减少多少 | 分类系统获得了多少信息 |

#### 2.4.1 互信息 MI

互信息衡量特征 $t$ 与类别 $c$ 的相关程度：

$$
MI(t,c)=\log \frac{P(t,c)}{P(t)P(c)}
$$

如果 $t$ 和 $c$ 独立，则 $P(t,c)=P(t)P(c)$，互信息为 0。互信息越大，说明这个词对这个类别越有提示作用。

#### 2.4.2 信息增益 IG

信息增益等于原始类别熵减去知道特征后的条件熵：

$$
IG(C,t)=H(C)-H(C|t)
$$

其中：

$$
H(C)=-\sum_{j=1}^{C}P(c_j)\log P(c_j)
$$

$$
H(C|t)=-P(t)\sum_jP(c_j|t)\log P(c_j|t)-P(\bar{t})\sum_jP(c_j|\bar{t})\log P(c_j|\bar{t})
$$

所以：

$$
IG(t)=-\sum_j P(c_j)\log P(c_j)
+ P(t)\sum_jP(c_j|t)\log P(c_j|t)
+ P(\bar{t})\sum_jP(c_j|\bar{t})\log P(c_j|\bar{t})
$$

> 注意：这里课件使用自然对数时，“计算机”的例题结果为 $0.1308$。

#### 2.4.3 课堂例题：“计算机”的信息增益

课件四篇文档中，教育类 2 篇、体育类 2 篇。特征“计算机”出现于 2 篇教育文档，不出现于 2 篇体育文档。课件为了避免零概率，在条件概率中做了加 1 式处理，得到：

$$
P(\text{教育}|\text{计算机})=\frac{2+1}{2+2}=\frac{3}{4}
$$

$$
P(\text{体育}|\text{计算机})=\frac{0+1}{2+2}=\frac{1}{4}
$$

同理：

$$
P(\text{教育}|\bar{\text{计算机}})=\frac{1}{4},\quad
P(\text{体育}|\bar{\text{计算机}})=\frac{3}{4}
$$

代入：

$$
IG(\text{计算机})=-\log 0.5 + 0.75\log 0.75 + 0.25\log 0.25 = 0.1308
$$

如果考“北京”的信息增益，先数出“北京”在教育/体育中出现与未出现的次数，再按同一模板代入。不要死背结果，重点是会列 $P(t)$、$P(\bar{t})$、$P(c|t)$、$P(c|\bar{t})$。

### 2.5 多项式朴素贝叶斯

这是本章最重要的计算题。学长考后笔记也明确提到“多项式朴素贝叶斯用于文本分类”是必背点。

#### 2.5.1 为什么引入朴素贝叶斯

VSM 和 TF-IDF 解决了“文本如何表示”，但还没有解决“如何分类”。朴素贝叶斯用概率模型描述文本属于某类的可能性。它属于**生成式模型**，因为它建模的是：

$$
P(c_j) \quad \text{和} \quad P(\mathbf{x}|c_j)
$$

也就是先选一个类别，再由该类别生成一篇文档。

#### 2.5.2 朴素假设与决策规则

朴素假设：给定类别后，各个词的出现相互独立。

$$
P(\mathbf{x}|c_j)\approx \prod_{i=1}^{M}P(w_i|c_j)^{N(w_i)}
$$

决策规则：

$$
c^*=\arg\max_j P(c_j)\prod_{i=1}^{M}P(w_i|c_j)^{N(w_i)}
$$

实际计算时常取对数，避免很多小概率相乘造成下溢：

$$
c^*=\arg\max_j \left(\log P(c_j)+\sum_i N(w_i)\log P(w_i|c_j)\right)
$$

#### 2.5.3 参数估计

类别先验：

$$
P(c_j)=\frac{N(c_j)}{N_{\text{all}}}
$$

条件概率：

$$
P(w_i|c_j)=\frac{N(w_i,c_j)}{\sum_{i'}N(w_{i'},c_j)}
$$

如果某个词在某类中没有出现，概率为 0，整个乘积会变 0，所以要做拉普拉斯平滑。

拉普拉斯平滑：

$$
P(w_i|c_j)=\frac{1+N(w_i,c_j)}{M+\sum_{i'}N(w_{i'},c_j)}
$$

其中 $M$ 是特征词表大小。

#### 2.5.4 课堂练习完整模板

Feature Set = `[计算机, 排球, 运动会, 高校, 大学]`

| 类别 | 文档 |
|------|------|
| 教育 | 大学 计算机 高校 |
| 体育 | 大学 运动会 排球 |
| 教育 | 大学 计算机 |
| 体育 | 运动会 排球 |

第一步：先验概率（不平滑）：

$$
P(\text{教育})=\frac{2}{4}=0.5,\quad P(\text{体育})=\frac{2}{4}=0.5
$$

第二步：数词频。

教育类中：

| 词 | 次数 |
|----|------|
| 大学 | 2 |
| 计算机 | 2 |
| 高校 | 1 |
| 排球 | 0 |
| 运动会 | 0 |

教育类总词频为 $5$，$M=5$，所以分母为 $5+5=10$：

$$
P(\text{大学}|\text{教育})=\frac{1+2}{10}=\frac{3}{10}
$$

$$
P(\text{计算机}|\text{教育})=\frac{3}{10},\quad
P(\text{高校}|\text{教育})=\frac{2}{10}
$$

$$
P(\text{排球}|\text{教育})=\frac{1}{10},\quad
P(\text{运动会}|\text{教育})=\frac{1}{10}
$$

体育类中：

| 词 | 次数 |
|----|------|
| 大学 | 1 |
| 运动会 | 2 |
| 排球 | 2 |
| 计算机 | 0 |
| 高校 | 0 |

同理：

$$
P(\text{大学}|\text{体育})=\frac{2}{10},\quad
P(\text{运动会}|\text{体育})=\frac{3}{10},\quad
P(\text{排球}|\text{体育})=\frac{3}{10}
$$

$$
P(\text{计算机}|\text{体育})=\frac{1}{10},\quad
P(\text{高校}|\text{体育})=\frac{1}{10}
$$

如果测试文档是“大学 计算机”，则：

$$
Score(\text{教育})=0.5\times \frac{3}{10}\times \frac{3}{10}=0.045
$$

$$
Score(\text{体育})=0.5\times \frac{2}{10}\times \frac{1}{10}=0.01
$$

所以预测为**教育**。

### 2.6 分类评估指标

分类器不只看准确率，还常看 Precision、Recall、F1。

| 指标 | 公式 | 含义 |
|------|------|------|
| Precision | $P=\frac{TP}{TP+FP}$ | 预测为正的里面有多少真是正 |
| Recall | $R=\frac{TP}{TP+FN}$ | 真正为正的里面找回了多少 |
| F1 | $F_1=\frac{2PR}{P+R}$ | P 和 R 的调和平均 |

多类别时：

| 平均方式 | 做法 | 特点 |
|----------|------|------|
| Macro | 每类分别算，再平均 | 每个类别权重相同 |
| Micro | 合并所有 TP/FP/FN 后再算 | 样本多的类别影响更大 |

---

## 3. 第 4 章：语言模型

### 3.1 为什么引入语言模型

文本分类回答“这篇文本属于哪类”，语言模型回答“这句话像不像自然语言”。机器翻译、语音识别、输入法、文本生成都需要判断一个词序列的概率。

语言模型目标：

$$
P(s)=P(w_1,w_2,\ldots,w_m)
$$

根据链式法则：

$$
P(s)=\prod_{i=1}^{m}P(w_i|w_1,\ldots,w_{i-1})
$$

问题是历史太长，参数太多，所以引入 Markov 假设。

### 3.2 N-gram 模型

N-gram 假设当前词只依赖前 $n-1$ 个词。

$$
P(s)=\prod_i P(w_i|w_{i-n+1},\ldots,w_{i-1})
$$

常见模型：

| 模型      | 条件概率                    | 含义        |     |
| ------- | ----------------------- | --------- | --- |
| Unigram | $P(w_i)$                | 当前词不依赖上下文 |     |
| Bigram  | $P(w_i,w_{i-1})$        | 当前词只看前一个词 |     |
| Trigram | $P(w_i,w_{i-2}w_{i-1})$ | 当前词看前两个词  |     |

最大似然估计：

$$
P(w_i|w_{i-1})=\frac{c(w_{i-1},w_i)}{\sum_w c(w_{i-1},w)}
$$

也就是：

$$
P(\text{后词}|\text{前词})=\frac{\text{前词后面接这个后词的次数}}{\text{前词出现后接任意词的次数}}
$$

### 3.3 Bigram 课堂练习：John read a book

训练语料：

```text
<BOS> John read Moby Dick <EOS>
<BOS> Mary read a different book <EOS>
<BOS> She read a book by Cher <EOS>
```

计算：

$$
P(\text{John read a book})
=P(\text{John}|<BOS>)P(\text{read}|\text{John})P(a|\text{read})P(\text{book}|a)P(<EOS>|\text{book})
$$

逐项：

$$
P(\text{John}|<BOS>)=\frac{1}{3}
$$

$$
P(\text{read}|\text{John})=\frac{1}{1}=1
$$

$$
P(a|\text{read})=\frac{2}{3}
$$

$$
P(\text{book}|a)=\frac{1}{2}
$$

$$
P(<EOS>|\text{book})=\frac{1}{2}
$$

所以：

$$
P(\text{John read a book})=\frac{1}{3}\times1\times\frac{2}{3}\times\frac{1}{2}\times\frac{1}{2}=\frac{1}{18}\approx0.0556
$$

### 3.4 数据稀疏与加 1 平滑

N-gram 最大的问题是：没见过的组合概率为 0。例如 `Cher read a book` 中：

$$
P(\text{Cher}|<BOS>)=0
$$

所以整句概率变成 0。这显然不合理，因为未在训练集中出现不代表不可能。

加 1 平滑：

$$
P(w_i|w_{i-1})=\frac{1+c(w_{i-1},w_i)}{|V|+\sum_w c(w_{i-1},w)}
$$

同上语料，$|V|=13$：

$$
P(\text{Cher}|<BOS>)=\frac{0+1}{13+3}=\frac{1}{16}
$$

$$
P(\text{read}|\text{Cher})=\frac{0+1}{13+1}=\frac{1}{14}
$$

$$
P(a|\text{read})=\frac{2+1}{13+3}=\frac{3}{16}
$$

$$
P(\text{book}|a)=\frac{1+1}{13+2}=\frac{2}{15}
$$

$$
P(<EOS>|\text{book})=\frac{1+1}{13+2}=\frac{2}{15}
$$

因此：

$$
P(\text{Cher read a book})=\frac{1}{16}\times\frac{1}{14}\times\frac{3}{16}\times\frac{2}{15}\times\frac{2}{15}\approx1.49\times10^{-5}
$$

> [!warning] 存疑
> 课件提取文本中出现 `$P(\text{book}|a)=(1+1)/(13+2)=1/15$`，按公式应为 $2/15$。考试手算时以计数和公式为准。

### 3.5 其他平滑方法

| 方法 | 思想 |
|------|------|
| 加 1 法 | 所有计数都加 1，简单但会过度平滑 |
| 减值/折扣法 | 从出现过的事件中扣出一部分概率给未出现事件 |
| 删除插值 | 将高阶 n-gram 与低阶 n-gram 加权组合 |

### 3.6 困惑度 Perplexity

困惑度衡量语言模型对测试集的“惊讶程度”，越小越好。

$$
PP(T)=P(T)^{-\frac{1}{w_T}}
$$

等价写法：

$$
PP(T)=2^{-\frac{1}{w_T}\log_2P(T)}
$$

| 符号 | 含义 |
|------|------|
| $T$ | 测试集 |
| $w_T$ | 测试集词数 |
| $P(T)$ | 模型给测试集的概率 |

直觉：如果模型很确定测试文本，$P(T)$ 大，困惑度小；如果模型很不确定，困惑度大。

### 3.7 神经语言模型

N-gram 的局限是数据稀疏、不能表达语义相似性。例如“cat”和“dog”在 one-hot 中完全正交。神经语言模型用词向量表示词，使相似词可以共享统计信息。

FNN-LM 的基本结构：

1. 输入前 $n-1$ 个词；
2. 查表得到词向量；
3. 拼接词向量；
4. 经过隐藏层；
5. Softmax 输出下一个词概率。

RNN-LM 则用隐状态 $h_t$ 表示历史上下文：

$$
h_t=f(Uh_{t-1}+Wx_t+b)
$$

---

## 4. 第 5 章：文本表示

### 4.1 为什么引入文本表示学习

BOW/TF-IDF 能用于分类，但它把每个词当成独立维度，不能表达语义相似性。比如“大学”和“高校”语义相近，但 one-hot 中它们的内积为 0。

所以需要分布式表示：用低维稠密向量表示词，让语义相似的词在向量空间中更接近。

分布式假说：

> You shall know a word by the company it keeps.  
> 一个词的意义由它的上下文决定。

### 4.2 One-hot 与分布式表示对比

| 维度   | One-hot  | 分布式词向量             |     |     |
| ---- | -------- | ------------------ | --- | --- |
| 维度   | 词表大小 $V$ | 人工设定，如 100/300/768 |     |     |
| 稀疏性  | 极稀疏      | 稠密                 |     |     |
| 相似性  | 任意不同词正交  | 相似词向量接近            |     |     |
| 学习方式 | 人工编码     | 从语料中学习             |     |     |

### 4.3 Word2Vec：CBOW 与 Skip-Gram

Word2Vec 的核心是用浅层神经网络从上下文中学习词向量。

| 模型 | 输入 | 输出 | 直觉 |
|------|------|------|------|
| CBOW | 上下文词 | 中心词 | 根据周围词猜中间词 |
| Skip-Gram | 中心词 | 上下文词 | 根据一个词预测周围词 |

CBOW：

$$
h=\frac{1}{2C}\sum_{k=-C,k\ne0}^{C}e(w_{t+k})
$$

$$
P(w_t|context)=\frac{\exp(h\cdot e(w_t))}{\sum_{w\in V}\exp(h\cdot e(w))}
$$

Skip-Gram：

$$
P(w_{t+j}|w_t)=\frac{\exp(e(w_t)\cdot e(w_{t+j}))}{\sum_{w\in V}\exp(e(w_t)\cdot e(w))}
$$

| 对比 | CBOW | Skip-Gram |
|------|------|-----------|
| 训练速度 | 快 | 慢 |
| 低频词效果 | 一般 | 更好 |
| 数据需求 | 更适合大语料高频词 | 小语料/低频词相对更好 |
| 是否考虑词序 | 基本不考虑 | 基本不考虑 |

### 4.4 GloVe

Word2Vec 更偏局部窗口预测，GloVe 引入全局共现统计。它希望词向量内积能够拟合词共现次数的对数。

目标函数：

$$
J=\sum_{i,j=1}^{V}f(X_{ij})\left(e(w_i)^T\tilde e(w_j)+b_i+\tilde b_j-\log X_{ij}\right)^2
$$

| 符号 | 含义 |
|------|------|
| $X_{ij}$ | 词 $i$ 和词 $j$ 的共现次数 |
| $f(X_{ij})$ | 权重函数，降低极高/极低频共现的影响 |
| $e(w_i)$ | 中心词向量 |
| $\tilde e(w_j)$ | 上下文词向量 |

考试如果问 GloVe，答出“全局共现矩阵 + 加权最小二乘 + 拟合 $\log X_{ij}$”即可。

### 4.5 句子和文档表示

词向量只表示单词，句子/文档还需要组合。

| 方法 | 思想 | 局限 |
|------|------|------|
| 平均池化 | 句向量 = 词向量平均 | 丢失词序和重点 |
| 最大池化 | 每一维取最大响应 | 只保留最强特征 |
| CNN | 捕获局部 n-gram 特征 | 长距离依赖弱 |
| RNN | 按顺序递推，保留历史 | 串行、梯度问题 |
| Transformer | Self-Attention 全局交互 | 数据和算力要求高 |

---

## 5. 第 6 章：卷积神经网络 CNN

### 5.1 为什么引入 CNN

词向量解决了“词怎么表示”，但文本分类常常取决于局部短语，例如“非常好”“不满意”“not good”。CNN 用卷积核在序列上滑动，自动捕获局部 n-gram 特征。

### 5.2 卷积输出长度公式

一维卷积输出长度：

$$
L_{\text{out}}=\left\lfloor\frac{M+2P-K}{S}+1\right\rfloor
$$

二维卷积每个维度同理：

$$
H_{\text{out}}=\left\lfloor\frac{H+2P-K_H}{S}+1\right\rfloor
$$

$$
W_{\text{out}}=\left\lfloor\frac{W+2P-K_W}{S}+1\right\rfloor
$$

| 符号 | 含义 |
|------|------|
| $M$ | 输入长度 |
| $K$ | 卷积核大小 |
| $P$ | 单侧 padding |
| $S$ | stride |

三种常见卷积：

| 类型 | 参数 | 输出长度 |
|------|------|----------|
| 窄卷积 narrow | $P=0,S=1$ | $M-K+1$ |
| 等宽卷积 same | $P=(K-1)/2,S=1$，$K$ 为奇数 | $M$ |
| 宽卷积 wide | $P=K-1,S=1$ | $M+K-1$ |

### 5.3 课堂练习：二维卷积

给定 $6\times6$ 图像和 $3\times3$ 滤波器，$P=0,S=1$。输出大小：

$$
H_{\text{out}}=W_{\text{out}}=\frac{6-3}{1}+1=4
$$

所以输出是 $4\times4$。

滤波器：

$$
W=
\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1
\end{bmatrix}
$$

左上角窗口若为：

$$
\begin{bmatrix}
3&0&1\\
1&5&8\\
2&7&2
\end{bmatrix}
$$

卷积结果：

$$
3\cdot1+0\cdot0+1\cdot(-1)+1\cdot1+5\cdot0+8\cdot(-1)+2\cdot1+7\cdot0+2\cdot(-1)=-5
$$

考试手算卷积的步骤：

1. 先用公式算输出尺寸；
2. 取对应窗口；
3. 窗口元素与卷积核逐项相乘；
4. 全部相加；
5. 移动窗口继续算。

### 5.4 CNN 基本结构

CNN 通常由卷积层、激活函数、池化层、全连接层组成。

| 层 | 作用 |
|----|------|
| 卷积层 | 提取局部模式 |
| 激活函数 | 引入非线性，如 ReLU |
| 池化层 | 降维并增强平移不变性 |
| 全连接层 | 汇总特征并分类 |

卷积层可写为：

$$
Y_p=f(W_p\otimes X+b_p)
$$

其中 $W_p$ 是第 $p$ 个卷积核，$b_p$ 是偏置，$f$ 是激活函数。

### 5.5 TextCNN

TextCNN 是 CNN 在文本分类中的经典应用。

流程：

1. 输入句子，查词向量表得到词向量矩阵；
2. 使用多个不同高度的卷积核，如 2、3、4，对应不同 n-gram；
3. 每个卷积核得到特征图；
4. 对每个特征图做 Global Max Pooling；
5. 拼接池化结果；
6. 全连接 + Softmax 分类。

如果句子长度为 $n$，词向量维度为 $d$，输入矩阵大小是：

$$
n\times d
$$

文本卷积核通常大小为：

$$
k\times d
$$

其中 $k$ 是窗口大小，如 2-gram、3-gram，$d$ 覆盖整个词向量维度。

### 5.6 TextCNN 为什么有效

TextCNN 适合文本分类，因为分类往往由局部关键词或短语决定。例如情感分类中，“非常失望”“太好看了”“not worth”这类短语本身就有很强类别信号。

Global Max Pooling 的作用是从全句中找出某个卷积核最强响应的位置，也就是“这个模式是否在句中出现过”。

### 5.7 CNN 的优缺点

| 方面 | 优点 | 缺点 |
|------|------|------|
| 并行性 | 卷积可并行，训练快 | 局部窗口限制 |
| 特征提取 | 擅长 n-gram 局部模式 | 长距离依赖弱 |
| 参数 | 参数共享，较高效 | 对顺序全局结构建模不如 RNN/Transformer |

考试答 TextCNN 模板：

> TextCNN 将句子表示为词向量矩阵，用多个不同窗口大小的卷积核提取局部 n-gram 特征，经非线性激活和 Global Max Pooling 得到每类局部模式的最强响应，最后拼接后用全连接层和 Softmax 分类。它的优点是并行性好、能捕获关键短语；局限是感受野有限，难以直接建模长距离依赖。

---

## 6. 第 7 章：循环神经网络 RNN

### 6.1 为什么引入 RNN

CNN 擅长局部特征，但语言是序列，很多任务需要历史信息。例如“我今天很开心，因为……”后面的词依赖前文。RNN 通过隐状态在时间上传递信息，适合建模变长序列。

### 6.2 SRNN 基本公式

标准 RNN：

$$
h_t=f(Uh_{t-1}+Wx_t+b)
$$

输出可写为：

$$
y_t=g(Vh_t+c)
$$

| 符号 | 含义 |
|------|------|
| $x_t$ | 当前输入 |
| $h_t$ | 当前隐状态 |
| $h_{t-1}$ | 上一时刻隐状态 |
| $U$ | 隐状态到隐状态的权重 |
| $W$ | 输入到隐状态的权重 |
| $V$ | 隐状态到输出的权重 |

RNN 的关键是参数共享：每个时间步使用同一组 $U,W,b$。

### 6.3 BPTT：随时间反向传播

RNN 的训练方式是把循环结构沿时间展开，变成一个深层前馈网络，然后用反向传播求梯度。这称为 BPTT（Back Propagation Through Time）。

考场答题模板：

1. 前向传播：按时间步计算 $h_1,h_2,\ldots,h_T$ 和输出；
2. 计算损失：把每个时间步或最终输出与标签比较；
3. 时间展开：把共享参数的 RNN 展开成多层网络；
4. 反向传播：从后往前沿时间链式求导；
5. 参数共享：不同时间步的梯度累加到同一组参数上。

RNN 梯度中会出现连续相乘：

$$
\frac{\partial h_t}{\partial h_k}=\prod_{i=k+1}^{t}\frac{\partial h_i}{\partial h_{i-1}}
$$

如果连乘项模长小于 1，梯度消失；大于 1，梯度爆炸。

### 6.4 梯度消失与梯度爆炸

| 问题 | 原因 | 后果 | 常见解决 |
|------|------|------|----------|
| 梯度消失 | 多个小于 1 的导数连乘 | 远距离历史学不到 | LSTM/GRU、残差结构 |
| 梯度爆炸 | 多个大于 1 的导数连乘 | 参数更新不稳定 | 梯度裁剪 |

考试如果问“为什么 RNN 难学长程依赖”，核心答：

> RNN 反向传播时梯度需要沿时间步不断连乘，长序列中远处时间步的梯度会指数衰减或指数增长，导致梯度消失或爆炸。因此普通 RNN 很难稳定学习长距离依赖。

### 6.5 GRU

GRU 用门控机制控制旧信息保留和新信息写入。

更新门：

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z)
$$

重置门：

$$
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r)
$$

候选状态：

$$
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})+b_h)
$$

最终状态：

$$
h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t
$$

记忆方式：

| 门 | 作用 |
|----|------|
| 更新门 $z_t$ | 控制保留多少旧状态 |
| 重置门 $r_t$ | 控制计算候选状态时看多少旧状态 |

如果 $z_t$ 接近 1，更多保留旧状态；如果 $z_t$ 接近 0，更多使用新候选状态。

### 6.6 LSTM

LSTM 引入记忆单元 $c_t$，用输入门、遗忘门、输出门控制信息流。

遗忘门：

$$
f_t=\sigma(W_fx_t+U_fh_{t-1}+b_f)
$$

输入门：

$$
i_t=\sigma(W_ix_t+U_ih_{t-1}+b_i)
$$

输出门：

$$
o_t=\sigma(W_ox_t+U_oh_{t-1}+b_o)
$$

候选记忆：

$$
\tilde c_t=\tanh(W_cx_t+U_ch_{t-1}+b_c)
$$

记忆更新：

$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
$$

隐状态：

$$
h_t=o_t\odot\tanh(c_t)
$$

| 门 | 作用 |
|----|------|
| 遗忘门 | 决定旧记忆保留多少 |
| 输入门 | 决定新信息写入多少 |
| 输出门 | 决定记忆输出多少 |

### 6.7 GRU 与 LSTM 对比

| 对比 | GRU | LSTM |
|------|-----|------|
| 门数量 | 2 个 | 3 个 |
| 状态 | 只有 $h_t$ | 有 $c_t$ 和 $h_t$ |
| 参数量 | 少 | 多 |
| 表达能力 | 较强，训练快 | 更复杂，长期记忆能力强 |

### 6.8 Attention

RNN Encoder-Decoder 有一个瓶颈：如果把整个源句压缩成一个固定长度向量，长句信息容易丢失。Attention 的思想是解码每一步时动态关注源序列不同位置。

通用形式：

$$
\alpha_n=\text{softmax}(s(x_n,q))
$$

$$
\text{att}(X,q)=\sum_n\alpha_nx_n
$$

常见打分函数：

| 类型 | 公式 |
|------|------|
| 点积 | $s(x,q)=x^Tq$ |
| 缩放点积 | $s(x,q)=\frac{x^Tq}{\sqrt{d_k}}$ |
| 双线性 | $s(x,q)=x^TWq$ |
| 加性 | $s(x,q)=v^T\tanh(W_1x+W_2q)$ |

课堂练习中如果给定 $q$ 和 $x_1,x_2,x_3$：

1. 算打分 $s_i=x_i^Tq$；
2. 做 softmax：

$$
\alpha_i=\frac{e^{s_i}}{\sum_j e^{s_j}}
$$

3. 加权求和：

$$
\text{att}=\sum_i\alpha_ix_i
$$

### 6.9 NER 评估

命名实体识别常按实体级别严格匹配，边界和类型都对才算 TP。

$$
P=\frac{TP}{TP+FP}
$$

$$
R=\frac{TP}{TP+FN}
$$

$$
F_1=\frac{2PR}{P+R}
$$

| 符号 | 含义 |
|------|------|
| TP | 系统输出实体与标准答案完全一致 |
| FP | 系统输出了但标准答案没有 |
| FN | 标准答案有但系统漏掉 |

---

## 7. 第 8 章：Transformer

### 7.1 为什么引入 Transformer

RNN 能建模序列，但有两个问题：

1. **难并行**：必须按时间步递推；
2. **长距离依赖仍困难**：远距离信息要经过很多步传递。

Self-Attention 让序列中任意两个位置直接交互，不再依赖递推路径，因此可以并行计算并捕获全局依赖。

### 7.2 Self-Attention

输入序列表示为矩阵 $X$，通过线性变换得到：

$$
Q=XW^Q,\quad K=XW^K,\quad V=XW^V
$$

Scaled Dot-Product Attention：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

如果课件使用列向量排布，也可能写成 $K^TQ$ 的形式，本质一样：用 Query 和 Key 算相关性，再用相关性加权 Value。

| 符号 | 含义 |
|------|------|
| Query | 当前词想查询什么信息 |
| Key | 每个词提供的索引特征 |
| Value | 每个词真正被聚合的信息 |
| $\sqrt{d_k}$ | 缩放因子，防止点积过大导致 softmax 过尖 |

### 7.3 Multi-Head Attention

单头注意力只在一个表示子空间中计算关系，多头注意力让模型在多个子空间中并行关注不同关系。

$$
head_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
MHA(Q,K,V)=\text{Concat}(head_1,\ldots,head_h)W^O
$$

记忆方式：多头 = 多组 Q/K/V 投影 + 并行注意力 + 拼接 + 输出投影。

### 7.4 Transformer Encoder

Encoder Block 包含：

1. Multi-Head Self-Attention；
2. Add & Norm；
3. Feed Forward Network；
4. Add & Norm。

Add & Norm = 残差连接 + LayerNorm。残差帮助梯度传播，LayerNorm 稳定训练。

FFN 通常是两层全连接：

$$
FFN(x)=\max(0,xW_1+b_1)W_2+b_2
$$

位置编码用于补充顺序信息，因为 Self-Attention 本身不包含位置顺序。输入一般为：

$$
\text{Input Embedding}+\text{Positional Encoding}
$$

### 7.5 Transformer Decoder

Decoder Block 包含：

1. Masked Multi-Head Self-Attention；
2. Add & Norm；
3. Cross-Attention；
4. Add & Norm；
5. FFN；
6. Add & Norm。

Masked Self-Attention 的作用：生成第 $t$ 个词时不能看到未来词，只能看已经生成的前缀。训练时用 mask 防止偷看答案；推理时本来就是一步一步生成。

Cross-Attention 中：

| 矩阵 | 来源 | 作用 |
|------|------|------|
| $Q$ | Decoder 当前状态 | 查询源句相关信息 |
| $K,V$ | Encoder 输出 | 提供源句表示 |

### 7.6 Transformer 用于机器翻译

机器翻译通常使用 Encoder-Decoder：

1. Encoder 读入源语言句子；
2. Decoder 自回归生成目标语言；
3. Decoder 的 Cross-Attention 动态对齐源句信息；
4. 输出层对目标词表做 softmax。

训练时常用 Teacher Forcing：给 Decoder 输入真实前缀，预测下一个词。推理时用已经生成的词作为下一步输入。

---

## 8. BLEU：机器翻译高频考点

### 8.1 为什么引入 BLEU

机器翻译需要自动评价候选译文和参考译文的接近程度。BLEU 使用 n-gram 精确率衡量候选译文中有多少片段出现在参考译文中，并用 BP 惩罚过短译文。

公式：

$$
BLEU=BP\cdot \exp\left(\sum_{n=1}^{N}w_n\log p_n\right)
$$

通常 $N=4$，$w_n=\frac{1}{N}$。

### 8.2 $p_n$：n-gram 精确率

$$
p_n=\frac{\text{候选译文中匹配参考译文的 n-gram 数}}{\text{候选译文 n-gram 总数}}
$$

严格 BLEU 中要使用 clipped count，避免候选译文重复某个词刷高分。

### 8.3 BP：短句惩罚

$$
BP=
\begin{cases}
1,& c>r\\
e^{1-r/c},& c\le r
\end{cases}
$$

| 符号 | 含义 |
|------|------|
| $c$ | 候选译文长度 |
| $r$ | 参考译文长度，或与候选长度最接近的参考长度 |

如果候选译文太短，$c<r$，则 $BP<1$。

### 8.4 BLEU 计算模板

考试步骤：

1. 写出候选译文长度 $c$、参考译文长度 $r$；
2. 计算 $BP$；
3. 列出 1-gram、2-gram、3-gram、4-gram；
4. 分别计算 $p_1,p_2,p_3,p_4$；
5. 代入几何平均；
6. 乘以 BP。

如果某阶 $p_n=0$，且题目没有要求平滑，则：

$$
BLEU=0
$$

因为几何平均中有一项为 0。

### 8.5 BLEU 答题注意

| 易错点 | 正确做法 |
|--------|----------|
| 把 BLEU 当召回率 | BLEU 主体是 n-gram 精确率 |
| 忘记 BP | 候选译文过短时必须惩罚 |
| $p_4=0$ 还继续取 log | 无平滑时 BLEU 为 0 |
| n-gram 数数错 | 候选长度为 $m$ 时，n-gram 数为 $m-n+1$ |
| 没写 clipped count | 至少说明匹配数不能超过参考中出现次数 |

---

## 9. 第 8 章：预训练模型

### 9.1 为什么引入预训练模型

Transformer 架构强，但标注数据有限。预训练模型先在大规模无标注语料上做自监督学习，再迁移到下游任务。这样可以利用海量文本中的语言知识。

通用范式：

$$
\text{大规模无标注预训练} \rightarrow \text{下游任务微调/提示}
$$

### 9.2 ELMo

ELMo 使用双向 LSTM 训练语言模型，得到上下文相关词表示。同一个词在不同句子中可以有不同向量。

特点：

| 项 | 内容 |
|----|------|
| 架构 | 双向 LSTM |
| 预训练任务 | 前向 LM + 后向 LM |
| 表示 | 上下文相关 |
| 使用方式 | 将 ELMo 表示加入下游模型 |

### 9.3 BERT

BERT 使用 Transformer Encoder，核心是双向上下文表示。

预训练任务：

1. MLM（Masked Language Modeling）
2. NSP（Next Sentence Prediction）

MLM：随机 mask 15% token，其中：

| 比例 | 操作 |
|------|------|
| 80% | 替换为 `[MASK]` |
| 10% | 替换为随机词 |
| 10% | 保持不变 |

这么做是为了缓解预训练和微调不一致，因为微调时输入中通常没有 `[MASK]`。

NSP：判断句子 B 是否是句子 A 的下一句，用于学习句间关系。

BERT 适合理解任务，如文本分类、NER、阅读理解。常见做法是使用 `[CLS]` 表示做分类。

### 9.4 T5：最高优先级

T5 是 Text-to-Text Transfer Transformer。它把所有 NLP 任务统一成“输入文本 → 输出文本”。

| 任务 | T5 输入 | T5 输出 |
|------|---------|---------|
| 翻译 | `translate English to German: That is good.` | `Das ist gut.` |
| 情感分类 | `sst2 sentence: This movie is great.` | `positive` |
| 摘要 | `summarize: ...` | 摘要文本 |
| 问答 | `question: ... context: ...` | 答案 |

#### 9.4.1 T5 的 prefix

prefix 是任务前缀，用来告诉模型当前要做什么任务。T5 不需要为不同任务设计不同模型结构，而是通过输入文本中的任务提示来统一建模。

考场答法：

> T5 将所有任务统一为 text-to-text 框架，通过在输入前添加任务 prefix 指示任务类型，例如翻译任务输入 `translate English to German: ...`，分类任务输入 `sst2 sentence: ...`。模型架构不变，输出也统一为文本。

#### 9.4.2 T5 的预训练任务：Masked Span Prediction

T5 不是只 mask 单个 token，而是 mask 连续 span，并用 sentinel token 表示被遮盖片段。

示例：

原句：

```text
Thank you for inviting me to your party last week.
```

遮盖后输入：

```text
Thank you <X> me to your party <Y> week.
```

目标输出：

```text
<X> for inviting <Y> last <Z>
```

核心：

| 点 | T5 MSP |
|----|--------|
| 遮盖对象 | 连续片段 span |
| 标记方式 | sentinel token |
| 输出方式 | Decoder 生成被遮盖片段 |
| 架构 | Encoder-Decoder |

### 9.5 BART

BART 是 Denoising Seq2Seq 预训练模型，架构是双向 Encoder + 自回归 Decoder。

预训练思路：先破坏输入文档，再让模型重建原文。

常见破坏方式：

| 方式 | 含义 |
|------|------|
| Token Masking | 遮盖 token |
| Token Deletion | 删除 token |
| Text Infilling | 用 mask 替换一段文本 |
| Sentence Permutation | 打乱句子顺序 |
| Document Rotation | 旋转文档 |

BART 适合生成、翻译、摘要，也可用于分类。

### 9.6 GPT 系列

GPT 使用 Transformer Decoder，核心任务是因果语言模型（Causal Language Modeling, CLM）：

$$
P(w_1,\ldots,w_T)=\prod_{t=1}^{T}P(w_t|w_{<t})
$$

也就是只能看左侧上下文，预测下一个词。

| 模型 | 特点 |
|------|------|
| GPT-1 | 预训练 + 微调 |
| GPT-2 | 更大规模，展示零样本能力 |
| GPT-3 | 175B，强调 In-context Learning |
| ChatGPT | SFT + RLHF，使模型更符合人类偏好 |
| GPT-4 | 多模态能力更强，了解即可 |

In-context Learning：不更新模型参数，只在 prompt 中给任务描述和示例，让模型根据上下文完成任务。

### 9.7 ChatGPT 训练流程

课件提到 ChatGPT 训练过程与 InstructGPT 类似，使用 RLHF。

可按三阶段答：

1. **预训练**：在大规模文本上训练语言模型；
2. **SFT**：用人工标注的指令-回答数据监督微调；
3. **RLHF**：训练奖励模型，再用强化学习优化模型输出，使其更符合人类偏好。

### 9.8 BERT vs GPT vs T5

| 模型 | 架构 | 上下文 | 任务形式 | 适合 |
|------|------|--------|----------|------|
| BERT | Encoder-only | 双向 | 表示学习 + 分类/抽取 | 理解任务 |
| GPT | Decoder-only | 单向，自回归 | 续写/生成 | 生成任务 |
| T5 | Encoder-Decoder | Encoder 双向，Decoder 自回归 | Text-to-Text | 多任务统一、生成 |

### 9.9 8.4.4 不考

8.4.4 基于 Transformer 的文本-视觉预训练模型不考。只需知道它包括图文配对数据、MLM/MRM/ITM、CLIP 等概念即可，不要在 6 小时冲刺中投入时间。

---

## 10. 开放题模板：机器翻译

如果开放题问“设计机器翻译系统”或“说明 Transformer 如何用于机器翻译”，按以下结构写。

### 10.1 任务形式化

输入源语言句子：

$$
X=(x_1,x_2,\ldots,x_m)
$$

输出目标语言句子：

$$
Y=(y_1,y_2,\ldots,y_n)
$$

目标是最大化：

$$
P(Y|X)=\prod_{t=1}^{n}P(y_t|y_{<t},X)
$$

### 10.2 模型架构

使用 Transformer Encoder-Decoder：

1. Encoder 对源句进行 Self-Attention 编码；
2. Decoder 用 Masked Self-Attention 建模已生成目标前缀；
3. Decoder 用 Cross-Attention 查询 Encoder 输出；
4. Softmax 预测下一个目标词。

### 10.3 训练

训练数据：平行语料 $(X,Y)$。

损失函数：交叉熵。

$$
\mathcal{L}=-\sum_t \log P(y_t^*|y_{<t}^*,X)
$$

训练技巧：Teacher Forcing，即训练时给 Decoder 真实前缀。

### 10.4 推理

推理时自回归生成：

1. 输入 `<BOS>`；
2. 预测第一个词；
3. 将预测词拼回输入；
4. 继续生成直到 `<EOS>`；
5. 可使用 greedy search 或 beam search。

### 10.5 评价

使用 BLEU：

$$
BLEU=BP\cdot \exp\left(\sum_{n=1}^{N}w_n\log p_n\right)
$$

补充评价可写：人工评价、BERTScore、COMET、ROUGE（摘要更常用）。

---

## 11. 开放题模板：生成式模型

如果问“什么是生成式模型”“比较 BERT/GPT/T5”“生成式模型如何训练”，可以按这个模板答。

### 11.1 生成式模型是什么

生成式模型学习数据分布或条件分布，能够生成新的文本。语言生成中常见目标是：

$$
P(Y|X)=\prod_tP(y_t|y_{<t},X)
$$

如果没有输入条件，就是普通语言模型：

$$
P(Y)=\prod_tP(y_t|y_{<t})
$$

### 11.2 典型模型

| 模型    | 是否生成式 | 原因                     |     |
| ----- | ----- | ---------------------- | --- |
| 朴素贝叶斯 | 是     | 建模 $P(xc)$ 与 $P(c)$    |     |
| GPT   | 是     | 自回归生成下一个词              |     |
| T5    | 是     | Encoder-Decoder 生成输出文本 |     |
| BART  | 是     | Denoising Seq2Seq 重建文本 |     |
| BERT  | 不典型   | 主要做双向表示学习，不自回归生成       |     |

### 11.3 GPT 与 BERT 对比

| 维度 | GPT | BERT |
|------|-----|------|
| 架构 | Decoder-only | Encoder-only |
| 注意力 | Masked Self-Attention，只看左边 | 双向 Self-Attention |
| 预训练 | Causal LM | MLM + NSP |
| 适合 | 生成、对话、续写 | 分类、NER、抽取式 QA |

### 11.4 T5 与 GPT 对比

| 维度 | T5 | GPT |
|------|----|-----|
| 架构 | Encoder-Decoder | Decoder-only |
| 输入输出 | Text-to-Text | Prompt 后续写 |
| 任务控制 | prefix | prompt/instruction |
| 适合 | 翻译、摘要、多任务生成 | 对话、开放生成、续写 |

---

## 12. 开放题模板：自动对下联系统

这是学长笔记提到的模型设计题。按“任务形式化 → 模型 → 数据与损失 → 推理约束 → 评价”写，一般能拿结构分。

### 12.1 任务形式化

输入上联：

$$
X=(x_1,x_2,\ldots,x_n)
$$

输出下联：

$$
Y=(y_1,y_2,\ldots,y_n)
$$

要求输出与输入长度一致，并满足语义相关、词性对仗、平仄协调、风格一致。

### 12.2 模型选择

方案一：Seq2Seq + Attention。

优点：结构清晰，可解释对齐；缺点：RNN 串行，长距离建模有限。

方案二：Transformer Encoder-Decoder。

优点：并行、全局依赖强，适合生成；缺点：需要较多数据。

方案三：T5/GPT 微调。

优点：利用预训练语言知识；缺点：需要控制格式和格律，可能生成不符合约束的内容。

推荐答案：Transformer Encoder-Decoder 或 T5 Text-to-Text。

### 12.3 训练数据与损失

训练数据：上联-下联平行语料。

输入格式：

```text
对下联：春风得意马蹄疾
```

输出：

```text
一日看尽长安花
```

损失函数：

$$
\mathcal{L}=-\sum_t\log P(y_t^*|y_{<t}^*,X)
$$

### 12.4 推理约束

| 约束 | 做法 |
|------|------|
| 字数一致 | 强制输出长度等于输入长度 |
| 词性对仗 | 用词性标注或模板约束名词对名词、动词对动词 |
| 平仄 | Beam Search 中惩罚不符合平仄的候选 |
| 语义相关 | Cross-Attention 保持与上联对应 |
| 避免重复 | 对重复 token 加惩罚 |

### 12.5 评价

可以从自动指标与人工评价两方面写：

1. 自动指标：BLEU、困惑度、长度匹配率、词性对仗准确率；
2. 人工评价：语义相关性、格律、流畅性、文学性。

---

## 13. 课堂练习计算题总表

| 章节  | 题型             | 必会内容                      |     |
| --- | -------------- | ------------------------- | --- |
| Ch3 | 信息增益           | 会算 $H(C)$、$H(Ct)$、$IG$    |     |
| Ch3 | 多项式 NB         | 会算先验、条件概率、拉普拉斯平滑、分类决策     |     |
| Ch4 | Bigram         | 会数共现次数并连乘                 |     |
| Ch4 | 加 1 平滑         | 会处理零概率                    |     |
| Ch6 | 卷积             | 会算输出尺寸与窗口卷积值              |     |
| Ch7 | Attention      | 会算点积、softmax、加权和          |     |
| Ch7 | NER 指标         | 会数 TP/FP/FN、P/R/F1        |     |
| Ch8 | Self-Attention | 会算 Q/K/V、注意力矩阵            |     |
| Ch8 | BLEU           | 会数 n-gram、算 BP、处理 $p_n=0$ |     |

---

## 14. 考前速记清单

### 14.1 必背公式

文本分类：

$$
\text{tf-idf}_i=tf_i\log\frac{N}{df_i}
$$

信息增益：

$$
IG(C,t)=H(C)-H(C|t)
$$

朴素贝叶斯：

$$
c^*=\arg\max_jP(c_j)\prod_iP(w_i|c_j)^{N(w_i)}
$$

拉普拉斯平滑：

$$
P(w_i|c_j)=\frac{1+N(w_i,c_j)}{M+\sum_{i'}N(w_{i'},c_j)}
$$

Bigram：

$$
P(w_i|w_{i-1})=\frac{c(w_{i-1},w_i)}{\sum_wc(w_{i-1},w)}
$$

加 1 平滑：

$$
P(w_i|w_{i-1})=\frac{1+c(w_{i-1},w_i)}{|V|+\sum_wc(w_{i-1},w)}
$$

卷积输出：

$$
L_{\text{out}}=\left\lfloor\frac{M+2P-K}{S}+1\right\rfloor
$$

RNN：

$$
h_t=f(Uh_{t-1}+Wx_t+b)
$$

GRU：

$$
h_t=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t
$$

LSTM：

$$
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t
$$

Attention：

$$
\text{att}(X,q)=\sum_n\alpha_nx_n
$$

Self-Attention：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

BLEU：

$$
BLEU=BP\cdot\exp\left(\sum_{n=1}^{N}w_n\log p_n\right)
$$

BP：

$$
BP=
\begin{cases}
1,& c>r\\
e^{1-r/c},& c\le r
\end{cases}
$$

### 14.2 必背一句话

| 概念 | 一句话 |
|------|--------|
| TF-IDF | 当前文档高频、全局低频的词更重要 |
| 信息增益 | 知道某特征后类别不确定性减少多少 |
| NB | 生成式分类模型，假设给定类别后词独立 |
| N-gram | 当前词只依赖前 $n-1$ 个词 |
| 平滑 | 给未见事件分配非零概率 |
| Word2Vec | 用上下文预测学习词向量 |
| TextCNN | 用卷积核捕获局部 n-gram 分类特征 |
| RNN | 用隐状态沿时间传递历史信息 |
| LSTM/GRU | 用门控缓解长程依赖和梯度问题 |
| Attention | 根据相关性对输入位置加权求和 |
| Transformer | 用 Self-Attention 并行建模全局依赖 |
| BLEU | n-gram 精确率几何平均乘短句惩罚 |
| BERT | Encoder-only，MLM + NSP，适合理解 |
| GPT | Decoder-only，CLM，适合生成 |
| T5 | Text-to-Text，用 prefix 统一任务 |
| BART | Denoising Seq2Seq，破坏再重建 |

### 14.3 90 分优先级

| 优先级 | 内容 | 原因 |
|--------|------|------|
| 1 | 多项式朴素贝叶斯 | 学长明确踩坑，计算题高危 |
| 1 | BLEU | 学长明确踩坑，机器翻译开放题核心 |
| 1 | T5 prefix + MSP | 学长明确踩坑，生成式模型高危 |
| 2 | TextCNN | 出过题，适合模型分析 |
| 2 | RNN/BPTT | 出过题，容易写成大题 |
| 2 | Transformer Encoder/Decoder | 机器翻译和预训练基础 |
| 3 | Word2Vec/GloVe | 概念题 |
| 3 | NER 指标 | 小计算题 |
| 3 | 8.4.4 | 不考，最后不看 |

---

## 15. 最后 20 分钟默写模板

### 15.1 机器翻译模板

> 机器翻译可形式化为给定源句 $X$ 生成目标句 $Y$，目标是最大化 $P(Y|X)=\prod_tP(y_t|y_{<t},X)$。可采用 Transformer Encoder-Decoder 架构：Encoder 对源句进行 Self-Attention 编码，Decoder 通过 Masked Self-Attention 建模目标前缀，并通过 Cross-Attention 查询 Encoder 输出。训练时使用平行语料和交叉熵损失，推理时自回归生成，可用 beam search。评价可使用 BLEU，公式为 $BLEU=BP\cdot\exp(\sum_nw_n\log p_n)$。

### 15.2 生成式模型模板

> 生成式模型学习文本分布或条件分布，能够根据上下文生成新文本。GPT 使用 Decoder-only Transformer 和因果语言模型，适合自回归生成；T5 使用 Encoder-Decoder，并通过 text-to-text 和任务 prefix 统一翻译、分类、摘要等任务；BART 使用去噪 Seq2Seq，通过破坏输入再重建进行预训练。相比 BERT 这类 Encoder-only 理解模型，GPT/T5/BART 更适合生成任务。

### 15.3 RNN 训练模板

> RNN 的隐状态更新为 $h_t=f(Uh_{t-1}+Wx_t+b)$。训练时将循环结构沿时间展开，先前向计算所有时间步的隐状态和损失，再从后往前使用 BPTT 求梯度。由于梯度沿时间反向传播时包含多个雅可比矩阵连乘，长序列中容易出现梯度消失或爆炸。解决方法包括 LSTM/GRU 门控机制、梯度裁剪等。

### 15.4 TextCNN 模板

> TextCNN 将句子表示为词向量矩阵，用多个不同窗口大小的卷积核提取局部 n-gram 特征，经过 ReLU 和 Global Max Pooling 得到每个特征的最强响应，最后拼接后送入全连接层和 Softmax 分类。它并行性好、擅长捕获关键短语，但长距离依赖建模弱。

### 15.5 T5 模板

> T5 是 Text-to-Text Transfer Transformer，它把所有 NLP 任务统一为文本到文本的形式，并通过 prefix 指明任务类型，例如 `translate English to German:` 或 `sst2 sentence:`。T5 的预训练任务是 Masked Span Prediction，即遮盖连续文本片段，用 sentinel token 标记，Decoder 生成被遮盖的 span。它适合翻译、摘要、分类、问答等多种任务。

