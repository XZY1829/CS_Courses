# 课程实践1实验报告

## 基于循环神经网络的命名实体识别（CoNLL-2003）

> 姓名：许政阳  
> 学号：231880081

---

## 1. 任务描述

本实验基于 CoNLL-2003 英文数据集，使用循环神经网络完成命名实体识别（NER）任务，识别文本中的实体边界及类别（`PER`、`LOC`、`ORG`、`MISC`），并使用 `Precision`、`Recall`、`F1` 作为评价指标。

## 2. 数据集说明

- 数据集：`CoNLL-2003`
- 数据划分：`train / validation / test`
- 标注体系：BIO
- 标签示例：`B-PER`、`I-ORG`、`O` 等

本项目通过 HuggingFace `datasets` 自动下载 `conll2003`，可复现且便于实验管理。

## 3. 方法设计

### 3.1 基线模型：BiLSTM-CRF

模型结构如下：

1. **词嵌入层**：将每个token映射到稠密向量；
2. **双向LSTM编码器**：同时捕获前向和后向上下文；
3. **线性层**：输出每个位置到标签空间的发射分数；
4. **CRF层**：建模标签转移约束，使用Viterbi进行全局最优解码。

该结构的优势是：LSTM擅长上下文建模，CRF擅长标签依赖建模，二者结合是NER经典强基线。

### 3.2 改进点：字符级CharCNN

为提升对未登录词和词形变化的鲁棒性，在词向量之外加入字符级卷积特征：

- 对每个词的字符序列进行CNN提取；
- 与词向量拼接后送入BiLSTM；
- 在人名、地名、组织名等词形规律上通常能带来增益。

### 3.3 训练策略

- 优化器：AdamW
- 正则化：dropout + word dropout
- 稳定训练：gradient clipping
- 选择最优模型：early stopping（按验证集F1）

## 4. 实验设置

### 4.1 运行环境

- 框架：PyTorch
- 主要依赖：datasets、seqeval、tqdm
- 系统：Windows 10 + Python 3.11
- 硬件：CPU（本次实验）

### 4.2 关键超参数

- 词向量维度：100
- BiLSTM隐层维度：256
- Batch size：32
- 学习率：1e-3
- Epoch上限：30
- Patience：6

## 5. 实验结果与分析

### 5.1 总体指标

> 以下为本项目当前环境实测结果（由 `outputs/final_bilstm/test_metrics.json` 与 `outputs/final_charcnn/test_metrics.json` 生成）。


| 模型                   | Precision | Recall | F1     |
| -------------------- | --------- | ------ | ------ |
| BiLSTM-CRF           | 83.26%    | 73.81% | 78.25% |
| BiLSTM-CRF + CharCNN | 84.38%    | 83.96% | 84.17% |


### 5.2 结果分析

1. **基线有效性**：BiLSTM-CRF在测试集达到 78.25% F1，说明RNN+CRF结构具备较强序列标注能力；
2. **改进有效性**：加入CharCNN后，F1从 78.25% 提升到 84.17%，提升 **5.91** 个百分点，主要来自召回率显著提升（73.81% -> 83.96%）；
3. **错误来源**：实体边界模糊、实体类型语义接近（如 `ORG` vs `MISC`）时更易出错。

### 5.3 定性案例分析（可在答辩时展示）

输入句子示例：

`EU rejects German call to boycott British lamb .`

可展示模型输出的token级标签与实体抽取结果，证明模型具备端到端预测能力。

## 6. 消融实验（加分项）

建议至少做一组：

- `BiLSTM-CRF`（基线）
- `BiLSTM-CRF + CharCNN`（改进）

并比较F1增益。项目中提供脚本 `scripts/compare_experiments.py` 自动生成对比Markdown（本次实测输出见 `outputs/final_comparison.md`）。

## 7. 结论与展望

本实验完成了基于RNN的NER系统实现，达成了课程要求的完整流程：数据加载、模型训练、评估与推理。实验表明 BiLSTM-CRF 是有效基线，引入字符级特征后性能进一步提升。后续可考虑：

- 引入预训练词向量（GloVe/FastText）
- 使用更强的上下文表示（如BERT特征融合）
- 多随机种子重复实验并报告均值与方差，提升结论可靠性

---

## 附录：复现实验命令

```bash
python -m pip install -r requirements.txt
python train.py --config configs/bilstm_crf.yaml --output_dir outputs/bilstm_crf
python train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/bilstm_crf_charcnn
python evaluate.py --model_dir outputs/bilstm_crf_charcnn --split test
python predict.py --model_dir outputs/bilstm_crf_charcnn --sentence "EU rejects German call to boycott British lamb ."
```

