# LibFewShot SIFT 小样本复现报告

> 来源：从同目录 docx 原件抽取正文文本；用于学习检索，若版式或图片有疑问以 docx 原件为准。

## 报告信息

- 题目：将 SIFT 方法集成入 LibFewShot 框架
- 作者：许政阳、许浩博

## 一、实验目的

本实验旨在探索如何将用于小样本学习的基于语义的隐式特征转换算法 SIFT（Semantic-Based Implicit Feature Transform for Few-Shot Classification）融入到用于小样本学习的框架 LibFewShot 中。同时评估 SIFT 在 few-shot 学习任务中的特征表现能力及与神经网络特征的融合潜力。

## 二、背景介绍

### 1. SIFT 简介

SIFT 是基于语义的隐式特征转换算法，通过对原始数据集使用预训练后的特征网络进行特征提取，之后通过编码器-转换网络-解码器管道，以对管道进行优化，最终使用分类器分类，在小样本分类任务中取得不错的性能表现。

### 2. LibFewShot 简介

LibFewShot 是一个轻量化的少样本学习框架，支持多种元学习算法（如 Prototypical Networks、Matching Networks、MAML 等），方便进行 few-shot 任务实验与评估。

## 三、实验环境

- 操作系统：Windows11
- 编程语言：Python3
- 框架依赖：PyTorch、torchaudio 2.2.0+cu118、libfewshot
- GPU：NVIDIA RTX 4060 Laptop

## 四、SIFT 算法介绍（基于伪代码）

### 数据要求

- 样本：经过特征提取。
- 标签：经过 GloVe 语义嵌入。

### 步骤 1：选择基础类样本

- 根据 GloVe 语义嵌入计算新类与基础类之间的相似度（余弦相似度）。
- 使用最短路径规划（0-1 整数线性规划）优化选择，使每个新类唯一对应一个基础类，避免重复选择。

### 步骤 2：编码器（Encoder）

一个全连接层（FClayer），将基础类的视觉特征（图像提取的 CNN 特征）映射到语义空间（词向量）。使用重构损失（Reconstruction loss）+ 投影损失（Projection loss）确保编码器输出能精确还原原始视觉和语义特征。其中，该重构损失和解码器的重构损失使用岭回归损失（ridge regression loss）进行优化，同时使用权重绑定（Weight Tying）机制提高鲁棒性。

### 步骤 3：语义特征变换（Transformation Network）

一个全连接层（nn.Linear），将基础类的语义嵌入映射到新类的语义嵌入，得到转换后的嵌入。优化目标是最小化新类语义嵌入与转换结果的差异（同样使用岭回归 ridge regression）。

### 步骤 4：解码器（Decoder）

一个全连接层（FClayer），将转换后的语义嵌入重新映射回视觉特征空间，生成新类的增强特征。同样通过重构损失（Reconstruction loss）和投影损失（Projection loss）优化解码器。

### 步骤 5：特征增强与紧凑性约束

使用生成的特征 augment 支持集。引入紧凑性约束（Compactness Constraint），确保生成的特征聚集在其所属类别的中心周围。

### 步骤 6：原型修正

在 Transductive setting 中：将查询集聚类，使用最短路径优化和 kmeans 聚类算法将聚类中心分配给类别。构造新的“修正原型”，用于分类时增强判别性。

### 步骤 7：训练与分类

训练损失总览：使用交叉熵损失优化。基于增强后的支持集训练分类器对查询样本进行分类预测。

论文中 SIFT 在多个基准数据集（miniImageNet、tieredImageNet、CIFAR-FS、CUB）中多个任务（1/5-shot）均超越了多种先进方法，尤其在 1-shot 场景下优势显著，显示了其在小样本增强上的高效性。

## 五、实验过程

通过对 SIFT 源代码的理解，我们着手开始尝试将 SIFT 移植入 LibFewShot 作为 model 库的新模型，但是在这个过程中遇到了一些问题。

SIFT 源代码中，作者使用了懒惰学习算法（lazy learner），意味着只有遇到新的数据点才会进行预测。在每个任务内（support+query）做 50 步的 task-specific 优化，但 LibFewShot 中按 epoch 调用 set_forward_loss，只做单步前向+反向。所以如果只是对 SIFT 进行简单包装作为 MetaModel 会丢失论文原有的自适应优化的完整逻辑。如果不保留 50 步优化，SIFT 的性能会大幅下降。

此外，SIFT 作者似乎并没有体现出他们的数据集训练过程，SIFT 依赖 backbone（ResNet12）提取高质量视觉特征。意味着在 base classes 上训练 backbone，获得通用特征的过程和方法我们无从得知，只能自己通过 mini 数据集进行训练，可能对模型的性能造成损失。

如果直接用随机初始化，任务内优化无法收敛，生成的增强特征无意义，同时 encoder 学不到良好的视觉到语义映射，semantic consistency 损失形同虚设。此外 Prototype Rectification 假设支持集原型接近真实类别中心。如果 backbone 特征质量差，rectification 无法提高性能，甚至会恶化。

最终，我们决定在自己训练 backbone 的基础上通过新建 tools 文件的方法将 SIFT 的独有算法嵌入 LibFewShot。

## 六、核心亮点

### 1. 预训练相关

预训练特征提取和语义嵌入是我们本次实验的核心难点与亮点，利用预训练的 ResNet12 网络，批量提取 miniImageNet 中图像的中间特征，保存为 `.npz` 文件，供下游 few-shot 任务使用。同时利用 GloVe 方法，使用平均策略获得了每个标签的语义嵌入。

为了方便地进行数据特征和语义提取处理，我们构建了一套完整的数据链，保存在 Tools 目录里，其中核心文件和对应功能如下：

- `generate_synet_words.py`：从网上下载写明 miniImageNet 数据集标签编号和名称映射的 json 文件，读取并生成 synset 到 label 的映射，用于转换标签。
- `label2name.py`：把原始 miniImageNet 数据集 `.csv` 文件里的 labels 列编号通过字典转换为真实英文单词，并保存为新的 csv 文件（`train_mapped.csv`、`test_mapped.csv`）用于语义嵌入。
- `glove_embedding.py`：通过分割单词的方法，从类别名生成平均词向量，保存在 npz 文件（`few-shot-wordemb-train.npz`、`few-shot-wordemb-test.npz`）中。
- `convert_to_csv.py`：把原始数据集的 image 转换为 np 数组和 csv 文件组成一个文件（`few-shot-train/test.npz`），供源代码的 Genfeat 模块提取特征。

经过这些处理之后，再通过源代码的 Genfeat 模块使用预训练好的 ResNet12 网络提取特征，得到特征文件（`feat-train/test.npz`），至此准备工作结束。可以看出这里对代码阅读和编写要求很高，费时费力，在作者对预训练流程解释不清、没有代码文档的情况下跑通源代码并不是一件容易的事。

简化步骤如下：

1. 从 CSV 文件中读取图像路径及对应类别标签，将图像统一缩放到 84×84，并将图像转为 numpy 数组保存为 features + labels 的 `.npz` 文件。
2. 使用预训练的 ResNet12 模型对图像进行特征提取：加载训练好的模型参数，提取中间层输出（640 维向量），将所有图像特征与标签保存为 `.npz`，供下游 few-shot 模型使用。
3. 通过类名生成对应的词向量表示：加载 GloVe 向量表，将类名分词并查询词向量，取平均，输出每类的 300 维嵌入，保存为 `.npz` 文件。
4. 使用 ImageNet 提供的类别 ID 到类名映射，将原始 synset ID 替换为可读类名，并保存为辅助文本，便于语义嵌入。

### 2. 源代码改进

在阅读源代码时，我们发现作者在计算余弦相似度时没有考虑除零的问题，因为在语义嵌入时有可能找不到在 GloVe 中的对应向量，这时候可能产生零向量导致除 0 错误。我们改进了代码，当分母小于阈值时返回 `0.5 + 0.5 * 0`，避免了除零错误，提高了代码的鲁棒性。

## 七、实验结果

- 最终平均准确率：57.28% ± 0.80%。
- 每任务耗时：约 0.44 秒。
- 总运行时间：约 4 分钟。
- 准确率在 batch 200 后趋于收敛在 56.5% 到 57.5% 区间。
- 标准差较小，模型稳定性较好。
- 使用预训练 ResNet12 作为 backbone，无需 fine-tune 即可取得不错的 few-shot 表现。
- 相比于随机初始化，预训练模型提供了更好的初始化语义空间。
- 1-shot 场景下超过 57% 准确率是合理、具有代表性的 baseline。

## 八、总结与启示

本实验展示了在 LibFewShot 框架中，利用预训练视觉特征进行 few-shot 分类的效果。即使在 1-shot 情况下也能提供稳定可靠的性能。此结果可作为未来融合语义特征（GloVe、CLIP）或更复杂网络（如 ViT）的 baseline 参考。
