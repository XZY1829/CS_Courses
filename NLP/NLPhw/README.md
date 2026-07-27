# 课程实践1：基于循环神经网络的命名实体识别（CoNLL-2003）

本项目使用 **PyTorch** 实现 `BiLSTM-CRF` 命名实体识别，并提供可选改进 `CharCNN`（字符级卷积）模块。  
适配作业要求：**RNN实现 + CoNLL-2003 + Precision/Recall/F1 指标 + 代码可运行 + 完整报告材料**。

## 为什么选这个题目更容易冲高分

- 任务与课程要求高度匹配（老师点名 RNN-NER）。
- `BiLSTM-CRF` 在 CoNLL-2003 上属于经典强基线，容易做出高质量结果。
- 报告可写内容丰富：模型结构、消融实验、误差分析、可视化案例都容易展开。
- 本仓库额外包含 `CharCNN` 改进和对比脚本，便于写“创新点”。

## 项目亮点

- 标准实现：`BiLSTM + CRF` 序列标注。
- 创新增强：`CharCNN` 字符级形态特征（英文NER很常用且有效）。
- 训练技巧：`word dropout`、`gradient clipping`、`early stopping`。
- 实验完整：基线模型与改进模型均可一键训练并自动对比。
- 交付完整：含中文报告模板与 1-2 分钟演示视频脚本。

## 当前实测结果（本机CPU）

- `BiLSTM-CRF`：P=83.26, R=73.81, F1=78.25
- `BiLSTM-CRF + CharCNN`：P=84.38, R=83.96, F1=84.17
- F1 提升：+5.91

对应文件：

- `outputs/final_bilstm/test_metrics.json`
- `outputs/final_charcnn/test_metrics.json`
- `outputs/final_comparison.md`

## 目录结构

```text
NLPhw/
├─ configs/
│  ├─ bilstm_crf.yaml
│  └─ bilstm_crf_charcnn.yaml
├─ src/
│  ├─ crf.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ model.py
│  ├─ trainer.py
│  └─ utils.py
├─ scripts/
│  ├─ run_windows.bat
│  ├─ run_linux.sh
│  └─ compare_experiments.py
├─ report/
│  ├─ 课程实践1_基于RNN的NER实验报告.md
│  └─ 演示视频脚本.md
├─ train.py
├─ evaluate.py
├─ predict.py
└─ requirements.txt
```

## 环境要求

- Python 3.9+（推荐 3.10 或 3.11）
- PyTorch 2.2+
- 可联网下载 HuggingFace `conll2003` 数据集
- `datasets` 建议使用 2.x（已在 `requirements.txt` 固定）

完整操作说明见：`TRAINING_GUIDE.md`

## 快速开始

### 1) 安装依赖

```bash
python -m pip install -r requirements.txt
```

> 如果 `python` 命令不可用，Windows 可改用：`%LocalAppData%\Programs\Python\Python311\python.exe`

### 2) 训练基线模型

```bash
python train.py --config configs/bilstm_crf.yaml --output_dir outputs/bilstm_crf
```

### 3) 训练改进模型（CharCNN）

```bash
python train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/bilstm_crf_charcnn
```

### 4) 评估模型

```bash
python evaluate.py --model_dir outputs/bilstm_crf_charcnn --split test
```

### 5) 单句预测演示

```bash
python predict.py --model_dir outputs/bilstm_crf_charcnn --sentence "EU rejects German call to boycott British lamb ."
```

### 6) 生成对比结果（写报告用）

```bash
python scripts/compare_experiments.py --baseline_dir outputs/bilstm_crf --improved_dir outputs/bilstm_crf_charcnn --save_path outputs/experiment_comparison.md
```

## Windows 一键脚本

```bat
scripts\run_windows.bat
```

## 输出文件说明

- `outputs/*/best_model.pt`：最佳模型权重
- `outputs/*/history.json`：训练过程记录
- `outputs/*/test_metrics.json`：测试集 P/R/F1
- `outputs/*/test_report.txt`：分类别详细报告
- `outputs/experiment_comparison.md`：基线与改进对比

## 如何写成高分报告

建议结构（已给模板）：

1. 任务背景与目标
2. 方法：BiLSTM-CRF + CharCNN
3. 实验设置：数据、参数、训练策略
4. 结果对比：基线 vs 改进
5. 误差分析：典型正确/错误案例
6. 总结与改进方向

## 演示视频建议（1-2分钟）

直接参考 `report/演示视频脚本.md`：

- 10s 任务介绍
- 30s 代码结构
- 40s 训练与评估结果
- 20s 预测案例展示
- 10s 总结创新点

---

如果你要再冲更高分，可以进一步加：

- 预训练词向量（GloVe）
- 多次随机种子取均值
- 混淆实体类型的误差统计图

