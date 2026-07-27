# QLoRA 在中文情感分析任务上的复现与改进研究

> 自然语言处理课程 · 期末小组项目

## 项目概述

本项目复现了 QLoRA（NeurIPS 2023）在中文情感分析任务上的参数高效微调方法，并通过 DoRA 改进实验实现了进一步的性能提升。所有实验均在单张 RTX 4060（8GB）上完成。

- **基础模型**：Qwen2.5-7B-Instruct（7.6B 参数）
- **数据集**：ChnSentiCorp（12,000 条中文评论，二分类）
- **最佳结果**：Accuracy = 95.92%，Macro-F1 = 95.89%（I1 DoRA 改进）

## 提交内容

```
提交_QLoRA中文情感分析复现与改进/
├── README.md                                           ← 本文件
├── 项目报告_QLoRA中文情感分析复现与改进.pdf              ← 实验报告（PDF）
├── PPT演示_QLoRA中文情感分析复现与改进.pptx              ← 答辩 PPT
└── 代码/
    ├── requirements.txt                                ← Python 依赖
    ├── docs_report.tex                                 ← 报告 LaTeX 源文件
    ├── configs/                                        ← 实验配置（YAML）
    │   ├── exp_r3_qlora_nf4_dq_alllinear.yaml          ← R3 核心复现
    │   ├── exp_a1_ablation_no_dq.yaml                  ← A1 消融：去除 DQ
    │   ├── exp_a2_ablation_fp4.yaml                    ← A2 消融：FP4 替代 NF4
    │   ├── exp_a3_ablation_rank16.yaml                 ← A3 消融：rank=16
    │   └── exp_i1_dora.yaml                            ← I1 改进：DoRA
    ├── scripts/                                        ← 全部 Python 脚本
    │   ├── prepare_data.py                             ← 数据预处理
    │   ├── train_qlora.py                              ← QLoRA 训练
    │   ├── infer_qlora.py                              ← Adapter 推理
    │   ├── infer_zero_shot.py                          ← Zero-shot 推理
    │   ├── eval.py                                     ← 评测（→ metrics.csv）
    │   └── plot_results.py                             ← 可视化
    └── results/
        ├── metrics.csv                                 ← 全部实验指标记录
        └── figures/                                    ← 训练曲线与对比图
```

## 实验结果

| ID     | 方法          | Epochs | Accuracy (%) | Macro-F1 (%) | Δ Acc vs R3 |
| ------ | ------------- | ------ | ------------ | ------------ | ----------- |
| R1     | Zero-shot     | —      | 90.17        | 90.10        | —           |
| R3     | QLoRA 核心    | 3      | 95.00        | 94.99        | —           |
| A1     | 去除 DQ       | 1      | 94.00        | 93.98        | −1.00       |
| A2     | FP4 替代 NF4  | 1      | 93.58        | 93.57        | −1.42       |
| A3     | rank=16       | 1      | 94.08        | 94.05        | −0.92       |
| **I1** | **DoRA 改进** | **3**  | **95.92**    | **95.89**    | **+0.92**   |

## 环境配置

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r 代码/requirements.txt
```

## 复现流程

```bash
# 1. 数据预处理
python 代码/scripts/prepare_data.py --input <原始数据路径> --output-dir data/processed/paper_a_core

# 2. 训练（以 R3 为例）
python 代码/scripts/train_qlora.py --config 代码/configs/exp_r3_qlora_nf4_dq_alllinear.yaml --model-name-or-path <模型路径>

# 3. 推理
python 代码/scripts/infer_qlora.py --base-model <模型路径> --adapter-dir outputs/r3_qlora_nf4_dq_alllinear/adapter --input-file data/processed/paper_a_core/test.jsonl --output-file outputs/r3_qlora_nf4_dq_alllinear/predictions.jsonl

# 4. 评测
python 代码/scripts/eval.py --pred-file outputs/r3_qlora_nf4_dq_alllinear/predictions.jsonl --exp-id R3 --method qlora_nf4_dq_alllinear --dataset ChnSentiCorp

# 5. 可视化
python 代码/scripts/plot_results.py
```

## 团队分工

| 成员   | 主要工作                                                                                                                                                                |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 许政阳 | 项目框架搭建（仓库结构、Git 协作流程）；核心训练脚本开发（train_qlora.py）；R2 冒烟测试与 R3 核心复现实验；报告撰写（实验设置、结果分析）                               |
| 王丹义 | 推理与评测脚本开发（infer_qlora.py、eval.py）；A1、A2、A3 消融实验（去除 DQ）与 I1 DoRA 改进实验；可视化脚本（plot_results.py）；报告撰写（引言、相关工作、讨论与结论） |
| 卞兆洲 | 论文调研（QLoRA、DoRA、LoRA 相关文献）；数据预处理脚本（prepare_data.py）与数据集构建；R1 Zero-shot 基线；PPT 制作与演讲视频录制                                        |
