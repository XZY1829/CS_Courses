# 新模型完整训练 Guidance（严格 RNN 高分版）

本指南对应当前升级后的 NER 管线：

- 主干：`BiLSTM-CRF (+ CharCNN)`
- 结构增强：预训练词向量、BIO 约束 CRF、POS/Chunk 多任务
- 训练增强：warmup+scheduler、AMP、梯度累积、EMA
- 冲分流程：multi-seed 批量训练 + majority vote 集成

目标是给你一套从环境到最终报告都可直接执行的完整流程。

---

## 0. 一页版执行流程（先看这个）

### Step A：环境与依赖

```powershell
cd c:\Users\zhengyang.xu\Desktop\WorkDoc\Study\NLPhw
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

可选：准备预训练词向量（推荐，通常能提升上限）：

```powershell
mkdir data\embeddings -Force
Invoke-WebRequest -Uri "https://nlp.stanford.edu/data/glove.6B.zip" -OutFile "data\embeddings\glove.6B.zip"
Expand-Archive -Path "data\embeddings\glove.6B.zip" -DestinationPath "data\embeddings" -Force
```

### Step B：先跑通 smoke（1 epoch）

```powershell
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf.yaml --output_dir outputs/smoke_bilstm --epochs_override 1
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/smoke_charcnn --epochs_override 1
.\.venv\Scripts\python.exe evaluate.py --model_dir outputs/smoke_charcnn --split test
```

### Step C：full 单模型训练

```powershell
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/full_charcnn
.\.venv\Scripts\python.exe evaluate.py --model_dir outputs/full_charcnn --split test
```

### Step D：multi-seed + ensemble（冲分）

```powershell
.\.venv\Scripts\python.exe scripts/run_multiseed.py --config configs/bilstm_crf_charcnn.yaml --output_root outputs/full_multiseed --seeds 42,43,44,45,46 --top_k 3
.\.venv\Scripts\python.exe scripts/ensemble_eval.py --summary_json outputs/full_multiseed/multiseed_summary.json --top_k 3 --split test --output_dir outputs/full_ensemble
```

---

## 1. 新模型到底“新”在哪

相对基础 `BiLSTM-CRF`，当前强模型主要新增了下面几层能力：

1. **Pretrained Embedding**
   - `data.pretrained_embeddings_path`
   - 自动构建 embedding matrix，记录覆盖率到 `dataset_stats.json`
2. **BIO-Constrained CRF**
   - `model.crf_constraint: bio`
   - 屏蔽非法转移（如 `O -> I-XXX`）
3. **POS/Chunk 多任务辅助**
   - `model.use_pos_chunk_aux: true`
   - 利用 CoNLL 自带 `pos_tags/chunk_tags` 做辅助监督
4. **训练策略增强**
   - `training.scheduler`: `none|linear|cosine`
   - `training.warmup_ratio`
   - `training.use_amp`
   - `training.accumulation_steps`
   - `training.ema_decay`
   - `training.use_ema_for_eval`（默认建议 `false`，避免短训练时 EMA 过平滑）

---

## 2. 环境准备（Windows）

### 2.1 Python 与虚拟环境

```powershell
py -3.11 -m venv .venv
```

如果 PowerShell 激活脚本被策略拦截，直接用解释器即可：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2.2 CUDA（可选但推荐）

先看是否有 NVIDIA 驱动：

```powershell
nvidia-smi
```

安装 CUDA 版 torch（示例 cu124）：

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

验证：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

---

## 3. 数据与预训练词向量

### 3.1 CoNLL-2003 数据

项目通过 HuggingFace `datasets` 自动下载：

```powershell
.\.venv\Scripts\python.exe -c "from datasets import load_dataset; load_dataset('conll2003', trust_remote_code=True)"
```

默认缓存目录通常在：

- `%USERPROFILE%\.cache\huggingface\datasets`

### 3.2 预训练词向量（可选但强烈推荐）

这一节给你“从 0 到可训练”的完整流程。  
当前项目最推荐先用 **GloVe 6B 100d**，因为与你默认配置 `embedding_dim: 100` 完全匹配。

#### 3.2.1 选哪种词向量

优先建议：

1. `GloVe 6B 100d`（稳定、下载方便、和当前配置兼容）
2. `GloVe 840B 300d`（更强但体积很大，不建议首次使用）
3. `FastText` 英文词向量（可作为额外对比实验）

如果你现在目标是“快速提升且少踩坑”，就先上 **GloVe 6B 100d**。

#### 3.2.2 下载与解压（Windows / PowerShell）

在项目根目录执行：

```powershell
mkdir data\embeddings -Force
Invoke-WebRequest -Uri "https://nlp.stanford.edu/data/glove.6B.zip" -OutFile "data\embeddings\glove.6B.zip"
Expand-Archive -Path "data\embeddings\glove.6B.zip" -DestinationPath "data\embeddings" -Force
```

解压后应看到这些文件：

- `data/embeddings/glove.6B.50d.txt`
- `data/embeddings/glove.6B.100d.txt`
- `data/embeddings/glove.6B.200d.txt`
- `data/embeddings/glove.6B.300d.txt`

#### 3.2.3 文件格式要求（非常重要）

本项目 loader 读取格式为：

- 每行一个词向量
- 第一列是 token
- 后面是浮点维度值

示例：

```text
the 0.418 0.24968 -0.41242 ...
of 0.013441 0.23682 -0.16899 ...
```

要求：

- 向量维度必须等于 `model.embedding_dim`
- 文件可以是绝对路径或项目相对路径
- 文本文件编码一般 `utf-8` 即可（loader 对异常行会自动跳过）

#### 3.2.4 在配置中启用

以 `configs/bilstm_crf_charcnn.yaml` 为例：

```yaml
data:
  pretrained_embeddings_path: data/embeddings/glove.6B.100d.txt
  lowercase: true

model:
  embedding_dim: 100
  freeze_word_embeddings: false
```

推荐解释：

- `lowercase: true`：GloVe 6B 主要是小写词表，通常能显著提高覆盖率
- `freeze_word_embeddings: false`：让词向量可继续微调，NER 任务通常效果更好

#### 3.2.5 首次加载如何验证“真的生效”

跑一个 1 epoch smoke：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/smoke_glove --epochs_override 1
```

训练日志里应看到类似信息：

- `Loading pretrained embeddings from: ...`
- `Pretrained embedding coverage: xx.xx% (hits/target_vocab)`

并且 `outputs/smoke_glove/dataset_stats.json` 中会出现：

- `pretrained_hits`
- `pretrained_target_vocab`
- `pretrained_coverage`
- `pretrained_embedding_dim`

#### 3.2.6 常见错误与处理

1. **维度不匹配**
   - 现象：报 shape mismatch / embedding dim 错误
   - 处理：保证 `model.embedding_dim` 与词向量文件维度一致（如 100 对 100d）

2. **路径错误**
   - 现象：`Pretrained embedding file not found`
   - 处理：检查路径拼写；优先用项目相对路径 `data/embeddings/...`

3. **覆盖率很低**
   - 处理：将 `data.lowercase` 设为 `true` 再试；优先使用英文通用词向量（如 GloVe）

4. **下载命令受网络限制**
   - 处理：可在浏览器手动下载 zip 后放入 `data/embeddings/` 再解压

#### 3.2.7 课程冲分建议（和预训练相关）

建议做一个最小消融对比：

- `baseline`：`pretrained_embeddings_path: null`
- `+pretrain`：启用 `glove.6B.100d.txt`

其它配置保持不变，然后对比：

- dev F1（训练中）
- test F1（`test_metrics.json`）

这样你可以在报告里清晰展示“预训练词向量”这项改进的净收益。

---

## 4. 配置建议（按阶段）

### 4.1 smoke（仅验证链路）

- `--epochs_override 1`
- 可用 `configs/bilstm_crf.yaml` 与 `configs/bilstm_crf_charcnn.yaml`

### 4.2 full 单模型（课程主结果）

建议从 `configs/bilstm_crf_charcnn.yaml` 起步，重点确认：

- `model.crf_constraint: bio`
- `model.use_pos_chunk_aux: true`
- `training.scheduler: cosine`
- `training.use_amp: true`（GPU）
- `training.ema_decay: 0.999`
- `training.use_ema_for_eval: false`（先稳；若跑很多 epoch 后再尝试设为 `true` 对比）

### 4.3 显存紧张时

- CLI 降 batch：`--batch_size_override 8`
- 配置提 `accumulation_steps`（如 2/4）补回等效 batch

---

## 5. 完整训练流程（可复制）

### 5.1 Smoke（强烈建议先跑）

```powershell
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf.yaml --output_dir outputs/smoke_bilstm --epochs_override 1
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/smoke_charcnn --epochs_override 1
.\.venv\Scripts\python.exe evaluate.py --model_dir outputs/smoke_charcnn --split test
.\.venv\Scripts\python.exe predict.py --model_dir outputs/smoke_charcnn --sentence "Barack Obama visited New York yesterday ."
.\.venv\Scripts\python.exe scripts/compare_experiments.py --baseline_dir outputs/smoke_bilstm --improved_dir outputs/smoke_charcnn --save_path outputs/smoke_comparison.md
```

### 5.2 Full 单模型训练

```powershell
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/full_charcnn
.\.venv\Scripts\python.exe evaluate.py --model_dir outputs/full_charcnn --split test
```

### 5.3 Multi-seed 批量训练

```powershell
.\.venv\Scripts\python.exe scripts/run_multiseed.py --config configs/bilstm_crf_charcnn.yaml --output_root outputs/full_multiseed --seeds 42,43,44,45,46 --top_k 3
```

这个脚本会：

- 逐 seed 训练并保存 `outputs/full_multiseed/seed_<seed>`
- 汇总 `best_dev_f1/test_f1`
- 输出：
  - `outputs/full_multiseed/multiseed_summary.json`
  - `outputs/full_multiseed/multiseed_summary.md`

### 5.4 Ensemble 评估（majority vote）

推荐直接从 summary 读取 top-k：

```powershell
.\.venv\Scripts\python.exe scripts/ensemble_eval.py --summary_json outputs/full_multiseed/multiseed_summary.json --top_k 3 --split test --output_dir outputs/full_ensemble
```

也可以手动指定：

```powershell
.\.venv\Scripts\python.exe scripts/ensemble_eval.py --model_dirs outputs/full_multiseed/seed_42,outputs/full_multiseed/seed_43,outputs/full_multiseed/seed_44 --split test --output_dir outputs/full_ensemble_manual
```

输出：

- `ensemble_metrics.json`
- `ensemble_report.txt`
- `ensemble_summary.md`

---

## 6. 输出文件怎么读

每个训练目录（如 `outputs/full_charcnn`）：

- `best_model.pt`：最佳 checkpoint（含 EMA 权重时优先用于评估）
- `history.json`：每个 epoch 的 `loss/dev_f1/lr`
- `dataset_stats.json`：词表规模、辅助标签规模、预训练覆盖率
- `test_metrics.json`：最终 `precision/recall/f1`
- `test_report.txt`：类别级报告
- `preprocessor.json`：词表和标签映射（推理/评估一致性关键）

---

## 7. 报告与复现素材

推荐直接使用模板并填数：

- `outputs/templates/ablation_template.md`
- `outputs/templates/ensemble_template.md`

建议呈现顺序：

1. baseline（`bilstm_crf`）
2. +pretrained embedding
3. +BIO constraint
4. +POS/Chunk multitask
5. +scheduler/AMP/accum/EMA
6. multi-seed ensemble

---

## 8. 常见问题排查

### 8.1 `python` 不可用

用 `py -3.11` 或 `.\.venv\Scripts\python.exe`。

### 8.2 PowerShell 无法 `Activate.ps1`

用：

```powershell
cmd /c ".venv\Scripts\activate.bat"
```

或直接不激活，用 `.\.venv\Scripts\python.exe`。

### 8.3 `Windows fatal exception: access violation`

```powershell
.\.venv\Scripts\python.exe -m pip install "numpy<2" "pandas<3" "pyarrow==17.0.0"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 8.4 有 GPU 但跑在 CPU

检查三件事：

1. `nvidia-smi` 正常
2. `torch.cuda.is_available()` 为 `True`
3. 训练日志出现 `Using device: cuda`

### 8.5 OOM（显存不足）

```powershell
.\.venv\Scripts\python.exe train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/tmp_small --batch_size_override 8
```

并在配置里提升 `training.accumulation_steps`。

---

## 9. 交付前检查清单

建议至少包含以下结果：

1. `outputs/full_charcnn/test_metrics.json`
2. `outputs/full_multiseed/multiseed_summary.json`
3. `outputs/full_ensemble/ensemble_metrics.json`
4. `outputs/templates/ablation_template.md`（已填）
5. `outputs/templates/ensemble_template.md`（已填）

满足这 5 项，基本就覆盖了“可复现 + 高分冲刺”需要的主链路。
