# 基于 Transformer 的机器翻译（EN→ZH）

使用 PyTorch 实现标准 Encoder-Decoder Transformer，在 `cmn-eng-simple` 中英平行语料上训练，评估指标为 BLEU。

## 环境准备

```bash
pip install -r requirements.txt
```

依赖：PyTorch (>=2.0, CUDA)、sacrebleu、jieba、opencc-python-reimplemented、subword-nmt、tqdm、pyyaml

## 数据预处理（已完成，可跳过）

`cmn-eng-simple/` 下的 `training.txt`、`validation.txt`、`testing.txt` 和词表 JSON 已就绪，无需重新预处理。

如需从头构建：

```powershell
cd cmn-eng-simple/preprocess
python tokenizer.py
Get-Content en.txt | subword-nmt learn-bpe -s 5000 | Set-Content en_code.txt
Get-Content en.txt | subword-nmt apply-bpe -c en_code.txt | Set-Content en_refine.txt
subword-nmt get-vocab --input en_refine.txt --output en_vocab.txt
python build_dataset.py
```

## 训练

```bash
python train_transformer.py --config configs/train_cuda.yaml
```

配置已针对 RTX 4060 (8GB) 优化（batch_size=64, AMP 开启）。训练约 60 epoch，耗时约 2-3 小时。

训练产物保存在 `runs_transformer_cuda/` 下的时间戳子目录中：
- `avg_model.pt` — 最终模型（top-5 checkpoint 平均）
- `best_model.pt` — 最佳单一 checkpoint
- `train_log.csv` — 逐 epoch 日志
- `metrics_train.json` — 结构化训练指标
- `test_predictions.txt` — 测试翻译结果（训练结束后自动生成）

## 单独测试

```bash
python train_transformer.py --mode test \
  --config configs/train_cuda.yaml \
  --checkpoint runs_transformer_cuda/<训练目录>/avg_model.pt \
  --max_bleu_samples 0
```

`--max_bleu_samples 0` 表示在全量测试集上评估。

## 项目结构

```
├── train_transformer.py          # 主脚本（模型定义 + 训练 + 测试 + beam search）
├── configs/
│   └── train_cuda.yaml           # 超参数配置
├── requirements.txt              # Python 依赖
├── cmn-eng-simple/               # 数据集
│   ├── training/validation/testing.txt
│   ├── word2int_*.json / int2word_*.json
│   └── preprocess/               # 原始数据与预处理脚本
└── runs_transformer_cuda/        # 训练输出（运行后生成）
```
