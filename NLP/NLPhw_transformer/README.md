# EN->ZH Transformer (PyTorch)

这个项目用于完成作业：英文翻译中文（`cmn-eng-simple`），评估指标为 BLEU。

## 1) 环境准备

```powershell
cd F:\documents_personal\NLP\NLPhw_transformer
pip install -r requirements.txt
```

## 2) 训练与测试解耦（支持时间戳目录）

脚本支持两种模式：

- `--mode train`：只训练 + 验证（默认不跑测试）
- `--mode test`：加载已有 checkpoint 单独测试

每次运行都会在 `output_dir` 下创建时间戳子目录，例如：

- `runs_transformer_cuda/train_20260530_183012_laptop4060`
- `runs_transformer_cuda/test_20260530_183540_epoch24`

不会覆盖历史实验，可并行“训练新模型 + 测试旧模型”。

## 3) 4060 Laptop 推荐训练命令

默认配置已针对 4060 Laptop（8GB）做了折中优化：

- `batch_size: 80`
- `eval_batch_size: 96`
- `AMP: on`
- `save_epoch_checkpoints: false`（仅保留最佳模型）
- 更稳的早停（`early_stop_min_delta` + `loss_explosion_*`）

运行：

```powershell
python .\train_transformer.py --config .\configs\train_cuda.yaml
```

## 4) 训练产物（实验报告可直接用）

每个训练 run 目录下会保存：

- `args.json`：完整参数快照
- `runtime.json`：设备/环境信息（含 GPU 名称和显存）
- `train_log.csv`：逐 epoch 日志（loss、BLEU、lr、耗时）
- `metrics_train.json`：结构化汇总（best/final/history/stop_reason）
- `best_model.pt`：最佳验证 BLEU 模型

## 5) 用任意已训练模型单独测试

```powershell
python .\train_transformer.py `
  --mode test `
  --data_dir .\cmn-eng-simple `
  --output_dir .\runs_transformer_cuda `
  --checkpoint ".\runs_transformer_cuda\train_20260530_183012_laptop4060\best_model.pt" `
  --beam_size 5 `
  --beam_alpha 0.6 `
  --test_split test
```

测试 run 目录下会保存：

- `metrics_test.json`
- `test_predictions.tok.txt`
- `test_predictions.txt`

## 6) 如果想训练结束后自动跑测试

加上：

```powershell
--run_test_after_train
```
