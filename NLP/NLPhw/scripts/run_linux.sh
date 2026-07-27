#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Installing dependencies..."
python -m pip install -r requirements.txt

echo "[2/6] Training baseline BiLSTM-CRF..."
python train.py --config configs/bilstm_crf.yaml --output_dir outputs/bilstm_crf

echo "[3/6] Training improved BiLSTM-CRF + CharCNN..."
python train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/bilstm_crf_charcnn

echo "[4/6] Evaluating improved model..."
python evaluate.py --model_dir outputs/bilstm_crf_charcnn --split test

echo "[5/6] Predicting one custom sentence..."
python predict.py --model_dir outputs/bilstm_crf_charcnn --sentence "EU rejects German call to boycott British lamb ."

echo "[6/6] Generating comparison markdown..."
python scripts/compare_experiments.py \
  --baseline_dir outputs/bilstm_crf \
  --improved_dir outputs/bilstm_crf_charcnn \
  --save_path outputs/experiment_comparison.md

echo "All steps completed."
