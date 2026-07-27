@echo off
setlocal enabledelayedexpansion

set "PYTHON_EXE=python"
%PYTHON_EXE% -V >nul 2>nul
if errorlevel 1 (
  set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"
)

"%PYTHON_EXE%" -V >nul 2>nul
if errorlevel 1 (
  echo Python not found. Please install Python 3.10+ first.
  exit /b 1
)

echo [1/6] Installing dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [2/6] Training baseline BiLSTM-CRF...
"%PYTHON_EXE%" train.py --config configs/bilstm_crf.yaml --output_dir outputs/bilstm_crf
if errorlevel 1 goto :error

echo [3/6] Training improved BiLSTM-CRF + CharCNN...
"%PYTHON_EXE%" train.py --config configs/bilstm_crf_charcnn.yaml --output_dir outputs/bilstm_crf_charcnn
if errorlevel 1 goto :error

echo [4/6] Evaluating improved model...
"%PYTHON_EXE%" evaluate.py --model_dir outputs/bilstm_crf_charcnn --split test
if errorlevel 1 goto :error

echo [5/6] Predicting one custom sentence...
"%PYTHON_EXE%" predict.py --model_dir outputs/bilstm_crf_charcnn --sentence "EU rejects German call to boycott British lamb ."
if errorlevel 1 goto :error

echo [6/6] Generating comparison markdown...
"%PYTHON_EXE%" scripts/compare_experiments.py --baseline_dir outputs/bilstm_crf --improved_dir outputs/bilstm_crf_charcnn --save_path outputs/experiment_comparison.md
if errorlevel 1 goto :error

echo All steps completed.
goto :end

:error
echo Script failed. Check the error messages above.
exit /b 1

:end
endlocal
