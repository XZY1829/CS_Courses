import json
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"P={metrics['precision'] * 100:.2f} "
        f"R={metrics['recall'] * 100:.2f} "
        f"F1={metrics['f1'] * 100:.2f}"
    )


def resolve_device(device_arg: Optional[str] = None) -> torch.device:
    """Pick torch.device from CLI. Auto mode prefers CUDA when torch is built with CUDA and a GPU is visible."""
    if device_arg is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            if torch.version.cuda is None:
                print(
                    "提示: 当前 PyTorch 为 CPU 构建 (torch.version.cuda 为 None)。"
                    "若本机有 NVIDIA 显卡，请安装 CUDA 版 PyTorch，见 TRAINING_GUIDE.md「GPU / CUDA」一节。"
                )
        return device

    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        msg = (
            "请求使用 CUDA，但 torch.cuda.is_available() 为 False。"
            "请确认已安装 NVIDIA 驱动 (nvidia-smi)，且安装的是 CUDA 版 PyTorch，不是默认的 CPU 轮子。"
        )
        if torch.version.cuda is None:
            msg += " 当前 wheel 为 CPU 构建。"
        raise RuntimeError(msg)
    return dev
