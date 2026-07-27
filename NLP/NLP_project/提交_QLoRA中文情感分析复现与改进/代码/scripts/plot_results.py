#!/usr/bin/env python
"""生成训练曲线与实验对比图。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load_latest_state(run_dir: Path) -> Path | None:
    direct = run_dir / "trainer_state.json"
    if direct.exists():
        return direct

    checkpoints = []
    for ckpt in run_dir.glob("checkpoint-*"):
        if not ckpt.is_dir():
            continue
        state = ckpt / "trainer_state.json"
        if not state.exists():
            continue
        try:
            step = int(ckpt.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, state))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def plot_loss_curve(state_file: Path, output_path: Path, title: str = "") -> None:
    with state_file.open("r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    if not train_loss and not eval_loss:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_loss:
        ax.plot(train_steps, train_loss, label="Train Loss", alpha=0.85)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="Eval Loss", marker="o", markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title or state_file.parent.name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


def plot_comparison_bar(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)

    valid_ids = ["R1", "R2", "R3", "A1", "A2", "A3", "I1"]
    df = df[df["exp_id"].isin(valid_ids)].copy()
    if df.empty:
        return

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    df_acc = (
        df[df["metric"] == "accuracy"]
        .sort_values(["exp_id", "date"])
        .drop_duplicates(subset=["exp_id"], keep="last")
    )
    df_f1 = (
        df[df["metric"] == "macro_f1"]
        .sort_values(["exp_id", "date"])
        .drop_duplicates(subset=["exp_id"], keep="last")
    )

    merged = pd.merge(
        df_acc[["exp_id", "value"]].rename(columns={"value": "accuracy"}),
        df_f1[["exp_id", "value"]].rename(columns={"value": "macro_f1"}),
        on="exp_id",
        how="inner",
    )
    if merged.empty:
        return

    order = [x for x in valid_ids if x in merged["exp_id"].tolist()]
    merged["exp_id"] = pd.Categorical(merged["exp_id"], categories=order, ordered=True)
    merged = merged.sort_values("exp_id")

    exp_ids = merged["exp_id"].astype(str).tolist()
    acc_values = merged["accuracy"].tolist()
    f1_values = merged["macro_f1"].tolist()

    x = range(len(exp_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar([i - width / 2 for i in x], acc_values, width, label="Accuracy")
    bars2 = ax.bar([i + width / 2 for i in x], f1_values, width, label="Macro-F1")

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Score")
    ax.set_title("实验结果对比")
    ax.set_xticks(list(x))
    ax.set_xticklabels(exp_ids)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    for bar in bars2:
        ax.annotate(
            f"{bar.get_height():.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    outputs_root = project_root / "outputs"
    figures_dir = project_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if outputs_root.exists():
        for run_dir in sorted(outputs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            state = _load_latest_state(run_dir)
            if state is None:
                continue
            out = figures_dir / f"loss_{run_dir.name}.png"
            plot_loss_curve(state, out, title=run_dir.name)

    metrics_csv = project_root / "results" / "metrics.csv"
    if metrics_csv.exists():
        plot_comparison_bar(metrics_csv, figures_dir / "comparison_bar.png")

    print(f"[DONE] Figures dir: {figures_dir}")
