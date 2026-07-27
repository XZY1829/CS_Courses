#!/usr/bin/env python
"""
Evaluate predictions and append records to results/metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score


METRIC_HEADER = [
    "exp_id",
    "method",
    "paper_ref",
    "dataset",
    "metric",
    "value",
    "seed",
    "notes",
    "owner",
    "date",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def ensure_metrics_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_HEADER)
        writer.writeheader()


def append_metric_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_HEADER)
        writer.writerow(row)


def normalize_label(text: Any) -> str:
    return str(text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions and record metrics.")
    parser.add_argument("--pred-file", required=True, help="Prediction file (jsonl)")
    parser.add_argument("--metrics-file", default="results/metrics.csv", help="Metrics CSV path")
    parser.add_argument("--exp-id", required=True, help="Experiment id, e.g. R3")
    parser.add_argument("--method", required=True, help="Method name, e.g. qlora_nf4_dq")
    parser.add_argument("--paper-ref", default="QLoRA Table3", help="Paper/table reference")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--seed", default="42", help="Seed used in training/eval")
    parser.add_argument("--owner", default="", help="Experiment owner")
    parser.add_argument("--notes", default="", help="Extra notes")
    parser.add_argument(
        "--metric-set",
        choices=["both", "acc", "f1"],
        default="both",
        help="Which metrics to save",
    )
    args = parser.parse_args()

    pred_file = Path(args.pred_file).resolve()
    metrics_file = Path(args.metrics_file).resolve()

    rows = read_jsonl(pred_file)
    if not rows:
        raise ValueError("Prediction file is empty.")

    y_true: list[str] = []
    y_pred: list[str] = []
    for row in rows:
        gold = row.get("gold", row.get("output"))
        pred = row.get("prediction", row.get("pred"))
        if gold is None or pred is None:
            raise ValueError("Each row must include gold/output and prediction/pred fields.")
        y_true.append(normalize_label(gold))
        y_pred.append(normalize_label(pred))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    ensure_metrics_file(metrics_file)
    today = datetime.now().strftime("%Y-%m-%d")

    if args.metric_set in {"both", "acc"}:
        append_metric_row(
            metrics_file,
            {
                "exp_id": args.exp_id,
                "method": args.method,
                "paper_ref": args.paper_ref,
                "dataset": args.dataset,
                "metric": "accuracy",
                "value": f"{acc:.6f}",
                "seed": args.seed,
                "notes": args.notes,
                "owner": args.owner,
                "date": today,
            },
        )
    if args.metric_set in {"both", "f1"}:
        append_metric_row(
            metrics_file,
            {
                "exp_id": args.exp_id,
                "method": args.method,
                "paper_ref": args.paper_ref,
                "dataset": args.dataset,
                "metric": "macro_f1",
                "value": f"{macro_f1:.6f}",
                "seed": args.seed,
                "notes": args.notes,
                "owner": args.owner,
                "date": today,
            },
        )

    print(f"Rows evaluated: {len(rows)}")
    print(f"accuracy: {acc:.6f}")
    print(f"macro_f1: {macro_f1:.6f}")
    print(f"Metrics appended to: {metrics_file}")


if __name__ == "__main__":
    main()
