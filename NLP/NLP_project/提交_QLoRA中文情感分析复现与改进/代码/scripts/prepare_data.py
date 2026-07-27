#!/usr/bin/env python
"""
Prepare a local dataset for Paper A QLoRA experiments.

Input formats supported:
- .jsonl
- .json (list of records, or {"data": [...]})
- .csv

Output format:
- Alpaca-style JSONL records with keys: instruction, input, output
- train/val/test split files under --output-dir
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def parse_label_map(raw: str) -> dict[str, str]:
    if not raw.strip():
        return {}
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        if ":" not in item:
            raise ValueError(f"Invalid label map entry: {item!r}, expected key:value")
        key, value = item.split(":", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL parse error at line {line_no}: {exc}") from exc
        return records

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [dict(x) for x in data["data"]]
        raise ValueError("JSON input must be a list or a dict with key 'data' as list")

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    raise ValueError(f"Unsupported input extension: {suffix}")


def to_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def to_alpaca_record(
    row: dict[str, Any],
    text_col: str,
    label_col: str,
    instruction: str,
    label_map: dict[str, str],
) -> dict[str, str]:
    # Already alpaca-style
    if "instruction" in row and "output" in row:
        return {
            "instruction": to_str(row.get("instruction")),
            "input": to_str(row.get("input")),
            "output": to_str(row.get("output")),
        }

    # Prompt/response style
    if "prompt" in row and "response" in row:
        return {
            "instruction": "Please answer the user request.",
            "input": to_str(row.get("prompt")),
            "output": to_str(row.get("response")),
        }

    # Text/label style (most common for classification)
    if text_col in row and label_col in row:
        label_value = to_str(row.get(label_col))
        label_value = label_map.get(label_value, label_value)
        return {
            "instruction": instruction,
            "input": to_str(row.get(text_col)),
            "output": label_value,
        }

    raise ValueError(
        "Cannot infer record format. Need one of: "
        "(instruction,input,output), (prompt,response), or custom --text-col/--label-col"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train-ratio must be between 0 and 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val-ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be < 1.0")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    shuffled = [records[i] for i in indices]

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    if n_test <= 0:
        raise ValueError(
            f"Split produced no test samples (total={n_total}, "
            f"train={n_train}, val={n_val}, test={n_test})"
        )
    return train_set, val_set, test_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local data for QLoRA experiments.")
    parser.add_argument("--input", required=True, help="Input data file (.jsonl/.json/.csv)")
    parser.add_argument(
        "--output-dir",
        default="data/processed/paper_a_smoke",
        help="Output directory for train/val/test JSONL files",
    )
    parser.add_argument("--text-col", default="text", help="Text column for classification data")
    parser.add_argument("--label-col", default="label", help="Label column for classification data")
    parser.add_argument(
        "--instruction",
        default="请根据输入文本完成任务，只输出最终标签，不要解释。",
        help="Instruction template for text/label data",
    )
    parser.add_argument(
        "--label-map",
        default="",
        help="Optional mapping like: 0:negative,1:positive",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use only first N samples after shuffle, 0 means all",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    label_map = parse_label_map(args.label_map)

    records = load_records(input_path)
    if not records:
        raise ValueError(f"No data found in {input_path}")

    converted = [
        to_alpaca_record(
            row=row,
            text_col=args.text_col,
            label_col=args.label_col,
            instruction=args.instruction,
            label_map=label_map,
        )
        for row in records
    ]

    random.Random(args.seed).shuffle(converted)
    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    train_set, val_set, test_set = split_records(
        records=converted,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    write_jsonl(train_path, train_set)
    write_jsonl(val_path, val_set)
    write_jsonl(test_path, test_set)

    meta = {
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "sizes": {
            "total": len(converted),
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
        },
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Data prepared successfully.")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")
    print(f"Sizes: {meta['sizes']}")


if __name__ == "__main__":
    main()
