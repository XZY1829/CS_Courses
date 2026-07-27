#!/usr/bin/env python
"""
Zero-shot inference (no adapter) for baseline comparison.

Usage:
  python scripts/infer_zero_shot.py \
    --base-model D:/models/Qwen2.5-7B-Instruct \
    --input-file data/processed/paper_a_core/test.jsonl \
    --output-file outputs/r1_zero_shot/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_prompt(instruction: str, user_input: str) -> str:
    instruction = instruction.strip()
    user_input = user_input.strip()
    if user_input:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{user_input}\n\n"
            "### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot inference (no adapter).")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path/name; fallback to env MODEL_NAME_OR_PATH",
    )
    parser.add_argument("--input-file", required=True, help="Input JSONL file")
    parser.add_argument("--output-file", required=True, help="Output predictions JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    base_model = args.base_model or os.getenv("MODEL_NAME_OR_PATH")
    if not base_model:
        raise ValueError("Provide --base-model or set MODEL_NAME_OR_PATH env var.")

    input_file = Path(args.input_file).resolve()
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No input rows to infer.")

    print(f"Loading model: {base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded. Starting inference...")

    with output_file.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            prompt = build_prompt(
                str(row.get("instruction", "")), str(row.get("input", ""))
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            prediction = (
                generated_text[len(prompt) :].strip()
                if generated_text.startswith(prompt)
                else generated_text.strip()
            )

            out_row = {
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
                "gold": row.get("output", ""),
                "prediction": prediction,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(rows)}] done")

    print(f"\nZero-shot inference completed: {output_file}")
    print(f"Total samples: {len(rows)}")


if __name__ == "__main__":
    main()
