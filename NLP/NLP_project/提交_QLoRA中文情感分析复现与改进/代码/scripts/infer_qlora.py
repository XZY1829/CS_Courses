#!/usr/bin/env python
"""
Run inference with a QLoRA adapter and export predictions.

Usage:
  python scripts/infer_qlora.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter-dir outputs/r2_smoke_qlora_qwen7b/adapter \
    --input-file data/processed/paper_a_smoke/test.jsonl \
    --output-file outputs/r2_smoke_qlora_qwen7b/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
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
    return "### Instruction:\n" f"{instruction}\n\n" "### Response:\n"


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
    parser = argparse.ArgumentParser(description="Inference for QLoRA adapter.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path/name; fallback to env MODEL_NAME_OR_PATH",
    )
    parser.add_argument("--adapter-dir", required=True, help="Adapter output directory")
    parser.add_argument("--input-file", required=True, help="Input JSONL file")
    parser.add_argument("--output-file", required=True, help="Output predictions JSONL")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    args = parser.parse_args()
    base_model_name_or_path = args.base_model or os.getenv("MODEL_NAME_OR_PATH")
    if not base_model_name_or_path:
        raise ValueError(
            "Base model is not set. Provide --base-model or set MODEL_NAME_OR_PATH."
        )

    input_file = Path(args.input_file).resolve()
    output_file = Path(args.output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_file)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No input rows to infer.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    with output_file.open("w", encoding="utf-8") as f:
        for row in rows:
            prompt = build_prompt(str(row.get("instruction", "")), str(row.get("input", "")))
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            do_sample = args.temperature > 0
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_p"] = args.top_p
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            prediction = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text

            out_row = {
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
                "gold": row.get("output", ""),
                "prediction": prediction,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"Inference completed: {output_file}")
    print(f"Samples: {len(rows)}")


if __name__ == "__main__":
    main()
