#!/usr/bin/env python
"""
Train Paper A (QLoRA) baseline with Transformers + PEFT.

Usage:
  python scripts/train_qlora.py --config configs/exp_r2_smoke_qlora_qwen7b.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Avoid importing TensorFlow/Keras stacks in this PyTorch-only pipeline.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

import yaml
from datasets import load_dataset
# Import datasets/pyarrow before torch on Windows.
# This avoids an intermittent access-violation seen when pyarrow initializes
# after torch runtime libraries are already loaded.
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

try:
    from transformers.training_args import OptimizerNames
except Exception:  # pragma: no cover - defensive for old/new transformers API changes
    OptimizerNames = None  # type: ignore[assignment]


DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class TeeTextIO:
    """Duplicate stdout/stderr to console and file."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    @property
    def encoding(self) -> str:
        return str(getattr(self._streams[0], "encoding", "utf-8"))

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._streams[0], "isatty", lambda: False)())

    def fileno(self) -> int:
        return int(getattr(self._streams[0], "fileno")())


def sanitize_filename(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip())
    return cleaned or "train"


def enable_file_logging(output_dir: Path, run_name: str | None) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = sanitize_filename(run_name or "train")
    log_path = log_dir / f"{run_tag}_{timestamp}.log"
    log_fp = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeTextIO(sys.__stdout__, log_fp)  # type: ignore[assignment]
    sys.stderr = TeeTextIO(sys.__stderr__, log_fp)  # type: ignore[assignment]
    print(f"[INFO] training log file: {log_path}")
    return log_path


class RobustCheckpointTrainer(Trainer):
    """Trainer with early trainer_state snapshot for crash resilience."""

    def __init__(self, *args: Any, checkpoint_train_batch_size: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._checkpoint_train_batch_size = checkpoint_train_batch_size

    def _ensure_checkpoint_train_batch_size(self) -> None:
        if getattr(self.state, "train_batch_size", None) is not None:
            return
        if self._checkpoint_train_batch_size is None:
            return
        self.state.train_batch_size = self._checkpoint_train_batch_size

    def _save_checkpoint(self, model: Any, trial: Any) -> None:
        if self.args.should_save:
            self._ensure_checkpoint_train_batch_size()
            run_dir = self._get_output_dir(trial=trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            checkpoint_dir = Path(run_dir) / checkpoint_folder
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.state.save_to_json(str(checkpoint_dir / TRAINER_STATE_NAME))
                print(f"[INFO] pre-saved trainer_state: {checkpoint_dir / TRAINER_STATE_NAME}")
            except Exception as exc:
                print(f"[WARN] failed to pre-save trainer_state before checkpoint write: {exc}")
            print(
                "[INFO] checkpoint save start: "
                f"step={self.state.global_step}, save_only_model={self.args.save_only_model}, "
                f"dir={checkpoint_dir}"
            )
        started_at = time.monotonic()
        super()._save_checkpoint(model, trial)
        if self.args.should_save:
            elapsed = time.monotonic() - started_at
            self._log_checkpoint_file_sizes(Path(self._get_output_dir(trial=trial)) / checkpoint_folder)

            print(f"[INFO] checkpoint save finished: step={self.state.global_step}, elapsed={elapsed:.1f}s")

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        print(f"[INFO] optimizer/scheduler save start: {output_dir}")
        started_at = time.monotonic()
        super()._save_optimizer_and_scheduler(output_dir)
        elapsed = time.monotonic() - started_at
        self._log_checkpoint_file_sizes(Path(output_dir))
        print(f"[INFO] optimizer/scheduler save finished: elapsed={elapsed:.1f}s")

    @staticmethod
    def _log_checkpoint_file_sizes(checkpoint_dir: Path) -> None:
        interesting_files = [
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            TRAINER_STATE_NAME,
        ]
        parts = []
        for name in interesting_files:
            path = checkpoint_dir / name
            if path.exists():
                parts.append(f"{name}={path.stat().st_size / (1024 * 1024):.1f}MB")
            else:
                parts.append(f"{name}=missing")
        print(f"[INFO] checkpoint files: {', '.join(parts)}")


class StopAfterSaveCallback(TrainerCallback):
    """Stop training only after a checkpoint has been written."""

    def __init__(self, stop_file: Path) -> None:
        self.stop_file = stop_file
        self._stop_requested = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self.stop_file.exists():
            if not self._stop_requested:
                print(
                    "[INFO] stop file detected; forcing a checkpoint before shutdown: "
                    f"{self.stop_file}"
                )
            self._stop_requested = True
            control.should_save = True
        return control

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self._stop_requested or self.stop_file.exists():
            handled_file = self.stop_file.with_name(
                f"{self.stop_file.name}.handled.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            try:
                if self.stop_file.exists():
                    self.stop_file.replace(handled_file)
                    print(f"[INFO] stop file marked handled: {handled_file}")
            except OSError as exc:
                print(f"[WARN] failed to mark stop file handled: {exc}")
            print(f"[INFO] checkpoint saved; stopping training at global_step={state.global_step}.")
            control.should_training_stop = True
        return control


def get_supported_optimizers() -> set[str]:
    if OptimizerNames is None:
        return set()
    return {str(item.value) for item in OptimizerNames}


def validate_optimizer_name(optim_name: str) -> None:
    supported = get_supported_optimizers()
    if not supported:
        return
    if optim_name in supported:
        return

    paged_supported = sorted([name for name in supported if name.startswith("paged_")])
    raise ValueError(
        f"Unsupported optimizer '{optim_name}'. "
        f"Your transformers supports: {', '.join(sorted(supported))}. "
        f"Try one of paged optimizers: {paged_supported}, or upgrade transformers."
    )


def build_max_memory_map(max_memory_mb: int | None) -> dict[int | str, str] | None:
    if max_memory_mb is None:
        return None
    if max_memory_mb <= 0:
        return None
    if not torch.cuda.is_available():
        return None

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 0:
        return None

    reserve_mb = 512

    def capped_limit_for_device(device_index: int) -> int:
        total_mb = int(torch.cuda.get_device_properties(device_index).total_memory / (1024 * 1024))
        safe_cap_mb = max(total_mb - reserve_mb, 512)
        if max_memory_mb > safe_cap_mb:
            print(
                "[WARN] max_memory_mb exceeds safe capacity on "
                f"cuda:{device_index} (requested={max_memory_mb}MB, total={total_mb}MB, capped={safe_cap_mb}MB)."
            )
            return safe_cap_mb
        return max_memory_mb

    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is not None:
        local_rank = int(local_rank_raw)
        if local_rank >= 0:
            capped_mb = capped_limit_for_device(local_rank)
            return {"": f"{capped_mb}MB"}
        return None

    return {i: f"{capped_limit_for_device(i)}MB" for i in range(n_gpus)}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(raw: str | None, base_dir: Path) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def resolve_optional_output_path(raw: Any, output_dir: Path) -> Path | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None if raw is False else output_dir / "STOP_TRAINING"
    value = str(raw).strip()
    if not value or value.lower() in {"false", "none", "off", "0"}:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (output_dir / path).resolve()


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


def ensure_required(cfg: dict[str, Any], keys: list[str]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")


def resolve_resume_checkpoint(resume_raw: str | None, output_dir: Path) -> Path | None:
    def checkpoint_latest_mtime(path: Path) -> float:
        model_file_names = [
            "adapter_model.safetensors",
            "model.safetensors",
            "pytorch_model.bin",
        ]
        model_mtimes = []
        for name in model_file_names:
            model_file = path / name
            if model_file.exists():
                model_mtimes.append(model_file.stat().st_mtime)
        if model_mtimes:
            return max(model_mtimes)

        latest = path.stat().st_mtime
        for child in path.iterdir():
            try:
                latest = max(latest, child.stat().st_mtime)
            except OSError:
                continue
        return latest

    def find_most_recent_checkpoint_dir(base_dir: Path) -> Path | None:
        checkpoint_dirs = [p for p in base_dir.glob("checkpoint-*") if p.is_dir()]
        if not checkpoint_dirs:
            return None
        checkpoint_dirs.sort(key=checkpoint_latest_mtime, reverse=True)
        return checkpoint_dirs[0]

    if resume_raw is None:
        return None
    resume_value = str(resume_raw).strip()
    if not resume_value:
        return None

    if resume_value.lower() == "last":
        most_recent_checkpoint = find_most_recent_checkpoint_dir(output_dir)
        if most_recent_checkpoint is None:
            # Fallback to transformers helper for compatibility with unusual layouts.
            last_checkpoint = get_last_checkpoint(str(output_dir))
            if last_checkpoint is not None:
                most_recent_checkpoint = Path(last_checkpoint).resolve()
        if most_recent_checkpoint is None:
            raise FileNotFoundError(
                "resume_from_checkpoint is set to 'last', "
                f"but no checkpoint found under output_dir: {output_dir}"
            )
        resolved = most_recent_checkpoint.resolve()
        print(f"[INFO] resume checkpoint (last): {resolved}")
        return resolved

    checkpoint_path = Path(resume_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    print(f"[INFO] resume checkpoint (explicit): {checkpoint_path}")
    return checkpoint_path


def resolve_resume_strategy(
    resume_raw: str | None, output_dir: Path, train_batch_size_hint: int | None = None
) -> tuple[str | None, str | None]:
    checkpoint_path = resolve_resume_checkpoint(resume_raw, output_dir)
    if checkpoint_path is None:
        return None, None

    trainer_state = checkpoint_path / "trainer_state.json"
    adapter_config = checkpoint_path / "adapter_config.json"
    adapter_weights = checkpoint_path / "adapter_model.safetensors"
    optimizer_state = checkpoint_path / "optimizer.pt"

    def parse_checkpoint_step(path: Path) -> int | None:
        match = re.search(r"checkpoint-(\d+)$", path.name)
        if match is None:
            return None
        return int(match.group(1))

    def synthesize_trainer_state(path: Path) -> bool:
        step = parse_checkpoint_step(path)
        if step is None:
            return False
        train_batch_size = train_batch_size_hint if train_batch_size_hint and train_batch_size_hint > 0 else 1
        synthetic_state = TrainerState(global_step=step, train_batch_size=train_batch_size)
        synthetic_state.save_to_json(str(path / TRAINER_STATE_NAME))
        print(
            "[WARN] checkpoint is missing trainer_state.json; "
            f"synthesized minimal state at step={step} (train_batch_size={train_batch_size}) to enable resume."
        )
        return True

    if trainer_state.exists():
        try:
            state_data = json.loads(trainer_state.read_text(encoding="utf-8"))
            if state_data.get("train_batch_size") is None and train_batch_size_hint and train_batch_size_hint > 0:
                state_data["train_batch_size"] = train_batch_size_hint
                trainer_state.write_text(json.dumps(state_data, indent=2), encoding="utf-8")
                print(
                    "[WARN] checkpoint trainer_state.json had null train_batch_size; "
                    f"patched to {train_batch_size_hint}."
                )
        except Exception as exc:
            print(f"[WARN] failed to inspect trainer_state.json: {exc}")
        return str(checkpoint_path), None

    if optimizer_state.exists():
        try:
            if synthesize_trainer_state(checkpoint_path):
                return str(checkpoint_path), None
        except Exception as exc:
            print(f"[WARN] failed to synthesize trainer_state.json: {exc}")

    if adapter_config.exists() and adapter_weights.exists():
        print(
            "[WARN] checkpoint is missing trainer_state.json; "
            "fallback to adapter-weights warm start (optimizer/scheduler state will not be resumed)."
        )
        return None, str(checkpoint_path)

    raise FileNotFoundError(
        "Checkpoint is incomplete: missing trainer_state.json and adapter files. "
        f"Path: {checkpoint_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train QLoRA baseline (Paper A).")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Override base model path/name (takes precedence over config and env)",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Resume from checkpoint path. "
            "Use 'last' to auto-detect the latest checkpoint in output_dir."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    cfg = load_yaml(config_path)

    ensure_required(cfg, ["train_file", "output_dir", "lora_r", "lora_alpha", "learning_rate"])

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    model_name_or_path = (
        args.model_name_or_path
        or cfg.get("model_name_or_path")
        or os.getenv("MODEL_NAME_OR_PATH")
    )
    if not model_name_or_path:
        raise ValueError(
            "Base model is not set. Provide --model-name-or-path, "
            "or set model_name_or_path in config, or export MODEL_NAME_OR_PATH."
        )
    model_name_or_path = str(model_name_or_path)
    train_file = resolve_path(cfg.get("train_file"), config_dir)
    validation_file = resolve_path(cfg.get("validation_file"), config_dir)
    output_dir = resolve_path(cfg.get("output_dir"), config_dir)
    max_samples = int(cfg.get("max_samples", 0))

    if train_file is None or not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if validation_file is not None and not validation_file.exists():
        print(f"[WARN] Validation file does not exist, skip eval: {validation_file}")
        validation_file = None
    if output_dir is None:
        raise ValueError("output_dir cannot be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    enable_file_logging(output_dir, str(cfg.get("run_name", "")))
    stop_after_save_file = resolve_optional_output_path(cfg.get("stop_after_save_file", "STOP_TRAINING"), output_dir)
    if stop_after_save_file is not None:
        print(f"[INFO] graceful stop file: {stop_after_save_file}")
    resume_from_checkpoint, warm_start_adapter_checkpoint = resolve_resume_strategy(
        args.resume_from_checkpoint,
        output_dir,
        train_batch_size_hint=int(cfg.get("per_device_train_batch_size", 1)),
    )

    use_4bit = bool(cfg.get("use_4bit", True))
    optim_name = str(cfg.get("optim", "paged_adamw_32bit"))
    validate_optimizer_name(optim_name)
    max_memory_mb_raw = cfg.get("max_memory_mb")
    max_memory_mb = int(max_memory_mb_raw) if max_memory_mb_raw is not None else None
    max_memory = build_max_memory_map(max_memory_mb)

    compute_dtype = DTYPE_MAP.get(
        str(cfg.get("bnb_4bit_compute_dtype", "bfloat16")).lower(),
        torch.bfloat16,
    )

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
        print(f"[INFO] max_memory enabled: {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    print(f"[INFO] optimizer: {optim_name}")

    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))
    # Keep historical runtime behavior by default while still passing this
    # argument explicitly to avoid PyTorch's deprecation warning.
    gradient_checkpointing_use_reentrant = bool(cfg.get("gradient_checkpointing_use_reentrant", True))
    gradient_checkpointing_kwargs = (
        {"use_reentrant": gradient_checkpointing_use_reentrant} if gradient_checkpointing else None
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    if warm_start_adapter_checkpoint is not None:
        model = PeftModel.from_pretrained(
            model,
            warm_start_adapter_checkpoint,
            is_trainable=True,
        )
        print(f"[INFO] warm-start adapter loaded from: {warm_start_adapter_checkpoint}")
    else:
        lora_target_modules = cfg.get("lora_target_modules", "all-linear")
        use_dora = bool(cfg.get("use_dora", False))
        lora_config = LoraConfig(
            r=int(cfg.get("lora_r", 64)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=use_dora,
        )
        if use_dora:
            print("[INFO] DoRA enabled (use_dora=True)")
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_files: dict[str, str] = {"train": str(train_file)}
    if validation_file is not None:
        data_files["validation"] = str(validation_file)

    raw_ds = load_dataset("json", data_files=data_files)
    if max_samples > 0:
        raw_ds["train"] = raw_ds["train"].select(range(min(max_samples, len(raw_ds["train"]))))
        if "validation" in raw_ds:
            raw_ds["validation"] = raw_ds["validation"].select(
                range(min(max_samples, len(raw_ds["validation"])))
            )

    max_seq_length = int(cfg.get("max_seq_length", 512))

    def preprocess(ex: dict[str, Any]) -> dict[str, Any]:
        instruction = str(ex.get("instruction", "")).strip()
        user_input = str(ex.get("input", "")).strip()
        output = str(ex.get("output", "")).strip()

        prompt_text = build_prompt(instruction, user_input)
        full_text = prompt_text + output + (tokenizer.eos_token or "")

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        prompt_tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )

        labels = tokenized["input_ids"].copy()
        prompt_len = min(len(prompt_tokenized["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
        return tokenized

    remove_cols = raw_ds["train"].column_names
    tokenized_ds = raw_ds.map(
        preprocess,
        remove_columns=remove_cols,
        num_proc=1,
        desc="Tokenizing",
    )

    eval_strategy = "steps" if "validation" in tokenized_ds else "no"
    save_strategy = str(cfg.get("save_strategy", "steps")).strip().lower()
    if save_strategy not in {"steps", "epoch", "no"}:
        raise ValueError("save_strategy must be one of: steps, epoch, no")
    save_steps = int(cfg.get("save_steps", 100))
    eval_steps = int(cfg.get("eval_steps", 100))
    load_best_model_at_end = bool("validation" in tokenized_ds and save_strategy != "no")
    if eval_strategy == "steps" and load_best_model_at_end:
        if eval_steps <= 0:
            raise ValueError("eval_steps must be > 0 when eval_strategy='steps'.")
        if save_strategy == "steps" and save_steps <= 0:
            raise ValueError("save_steps must be > 0 when save_strategy='steps'.")
        if save_strategy == "steps" and save_steps % eval_steps != 0:
            print(
                "[WARN] save_steps is not a multiple of eval_steps while load_best_model_at_end=True. "
                "Disabling load_best_model_at_end to avoid Trainer initialization failure."
            )
            load_best_model_at_end = False

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        bf16=bool(cfg.get("bf16", True)),
        fp16=bool(cfg.get("fp16", False)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        save_only_model=bool(cfg.get("save_only_model", False)),
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        optim=optim_name,
        load_best_model_at_end=load_best_model_at_end,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=False,
        label_names=["labels"],
        seed=seed,
    )

    trainer = RobustCheckpointTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"] if "validation" in tokenized_ds else None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        callbacks=([StopAfterSaveCallback(stop_after_save_file)] if stop_after_save_file is not None else None),
        checkpoint_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    eval_metrics: dict[str, Any] = {}
    if "validation" in tokenized_ds:
        eval_metrics = trainer.evaluate()

    adapter_dir = output_dir / "adapter"
    tokenizer_dir = output_dir / "tokenizer"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(tokenizer_dir))

    metrics = dict(train_result.metrics)
    metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
    metrics["train_samples"] = len(tokenized_ds["train"])
    metrics["validation_samples"] = len(tokenized_ds["validation"]) if "validation" in tokenized_ds else 0

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training completed.")
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Tokenizer saved to: {tokenizer_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
