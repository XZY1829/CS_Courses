import argparse
import math
import os
from typing import Any, Dict

import torch
import yaml

from src.data import build_data_bundle, save_preprocessor
from src.model import BiLSTMCRF
from src.trainer import Trainer
from src.utils import count_parameters, ensure_dir, format_metrics, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiLSTM-CRF on CoNLL-2003.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bilstm_crf_charcnn.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/bilstm_crf_charcnn",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda or cpu. Auto-select if not set.",
    )
    parser.add_argument(
        "--epochs_override",
        type=int,
        default=None,
        help="Override epoch number in config for quick experiments.",
    )
    parser.add_argument(
        "--batch_size_override",
        type=int,
        default=None,
        help="Override batch size in config for quick experiments.",
    )
    parser.add_argument(
        "--seed_override",
        type=int,
        default=None,
        help="Override random seed in config (useful for multi-seed runs).",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any], bundle) -> BiLSTMCRF:
    model_cfg = config["model"]
    label_names = [bundle.id2label[idx] for idx in sorted(bundle.id2label)]
    model = BiLSTMCRF(
        vocab_size=len(bundle.word_vocab),
        num_labels=len(bundle.label2id),
        word_pad_idx=bundle.word_vocab.pad_idx,
        word_unk_idx=bundle.word_vocab.unk_idx,
        embedding_dim=int(model_cfg.get("embedding_dim", 100)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        lstm_layers=int(model_cfg.get("lstm_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.33)),
        word_dropout=float(model_cfg.get("word_dropout", 0.05)),
        use_char_cnn=bool(model_cfg.get("use_char_cnn", False)),
        char_vocab_size=len(bundle.char_vocab) if bundle.char_vocab is not None else 0,
        char_pad_idx=bundle.char_vocab.pad_idx if bundle.char_vocab is not None else 0,
        char_embedding_dim=int(model_cfg.get("char_embedding_dim", 30)),
        char_num_filters=int(model_cfg.get("char_num_filters", 50)),
        char_kernel_sizes=model_cfg.get("char_kernel_sizes", [3, 4, 5]),
        pretrained_word_embeddings=bundle.pretrained_word_embeddings,
        freeze_word_embeddings=bool(model_cfg.get("freeze_word_embeddings", False)),
        crf_constraint=str(model_cfg.get("crf_constraint", "none")),
        label_names=label_names,
        use_pos_chunk_aux=bool(model_cfg.get("use_pos_chunk_aux", False)),
        num_pos_labels=int(bundle.num_pos_labels),
        num_chunk_labels=int(bundle.num_chunk_labels),
    )
    return model


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    name = str(scheduler_name).lower()
    if name in {"", "none"}:
        return None
    if total_steps <= 0:
        return None

    warmup_steps = int(total_steps * max(0.0, warmup_ratio))
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        decay_steps = max(1, total_steps - warmup_steps)
        progress = float(step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)

        if name == "linear":
            return max(0.0, 1.0 - progress)
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError("scheduler must be one of: none, linear, cosine")

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed_override is not None:
        config["seed"] = int(args.seed_override)
    if args.epochs_override is not None:
        config["training"]["epochs"] = int(args.epochs_override)
    if args.batch_size_override is not None:
        config["training"]["batch_size"] = int(args.batch_size_override)

    ensure_dir(args.output_dir)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    save_json(os.path.join(args.output_dir, "config_resolved.json"), config)

    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    print(
        "Loading dataset (HuggingFace) and building vocab - first run can take several minutes...",
        flush=True,
    )
    bundle = build_data_bundle(config)
    save_preprocessor(
        path=os.path.join(args.output_dir, "preprocessor.json"),
        word_vocab=bundle.word_vocab,
        char_vocab=bundle.char_vocab,
        label2id=bundle.label2id,
        lowercase=bool(config["data"].get("lowercase", False)),
    )

    dataset_stats = {
        "train_size": len(bundle.train_loader.dataset),
        "dev_size": len(bundle.dev_loader.dataset),
        "test_size": len(bundle.test_loader.dataset),
        "word_vocab_size": len(bundle.word_vocab),
        "char_vocab_size": len(bundle.char_vocab) if bundle.char_vocab is not None else 0,
        "num_labels": len(bundle.label2id),
        "num_pos_labels": int(bundle.num_pos_labels),
        "num_chunk_labels": int(bundle.num_chunk_labels),
    }
    dataset_stats["use_pos_chunk_aux"] = bool(config["model"].get("use_pos_chunk_aux", False))
    if bundle.pretrained_stats is not None:
        dataset_stats.update(bundle.pretrained_stats)
    save_json(os.path.join(args.output_dir, "dataset_stats.json"), dataset_stats)
    print("Dataset stats:", dataset_stats, flush=True)

    model = build_model(config, bundle).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}", flush=True)

    training_cfg = config["training"]
    epochs = int(training_cfg.get("epochs", 20))
    accumulation_steps = max(1, int(training_cfg.get("accumulation_steps", 1)))
    updates_per_epoch = math.ceil(len(bundle.train_loader) / accumulation_steps)
    total_update_steps = updates_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 0.001)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=str(training_cfg.get("scheduler", "none")),
        total_steps=total_update_steps,
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.0)),
    )
    print(
        f"Optimization setup: scheduler={training_cfg.get('scheduler', 'none')}, "
        f"warmup_ratio={float(training_cfg.get('warmup_ratio', 0.0)):.3f}, "
        f"accumulation_steps={accumulation_steps}, total_update_steps={total_update_steps}",
        flush=True,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        id2label=bundle.id2label,
        grad_clip=float(training_cfg.get("grad_clip", 5.0)),
        aux_loss_weight=float(training_cfg.get("aux_loss_weight", 0.0)),
        scheduler=scheduler,
        use_amp=bool(training_cfg.get("use_amp", False)),
        accumulation_steps=accumulation_steps,
        ema_decay=float(training_cfg.get("ema_decay", 0.0)),
        use_ema_for_eval=bool(training_cfg.get("use_ema_for_eval", False)),
    )

    fit_state = trainer.fit(
        train_loader=bundle.train_loader,
        dev_loader=bundle.dev_loader,
        epochs=epochs,
        patience=int(training_cfg.get("patience", 5)),
        output_dir=args.output_dir,
        run_config=config,
    )
    print("Best dev state:", fit_state, flush=True)

    checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
    ckpt = torch.load(checkpoint_path, map_location=device)
    use_ema_ckpt = (
        ckpt.get("best_model_variant") == "ema" and ckpt.get("ema_model_state_dict") is not None
    )
    if use_ema_ckpt:
        model.load_state_dict(ckpt["ema_model_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    test_metrics, test_report = trainer.evaluate(bundle.test_loader)
    save_json(os.path.join(args.output_dir, "test_metrics.json"), test_metrics)
    with open(os.path.join(args.output_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_report)

    print("Test:", format_metrics(test_metrics), flush=True)
    print("Detailed test report saved to test_report.txt", flush=True)


if __name__ == "__main__":
    main()
