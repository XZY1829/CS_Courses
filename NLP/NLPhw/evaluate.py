import argparse
import os
from typing import Any, Dict

import torch

from src.data import build_data_bundle, load_preprocessor
from src.model import BiLSTMCRF
from src.trainer import Trainer
from src.utils import format_metrics, load_json, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained BiLSTM-CRF model.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/bilstm_crf_charcnn",
        help="Directory that contains best_model.pt and preprocessor.json.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="Evaluate on dev or test split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use. Auto-select if not set.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    model_dir = args.model_dir
    config = load_json(os.path.join(model_dir, "config_resolved.json"))
    preprocessor = load_preprocessor(os.path.join(model_dir, "preprocessor.json"))

    bundle = build_data_bundle(
        config=config,
        word_vocab=preprocessor["word_vocab"],
        char_vocab=preprocessor["char_vocab"],
        label2id=preprocessor["label2id"],
    )
    data_loader = bundle.dev_loader if args.split == "dev" else bundle.test_loader

    device = resolve_device(args.device)
    model = build_model(config, bundle).to(device)

    checkpoint = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device)
    use_ema_ckpt = (
        checkpoint.get("best_model_variant") == "ema"
        and checkpoint.get("ema_model_state_dict") is not None
    )
    if use_ema_ckpt:
        model.load_state_dict(checkpoint["ema_model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=device,
        id2label=bundle.id2label,
        grad_clip=5.0,
    )

    metrics, report = trainer.evaluate(data_loader)
    print(f"{args.split} split metrics: {format_metrics(metrics)}")
    print(report)

    save_json(os.path.join(model_dir, f"{args.split}_metrics_eval.json"), metrics)
    with open(os.path.join(model_dir, f"{args.split}_report_eval.txt"), "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
