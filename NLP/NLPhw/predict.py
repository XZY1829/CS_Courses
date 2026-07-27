import argparse
import os
from typing import Any, Dict, List

import torch

from src.data import build_data_bundle, load_preprocessor
from src.model import BiLSTMCRF
from src.utils import load_json, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict entities for a custom sentence.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/bilstm_crf_charcnn",
        help="Directory containing trained model artifacts.",
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default="EU rejects German call to boycott British lamb .",
        help="Input sentence. Use whitespace tokenization.",
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


def extract_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    entities = []
    current_tokens: List[str] = []
    current_type = None
    start_idx = -1

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_type is not None:
                entities.append(
                    {
                        "type": current_type,
                        "text": " ".join(current_tokens),
                        "start": start_idx,
                        "end": i - 1,
                    }
                )
            current_type = tag[2:]
            current_tokens = [token]
            start_idx = i
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_tokens.append(token)
        else:
            if current_type is not None:
                entities.append(
                    {
                        "type": current_type,
                        "text": " ".join(current_tokens),
                        "start": start_idx,
                        "end": i - 1,
                    }
                )
            current_type = None
            current_tokens = []
            start_idx = -1

    if current_type is not None:
        entities.append(
            {
                "type": current_type,
                "text": " ".join(current_tokens),
                "start": start_idx,
                "end": len(tokens) - 1,
            }
        )
    return entities


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
    model.eval()

    tokens = args.sentence.strip().split()
    if len(tokens) == 0:
        raise ValueError("Input sentence is empty.")

    lowercase = preprocessor["lowercase"]
    word_ids = bundle.word_vocab.encode(tokens, lowercase=lowercase)
    word_tensor = torch.tensor([word_ids], dtype=torch.long, device=device)
    mask_tensor = torch.ones((1, len(tokens)), dtype=torch.bool, device=device)

    char_tensor = None
    if bundle.char_vocab is not None:
        max_char_len = max(max(len(tok), 1) for tok in tokens)
        char_tensor = torch.full(
            (1, len(tokens), max_char_len),
            fill_value=bundle.char_vocab.pad_idx,
            dtype=torch.long,
            device=device,
        )
        for i, token in enumerate(tokens):
            char_ids = bundle.char_vocab.encode(list(token), lowercase=False)
            char_tensor[0, i, : len(char_ids)] = torch.tensor(
                char_ids, dtype=torch.long, device=device
            )

    with torch.no_grad():
        pred_ids = model.decode(word_ids=word_tensor, mask=mask_tensor, char_ids=char_tensor)[0]

    pred_tags = [bundle.id2label[idx] for idx in pred_ids]
    entities = extract_entities(tokens, pred_tags)

    print("Token-level tags:")
    for token, tag in zip(tokens, pred_tags):
        print(f"{token:15s} -> {tag}")

    print("\nEntities:")
    if len(entities) == 0:
        print("No entities found.")
    else:
        for ent in entities:
            print(
                f"[{ent['type']}] {ent['text']} "
                f"(token_span={ent['start']}..{ent['end']})"
            )


if __name__ == "__main__":
    main()
