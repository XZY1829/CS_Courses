import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_data_bundle, load_preprocessor
from src.metrics import compute_ner_metrics, ids_to_tag_sequences, ner_classification_report
from src.model import BiLSTMCRF
from src.utils import load_json, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate majority-vote ensemble on CoNLL NER.")
    parser.add_argument(
        "--model_dirs",
        type=str,
        default="",
        help="Comma-separated model directories. Optional when --summary_json is used.",
    )
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Path to multiseed_summary.json; uses recommended_model_dirs from it.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="If set with --summary_json, keep only the first top_k recommended dirs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="Evaluate ensemble on dev or test split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ensemble_eval",
        help="Directory for ensemble metrics and report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/cpu).",
    )
    return parser.parse_args()


def build_model(config: Dict[str, Any], bundle) -> BiLSTMCRF:
    model_cfg = config["model"]
    label_names = [bundle.id2label[idx] for idx in sorted(bundle.id2label)]
    return BiLSTMCRF(
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


def parse_model_dirs(args: argparse.Namespace, project_root: Path) -> List[Path]:
    model_dirs: List[str] = []

    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        if not summary_path.is_absolute():
            summary_path = (project_root / summary_path).resolve()
        summary = load_json(str(summary_path))
        model_dirs = list(summary.get("recommended_model_dirs", []))
        if args.top_k is not None:
            model_dirs = model_dirs[: max(1, int(args.top_k))]
    elif args.model_dirs.strip():
        model_dirs = [item.strip() for item in args.model_dirs.split(",") if item.strip()]

    if len(model_dirs) == 0:
        raise ValueError("Provide --model_dirs or --summary_json with recommended_model_dirs.")

    resolved = []
    for item in model_dirs:
        path = Path(item)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        resolved.append(path)
    return resolved


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {
        "word_ids": batch["word_ids"].to(device),
        "mask": batch["mask"].to(device),
    }
    if isinstance(batch.get("char_ids"), torch.Tensor):
        moved["char_ids"] = batch["char_ids"].to(device)
    else:
        moved["char_ids"] = None
    return moved


def collect_targets(data_loader) -> Tuple[List[List[int]], List[List[bool]]]:
    gold_ids: List[List[int]] = []
    masks: List[List[bool]] = []
    for batch in data_loader:
        gold_ids.extend(batch["tag_ids"].tolist())
        masks.extend(batch["mask"].tolist())
    return gold_ids, masks


@torch.no_grad()
def collect_predictions(model: BiLSTMCRF, data_loader, device: torch.device) -> List[List[int]]:
    model.eval()
    preds: List[List[int]] = []
    for batch in data_loader:
        moved = move_batch(batch, device)
        batch_preds = model.decode(
            word_ids=moved["word_ids"],
            mask=moved["mask"],
            char_ids=moved["char_ids"],
        )
        preds.extend(batch_preds)
    return preds


def pad_predictions(preds: Sequence[Sequence[int]], masks: Sequence[Sequence[bool]]) -> List[List[int]]:
    padded = []
    for pred_seq, mask_seq in zip(preds, masks):
        max_len = len(mask_seq)
        padded.append(list(pred_seq) + [0] * (max_len - len(pred_seq)))
    return padded


def majority_vote_predictions(
    all_model_preds: Sequence[Sequence[Sequence[int]]],
    masks: Sequence[Sequence[bool]],
) -> List[List[int]]:
    num_models = len(all_model_preds)
    num_samples = len(masks)
    voted_padded: List[List[int]] = []

    for sample_idx in range(num_samples):
        mask_seq = masks[sample_idx]
        valid_len = int(sum(bool(v) for v in mask_seq))
        voted_seq: List[int] = []
        for token_idx in range(valid_len):
            votes = []
            for model_idx in range(num_models):
                model_seq = all_model_preds[model_idx][sample_idx]
                if token_idx < len(model_seq):
                    votes.append(int(model_seq[token_idx]))
                else:
                    votes.append(0)
            voted_tag = Counter(votes).most_common(1)[0][0]
            voted_seq.append(voted_tag)
        voted_padded.append(voted_seq + [0] * (len(mask_seq) - valid_len))
    return voted_padded


def build_member_metrics(
    member_names: Sequence[str],
    member_preds: Sequence[Sequence[Sequence[int]]],
    gold_ids: Sequence[Sequence[int]],
    masks: Sequence[Sequence[bool]],
    id2label: Dict[int, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, preds in zip(member_names, member_preds):
        padded_preds = pad_predictions(preds, masks)
        pred_tags, gold_tags = ids_to_tag_sequences(
            pred_ids=padded_preds,
            gold_ids=gold_ids,
            mask=masks,
            id2label=id2label,
        )
        metrics = compute_ner_metrics(pred_tags=pred_tags, gold_tags=gold_tags)
        rows.append(
            {
                "model_dir": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    model_dirs = parse_model_dirs(args, project_root=project_root)

    first_dir = model_dirs[0]
    config = load_json(str(first_dir / "config_resolved.json"))
    preprocessor = load_preprocessor(str(first_dir / "preprocessor.json"))
    bundle = build_data_bundle(
        config=config,
        word_vocab=preprocessor["word_vocab"],
        char_vocab=preprocessor["char_vocab"],
        label2id=preprocessor["label2id"],
    )
    data_loader = bundle.dev_loader if args.split == "dev" else bundle.test_loader
    gold_ids, masks = collect_targets(data_loader)

    device = resolve_device(args.device)
    member_names: List[str] = []
    member_preds: List[List[List[int]]] = []
    for model_dir in model_dirs:
        member_names.append(display_path(model_dir, project_root=project_root))
        member_config = load_json(str(model_dir / "config_resolved.json"))
        model = build_model(member_config, bundle).to(device)

        checkpoint = torch.load(str(model_dir / "best_model.pt"), map_location=device)
        if checkpoint.get("ema_model_state_dict") is not None:
            model.load_state_dict(checkpoint["ema_model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        model_predictions = collect_predictions(model=model, data_loader=data_loader, device=device)
        member_preds.append(model_predictions)

    ensemble_pred_ids = majority_vote_predictions(all_model_preds=member_preds, masks=masks)
    pred_tags, gold_tags = ids_to_tag_sequences(
        pred_ids=ensemble_pred_ids,
        gold_ids=gold_ids,
        mask=masks,
        id2label=bundle.id2label,
    )
    ensemble_metrics = compute_ner_metrics(pred_tags=pred_tags, gold_tags=gold_tags)
    ensemble_report = ner_classification_report(pred_tags=pred_tags, gold_tags=gold_tags)

    member_metrics = build_member_metrics(
        member_names=member_names,
        member_preds=member_preds,
        gold_ids=gold_ids,
        masks=masks,
        id2label=bundle.id2label,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "split": args.split,
        "model_dirs": member_names,
        "member_metrics": member_metrics,
        "ensemble_metrics": ensemble_metrics,
    }
    save_json(str(output_dir / "ensemble_metrics.json"), payload)
    (output_dir / "ensemble_report.txt").write_text(ensemble_report, encoding="utf-8")

    lines = [
        f"# Ensemble Result ({args.split})",
        "",
        "| Model | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for row in member_metrics:
        lines.append(
            f"| `{row['model_dir']}` | {row['precision'] * 100:.2f} | "
            f"{row['recall'] * 100:.2f} | {row['f1'] * 100:.2f} |"
        )
    lines.append(
        f"| **Ensemble (majority vote)** | {ensemble_metrics['precision'] * 100:.2f} | "
        f"{ensemble_metrics['recall'] * 100:.2f} | {ensemble_metrics['f1'] * 100:.2f} |"
    )
    lines.append("")
    (output_dir / "ensemble_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved ensemble payload: {output_dir / 'ensemble_metrics.json'}")
    print(f"Saved ensemble report: {output_dir / 'ensemble_report.txt'}")
    print(f"Saved ensemble summary: {output_dir / 'ensemble_summary.md'}")
    print(
        "Ensemble metrics -> "
        f"P={ensemble_metrics['precision'] * 100:.2f} "
        f"R={ensemble_metrics['recall'] * 100:.2f} "
        f"F1={ensemble_metrics['f1'] * 100:.2f}"
    )


if __name__ == "__main__":
    main()
