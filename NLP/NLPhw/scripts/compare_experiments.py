import argparse
import json
import os


def load_metrics(model_dir: str):
    path = os.path.join(model_dir, "test_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NER experiment metrics.")
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="outputs/bilstm_crf",
        help="Directory of baseline experiment.",
    )
    parser.add_argument(
        "--improved_dir",
        type=str,
        default="outputs/bilstm_crf_charcnn",
        help="Directory of improved experiment.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/experiment_comparison.md",
        help="Where to save markdown comparison.",
    )
    args = parser.parse_args()

    baseline = load_metrics(args.baseline_dir)
    improved = load_metrics(args.improved_dir)

    lines = [
        "| Model | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    if baseline is not None:
        lines.append(
            f"| BiLSTM-CRF | {fmt(baseline['precision'])} | {fmt(baseline['recall'])} | {fmt(baseline['f1'])} |"
        )
    if improved is not None:
        lines.append(
            f"| BiLSTM-CRF + CharCNN | {fmt(improved['precision'])} | {fmt(improved['recall'])} | {fmt(improved['f1'])} |"
        )

    if baseline is not None and improved is not None:
        delta = (improved["f1"] - baseline["f1"]) * 100
        lines.append("")
        lines.append(f"- F1 improvement: **{delta:.2f}** points")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved comparison to: {args.save_path}")


if __name__ == "__main__":
    main()
