import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed NER training experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bilstm_crf_charcnn.yaml",
        help="Training config file path.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/multiseed",
        help="Directory where per-seed runs and summaries are saved.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44,45,46",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top-K models to recommend for ensemble.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override for train.py (e.g., cuda, cpu).",
    )
    parser.add_argument(
        "--epochs_override",
        type=int,
        default=None,
        help="Optional epochs override for all runs.",
    )
    parser.add_argument(
        "--batch_size_override",
        type=int,
        default=None,
        help="Optional batch size override for all runs.",
    )
    return parser.parse_args()


def parse_seeds(seed_text: str) -> List[int]:
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def load_best_dev_f1(run_dir: Path) -> float:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return float("nan")
    history_payload = load_json(str(history_path))
    history = history_payload.get("history", [])
    if not history:
        return float("nan")
    return float(max(float(row.get("dev_f1", float("-inf"))) for row in history))


def load_test_f1(run_dir: Path) -> float:
    test_metrics_path = run_dir / "test_metrics.json"
    if not test_metrics_path.exists():
        return float("nan")
    metrics = load_json(str(test_metrics_path))
    return float(metrics.get("f1", float("nan")))


def display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def train_one_seed(
    project_root: Path,
    config_path: str,
    run_dir: Path,
    seed: int,
    device: str | None,
    epochs_override: int | None,
    batch_size_override: int | None,
) -> None:
    cmd = [
        sys.executable,
        "train.py",
        "--config",
        config_path,
        "--output_dir",
        str(run_dir),
        "--seed_override",
        str(seed),
    ]
    if device:
        cmd.extend(["--device", device])
    if epochs_override is not None:
        cmd.extend(["--epochs_override", str(epochs_override)])
    if batch_size_override is not None:
        cmd.extend(["--batch_size_override", str(batch_size_override)])

    print(f"[seed={seed}] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(project_root), check=True)


def build_markdown_summary(rows: List[Dict[str, Any]], top_k: int) -> str:
    lines = [
        "| Rank | Seed | Run Dir | Best Dev F1 | Test F1 |",
        "|---:|---:|---|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row['seed']} | `{row['run_dir']}` | "
            f"{row['best_dev_f1'] * 100:.2f} | {row['test_f1'] * 100:.2f} |"
        )

    lines.append("")
    lines.append(f"Recommended top-{top_k} runs for ensemble:")
    for row in rows[:top_k]:
        lines.append(f"- `{row['run_dir']}` (seed={row['seed']}, dev F1={row['best_dev_f1'] * 100:.2f})")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    top_k = max(1, int(args.top_k))

    project_root = PROJECT_ROOT
    output_root = (project_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        run_dir = output_root / f"seed_{seed}"
        train_one_seed(
            project_root=project_root,
            config_path=args.config,
            run_dir=run_dir,
            seed=seed,
            device=args.device,
            epochs_override=args.epochs_override,
            batch_size_override=args.batch_size_override,
        )

        best_dev_f1 = load_best_dev_f1(run_dir)
        test_f1 = load_test_f1(run_dir)
        run_rows.append(
            {
                "seed": seed,
                "run_dir": display_path(run_dir, project_root=project_root),
                "best_dev_f1": best_dev_f1,
                "test_f1": test_f1,
            }
        )

    run_rows.sort(key=lambda row: row["best_dev_f1"], reverse=True)
    summary_payload = {
        "config": args.config,
        "seeds": seeds,
        "top_k": top_k,
        "runs": run_rows,
        "recommended_model_dirs": [row["run_dir"] for row in run_rows[:top_k]],
    }
    save_json(str(output_root / "multiseed_summary.json"), summary_payload)

    summary_md = build_markdown_summary(rows=run_rows, top_k=top_k)
    (output_root / "multiseed_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Saved JSON summary: {output_root / 'multiseed_summary.json'}")
    print(f"Saved markdown summary: {output_root / 'multiseed_summary.md'}")
    print("Recommended model dirs for ensemble:")
    for model_dir in summary_payload["recommended_model_dirs"]:
        print(f"- {model_dir}")


if __name__ == "__main__":
    main()
