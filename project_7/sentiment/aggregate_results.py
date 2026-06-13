"""
Collect experiment metrics.json files into a CSV table.

Usage:
    python3 sentiment/aggregate_results.py --root sentiment/out --output sentiment/out/results.csv
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics.")
    parser.add_argument("--root", default="sentiment/out", help="Directory containing experiment outputs.")
    parser.add_argument("--output", default="sentiment/out/results.csv", help="CSV file to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []

    for metrics_path in sorted(root.rglob("metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        hparams = data.get("hyperparameters", {})
        rows.append(
            {
                "run_dir": str(metrics_path.parent),
                "model": data.get("model"),
                "seed": data.get("seed"),
                "max_train": data.get("max_train"),
                "max_test": data.get("max_test"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "test_accuracy": data.get("test", {}).get("accuracy"),
                "test_macro_f1": data.get("test", {}).get("macro_f1"),
                "test_precision": data.get("test", {}).get("macro_precision"),
                "test_recall": data.get("test", {}).get("macro_recall"),
                "val_accuracy": data.get("validation", {}).get("accuracy"),
                "val_macro_f1": data.get("validation", {}).get("macro_f1"),
                "lr": hparams.get("lr"),
                "epochs": hparams.get("epochs"),
                "batch_size": hparams.get("batch_size"),
                "max_len": hparams.get("max_len"),
                "embed_dim": hparams.get("embed_dim"),
                "hidden_dim": hparams.get("hidden_dim"),
                "num_layers": hparams.get("num_layers"),
                "bidirectional": hparams.get("bidirectional"),
                "dropout": hparams.get("dropout"),
                "weight_decay": hparams.get("weight_decay"),
                "freeze_embedding": hparams.get("freeze_embedding"),
                "freeze_classifier": hparams.get("freeze_classifier"),
                "num_layers_to_freeze": hparams.get("num_layers_to_freeze"),
                "warmup_ratio": hparams.get("warmup_ratio"),
                "max_features": hparams.get("max_features"),
                "C": hparams.get("C"),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else ["run_dir"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
