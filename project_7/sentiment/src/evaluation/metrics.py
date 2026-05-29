"""
evaluation/metrics.py
Evaluation utilities used by all three models.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "out" / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate(y_true: list[int], y_pred: list[int], model_name: str = "") -> dict:
    """
    Compute accuracy, macro F1, precision, and recall.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Optional label for display.

    Returns:
        Dictionary of metric name -> value.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    results = {
        "accuracy": acc,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }

    header = f"=== {model_name} ===" if model_name else "=== Results ==="
    print(f"\n{header}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

    return results


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Title prefix.
        save_path: If provided, save the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def compare_models(results: dict[str, dict]) -> None:
    """
    Print a comparison table of all evaluated models.

    Args:
        results: Dict mapping model_name -> metrics dict from evaluate().
    """
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("=" * 60)
    for name, metrics in results.items():
        print(
            f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}"
        )
    print("=" * 60)
