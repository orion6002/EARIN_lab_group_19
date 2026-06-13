"""
evaluation/metrics.py
Evaluation utilities used by all three models.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np


def confusion_matrix_binary(y_true: list[int], y_pred: list[int]) -> np.ndarray:
    """Return a 2x2 confusion matrix with rows=true labels and columns=predictions."""
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _class_report_row(cm: np.ndarray, cls: int) -> dict:
    tp = cm[cls, cls]
    fp = cm[1 - cls, cls]
    fn = cm[cls, 1 - cls]
    support = cm[cls].sum()
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": int(support),
    }


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
    cm = confusion_matrix_binary(y_true, y_pred)
    total = cm.sum()
    acc = _safe_div(cm.trace(), total)

    neg = _class_report_row(cm, 0)
    pos = _class_report_row(cm, 1)
    f1 = (neg["f1"] + pos["f1"]) / 2
    precision = (neg["precision"] + pos["precision"]) / 2
    recall = (neg["recall"] + pos["recall"]) / 2

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
    print("\n              precision    recall  f1-score   support\n")
    print(
        f"    Negative       {neg['precision']:.2f}      {neg['recall']:.2f}"
        f"      {neg['f1']:.2f}      {neg['support']:>6}"
    )
    print(
        f"    Positive       {pos['precision']:.2f}      {pos['recall']:.2f}"
        f"      {pos['f1']:.2f}      {pos['support']:>6}"
    )

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
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(__file__).resolve().parents[2] / "out" / ".matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix_binary(y_true, y_pred)
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
