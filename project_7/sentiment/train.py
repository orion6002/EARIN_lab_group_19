"""
train.py
Entry point to train one of the three sentiment classifiers.

Usage:
    python train.py --model logistic --train data/train.ft.txt --test data/test.ft.txt
    python train.py --model lstm     --train data/train.ft.txt --test data/test.ft.txt
    python train.py --model roberta  --train data/train.ft.txt --test data/test.ft.txt

Optional flags (all models):
    --max_train N       Use only N training examples (for quick tests)
    --max_test  N       Use only N test examples
    --val_ratio 0.2     Fraction of training data for validation
    --seed 42           Random seed
    --output_dir ./out  Where to save the trained model

LSTM extra flags:
    --max_len 256       Maximum sequence length
    --embed_dim 100     Embedding dimension
    --hidden_dim 128    LSTM hidden units per direction
    --batch_size 64     Batch size
    --epochs 5          Training epochs
    --lr 1e-3           Learning rate
    --device cpu        "cpu" or "cuda"

RoBERTa extra flags:
    --max_len 512
    --batch_size 32
    --epochs 3
    --lr 2e-5
    --device cpu
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

# Make src importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.loader import load_dataset
from data.preprocessor import clean_texts
from evaluation.metrics import evaluate, plot_confusion_matrix


def set_seed(seed: int) -> None:
    """Set random seeds used by the training pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_run_summary(
    args: argparse.Namespace,
    val_metrics: dict,
    test_metrics: dict,
    elapsed_seconds: float,
) -> None:
    """Save metrics and hyperparameters for reproducible experiments."""
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "seed": args.seed,
        "train_path": args.train,
        "test_path": args.test,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "val_ratio": args.val_ratio,
        "no_plots": args.no_plots,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "hyperparameters": {
            "max_features": args.max_features,
            "C": args.C,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "device": args.device,
            "num_workers": args.num_workers,
            "embed_dim": args.embed_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "bidirectional": args.bidirectional,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay
            if args.weight_decay is not None
            else (0.01 if args.model == "roberta" else 0.0),
            "freeze_embedding": args.freeze_embedding,
            "freeze_classifier": args.freeze_classifier,
            "num_layers_to_freeze": args.num_layers_to_freeze,
            "warmup_ratio": args.warmup_ratio,
            "transformer_model": args.transformer_model,
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    output_path = os.path.join(args.output_dir, "metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Run summary saved to {output_path}")


def save_training_history(model, output_dir: str) -> None:
    """Save per-epoch training history when the model exposes it."""
    history = getattr(model, "training_history", None)
    if not history:
        return
    output_path = os.path.join(output_dir, "training_history.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sentiment classifier.")

    # Required
    parser.add_argument("--model", required=True, choices=["logistic", "lstm", "roberta"], help="Which model to train.")
    parser.add_argument("--train", required=True, help="Path to train.ft.txt")
    parser.add_argument("--test", required=True, help="Path to test.ft.txt")

    # Data
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./out")
    parser.add_argument("--no_plots", action="store_true", help="Skip confusion matrix generation.")

    # LSTM / RoBERTa shared
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)

    # LSTM specific
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--unidirectional", action="store_false", dest="bidirectional")
    parser.add_argument("--freeze_embedding", action="store_true")
    parser.add_argument("--freeze_classifier", action="store_true")

    # Logistic Regression specific
    parser.add_argument("--max_features", type=int, default=100_000)
    parser.add_argument("--C", type=float, default=1.0)

    # Roberta specific
    parser.add_argument("--transformer_model", type=str, default="roberta-base")
    parser.add_argument("--num_layers_to_freeze", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    data = load_dataset(
        train_path=args.train,
        test_path=args.test,
        val_ratio=args.val_ratio,
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
        seed=args.seed,
    )

    train_texts = clean_texts(data["train_texts"])
    val_texts = clean_texts(data["val_texts"])
    test_texts = clean_texts(data["test_texts"])
    train_labels = data["train_labels"]
    val_labels = data["val_labels"]
    test_labels = data["test_labels"]

    # ------------------------------------------------------------------ #
    # 2. Train the selected model
    # ------------------------------------------------------------------ #
    if args.model == "logistic":
        val_metrics, test_metrics = _run_logistic(
            args, train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels
        )

    elif args.model == "lstm":
        val_metrics, test_metrics = _run_lstm(
            args, train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels
        )

    elif args.model == "roberta":
        val_metrics, test_metrics = _run_roberta(
            args, train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels
        )

    elapsed_seconds = time.perf_counter() - start_time
    save_run_summary(args, val_metrics, test_metrics, elapsed_seconds)
    print(f"Total runtime: {elapsed_seconds:.1f} seconds")


# ------------------------------------------------------------------ #
#  Model-specific runners
# ------------------------------------------------------------------ #

def _run_logistic(args, train_texts, train_labels, val_texts, val_labels,
                  test_texts, test_labels) -> tuple[dict, dict]:
    from models import logistic

    model = logistic.train(
        train_texts, train_labels,
        max_features=args.max_features,
        C=args.C,
        random_state=args.seed,
    )

    print("\n--- Validation ---")
    val_preds = logistic.predict(model, val_texts)
    val_metrics = evaluate(val_labels, val_preds, model_name="Logistic Regression (val)")

    print("\n--- Test ---")
    test_preds = logistic.predict(model, test_texts)
    test_metrics = evaluate(test_labels, test_preds, model_name="Logistic Regression (test)")

    if not args.no_plots:
        plot_confusion_matrix(
            test_labels, test_preds,
            model_name="Logistic Regression",
            save_path=os.path.join(args.output_dir, "confusion_logistic.png"),
        )
    logistic.save(model, os.path.join(args.output_dir, "logistic_pipeline.pkl"))
    return val_metrics, test_metrics


def _run_lstm(args, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels) -> tuple[dict, dict]:
    from models.lstm import Vocabulary, train as lstm_train, predict as lstm_predict, save as lstm_save

    vocab = Vocabulary()
    vocab.build(train_texts, max_vocab=50_000)

    model = lstm_train(
        train_texts, train_labels,
        val_texts, val_labels,
        vocab=vocab,
        max_len=args.max_len or 256,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_embedding=args.freeze_embedding,
        freeze_classifier=args.freeze_classifier,
        batch_size=args.batch_size or 64,
        num_epochs=args.epochs or 5,
        lr=args.lr or 1e-3,
        weight_decay=args.weight_decay if args.weight_decay is not None else 0.0,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
    )
    save_training_history(model, args.output_dir)

    print("\n--- Validation ---")
    val_preds = lstm_predict(model, val_texts, vocab, max_len=args.max_len or 256, device=args.device)
    model_name = "BiLSTM" if args.bidirectional else "LSTM"
    val_metrics = evaluate(val_labels, val_preds, model_name=f"{model_name} (val)")

    print("\n--- Test ---")
    test_preds = lstm_predict(model, test_texts, vocab, max_len=args.max_len or 256, device=args.device)
    test_metrics = evaluate(test_labels, test_preds, model_name=f"{model_name} (test)")

    if not args.no_plots:
        plot_confusion_matrix(
            test_labels, test_preds,
            model_name=model_name,
            save_path=os.path.join(args.output_dir, "confusion_lstm.png"),
        )
    lstm_save(model, vocab, os.path.join(args.output_dir, "lstm_model"))
    return val_metrics, test_metrics


def _run_roberta(args, train_texts, train_labels, val_texts, val_labels,
                 test_texts, test_labels) -> tuple[dict, dict]:
    from models import roberta

    model, tokenizer = roberta.train(
        train_texts, train_labels,
        val_texts, val_labels,
        model_name=args.transformer_model,
        max_len=args.max_len or 512,
        batch_size=args.batch_size or 32,
        num_epochs=args.epochs or 3,
        lr=args.lr or 2e-5,
        weight_decay=args.weight_decay if args.weight_decay is not None else 0.01,
        warmup_ratio=args.warmup_ratio,
        device=args.device,
        num_layers_to_freeze=args.num_layers_to_freeze,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    save_training_history(model, args.output_dir)

    print("\n--- Validation ---")
    val_preds = roberta.predict(model, tokenizer, val_texts, max_len=args.max_len or 512, device=args.device)
    val_metrics = evaluate(val_labels, val_preds, model_name="RoBERTa (val)")

    print("\n--- Test ---")
    test_preds = roberta.predict(model, tokenizer, test_texts, max_len=args.max_len or 512, device=args.device)
    test_metrics = evaluate(test_labels, test_preds, model_name="RoBERTa (test)")

    if not args.no_plots:
        plot_confusion_matrix(
            test_labels, test_preds,
            model_name="RoBERTa",
            save_path=os.path.join(args.output_dir, "confusion_roberta.png"),
        )
    roberta.save(model, tokenizer, os.path.join(args.output_dir, "roberta_model"))
    return val_metrics, test_metrics


if __name__ == "__main__":
    main()
