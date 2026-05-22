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
import os
import sys

# Make src importable from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data.loader import load_dataset
from data.preprocessor import clean_texts
from evaluation.metrics import evaluate, plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sentiment classifier.")

    # Required
    parser.add_argument("--model", required=True, choices=["logistic", "lstm", "roberta"],
                        help="Which model to train.")
    parser.add_argument("--train", required=True, help="Path to train.ft.txt")
    parser.add_argument("--test", required=True, help="Path to test.ft.txt")

    # Data
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./out")

    # LSTM / RoBERTa shared
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    # LSTM specific
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)

    # Logistic Regression specific
    parser.add_argument("--max_features", type=int, default=100_000)
    parser.add_argument("--C", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

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
        _run_logistic(args, train_texts, train_labels, val_texts, val_labels,
                      test_texts, test_labels)

    elif args.model == "lstm":
        _run_lstm(args, train_texts, train_labels, val_texts, val_labels,
                  test_texts, test_labels)

    elif args.model == "roberta":
        _run_roberta(args, train_texts, train_labels, val_texts, val_labels,
                     test_texts, test_labels)


# ------------------------------------------------------------------ #
#  Model-specific runners
# ------------------------------------------------------------------ #

def _run_logistic(args, train_texts, train_labels, val_texts, val_labels,
                  test_texts, test_labels) -> None:
    from models import logistic

    model = logistic.train(
        train_texts, train_labels,
        max_features=args.max_features,
        C=args.C,
    )

    print("\n--- Validation ---")
    val_preds = logistic.predict(model, val_texts)
    evaluate(val_labels, val_preds, model_name="Logistic Regression (val)")

    print("\n--- Test ---")
    test_preds = logistic.predict(model, test_texts)
    evaluate(test_labels, test_preds, model_name="Logistic Regression (test)")

    plot_confusion_matrix(
        test_labels, test_preds,
        model_name="Logistic Regression",
        save_path=os.path.join(args.output_dir, "confusion_logistic.png"),
    )
    logistic.save(model, os.path.join(args.output_dir, "logistic_pipeline.pkl"))


def _run_lstm(args, train_texts, train_labels, val_texts, val_labels,
              test_texts, test_labels) -> None:
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
        batch_size=args.batch_size or 64,
        num_epochs=args.epochs or 5,
        lr=args.lr or 1e-3,
        device=args.device,
    )

    print("\n--- Validation ---")
    val_preds = lstm_predict(model, val_texts, vocab, device=args.device)
    evaluate(val_labels, val_preds, model_name="BiLSTM (val)")

    print("\n--- Test ---")
    test_preds = lstm_predict(model, test_texts, vocab, device=args.device)
    evaluate(test_labels, test_preds, model_name="BiLSTM (test)")

    plot_confusion_matrix(
        test_labels, test_preds,
        model_name="BiLSTM",
        save_path=os.path.join(args.output_dir, "confusion_lstm.png"),
    )
    lstm_save(model, vocab, os.path.join(args.output_dir, "lstm_model"))


def _run_roberta(args, train_texts, train_labels, val_texts, val_labels,
                 test_texts, test_labels) -> None:
    from models import roberta

    model, tokenizer = roberta.train(
        train_texts, train_labels,
        val_texts, val_labels,
        max_len=args.max_len or 512,
        batch_size=args.batch_size or 32,
        num_epochs=args.epochs or 3,
        lr=args.lr or 2e-5,
        device=args.device,
    )

    print("\n--- Validation ---")
    val_preds = roberta.predict(model, tokenizer, val_texts, device=args.device)
    evaluate(val_labels, val_preds, model_name="RoBERTa (val)")

    print("\n--- Test ---")
    test_preds = roberta.predict(model, tokenizer, test_texts, device=args.device)
    evaluate(test_labels, test_preds, model_name="RoBERTa (test)")

    plot_confusion_matrix(
        test_labels, test_preds,
        model_name="RoBERTa",
        save_path=os.path.join(args.output_dir, "confusion_roberta.png"),
    )
    roberta.save(model, tokenizer, os.path.join(args.output_dir, "roberta_model"))


if __name__ == "__main__":
    main()
