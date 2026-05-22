"""
data/loader.py
Loads the Amazon Reviews dataset (bittlingmayer fastText format)
and creates train / validation / test splits.
"""

import re
import random
from pathlib import Path
from typing import Optional


def parse_fasttext_file(filepath: str, max_samples: Optional[int] = None) -> tuple[list[str], list[int]]:
    """
    Parse a fastText-formatted file into texts and labels.

    Each line has the format:
        __label__1 <review text>   -> negative (label 0)
        __label__2 <review text>   -> positive (label 1)

    Args:
        filepath: Path to the .ft.txt file.
        max_samples: If set, only load the first N samples (useful for dev).

    Returns:
        texts: List of raw review strings.
        labels: List of integer labels (0 = negative, 1 = positive).
    """
    texts, labels = [], []
    pattern = re.compile(r"^__label__([12])\s+(.+)$")

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            match = pattern.match(line)
            if match:
                label_str, text = match.group(1), match.group(2)
                labels.append(int(label_str) - 1)   # __label__1 -> 0, __label__2 -> 1
                texts.append(text)

    return texts, labels


def train_val_split(
    texts: list[str],
    labels: list[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list, list, list]:
    """
    Split texts and labels into train and validation sets.
    Stratified by label to preserve the 50/50 balance.

    Args:
        texts: List of review strings.
        labels: List of integer labels.
        val_ratio: Fraction of data used for validation.
        seed: Random seed for reproducibility.

    Returns:
        train_texts, val_texts, train_labels, val_labels
    """
    random.seed(seed)

    # Separate indices by class
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]

    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    n_val_pos = int(len(pos_idx) * val_ratio)
    n_val_neg = int(len(neg_idx) * val_ratio)

    val_idx = pos_idx[:n_val_pos] + neg_idx[:n_val_neg]
    train_idx = pos_idx[n_val_pos:] + neg_idx[n_val_neg:]

    random.shuffle(val_idx)
    random.shuffle(train_idx)

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    return train_texts, val_texts, train_labels, val_labels


def load_dataset(
    train_path: str,
    test_path: str,
    val_ratio: float = 0.2,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """
    Full pipeline: load train + test files, create val split.

    Args:
        train_path: Path to train.ft.txt (or .bz2 extracted).
        test_path: Path to test.ft.txt.
        val_ratio: Fraction of training data for validation.
        max_train_samples: Cap training data (for quick experiments).
        max_test_samples: Cap test data.
        seed: Random seed.

    Returns:
        Dictionary with keys: train_texts, train_labels,
                               val_texts,   val_labels,
                               test_texts,  test_labels.
    """
    print(f"Loading training data from {train_path} ...")
    all_texts, all_labels = parse_fasttext_file(train_path, max_samples=max_train_samples)
    print(f"  -> {len(all_texts):,} examples loaded.")

    train_texts, val_texts, train_labels, val_labels = train_val_split(
        all_texts, all_labels, val_ratio=val_ratio, seed=seed
    )

    print(f"Loading test data from {test_path} ...")
    test_texts, test_labels = parse_fasttext_file(test_path, max_samples=max_test_samples)
    print(f"  -> {len(test_texts):,} examples loaded.")

    print(
        f"\nSplit summary:\n"
        f"  Train : {len(train_texts):>10,}\n"
        f"  Val   : {len(val_texts):>10,}\n"
        f"  Test  : {len(test_texts):>10,}\n"
    )

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }
