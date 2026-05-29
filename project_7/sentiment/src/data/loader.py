import re
import random
from pathlib import Path
from typing import Optional


def parse_fasttext_file(filepath: str, max_samples=None) -> tuple[list[str], list[int]]:
    """
    Parse a fastText-formatted file into texts and labels (positive/negative).

    Each line has the following format:
        __label__1 <review text>   -> negative (label 0)
        __label__2 <review text>   -> positive (label 1)

    Arguments:
        filepath: Path to the .ft.txt file.
        max_samples: If set, only load the first N samples in order to have a way to get only a fraction of the set (saving time)

    The programm returns a tuple of lists of the strings (comments) and integers (labels (0/1))
    """
    texts, labels = [], []
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            "Download and extract the Kaggle Amazon Reviews dataset, then place "
            "train.ft.txt and test.ft.txt in sentiment/data/ or pass the correct "
            "paths with --train and --test."
        )

    pattern = re.compile(
        r"^__label__([12])\s+(.+)$"
    )  # defines a patern __label__x with x = 1 or 2

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if (
                max_samples is not None and i >= max_samples
            ):  # if we are over the max samples nb
                break
            line = line.strip()
            match = pattern.match(line)
            if match:
                label_str, text = match.group(1), match.group(2)
                labels.append(int(label_str) - 1)  # __label__1 -> 0, __label__2 -> 1
                texts.append(text)
    return texts, labels


def train_val_split(
    texts: list[str],
    labels: list[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list, list, list]:
    """
    Split texts and labels into train and validation sets. We make sure that the 50/50 balance remains the same.

    Args:
        texts: List of review strings.
        labels: List of integer labels.
        val_ratio: Fraction of data used for validation (here 80/20).
        seed: Random seed for reproducibility.

    The programm returns all sets (text and labels used for training and validation): train_texts, val_texts, train_labels, val_labels
    """
    random.seed(seed)  # make the random everytime the same

    pos_idx = [
        i for i, l in enumerate(labels) if l == 1
    ]  # searches ids where label = 1
    neg_idx = [
        i for i, l in enumerate(labels) if l == 0
    ]  # searches ids where label = 0

    random.shuffle(pos_idx)  # prevents biases
    random.shuffle(neg_idx)

    n_val_pos = int(
        len(pos_idx) * val_ratio
    )  # number of positive values to take into the validation
    n_val_neg = int(len(neg_idx) * val_ratio)

    val_idx = pos_idx[:n_val_pos] + neg_idx[:n_val_neg]
    train_idx = pos_idx[n_val_pos:] + neg_idx[n_val_neg:]

    random.shuffle(val_idx)  # prevent from having only positive then negative
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
    max_train_samples=None,
    max_test_samples=None,
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
    all_texts, all_labels = parse_fasttext_file(
        train_path, max_samples=max_train_samples
    )
    print(f"  -> {len(all_texts):,} examples loaded.")

    train_texts, val_texts, train_labels, val_labels = train_val_split(
        all_texts, all_labels, val_ratio=val_ratio, seed=seed
    )

    print(f"Loading test data from {test_path} ...")
    test_texts, test_labels = parse_fasttext_file(
        test_path, max_samples=max_test_samples
    )
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
