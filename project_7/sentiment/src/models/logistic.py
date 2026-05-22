"""
models/logistic.py
Logistic Regression classifier with TF-IDF features.

Pipeline:
  1. TF-IDF vectorization (unigrams + bigrams, top 100k features)
  2. Logistic Regression with L2 regularization
"""

import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline(
    max_features: int = 100_000,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline:  TF-IDF  ->  Logistic Regression.

    Args:
        max_features: Maximum number of TF-IDF features to keep.
        ngram_range: Range of n-gram sizes. (1, 2) = unigrams + bigrams.
        C: Inverse of regularization strength (larger = less regularization).
        max_iter: Maximum iterations for the solver.
        n_jobs: Number of CPU cores (-1 = all available).

    Returns:
        Untrained sklearn Pipeline.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,        # apply log(1 + tf) scaling
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        n_jobs=n_jobs,
    )

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train(
    train_texts: list[str],
    train_labels: list[int],
    **pipeline_kwargs,
) -> Pipeline:
    """
    Fit the TF-IDF + Logistic Regression pipeline.

    Args:
        train_texts: List of preprocessed review strings.
        train_labels: List of integer labels (0 / 1).
        **pipeline_kwargs: Forwarded to build_pipeline().

    Returns:
        Fitted Pipeline.
    """
    print("Building TF-IDF + Logistic Regression pipeline ...")
    pipeline = build_pipeline(**pipeline_kwargs)

    print(f"Fitting on {len(train_texts):,} examples ...")
    pipeline.fit(train_texts, train_labels)
    print("Training complete.")

    return pipeline


def predict(pipeline: Pipeline, texts: list[str]) -> list[int]:
    """
    Run inference with a fitted pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        texts: List of preprocessed review strings.

    Returns:
        List of predicted integer labels.
    """
    return pipeline.predict(texts).tolist()


def save(pipeline: Pipeline, path: str) -> None:
    """Persist the fitted pipeline to disk with pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved to {path}")


def load(path: str) -> Pipeline:
    """Load a fitted pipeline from disk."""
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"Pipeline loaded from {path}")
    return pipeline
