"""
TF-IDF + Logistic Regression baseline for binary sentiment classification.
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train(
    train_texts: list[str],
    train_labels: list[int],
    max_features: int = 100_000,
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a Logistic Regression classifier on TF-IDF features.

    Args:
        train_texts: Cleaned review texts.
        train_labels: Binary labels, where 0 is negative and 1 is positive.
        max_features: Maximum TF-IDF vocabulary size.
        C: Inverse regularization strength for Logistic Regression.
        random_state: Seed used by the Logistic Regression solver.

    Returns:
        A fitted scikit-learn Pipeline.
    """
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=1_000,
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )

    print("Training TF-IDF + Logistic Regression model ...")
    model.fit(train_texts, train_labels)
    return model


def predict(model: Pipeline, texts: list[str]) -> list[int]:
    """
    Predict binary sentiment labels for the given texts.
    """
    return model.predict(texts).tolist()


def save(model: Pipeline, path: str) -> None:
    """
    Save the fitted pipeline to disk.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def load(path: str) -> Pipeline:
    """
    Load a saved Logistic Regression pipeline from disk.
    """
    return joblib.load(path)
