"""
data/preprocessor.py
Base text preprocessing shared by all three models.
Model-specific preprocessing lives in each model's own module.
"""

import re


def clean_text(text: str) -> str:
    """
    Apply base cleaning to a single review string.

    Steps:
      1. Remove HTML tags.
      2. Lowercase.
      3. Collapse multiple whitespace into one.

    Punctuation that carries sentiment (!, ?) is intentionally kept.

    Args:
        text: Raw review string (already stripped of __label__ prefix).

    Returns:
        Cleaned string.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_texts(texts: list[str]) -> list[str]:
    """Apply clean_text to a list of strings."""
    return [clean_text(t) for t in texts]
