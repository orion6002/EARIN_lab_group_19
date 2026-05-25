import re


def clean_text(text: str) -> str:
    """
    Takes a string removed from the label and returns the cleaned string.
    We use the regex re lib to make is smoother.
    The punctuation which carries sentiment is intentionally kept.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    # Collapse multiple blanks into one only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_texts(texts: list[str]) -> list[str]:
    """Cleans every text given in the list"""
    return [clean_text(t) for t in texts]