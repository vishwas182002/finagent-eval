"""Normalized exact match grader."""
from .normalize import normalize_text, normalize_financial_number


def exact_match_score(prediction: str, gold: str) -> bool:
    """
    Check if prediction matches gold after normalization.
    Tries financial number normalization first, falls back to text normalization.
    """
    # Try financial number normalization for both
    pred_num = normalize_financial_number(prediction)
    gold_num = normalize_financial_number(gold)

    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num

    # Fall back to text normalization
    return normalize_text(prediction) == normalize_text(gold)
