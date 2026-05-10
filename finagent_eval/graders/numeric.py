"""Numeric tolerance grader (2% default threshold)."""
from .normalize import normalize_financial_number


def numeric_match_score(prediction: str, gold: str, tolerance: float = 0.02) -> bool | None:
    """
    Check if prediction is within tolerance of gold answer.
    Returns True/False if both are numbers, None if either is not a number.
    """
    pred_str = normalize_financial_number(prediction)
    gold_str = normalize_financial_number(gold)

    if pred_str is None or gold_str is None:
        return None

    # Strip percentage sign for comparison
    pred_val = float(pred_str.rstrip('%'))
    gold_val = float(gold_str.rstrip('%'))

    # Both must be percentages or both non-percentages
    pred_is_pct = pred_str.endswith('%')
    gold_is_pct = gold_str.endswith('%')
    if pred_is_pct != gold_is_pct:
        return False

    if gold_val == 0:
        return pred_val == 0

    return abs(pred_val - gold_val) / abs(gold_val) <= tolerance
