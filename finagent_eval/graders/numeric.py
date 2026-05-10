"""Numeric tolerance grader (2% default threshold)."""
import re

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


def _is_year_like_answer(text: str) -> bool:
    """Return True for standalone year answers such as 2025 or FY2025."""
    if not text:
        return False

    raw = str(text).strip().lower()

    # Financial values with formatting should not be treated as years.
    if any(symbol in raw for symbol in ["$", "€", "£", "%", ",", "."]):
        return False

    return bool(re.fullmatch(r"fy\s*(?:19|20)\d{2}|(?:19|20)\d{2}", raw))


def numeric_match_is_correct(prediction: str, gold: str, tolerance: float = 0.02) -> bool:
    """
    Return True when numeric tolerance should count as correctness.

    This deliberately excludes year-like answers. For example, 2024 and 2025
    are within 2% numerically, but that must never make a wrong year correct.
    """
    if numeric_match_score(prediction, gold, tolerance=tolerance) is not True:
        return False

    if _is_year_like_answer(prediction) or _is_year_like_answer(gold):
        return False

    return True
