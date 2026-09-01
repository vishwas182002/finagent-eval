"""Numeric tolerance grader (2% default threshold)."""
from __future__ import annotations

import re

from .normalize import financial_numbers_equivalent, parse_financial_number

DEFAULT_TOLERANCE = 0.02


def numeric_match_score(prediction: str, gold: str, tolerance: float = DEFAULT_TOLERANCE) -> bool | None:
    """
    Check if prediction is within relative ``tolerance`` of the gold answer.
    Returns True/False if both are numbers, None if either is not a number.

    Percent and non-percent answers never match each other. An explicit unit
    word on exactly one side (``"$85,269 million"`` vs ``"85,269"``) is treated
    as the table's implicit unit, see :func:`financial_numbers_equivalent`.
    """
    pred_num = parse_financial_number(prediction)
    gold_num = parse_financial_number(gold)

    if pred_num is None or gold_num is None:
        return None

    return financial_numbers_equivalent(pred_num, gold_num, tolerance=tolerance)


def _is_year_like_answer(text: str) -> bool:
    """Return True for standalone year answers such as 2025 or FY2025."""
    if not text:
        return False

    raw = str(text).strip().lower()

    # Financial values with formatting should not be treated as years.
    if any(symbol in raw for symbol in ["$", "€", "£", "%", ",", "."]):
        return False

    return bool(re.fullmatch(r"fy\s*(?:19|20)\d{2}|(?:19|20)\d{2}", raw))


def numeric_match_is_correct(prediction: str, gold: str, tolerance: float = DEFAULT_TOLERANCE) -> bool:
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
