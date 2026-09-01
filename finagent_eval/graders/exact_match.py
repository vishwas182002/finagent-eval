"""Normalized exact match grader."""
from __future__ import annotations

from .normalize import financial_numbers_equivalent, normalize_answer_text, parse_financial_number


def exact_match_score(prediction: str, gold: str) -> bool:
    """
    Check if prediction matches gold after normalization.

    If both sides parse as financial numbers they are compared with
    :func:`financial_numbers_equivalent` (currency, commas, accounting
    negatives, percent status and unit words are all handled, and an implicit
    table unit on one side is tolerated). Otherwise the answers are compared as
    normalized text (case, whitespace, unicode punctuation, answer prefixes and
    a leading article are ignored).
    """
    pred_num = parse_financial_number(prediction)
    gold_num = parse_financial_number(gold)

    if pred_num is not None and gold_num is not None:
        return financial_numbers_equivalent(pred_num, gold_num)

    pred_text = normalize_answer_text(prediction)
    gold_text = normalize_answer_text(gold)
    if not gold_text:
        return not pred_text
    return pred_text == gold_text
