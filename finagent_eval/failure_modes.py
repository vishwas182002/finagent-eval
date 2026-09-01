"""Semi-automatic failure mode classifier for incorrect answers.

The classifier is only meaningful for predictions that are *not* project-correct
(see :func:`finagent_eval.graders.grade`). Labels, in the order they are tested:

``empty_output``
    Model returned nothing.
``abstention``
    Model explicitly refused or said the answer is unavailable.
``format_mismatch``
    Numerically identical to gold but the grader in use rejected it (only
    reachable with a stricter grader than the project default; kept for API
    completeness).
``sign_mismatch``
    Same magnitude, opposite sign - e.g. ``39,371`` for an accounting-negative
    gold ``(39,371)``.
``unit_mismatch``
    Percent vs. non-percent, or both sides carry explicit but conflicting units.
``wrong_year``
    The prediction references a year that the question does not.
``hallucinated_number``
    Predicted number does not occur in the supplied ``source_text``. Only
    reachable when OCR/source text is provided.
``numeric_mismatch``
    Both numeric, values differ beyond tolerance.
``near_miss_numeric``
    Both numeric and within tolerance, but tolerance does not apply to this
    category (extractive/layout answers must match the printed digits) - a
    misread digit such as ``37,586`` for ``37,855``.
``type_mismatch``
    One side is numeric, the other is text (``"Yes"`` for a gold ``4,396``).
``partial_text_match``
    Text answers where one normalized string contains the other
    (``"Daily VaR"`` vs ``"Daily VaR (Value at Risk)"``). Usually an
    over- or under-specified but semantically right answer.
``text_mismatch``
    Both text, unrelated (``"Net product sales"`` vs ``"Net service sales"``).
``manual_diagnosis_required``
    Fallback; should be rare.
"""
from __future__ import annotations

import re

from .graders.normalize import (
    FinancialNumber,
    financial_numbers_equivalent,
    normalize_answer_text,
    parse_financial_number,
)
from .graders.numeric import DEFAULT_TOLERANCE, _is_year_like_answer, numeric_match_score

FAILURE_LABELS: tuple[str, ...] = (
    "empty_output",
    "abstention",
    "format_mismatch",
    "sign_mismatch",
    "unit_mismatch",
    "wrong_year",
    "hallucinated_number",
    "numeric_mismatch",
    "near_miss_numeric",
    "type_mismatch",
    "partial_text_match",
    "text_mismatch",
    "manual_diagnosis_required",
)

_ABSTENTION_PATTERNS = [
    r"^n/?a$",
    r"^none$",
    r"^null$",
    r"^unknown$",
    r"^i (?:don'?t|do not) know",
    r"^(?:i am |i'm )?unable to (?:determine|answer|find)",
    r"^not (?:available|found|provided|specified|applicable|shown|stated|given|visible)",
    r"^(?:i )?cannot (?:determine|answer|find|see)",
    r"^(?:i )?can'?t (?:determine|answer|find|see)",
    r"^no answer",
    r"^(?:the )?(?:information|data|value|answer) (?:is )?not (?:available|provided|shown|present)",
    r"^(?:the )?(?:document|image|table|chart) does not",
    r"^there is no ",
    r"^insufficient (?:information|data)",
]
_ABSTENTION_RES = [re.compile(p) for p in _ABSTENTION_PATTERNS]


def _extract_years(text: str) -> set[str]:
    """Extract 4-digit years (2000-2099), including FY-prefixed and ranges."""
    return set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", text or ""))


def _numbers_in(text: str) -> set[str]:
    """All canonical numbers that appear in a block of text."""
    out: set[str] = set()
    for match in re.findall(r"\(?[+-]?\$?\d[\d,]*\.?\d*\)?%?", text):
        parsed = parse_financial_number(match)
        if parsed is not None:
            out.add(parsed.canonical())
            # Also record the unscaled mantissa so table units are tolerated.
            out.add(parse_financial_number(str(parsed.value)).canonical())
    return out


def is_abstention(prediction: str) -> bool:
    """True if the prediction is an explicit refusal / not-available answer."""
    pred = normalize_answer_text(prediction)
    return any(p.search(pred) for p in _ABSTENTION_RES)


def classify_failure(
    prediction: str,
    gold_answer: str,
    category: str,
    question: str = "",
    source_text: str = "",
    tolerance: float = DEFAULT_TOLERANCE,
) -> str:
    """Classify why a prediction is wrong. Returns one of :data:`FAILURE_LABELS`."""
    del category  # reserved for category-specific rules; keeps the public signature stable
    pred_stripped = prediction.strip() if prediction else ""

    # 1. Empty output
    if not pred_stripped:
        return "empty_output"

    # 2. Abstention
    if is_abstention(pred_stripped):
        return "abstention"

    pred_num = parse_financial_number(prediction)
    gold_num = parse_financial_number(gold_answer)
    both_numeric = pred_num is not None and gold_num is not None

    if both_numeric:
        # 3a. Format mismatch - identical under the project rules. Only reachable
        #     if the caller used a stricter grader than the project default.
        if financial_numbers_equivalent(pred_num, gold_num):
            return "format_mismatch"

        # 3b. Percent vs non-percent, or conflicting explicit units.
        if pred_num.is_percent != gold_num.is_percent:
            return "unit_mismatch"
        if (
            pred_num.has_unit
            and gold_num.has_unit
            and pred_num.scale != gold_num.scale
            and pred_num.value == gold_num.value
        ):
            return "unit_mismatch"

        # 3c. Same magnitude, opposite sign (accounting negatives).
        if pred_num.value != 0:
            flipped = FinancialNumber(
                value=-pred_num.value,
                scale=pred_num.scale,
                has_unit=pred_num.has_unit,
                is_percent=pred_num.is_percent,
            )
            if financial_numbers_equivalent(flipped, gold_num):
                return "sign_mismatch"

    # 4. Wrong year - checked before numeric mismatch so "2024" vs question "2025"
    #    is a temporal error, not a numeric one.
    q_years = _extract_years(question)
    pred_years = _extract_years(pred_stripped)
    gold_years = _extract_years(gold_answer or "")
    if pred_years and q_years and not pred_years.intersection(q_years):
        return "wrong_year"
    # Gold is itself a year ("Which year had the highest ...?") and the model named another.
    if _is_year_like_answer(gold_answer) and pred_years and pred_years != gold_years:
        return "wrong_year"

    # 5. Remaining numeric checks
    if both_numeric:
        # 5a. Hallucinated number - predicted number not in source document
        if source_text:
            source_numbers = _numbers_in(source_text)
            mantissa = parse_financial_number(str(pred_num.value)).canonical()
            if pred_num.canonical() not in source_numbers and mantissa not in source_numbers:
                return "hallucinated_number"

        # 5b. Numeric mismatch - outside tolerance
        if numeric_match_score(prediction, gold_answer, tolerance=tolerance) is False:
            return "numeric_mismatch"
        # 5c. Within tolerance but not accepted: an extractive/layout answer
        #     whose digits differ (37,586 for 37,855), or a year-like value.
        return "near_miss_numeric"

    # 6. One side numeric, the other not
    if (pred_num is None) != (gold_num is None):
        return "type_mismatch"

    # 7. Text vs text
    if pred_num is None and gold_num is None:
        pred_text = normalize_answer_text(prediction)
        gold_text = normalize_answer_text(gold_answer)
        if pred_text and gold_text and (pred_text in gold_text or gold_text in pred_text):
            return "partial_text_match"
        if pred_text != gold_text:
            return "text_mismatch"

    # 8. Fallback
    return "manual_diagnosis_required"
