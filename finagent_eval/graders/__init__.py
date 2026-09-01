"""Grading primitives and the single ``grade`` entry point.

``grade`` is the one place that defines what "project-correct" means. Every
consumer (enrichment, the adapter runner, the reliability probe re-grader, the
notebook) must go through it so the definition cannot drift.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .anls import anls_score
from .exact_match import exact_match_score
from .normalize import (
    GRADER_VERSION,
    FinancialNumber,
    financial_numbers_equivalent,
    normalize_answer_text,
    normalize_financial_number,
    normalize_text,
    parse_financial_number,
)
from .numeric import DEFAULT_TOLERANCE, numeric_match_is_correct, numeric_match_score


@dataclass(frozen=True)
class GradeResult:
    """Outcome of grading one prediction against one gold answer."""

    anls: float
    raw_anls: float
    exact_match: bool
    numeric_match: bool | None
    numeric_tolerance_correct: bool
    accepted_by_numeric_tolerance: bool
    project_correct: bool

    def to_dict(self) -> dict:
        return asdict(self)


# Categories where the gold answer is itself computed or read off a chart, so a
# rounded answer (23% for 23.3%) is legitimately correct. For extractive and
# layout questions the number is printed verbatim in the document, and a value
# that is "close" (37,586 for 37,855) is a misread digit, i.e. wrong.
TOLERANCE_CATEGORIES: frozenset[str] = frozenset({"numerical_reasoning", "chart_interpretation"})


def tolerance_applies(category: str | None) -> bool:
    return category is None or category in TOLERANCE_CATEGORIES


def grade(
    prediction: str | None,
    gold: str,
    category: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> GradeResult:
    """Grade a prediction. ``project_correct`` = exact match OR numeric tolerance match.

    Numeric tolerance only counts as correctness for categories in
    :data:`TOLERANCE_CATEGORIES` (or when ``category`` is not given).
    ``numeric_match`` is always reported for analysis regardless of category.
    """
    pred = "" if prediction is None else str(prediction)
    anls = anls_score(pred, gold)
    raw_anls = anls_score(pred, gold, numeric_aware=False)
    em = exact_match_score(pred, gold)
    num = numeric_match_score(pred, gold, tolerance=tolerance)
    num_correct = numeric_match_is_correct(pred, gold, tolerance=tolerance) and tolerance_applies(category)
    return GradeResult(
        anls=round(anls, 6),
        raw_anls=round(raw_anls, 6),
        exact_match=em,
        numeric_match=num,
        numeric_tolerance_correct=num_correct,
        accepted_by_numeric_tolerance=bool(num_correct and not em),
        project_correct=bool(em or num_correct),
    )


__all__ = [
    "DEFAULT_TOLERANCE",
    "GRADER_VERSION",
    "TOLERANCE_CATEGORIES",
    "FinancialNumber",
    "GradeResult",
    "anls_score",
    "exact_match_score",
    "financial_numbers_equivalent",
    "grade",
    "normalize_answer_text",
    "normalize_financial_number",
    "normalize_text",
    "numeric_match_is_correct",
    "numeric_match_score",
    "parse_financial_number",
    "tolerance_applies",
]
