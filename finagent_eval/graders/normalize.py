"""Shared normalization functions for financial text answers.

Two layers live here:

* ``normalize_text`` / ``normalize_answer_text`` - plain-text canonicalisation
  (case, whitespace, unicode punctuation, answer prefixes, trailing punctuation).
* ``parse_financial_number`` - a structured parse of a financial quantity that
  keeps the written mantissa and the unit scale separate, so that
  ``"$85,269 million"`` can be compared against a gold answer of ``"85,269"``
  taken from a table headed "in millions".
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Version of the grading rules. Bump when a change alters scores, and record it
# in CHANGELOG.md so historical leaderboards can be interpreted.
GRADER_VERSION = "2.0"

_UNICODE_MAP = str.maketrans(
    {
        "’": "'",  # right single quotation mark
        "‘": "'",  # left single quotation mark
        "“": '"',
        "”": '"',
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus sign
        " ": " ",  # non-breaking space
    }
)

_ANSWER_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:answer|final answer)\s*(?:is|:)\s*[:\-]?\s*|^(?:it\s+is|it's)\s+",
    re.IGNORECASE,
)
_APPROX_PREFIX_RE = re.compile(r"^(?:approximately|approx\.?|about|around|roughly|~)\s*", re.IGNORECASE)
_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[\.\;\:,\s]+$")
_CURRENCY_RE = re.compile(r"(?:us\s?\$|u\.s\.\s?\$|[\$€£¥₹])", re.IGNORECASE)
_CURRENCY_WORD_RE = re.compile(r"\s*(?:u\.?s\.?\s+)?(?:dollars?|usd|eur|gbp)\s*$", re.IGNORECASE)
_PERCENT_WORD_RE = re.compile(r"\s*(?:%|percent|pct\.?|percentage points?)\s*$", re.IGNORECASE)

SCALE_WORDS: dict[str, int] = {
    "k": 1_000,
    "thousand": 1_000,
    "thousands": 1_000,
    "m": 1_000_000,
    "mm": 1_000_000,
    "mn": 1_000_000,
    "mil": 1_000_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "bil": 1_000_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
    "t": 1_000_000_000_000,
    "tn": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,
    "trillions": 1_000_000_000_000,
}
_SCALE_RE = re.compile(
    r"^([+-]?\d[\d.]*|[+-]?\.\d+)\s*(" + "|".join(sorted(SCALE_WORDS, key=len, reverse=True)) + r")$",
    re.IGNORECASE,
)
_PLAIN_NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$", re.IGNORECASE)


def normalize_text(text: str | None) -> str:
    """Basic text normalization: unicode punctuation, lowercase, strip, collapse whitespace.

    Trailing sentence punctuation is removed so that ``"First half."`` and
    ``"First half"`` compare equal. Curly quotes become straight quotes.
    """
    if not text:
        return ""
    s = str(text).translate(_UNICODE_MAP)
    s = re.sub(r"\s+", " ", s.strip().lower())
    return _TRAILING_PUNCT_RE.sub("", s)


def normalize_answer_text(text: str | None) -> str:
    """Stronger text normalization used for exact match on non-numeric answers.

    Beyond :func:`normalize_text` this strips answer prefixes ("Answer:", "The
    answer is"), a leading article, and surrounding quotes.
    """
    s = normalize_text(text)
    if not s:
        return ""
    s = _ANSWER_PREFIX_RE.sub("", s)
    s = s.strip().strip("\"'")
    s = _LEADING_ARTICLE_RE.sub("", s)
    return _TRAILING_PUNCT_RE.sub("", s.strip())


@dataclass(frozen=True)
class FinancialNumber:
    """A parsed financial quantity.

    ``value`` is the number as written (after sign handling), ``scale`` is the
    multiplier implied by a unit word or suffix (1 when none was written), and
    ``scaled`` is the absolute quantity.
    """

    value: float
    scale: int = 1
    has_unit: bool = False
    is_percent: bool = False

    @property
    def scaled(self) -> float:
        return self.value * self.scale

    def canonical(self) -> str:
        """Canonical string of the scaled quantity, e.g. ``1500000`` or ``12.5%``."""
        val = float(self.scaled)
        out = str(int(val)) if val.is_integer() else repr(val)
        return f"{out}%" if self.is_percent else out


def parse_financial_number(text: str | None) -> FinancialNumber | None:
    """Parse a financial number string. Returns ``None`` if it is not a number.

    Handles:
    - answer prefixes: ``"Answer: 1,200"``, ``"approximately 115 million"``
    - currency symbols and words: ``$``, ``EUR``, ``GBP``, ``US$``, ``USD``, ``dollars``
    - thousands separators: ``1,452,000``
    - accounting negatives: ``(500)``, ``($500)``, ``-$1,200``
    - percentages: ``12.5%``, ``12.5 percent``
    - unit suffixes and words: ``1.5M``, ``2.3B``, ``85,269 million``, ``$53.8 billion``
    """
    if text is None:
        return None
    s = str(text).translate(_UNICODE_MAP).strip()
    if not s:
        return None

    s = _ANSWER_PREFIX_RE.sub("", s).strip()
    s = _APPROX_PREFIX_RE.sub("", s).strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    if not s:
        return None

    negative = False
    paren_match = re.match(r"^\((.+)\)$", s)
    if paren_match:
        negative = True
        s = paren_match.group(1).strip()

    s = _CURRENCY_WORD_RE.sub("", s)
    s = _CURRENCY_RE.sub("", s).strip()

    is_percent = False
    pct = _PERCENT_WORD_RE.search(s)
    if pct and pct.start() > 0:
        is_percent = True
        s = s[: pct.start()].strip()

    if s.startswith("-"):
        negative = not negative
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    s = s.replace(",", "").strip()
    if not s:
        return None

    scale = 1
    has_unit = False
    scale_match = _SCALE_RE.match(s)
    if scale_match:
        num_str, unit = scale_match.group(1), scale_match.group(2).lower()
        scale = SCALE_WORDS[unit]
        has_unit = True
        s = num_str

    if not _PLAIN_NUMBER_RE.match(s):
        return None
    try:
        value = float(s)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    if negative:
        value = -value
    return FinancialNumber(value=value, scale=scale, has_unit=has_unit, is_percent=is_percent)


def financial_numbers_equivalent(
    prediction: FinancialNumber,
    gold: FinancialNumber,
    tolerance: float = 0.0,
) -> bool:
    """Compare two parsed numbers, tolerating an implicit unit on one side.

    Gold answers in this benchmark are copied from tables whose header states
    the unit ("in millions"), so ``85,269`` and ``$85,269 million`` are the same
    answer. Rules:

    1. A percentage never equals a non-percentage.
    2. If the fully scaled quantities agree (within ``tolerance``), match.
    3. Otherwise, if exactly one side carries an explicit unit, compare the
       written mantissas instead (the unit is treated as implicit on the other).
    """
    if prediction.is_percent != gold.is_percent:
        return False

    def close(a: float, b: float) -> bool:
        if b == 0:
            return a == 0
        if tolerance <= 0:
            return a == b
        return abs(a - b) / abs(b) <= tolerance

    if close(prediction.scaled, gold.scaled):
        return True
    if prediction.has_unit != gold.has_unit:
        return close(prediction.value, gold.value)
    return False


def normalize_financial_number(text: str | None) -> str | None:
    """Canonical string for a financial number, or ``None`` if not a number.

    Kept for backwards compatibility with the v1 grader. The scaled quantity is
    returned (``"1.5M"`` -> ``"1500000"``); use :func:`parse_financial_number`
    when you need unit-aware comparison.
    """
    parsed = parse_financial_number(text)
    return None if parsed is None else parsed.canonical()
