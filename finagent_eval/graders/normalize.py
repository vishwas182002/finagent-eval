"""Shared normalization functions for financial text answers."""
import re


def normalize_text(text: str) -> str:
    """Basic text normalization: lowercase, strip, collapse whitespace."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip().lower())


def normalize_financial_number(text: str) -> str | None:
    """
    Attempt to parse a financial number string into a canonical string.
    Returns None if the text is not a recognizable number.

    Handles:
    - Currency symbols: $, €, £ (including -$1,200 and ($500))
    - Commas: 1,452,000 -> 1452000
    - Parentheses for negatives: (500) -> -500, ($500) -> -500
    - Percentage signs: 12.5% -> 12.5%
    - Suffixes: 1.5M -> 1500000, 2.3B -> 2300000000
    """
    if not text:
        return None

    s = text.strip()

    # Handle parentheses for negatives: (500) or ($500) -> -500
    paren_match = re.match(r'^\((.+)\)$', s)
    if paren_match:
        s = '-' + paren_match.group(1)

    # Strip currency symbols anywhere after sign
    s = re.sub(r'[\$€£]', '', s)

    # Track and strip percentage sign
    is_percent = s.endswith('%')
    if is_percent:
        s = s[:-1].strip()

    # Strip commas
    s = s.replace(',', '')

    # Strip whitespace that might remain
    s = s.strip()

    # Handle suffixes: M, B, T, K
    suffix_multipliers = {
        'k': 1_000,
        'm': 1_000_000,
        'b': 1_000_000_000,
        't': 1_000_000_000_000,
    }
    suffix_match = re.match(r'^(-?[\d.]+)\s*([kmbt])$', s, re.IGNORECASE)
    if suffix_match:
        num_str = suffix_match.group(1)
        suffix = suffix_match.group(2).lower()
        try:
            val = float(num_str) * suffix_multipliers[suffix]
            s = str(val)
        except ValueError:
            return None

    # Try parsing as float
    try:
        val = float(s)
        # Clean integer output: 1452000 not 1452000.0
        if val == int(val) and not is_percent:
            result = str(int(val))
        else:
            result = str(val)
        if is_percent:
            return f"{result}%"
        return result
    except ValueError:
        return None
