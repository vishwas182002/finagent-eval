"""Semi-automatic failure mode classifier for incorrect answers."""
import re
from .graders.normalize import normalize_financial_number
from .graders.numeric import numeric_match_score


def _extract_years(text: str) -> set:
    """Extract 4-digit years (2000-2099), including FY-prefixed."""
    return set(re.findall(r'(20\d{2})(?!\d)', text))


def classify_failure(
    prediction: str,
    gold_answer: str,
    category: str,
    question: str = "",
    source_text: str = "",
) -> str:
    """
    Classify why a prediction is wrong.
    Returns one of:
        - empty_output
        - abstention
        - format_mismatch
        - hallucinated_number (only if source_text provided)
        - wrong_year
        - numeric_mismatch
        - manual_diagnosis_required
    """
    pred_stripped = prediction.strip() if prediction else ""

    # 1. Empty output
    if not pred_stripped:
        return "empty_output"

    # 2. Abstention
    abstention_patterns = [
        r"^n/?a$", r"^none$", r"^unknown$", r"^i don'?t know$",
        r"^unable to determine$", r"^not available$", r"^cannot determine$",
        r"^not found$", r"^no answer$",
    ]
    pred_lower = pred_stripped.lower()
    for pattern in abstention_patterns:
        if re.match(pattern, pred_lower):
            return "abstention"

    # 3. Check if both are numbers
    pred_num = normalize_financial_number(prediction)
    gold_num = normalize_financial_number(gold_answer)

    if pred_num is not None and gold_num is not None:
        pred_is_pct = pred_num.endswith('%')
        gold_is_pct = gold_num.endswith('%')

        try:
            pred_val = float(pred_num.rstrip('%'))
            gold_val = float(gold_num.rstrip('%'))
        except ValueError:
            pred_val, gold_val = None, None

        # 3a. Format mismatch — same value AND same percent status
        if pred_val is not None and pred_val == gold_val and pred_is_pct == gold_is_pct:
            return "format_mismatch"

    # 4. Wrong year — check BEFORE numeric mismatch so "2024" vs question "2025" isn't numeric
    q_years = _extract_years(question)
    pred_years = _extract_years(pred_stripped)
    if pred_years and q_years and not pred_years.intersection(q_years):
        return "wrong_year"

    # 5. Remaining numeric checks
    if pred_num is not None and gold_num is not None:
        # 5a. Hallucinated number — predicted number not in source document
        if source_text:
            source_numbers = set()
            for match in re.findall(r'[\d,]+\.?\d*', source_text):
                n = normalize_financial_number(match)
                if n is not None:
                    source_numbers.add(n)
            if pred_num not in source_numbers:
                return "hallucinated_number"

        # 5b. Numeric mismatch — outside 2% tolerance
        if numeric_match_score(prediction, gold_answer) is False:
            return "numeric_mismatch"

    # 6. Fallback
    return "manual_diagnosis_required"
