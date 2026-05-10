"""ANLS (Average Normalized Levenshtein Similarity) grader."""
from .normalize import normalize_text


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,        # insert
                prev_row[j + 1] + 1,    # delete
                prev_row[j] + cost      # substitute
            ))
        prev_row = curr_row

    return prev_row[-1]


def anls_score(prediction: str, gold: str, threshold: float = 0.5) -> float:
    """
    Compute ANLS between prediction and gold answer.

    ANLS = 1 - NL if NL < threshold, else 0
    where NL = levenshtein(pred, gold) / max(len(pred), len(gold))

    Standard threshold is 0.5 (used in DocVQA).
    """
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)

    if not gold_norm:
        return 1.0 if not pred_norm else 0.0

    if not pred_norm:
        return 0.0

    dist = _levenshtein_distance(pred_norm, gold_norm)
    max_len = max(len(pred_norm), len(gold_norm))
    nl = dist / max_len

    if nl < threshold:
        return 1.0 - nl
    else:
        return 0.0
