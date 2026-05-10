"""Scorecard generation from enriched result files."""
import json
import random
from collections import Counter
from pathlib import Path


def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = int((1 - ci) / 2 * n_boot)
    hi = int((1 + ci) / 2 * n_boot)
    return round(means[lo], 4), round(means[hi], 4)


def generate_scorecard(enriched_results: list[dict]) -> dict:
    """Generate a per-model scorecard from enriched results."""
    model = enriched_results[0]["model"]
    total = len(enriched_results)

    # Overall metrics
    anls_values = [r["recomputed_anls"] for r in enriched_results]
    avg_anls = round(sum(anls_values) / total, 4)
    anls_ci = _bootstrap_ci(anls_values)

    em_values = [1.0 if r["recomputed_exact_match"] else 0.0 for r in enriched_results]
    em_rate = round(sum(em_values) / total, 4)
    em_ci = _bootstrap_ci(em_values)

    # Per-category breakdown
    by_category = {}
    for r in enriched_results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"anls": [], "em": []}
        by_category[cat]["anls"].append(r["recomputed_anls"])
        by_category[cat]["em"].append(1.0 if r["recomputed_exact_match"] else 0.0)

    category_breakdown = {}
    for cat, vals in sorted(by_category.items()):
        n = len(vals["anls"])
        cat_anls = round(sum(vals["anls"]) / n, 4)
        cat_em = round(sum(vals["em"]) / n, 4)
        category_breakdown[cat] = {
            "anls": cat_anls,
            "anls_ci": _bootstrap_ci(vals["anls"]),
            "exact_match": cat_em,
            "exact_match_ci": _bootstrap_ci(vals["em"]),
            "n": n,
        }

    # Failure mode distribution
    failures = [r for r in enriched_results if r["failure_label"] is not None]
    label_counts = dict(Counter(r["failure_label"] for r in failures).most_common())

    # Weakest category
    weakest = min(category_breakdown, key=lambda c: category_breakdown[c]["anls"])

    # Top 5 worst failures by ANLS
    worst = sorted(failures, key=lambda r: r["recomputed_anls"])[:5]
    top_5_worst = [
        {
            "task_id": r["task_id"],
            "ticker": r.get("ticker"),
            "category": r["category"],
            "question": r.get("question", ""),
            "gold_answer": r["gold_answer"],
            "prediction": r["prediction"][:200],
            "failure_label": r["failure_label"],
            "anls": r["recomputed_anls"],
            "image_path": r.get("image_path"),
        }
        for r in worst
    ]

    return {
        "model": model,
        "total_questions": total,
        "overall_anls": avg_anls,
        "overall_anls_ci": anls_ci,
        "overall_exact_match": em_rate,
        "overall_exact_match_ci": em_ci,
        "category_breakdown": category_breakdown,
        "weakest_category": weakest,
        "failure_mode_distribution": label_counts,
        "total_failures": len(failures),
        "top_5_worst_failures": top_5_worst,
    }


def generate_comparison_table(scorecards: list[dict]) -> dict:
    """Generate cross-model comparison from multiple scorecards."""
    categories = list(scorecards[0]["category_breakdown"].keys())

    rows = []
    for sc in scorecards:
        row = {
            "model": sc["model"],
            "overall_anls": sc["overall_anls"],
            "overall_exact_match": sc["overall_exact_match"],
        }
        for cat in categories:
            row[f"{cat}_anls"] = sc["category_breakdown"][cat]["anls"]
        rows.append(row)

    return {"categories": categories, "models": rows}
