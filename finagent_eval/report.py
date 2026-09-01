"""Scorecard generation from enriched result files."""
from __future__ import annotations

import random
from collections import Counter

from .graders.normalize import GRADER_VERSION


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean (deterministic for a fixed seed)."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo = min(max(int(round(alpha * (n_boot - 1))), 0), n_boot - 1)
    hi = min(max(int(round((1 - alpha) * (n_boot - 1))), 0), n_boot - 1)
    return round(means[lo], 4), round(means[hi], 4)


# Backwards-compatible private alias.
_bootstrap_ci = bootstrap_ci


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def generate_scorecard(enriched_results: list[dict], run_metadata: dict | None = None) -> dict:
    """Generate a per-model scorecard from enriched results."""
    if not enriched_results:
        raise ValueError("generate_scorecard needs at least one enriched row")
    model = enriched_results[0]["model"]
    total = len(enriched_results)

    anls_values = [r["recomputed_anls"] for r in enriched_results]
    em_values = [1.0 if r["recomputed_exact_match"] else 0.0 for r in enriched_results]
    pc_values = [1.0 if r.get("project_correct") else 0.0 for r in enriched_results]
    raw_anls_values = [r.get("raw_anls", r["recomputed_anls"]) for r in enriched_results]

    by_category: dict[str, dict[str, list[float]]] = {}
    for r, em, pc in zip(enriched_results, em_values, pc_values, strict=True):
        cat = by_category.setdefault(r["category"], {"anls": [], "em": [], "pc": []})
        cat["anls"].append(r["recomputed_anls"])
        cat["em"].append(em)
        cat["pc"].append(pc)

    category_breakdown = {}
    for cat, vals in sorted(by_category.items()):
        n = len(vals["anls"])
        category_breakdown[cat] = {
            "anls": round(_mean(vals["anls"]), 4),
            "anls_ci": bootstrap_ci(vals["anls"]),
            "exact_match": round(_mean(vals["em"]), 4),
            "exact_match_ci": bootstrap_ci(vals["em"]),
            "project_correct": round(_mean(vals["pc"]), 4),
            "project_correct_count": int(sum(vals["pc"])),
            "n": n,
        }

    failures = [r for r in enriched_results if r.get("failure_label") is not None]
    label_counts = dict(Counter(r["failure_label"] for r in failures).most_common())

    weakest = min(category_breakdown, key=lambda c: category_breakdown[c]["anls"])

    worst = sorted(failures, key=lambda r: (r["recomputed_anls"], r["task_id"]))[:5]
    top_5_worst = [
        {
            "task_id": r["task_id"],
            "ticker": r.get("ticker"),
            "category": r["category"],
            "question": r.get("question", ""),
            "gold_answer": r["gold_answer"],
            "prediction": (r["prediction"] or "")[:200],
            "failure_label": r["failure_label"],
            "anls": r["recomputed_anls"],
            "image_path": r.get("image_path"),
        }
        for r in worst
    ]

    latencies = [r["latency_ms"] for r in enriched_results if r.get("latency_ms") is not None]

    return {
        "model": model,
        "grader_version": GRADER_VERSION,
        "total_questions": total,
        "overall_anls": round(_mean(anls_values), 4),
        "overall_anls_ci": bootstrap_ci(anls_values),
        "overall_raw_anls": round(_mean(raw_anls_values), 4),
        "overall_exact_match": round(_mean(em_values), 4),
        "overall_exact_match_ci": bootstrap_ci(em_values),
        "overall_project_correct": round(_mean(pc_values), 4),
        "overall_project_correct_ci": bootstrap_ci(pc_values),
        "project_correct_count": int(sum(pc_values)),
        "mean_latency_ms": round(_mean(latencies), 1) if latencies else None,
        "category_breakdown": category_breakdown,
        "weakest_category": weakest,
        "failure_mode_distribution": label_counts,
        "total_failures": len(failures),
        "top_5_worst_failures": top_5_worst,
        "run_metadata": run_metadata or {},
    }


def generate_comparison_table(scorecards: list[dict]) -> dict:
    """Generate cross-model comparison from multiple scorecards, best ANLS first."""
    if not scorecards:
        raise ValueError("generate_comparison_table needs at least one scorecard")
    categories: list[str] = []
    for sc in scorecards:
        for cat in sc["category_breakdown"]:
            if cat not in categories:
                categories.append(cat)

    rows = []
    for sc in sorted(scorecards, key=lambda s: s["overall_anls"], reverse=True):
        row = {
            "model": sc["model"],
            "overall_anls": sc["overall_anls"],
            "overall_exact_match": sc["overall_exact_match"],
            "overall_project_correct": sc.get("overall_project_correct"),
        }
        for cat in categories:
            row[f"{cat}_anls"] = sc["category_breakdown"].get(cat, {}).get("anls")
        rows.append(row)

    return {"grader_version": GRADER_VERSION, "categories": categories, "models": rows}
