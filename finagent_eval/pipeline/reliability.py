"""Re-grade the paraphrase reliability probe with the project grader.

The probe notebook records raw predictions for each question variant. Its
in-notebook correctness flags used an older, exact-only matcher; this stage
recomputes ``correct`` with :func:`finagent_eval.graders.grade` so that
"correct" means the same thing on the leaderboard and in the reliability tab,
and writes a summary file with the aggregate metrics quoted in the README.
"""
from __future__ import annotations

import logging
from collections import Counter

from ..graders import GRADER_VERSION, grade, normalize_answer_text, parse_financial_number
from ..paths import ProjectPaths
from ._io import read_json, write_json

log = logging.getLogger(__name__)


def canonical_answer(answer: str | None) -> str:
    """Stable key used to decide whether two phrasings produced the *same* answer."""
    num = parse_financial_number(answer)
    if num is not None:
        return f"NUM::{num.canonical()}"
    return f"TXT::{normalize_answer_text(answer)}"


def regrade_row(row: dict) -> dict:
    gold = row["gold_answer"]
    variants = []
    for v in row["variants"]:
        g = grade(v.get("prediction"), gold, category=row.get("category"))
        variants.append({**v, "canonical_prediction": canonical_answer(v.get("prediction")), "correct": g.project_correct, "anls": g.anls})
    canon = {v["canonical_prediction"] for v in variants}
    corrects = [v["correct"] for v in variants]
    original = next((v for v in variants if v.get("variant") == "original"), variants[0])
    return {
        **row,
        "canonical_gold": canonical_answer(gold),
        "variants": variants,
        "consistent_answer": len(canon) == 1,
        "correctness_stable": len(set(corrects)) == 1,
        "original_correct": bool(original["correct"]),
        "all_correct": all(corrects),
        "any_correct": any(corrects),
        "grader_version": GRADER_VERSION,
    }


def summarize_probe(rows: list[dict]) -> dict:
    n = len(rows)

    def frac(key: str, subset: list[dict]) -> dict:
        k = sum(1 for r in subset if r.get(key))
        return {"count": k, "total": len(subset), "rate": round(k / len(subset), 4) if subset else None}

    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    return {
        "grader_version": GRADER_VERSION,
        "n_tasks": n,
        "variants_per_task": len(rows[0]["variants"]) if rows else 0,
        "answer_consistency": frac("consistent_answer", rows),
        "correctness_stability": frac("correctness_stable", rows),
        "original_accuracy": frac("original_correct", rows),
        "all_variants_correct": frac("all_correct", rows),
        "any_variant_correct": frac("any_correct", rows),
        "by_category": {
            cat: {
                "n": len(sub),
                "consistent": frac("consistent_answer", sub)["count"],
                "all_correct": frac("all_correct", sub)["count"],
                "original_correct": frac("original_correct", sub)["count"],
            }
            for cat, sub in sorted(by_cat.items())
        },
        "category_counts": dict(Counter(r["category"] for r in rows)),
    }


def regrade_reliability_probe(paths: ProjectPaths) -> dict | None:
    if not paths.reliability_results.exists():
        log.warning("no reliability probe results at %s, skipping", paths.reliability_results)
        return None
    rows = [regrade_row(r) for r in read_json(paths.reliability_results)]
    write_json(paths.reliability_results, rows)
    summary = summarize_probe(rows)
    write_json(paths.reliability_summary, summary)
    log.info(
        "Reliability probe: consistency %d/%d, stability %d/%d, all-correct %d/%d",
        summary["answer_consistency"]["count"], summary["n_tasks"],
        summary["correctness_stability"]["count"], summary["n_tasks"],
        summary["all_variants_correct"]["count"], summary["n_tasks"],
    )
    return summary
