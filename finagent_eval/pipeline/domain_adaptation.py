"""Pix2Struct LoRA domain adaptation on the 90-question held-out subset.

This is separate from the main 397-question leaderboard: the LoRA model was
evaluated only on JPM/MSFT/WMT held-out questions.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from ..graders import GRADER_VERSION, grade
from ..paths import ProjectPaths
from ._io import read_json, write_json
from .normalize_results import task_key

log = logging.getLogger(__name__)

EXPECTED_HELDOUT = 90


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict], anls_key: str, em_key: str, pc_key: str | None = None) -> dict:
    by_category: dict[str, list[dict]] = defaultdict(list)
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
        by_ticker[r["ticker"]].append(r)

    def group_summary(group: list[dict]) -> dict:
        out = {
            "n": len(group),
            "anls": round(_mean([r[anls_key] for r in group]), 4),
            "exact_match": round(_mean([1.0 if r[em_key] else 0.0 for r in group]), 4),
        }
        if pc_key:
            out["project_correct"] = round(_mean([1.0 if r[pc_key] else 0.0 for r in group]), 4)
        return out

    summary = group_summary(rows)
    return {
        "n": len(rows),
        "overall_anls": summary["anls"],
        "overall_exact_match": summary["exact_match"],
        **({"overall_project_correct": summary["project_correct"]} if pc_key else {}),
        "by_category": {k: group_summary(v) for k, v in sorted(by_category.items())},
        "by_ticker": {k: group_summary(v) for k, v in sorted(by_ticker.items())},
    }


def _row(task: dict, model: str, prediction: str, stored_anls: float, stored_em: bool) -> dict:
    g = grade(prediction, task["gold_answer"], category=task["category"])
    return {
        "task_id": task["task_id"],
        "model": model,
        "ticker": task["ticker"],
        "company": task["company"],
        "category": task["category"],
        "question": task["question"],
        "gold_answer": task["gold_answer"],
        "prediction": prediction,
        "image_path": task["image_path"],
        "stored_anls": float(stored_anls),
        "stored_exact_match": bool(stored_em),
        "recomputed_anls": g.anls,
        "recomputed_exact_match": g.exact_match,
        "project_correct": g.project_correct,
    }


def analyze_domain_adaptation(paths: ProjectPaths, strict: bool = True) -> dict:
    zero_path = paths.raw_dir / "final_pix2struct_results.json"
    lora_path = paths.raw_dir / "financial_finetuned_results.json"
    tasks = read_json(paths.tasks)
    zero_raw = read_json(zero_path)["per_example"]
    lora_raw = read_json(lora_path)

    task_lookup = {task_key(t["ticker"], t["question"], t["gold_answer"], t["category"]): t for t in tasks}
    zero_lookup = {task_key(e["ticker"], e["question"], e["ground_truth"], e["question_type"]): e for e in zero_raw}

    zero_rows, lora_rows, paired_rows = [], [], []
    for ex in lora_raw["per_example"]:
        key = task_key(ex["ticker"], ex["question"], ex["ground_truth"], ex["question_type"])
        task = task_lookup[key]
        zero = zero_lookup[key]
        zero_row = _row(task, "Pix2Struct zero-shot", zero["prediction"], zero["anls"], zero["exact_match"])
        lora_row = _row(task, "Pix2Struct LoRA", ex["prediction"], ex["anls"], ex["exact_match"])
        zero_rows.append(zero_row)
        lora_rows.append(lora_row)
        paired_rows.append(
            {
                "task_id": task["task_id"],
                "ticker": task["ticker"],
                "category": task["category"],
                "question": task["question"],
                "gold_answer": task["gold_answer"],
                "zero_shot_prediction": zero_row["prediction"],
                "lora_prediction": lora_row["prediction"],
                "zero_shot_stored_anls": zero_row["stored_anls"],
                "lora_stored_anls": lora_row["stored_anls"],
                "stored_delta_anls": round(lora_row["stored_anls"] - zero_row["stored_anls"], 6),
                "zero_shot_recomputed_anls": zero_row["recomputed_anls"],
                "lora_recomputed_anls": lora_row["recomputed_anls"],
                "recomputed_delta_anls": round(lora_row["recomputed_anls"] - zero_row["recomputed_anls"], 6),
                "zero_shot_exact_match": zero_row["stored_exact_match"],
                "lora_exact_match": lora_row["stored_exact_match"],
                "zero_shot_project_correct": zero_row["project_correct"],
                "lora_project_correct": lora_row["project_correct"],
            }
        )

    if strict and (len(lora_rows) != EXPECTED_HELDOUT or len({r["task_id"] for r in lora_rows}) != EXPECTED_HELDOUT):
        raise RuntimeError(f"expected {EXPECTED_HELDOUT} unique held-out rows, got {len(lora_rows)}")

    zero_stored = summarize(zero_rows, "stored_anls", "stored_exact_match")
    lora_stored = summarize(lora_rows, "stored_anls", "stored_exact_match")
    zero_rec = summarize(zero_rows, "recomputed_anls", "recomputed_exact_match", "project_correct")
    lora_rec = summarize(lora_rows, "recomputed_anls", "recomputed_exact_match", "project_correct")

    improved = sum(1 for r in paired_rows if r["stored_delta_anls"] > 0)
    worse = sum(1 for r in paired_rows if r["stored_delta_anls"] < 0)
    unchanged = len(paired_rows) - improved - worse
    rec_improved = sum(1 for r in paired_rows if r["recomputed_delta_anls"] > 0)
    rec_worse = sum(1 for r in paired_rows if r["recomputed_delta_anls"] < 0)

    summary = {
        "analysis": "Pix2Struct LoRA domain adaptation",
        "important_note": "Separate analysis only. Do not include Pix2Struct LoRA in the main 397-question leaderboard.",
        "grader_version": GRADER_VERSION,
        "train_companies": lora_raw["train_companies"],
        "test_companies": lora_raw["test_companies"],
        "train_pairs": lora_raw["train_pairs"],
        "test_pairs": lora_raw["test_pairs"],
        "heldout_questions": len(lora_rows),
        "stored_original_evaluator": {
            "zero_shot": zero_stored,
            "lora": lora_stored,
            "delta": {
                "anls": round(lora_stored["overall_anls"] - zero_stored["overall_anls"], 4),
                "exact_match": round(lora_stored["overall_exact_match"] - zero_stored["overall_exact_match"], 4),
                "relative_anls_change_pct": round(
                    (lora_stored["overall_anls"] - zero_stored["overall_anls"]) / zero_stored["overall_anls"] * 100, 1
                ) if zero_stored["overall_anls"] else None,
            },
        },
        "recomputed_project_grader": {
            "zero_shot": zero_rec,
            "lora": lora_rec,
            "delta": {
                "anls": round(lora_rec["overall_anls"] - zero_rec["overall_anls"], 4),
                "exact_match": round(lora_rec["overall_exact_match"] - zero_rec["overall_exact_match"], 4),
                "project_correct": round(lora_rec["overall_project_correct"] - zero_rec["overall_project_correct"], 4),
            },
        },
        "paired_outcomes_stored_anls": {"improved": improved, "worse": worse, "unchanged": unchanged},
        "paired_outcomes_recomputed_anls": {
            "improved": rec_improved,
            "worse": rec_worse,
            "unchanged": len(paired_rows) - rec_improved - rec_worse,
        },
        "top_improvements": sorted(paired_rows, key=lambda r: (-r["stored_delta_anls"], r["task_id"]))[:5],
        "top_regressions": sorted(paired_rows, key=lambda r: (r["stored_delta_anls"], r["task_id"]))[:5],
    }

    out = paths.domain_adaptation_dir
    write_json(out / "pix2struct_zero_shot_heldout.json", zero_rows)
    write_json(out / "pix2struct_lora_heldout.json", lora_rows)
    write_json(out / "pix2struct_domain_adaptation_pairs.json", paired_rows)
    write_json(out / "domain_adaptation_summary.json", summary)
    write_json(paths.reports_dir / "domain_adaptation_summary.json", summary)

    log.info("Domain adaptation: stored zero-shot ANLS=%.4f LoRA=%.4f | recomputed zero-shot=%.4f LoRA=%.4f",
             zero_stored["overall_anls"], lora_stored["overall_anls"], zero_rec["overall_anls"], lora_rec["overall_anls"])
    return summary
