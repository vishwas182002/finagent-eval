"""Analyze Pix2Struct LoRA domain adaptation on the 90-question held-out subset.

Important: this is separate from the main 397-question leaderboard.
The LoRA model was evaluated only on JPM/MSFT/WMT held-out questions.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = PROJECT_ROOT / "data/track_a/tasks.json"
ZERO_RAW_PATH = PROJECT_ROOT / "data/track_a/raw/final_pix2struct_results.json"
LORA_RAW_PATH = PROJECT_ROOT / "data/track_a/raw/financial_finetuned_results.json"
OUT_DIR = PROJECT_ROOT / "results/domain_adaptation"
REPORTS_DIR = PROJECT_ROOT / "reports"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from finagent_eval.graders.anls import anls_score
from finagent_eval.graders.exact_match import exact_match_score


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(rows, anls_key="stored_anls", em_key="stored_exact_match"):
    by_category = defaultdict(list)
    by_ticker = defaultdict(list)

    for r in rows:
        by_category[r["category"]].append(r)
        by_ticker[r["ticker"]].append(r)

    def group_summary(group):
        return {
            "n": len(group),
            "anls": round(mean([r[anls_key] for r in group]), 4),
            "exact_match": round(mean([1.0 if r[em_key] else 0.0 for r in group]), 4),
        }

    return {
        "n": len(rows),
        "overall_anls": round(mean([r[anls_key] for r in rows]), 4),
        "overall_exact_match": round(mean([1.0 if r[em_key] else 0.0 for r in rows]), 4),
        "by_category": {k: group_summary(v) for k, v in sorted(by_category.items())},
        "by_ticker": {k: group_summary(v) for k, v in sorted(by_ticker.items())},
    }


with TASKS_PATH.open(encoding="utf-8") as f:
    tasks = json.load(f)

with ZERO_RAW_PATH.open(encoding="utf-8") as f:
    zero_raw = json.load(f)["per_example"]

with LORA_RAW_PATH.open(encoding="utf-8") as f:
    lora_raw = json.load(f)

task_lookup = {
    (t["ticker"], t["question"], t["gold_answer"], t["category"]): t
    for t in tasks
}

zero_lookup = {
    (e["ticker"], e["question"], e["ground_truth"], e["question_type"]): e
    for e in zero_raw
}

zero_rows = []
lora_rows = []
paired_rows = []

for ex in lora_raw["per_example"]:
    key = (ex["ticker"], ex["question"], ex["ground_truth"], ex["question_type"])
    task = task_lookup[key]
    zero = zero_lookup[key]

    zero_row = {
        "task_id": task["task_id"],
        "model": "Pix2Struct zero-shot",
        "ticker": task["ticker"],
        "company": task["company"],
        "category": task["category"],
        "question": task["question"],
        "gold_answer": task["gold_answer"],
        "prediction": zero["prediction"],
        "image_path": task["image_path"],
        "stored_anls": float(zero["anls"]),
        "stored_exact_match": bool(zero["exact_match"]),
        "recomputed_anls": round(anls_score(zero["prediction"], zero["ground_truth"]), 6),
        "recomputed_exact_match": exact_match_score(zero["prediction"], zero["ground_truth"]),
    }

    lora_row = {
        "task_id": task["task_id"],
        "model": "Pix2Struct LoRA",
        "ticker": task["ticker"],
        "company": task["company"],
        "category": task["category"],
        "question": task["question"],
        "gold_answer": task["gold_answer"],
        "prediction": ex["prediction"],
        "image_path": task["image_path"],
        "stored_anls": float(ex["anls"]),
        "stored_exact_match": bool(ex["exact_match"]),
        "recomputed_anls": round(anls_score(ex["prediction"], ex["ground_truth"]), 6),
        "recomputed_exact_match": exact_match_score(ex["prediction"], ex["ground_truth"]),
    }

    zero_rows.append(zero_row)
    lora_rows.append(lora_row)

    paired_rows.append({
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
    })

assert len(zero_rows) == 90
assert len(lora_rows) == 90
assert len({r["task_id"] for r in lora_rows}) == 90

zero_stored = summarize(zero_rows, "stored_anls", "stored_exact_match")
lora_stored = summarize(lora_rows, "stored_anls", "stored_exact_match")
zero_recomputed = summarize(zero_rows, "recomputed_anls", "recomputed_exact_match")
lora_recomputed = summarize(lora_rows, "recomputed_anls", "recomputed_exact_match")

improved = sum(1 for r in paired_rows if r["stored_delta_anls"] > 0)
worse = sum(1 for r in paired_rows if r["stored_delta_anls"] < 0)
unchanged = sum(1 for r in paired_rows if r["stored_delta_anls"] == 0)

summary = {
    "analysis": "Pix2Struct LoRA domain adaptation",
    "important_note": "Separate analysis only. Do not include Pix2Struct LoRA in the main 397-question leaderboard.",
    "train_companies": lora_raw["train_companies"],
    "test_companies": lora_raw["test_companies"],
    "train_pairs": lora_raw["train_pairs"],
    "test_pairs": lora_raw["test_pairs"],
    "heldout_questions": 90,
    "stored_original_evaluator": {
        "zero_shot": zero_stored,
        "lora": lora_stored,
        "delta": {
            "anls": round(lora_stored["overall_anls"] - zero_stored["overall_anls"], 4),
            "exact_match": round(lora_stored["overall_exact_match"] - zero_stored["overall_exact_match"], 4),
            "relative_anls_change_pct": round(
                (lora_stored["overall_anls"] - zero_stored["overall_anls"])
                / zero_stored["overall_anls"] * 100,
                1,
            ),
        },
    },
    "recomputed_project_grader": {
        "zero_shot": zero_recomputed,
        "lora": lora_recomputed,
        "delta": {
            "anls": round(lora_recomputed["overall_anls"] - zero_recomputed["overall_anls"], 4),
            "exact_match": round(lora_recomputed["overall_exact_match"] - zero_recomputed["overall_exact_match"], 4),
        },
    },
    "paired_outcomes_stored_anls": {
        "improved": improved,
        "worse": worse,
        "unchanged": unchanged,
    },
    "top_improvements": sorted(paired_rows, key=lambda r: r["stored_delta_anls"], reverse=True)[:5],
    "top_regressions": sorted(paired_rows, key=lambda r: r["stored_delta_anls"])[:5],
}

for path, payload in {
    OUT_DIR / "pix2struct_zero_shot_heldout.json": zero_rows,
    OUT_DIR / "pix2struct_lora_heldout.json": lora_rows,
    OUT_DIR / "pix2struct_domain_adaptation_pairs.json": paired_rows,
    OUT_DIR / "domain_adaptation_summary.json": summary,
    REPORTS_DIR / "domain_adaptation_summary.json": summary,
}.items():
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

print("Domain adaptation analysis complete")
print(f"Train: {', '.join(summary['train_companies'])} ({summary['train_pairs']} pairs)")
print(f"Held-out test: {', '.join(summary['test_companies'])} ({summary['test_pairs']} pairs)")
print()
print("Stored original evaluator:")
print(f"  Pix2Struct zero-shot ANLS={zero_stored['overall_anls']:.4f} EM={zero_stored['overall_exact_match']:.4f}")
print(f"  Pix2Struct LoRA      ANLS={lora_stored['overall_anls']:.4f} EM={lora_stored['overall_exact_match']:.4f}")
print(f"  Delta                ANLS={summary['stored_original_evaluator']['delta']['anls']:+.4f} EM={summary['stored_original_evaluator']['delta']['exact_match']:+.4f}")
print(f"  Relative ANLS change {summary['stored_original_evaluator']['delta']['relative_anls_change_pct']:+.1f}%")
print()
print("Recomputed project grader:")
print(f"  Pix2Struct zero-shot ANLS={zero_recomputed['overall_anls']:.4f} EM={zero_recomputed['overall_exact_match']:.4f}")
print(f"  Pix2Struct LoRA      ANLS={lora_recomputed['overall_anls']:.4f} EM={lora_recomputed['overall_exact_match']:.4f}")
print(f"  Delta                ANLS={summary['recomputed_project_grader']['delta']['anls']:+.4f} EM={summary['recomputed_project_grader']['delta']['exact_match']:+.4f}")
print()
print(f"Paired outcomes by stored ANLS: improved={improved}, worse={worse}, unchanged={unchanged}")
print()
print("Category deltas, stored evaluator:")
for cat in zero_stored["by_category"]:
    z = zero_stored["by_category"][cat]
    l = lora_stored["by_category"][cat]
    print(f"  {cat}: n={z['n']} delta_anls={l['anls'] - z['anls']:+.4f} delta_em={l['exact_match'] - z['exact_match']:+.4f}")
