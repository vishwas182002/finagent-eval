"""Enrich normalized results with recomputed grader scores and failure labels."""
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = PROJECT_ROOT / "data/track_a/tasks.json"
RESULTS_DIR = PROJECT_ROOT / "results/track_a"
OUT_DIR = PROJECT_ROOT / "results/track_a/enriched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from finagent_eval.graders.anls import anls_score
from finagent_eval.graders.exact_match import exact_match_score
from finagent_eval.graders.numeric import numeric_match_is_correct, numeric_match_score
from finagent_eval.failure_modes import classify_failure

# Load tasks for question lookup
with TASKS_PATH.open() as f:
    tasks = json.load(f)
task_by_id = {t["task_id"]: t for t in tasks}

for result_file in sorted(RESULTS_DIR.glob("*.json")):
    with result_file.open() as f:
        results = json.load(f)

    model = results[0]["model"]
    enriched = []

    for r in results:
        task = task_by_id[r["task_id"]]

        r_anls = anls_score(r["prediction"], r["gold_answer"])
        r_em = exact_match_score(r["prediction"], r["gold_answer"])
        r_num = numeric_match_score(r["prediction"], r["gold_answer"])
        r_numeric_correct = numeric_match_is_correct(r["prediction"], r["gold_answer"])
        r_accepted_by_tolerance = r_numeric_correct and not r_em
        r_project_correct = r_em or r_numeric_correct

        failure_label = None if r_project_correct else classify_failure(
            r["prediction"],
            r["gold_answer"],
            r["category"],
            question=task["question"],
        )

        enriched.append({
            **r,
            "question": task["question"],
            "ticker": task["ticker"],
            "company": task["company"],
            "image_path": task["image_path"],
            "source_section": task.get("source_section"),
            "image_type": task.get("image_type"),
            "recomputed_anls": round(r_anls, 6),
            "recomputed_exact_match": r_em,
            "numeric_match": r_num,
            "numeric_tolerance_correct": r_numeric_correct,
            "accepted_by_numeric_tolerance": r_accepted_by_tolerance,
            "project_correct": r_project_correct,
            "failure_label": failure_label,
        })

    out_path = OUT_DIR / result_file.name
    with out_path.open("w") as f:
        json.dump(enriched, f, indent=2)

    # Summary stats
    total = len(enriched)
    avg_anls = sum(r["recomputed_anls"] for r in enriched) / total
    em_count = sum(1 for r in enriched if r["recomputed_exact_match"])
    project_correct_count = sum(1 for r in enriched if r["project_correct"])
    failures = [r for r in enriched if r["failure_label"] is not None]
    label_dist = Counter(r["failure_label"] for r in failures)

    print(f"{model}:")
    print(f"  ANLS: {avg_anls:.4f}  EM: {em_count}/{total} ({100*em_count/total:.1f}%)")
    print(f"  Project-correct: {project_correct_count}/{total} ({100*project_correct_count/total:.1f}%)")
    print(f"  Failures: {len(failures)}/{total}")
    print(f"  Labels: {dict(label_dist.most_common())}")
    print()
