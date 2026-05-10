"""Normalize baseline result files into common schema with task_id linkage."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = PROJECT_ROOT / "data/track_a/tasks.json"
RAW_DIR = PROJECT_ROOT / "data/track_a/raw"
OUT_DIR = PROJECT_ROOT / "results/track_a"

with TASKS_PATH.open() as f:
    tasks = json.load(f)

# Build lookup: (ticker, question, gold_answer, category) -> task
task_lookup = {}
for t in tasks:
    key = (t["ticker"], t["question"], t["gold_answer"], t["category"])
    assert key not in task_lookup, f"Duplicate key: {key}"
    task_lookup[key] = t

# Baseline files to normalize
baselines = {
    "ocr_roberta": ("final_baseline_results.json", "OCR+RoBERTa"),
    "layoutlmv3": ("final_layoutlmv3_results.json", "LayoutLMv3"),
    "donut": ("final_donut_results.json", "Donut"),
    "pix2struct": ("final_pix2struct_results.json", "Pix2Struct"),
}

for out_name, (filename, model_name) in baselines.items():
    raw_path = RAW_DIR / filename
    with raw_path.open() as f:
        raw = json.load(f)

    results = []
    matched = 0

    for ex in raw["per_example"]:
        key = (
            ex["ticker"],
            ex["question"],
            ex["ground_truth"],
            ex["question_type"],
        )
        task = task_lookup.get(key)

        if task is None:
            print(f"  WARNING: no task match for {key[0]}: {key[1][:60]}...")
            continue

        matched += 1
        results.append({
            "task_id": task["task_id"],
            "model": model_name,
            "prediction": ex["prediction"],
            "gold_answer": ex["ground_truth"],
            "category": ex["question_type"],
            "anls": ex["anls"],
            "exact_match": bool(ex["exact_match"]),
            "latency_ms": None,
            "evidence_available": False,
            "evidence": None,
            "raw_output": ex["prediction"]
        })

    out_path = OUT_DIR / f"{out_name}.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"{model_name}: {matched}/397 matched -> {out_path.name}")
    assert matched == 397, f"{model_name} only matched {matched}/397"

print("\nAll 4 baselines normalized successfully.")
