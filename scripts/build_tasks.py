"""Convert raw dataset JSON into normalized tasks.json."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/track_a/raw/combined_financial_vqa_dataset_fixed_types.json"
OUT_PATH = PROJECT_ROOT / "data/track_a/tasks.json"

with RAW_PATH.open(encoding="utf-8") as f:
    raw = json.load(f)

tasks = []
counters = {}

for entry in raw:
    ticker = entry["ticker"]
    company = entry["company"]
    rel_image_path = f"data/track_a/images/{ticker}/{entry['image_path']}"
    abs_image_path = PROJECT_ROOT / rel_image_path
    source_section = entry.get("section", "unknown")
    image_type = entry.get("image_type", "unknown")

    if not abs_image_path.exists():
        raise FileNotFoundError(f"Missing image: {abs_image_path}")

    for q in entry["questions"]:
        counters[ticker] = counters.get(ticker, 0) + 1
        task_id = f"{ticker}_{counters[ticker]:03d}"

        tasks.append({
            "task_id": task_id,
            "track": "track_a",
            "ticker": ticker,
            "company": company,
            "image_path": rel_image_path,
            "question": q["question"],
            "gold_answer": q["answer"],
            "category": q["type"],
            "source_section": source_section,
            "image_type": image_type,
            "source": f"{company} SEC filing"
        })

with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)

cats = {}
for t in tasks:
    cats[t["category"]] = cats.get(t["category"], 0) + 1

print(f"Wrote {len(tasks)} tasks to {OUT_PATH}")
print(f"Categories: {cats}")
print(f"Unique tickers: {len(counters)}")
print(f"Sample task_ids: {tasks[0]['task_id']}, {tasks[-1]['task_id']}")

assert len(tasks) == 397
assert len(counters) == 10
assert cats == {
    "extractive": 243,
    "layout_understanding": 72,
    "numerical_reasoning": 65,
    "chart_interpretation": 17,
}
