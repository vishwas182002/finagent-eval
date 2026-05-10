"""Validate tasks.json and normalized Track A result files."""
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = PROJECT_ROOT / "data/track_a/tasks.json"
RESULTS_DIR = PROJECT_ROOT / "results/track_a"
EXPECTED_RESULT_FILES = {
    "donut.json",
    "layoutlmv3.json",
    "ocr_roberta.json",
    "pix2struct.json",
    "qwen25vl_7b_results.json",
}

with TASKS_PATH.open(encoding="utf-8") as f:
    tasks = json.load(f)

task_ids = [t["task_id"] for t in tasks]
task_id_counts = Counter(task_ids)
task_by_id = {t["task_id"]: t for t in tasks}

errors = 0

print(f"tasks.json: {len(tasks)} tasks, {len(task_by_id)} unique task_ids")

if len(tasks) != 397:
    print(f"  ERROR: tasks.json has {len(tasks)} tasks, expected 397")
    errors += 1

duplicate_task_ids = [task_id for task_id, count in task_id_counts.items() if count > 1]
if duplicate_task_ids:
    print(f"  ERROR: duplicate task_ids in tasks.json: {duplicate_task_ids[:10]}")
    errors += 1

result_files = sorted(RESULTS_DIR.glob("*.json"))
result_file_names = {path.name for path in result_files}
missing_result_files = EXPECTED_RESULT_FILES - result_file_names
unexpected_result_files = result_file_names - EXPECTED_RESULT_FILES

if missing_result_files:
    print(f"  ERROR: missing expected result files: {sorted(missing_result_files)}")
    errors += 1

if unexpected_result_files:
    print(f"  ERROR: unexpected result files in results/track_a: {sorted(unexpected_result_files)}")
    errors += 1

for result_file in result_files:
    with result_file.open(encoding="utf-8") as f:
        results = json.load(f)

    file_errors = 0
    model = results[0]["model"] if results else "unknown"

    result_ids = [r["task_id"] for r in results]
    result_id_counts = Counter(result_ids)
    result_id_set = set(result_ids)

    missing = result_id_set - set(task_by_id)
    orphan_tasks = set(task_by_id) - result_id_set
    duplicate_result_ids = [
        task_id for task_id, count in result_id_counts.items() if count > 1
    ]

    if len(results) != 397:
        print(f"  ERROR: {result_file.name} has {len(results)} results, expected 397")
        file_errors += 1

    if missing:
        print(f"  ERROR: {result_file.name} has {len(missing)} task_ids not in tasks.json")
        file_errors += 1

    if orphan_tasks:
        print(f"  ERROR: {result_file.name} missing {len(orphan_tasks)} task_ids from tasks.json")
        file_errors += 1

    if duplicate_result_ids:
        print(f"  ERROR: {result_file.name} has duplicate task_ids: {duplicate_result_ids[:10]}")
        file_errors += 1

    for r in results:
        task = task_by_id.get(r["task_id"])
        if not task:
            continue
        if r["gold_answer"] != task["gold_answer"] or r["category"] != task["category"]:
            print(f"  ERROR: {result_file.name} mismatch at {r['task_id']}")
            file_errors += 1
            break

    if file_errors == 0:
        print(f"{result_file.name} ({model}): {len(results)} results - OK")

    errors += file_errors

if errors == 0:
    print("\nAll validations passed.")
else:
    print(f"\n{errors} error(s) found!")
    raise SystemExit(1)
