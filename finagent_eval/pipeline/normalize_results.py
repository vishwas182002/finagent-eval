"""Normalize the 2024 baseline result files into the common prediction schema."""
from __future__ import annotations

import logging

from ..paths import EXPECTED_TASK_COUNT, ProjectPaths
from ._io import read_json, write_json
from .build_tasks import DatasetError

log = logging.getLogger(__name__)

BASELINES: dict[str, tuple[str, str]] = {
    "ocr_roberta": ("final_baseline_results.json", "OCR+RoBERTa"),
    "layoutlmv3": ("final_layoutlmv3_results.json", "LayoutLMv3"),
    "donut": ("final_donut_results.json", "Donut"),
    "pix2struct": ("final_pix2struct_results.json", "Pix2Struct"),
}


def task_key(ticker: str, question: str, gold: str, category: str) -> tuple[str, str, str, str]:
    return (ticker, question, gold, category)


def build_task_lookup(tasks: list[dict]) -> dict[tuple, dict]:
    lookup: dict[tuple, dict] = {}
    for t in tasks:
        key = task_key(t["ticker"], t["question"], t["gold_answer"], t["category"])
        if key in lookup:
            raise DatasetError(f"Duplicate task key: {key}")
        lookup[key] = t
    return lookup


def normalize_rows(raw_examples: list[dict], model_name: str, task_lookup: dict[tuple, dict]) -> tuple[list[dict], int]:
    results = []
    unmatched = 0
    for ex in raw_examples:
        key = task_key(ex["ticker"], ex["question"], ex["ground_truth"], ex["question_type"])
        task = task_lookup.get(key)
        if task is None:
            unmatched += 1
            log.warning("no task match for %s: %.60s...", key[0], key[1])
            continue
        results.append(
            {
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
                "raw_output": ex["prediction"],
            }
        )
    return results, unmatched


def normalize_baseline_results(paths: ProjectPaths, strict: bool = True) -> dict[str, int]:
    tasks = read_json(paths.tasks)
    task_lookup = build_task_lookup(tasks)
    matched: dict[str, int] = {}

    for out_name, (filename, model_name) in BASELINES.items():
        raw_path = paths.raw_dir / filename
        if not raw_path.exists():
            log.warning("baseline file missing, skipping: %s", raw_path)
            continue
        raw = read_json(raw_path)
        results, unmatched = normalize_rows(raw["per_example"], model_name, task_lookup)
        write_json(paths.results_dir / f"{out_name}.json", results)
        matched[model_name] = len(results)
        log.info("%s: %d/%d matched (%d unmatched) -> %s.json", model_name, len(results), len(tasks), unmatched, out_name)
        if strict and len(results) != EXPECTED_TASK_COUNT:
            raise DatasetError(f"{model_name} only matched {len(results)}/{EXPECTED_TASK_COUNT}")

    return matched
