"""Convert the raw annotation JSON into the normalized ``tasks.json``."""
from __future__ import annotations

import logging
from collections import Counter

from ..paths import (
    EXPECTED_CATEGORY_COUNTS,
    EXPECTED_TASK_COUNT,
    EXPECTED_TICKER_COUNT,
    ProjectPaths,
)
from ._io import read_json, write_json

log = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the dataset does not match its documented shape."""


def tasks_from_raw(raw: list[dict], paths: ProjectPaths, check_images: bool = True) -> list[dict]:
    tasks: list[dict] = []
    counters: Counter[str] = Counter()

    for entry in raw:
        ticker = entry["ticker"]
        company = entry["company"]
        rel_image_path = f"data/track_a/images/{ticker}/{entry['image_path']}"
        source_section = entry.get("section", "unknown")
        image_type = entry.get("image_type", "unknown")

        if check_images and not (paths.root / rel_image_path).exists():
            raise DatasetError(f"Missing image: {paths.root / rel_image_path}")

        for q in entry["questions"]:
            counters[ticker] += 1
            tasks.append(
                {
                    "task_id": f"{ticker}_{counters[ticker]:03d}",
                    "track": "track_a",
                    "ticker": ticker,
                    "company": company,
                    "image_path": rel_image_path,
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "category": q["type"],
                    "source_section": source_section,
                    "image_type": image_type,
                    "source": f"{company} SEC filing",
                }
            )
    return tasks


def check_dataset_shape(tasks: list[dict]) -> None:
    cats = Counter(t["category"] for t in tasks)
    tickers = {t["ticker"] for t in tasks}
    problems = []
    if len(tasks) != EXPECTED_TASK_COUNT:
        problems.append(f"expected {EXPECTED_TASK_COUNT} tasks, got {len(tasks)}")
    if len(tickers) != EXPECTED_TICKER_COUNT:
        problems.append(f"expected {EXPECTED_TICKER_COUNT} tickers, got {len(tickers)}")
    if dict(cats) != EXPECTED_CATEGORY_COUNTS:
        problems.append(f"category counts {dict(cats)} != {EXPECTED_CATEGORY_COUNTS}")
    if problems:
        raise DatasetError("; ".join(problems))


def build_tasks(paths: ProjectPaths, strict: bool = True) -> list[dict]:
    raw = read_json(paths.raw_dataset)
    tasks = tasks_from_raw(raw, paths)
    if strict:
        check_dataset_shape(tasks)
    write_json(paths.tasks, tasks)

    cats = Counter(t["category"] for t in tasks)
    log.info("Wrote %d tasks to %s", len(tasks), paths.tasks)
    log.info("Categories: %s", dict(cats))
    log.info("Unique tickers: %d", len({t["ticker"] for t in tasks}))
    return tasks
