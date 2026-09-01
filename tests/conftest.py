"""Shared fixtures: a tiny synthetic project tree so pipeline stages can run in isolation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from finagent_eval.paths import ProjectPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_paths() -> ProjectPaths:
    """The real, committed project (read-only in tests)."""
    return ProjectPaths(REPO_ROOT)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def mini_project(tmp_path: Path) -> ProjectPaths:
    """A 6-task project with two tickers, four categories and a 1x1 PNG per image."""
    paths = ProjectPaths(tmp_path)
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe\xa7V\xbd\xfa\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    raw = [
        {
            "ticker": "AAA",
            "company": "Alpha Inc",
            "image_path": "AAA_income.png",
            "section": "income_statement",
            "image_type": "financial_table",
            "questions": [
                {"question": "What was revenue in 2025?", "answer": "143,756", "type": "extractive"},
                {"question": "What was revenue in 2024?", "answer": "124,300", "type": "extractive"},
                {"question": "By what percentage did revenue grow from 2024 to 2025?", "answer": "15.7%", "type": "numerical_reasoning"},
                {"question": "Which section is below total assets?", "answer": "LIABILITIES AND STOCKHOLDERS’ EQUITY", "type": "layout_understanding"},
            ],
        },
        {
            "ticker": "BBB",
            "company": "Beta Corp",
            "image_path": "BBB_chart.png",
            "section": "stock_performance",
            "image_type": "chart",
            "questions": [
                {"question": "What type of chart is shown?", "answer": "Line chart", "type": "chart_interpretation"},
                {"question": "What was the approximate value in 2025?", "answer": "Around 190", "type": "chart_interpretation"},
            ],
        },
    ]
    for entry in raw:
        img = paths.images_dir / entry["ticker"] / entry["image_path"]
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(png)
    _write(paths.raw_dataset, raw)
    shutil.copytree(REPO_ROOT / "templates", paths.templates_dir)
    return paths


@pytest.fixture
def mini_tasks(mini_project: ProjectPaths) -> list[dict]:
    from finagent_eval.pipeline import build_tasks

    return build_tasks(mini_project, strict=False)


def prediction_rows(tasks: list[dict], model: str, predictions: dict[str, str]) -> list[dict]:
    return [
        {
            "task_id": t["task_id"],
            "model": model,
            "prediction": predictions.get(t["task_id"], ""),
            "gold_answer": t["gold_answer"],
            "category": t["category"],
            "latency_ms": 10,
            "evidence_available": False,
            "evidence": None,
            "raw_output": predictions.get(t["task_id"], ""),
        }
        for t in tasks
    ]
