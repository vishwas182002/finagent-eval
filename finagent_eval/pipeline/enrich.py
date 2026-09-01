"""Enrich prediction files with recomputed grader scores and failure labels."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from ..failure_modes import classify_failure
from ..graders import GRADER_VERSION, grade
from ..paths import ProjectPaths, meta_path_for
from ._io import read_json, read_json_if_exists, write_json

log = logging.getLogger(__name__)


def enrich_rows(results: list[dict], task_by_id: dict[str, dict], source_text_by_id: dict[str, str] | None = None) -> list[dict]:
    """Attach task context, recomputed grades and a failure label to each row."""
    enriched = []
    for r in results:
        task = task_by_id[r["task_id"]]
        prediction = "" if r.get("prediction") is None else str(r["prediction"])
        g = grade(prediction, r["gold_answer"], category=r["category"])
        source_text = (source_text_by_id or {}).get(r["task_id"], "")
        failure_label = (
            None
            if g.project_correct
            else classify_failure(
                prediction,
                r["gold_answer"],
                r["category"],
                question=task["question"],
                source_text=source_text,
            )
        )
        enriched.append(
            {
                **r,
                "prediction": prediction,
                "question": task["question"],
                "ticker": task["ticker"],
                "company": task["company"],
                "image_path": task["image_path"],
                "source_section": task.get("source_section"),
                "image_type": task.get("image_type"),
                "recomputed_anls": g.anls,
                "raw_anls": g.raw_anls,
                "recomputed_exact_match": g.exact_match,
                "numeric_match": g.numeric_match,
                "numeric_tolerance_correct": g.numeric_tolerance_correct,
                "accepted_by_numeric_tolerance": g.accepted_by_numeric_tolerance,
                "project_correct": g.project_correct,
                "failure_label": failure_label,
                "grader_version": GRADER_VERSION,
            }
        )
    return enriched


def summarize_enriched(enriched: list[dict]) -> dict:
    total = len(enriched)
    failures = [r for r in enriched if r["failure_label"] is not None]
    return {
        "model": enriched[0]["model"] if enriched else None,
        "total": total,
        "anls": round(sum(r["recomputed_anls"] for r in enriched) / total, 4) if total else 0.0,
        "exact_match_count": sum(1 for r in enriched if r["recomputed_exact_match"]),
        "project_correct_count": sum(1 for r in enriched if r["project_correct"]),
        "failure_count": len(failures),
        "failure_labels": dict(Counter(r["failure_label"] for r in failures).most_common()),
    }


def enrich_file(result_file: Path, paths: ProjectPaths, task_by_id: dict[str, dict]) -> dict:
    results = read_json(result_file)
    enriched = enrich_rows(results, task_by_id)
    out_path = paths.enriched_dir / result_file.name
    write_json(out_path, enriched)

    meta = read_json_if_exists(meta_path_for(result_file))
    if meta is not None:
        write_json(meta_path_for(out_path), meta)

    summary = summarize_enriched(enriched)
    log.info(
        "%s: ANLS=%.4f EM=%d/%d project-correct=%d/%d failures=%d labels=%s",
        summary["model"], summary["anls"], summary["exact_match_count"], summary["total"],
        summary["project_correct_count"], summary["total"], summary["failure_count"], summary["failure_labels"],
    )
    return summary


def enrich_results(paths: ProjectPaths, only: list[Path] | None = None) -> list[dict]:
    tasks = read_json(paths.tasks)
    task_by_id = {t["task_id"]: t for t in tasks}
    paths.enriched_dir.mkdir(parents=True, exist_ok=True)
    files = only if only is not None else paths.result_files()
    return [enrich_file(Path(f), paths, task_by_id) for f in files]
