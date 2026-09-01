"""Validate ``tasks.json`` and the prediction files in ``results/track_a``."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from ..paths import EXPECTED_TASK_COUNT, ProjectPaths
from ._io import read_json

log = logging.getLogger(__name__)

REQUIRED_RESULT_FIELDS = ("task_id", "model", "prediction", "gold_answer", "category")


class ValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    checked_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        log.error(msg)


def validate_tasks(tasks: list[dict], report: ValidationReport, expected_count: int | None = EXPECTED_TASK_COUNT) -> dict[str, dict]:
    task_ids = [t["task_id"] for t in tasks]
    dupes = [tid for tid, c in Counter(task_ids).items() if c > 1]
    if expected_count is not None and len(tasks) != expected_count:
        report.error(f"tasks.json has {len(tasks)} tasks, expected {expected_count}")
    if dupes:
        report.error(f"duplicate task_ids in tasks.json: {dupes[:10]}")
    for t in tasks:
        for key in ("task_id", "question", "gold_answer", "category", "image_path"):
            if not t.get(key):
                report.error(f"task {t.get('task_id')!r} missing field {key}")
    return {t["task_id"]: t for t in tasks}


def validate_result_rows(name: str, results: list[dict], task_by_id: dict[str, dict], report: ValidationReport) -> None:
    if not isinstance(results, list) or not results:
        report.error(f"{name}: expected a non-empty list of prediction rows")
        return

    for i, r in enumerate(results):
        missing = [k for k in REQUIRED_RESULT_FIELDS if k not in r]
        if missing:
            report.error(f"{name}: row {i} missing fields {missing}")
            return

    models = {r["model"] for r in results}
    if len(models) != 1:
        report.error(f"{name}: rows reference {len(models)} model names {sorted(models)[:5]}, expected exactly 1")

    result_ids = [r["task_id"] for r in results]
    counts = Counter(result_ids)
    dupes = [tid for tid, c in counts.items() if c > 1]
    unknown = set(result_ids) - set(task_by_id)
    orphan = set(task_by_id) - set(result_ids)

    if len(results) != len(task_by_id):
        report.error(f"{name}: has {len(results)} results, expected {len(task_by_id)}")
    if unknown:
        report.error(f"{name}: {len(unknown)} task_ids not in tasks.json (e.g. {sorted(unknown)[:3]})")
    if orphan:
        report.error(f"{name}: missing {len(orphan)} task_ids from tasks.json (e.g. {sorted(orphan)[:3]})")
    if dupes:
        report.error(f"{name}: duplicate task_ids {dupes[:10]}")

    mismatches = 0
    for r in results:
        task = task_by_id.get(r["task_id"])
        if not task:
            continue
        if r["gold_answer"] != task["gold_answer"] or r["category"] != task["category"]:
            mismatches += 1
            if mismatches <= 3:
                report.error(f"{name}: gold/category mismatch at {r['task_id']}")
        if r["prediction"] is not None and not isinstance(r["prediction"], str):
            report.error(f"{name}: prediction at {r['task_id']} is {type(r['prediction']).__name__}, expected str")
    if mismatches > 3:
        report.error(f"{name}: ... {mismatches} gold/category mismatches in total")


def validate_data(paths: ProjectPaths, strict: bool = True, expected_files: set[str] | None = None) -> ValidationReport:
    report = ValidationReport()
    tasks = read_json(paths.tasks)
    task_by_id = validate_tasks(tasks, report, EXPECTED_TASK_COUNT if strict else None)
    log.info("tasks.json: %d tasks, %d unique task_ids", len(tasks), len(task_by_id))

    result_files = paths.result_files()
    if expected_files is not None:
        names = {p.name for p in result_files}
        if expected_files - names:
            report.error(f"missing expected result files: {sorted(expected_files - names)}")
        if names - expected_files:
            report.error(f"unexpected result files in results/track_a: {sorted(names - expected_files)}")

    for path in result_files:
        before = len(report.errors)
        validate_result_rows(path.name, read_json(path), task_by_id, report)
        report.checked_files.append(path.name)
        if len(report.errors) == before:
            log.info("%s: OK", path.name)

    if report.ok:
        log.info("All validations passed.")
    elif strict:
        raise ValidationError(f"{len(report.errors)} validation error(s): " + "; ".join(report.errors[:5]))
    return report
