"""Run an adapter over the task set and write a prediction file + metadata sidecar."""
from __future__ import annotations

import hashlib
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .adapters.base import BaseAdapter
from .graders import GRADER_VERSION
from .paths import ProjectPaths, meta_path_for
from .pipeline._io import read_json, write_json

log = logging.getLogger(__name__)


def _git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_run_metadata(adapter: BaseAdapter, paths: ProjectPaths, n_tasks: int, started: datetime) -> dict:
    return {
        "model": adapter.name,
        "adapter": f"{type(adapter).__module__}:{type(adapter).__name__}",
        "adapter_metadata": adapter.metadata(),
        "n_tasks": n_tasks,
        "tasks_sha256": _sha256(paths.tasks) if paths.tasks.exists() else None,
        "started_at": started.isoformat(),
        "finished_at": None,
        "finagent_eval_version": __version__,
        "grader_version": GRADER_VERSION,
        "git_sha": _git_sha(paths.root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def make_row(task: dict, model: str, prediction: str, latency_ms: int | None, evidence: dict | None = None) -> dict:
    return {
        "task_id": task["task_id"],
        "model": model,
        "prediction": prediction,
        "gold_answer": task["gold_answer"],
        "category": task["category"],
        "latency_ms": latency_ms,
        "evidence_available": evidence is not None,
        "evidence": evidence,
        "raw_output": prediction,
    }


def run_adapter(
    adapter: BaseAdapter,
    paths: ProjectPaths,
    out_path: Path,
    limit: int | None = None,
    resume: bool = True,
    checkpoint_every: int = 25,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Predict every task with ``adapter`` and write the prediction file.

    Rows are checkpointed to ``out_path`` every ``checkpoint_every`` tasks; with
    ``resume=True`` an interrupted run continues from the last checkpoint.
    Exceptions from the adapter are recorded as empty predictions so one bad
    image cannot kill a long run (they are logged and counted in metadata).
    """
    tasks = read_json(paths.tasks)
    if limit is not None:
        tasks = tasks[:limit]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if resume and out_path.exists():
        rows = [r for r in read_json(out_path) if r.get("model") == adapter.name]
        log.info("resuming: %d rows already in %s", len(rows), out_path)
    done = {r["task_id"] for r in rows}

    started = datetime.now(timezone.utc)
    meta = build_run_metadata(adapter, paths, len(tasks), started)
    errors = 0

    adapter.setup()
    try:
        for i, task in enumerate(tasks, start=1):
            if task["task_id"] in done:
                continue
            image_path = paths.root / task["image_path"]
            t0 = time.perf_counter()
            try:
                prediction, evidence = adapter.predict_with_evidence(image_path, task["question"])
                latency = round((time.perf_counter() - t0) * 1000)
            except Exception as exc:  # noqa: BLE001 - one failure must not abort a GPU run
                errors += 1
                log.exception("adapter failed on %s: %r", task["task_id"], exc)
                prediction, evidence, latency = "", None, None
            rows.append(make_row(task, adapter.name, "" if prediction is None else str(prediction), latency, evidence))
            if progress:
                progress(i, len(tasks))
            if checkpoint_every and len(rows) % checkpoint_every == 0:
                write_json(out_path, rows)
    finally:
        adapter.teardown()

    write_json(out_path, rows)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["adapter_errors"] = errors
    write_json(meta_path_for(out_path), meta)
    log.info("wrote %d predictions to %s (%d adapter errors)", len(rows), out_path, errors)
    return rows
