"""Pipeline stages. Each stage is a plain function that takes a :class:`ProjectPaths`.

Order: build_tasks -> normalize_results -> validate -> enrich -> reports ->
domain_adaptation -> reliability. ``run_all`` executes them in sequence.
"""
from __future__ import annotations

from ..paths import ProjectPaths
from .build_tasks import build_tasks
from .domain_adaptation import analyze_domain_adaptation
from .enrich import enrich_results
from .normalize_results import normalize_baseline_results
from .reliability import regrade_reliability_probe
from .reports import generate_reports
from .validate import validate_data


def run_all(paths: ProjectPaths, strict: bool = True) -> None:
    build_tasks(paths, strict=strict)
    normalize_baseline_results(paths, strict=strict)
    validate_data(paths, strict=strict)
    enrich_results(paths)
    generate_reports(paths)
    analyze_domain_adaptation(paths)
    regrade_reliability_probe(paths)


__all__ = [
    "analyze_domain_adaptation",
    "build_tasks",
    "enrich_results",
    "generate_reports",
    "normalize_baseline_results",
    "regrade_reliability_probe",
    "run_all",
    "validate_data",
]
