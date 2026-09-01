"""Project layout helpers.

All pipeline functions take an explicit ``root`` so they can be pointed at a
temporary directory in tests. ``find_project_root`` resolves the default from
``FINAGENT_EVAL_ROOT``, then by walking up from the current directory looking
for ``data/track_a``, then falls back to the repository that contains this
package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

EXPECTED_TASK_COUNT = 397
EXPECTED_CATEGORY_COUNTS = {
    "extractive": 243,
    "layout_understanding": 72,
    "numerical_reasoning": 65,
    "chart_interpretation": 17,
}
EXPECTED_TICKER_COUNT = 10

# Result files are lists of prediction rows; run metadata lives next to them in
# ``<name>.meta.json`` so the row schema stays flat and backwards compatible.
META_SUFFIX = ".meta.json"


def is_meta_file(path: Path) -> bool:
    return path.name.endswith(META_SUFFIX)


def meta_path_for(result_path: Path) -> Path:
    return result_path.with_name(result_path.stem + META_SUFFIX)


def find_project_root(start: Path | None = None) -> Path:
    env = os.environ.get("FINAGENT_EVAL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "data" / "track_a").is_dir():
            return candidate
    return REPO_ROOT


@dataclass(frozen=True)
class ProjectPaths:
    """Every file the pipeline reads or writes, derived from one root."""

    root: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> ProjectPaths:
        return cls(Path(root).resolve() if root else find_project_root())

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / "track_a"

    @property
    def raw_dataset(self) -> Path:
        return self.data_dir / "raw" / "combined_financial_vqa_dataset_fixed_types.json"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def tasks(self) -> Path:
        return self.data_dir / "tasks.json"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def results_dir(self) -> Path:
        return self.root / "results" / "track_a"

    @property
    def enriched_dir(self) -> Path:
        return self.results_dir / "enriched"

    @property
    def reliability_results(self) -> Path:
        return self.root / "results" / "reliability" / "reliability_probe_results.json"

    @property
    def reliability_summary(self) -> Path:
        return self.root / "results" / "reliability" / "reliability_probe_summary.json"

    @property
    def domain_adaptation_dir(self) -> Path:
        return self.root / "results" / "domain_adaptation"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def templates_dir(self) -> Path:
        return self.root / "templates"

    def result_files(self) -> list[Path]:
        """Prediction files in ``results/track_a`` (metadata sidecars excluded)."""
        if not self.results_dir.is_dir():
            return []
        return sorted(p for p in self.results_dir.glob("*.json") if not is_meta_file(p))

    def enriched_files(self) -> list[Path]:
        if not self.enriched_dir.is_dir():
            return []
        return sorted(p for p in self.enriched_dir.glob("*.json") if not is_meta_file(p))
