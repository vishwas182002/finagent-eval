"""Generate JSON + HTML scorecards for every enriched result file and a comparison."""
from __future__ import annotations

import logging

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from ..graders import GRADER_VERSION
from ..paths import ProjectPaths, meta_path_for
from ..report import generate_comparison_table, generate_scorecard
from ._io import read_json, read_json_if_exists, write_json

log = logging.getLogger(__name__)


def _template(paths: ProjectPaths):
    env = Environment(
        loader=FileSystemLoader(str(paths.templates_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env.get_template("scorecard.html.j2")


def generate_reports(paths: ProjectPaths) -> list[dict]:
    template = _template(paths)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic footer: the same inputs must produce byte-identical reports
    # so CI can diff regenerated artifacts against the committed ones.
    stamp = f"finagent-eval {__version__} | grader v{GRADER_VERSION}"

    scorecards = []
    for enriched_file in paths.enriched_files():
        results = read_json(enriched_file)
        meta = read_json_if_exists(meta_path_for(enriched_file), {})
        sc = generate_scorecard(results, run_metadata=meta)
        scorecards.append(sc)

        write_json(paths.reports_dir / f"scorecard_{enriched_file.stem}.json", sc)
        html = template.render(title=sc["model"], scorecards=[sc], comparison=None, generated_at=stamp)
        (paths.reports_dir / f"scorecard_{enriched_file.stem}.html").write_text(html, encoding="utf-8")
        log.info("%-24s ANLS=%.4f EM=%.4f PC=%.4f", sc["model"], sc["overall_anls"], sc["overall_exact_match"], sc["overall_project_correct"])

    if not scorecards:
        log.warning("no enriched result files found in %s", paths.enriched_dir)
        return []

    comparison = generate_comparison_table(scorecards)
    write_json(paths.reports_dir / "comparison.json", comparison)
    html = template.render(title="Cross-Model Comparison", scorecards=scorecards, comparison=comparison, generated_at=stamp)
    (paths.reports_dir / "comparison.html").write_text(html, encoding="utf-8")
    log.info("Generated %d scorecards + comparison in %s", len(scorecards), paths.reports_dir)
    return scorecards
