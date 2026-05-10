"""Generate JSON + HTML scorecards for all models and a cross-model comparison."""
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENRICHED_DIR = PROJECT_ROOT / "results/track_a/enriched"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
from finagent_eval.report import generate_scorecard, generate_comparison_table
from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)
template = env.get_template("scorecard.html.j2")
now = datetime.now().strftime("%Y-%m-%d %H:%M")

scorecards = []

for result_file in sorted(ENRICHED_DIR.glob("*.json")):
    with result_file.open() as f:
        results = json.load(f)

    sc = generate_scorecard(results)
    scorecards.append(sc)

    # Per-model JSON
    out_json = REPORTS_DIR / f"scorecard_{result_file.stem}.json"
    with out_json.open("w") as f:
        json.dump(sc, f, indent=2)

    # Per-model HTML
    html = template.render(title=sc["model"], scorecards=[sc], comparison=None, generated_at=now)
    out_html = REPORTS_DIR / f"scorecard_{result_file.stem}.html"
    out_html.write_text(html)

    print(f"{sc['model']:15s}  ANLS={sc['overall_anls']:.4f}  EM={sc['overall_exact_match']:.4f}  -> {out_json.name}, {out_html.name}")

# Cross-model comparison JSON
comparison = generate_comparison_table(scorecards)
with (REPORTS_DIR / "comparison.json").open("w") as f:
    json.dump(comparison, f, indent=2)

# Cross-model comparison HTML
html = template.render(title="Cross-Model Comparison", scorecards=scorecards, comparison=comparison, generated_at=now)
(REPORTS_DIR / "comparison.html").write_text(html)

print(f"\nGenerated {len(scorecards)} scorecards + comparison (JSON + HTML) in reports/")
