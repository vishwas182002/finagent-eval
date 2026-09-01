"""End-to-end pipeline tests on a synthetic mini project, plus the adapter runner and CLI."""
import json
from pathlib import Path

import pytest

from finagent_eval import pipeline
from finagent_eval.adapters import ConstantAdapter, GoldOracleAdapter, resolve_adapter
from finagent_eval.adapters.base import BaseAdapter
from finagent_eval.cli import main
from finagent_eval.paths import ProjectPaths, meta_path_for
from finagent_eval.pipeline.build_tasks import DatasetError
from finagent_eval.pipeline.validate import ValidationError
from finagent_eval.runner import run_adapter
from tests.conftest import prediction_rows


def test_build_tasks_assigns_stable_ids(mini_project, mini_tasks):
    assert [t["task_id"] for t in mini_tasks] == ["AAA_001", "AAA_002", "AAA_003", "AAA_004", "BBB_001", "BBB_002"]
    assert mini_tasks[0]["image_path"] == "data/track_a/images/AAA/AAA_income.png"
    assert mini_project.tasks.exists()
    with pytest.raises(DatasetError):
        pipeline.build_tasks(mini_project, strict=True)  # 6 tasks != 397


def test_validate_catches_bad_result_files(mini_project, mini_tasks):
    rows = prediction_rows(mini_tasks, "Good", {"AAA_001": "143,756"})
    (mini_project.results_dir).mkdir(parents=True)
    (mini_project.results_dir / "good.json").write_text(json.dumps(rows))
    assert pipeline.validate_data(mini_project, strict=False).ok

    bad = rows[:-1] + [{**rows[-1], "task_id": "ZZZ_999", "gold_answer": "nope"}]
    (mini_project.results_dir / "bad.json").write_text(json.dumps(bad))
    report = pipeline.validate_data(mini_project, strict=False)
    assert not report.ok
    assert any("ZZZ_999" in e or "missing 1 task_ids" in e for e in report.errors)
    with pytest.raises(ValidationError):
        pipeline.validate_data(mini_project, strict=True)


def test_enrich_and_report_end_to_end(mini_project, mini_tasks):
    rows = prediction_rows(
        mini_tasks,
        "Toy",
        {
            "AAA_001": "$143,756 million",  # correct via unit-aware exact match
            "AAA_002": "124,900",           # near miss on an extractive number => wrong
            "AAA_003": "16%",               # within 2% on numerical reasoning => correct
            "AAA_004": "LIABILITIES AND STOCKHOLDERS' EQUITY",  # curly vs straight quote
            "BBB_001": "Bar chart",
            "BBB_002": "$190",
        },
    )
    mini_project.results_dir.mkdir(parents=True)
    (mini_project.results_dir / "toy.json").write_text(json.dumps(rows))
    meta_path_for(mini_project.results_dir / "toy.json").write_text(json.dumps({"prompt_template": "Q: {question}"}))

    summaries = pipeline.enrich_results(mini_project)
    assert summaries[0]["project_correct_count"] == 4
    assert summaries[0]["failure_labels"] == {"near_miss_numeric": 1, "text_mismatch": 1}

    enriched = json.loads((mini_project.enriched_dir / "toy.json").read_text())
    by_id = {r["task_id"]: r for r in enriched}
    assert by_id["AAA_001"]["project_correct"] and by_id["AAA_001"]["raw_anls"] == 0.0
    assert by_id["AAA_003"]["accepted_by_numeric_tolerance"]
    assert by_id["AAA_002"]["failure_label"] == "near_miss_numeric"
    assert by_id["AAA_001"]["question"] == "What was revenue in 2025?"
    assert meta_path_for(mini_project.enriched_dir / "toy.json").exists()

    scorecards = pipeline.generate_reports(mini_project)
    assert scorecards[0]["run_metadata"]["prompt_template"] == "Q: {question}"
    assert (mini_project.reports_dir / "scorecard_toy.json").exists()
    html = (mini_project.reports_dir / "comparison.html").read_text()
    assert "Toy" in html and "grader v" in html
    # Regenerating must be byte-identical (no timestamps) so CI can diff artifacts.
    pipeline.generate_reports(mini_project)
    assert (mini_project.reports_dir / "comparison.html").read_text() == html


class _ExplodingAdapter(BaseAdapter):
    name = "boom"

    def predict(self, image_path: Path, question: str) -> str:
        if question == "What was revenue in 2024?":
            raise RuntimeError("cuda OOM")
        return "42"


def test_runner_writes_rows_metadata_and_survives_adapter_errors(mini_project, mini_tasks):
    out = mini_project.results_dir / "boom.json"
    rows = run_adapter(_ExplodingAdapter(), mini_project, out, checkpoint_every=2)
    assert len(rows) == 6
    assert {r["task_id"] for r in rows} == {t["task_id"] for t in mini_tasks}
    assert rows[1]["prediction"] == "" and rows[1]["latency_ms"] is None
    meta = json.loads(meta_path_for(out).read_text())
    assert meta["model"] == "boom" and meta["adapter_errors"] == 1
    assert meta["n_tasks"] == 6 and meta["grader_version"] and meta["tasks_sha256"]
    # The file passes validation and can be enriched like any other result file.
    assert pipeline.validate_data(mini_project, strict=False).ok
    pipeline.enrich_results(mini_project)


def test_runner_resume_and_limit(mini_project, mini_tasks):
    out = mini_project.results_dir / "const.json"
    first = run_adapter(ConstantAdapter("N/A"), mini_project, out, limit=2)
    assert len(first) == 2
    calls = []

    class Counting(ConstantAdapter):
        def predict(self, image_path, question):
            calls.append(question)
            return super().predict(image_path, question)

    Counting.name = "constant"
    resumed = run_adapter(Counting("N/A"), mini_project, out)
    assert len(resumed) == 6 and len(calls) == 4  # only the 4 missing tasks were predicted


def test_gold_oracle_scores_perfectly(mini_project, mini_tasks):
    oracle = GoldOracleAdapter({(str(mini_project.root / t["image_path"]), t["question"]): t["gold_answer"] for t in mini_tasks})
    run_adapter(oracle, mini_project, mini_project.results_dir / "oracle.json")
    summary = pipeline.enrich_results(mini_project)[0]
    assert summary["project_correct_count"] == 6 and summary["anls"] == 1.0


def test_resolve_adapter_registry_and_dotted_path():
    assert resolve_adapter("constant") is ConstantAdapter
    assert resolve_adapter("finagent_eval.adapters.dummy:ConstantAdapter") is ConstantAdapter
    with pytest.raises(KeyError):
        resolve_adapter("does-not-exist")
    with pytest.raises(TypeError):
        resolve_adapter("pathlib:Path")


def test_cli_grade_and_run(mini_project, mini_tasks, capsys):
    assert main(["grade", "$85,269 million", "85,269", "--category", "extractive"]) == 0
    assert json.loads(capsys.readouterr().out)["project_correct"] is True
    # CLI grading must agree with the leaderboard: tolerance is category-aware.
    assert main(["grade", "37586", "37855", "--category", "extractive"]) == 0
    assert json.loads(capsys.readouterr().out)["project_correct"] is False
    assert main(["grade", "37586", "37855", "--category", "chart_interpretation"]) == 0
    assert json.loads(capsys.readouterr().out)["project_correct"] is True
    with pytest.raises(SystemExit):
        main(["grade", "1", "1"])  # --category is required

    root = str(mini_project.root)
    out = mini_project.results_dir / "cli.json"
    assert main(["run", "--root", root, "--adapter", "constant", "--adapter-arg", "answer=Line chart", "--out", str(out), "--then-enrich"]) == 0
    assert (mini_project.reports_dir / "scorecard_cli.json").exists()
    assert main(["validate", "--root", root, "--no-strict"]) == 0
    assert main(["validate", "--root", root]) == 1  # strict: 6 tasks != 397
    (mini_project.results_dir / "broken.json").write_text("[]")
    assert main(["validate", "--root", root, "--no-strict"]) == 1


def test_project_paths_ignores_meta_sidecars(tmp_path):
    p = ProjectPaths(tmp_path)
    p.results_dir.mkdir(parents=True)
    (p.results_dir / "a.json").write_text("[]")
    (p.results_dir / "a.meta.json").write_text("{}")
    assert [f.name for f in p.result_files()] == ["a.json"]
