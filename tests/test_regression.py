"""Regression tests against the committed dataset and results.

These pin the published leaderboard numbers. If a grader change moves them,
the test fails on purpose: bump GRADER_VERSION, regenerate the artifacts,
update README.md and CHANGELOG.md, then update the expectations here.
"""
import json

import pytest

from finagent_eval.graders import GRADER_VERSION, grade
from finagent_eval.paths import EXPECTED_CATEGORY_COUNTS, EXPECTED_TASK_COUNT
from finagent_eval.pipeline.enrich import enrich_rows, summarize_enriched
from finagent_eval.pipeline.reliability import regrade_row, summarize_probe
from finagent_eval.report import generate_scorecard

EXPECTED_LEADERBOARD = {
    # model: (overall ANLS, exact match, project-correct count)
    "Qwen2.5-VL-7B-Instruct": (0.7378, 0.6877, 277),
    "LayoutLMv3": (0.1533, 0.1335, 53),
    "Pix2Struct": (0.1343, 0.1134, 45),
    "OCR+RoBERTa": (0.1132, 0.0856, 34),
    "Donut": (0.1061, 0.0756, 30),
}
RESULT_FILES = {
    "Qwen2.5-VL-7B-Instruct": "qwen25vl_7b_results.json",
    "LayoutLMv3": "layoutlmv3.json",
    "Pix2Struct": "pix2struct.json",
    "OCR+RoBERTa": "ocr_roberta.json",
    "Donut": "donut.json",
}


@pytest.fixture(scope="module")
def tasks(repo_paths):
    return json.loads(repo_paths.tasks.read_text(encoding="utf-8"))


def test_grader_version_matches_pinned_numbers():
    assert GRADER_VERSION == "2.0", "grader changed: regenerate artifacts and update EXPECTED_LEADERBOARD"


def test_dataset_shape(tasks):
    assert len(tasks) == EXPECTED_TASK_COUNT
    counts = {}
    for t in tasks:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    assert counts == EXPECTED_CATEGORY_COUNTS
    assert len({t["task_id"] for t in tasks}) == EXPECTED_TASK_COUNT
    assert len({t["image_path"] for t in tasks}) == 79
    assert len({t["ticker"] for t in tasks}) == 10


def test_every_gold_answer_is_gradable_against_itself(tasks):
    """The grader must give full credit to the gold answer for every task."""
    bad = [t["task_id"] for t in tasks if not grade(t["gold_answer"], t["gold_answer"], category=t["category"]).exact_match]
    assert bad == []


@pytest.mark.parametrize("model", sorted(EXPECTED_LEADERBOARD))
def test_leaderboard_numbers_reproduce_from_raw_predictions(repo_paths, tasks, model):
    task_by_id = {t["task_id"]: t for t in tasks}
    rows = json.loads((repo_paths.results_dir / RESULT_FILES[model]).read_text(encoding="utf-8"))
    enriched = enrich_rows(rows, task_by_id)
    sc = generate_scorecard(enriched)
    anls, em, pc = EXPECTED_LEADERBOARD[model]
    assert sc["overall_anls"] == pytest.approx(anls, abs=5e-5)
    assert sc["overall_exact_match"] == pytest.approx(em, abs=5e-5)
    assert sc["project_correct_count"] == pc
    assert summarize_enriched(enriched)["failure_labels"].get("manual_diagnosis_required", 0) == 0


def test_committed_enriched_files_match_recomputation(repo_paths, tasks):
    """The committed enriched artifacts must be exactly what the current code produces."""
    task_by_id = {t["task_id"]: t for t in tasks}
    for model, name in RESULT_FILES.items():
        rows = json.loads((repo_paths.results_dir / name).read_text(encoding="utf-8"))
        committed = json.loads((repo_paths.enriched_dir / name).read_text(encoding="utf-8"))
        assert enrich_rows(rows, task_by_id) == committed, f"{model}: run `finagent-eval enrich` and commit"


def test_qwen_numerical_reasoning_collapse_headline(repo_paths, tasks):
    """The README's headline finding: Qwen is correct on 10 of 65 numerical-reasoning questions."""
    task_by_id = {t["task_id"]: t for t in tasks}
    rows = json.loads((repo_paths.results_dir / RESULT_FILES["Qwen2.5-VL-7B-Instruct"]).read_text(encoding="utf-8"))
    enriched = enrich_rows(rows, task_by_id)
    nr = [r for r in enriched if r["category"] == "numerical_reasoning"]
    assert len(nr) == 65
    assert sum(r["project_correct"] for r in nr) == 10
    sc = generate_scorecard(enriched)
    assert sc["weakest_category"] == "numerical_reasoning"
    assert sc["category_breakdown"]["numerical_reasoning"]["anls"] == pytest.approx(0.1938, abs=5e-5)


def test_reliability_probe_summary(repo_paths):
    rows = [regrade_row(r) for r in json.loads(repo_paths.reliability_results.read_text(encoding="utf-8"))]
    s = summarize_probe(rows)
    assert s["n_tasks"] == 25 and s["variants_per_task"] == 4
    assert s["answer_consistency"]["count"] == 11
    assert s["correctness_stability"]["count"] == 18
    assert s["all_variants_correct"]["count"] == 9
    assert s["by_category"]["numerical_reasoning"] == {"n": 6, "consistent": 0, "all_correct": 0, "original_correct": 1}
