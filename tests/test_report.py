"""Tests for scorecard generation and the bootstrap confidence interval."""
import statistics

import pytest

from finagent_eval.report import bootstrap_ci, generate_comparison_table, generate_scorecard


def _row(task_id, category, anls, em, pc, label=None, model="M"):
    return {
        "task_id": task_id,
        "model": model,
        "category": category,
        "gold_answer": "g",
        "prediction": "p",
        "recomputed_anls": anls,
        "raw_anls": anls,
        "recomputed_exact_match": em,
        "project_correct": pc,
        "failure_label": label,
        "latency_ms": 100,
    }


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    values = [0.0] * 40 + [1.0] * 60
    lo, hi = bootstrap_ci(values)
    assert (lo, hi) == bootstrap_ci(values)  # same seed => same interval
    assert lo <= statistics.mean(values) <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_shrinks_with_more_data_and_is_degenerate_for_constants():
    small = bootstrap_ci([0.0, 1.0] * 5)
    large = bootstrap_ci([0.0, 1.0] * 500)
    assert (large[1] - large[0]) < (small[1] - small[0])
    assert bootstrap_ci([0.5] * 20) == (0.5, 0.5)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_generate_scorecard_fields_and_counts():
    rows = [
        _row("A_001", "extractive", 1.0, True, True),
        _row("A_002", "extractive", 0.0, False, False, "numeric_mismatch"),
        _row("A_003", "numerical_reasoning", 0.5, False, True),
        _row("A_004", "numerical_reasoning", 0.0, False, False, "numeric_mismatch"),
        _row("A_005", "chart_interpretation", 0.0, False, False, "text_mismatch"),
    ]
    sc = generate_scorecard(rows, run_metadata={"prompt": "x"})
    assert sc["model"] == "M"
    assert sc["total_questions"] == 5
    assert sc["overall_anls"] == pytest.approx(0.3)
    assert sc["overall_exact_match"] == pytest.approx(0.2)
    assert sc["overall_project_correct"] == pytest.approx(0.4)
    assert sc["project_correct_count"] == 2
    assert sc["total_failures"] == 3
    assert sc["failure_mode_distribution"] == {"numeric_mismatch": 2, "text_mismatch": 1}
    assert sc["weakest_category"] == "chart_interpretation"
    assert sc["category_breakdown"]["extractive"]["n"] == 2
    assert sc["category_breakdown"]["extractive"]["project_correct_count"] == 1
    assert sc["run_metadata"] == {"prompt": "x"}
    assert sc["mean_latency_ms"] == 100
    assert len(sc["top_5_worst_failures"]) == 3
    assert sc["grader_version"]
    with pytest.raises(ValueError):
        generate_scorecard([])


def test_comparison_table_sorted_by_anls_and_tolerates_missing_categories():
    a = generate_scorecard([_row("A_001", "extractive", 0.2, False, False, "text_mismatch", model="A")])
    b = generate_scorecard([
        _row("B_001", "extractive", 0.9, True, True, model="B"),
        _row("B_002", "chart_interpretation", 0.7, True, True, model="B"),
    ])
    table = generate_comparison_table([a, b])
    assert [m["model"] for m in table["models"]] == ["B", "A"]
    assert table["categories"] == ["extractive", "chart_interpretation"]
    assert table["models"][1]["chart_interpretation_anls"] is None
