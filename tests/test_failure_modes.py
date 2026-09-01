"""Test suite for semi-automatic failure mode classification."""
import pytest

from finagent_eval.failure_modes import FAILURE_LABELS, classify_failure, is_abstention


def test_failure_empty_and_abstention():
    assert classify_failure("", "394,328", "extractive") == "empty_output"
    assert classify_failure("   ", "394,328", "extractive") == "empty_output"
    assert classify_failure("N/A", "394,328", "extractive") == "abstention"
    assert classify_failure("Unable to determine", "Products", "extractive") == "abstention"
    assert classify_failure("The image does not contain that information.", "37,855", "extractive") == "abstention"
    assert classify_failure("I cannot determine the value from the chart", "190", "chart_interpretation") == "abstention"
    assert not is_abstention("Products")
    assert not is_abstention("1,200")


def test_failure_format_mismatch_only_for_stricter_graders():
    # Under the project grader these are correct, so the label is never assigned by
    # the pipeline; the classifier still reports it for callers with stricter rules.
    assert classify_failure("$394,328", "394,328", "extractive") == "format_mismatch"
    assert classify_failure("$85,269 million", "85,269", "extractive") == "format_mismatch"


def test_failure_numeric_labels():
    assert classify_failure("67,478", "143,756", "extractive") == "numeric_mismatch"
    assert classify_failure("10%", "15.7%", "numerical_reasoning") == "numeric_mismatch"
    assert classify_failure("142000", "143,756", "extractive") == "near_miss_numeric"
    assert classify_failure("37,586", "37,855", "extractive") == "near_miss_numeric"
    assert classify_failure("12.5%", "12.5", "extractive") == "unit_mismatch"
    assert classify_failure("10.1%", "722", "numerical_reasoning") == "unit_mismatch"
    assert classify_failure("1.5 million", "1.5 billion", "extractive") == "unit_mismatch"
    assert classify_failure("39,371", "(39,371)", "extractive") == "sign_mismatch"
    assert classify_failure("-$500", "500", "extractive") == "sign_mismatch"


def test_failure_wrong_year():
    assert classify_failure("FY2024 results", "Products", "extractive", question="What sold most in FY2025?") == "wrong_year"
    assert classify_failure("2024", "143,756", "extractive", question="What was revenue in 2025?") == "wrong_year"
    # Gold is a year and the model named a different one, question has no year.
    assert classify_failure("2024", "2023", "extractive", question="Which year had the highest net earnings?") == "wrong_year"
    # A number that merely looks like a year inside a longer figure is not a year.
    assert classify_failure("12,024", "143,756", "extractive", question="What was revenue in 2025?") == "numeric_mismatch"


def test_failure_hallucinated_number_needs_source_text():
    assert classify_failure("412,500", "394,328", "extractive", source_text="Revenue was 394,328 in 2025") == "hallucinated_number"
    # Present in the source (even with a unit word) => not hallucinated, just wrong.
    assert classify_failure("$412,500 million", "394,328", "extractive", source_text="Costs 412,500; revenue 394,328") == "numeric_mismatch"
    # Without source text the label is unreachable.
    assert classify_failure("412,500", "394,328", "extractive") == "numeric_mismatch"


def test_failure_text_labels():
    assert classify_failure("Net product sales", "Net service sales", "extractive") == "text_mismatch"
    assert classify_failure("Daily VaR", "Daily VaR (Value at Risk)", "extractive") == "partial_text_match"
    assert classify_failure("First Half", "First half of the year", "chart_interpretation") == "partial_text_match"
    assert classify_failure("Yes", "4,396", "numerical_reasoning") == "type_mismatch"
    assert classify_failure("4,396", "Total liabilities", "layout_understanding") == "type_mismatch"
    assert classify_failure("some random text", "Products", "extractive") == "text_mismatch"


def test_label_set_is_the_public_contract():
    import finagent_eval.failure_modes as fm

    assert len(FAILURE_LABELS) == len(set(FAILURE_LABELS))
    for label in FAILURE_LABELS:
        assert f"``{label}``" in fm.__doc__, f"{label} is not documented in the module docstring"


@pytest.mark.parametrize(
    "pred,gold,category",
    [
        ("$28,689 million", "29,972", "extractive"),
        ("Third Quarter 2025", "Second Quarter 2025", "chart_interpretation"),
        ("Board of Directors", "Corporate Oversight", "extractive"),
        ("120 million", "Around 115 million", "numerical_reasoning"),
    ],
)
def test_classifier_always_returns_a_known_label(pred, gold, category):
    assert classify_failure(pred, gold, category, question="What was it in 2025?") in FAILURE_LABELS
