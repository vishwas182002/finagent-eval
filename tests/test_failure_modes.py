"""Test suite for semi-automatic failure mode classification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finagent_eval.failure_modes import classify_failure


def test_failure_empty_and_abstention():
    assert classify_failure("", "394,328", "extractive") == "empty_output"
    assert classify_failure("N/A", "394,328", "extractive") == "abstention"
    assert classify_failure("Unable to determine", "Products", "extractive") == "abstention"


def test_failure_format_mismatch():
    assert classify_failure("$394,328", "394,328", "extractive") == "format_mismatch"


def test_failure_numeric():
    assert classify_failure("67,478", "143,756", "extractive") == "numeric_mismatch"
    assert classify_failure("142000", "143,756", "extractive") == "manual_diagnosis_required"
    assert classify_failure("12.5%", "12.5", "extractive") == "numeric_mismatch"


def test_failure_wrong_year():
    assert classify_failure(
        "FY2024 results",
        "Products",
        "extractive",
        question="What sold most in FY2025?",
    ) == "wrong_year"
    assert classify_failure(
        "2024",
        "143,756",
        "extractive",
        question="What was revenue in 2025?",
    ) == "wrong_year"


def test_failure_hallucinated_number():
    assert classify_failure(
        "412,500",
        "394,328",
        "extractive",
        source_text="Revenue was 394,328 in 2025",
    ) == "hallucinated_number"


def test_failure_fallback():
    assert classify_failure(
        "some random text",
        "Products",
        "extractive",
    ) == "manual_diagnosis_required"
