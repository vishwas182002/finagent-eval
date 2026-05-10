"""Test suite for graders: normalize, ANLS, exact match, numeric tolerance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finagent_eval.graders.anls import anls_score
from finagent_eval.graders.exact_match import exact_match_score
from finagent_eval.graders.normalize import normalize_financial_number, normalize_text
from finagent_eval.graders.numeric import numeric_match_is_correct, numeric_match_score


def test_normalize_text():
    assert normalize_text("Hello World") == "hello world"
    assert normalize_text("  spaced  ") == "spaced"
    assert normalize_text("a   b   c") == "a b c"
    assert normalize_text("") == ""
    assert normalize_text("line1\n  line2") == "line1 line2"


def test_normalize_financial_number():
    assert normalize_financial_number("$1,452,000") == "1452000"
    assert normalize_financial_number("(500)") == "-500"
    assert normalize_financial_number("($500)") == "-500"
    assert normalize_financial_number("-$1,200") == "-1200"
    assert normalize_financial_number("12.5%") == "12.5%"
    assert normalize_financial_number("1.5M") == "1500000"
    assert normalize_financial_number("2.3B") == "2300000000"
    assert normalize_financial_number("hello") is None
    assert normalize_financial_number("") is None
    assert normalize_financial_number("3.14") == "3.14"


def test_anls_score():
    assert anls_score("394,328", "394,328") == 1.0
    assert anls_score("apple", "Apple") == 1.0
    assert anls_score("", "394,328") == 0.0
    assert anls_score("394,328", "") == 0.0
    assert anls_score("", "") == 1.0
    assert round(anls_score("394328", "$394,328"), 4) == 0.75


def test_exact_match_score():
    assert exact_match_score("394,328", "$394,328") is True
    assert exact_match_score("1500000", "1.5M") is True
    assert exact_match_score("(500)", "-$500") is True
    assert exact_match_score("Products", "products") is True
    assert exact_match_score("394,328", "394,329") is False
    assert exact_match_score("", "394,328") is False


def test_numeric_match_score():
    assert numeric_match_score("143756", "143,756") is True
    assert numeric_match_score("142000", "143,756") is True
    assert numeric_match_score("100000", "143,756") is False
    assert numeric_match_score("15.5%", "15.7%") is True
    assert numeric_match_score("12%", "15.7%") is False
    assert numeric_match_score("hello", "143,756") is None
    assert numeric_match_score("0.157", "15.7%") is False


def test_numeric_match_is_correct():
    assert numeric_match_is_correct("142000", "143,756") is True
    assert numeric_match_is_correct("15.5%", "15.7%") is True
    assert numeric_match_is_correct("100000", "143,756") is False
    assert numeric_match_score("2024", "2025") is True
    assert numeric_match_is_correct("2024", "2025") is False
    assert numeric_match_is_correct("FY2024", "FY2025") is False
