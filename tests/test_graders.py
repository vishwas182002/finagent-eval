"""Test suite for graders: normalize, ANLS, exact match, numeric tolerance, grade()."""
import pytest

from finagent_eval.graders import (
    TOLERANCE_CATEGORIES,
    anls_score,
    exact_match_score,
    financial_numbers_equivalent,
    grade,
    normalize_answer_text,
    normalize_financial_number,
    normalize_text,
    numeric_match_is_correct,
    numeric_match_score,
    parse_financial_number,
)


# --------------------------------------------------------------------------- text
def test_normalize_text():
    assert normalize_text("Hello World") == "hello world"
    assert normalize_text("  spaced  ") == "spaced"
    assert normalize_text("a   b   c") == "a b c"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""
    assert normalize_text("line1\n  line2") == "line1 line2"


def test_normalize_text_unicode_and_trailing_punctuation():
    assert normalize_text("STOCKHOLDERS’ EQUITY") == normalize_text("STOCKHOLDERS' EQUITY")
    assert normalize_text("“Products”") == '"products"'
    assert normalize_text("First half.") == "first half"
    assert normalize_text("2024–2025") == "2024-2025"


def test_normalize_answer_text_strips_prefixes_and_articles():
    assert normalize_answer_text("The Goldman Sachs Group, Inc.") == "goldman sachs group, inc"
    assert normalize_answer_text("Answer: Products") == "products"
    assert normalize_answer_text("The answer is iPhone.") == "iphone"
    assert normalize_answer_text('"Americas"') == "americas"


# ------------------------------------------------------------------------ numbers
@pytest.mark.parametrize(
    "text,expected",
    [
        ("$1,452,000", "1452000"),
        ("(500)", "-500"),
        ("($500)", "-500"),
        ("-$1,200", "-1200"),
        ("12.5%", "12.5%"),
        ("12%", "12%"),
        ("1.5M", "1500000"),
        ("2.3B", "2300000000"),
        ("3.14", "3.14"),
        ("hello", None),
        ("", None),
        (None, None),
        ("FY2025", None),
        ("Q3", None),
        ("$85,269 million", "85269000000"),
        ("$53.8 billion", "53800000000"),
        ("115 million", "115000000"),
        ("Around 115 million", "115000000"),
        ("approximately $190", "190"),
        ("12.5 percent", "12.5%"),
        ("1,062 USD", "1062"),
        ("$4,316 million dollars", "4316000000"),
        ("US$ 3.2 bn", "3200000000"),
        ("+7.5%", "7.5%"),
        ("−39,371", "-39371"),  # unicode minus
        ("Answer: 143,756", "143756"),
        ("1,234.", "1234"),
        ("%", None),
        ("$", None),
        ("2024-2025", None),
    ],
)
def test_normalize_financial_number(text, expected):
    assert normalize_financial_number(text) == expected


def test_parse_financial_number_structure():
    n = parse_financial_number("$85,269 million")
    assert n is not None
    assert (n.value, n.scale, n.has_unit, n.is_percent) == (85269.0, 1_000_000, True, False)
    plain = parse_financial_number("85,269")
    assert (plain.value, plain.scale, plain.has_unit) == (85269.0, 1, False)
    pct = parse_financial_number("(12.5%)")
    assert pct.value == -12.5 and pct.is_percent


def test_financial_numbers_equivalent_unit_rules():
    p = parse_financial_number
    # implicit table unit on one side
    assert financial_numbers_equivalent(p("$85,269 million"), p("85,269"))
    assert financial_numbers_equivalent(p("85,269"), p("$85,269 million"))
    # both explicit: must agree after scaling
    assert financial_numbers_equivalent(p("1.5 billion"), p("1,500 million"))
    assert not financial_numbers_equivalent(p("1.5 billion"), p("1.5 million"))
    # neither explicit: plain equality
    assert financial_numbers_equivalent(p("1500000"), p("1,500,000"))
    assert not financial_numbers_equivalent(p("1500000"), p("1,500"))
    # percent never equals non-percent
    assert not financial_numbers_equivalent(p("12.5%"), p("12.5"))
    # tolerance is relative to gold (second argument)
    assert financial_numbers_equivalent(p("102"), p("100"), tolerance=0.02)
    assert not financial_numbers_equivalent(p("103"), p("100"), tolerance=0.02)
    assert financial_numbers_equivalent(p("0"), p("0"), tolerance=0.02)
    assert not financial_numbers_equivalent(p("1"), p("0"), tolerance=0.02)


# --------------------------------------------------------------------------- ANLS
def test_anls_score_text_behaviour():
    assert anls_score("394,328", "394,328") == 1.0
    assert anls_score("apple", "Apple") == 1.0
    assert anls_score("", "394,328") == 0.0
    assert anls_score("394,328", "") == 0.0
    assert anls_score("", "") == 1.0
    assert anls_score("Net product sales", "Net service sales") > 0.5
    assert anls_score("completely different", "Products") == 0.0


def test_anls_score_numeric_aware_vs_raw():
    # v1 behaviour (plain DocVQA text metric) is still available
    assert round(anls_score("394328", "$394,328", numeric_aware=False), 4) == 0.75
    assert anls_score("$85,269 million", "85,269", numeric_aware=False) == 0.0
    # v2 default: equivalent numbers score 1.0
    assert anls_score("394328", "$394,328") == 1.0
    assert anls_score("$85,269 million", "85,269") == 1.0
    # non-equivalent numbers fall back to the text metric, never a free 1.0
    assert anls_score("10%", "15.7%") < 1.0
    assert anls_score("$28,689 million", "29,972") == 0.0


# -------------------------------------------------------------------- exact match
@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("394,328", "$394,328", True),
        ("1500000", "1.5M", True),
        ("(500)", "-$500", True),
        ("Products", "products", True),
        ("394,328", "394,329", False),
        ("", "394,328", False),
        ("$85,269 million", "85,269", True),
        ("$53.8 billion", "53.8", True),
        ("$190", "Around 190", True),
        ("STOCKHOLDERS' EQUITY", "STOCKHOLDERS’ EQUITY", True),
        ("The Goldman Sachs Group, Inc.", "Goldman Sachs Group, Inc.", True),
        ("Line chart.", "Line chart", True),
        ("12.5%", "12.5", False),
        ("39,371", "(39,371)", False),
        ("Net product sales", "Net service sales", False),
        ("Yes", "4,396", False),
        ("2024", "2025", False),
        ("Line graph", "Line chart", False),
    ],
)
def test_exact_match_score(pred, gold, expected):
    assert exact_match_score(pred, gold) is expected


# --------------------------------------------------------------- numeric tolerance
def test_numeric_match_score():
    assert numeric_match_score("143756", "143,756") is True
    assert numeric_match_score("142000", "143,756") is True
    assert numeric_match_score("100000", "143,756") is False
    assert numeric_match_score("15.5%", "15.7%") is True
    assert numeric_match_score("12%", "15.7%") is False
    assert numeric_match_score("hello", "143,756") is None
    assert numeric_match_score("0.157", "15.7%") is False
    assert numeric_match_score("$4,316 million", "4,396") is True


def test_numeric_match_is_correct_excludes_years():
    assert numeric_match_is_correct("142000", "143,756") is True
    assert numeric_match_is_correct("15.5%", "15.7%") is True
    assert numeric_match_is_correct("100000", "143,756") is False
    assert numeric_match_score("2024", "2025") is True
    assert numeric_match_is_correct("2024", "2025") is False
    assert numeric_match_is_correct("FY2024", "FY2025") is False


# ------------------------------------------------------------------------- grade()
def test_grade_project_correct_definition():
    g = grade("$85,269 million", "85,269", category="extractive")
    assert g.exact_match and g.project_correct and g.anls == 1.0 and g.raw_anls == 0.0
    assert not g.accepted_by_numeric_tolerance

    g = grade("23%", "23.3%", category="numerical_reasoning")
    assert not g.exact_match and g.project_correct and g.accepted_by_numeric_tolerance


def test_grade_tolerance_is_category_aware():
    assert TOLERANCE_CATEGORIES == {"numerical_reasoning", "chart_interpretation"}
    # A misread digit on a verbatim extractive number is wrong ...
    g = grade("37,586", "37,855", category="extractive")
    assert g.numeric_match is True and not g.project_correct
    # ... but the same closeness on a computed/chart answer is accepted.
    assert grade("37,586", "37,855", category="chart_interpretation").project_correct
    # No category => tolerance applies (backwards compatible with v1 callers).
    assert grade("37,586", "37,855").project_correct


def test_grade_handles_none_prediction():
    g = grade(None, "143,756", category="extractive")
    assert g.anls == 0.0 and not g.project_correct
    assert set(g.to_dict()) >= {"anls", "raw_anls", "exact_match", "project_correct"}
