"""Headless smoke test of the Streamlit dashboard against the committed artifacts."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import REPO_ROOT  # noqa: E402


@pytest.mark.slow
def test_dashboard_renders_without_exceptions():
    at = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 7
    assert at.dataframe  # leaderboard rendered
