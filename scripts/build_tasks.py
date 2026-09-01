"""Thin wrapper kept for backwards compatibility. Equivalent to: finagent-eval build-tasks"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install -e .

from finagent_eval.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["build-tasks", "--root", str(Path(__file__).resolve().parents[1]), *sys.argv[1:]]))
