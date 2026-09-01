"""Thin wrapper kept for backwards compatibility. Equivalent to: finagent-eval validate"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # works without pip install -e .

from finagent_eval.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["validate", "--root", str(Path(__file__).resolve().parents[1]), *sys.argv[1:]]))
