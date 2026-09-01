"""Adapters with no model behind them, for tests and for exercising the runner."""
from __future__ import annotations

from pathlib import Path

from .base import BaseAdapter


class ConstantAdapter(BaseAdapter):
    """Always answers the same string. Useful as a floor / smoke test."""

    name = "constant"

    def __init__(self, answer: str = "N/A") -> None:
        self.answer = answer

    def predict(self, image_path: Path, question: str) -> str:
        return self.answer

    def metadata(self) -> dict:
        return {"answer": self.answer}


class EchoQuestionAdapter(BaseAdapter):
    """Returns the question - guaranteed wrong, deterministic, no dependencies."""

    name = "echo-question"

    def predict(self, image_path: Path, question: str) -> str:
        return question


class GoldOracleAdapter(BaseAdapter):
    """Answers from a task lookup. Used to verify the grader gives 100% on gold."""

    name = "gold-oracle"

    def __init__(self, gold_by_key: dict[tuple[str, str], str] | None = None) -> None:
        self.gold_by_key = gold_by_key or {}

    def predict(self, image_path: Path, question: str) -> str:
        return self.gold_by_key.get((str(image_path), question), "")
