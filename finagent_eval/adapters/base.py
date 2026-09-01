"""Adapter interface: the only thing a model needs to implement to be evaluated."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAdapter(ABC):
    """Wraps one model so the runner can call it uniformly.

    Subclasses set ``name`` (written into every prediction row as ``model``) and
    implement :meth:`predict`. Override :meth:`setup` for lazy, one-time model
    loading and :meth:`metadata` to record prompts, decoding parameters,
    quantization, checkpoints - anything needed to reproduce the run.
    """

    name: str = "unnamed-adapter"

    def setup(self) -> None:  # noqa: B027 - intentionally optional hook
        """Load weights / clients. Called once before the first prediction."""

    @abstractmethod
    def predict(self, image_path: Path, question: str) -> str:
        """Return the model's final answer string for one document image + question."""

    def predict_with_evidence(self, image_path: Path, question: str) -> tuple[str, dict | None]:
        """Optionally return evidence (bounding boxes, cited text). Default: none."""
        return self.predict(image_path, question), None

    def metadata(self) -> dict:
        """Reproducibility metadata recorded in the ``.meta.json`` sidecar."""
        return {}

    def teardown(self) -> None:  # noqa: B027 - intentionally optional hook
        """Release resources after the run."""
