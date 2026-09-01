"""Model adapters and the registry the CLI uses to find them.

Register your own adapter either by adding it to :data:`REGISTRY` or by
passing a dotted path on the command line::

    finagent-eval run --adapter my_package.my_module:MyAdapter
"""
from __future__ import annotations

import importlib

from .base import BaseAdapter
from .dummy import ConstantAdapter, EchoQuestionAdapter, GoldOracleAdapter
from .qwen_vl import QwenVLAdapter

REGISTRY: dict[str, type[BaseAdapter]] = {
    "constant": ConstantAdapter,
    "echo-question": EchoQuestionAdapter,
    "gold-oracle": GoldOracleAdapter,
    "qwen2.5-vl": QwenVLAdapter,
}


def resolve_adapter(spec: str) -> type[BaseAdapter]:
    """Look up an adapter class by registry name or ``module:Class`` path."""
    if spec in REGISTRY:
        return REGISTRY[spec]
    if ":" not in spec:
        raise KeyError(f"Unknown adapter {spec!r}. Known: {sorted(REGISTRY)} or use 'module:ClassName'.")
    module_name, class_name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseAdapter)):
        raise TypeError(f"{spec} is not a BaseAdapter subclass")
    return cls


__all__ = [
    "REGISTRY",
    "BaseAdapter",
    "ConstantAdapter",
    "EchoQuestionAdapter",
    "GoldOracleAdapter",
    "QwenVLAdapter",
    "resolve_adapter",
]
