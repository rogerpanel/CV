"""Common explainer contract.

Every explainer returns an `Explanation` object carrying (a) a per-feature
attribution vector, (b) optional supporting artefacts (attention maps,
counterfactuals, Nash strategies), and (c) a latency measurement in
milliseconds — needed for the "Cost(E) ≤ τ_max (100 ms per alert)"
constraint in Problem 1 of the manuscript.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class Explanation:
    attribution: torch.Tensor                       # (d,) per-feature importance
    method: str
    latency_ms: float
    supporting: dict[str, Any] = field(default_factory=dict)
    confidence_interval: Optional[torch.Tensor] = None
    target_node: Optional[int] = None


class BaseExplainer(ABC):
    """Abstract base class used by every technique in §3 of the paper."""

    name: str = "base"

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @abstractmethod
    def explain(self, *args, **kwargs) -> Explanation:  # noqa: D401
        """Produce an Explanation for a single alert / node."""

    # ------------------------------------------------------------------ #
    # Utility — time a sub-block
    # ------------------------------------------------------------------ #
    @staticmethod
    def _timed(func, *args, **kwargs) -> tuple[Any, float]:
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        return out, (time.perf_counter() - t0) * 1000.0
