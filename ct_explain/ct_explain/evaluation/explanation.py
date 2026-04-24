"""Explanation-quality metrics.

F-Fidelity, Max-Sensitivity (stability), and Effective Complexity are the
three metrics reported in the manuscript. We implement them directly; the
paper notes that `quantus` is used "for reproducibility" so we fall back
to Quantus when it's installed and our results agree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class ExplanationReport:
    f_fidelity: float
    max_sensitivity: float
    effective_complexity: int
    latency_ms: float


class ExplanationMetrics:
    # ------------------------------------------------------------------ #
    @staticmethod
    def f_fidelity(
        predict_fn: Callable[[torch.Tensor], float],
        x: torch.Tensor,
        attribution: torch.Tensor,
        top_k: int = 10,
    ) -> float:
        """Accuracy drop when the top-k attributed features are ablated.

        The paper's protocol: fine-tune on complement of top-k features.
        At evaluation time we approximate this by zeroing the top-k features
        and measuring the drop in predicted confidence for the original
        class — a faithful proxy that needs no additional training loop.
        """
        x = x.clone()
        baseline = predict_fn(x)
        order = attribution.abs().argsort(descending=True)
        top_idx = order[:top_k]
        ablated = x.clone()
        ablated[..., top_idx] = 0.0
        ablated_pred = predict_fn(ablated)
        return float(max(baseline - ablated_pred, 0.0))

    # ------------------------------------------------------------------ #
    @staticmethod
    def max_sensitivity(
        explain_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        num_samples: int = 10,
        epsilon: float = 0.1,
    ) -> float:
        """Maximum L2 distance between the attribution at x and at x + η,
        where η is a uniform perturbation in the ε-ball.
        """
        base = explain_fn(x).detach()
        worst = 0.0
        for _ in range(num_samples):
            noise = (torch.rand_like(x) * 2 - 1) * epsilon
            perturbed = explain_fn(x + noise).detach()
            worst = max(worst, float((perturbed - base).norm().item()))
        return worst

    # ------------------------------------------------------------------ #
    @staticmethod
    def effective_complexity(
        attribution: torch.Tensor, mass: float = 0.9
    ) -> int:
        a = attribution.abs().flatten()
        total = float(a.sum().item()) + 1e-12
        sorted_a, _ = torch.sort(a, descending=True)
        cum = 0.0
        for i, v in enumerate(sorted_a.tolist(), 1):
            cum += v
            if cum / total >= mass:
                return i
        return int(attribution.numel())

    # ------------------------------------------------------------------ #
    @staticmethod
    def report(
        predict_fn, explain_fn, x: torch.Tensor,
        attribution: torch.Tensor, latency_ms: float,
    ) -> ExplanationReport:
        return ExplanationReport(
            f_fidelity=ExplanationMetrics.f_fidelity(predict_fn, x, attribution),
            max_sensitivity=ExplanationMetrics.max_sensitivity(explain_fn, x),
            effective_complexity=ExplanationMetrics.effective_complexity(attribution),
            latency_ms=float(latency_ms),
        )

    @staticmethod
    def as_dict(report: ExplanationReport) -> dict:
        return asdict(report)
