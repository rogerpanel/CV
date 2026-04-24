"""Calibrated Explanations (Equation 9).

    C_α(a_i) = [a_i − r_i, a_i + r_i]
    P[a_i^true ∈ C_α(a_i)] ≥ 1 − α

Wraps any base explainer and attaches a conformal half-width r_i computed
from a calibration set of (attribution_pred, attribution_true) residuals.

Typical use:
    ce = CalibratedExplanations(explainer, alpha=0.05)
    ce.calibrate(calibration_graphs, reference_explainer=shap)
    explanation = ce.explain(graph, target_node=42)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from ct_explain.data.graph_builder import TemporalGraph
from ct_explain.explainers.base import BaseExplainer, Explanation


@dataclass
class CalibrationState:
    residuals: np.ndarray      # (n, d) absolute residuals per feature
    half_widths: np.ndarray    # (d,)   (1-α)-quantile per feature
    alpha: float


class CalibratedExplanations(BaseExplainer):
    name = "calibrated"

    def __init__(
        self,
        base: BaseExplainer,
        alpha: float = 0.05,
    ) -> None:
        super().__init__(base.model)
        self.base = base
        self.alpha = alpha
        self.state: Optional[CalibrationState] = None

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #
    def calibrate(
        self,
        calibration_set: Iterable[tuple[TemporalGraph, int, torch.Tensor]],
    ) -> CalibrationState:
        """Each calibration entry = (graph, target_node, attribution_truth).

        ``attribution_truth`` is a reference explanation — typically from a
        high-fidelity but slow baseline such as SHAP or integrated gradients
        — against which the base explainer's residuals are computed.
        """
        residuals: list[np.ndarray] = []
        for graph, node, truth in calibration_set:
            expl = self.base.explain(graph=graph, target_node=node)
            pred = expl.attribution.cpu().numpy()
            tr = truth.cpu().numpy()
            d = min(len(pred), len(tr))
            residuals.append(np.abs(pred[:d] - tr[:d]))
        R = np.stack(residuals, axis=0)
        q = np.quantile(R, 1 - self.alpha, axis=0)
        self.state = CalibrationState(residuals=R, half_widths=q, alpha=self.alpha)
        return self.state

    # ------------------------------------------------------------------ #
    # Explanation with CIs
    # ------------------------------------------------------------------ #
    def explain(
        self, graph: TemporalGraph, target_node: int, **kwargs
    ) -> Explanation:
        base_expl = self.base.explain(graph=graph, target_node=target_node, **kwargs)
        attr = base_expl.attribution.cpu().numpy()
        if self.state is None:
            half = np.zeros_like(attr)
        else:
            half = np.resize(self.state.half_widths, len(attr))
        ci = torch.from_numpy(
            np.stack([attr - half, attr + half], axis=-1)
        ).float()
        return Explanation(
            attribution=base_expl.attribution,
            method=f"{self.base.name}+calibrated",
            latency_ms=base_expl.latency_ms,
            supporting=base_expl.supporting | {
                "alpha": self.alpha,
                "half_width": torch.from_numpy(half).float(),
            },
            confidence_interval=ci,
            target_node=target_node,
        )
