"""SOC analyst dashboard objects.

The paper's §5 describes three analyst-facing views:

    AlertTriageDashboard   – uncertainty-ranked queue with conformal CIs
    InvestigationPanel     – all four explanations + counterfactual explorer
    FeedbackLoop (active learning, separate module)

These classes return JSON-serialisable dictionaries consumed by both the
Flask REST API (`ct_explain.api.server`) and the RobustIDPS.ai React SPA.
They do NOT render HTML — rendering is a frontend concern. The goal is to
give a stable, well-typed contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from ct_explain.conformal.conformal_guard import ConformalVerdict


@dataclass
class Alert:
    alert_id: str
    source: str
    target_node: int
    prediction: int
    predicted_class: str
    confidence: float
    uncertainty: float
    conformal_verdict: str                  # "Safe" | "Abstain" | "Escalate"
    timestamp: float
    explanation_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AlertTriageDashboard:
    """Uncertainty-ranked alert queue."""

    def __init__(self) -> None:
        self._queue: list[Alert] = []

    def push(self, alert: Alert) -> None:
        self._queue.append(alert)
        # Higher uncertainty first; conformal-escalated alerts always bubble up.
        self._queue.sort(
            key=lambda a: (
                a.conformal_verdict != "Escalate",
                -a.uncertainty,
                -a.confidence,
            )
        )

    def top(self, k: int = 10) -> list[dict]:
        return [a.to_dict() for a in self._queue[:k]]

    def __len__(self) -> int:
        return len(self._queue)


class InvestigationPanel:
    """Aggregates the four explanations + calibrated CIs for a single alert."""

    def build(
        self,
        *,
        alert: Alert,
        attention_explanation,
        uncertainty_explanation,
        counterfactual_explanation,
        game_explanation,
        calibrated_intervals: torch.Tensor | None = None,
        conformal: ConformalVerdict | None = None,
    ) -> dict:
        return {
            "alert": alert.to_dict(),
            "attention_flow": {
                "latency_ms": attention_explanation.latency_ms,
                "edge_attention": attention_explanation.supporting["attention_flow"]
                    .edge_attention.tolist(),
                "stages": attention_explanation.supporting["attention_flow"].stages,
                "colors": attention_explanation.supporting["attention_flow"].stage_colors,
                "rendered": attention_explanation.supporting["attention_flow"].rendered_paths,
            },
            "uncertainty_attribution": {
                "latency_ms": uncertainty_explanation.latency_ms,
                "per_feature_share": uncertainty_explanation
                    .supporting["decomposition"].per_feature_share.tolist(),
                "total_variance": uncertainty_explanation.supporting["decomposition"].variance_total,
                "method": uncertainty_explanation.supporting["decomposition"].method,
            },
            "counterfactual": {
                "latency_ms": counterfactual_explanation.latency_ms,
                "delta": counterfactual_explanation.supporting["counterfactual"].delta.tolist(),
                "l2_norm": counterfactual_explanation.supporting["counterfactual"].l2_norm,
                "original_prediction": counterfactual_explanation
                    .supporting["counterfactual"].original_prediction,
                "perturbed_prediction": counterfactual_explanation
                    .supporting["counterfactual"].perturbed_prediction,
                "iterations": counterfactual_explanation.supporting["counterfactual"].iterations,
            },
            "game_theory": {
                "latency_ms": game_explanation.latency_ms,
                "attacker_strategy": game_explanation.supporting["strategy"]
                    .attacker_strategy.tolist(),
                "defender_strategy": game_explanation.supporting["strategy"]
                    .defender_strategy.tolist(),
                "value": game_explanation.supporting["strategy"].value,
                "tactic_rank": game_explanation.supporting["strategy"].tactic_rank,
                "evasion_budget": game_explanation.supporting["strategy"].evasion_budget,
                "narrative": game_explanation.supporting["strategy"].narrative,
            },
            "calibrated_intervals": (
                calibrated_intervals.tolist() if calibrated_intervals is not None else None
            ),
            "conformal": (
                {
                    "verdict": conformal.verdict,
                    "score": conformal.score,
                    "q_hat": conformal.q_hat,
                    "e_value": conformal.e_value,
                    "triggered": conformal.triggered,
                    "reason": conformal.reason,
                }
                if conformal is not None else None
            ),
        }
