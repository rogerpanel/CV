"""Pydantic v2 schemas for the REST API surface."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NetflowEvent(BaseModel):
    src_ip: str
    dst_ip: str
    timestamp: float
    features: dict[str, float] = Field(default_factory=dict)
    label: Optional[int] = None


class PredictRequest(BaseModel):
    events: list[NetflowEvent]
    target_node: Optional[int] = None


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]
    safety_probabilities: list[float]
    uncertainty: list[float]


class ExplainRequest(BaseModel):
    events: list[NetflowEvent]
    target_node: int
    techniques: list[str] = Field(
        default_factory=lambda: [
            "attention_flow",
            "uncertainty_attribution",
            "counterfactual",
            "game_theory",
        ]
    )
    render_dir: Optional[str] = None


class CertifyRequest(BaseModel):
    action_id: str
    exec_graph: dict[str, Any]
    p_safe: dict[str, float]


class CalibrationRequest(BaseModel):
    alpha: float = 0.05
    k_hops: int = 2
    lambda_cascade: float = 0.5
    calibration_set: list[dict[str, Any]]


class FeedbackRequest(BaseModel):
    alert_id: str
    corrected_label: int
    rationale: str = ""
    analyst_confidence: float = 1.0
    features: Optional[list[float]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    backbone: str
    calibrated: bool
