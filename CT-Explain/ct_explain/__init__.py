"""CT-Explain: Explainable Continuous-Time Dynamic GNNs with Conformal Safety
Certification.

Top-level re-exports give users one import for the most common objects. Sub-
packages are lazy-loaded so importing the top-level does not pull in heavy
optional dependencies (torch_geometric, torchsde) until actually used.
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Roger Nick Anaedevha"

from importlib import import_module
from typing import TYPE_CHECKING, Any

_LAZY = {
    "CTTemporalGNN": ("ct_explain.models.ct_tgnn", "CTTemporalGNN"),
    "SDETemporalGNN": ("ct_explain.models.sde_tgnn", "SDETemporalGNN"),
    "ConformalGuard": ("ct_explain.conformal.conformal_guard", "ConformalGuard"),
    "GraphConformalCalibrator": (
        "ct_explain.conformal.graph_conformal",
        "GraphConformalCalibrator",
    ),
    "EValueMartingale": ("ct_explain.conformal.e_value", "EValueMartingale"),
    "AttentionFlowExplainer": (
        "ct_explain.explainers.attention_flow",
        "AttentionFlowExplainer",
    ),
    "UncertaintyAttributionExplainer": (
        "ct_explain.explainers.uncertainty_attribution",
        "UncertaintyAttributionExplainer",
    ),
    "CounterfactualExplainer": (
        "ct_explain.explainers.counterfactual",
        "CounterfactualExplainer",
    ),
    "GameTheoreticExplainer": (
        "ct_explain.explainers.game_theory",
        "GameTheoreticExplainer",
    ),
    "CalibratedExplanations": (
        "ct_explain.explainers.calibrated",
        "CalibratedExplanations",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        return getattr(import_module(module_path), attr)
    raise AttributeError(f"module 'ct_explain' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(_LAZY) + ["__version__", "__author__"])


if TYPE_CHECKING:  # pragma: no cover
    from ct_explain.conformal.conformal_guard import ConformalGuard  # noqa: F401
    from ct_explain.conformal.e_value import EValueMartingale  # noqa: F401
    from ct_explain.conformal.graph_conformal import GraphConformalCalibrator  # noqa: F401
    from ct_explain.explainers.attention_flow import AttentionFlowExplainer  # noqa: F401
    from ct_explain.explainers.calibrated import CalibratedExplanations  # noqa: F401
    from ct_explain.explainers.counterfactual import CounterfactualExplainer  # noqa: F401
    from ct_explain.explainers.game_theory import GameTheoreticExplainer  # noqa: F401
    from ct_explain.explainers.uncertainty_attribution import (  # noqa: F401
        UncertaintyAttributionExplainer,
    )
    from ct_explain.models.ct_tgnn import CTTemporalGNN  # noqa: F401
    from ct_explain.models.sde_tgnn import SDETemporalGNN  # noqa: F401
