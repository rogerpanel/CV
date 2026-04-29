"""USEPA-2011 probabilistic human-health risk assessment.

The module exposes:

* :func:`add_ingestion`, :func:`add_dermal`, :func:`add_inhalation` — average
  daily-dose calculators for each exposure pathway.
* :func:`hazard_quotient`, :func:`hazard_index` — non-carcinogenic risk.
* :func:`cancer_risk` — Incremental Lifetime Cancer Risk (ILCR).
* :class:`MonteCarloRiskAssessment` — 10,000-iteration probabilistic engine
  identical to the configuration used in Anaedevha et al. (2026).
* :func:`sobol_sensitivity` — global sensitivity analysis (Saltelli scheme).
"""
from .add_calculation import add_ingestion, add_dermal, add_inhalation
from .hazard_quotient import hazard_quotient, hazard_index
from .cancer_risk import cancer_risk, classify_cancer_risk
from .monte_carlo import MonteCarloRiskAssessment, RiskResult
from .sensitivity_analysis import sobol_sensitivity, spearman_sensitivity

__all__ = [
    "add_ingestion",
    "add_dermal",
    "add_inhalation",
    "hazard_quotient",
    "hazard_index",
    "cancer_risk",
    "classify_cancer_risk",
    "MonteCarloRiskAssessment",
    "RiskResult",
    "sobol_sensitivity",
    "spearman_sensitivity",
]
