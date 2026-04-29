"""Carcinogenic-risk calculator (USEPA 2011).

For each carcinogenic element / pathway:

.. math::  CR = ADD \\times CSF
"""
from __future__ import annotations

import numpy as np

from cmhr.utils import load_config


def _csf_lookup(element: str, pathway: str = "ingestion") -> float:
    cfg = load_config("reference_doses")
    table = {
        "ingestion": cfg["cancer_slope_factor_oral"],
        "oral":      cfg["cancer_slope_factor_oral"],
        "dermal":    cfg["cancer_slope_factor_dermal"],
        "inhalation": cfg["cancer_slope_factor_inhalation"],
    }[pathway]
    if element not in table:
        raise KeyError(f"No CSF for {element!r} via {pathway}")
    return float(table[element])


def cancer_risk(add, element: str, pathway: str = "ingestion"):
    """Incremental Lifetime Cancer Risk."""
    csf = _csf_lookup(element, pathway)
    return np.asarray(add, dtype=float) * csf


def classify_cancer_risk(value: float) -> str:
    """USEPA acceptability classes."""
    cfg = load_config("thresholds")["cancer_risk"]
    if value < cfg["acceptable"]:
        return "Acceptable"
    if value <= cfg["tolerable_max"]:
        return "Tolerable"
    return "Unacceptable"
