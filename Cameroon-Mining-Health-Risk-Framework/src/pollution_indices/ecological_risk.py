"""Potential Ecological Risk Index (Håkanson, 1980).

For each element :math:`i`:

.. math::  E_r^i = T_r^i \\cdot CF_i

The total ecological risk is the sum of individual factors:

.. math::  RI = \\sum_i E_r^i
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cmhr.utils import load_config
from .contamination_factor import contamination_factor
from ._classify import classify


def _toxic_response_factors() -> Mapping[str, float]:
    return load_config("reference_doses")["toxic_response_factor"]


def ecological_risk_factor(
    measured,
    background: float,
    element: str,
):
    """Per-element ecological risk factor :math:`E_r^i`."""
    tr = _toxic_response_factors().get(element)
    if tr is None:
        raise KeyError(f"No toxic-response factor configured for '{element}'")
    cf = contamination_factor(measured, background)
    if isinstance(cf, pd.Series):
        return (cf * tr).rename(f"Er_{element}")
    return tr * cf


def ecological_risk_index(
    df: pd.DataFrame,
    backgrounds: Mapping[str, float],
    columns: Sequence[str] | None = None,
) -> pd.Series:
    """Sum of element-wise risks per sample (row).

    Column naming convention: ``<Element>_mg_kg`` (or ``_mg_l`` for water).
    """
    tr = _toxic_response_factors()
    if columns is None:
        columns = [c for c in df.columns if "_mg_" in c]

    er_cols = []
    for col in columns:
        element = col.split("_mg_")[0]
        if element not in backgrounds or element not in tr:
            continue
        er_cols.append(ecological_risk_factor(df[col], backgrounds[element], element).values)

    if not er_cols:
        raise ValueError("No usable element columns found for RI calculation")
    ri = np.nansum(np.column_stack(er_cols), axis=1)
    return pd.Series(ri, index=df.index, name="RI")


def classify_ri(ri_values, granularity: str = "total"):
    """Classify ``ri_values`` using either ``"total"`` (RI) or ``"individual"`` (Er)."""
    cfg = load_config("thresholds")
    rules = cfg["ecological_risk_index"] if granularity == "total" else cfg["individual_ecological_risk"]
    return classify(ri_values, rules)
