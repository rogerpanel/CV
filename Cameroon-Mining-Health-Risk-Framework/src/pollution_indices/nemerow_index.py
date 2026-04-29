"""Nemerow Pollution Index (Nemerow, 1991; Cheng et al., 2007).

Combines mean and maximum single-factor pollution indices to flag the most
affected element while incorporating overall load:

.. math::

    PI_{N} = \\sqrt{\\frac{\\overline{PI}^{2} + PI_{max}^{2}}{2}}

where each single-factor index :math:`PI_i = C_i / S_i` (S_i = standard).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cmhr.utils import load_config
from ._classify import classify


def _single_factor(df: pd.DataFrame, standards: Mapping[str, float],
                   columns: Sequence[str] | None = None) -> pd.DataFrame:
    if columns is None:
        columns = [c for c in df.columns if "_mg_" in c]
    out = {}
    for col in columns:
        element = col.split("_mg_")[0]
        std = standards.get(element)
        if std is None or std <= 0:
            continue
        out[f"PI_{element}"] = df[col].astype(float) / std
    return pd.DataFrame(out, index=df.index)


def nemerow_pollution_index(
    df: pd.DataFrame,
    standards: Mapping[str, float],
    columns: Sequence[str] | None = None,
    return_components: bool = False,
):
    """Compute the Nemerow PI per sample.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format concentrations.
    standards : mapping
        Element → guideline / standard value (matching the units of ``df``).
        Defaults: WHO drinking-water guidelines for water samples
        (``config/reference_doses.yaml :: who_water_standards``).
    """
    pi = _single_factor(df, standards, columns)
    if pi.empty:
        raise ValueError("No matching element columns found for Nemerow PI")
    pi_mean = pi.mean(axis=1)
    pi_max = pi.max(axis=1)
    pin = np.sqrt((pi_mean.pow(2) + pi_max.pow(2)) / 2.0)
    pin.name = "Nemerow_PI"
    if return_components:
        return pin, pi
    return pin


def classify_nemerow(values):
    rules = load_config("thresholds")["nemerow_pollution_index"]
    return classify(values, rules)
