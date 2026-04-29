"""Additional commonly-used pollution indices.

* Single-element Pollution Index (PI = C / S)
* Modified Contamination Degree (mCdeg, Abrahim & Parker 2008)
* Degree of Contamination (Cdeg, Håkanson 1980)
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contamination_factor import contamination_factor


def pollution_index_single(measured, standard: float):
    """Per-element single-factor index ``PI = C / S``."""
    if standard is None or standard <= 0:
        raise ValueError("standard must be positive")
    pi = np.asarray(measured, dtype=float) / standard
    if isinstance(measured, pd.Series):
        return pd.Series(pi, index=measured.index, name=f"PI_{measured.name}")
    if np.isscalar(measured):
        return float(pi)
    return pi


def modified_contamination_degree(
    df: pd.DataFrame,
    backgrounds: Mapping[str, float],
    columns: Sequence[str] | None = None,
) -> pd.Series:
    """Abrahim & Parker (2008) modified contamination degree.

    .. math::  mCdeg = \\frac{1}{n} \\sum_{i=1}^{n} CF_i

    Classification (per Abrahim & Parker):
    - <1.5  Nil to very low
    - 1.5–2 Low
    - 2–4   Moderate
    - 4–8   High
    - 8–16  Very high
    - 16–32 Extremely high
    - >32   Ultra-high
    """
    if columns is None:
        columns = [c for c in df.columns if "_mg_" in c]
    cf_cols = []
    for col in columns:
        element = col.split("_mg_")[0]
        if element not in backgrounds:
            continue
        cf_cols.append(contamination_factor(df[col], backgrounds[element]).values)
    if not cf_cols:
        raise ValueError("No usable element columns")
    arr = np.column_stack(cf_cols)
    return pd.Series(np.nanmean(arr, axis=1), index=df.index, name="mCdeg")
