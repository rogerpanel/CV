"""Pollution Load Index (Tomlinson et al., 1980).

.. math::  PLI = \\bigl(\\prod_{i=1}^n CF_i\\bigr)^{1/n}

PLI > 1 indicates progressive deterioration; PLI < 1 indicates baseline
conditions.  The function accepts either a wide-format DataFrame of
concentrations + a background dictionary, or pre-computed CFs.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contamination_factor import contamination_factor


def pollution_load_index(
    df: pd.DataFrame,
    backgrounds: Mapping[str, float] | None = None,
    columns: Sequence[str] | None = None,
    cf_columns: Sequence[str] | None = None,
) -> pd.Series:
    """Compute the PLI per sample (per row).

    Two modes:
    1. Pass ``backgrounds`` plus the concentration columns → CF is computed
       internally.  Background keys must match the column suffix before
       ``"_mg_kg"`` / ``"_mg_l"`` (e.g. ``"As_mg_kg"`` ↔ ``backgrounds["As"]``).
    2. Pass ``cf_columns`` of already-computed CFs.

    Returns
    -------
    pd.Series
        PLI per sample, indexed identically to ``df``.
    """
    if cf_columns is not None:
        cf_matrix = df[list(cf_columns)].astype(float).values
    else:
        if backgrounds is None:
            raise ValueError("Either `backgrounds` or `cf_columns` must be provided")
        if columns is None:
            columns = [c for c in df.columns if "_mg_" in c]
        cfs = []
        for col in columns:
            element = col.split("_mg_")[0]
            if element not in backgrounds:
                continue
            cfs.append(contamination_factor(df[col], backgrounds[element]).values)
        if not cfs:
            raise ValueError("No usable element columns found")
        cf_matrix = np.column_stack(cfs)

    cf_matrix = np.where(cf_matrix > 0, cf_matrix, np.nan)
    log_mean = np.nanmean(np.log(cf_matrix), axis=1)
    pli = np.exp(log_mean)
    return pd.Series(pli, index=df.index, name="PLI")
