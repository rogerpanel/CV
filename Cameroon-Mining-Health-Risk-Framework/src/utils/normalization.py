"""Unit standardisation and transformations used by the indices."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


_UNIT_TO_MG_PER_KG = {
    "mg/kg": 1.0,
    "ug/g": 1.0,
    "ppm":  1.0,
    "g/kg": 1000.0,
    "wt%":  10000.0,
    "ug/kg": 1e-3,
    "ppb":   1e-3,
}

_UNIT_TO_MG_PER_L = {
    "mg/L": 1.0,
    "ppm":  1.0,
    "ug/L": 1e-3,
    "ppb":  1e-3,
    "g/L":  1000.0,
}


def standardise_units(value: float | np.ndarray, unit: str, kind: str = "solid"):
    """Convert concentrations to mg/kg (solids) or mg/L (water)."""
    table = _UNIT_TO_MG_PER_KG if kind == "solid" else _UNIT_TO_MG_PER_L
    try:
        factor = table[unit]
    except KeyError as err:
        raise ValueError(f"Unsupported unit '{unit}' for kind '{kind}'") from err
    return value * factor


def log_transform(df: pd.DataFrame, columns: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Return a copy of ``df`` with selected columns log-transformed (base e).

    Parameters
    ----------
    columns : mapping, optional
        ``{source_column: new_column}``.  If omitted, every numeric column is
        log-transformed in place with the suffix ``_log``.
    """
    out = df.copy()
    if columns is None:
        for c in out.select_dtypes(include="number").columns:
            out[f"{c}_log"] = np.log(out[c].replace(0, np.nan))
    else:
        for src, dst in columns.items():
            out[dst] = np.log(out[src].replace(0, np.nan))
    return out
