"""Environmental predictors for SDM (WorldClim, SRTM, ESA WorldCover, etc.)."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def build_predictor_stack(raster_paths: Mapping[str, str | Path]) -> dict:
    """Read a set of raster paths into ``{name: array, transform, crs}``.

    Loads via rasterio when available; otherwise raises an informative error.
    """
    try:
        import rasterio
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("rasterio is required: pip install rasterio") from err

    stack = {}
    for name, path in raster_paths.items():
        with rasterio.open(path) as ds:
            stack[name] = {
                "array": ds.read(1).astype(np.float32),
                "transform": ds.transform,
                "crs": ds.crs,
                "nodata": ds.nodata,
            }
    return stack


def screen_multicollinearity(
    samples: pd.DataFrame,
    max_pearson: float = 0.8,
    keep_priority: Sequence[str] | None = None,
) -> list[str]:
    """Greedy removal of highly correlated covariates (|r| > max_pearson).

    Parameters
    ----------
    samples : DataFrame
        One row per occurrence point, columns = predictors.
    max_pearson : float, default 0.8 (manuscript value).
    keep_priority : sequence of str, optional
        Predictors to retain when collinear pairs are found.
    """
    cols = list(samples.columns)
    if keep_priority:
        cols.sort(key=lambda c: 0 if c in keep_priority else 1)

    keep = []
    corr = samples[cols].corr().abs()
    for c in cols:
        if all(corr.at[c, k] <= max_pearson for k in keep):
            keep.append(c)
    return keep
