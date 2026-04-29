"""Lightweight IO wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_geochem_table(path: str | Path) -> pd.DataFrame:
    """Load a geochemical CSV with consistent dtypes.

    Expects the schema documented in ``data/geochemical/sample_template.csv``::

        sample_id, district, sub_district, lat, lon, matrix, depth_cm, year,
        method, As_mg_kg, Cd_mg_kg, Cr_mg_kg, ...
    """
    df = pd.read_csv(path)
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df


def save_results(obj: Any, path: str | Path) -> Path:
    """Save a DataFrame, dict or series to disk based on the file extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        if p.suffix == ".csv":
            obj.to_csv(p, index=False)
        elif p.suffix in {".parquet", ".pq"}:
            obj.to_parquet(p, index=False)
        else:                                                        # pragma: no cover
            raise ValueError(f"Unsupported extension {p.suffix} for DataFrame")
    elif isinstance(obj, dict):
        import json
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    else:
        import pickle
        with p.open("wb") as f:
            import pickle as _pkl
            _pkl.dump(obj, f)
    return p
