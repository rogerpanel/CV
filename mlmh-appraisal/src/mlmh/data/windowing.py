"""Segment per-minute activity into windows, carrying subject_id through.

Default window = one calendar day (00:00-23:59), the unit used by the dataset
authors' own baselines.  Days with fewer than ``min_minutes`` valid minutes
(default 80 % of 1440) are discarded, as are the first and last partial days
when ``drop_edge_days`` is set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440


def make_day_windows(minutes: pd.DataFrame, min_minutes: int = 1152, drop_edge_days: bool = True) -> pd.DataFrame:
    """Return a long table with an extra ``window_id`` and ``minute_of_day`` column."""
    df = minutes.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["valid"] = df["activity"].notna()
    counts = df.groupby(["subject_id", "date"]).agg(n_valid=("valid", "sum"), n_rows=("valid", "size")).reset_index()
    keep = counts[counts["n_valid"] >= min_minutes]
    if drop_edge_days:
        first_last = df.groupby("subject_id")["date"].agg(["min", "max"]).reset_index()
        keep = keep.merge(first_last, on="subject_id")
        # Only drop an edge day if it is genuinely partial: the recording does not span the full day.
        keep = keep[~(((keep["date"] == keep["min"]) | (keep["date"] == keep["max"])) & (keep["n_rows"] < MINUTES_PER_DAY))]
        keep = keep[["subject_id", "date", "n_valid"]]
    df = df.merge(keep[["subject_id", "date"]], on=["subject_id", "date"], how="inner")
    df["window_id"] = df["cohort"] + ":" + df["subject_id"].astype(str) + ":" + df["date"].dt.strftime("%Y-%m-%d")
    # De-duplicate minutes (some devices log a minute twice at DST changes).
    df = df.drop_duplicates(subset=["window_id", "minute_of_day"], keep="first")
    return df.drop(columns=["valid"]).reset_index(drop=True)


def window_matrix(windows: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Pivot to a (n_windows, 1440) matrix of raw activity, NaN where missing.

    Returns the matrix and an index frame (window_id, subject_id, cohort).
    """
    piv = windows.pivot_table(index="window_id", columns="minute_of_day", values="activity", aggfunc="first")
    piv = piv.reindex(columns=range(MINUTES_PER_DAY))
    idx = windows.drop_duplicates("window_id").set_index("window_id").loc[piv.index, ["subject_id", "cohort"]].reset_index()
    return piv.to_numpy(dtype=float), idx
