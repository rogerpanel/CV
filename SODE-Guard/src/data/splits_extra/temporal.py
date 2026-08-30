"""Temporal (time-ordered) train/val/test split.

Reviewer 2: random 70/15/15 splits allow flows from the same attack
campaign or generator run to appear in both train and test. The
temporal-holdout protocol sorts by ``timestamp`` (or a monotonically
increasing surrogate) and takes the first 70 % of the timeline for
training, the next 15 % for validation, and the final 15 % for test.
Attack campaigns therefore cannot leak forward in time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def temporal_holdout_split(df: pd.DataFrame, *,
                           timestamp_col: str = "timestamp",
                           train: float = 0.70,
                           val: float = 0.15,
                           test: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("splits must sum to 1")
    if timestamp_col not in df.columns:
        # Fall back to row order if no timestamp column exists.
        order = np.arange(len(df))
    else:
        order = np.argsort(pd.to_datetime(df[timestamp_col]).values)
    df_sorted = df.iloc[order].reset_index(drop=True)
    n = len(df_sorted)
    a = int(train * n); b = int((train + val) * n)
    return df_sorted.iloc[:a], df_sorted.iloc[a:b], df_sorted.iloc[b:]
