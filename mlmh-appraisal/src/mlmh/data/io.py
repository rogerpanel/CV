"""Robust CSV reading for the Simula actigraphy family.

The three source datasets were recorded with the same Actiwatch device but use
different delimiters, column names and date formats.  Rather than hard-code a
convention per file, we sniff the delimiter and normalise column names, and we
fail loudly when a file does not contain a recognisable timestamp and activity
column.  ``python -m mlmh verify-data`` prints what was recognised so the layout
can be checked before any modelling runs.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

_TS_CANDIDATES = ("timestamp", "time", "datetime", "date_time")
_ACT_CANDIDATES = ("activity", "act", "counts", "activity_counts")


def sniff_delimiter(path: Path) -> str:
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def read_activity_csv(path: Path) -> pd.DataFrame:
    """Return a frame with columns ``timestamp`` (datetime64) and ``activity`` (float)."""
    delim = sniff_delimiter(path)
    df = pd.read_csv(path, sep=delim, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    ts_col = next((c for c in _TS_CANDIDATES if c in df.columns), None)
    act_col = next((c for c in _ACT_CANDIDATES if c in df.columns), None)
    if ts_col is None or act_col is None:
        raise ValueError(f"{path}: could not find timestamp/activity columns in {list(df.columns)}")
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce"),
            "activity": pd.to_numeric(df[act_col], errors="coerce"),
        }
    )
    if out["timestamp"].isna().all():
        raise ValueError(f"{path}: timestamps could not be parsed")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sniff_delimiter(path), engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def series_hash(activity: np.ndarray) -> str:
    """Hash of an activity series, used to detect the same participant across cohorts."""
    arr = np.ascontiguousarray(np.nan_to_num(np.asarray(activity, dtype=np.float64), nan=-1.0))
    return hashlib.sha256(arr.tobytes()).hexdigest()
