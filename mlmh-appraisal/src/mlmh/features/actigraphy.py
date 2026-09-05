"""Window-level features from minute-level activity counts.

Two representations:

* ``engineered`` : statistical + circadian descriptors (the tabular models)
* ``raw``        : the log1p-transformed 1440-minute series (the 1D-CNN)

No demographic variable is used.  Age and sex are available for some cohorts
only, and mixing them in would confound the cross-cohort comparison in E2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..data.windowing import MINUTES_PER_DAY, window_matrix

DAY_START, DAY_END = 8 * 60, 22 * 60  # 08:00-22:00 = daytime


def _nanpercentile(a, q):
    return np.nanpercentile(a, q) if np.isfinite(a).any() else np.nan


def engineered_features_one(row: np.ndarray) -> dict[str, float]:
    a = row.astype(float)
    valid = np.isfinite(a)
    x = a[valid]
    f: dict[str, float] = {}
    f["mean"] = x.mean() if x.size else np.nan
    f["sd"] = x.std(ddof=1) if x.size > 1 else np.nan
    f["cv"] = f["sd"] / f["mean"] if f["mean"] else np.nan
    f["median"] = np.median(x) if x.size else np.nan
    f["p10"] = _nanpercentile(a, 10)
    f["p90"] = _nanpercentile(a, 90)
    f["max"] = x.max() if x.size else np.nan
    f["skew"] = stats.skew(x) if x.size > 2 else np.nan
    f["kurtosis"] = stats.kurtosis(x) if x.size > 3 else np.nan
    f["prop_zero"] = float((x == 0).mean()) if x.size else np.nan
    f["mean_log1p"] = np.log1p(x).mean() if x.size else np.nan
    f["sd_log1p"] = np.log1p(x).std(ddof=1) if x.size > 1 else np.nan
    # Day / night
    mins = np.arange(MINUTES_PER_DAY)
    day = valid & (mins >= DAY_START) & (mins < DAY_END)
    night = valid & ~((mins >= DAY_START) & (mins < DAY_END))
    f["day_mean"] = a[day].mean() if day.any() else np.nan
    f["night_mean"] = a[night].mean() if night.any() else np.nan
    f["night_day_ratio"] = f["night_mean"] / f["day_mean"] if f["day_mean"] else np.nan
    f["night_prop_zero"] = float((a[night] == 0).mean()) if night.any() else np.nan
    # Non-parametric circadian measures on the hourly profile
    hourly = np.array([np.nanmean(a[h * 60 : (h + 1) * 60]) if np.isfinite(a[h * 60 : (h + 1) * 60]).any() else np.nan for h in range(24)])
    if np.isfinite(hourly).sum() >= 12:
        hh = np.where(np.isfinite(hourly), hourly, np.nanmean(hourly))
        circ = np.concatenate([hh, hh])
        m10 = max(circ[i : i + 10].mean() for i in range(24))
        l5 = min(circ[i : i + 5].mean() for i in range(24))
        f["M10"], f["L5"] = m10, l5
        f["relative_amplitude"] = (m10 - l5) / (m10 + l5) if (m10 + l5) else np.nan
        # Intradaily variability (hourly)
        d = np.diff(hh)
        f["IV"] = (24 * np.sum(d**2)) / ((24 - 1) * np.sum((hh - hh.mean()) ** 2)) if np.sum((hh - hh.mean()) ** 2) else np.nan
        # Time (hour) of the most active 10 h onset: circadian phase proxy
        onset = int(np.argmax([circ[i : i + 10].mean() for i in range(24)]))
        f["m10_onset_sin"] = np.sin(2 * np.pi * onset / 24)
        f["m10_onset_cos"] = np.cos(2 * np.pi * onset / 24)
    else:
        for k in ("M10", "L5", "relative_amplitude", "IV", "m10_onset_sin", "m10_onset_cos"):
            f[k] = np.nan
    # Autocorrelation at 60 min and 5 min (activity persistence)
    for lag, key in ((5, "acf_5"), (60, "acf_60")):
        xx = np.where(valid, a, np.nanmean(a) if valid.any() else 0.0)
        if xx.size > lag + 2 and xx.std() > 0:
            f[key] = np.corrcoef(xx[:-lag], xx[lag:])[0, 1]
        else:
            f[key] = np.nan
    # Spectral: mean power spectral density and dominant period (min) within the day
    xx = np.where(valid, a, np.nanmean(a) if valid.any() else 0.0) - (np.nanmean(a) if valid.any() else 0.0)
    if xx.size > 16 and np.any(xx != 0):
        psd = np.abs(np.fft.rfft(xx)) ** 2 / xx.size
        f["psd_mean"] = float(np.log1p(psd[1:].mean()))
        freqs = np.fft.rfftfreq(xx.size, d=1.0)
        f["dominant_period_min"] = float(1.0 / freqs[1:][np.argmax(psd[1:])])
    else:
        f["psd_mean"], f["dominant_period_min"] = np.nan, np.nan
    # Activity bouts: number of transitions between rest (0) and activity
    z = (np.where(valid, a, 0) > 0).astype(int)
    f["n_transitions"] = float(np.abs(np.diff(z)).sum())
    f["missing_frac"] = float(1 - valid.mean())
    return f


def build_features(windows: pd.DataFrame, representation: str = "engineered") -> tuple[pd.DataFrame, list[str]]:
    """Return (frame with subject_id/cohort/window_id + features, feature_names)."""
    M, idx = window_matrix(windows)
    if representation == "engineered":
        rows = [engineered_features_one(M[i]) for i in range(M.shape[0])]
        feats = pd.DataFrame(rows)
    elif representation == "raw":
        feats = pd.DataFrame(np.log1p(np.clip(M, 0, None)), columns=[f"raw_{i:04d}" for i in range(M.shape[1])])
    else:
        raise ValueError(representation)
    names = list(feats.columns)
    out = pd.concat([idx.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)
    return out, names
