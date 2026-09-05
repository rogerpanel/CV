"""Bias-corrected and accelerated (BCa) bootstrap confidence intervals with
resampling at the *subject* level.

Windows from one participant are correlated; resampling windows would treat
them as independent and give intervals that are far too narrow.  We therefore
resample subjects with replacement and take all of each sampled subject's
windows.  The acceleration term comes from a leave-one-subject-out jackknife.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def subject_bootstrap_ci(
    pred: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    method: str = "bca",
) -> tuple[float, float, float]:
    """Return (estimate, lower, upper) for ``metric(pred)`` with subject-level BCa bootstrap."""
    rng = np.random.default_rng(seed)
    subjects = pred["subject_id"].unique()
    by_subj = {s: g for s, g in pred.groupby("subject_id")}
    theta = metric(pred)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        frame = pd.concat([by_subj[s] for s in sample], ignore_index=True)
        try:
            v = metric(frame)
        except Exception:
            v = np.nan
        boots.append(v)
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if len(boots) < 10 or not np.isfinite(theta):
        return theta, np.nan, np.nan
    if method == "percentile":
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi)
    # BCa
    z0 = stats.norm.ppf(np.clip((boots < theta).mean(), 1e-6, 1 - 1e-6))
    jack = []
    for s in subjects:
        frame = pred[pred["subject_id"] != s]
        try:
            jack.append(metric(frame))
        except Exception:
            jack.append(np.nan)
    jack = np.asarray(jack, dtype=float)
    jack = jack[np.isfinite(jack)]
    if len(jack) < 3:
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0
    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)

    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo, hi = np.percentile(boots, [100 * adj(z_lo), 100 * adj(z_hi)])
    return float(theta), float(lo), float(hi)


def paired_subject_bootstrap_diff(
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """CI for metric(a) - metric(b) resampling the *same* subjects in both arms.

    Used in E1, where the record-wise and subject-wise arms score the same
    participants, so the difference is paired.
    """
    rng = np.random.default_rng(seed)
    subjects = np.intersect1d(pred_a["subject_id"].unique(), pred_b["subject_id"].unique())
    ga = {s: g for s, g in pred_a.groupby("subject_id")}
    gb = {s: g for s, g in pred_b.groupby("subject_id")}
    theta = metric(pred_a) - metric(pred_b)
    diffs = []
    for _ in range(n_boot):
        sample = rng.choice(subjects, size=len(subjects), replace=True)
        fa = pd.concat([ga[s] for s in sample], ignore_index=True)
        fb = pd.concat([gb[s] for s in sample], ignore_index=True)
        try:
            diffs.append(metric(fa) - metric(fb))
        except Exception:
            diffs.append(np.nan)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 10:
        return theta, np.nan, np.nan
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(theta), float(lo), float(hi)
