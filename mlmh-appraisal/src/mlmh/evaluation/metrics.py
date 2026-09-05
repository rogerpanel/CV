"""Discrimination *and* calibration, at window level and at subject level.

calibration_slope / calibration_intercept follow Van Calster et al. (2019):
slope from ``y ~ logit(p)``, intercept from ``y ~ 1 + offset(logit(p))``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope(y, p) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(C=1e6, max_iter=5000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0])


def calibration_intercept(y, p) -> float:
    """Calibration-in-the-large with slope fixed at 1: intercept of y ~ 1 + offset(logit p).

    Solved by one-dimensional Newton iterations on the log-likelihood.
    """
    y = np.asarray(y, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan
    lp = _logit(p)
    a = 0.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-(lp + a)))
        grad = np.sum(y - q)
        hess = -np.sum(q * (1 - q))
        step = grad / hess if hess != 0 else 0.0
        a -= step
        if abs(step) < 1e-10:
            break
    return float(a)


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y, p, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "p_mean": p[m].mean(), "y_mean": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def binary_metrics(y, p, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    two_class = len(np.unique(y)) == 2
    out = {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)) if two_class else np.nan,
        "macro_f1": float(f1_score(y, yhat, average="macro")),
        "mcc": float(matthews_corrcoef(y, yhat)) if two_class else np.nan,
        "auroc": float(roc_auc_score(y, p)) if two_class else np.nan,
        "auprc": float(average_precision_score(y, p)) if two_class else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "calibration_slope": calibration_slope(y, p),
        "calibration_intercept": calibration_intercept(y, p),
        "ece": expected_calibration_error(y, p),
    }
    return out


def subject_level(pred: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window predictions to one probability per subject (mean)."""
    g = pred.groupby("subject_id").agg(y=("y", "first"), p=("p", "mean"), n_windows=("p", "size")).reset_index()
    return g


def all_metrics(pred: pd.DataFrame) -> dict[str, float]:
    """Window-level and subject-level metrics from a predictions frame (y, p, subject_id)."""
    w = binary_metrics(pred["y"], pred["p"])
    s = binary_metrics(*subject_level(pred)[["y", "p"]].to_numpy().T)
    out = {f"window_{k}": v for k, v in w.items()}
    out.update({f"subject_{k}": v for k, v in s.items()})
    out["n_subjects"] = int(pred["subject_id"].nunique())
    return out


PRIMARY = ["auroc", "accuracy", "macro_f1", "brier", "calibration_slope", "calibration_intercept", "ece"]
