"""Metrics for the problem-space study.

  * detection: per-flow detection rate, per-attack-instance detection (a run is
    detected if any of its flows alarms), macro-F1 on realised attacks;
  * evasion success: fraction of evasive runs that stay functional and undetected,
    with overhead;
  * cost: per-flow latency, throughput, memory (filled by scoring timers).

Statistics reuse the project's paired bootstrap with Holm correction so the
contrasts match the paper's family-wise control.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from src.evaluation.stats import paired_bootstrap, holm   # noqa: F401
    HAVE_STATS = True
except Exception:
    HAVE_STATS = False


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def detection_rate(is_attack: np.ndarray, alarm: np.ndarray) -> float:
    return float(alarm[is_attack].mean()) if is_attack.any() else float("nan")


def instance_detection(run_ids: np.ndarray, is_attack: np.ndarray, alarm: np.ndarray) -> float:
    """A run is detected if any of its attack flows alarms."""
    detected, total = 0, 0
    for rid in np.unique(run_ids[is_attack]):
        m = (run_ids == rid) & is_attack
        total += 1
        detected += int(alarm[m].any())
    return detected / total if total else float("nan")


def evasion_success_rate(functional: np.ndarray, detected: np.ndarray) -> float:
    ev = functional & ~detected
    return float(ev.sum() / functional.sum()) if functional.any() else float("nan")


def contrast(a_correct: np.ndarray, b_correct: np.ndarray, seeds: np.ndarray):
    """Paired difference A-B with CI and p, via the project's bootstrap if present."""
    if HAVE_STATS:
        return paired_bootstrap(a_correct, b_correct, seeds)
    diff = a_correct.mean() - b_correct.mean()
    return {"diff": float(diff), "ci": (None, None), "p": None}
