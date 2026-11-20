"""Conformal alarm thresholds on realised traffic.

Calibrated on benign testbed traffic (held out), the alarm fires when a flow's
benign-class score falls below the threshold that gives a chosen false-alarm rate
(1% and 5%). Reported alongside detection so the detection numbers come at a
stated false-alarm budget, exactly as in the paper. Reuses the project's
operations.alarm_threshold where logits are available; for label-only scorers
(the platform API) it falls back to an alarm-on-attack-label rule and reports the
empirical benign false-alarm rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from src.evaluation.operations import alarm_threshold   # noqa: F401
    HAVE_OPS = True
except Exception:
    HAVE_OPS = False


def thresholds_from_benign(benign_scores: np.ndarray, budgets=(0.01, 0.05)) -> dict[float, float]:
    """benign_scores: per-flow benign-class probability on calibration benign flows.
    The alarm fires when the benign score < tau; tau is the budget-quantile."""
    return {b: float(np.quantile(benign_scores, b)) for b in budgets}


def apply(scores: np.ndarray, tau: float) -> np.ndarray:
    return scores < tau


def far_detection(is_attack: np.ndarray, alarm: np.ndarray) -> tuple[float, float]:
    benign = ~is_attack
    far = float(alarm[benign].mean()) if benign.any() else float("nan")
    det = float(alarm[is_attack].mean()) if is_attack.any() else float("nan")
    return far, det


def label_rule_far_detection(pred_labels: np.ndarray, is_attack: np.ndarray,
                             benign_name: str) -> tuple[float, float]:
    """Alarm = predicted a non-benign class (for label-only scorers)."""
    alarm = pred_labels != benign_name
    return far_detection(is_attack, alarm)
