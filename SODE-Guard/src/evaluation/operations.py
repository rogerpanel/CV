"""Operational calibration for deployed detectors (future-work item: conformal alarms).

All functions take logits computed on a held-out *calibration* split and on the
test split. They need exchangeable calibration and test data, which holds for
the random splits of the v2 pipeline.

* ``fit_temperature``: temperature scaling (Guo et al., 2017), one scalar fitted
  by minimising the calibration negative log-likelihood.
* ``conformal_sets``: split-conformal prediction sets with the LAC score
  1 - p_y (Sadinle et al., 2019). The marginal version guarantees coverage
  >= 1 - alpha on average; the class-conditional (Mondrian) version guarantees it
  for every class, which matters for rare attacks.
* ``alarm_threshold``: a conformal false-alarm threshold. The alarm score is
  1 - p_benign; the threshold is the conformal quantile of the benign
  calibration scores, so the false-alarm rate on benign test flows is at most
  the target in expectation (up to 1/(n_benign + 1)).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, iters: int = 200) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")
    z, y = logits.double(), y.long()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.double().exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += abs((pred[m] == y[m]).mean() - conf[m].mean()) * m.mean()
    return float(out)


def _q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0
    k = math.ceil((n + 1) * (1 - alpha))
    return 1.0 if k > n else float(np.sort(scores)[k - 1])


def conformal_sets(p_cal: np.ndarray, y_cal: np.ndarray, p_te: np.ndarray, y_te: np.ndarray,
                   alpha: float, classwise: bool = False) -> dict:
    K = p_cal.shape[1]
    s_cal = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    if classwise:
        q = np.array([_q(s_cal[y_cal == k], alpha) for k in range(K)])
    else:
        q = np.full(K, _q(s_cal, alpha))
    sets = (1.0 - p_te) <= q[None, :]
    covered = sets[np.arange(len(y_te)), y_te]
    per_class = [float(covered[y_te == k].mean()) if (y_te == k).any() else None for k in range(K)]
    return {"alpha": alpha, "classwise": classwise, "coverage": float(covered.mean()),
            "min_class_coverage": float(min(c for c in per_class if c is not None)),
            "mean_set_size": float(sets.sum(1).mean()), "singleton_rate": float((sets.sum(1) == 1).mean()),
            "per_class_coverage": per_class}


def alarm_threshold(p_cal: np.ndarray, y_cal: np.ndarray, p_te: np.ndarray, y_te: np.ndarray,
                    benign: int, far: float) -> dict:
    s_cal = 1.0 - p_cal[:, benign]
    s_te = 1.0 - p_te[:, benign]
    t = _q(s_cal[y_cal == benign], far)
    alarm = s_te > t
    ben, att = y_te == benign, y_te != benign
    K = p_te.shape[1]
    return {"target_far": far, "threshold": t, "test_far": float(alarm[ben].mean()),
            "detection_rate": float(alarm[att].mean()),
            "per_class_detection": [float(alarm[y_te == k].mean()) if (y_te == k).any() and k != benign else None
                                    for k in range(K)]}
