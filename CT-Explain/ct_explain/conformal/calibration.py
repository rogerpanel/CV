"""Classical split-conformal calibration for the CT-TGNN detection backbone.

Used for point-wise intrusion decisions (the paper's Problem 1 stream).
ConformalGuard's graph-aware calibrator lives in `graph_conformal.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ConformalResult:
    q_hat: float
    alpha: float
    n_calibration: int


class SplitConformalCalibrator:
    """Classical split conformal over a score function s(x,y)."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._result: Optional[ConformalResult] = None

    def calibrate(self, scores: torch.Tensor | np.ndarray) -> ConformalResult:
        s = np.asarray(scores).reshape(-1)
        n = len(s)
        rank = int(np.ceil((1 - self.alpha) * (n + 1))) - 1
        rank = max(0, min(rank, n - 1))
        q = float(np.sort(s)[rank])
        self._result = ConformalResult(q_hat=q, alpha=self.alpha, n_calibration=n)
        return self._result

    @property
    def q_hat(self) -> float:
        if self._result is None:
            raise RuntimeError("Not calibrated yet.")
        return self._result.q_hat

    def prediction_set(self, softmax: torch.Tensor) -> torch.Tensor:
        """Return Boolean mask (batch × classes) of included classes."""
        q = self.q_hat
        scores = 1.0 - softmax                                    # 1 − p_y
        return scores <= q
