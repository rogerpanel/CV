"""Active-learning feedback loop.

Paper: "Active learning feedback: 12.3 corrections/week.
The UC-HGP framework is extended for online Bayesian updating."

We implement two building blocks:

* `ActiveLearningBuffer`  – ring buffer of analyst corrections with
  structured fields (alert id, ground-truth label, rationale, confidence).
* `BayesianUpdater`       – Gaussian-NG online posterior update over a
  classifier's final linear layer. This matches the "Bayesian online
  updating" description without touching the whole ODE backbone.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import torch
import torch.nn as nn


@dataclass
class Feedback:
    alert_id: str
    corrected_label: int
    rationale: str = ""
    analyst_confidence: float = 1.0          # 0..1
    features: Optional[torch.Tensor] = None  # (d,) — for Bayesian update


class ActiveLearningBuffer:
    def __init__(self, capacity: int = 4096) -> None:
        self._q: Deque[Feedback] = deque(maxlen=capacity)

    def push(self, fb: Feedback) -> None:
        self._q.append(fb)

    def __len__(self) -> int:
        return len(self._q)

    def drain(self) -> list[Feedback]:
        items = list(self._q)
        self._q.clear()
        return items


class BayesianUpdater:
    """Online Gaussian posterior update on a linear classifier head.

    Maintains (W_mean, W_cov) with a diagonal approximation. Each analyst
    correction (x, y) yields a one-step update weighted by analyst
    confidence. Stable under thousands of updates without degrading the
    base detector, because only the final linear head is modified.
    """

    def __init__(
        self,
        linear: nn.Linear,
        prior_var: float = 1.0,
        noise_var: float = 0.1,
    ) -> None:
        self.linear = linear
        with torch.no_grad():
            self.mu = linear.weight.detach().clone()
            self.sigma2 = torch.full_like(self.mu, prior_var)
        self.noise_var = noise_var

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update(self, x: torch.Tensor, y: int, confidence: float = 1.0) -> None:
        """One-sample Gaussian NG update on the row of class y."""
        x = x.to(self.mu.device)
        sigma2_y = self.sigma2[y]
        sigma2_new = 1.0 / (1.0 / sigma2_y + confidence * (x ** 2) / self.noise_var)
        mu_y = self.mu[y]
        mu_new = sigma2_new * (
            mu_y / sigma2_y + confidence * x / self.noise_var
        )
        self.mu[y] = mu_new
        self.sigma2[y] = sigma2_new
        self.linear.weight.data.copy_(self.mu)

    @torch.no_grad()
    def apply(self, feedback: list[Feedback]) -> dict:
        applied = 0
        for fb in feedback:
            if fb.features is None:
                continue
            self.update(fb.features, fb.corrected_label, fb.analyst_confidence)
            applied += 1
        return {"applied": applied, "buffer_size": len(feedback) - applied}
