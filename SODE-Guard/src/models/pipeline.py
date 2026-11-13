"""Deployed-model wrapper used by the v2 re-run.

``Pipeline(base, prep, K)`` maps z-scored flow features x to logits as

    logits(x) = base(prep(x)) + logit_bias

* ``prep`` is an optional differentiable input encoding (``QuantileEncoder``),
  so attacks and certificates still act in the z-scored feature space and the
  threat model (eps in z-units) is unchanged.
* ``logit_bias`` holds the post-hoc logit adjustment tuned on validation data
  (module 1). It is part of the deployed predictor, so attacks see it too.

For SODE-Guard-style bases the wrapper forwards ``sample_logits`` and
``forward_mean`` (and ``cfg``), so every attack in ``src/attacks`` works on it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm


class QuantileEncoder(nn.Module):
    """Differentiable, continuous piecewise-linear quantile map to N(0,1) scores.

    Knots are training quantiles; tied quantiles are merged (their target is
    the mean of the tied levels, as in sklearn's QuantileTransformer), and knots
    closer than ``min_gap`` z-units are merged too, so every slope is finite.
    Inputs beyond the outermost knots are clamped (flat).
    """

    def __init__(self, X: np.ndarray, n_quantiles: int = 64, min_gap: float = 0.05,
                 clip: float = 3.0):
        super().__init__()
        levels = np.linspace(0.0, 1.0, n_quantiles)
        target = np.clip(norm.ppf(np.clip(levels, 1e-3, 1 - 1e-3)), -clip, clip)
        xs_all, ys_all = [], []
        for j in range(X.shape[1]):
            q = np.quantile(X[:, j], levels)
            xs, ys, grp = [], [], []
            for qi, ti in zip(q, target):
                if xs and qi - xs[-1] < min_gap:
                    grp.append(ti)
                    ys[-1] = float(np.mean(grp))
                else:
                    xs.append(float(qi)); ys.append(float(ti)); grp = [ti]
            if len(xs) < 2:                       # (almost) constant feature: identity on [-10, 10]
                xs, ys = [-10.0, 10.0], [-10.0, 10.0]
            xs_all.append(xs); ys_all.append(ys)
        L = max(len(v) for v in xs_all)
        kx = np.zeros((X.shape[1], L)); ky = np.zeros((X.shape[1], L))
        for j, (xs, ys) in enumerate(zip(xs_all, ys_all)):
            pad = L - len(xs)
            kx[j] = xs + [xs[-1] + 1.0 + i for i in range(pad)]
            ky[j] = ys + [ys[-1]] * pad
        self.register_buffer("kx", torch.tensor(kx, dtype=torch.float32))
        self.register_buffer("ky", torch.tensor(ky, dtype=torch.float32))

    def max_slope(self) -> float:
        dx = self.kx[:, 1:] - self.kx[:, :-1]
        dy = (self.ky[:, 1:] - self.ky[:, :-1]).abs()
        return float((dy / dx).max())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.t().contiguous()                                   # (p, B)
        L = self.kx.shape[1]
        idx = torch.searchsorted(self.kx, xt).clamp(1, L - 1)
        x0, x1 = self.kx.gather(1, idx - 1), self.kx.gather(1, idx)
        y0, y1 = self.ky.gather(1, idx - 1), self.ky.gather(1, idx)
        t = ((xt - x0) / (x1 - x0)).clamp(0.0, 1.0)
        return (y0 + t * (y1 - y0)).t()


class Pipeline(nn.Module):
    def __init__(self, base: nn.Module, prep: Optional[nn.Module], num_classes: int):
        super().__init__()
        self.base = base
        self.prep = prep if prep is not None else nn.Identity()
        self.register_buffer("logit_bias", torch.zeros(num_classes))

    @property
    def cfg(self):
        return self.base.cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(self.prep(x)) + self.logit_bias

    def sample_logits(self, x: torch.Tensor, seeds: Sequence[int]) -> torch.Tensor:
        return self.base.sample_logits(self.prep(x), seeds) + self.logit_bias

    def forward_mean(self, x: torch.Tensor, n_paths: Optional[int] = None,
                     seeds: Optional[Sequence[int]] = None) -> torch.Tensor:
        return self.base.forward_mean(self.prep(x), n_paths=n_paths, seeds=seeds) + self.logit_bias
