"""Carlini–Wagner l2 attack (Carlini & Wagner, S&P 2017), untargeted.

Optimises the perturbation δ directly (features are z-scored, so there is no
natural [0, 1] box for the tanh reparameterisation); an optional box clamps
the result.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


class CarliniWagnerL2:
    def __init__(self, model, *, c: float = 1.0, kappa: float = 0.0,
                 iterations: int = 100, lr: float = 0.01,
                 clip_min: Optional[float] = None, clip_max: Optional[float] = None):
        self.model = model
        self.c = float(c)
        self.kappa = float(kappa)
        self.iterations = int(iterations)
        self.lr = float(lr)
        self.clip = (clip_min, clip_max)

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.clip
        return x if lo is None and hi is None else x.clamp(lo, hi)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        delta = torch.zeros_like(x, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=self.lr)
        for _ in range(self.iterations):
            logits = self.model(self._clip(x + delta))
            one_hot = F.one_hot(y, num_classes=logits.shape[-1]).bool()
            real = logits[one_hot]
            other = logits.masked_fill(one_hot, float("-inf")).max(dim=-1).values
            f_loss = torch.clamp(real - other + self.kappa, min=0.0)
            l2 = (delta ** 2).flatten(1).sum(dim=-1)
            loss = (l2 + self.c * f_loss).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self._clip(x + delta).detach()
