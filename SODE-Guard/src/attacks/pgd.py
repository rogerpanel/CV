"""Projected Gradient Descent (Madry et al., 2018), l∞ or l2.

``model`` is any callable returning logits. For SODE-Guard pass the same
predictor that is evaluated (e.g. ``eot_autoattack.fixed_predictor``) so the
attack does not target a different function than the one being scored.
Inputs are z-scored features, so no box constraint is applied by default.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from .eot_autoattack import _project, _step, _random_start


class PGD:
    def __init__(self, model, *, eps: float, steps: int = 40,
                 alpha: Optional[float] = None, norm: str = "linf",
                 random_start: bool = True, eot_samples: int = 1,
                 clip_min: Optional[float] = None, clip_max: Optional[float] = None):
        if norm not in {"linf", "l2"}:
            raise ValueError("norm must be 'linf' or 'l2'")
        self.model = model
        self.eps = float(eps)
        self.steps = int(steps)
        self.alpha = float(alpha if alpha is not None else 2.5 * eps / steps)
        self.norm = norm
        self.random_start = bool(random_start)
        self.eot_samples = int(eot_samples)
        self.clip = (clip_min, clip_max)

    def _clip(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.clip
        return x if lo is None and hi is None else x.clamp(lo, hi)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        delta = _random_start(x, self.eps, self.norm) if self.random_start else torch.zeros_like(x)
        x_adv = self._clip(x + delta).detach()
        for _ in range(self.steps):
            grad = torch.zeros_like(x_adv)
            for _e in range(self.eot_samples):
                xr = x_adv.detach().requires_grad_(True)
                loss = F.cross_entropy(self.model(xr), y)
                grad += torch.autograd.grad(loss, xr)[0]
            x_adv = x_adv.detach() + self.alpha * _step(grad, self.norm)
            x_adv = self._clip(x + _project(x_adv - x, self.eps, self.norm)).detach()
        return x_adv


def pgd_attack(model, x: torch.Tensor, y: torch.Tensor, *,
               eps: float, steps: int = 40, alpha: Optional[float] = None,
               norm: str = "linf") -> torch.Tensor:
    return PGD(model, eps=eps, steps=steps, alpha=alpha, norm=norm)(x, y)
