"""Carlini–Wagner ℓ2 attack (Carlini & Wagner, S&P 2017).

Simplified, untargeted variant — sufficient for the comparison numbers in
Table 3 of the manuscript.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CarliniWagnerL2:
    def __init__(self, model: nn.Module, *, c: float = 1.0,
                 kappa: float = 0.0, iterations: int = 100,
                 lr: float = 0.01, clip_min: float = 0.0, clip_max: float = 1.0):
        self.model = model
        self.c = float(c)
        self.kappa = float(kappa)
        self.iterations = int(iterations)
        self.lr = float(lr)
        self.clip = (clip_min, clip_max)

    @staticmethod
    def _to_tanh(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        scaled = (x - lo) / (hi - lo) * 2 - 1
        scaled = scaled.clamp(-1 + 1e-6, 1 - 1e-6)
        return torch.atanh(scaled)

    @staticmethod
    def _from_tanh(z: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return (torch.tanh(z) + 1) / 2 * (hi - lo) + lo

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        lo, hi = self.clip
        w = self._to_tanh(x.clone().detach(), lo, hi).requires_grad_(True)
        opt = torch.optim.Adam([w], lr=self.lr)
        x_orig = x.detach()
        for _ in range(self.iterations):
            x_adv = self._from_tanh(w, lo, hi)
            logits = self.model(x_adv)
            one_hot = F.one_hot(y, num_classes=logits.shape[-1]).bool()
            real = logits[one_hot]
            other = logits.masked_fill(one_hot, float("-inf")).max(dim=-1).values
            f_loss = torch.clamp(real - other + self.kappa, min=0.0)
            l2 = ((x_adv - x_orig) ** 2).flatten(1).sum(dim=-1)
            loss = (l2 + self.c * f_loss).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return self._from_tanh(w, lo, hi).detach()
