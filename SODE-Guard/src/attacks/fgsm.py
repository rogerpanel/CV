"""Fast Gradient Sign Method (Goodfellow et al., 2015)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSM:
    def __init__(self, model: nn.Module, eps: float,
                 clip_min: float = 0.0, clip_max: float = 1.0):
        self.model = model
        self.eps = float(eps)
        self.clip = (clip_min, clip_max)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        loss = F.cross_entropy(self.model(x), y)
        grad = torch.autograd.grad(loss, x)[0]
        x_adv = (x + self.eps * grad.sign()).clamp(*self.clip).detach()
        return x_adv


def fgsm_attack(model, x, y, eps):
    return FGSM(model, eps)(x, y)
