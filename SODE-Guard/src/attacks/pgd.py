"""Projected Gradient Descent ℓ∞ attack — PGD-40 of the manuscript.

Identical numerics to Madry et al. (2018) with the SODE-Guard tweak that the
forward pass is *stochastic*: each step we use a fresh single-path sample of
the SDE, which corresponds to the EOT (Expectation over Transformations)
adaptive attack of Athalye et al. (2018).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PGD:
    def __init__(self, model: nn.Module, *, eps: float, steps: int = 40,
                 alpha: float | None = None, norm: str = "linf",
                 random_start: bool = True, eot_samples: int = 1,
                 clip_min: float = 0.0, clip_max: float = 1.0):
        if norm != "linf":
            raise NotImplementedError("Only ℓ∞ PGD is reproduced; extend here for ℓ2.")
        self.model = model
        self.eps = float(eps)
        self.steps = int(steps)
        self.alpha = float(alpha if alpha is not None else 2.5 * eps / steps)
        self.random_start = bool(random_start)
        self.eot_samples = int(eot_samples)
        self.clip = (clip_min, clip_max)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_adv = x.clone().detach()
        if self.random_start:
            x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.eps, self.eps)
            x_adv = x_adv.clamp(*self.clip).detach()

        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            # EOT: average gradient across stochastic paths
            grad_acc = torch.zeros_like(x_adv)
            for _e in range(self.eot_samples):
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)
                grad = torch.autograd.grad(loss, x_adv, retain_graph=False)[0]
                grad_acc = grad_acc + grad.detach()
            grad_mean = grad_acc / self.eot_samples

            x_adv = x_adv.detach() + self.alpha * grad_mean.sign()
            delta = (x_adv - x).clamp(-self.eps, self.eps)
            x_adv = (x + delta).clamp(*self.clip).detach()
        return x_adv


def pgd_attack(model: nn.Module, x: torch.Tensor, y: torch.Tensor, *,
               eps: float, steps: int = 40, alpha: float | None = None) -> torch.Tensor:
    return PGD(model, eps=eps, steps=steps, alpha=alpha)(x, y)
