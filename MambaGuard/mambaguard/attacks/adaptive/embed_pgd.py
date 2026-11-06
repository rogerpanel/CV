"""Embedding-space PGD with Expectation-Over-Transformation (EoT).

S&P 2027 audit (C.12): the certified smoothing defence breaks whenever
the attacker averages gradients across the smoothing noise distribution
(Athalye-Carlini-Wagner 2018). This module implements the adaptive
baseline the manuscript must defeat:

    delta_{t+1} = Proj_{||.||_2 <= eps} (
        delta_t + alpha * sign( (1/K) sum_k  grad_{delta} L(f(x + delta + eta_k), y) )
    )

with ``K`` EoT samples per step and ``eta_k ~ N(0, sigma^2 I)``.

The attack lives in payload-embedding space (key ``"p"``) because the
MambaGuard classifier operates over canonicalised embeddings, matching
the smoothing certificate's input domain.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_logits(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        return out["logits"]
    return out


def _project_l2(delta: torch.Tensor, epsilon: float) -> torch.Tensor:
    flat = delta.reshape(delta.shape[0], -1)
    n = flat.norm(p=2, dim=-1, keepdim=True)
    factor = torch.clamp(epsilon / n.clamp(min=1e-12), max=1.0)
    return (flat * factor).reshape_as(delta)


class EmbedPGD:
    """Adaptive PGD in payload-embedding space with EoT.

    Parameters
    ----------
    epsilon : float
        L2 radius of the allowed perturbation.
    steps : int
        PGD iterations.
    alpha : float
        Per-step size.
    eot_sigma : float
        Smoothing std used for EoT (should match the defender's).
    eot_samples : int
        Number of noise draws per gradient estimate.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        steps: int = 100,
        alpha: float | None = None,
        eot_sigma: float = 0.05,
        eot_samples: int = 8,
        random_start: bool = True,
        input_key: str = "p",
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if eot_samples <= 0:
            raise ValueError("eot_samples must be > 0")
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.alpha = float(alpha) if alpha is not None else float(epsilon / max(steps // 4, 1))
        self.eot_sigma = float(eot_sigma)
        self.eot_samples = int(eot_samples)
        self.random_start = bool(random_start)
        self.input_key = input_key

    # ------------------------------------------------------------------
    def forward(
        self,
        model: nn.Module,
        batch: Mapping[str, Any],
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        was_training = model.training
        model.eval()
        x0 = batch[self.input_key].detach()
        if y is None:
            y = batch["labels"]
        if self.random_start:
            delta = torch.randn_like(x0) * (self.epsilon / max(x0[0].numel() ** 0.5, 1.0))
            delta = _project_l2(delta, self.epsilon)
        else:
            delta = torch.zeros_like(x0)

        for _ in range(self.steps):
            delta = delta.detach().requires_grad_(True)
            grad_accum = torch.zeros_like(delta)
            for _k in range(self.eot_samples):
                eta = torch.randn_like(x0) * self.eot_sigma
                adv_batch = dict(batch)
                adv_batch[self.input_key] = x0 + delta + eta
                out = model(adv_batch)
                logits = _extract_logits(out)
                loss = F.cross_entropy(logits, y)
                g = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                grad_accum = grad_accum + g.detach()
            grad = grad_accum / self.eot_samples
            # L2 PGD step.
            flat = grad.reshape(grad.shape[0], -1)
            nrm = flat.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            step = (self.alpha * (flat / nrm)).reshape_as(grad)
            delta = _project_l2(delta.detach() + step, self.epsilon)

        x_adv = (x0 + delta).detach()
        if was_training:
            model.train()
        out_batch = dict(batch)
        out_batch[self.input_key] = x_adv
        return out_batch

    def __call__(
        self, model: nn.Module, batch: Mapping[str, Any], y: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        return self.forward(model, batch, y)
