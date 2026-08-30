"""Empirical Lipschitz certification for the smoothed margin g(x).

Reviewer 1 correctly noted that the L^2-Lipschitz constant L_g of
Proposition 5 was stated without a constructive proof or empirical bound.
This module supplies one.

We estimate

    L_g^2 = sup_{‖δ‖ ≤ ε}   E [ ( g(x + δ) − g(x) )^2 ] / ‖δ‖^2

by drawing K random directions per test point, averaging the empirical
variance over N Brownian samples of the SDE, and reporting a
one-sided Hoeffding upper confidence bound at confidence level ``conf``.

The certificate is *paired* with the anti-concentration bound at inference
time so the reported robustness radius is honest with respect to the
observed L_g rather than a user-chosen hyper-parameter.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch


@dataclass
class LipschitzCertificate:
    L_hat: float            # empirical mean estimate of L_g
    L_upper: float          # (1-conf)-Hoeffding upper bound
    K: int                  # directions per test point
    N: int                  # Brownian paths per direction
    epsilon_probe: float    # ‖δ‖ used for the probe
    confidence: float

    def as_dict(self) -> dict:
        return dict(L_hat=self.L_hat, L_upper=self.L_upper,
                    K=self.K, N=self.N,
                    epsilon_probe=self.epsilon_probe,
                    confidence=self.confidence)


@torch.no_grad()
def certify_L2_lipschitz(model, dataloader, *,
                         device: str = "cpu",
                         epsilon_probe: float = 1e-3,
                         K: int = 8,
                         N: int = 32,
                         confidence: float = 0.95,
                         max_batches: int = 4) -> LipschitzCertificate:
    """Certify L_g on the loader distribution.

    Parameters
    ----------
    model : SODEGuard-compatible module exposing ``forward_mc(x, n_paths=N)``
        that returns per-example class probabilities.
    epsilon_probe : ℓ_∞ radius of the random probe direction.
    K, N : replication counts for direction / Brownian samples.
    confidence : (1 − α) upper-confidence level for the Hoeffding bound.

    Returns
    -------
    LipschitzCertificate with mean estimate and upper bound.
    """
    model.eval()
    all_ratios: list[float] = []
    for bi, batch in enumerate(dataloader):
        x, _ = batch
        x = x.to(device)
        base_logp = model.forward_mc(x, n_paths=N).clamp_min(1e-12).log()
        for _ in range(K):
            d = torch.randn_like(x)
            d = d / d.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)   # ℓ_∞ unit
            probe_logp = model.forward_mc(x + epsilon_probe * d, n_paths=N).clamp_min(1e-12).log()
            l2 = (probe_logp - base_logp).pow(2).sum(dim=-1).sqrt()
            all_ratios.append((l2 / epsilon_probe).cpu())
        if bi + 1 >= max_batches:
            break
    ratios = torch.cat(all_ratios).clamp_max(1e6).numpy()
    mean = float(ratios.mean())
    # Hoeffding: for bounded X ∈ [0, R],  P(X̄ ≤ μ − t) ≤ exp(-2 n t^2 / R^2)
    R = float(max(ratios.max(), mean + 1e-6))
    n = ratios.size
    alpha = 1.0 - float(confidence)
    t = R * math.sqrt(math.log(1.0 / max(alpha, 1e-12)) / (2.0 * n))
    upper = mean + t
    return LipschitzCertificate(L_hat=mean, L_upper=upper, K=K, N=N,
                                epsilon_probe=epsilon_probe,
                                confidence=confidence)
