"""Randomised-smoothing certificate for a randomised base classifier (module 3).

The smoothed classifier is

    G(x) = argmax_c  P_{eps, W}[ f(x + sigma * eps, W) = c ],   eps ~ N(0, I),

where f is the deployed SODE-Guard predictor (argmax of the M-path mean logits)
and W are its Brownian paths, drawn independently of x. Writing
h_c(z) = P_W[f(z, W) = c] in [0, 1], G(x) = argmax_c E_eps[h_c(x + sigma eps)],
and the certificate of Cohen et al. (2019) holds for soft base classifiers with
values in [0, 1] (Salman et al., 2019, Lemma 2), so the usual CERTIFY procedure
applies with samples drawn jointly over (eps, W): if the one-sided
Clopper-Pearson lower bound p_A on P[f = c_A] exceeds 1/2, G is constant on the
l2 ball of radius sigma * Phi^{-1}(p_A) around x.

Equivalently, the input noise is a pure-diffusion prefix of the SDE: on t in
[0, 1], dX = sigma dW from X_0 = x gives X_1 = x + sigma eps, after which the
learned SDE runs as before.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.stats import beta as beta_dist, norm

Predict = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def _counts(f: Predict, x: torch.Tensor, sigma: float, n: int, K: int, batch: int) -> torch.Tensor:
    counts = torch.zeros(x.shape[0], K, device=x.device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        noisy = x.repeat_interleave(b, dim=0) + sigma * torch.randn(x.shape[0] * b, *x.shape[1:], device=x.device)
        pred = f(noisy).argmax(-1).view(x.shape[0], b)
        counts.scatter_add_(1, pred, torch.ones_like(pred, dtype=counts.dtype))
        done += b
    return counts


@torch.no_grad()
def certify_randomized(f: Predict, x: torch.Tensor, *, sigma: float, K: int, n0: int = 100,
                       n: int = 10_000, alpha: float = 1e-3, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Returns (prediction, radius); prediction -1 means abstain (radius 0).

    ``f`` must draw fresh internal randomness on every call (for SODE-Guard,
    ``lambda z: model.forward_mean(z, n_paths=M)``)."""
    c_hat = _counts(f, x, sigma, n0, K, batch).argmax(-1)
    counts = _counts(f, x, sigma, n, K, batch)
    k = counts.gather(1, c_hat[:, None]).squeeze(1).cpu().numpy().astype(int)
    p_lower = np.where(k > 0, beta_dist.ppf(alpha, k, n - k + 1), 0.0)
    radius = np.where(p_lower > 0.5, sigma * norm.ppf(np.clip(p_lower, 1e-12, 1 - 1e-12)), 0.0)
    pred = np.where(p_lower > 0.5, c_hat.cpu().numpy(), -1)
    return pred, radius


def radius_ceiling(sigma: float, n: int, alpha: float = 1e-3) -> float:
    """Largest radius certifiable with n samples (all n votes for c_A)."""
    return float(sigma * norm.ppf(alpha ** (1.0 / n)))
