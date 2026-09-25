"""Randomised-smoothing baseline (Cohen et al., 2019) with a σ sweep.

CERTIFY: n0 noisy samples select the class, n independent samples give a
one-sided Clopper–Pearson lower bound p_A on its probability; the l2 radius is
σ·Φ⁻¹(p_A) when p_A > 1/2, otherwise abstain. The base classifier should be
trained with Gaussian noise augmentation at the same σ.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import beta as beta_dist, norm


@dataclass
class SmoothingReport:
    sigma: float
    clean_accuracy: float          # certified-and-correct counted at r = 0
    radii_l2: np.ndarray           # 0 for abstentions and wrong predictions

    def certified_accuracy(self, r: float) -> float:
        return float((self.radii_l2 >= r).mean()) if r > 0 else self.clean_accuracy


def _counts(base, x: torch.Tensor, sigma: float, n: int, K: int, batch: int = 64) -> torch.Tensor:
    counts = torch.zeros(x.shape[0], K, device=x.device)
    done = 0
    while done < n:
        b = min(batch, n - done)
        noisy = x.repeat_interleave(b, dim=0) + sigma * torch.randn(x.shape[0] * b, *x.shape[1:], device=x.device)
        pred = base(noisy).argmax(-1).view(x.shape[0], b)
        counts.scatter_add_(1, pred, torch.ones_like(pred, dtype=counts.dtype))
        done += b
    return counts


@torch.no_grad()
def certify_smoothing(base, x: torch.Tensor, y: torch.Tensor, *, sigma: float, K: int,
                      n0: int = 100, n: int = 10_000, alpha: float = 1e-3) -> np.ndarray:
    c_hat = _counts(base, x, sigma, n0, K).argmax(-1)
    counts = _counts(base, x, sigma, n, K)
    k = counts.gather(1, c_hat[:, None]).squeeze(1).cpu().numpy().astype(int)
    p_lower = np.where(k > 0, beta_dist.ppf(alpha, k, n - k + 1), 0.0)
    radius = np.where(p_lower > 0.5, sigma * norm.ppf(np.clip(p_lower, 1e-12, 1 - 1e-12)), 0.0)
    correct = (c_hat == y).cpu().numpy()
    return np.where(correct & (p_lower > 0.5), radius, 0.0)


@torch.no_grad()
def randomized_smoothing_sensitivity(base, dataloader, *, K: int, device: str = "cpu",
                                     sigmas: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
                                     n0: int = 100, n: int = 10_000, alpha: float = 1e-3,
                                     max_batches: int | None = None) -> list[SmoothingReport]:
    base.eval()
    reports = []
    for sigma in sigmas:
        radii, correct_abstain_free = [], []
        for i, (x, y) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            r = certify_smoothing(base, x, y, sigma=sigma, K=K, n0=n0, n=n, alpha=alpha)
            radii.append(r)
            correct_abstain_free.append(r > 0)
        r = np.concatenate(radii)
        reports.append(SmoothingReport(sigma=sigma,
                                       clean_accuracy=float(np.concatenate(correct_abstain_free).mean()),
                                       radii_l2=r))
    return reports
