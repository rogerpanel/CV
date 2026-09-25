"""Proposition B: single-path decision-flip bound for a fixed perturbation δ.

Let k* = argmax F(x) and G(x) = ψ_{k*}(X_T^x) − max_{k≠k*} ψ_k(X_T^x) be the
path-wise margin. Under the synchronous coupling,
|G(x+δ) − G(x)| ≤ sqrt(2)·||W_ψ||·||X_T^{x+δ} − X_T^x||, so for every β > 0

    P[path decision at x+δ ≠ k*] ≤ P[G(x) ≤ β] + 2 L² ||δ||² / β².

The first term involves the clean input only and is bounded from N fresh paths
by a Clopper–Pearson upper confidence limit; the β grid is covered by a union
bound. No union over classes is needed.
"""
from __future__ import annotations
import math

import torch
from scipy.stats import beta as beta_dist

from ..models.sode_guard import fresh_seeds


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))


@torch.no_grad()
def path_flip_bound(model, x: torch.Tensor, *, L: float, eps_l2: float,
                    n: int = 512, alpha: float = 1e-3,
                    beta_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
                    k_star: torch.Tensor | None = None, n0: int = 64) -> dict:
    """Per-example upper bound on the flip probability at ||δ||_2 ≤ eps_l2.

    k* is chosen on n0 separate paths so the counting sample stays independent.
    """
    model.eval()
    if k_star is None:
        k_star = model.sample_logits(x, fresh_seeds(n0)).mean(dim=1).argmax(dim=-1)
    logits = model.sample_logits(x, fresh_seeds(n))                     # (B, N, K)
    top = logits.gather(-1, k_star.view(-1, 1, 1).expand(-1, n, 1)).squeeze(-1)
    others = logits.scatter(-1, k_star.view(-1, 1, 1).expand(-1, n, 1), float("-inf"))
    G = top - others.max(dim=-1).values                                  # (B, N)

    a = alpha / len(beta_grid)
    best = torch.ones(x.shape[0])
    best_beta = torch.zeros(x.shape[0])
    for b in beta_grid:
        counts = (G <= b).sum(dim=1).cpu()
        near = torch.tensor([clopper_pearson_upper(int(c), n, a) for c in counts])
        total = (near + 2.0 * L ** 2 * eps_l2 ** 2 / b ** 2).clamp_max(1.0)
        better = total < best
        best = torch.where(better, total, best)
        best_beta = torch.where(better, torch.full_like(best_beta, b), best_beta)
    return {"flip_bound": best, "beta": best_beta, "k_star": k_star.cpu(),
            "path_error_rate": (G <= 0).float().mean(dim=1).cpu()}
