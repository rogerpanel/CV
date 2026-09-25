"""Theorem A: certified radius of the SDE-mean predictor, with MC correction.

Certified predictor: F^B(x) = E[P_B(ψ(X_T))], where P_B projects the logit
vector onto the l2 ball of radius B (1-Lipschitz, so F^B is L-Lipschitz with L
from ``lipschitz_constants``). Every pairwise margin F_y − F_k is then
sqrt(2)·L-Lipschitz, so a margin M(x) certifies the l2 radius M / (sqrt(2) L).

Monte-Carlo protocol (Cohen et al., 2019 style, fresh CSPRNG seeds):
    1. n0 selection paths → ŷ = argmax of the mean projected logits.
    2. n independent estimation paths → pairwise margins Z_k = (e_ŷ − e_k)ᵀ P_B(ψ),
       each bounded in [−sqrt(2) B, sqrt(2) B].
    3. One-sided lower confidence bound on every E Z_k with a union bound
       over the K − 1 competitors (Hoeffding or empirical Bernstein).
    4. If min_k LCB_k > 0: ŷ = argmax F^B(x) and r̂ = min_k LCB_k / (sqrt(2) L)
       is a valid l2 radius with probability ≥ 1 − α; otherwise abstain.
The l∞ radius follows from ||δ||_2 ≤ sqrt(p) ||δ||_∞ (p = 83 features).
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch

from ..models.sode_guard import fresh_seeds


def project_ball(v: torch.Tensor, B: float) -> torch.Tensor:
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v * torch.clamp(B / norm, max=1.0)


def hoeffding_width(n: int, value_range: float, log_term: float) -> float:
    return value_range * math.sqrt(log_term / (2.0 * n))


def bernstein_width(sample_var: torch.Tensor, n: int, value_range: float,
                    log_term: float) -> torch.Tensor:
    """Empirical Bernstein (Maurer & Pontil, 2009, Thm. 4), one-sided."""
    return torch.sqrt(2.0 * sample_var * log_term / n) + 7.0 * value_range * log_term / (3.0 * (n - 1))


@dataclass
class MeanCertificate:
    prediction: torch.Tensor      # (B,) ŷ, −1 when abstaining
    margin_lcb: torch.Tensor      # (B,) lower confidence bound on M(x)
    radius_l2: torch.Tensor       # (B,) 0 when abstaining
    radius_linf: torch.Tensor     # (B,)
    frac_projected: float         # fraction of path logits clipped by P_B
    L: float
    alpha: float
    n0: int
    n: int
    bound: str


@torch.no_grad()
def certify_mean(model, x: torch.Tensor, *, L: float, B: float,
                 n0: int = 64, n: int = 512, alpha: float = 1e-3,
                 bound: str = "bernstein", feature_dim: int | None = None,
                 chunk: int = 64) -> MeanCertificate:
    if bound not in {"hoeffding", "bernstein"}:
        raise ValueError("bound must be 'hoeffding' or 'bernstein'")
    model.eval()
    K = model.cfg.num_classes
    p = feature_dim or model.cfg.feature_dim

    sel = project_ball(model.sample_logits(x, fresh_seeds(n0)), B).mean(dim=1)
    y_hat = sel.argmax(dim=-1)                                            # (B,)

    sums = torch.zeros(x.shape[0], K, device=x.device)
    sq_sums = torch.zeros_like(sums)
    projected = 0
    seeds = fresh_seeds(n)
    for i in range(0, n, chunk):
        raw = model.sample_logits(x, seeds[i:i + chunk])                  # (B, c, K)
        projected += int((raw.norm(dim=-1) > B).sum())
        z = project_ball(raw, B)
        z = z.gather(-1, y_hat.view(-1, 1, 1).expand(-1, z.shape[1], 1)) - z   # (B, c, K) pairwise margins
        sums += z.sum(dim=1)
        sq_sums += (z ** 2).sum(dim=1)

    mean = sums / n
    value_range = 2.0 * math.sqrt(2.0) * B
    competitors = max(K - 1, 1)
    if bound == "hoeffding":
        width = torch.full_like(mean, hoeffding_width(n, value_range, math.log(competitors / alpha)))
    else:
        var = ((sq_sums - n * mean ** 2) / (n - 1)).clamp_min(0.0)
        width = bernstein_width(var, n, value_range, math.log(2.0 * competitors / alpha))
    lcb = mean - width
    lcb.scatter_(1, y_hat.view(-1, 1), float("inf"))                      # ignore k = ŷ
    margin_lcb = lcb.min(dim=-1).values

    certified = margin_lcb > 0
    radius_l2 = torch.where(certified, margin_lcb / (math.sqrt(2.0) * L), torch.zeros_like(margin_lcb))
    return MeanCertificate(
        prediction=torch.where(certified, y_hat, torch.full_like(y_hat, -1)),
        margin_lcb=margin_lcb,
        radius_l2=radius_l2,
        radius_linf=radius_l2 / math.sqrt(p),
        frac_projected=projected / (x.shape[0] * n),
        L=L, alpha=alpha, n0=n0, n=n, bound=bound,
    )
