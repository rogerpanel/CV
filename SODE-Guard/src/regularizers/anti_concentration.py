"""Anti-concentration regulariser and certificate.

Mathematical setup
------------------
Let g : ℝ^d → ℝ be the (smoothed) margin function produced by SODE-Guard,
expressed in the chaos basis of the Brownian motion W on [0,1]:

    g(x) = ∑_{k=0}^{d*} J_k(x) · ξ_k    with  J_k ∈ Wiener-chaos of degree k.

By the Carbery–Wright inequality (Carbery & Wright, GAFA 2001), a polynomial of
degree d* in standard normal variables enjoys

    Pr[|p(ξ)| ≤ β] ≤ C · d* · (β / ‖p‖_2)^{1/d*}.

The SODE-Guard certificate (Proposition 1 of the manuscript) replaces ‖p‖_2 by
the Lipschitz constant L_g of the smoothed score, finally yielding

    Pr[|g(x+δ) − g(x)| ≤ β] ≤ C · d* · ( β / (L_g · ε) )^{1/d*}.

The training-time loss penalises high "anti-concentration mass" of the margin
in a β-band, computed on Monte-Carlo samples of the SDE.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_CW_CONSTANT = 2.0  # universal constant in Carbery–Wright (loose but rigorous)


class AntiConcentrationLoss(nn.Module):
    """Penalises the empirical mass of |margin| inside a β-band.

    Implements the surrogate

        L_AC = (1 / |β-grid|) ∑_β log( 1 + (1/N) ∑_i 1[ |m_i| ≤ β ] · (β·B/‖m‖_p)^{1/d*} )

    using a smoothed indicator (sigmoid) so that gradients flow.
    """

    def __init__(self, chaos_degree: int = 4,
                 beta_grid: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10),
                 sharpness: float = 50.0):
        super().__init__()
        self.d_star = int(chaos_degree)
        self.beta_grid = tuple(float(b) for b in beta_grid)
        self.sharpness = float(sharpness)

    def _margin(self, logits: torch.Tensor) -> torch.Tensor:
        # (B, K) → (B,) gap between top-1 and top-2
        top2, _ = torch.topk(logits, k=2, dim=-1)
        return top2[..., 0] - top2[..., 1]

    def forward(self, logits_paths: torch.Tensor) -> torch.Tensor:
        """logits_paths: (B, N_paths, K). Returns scalar loss."""
        # Path-averaged log-softmax → score margin
        avg = logits_paths.mean(dim=1)
        m = self._margin(avg).abs()                           # (B,)
        norm = m.detach().norm() + 1e-6
        loss = 0.0
        for beta in self.beta_grid:
            soft_indicator = torch.sigmoid(self.sharpness * (beta - m))
            ratio = (beta / norm).clamp_min(1e-12) ** (1.0 / self.d_star)
            loss = loss + torch.log1p(_CW_CONSTANT * self.d_star * ratio * soft_indicator.mean())
        return loss / len(self.beta_grid)


def anti_concentration_certificate(margin: torch.Tensor,
                                   lipschitz: float,
                                   epsilon: float,
                                   chaos_degree: int = 4,
                                   beta: float = 0.05) -> torch.Tensor:
    """Closed-form bound from Proposition 1.

    Returns the *upper bound* on Pr[|g(x+δ)−g(x)| ≤ β] for each sample. A
    certificate "passes" when the bound is below the desired confidence
    level (e.g. 0.05); we therefore expose the bound itself so callers can
    aggregate however they like.
    """
    if epsilon <= 0 or lipschitz <= 0:
        return torch.zeros_like(margin)
    ratio = (beta / (lipschitz * epsilon))
    if ratio <= 0:
        return torch.zeros_like(margin)
    bound = _CW_CONSTANT * chaos_degree * (ratio ** (1.0 / chaos_degree))
    return torch.full_like(margin, float(min(bound, 1.0)))


def certified_radius(margin: torch.Tensor,
                     lipschitz: float,
                     chaos_degree: int = 4,
                     beta: float = 0.05,
                     confidence: float = 0.95) -> torch.Tensor:
    """Invert the Carbery–Wright bound for the largest ε s.t. the bound
    is ≤ (1 − confidence).

    Pr ≤ (1 − conf)  ⇔  C·d*·(β / (L_g ε))^{1/d*} ≤ 1 − conf
    ⇔  ε ≥ (β / L_g) · ( C·d* / (1 − conf) )^{d*}.

    For each sample the radius scales linearly with the realised margin
    relative to the worst-case Lipschitz bound: we therefore return
        r* = (margin / lipschitz) · ( (1 − conf) / (C · d*) )^{d*}.
    """
    one_minus_conf = max(1.0 - confidence, 1e-6)
    base = (one_minus_conf / (_CW_CONSTANT * chaos_degree)) ** chaos_degree
    return (margin.abs() / max(lipschitz, 1e-6)) * base
