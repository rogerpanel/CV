"""Moore–Penrose pseudo-inverse of the (128 × 16) diffusion.

Reviewer 2 asked how the BEL identity uses g^{-1} when g_θ is not
square. The answer used by the revised Theorem 2 is the Moore–Penrose
pseudo-inverse

    g^+ = (g^T g)^{-1} g^T ∈ R^{16 × 128}

which becomes numerically well-conditioned when the ellipticity floor
``λ_0 = 10^{-3}`` is applied so that g^T g ⪰ λ_0 · I. We expose the
projection here so downstream callers (the BEL gradient estimator and
the analysis notebooks) can share the same implementation.
"""
from __future__ import annotations
import torch


def moore_penrose_diffusion(g: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    """Return g^+ for a (batch, d, m) diffusion matrix.

    Applies the ellipticity floor by ridging g^T g with ``floor · I``
    before inversion, which corresponds exactly to the training-time
    projector in ``regularizers/ellipticity.py``.
    """
    if g.ndim != 3:
        raise ValueError("g must have shape (batch, d, m)")
    B, d, m = g.shape
    gtg = torch.einsum("bij,bik->bjk", g, g)         # (B, m, m)
    reg = floor * torch.eye(m, device=g.device, dtype=g.dtype).expand(B, -1, -1)
    inv = torch.linalg.solve(gtg + reg, torch.eye(m, device=g.device, dtype=g.dtype).expand(B, -1, -1))
    g_plus = torch.einsum("bjk,bik->bji", inv, g)    # (B, m, d)
    return g_plus


def condition_number(g: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
    """Return the batched condition number κ(g^T g + floor·I) for diagnostics."""
    gtg = torch.einsum("bij,bik->bjk", g, g)
    reg = floor * torch.eye(gtg.shape[-1], device=g.device, dtype=g.dtype).expand_as(gtg)
    s = torch.linalg.svdvals(gtg + reg)
    return s[:, 0] / s[:, -1].clamp_min(1e-12)
