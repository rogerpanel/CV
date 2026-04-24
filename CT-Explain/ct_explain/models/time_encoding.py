"""Multi-scale learnable time encoding φ(Δt).

Paper: "Temporal encoding φ(Δt) with learnable time constants {τ_k}."
Implementation follows Xu et al. (2020) Time2Vec-style sinusoidal encoding
with per-frequency trainable scales so that the temporal attention of
Equation 2 can adapt to process-specific inter-arrival distributions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEncoding(nn.Module):
    """Time2Vec-style encoding Δt → ℝ^{d_φ}.

    φ(Δt)[0]      = ω_0 · Δt + b_0
    φ(Δt)[1..d_φ] = sin(ω_k · Δt + b_k)      (k = 1 … d_φ − 1)

    Each ω_k = 1/τ_k is a learnable inverse-time-constant initialised on a
    geometric grid spanning [1, 1e4] so that multiple temporal scales
    (milliseconds, seconds, minutes, hours) are represented from the start.
    """

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("TimeEncoding dim must be >= 2")
        self.dim = dim

        # Geometric init: τ_k ∈ [1, 1e4] ⇒ ω_k ∈ [1e-4, 1].
        k = torch.arange(dim, dtype=torch.float32)
        log_omega = -k / max(dim - 1, 1) * math.log(1e4)
        self.omega = nn.Parameter(log_omega.exp())        # ω_k
        self.phase = nn.Parameter(torch.zeros(dim))       # b_k

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Δt: (...,) or (..., 1) → (..., dim)."""
        if delta_t.dim() > 0 and delta_t.shape[-1] != 1:
            delta_t = delta_t.unsqueeze(-1)
        theta = delta_t * self.omega + self.phase          # broadcast over last dim
        out = torch.empty_like(theta)
        out[..., :1] = theta[..., :1]                      # linear component
        out[..., 1:] = torch.sin(theta[..., 1:])
        return out
