"""Diffusion-ellipticity projector ``g g^T ⪰ λ_0 I``.

Kept here for re-use by analysis scripts; the live training path already
applies the floor inside ``sde/integrator.py`` for performance reasons.
"""
from __future__ import annotations
import math
import torch


class EllipticityProjector:
    def __init__(self, floor: float = 1.0e-3):
        self.floor = float(floor)
        self._sqrt_floor = math.sqrt(self.floor)

    def __call__(self, g: torch.Tensor) -> torch.Tensor:
        """g : (..., d, m) — add a diagonal sqrt(λ_0) on the first min(d,m) entries."""
        d, m = g.shape[-2], g.shape[-1]
        k = min(d, m)
        idx = torch.arange(k, device=g.device)
        out = g.clone()
        out[..., idx, idx] = out[..., idx, idx] + self._sqrt_floor
        return out
