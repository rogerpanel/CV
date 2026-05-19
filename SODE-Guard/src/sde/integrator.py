"""Euler–Maruyama integrator for Itô SDEs used by SODE-Guard.

Solves
    dX_t = f_θ(X_t, t) dt + g_θ(X_t, t) dW_t,   t ∈ [t0, t1],
where ``g_θ(X_t, t)`` returns a (d, m) diffusion matrix with m = ``noise_dim``.
Supports virtual-Brownian-tree variance reduction and an ellipticity floor so that
``g g^⊤ ⪰ λ_0 I`` (a hard requirement of the Bismut–Elworthy–Li gradient bound used
in the anti-concentration certificate).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .brownian import VirtualBrownianTree


DriftFn = Callable[[Tensor, Tensor], Tensor]
DiffFn = Callable[[Tensor, Tensor], Tensor]  # returns (batch, d, m)


@dataclass
class EMConfig:
    t0: float = 0.0
    t1: float = 1.0
    dt: float = 0.05
    noise_dim: int = 16
    ellipticity_floor: float = 1.0e-3
    use_virtual_brownian: bool = True
    save_trajectory: bool = False


class EulerMaruyama:
    """Vectorised Euler–Maruyama integrator with optional path saving."""

    def __init__(self, cfg: EMConfig):
        self.cfg = cfg
        if cfg.t1 <= cfg.t0:
            raise ValueError("t1 must exceed t0")
        n = int(round((cfg.t1 - cfg.t0) / cfg.dt))
        if abs(n * cfg.dt - (cfg.t1 - cfg.t0)) > 1e-9:
            raise ValueError("dt must divide (t1 - t0) evenly")
        self.num_steps = n

    def _project_diffusion(self, g: Tensor) -> Tensor:
        """Add λ_0 I^{1/2} to enforce the ellipticity floor componentwise.

        We implement the floor by scaling the square root of each diagonal block
        so that the smallest singular value of g is at least sqrt(λ_0). This is
        cheap because g is (batch, d, m).
        """
        floor = math.sqrt(self.cfg.ellipticity_floor)
        # Frobenius-stable lower bound:   g ← g + floor · I_{:m}
        d = g.shape[-2]
        m = g.shape[-1]
        eye = torch.zeros_like(g)
        k = min(d, m)
        idx = torch.arange(k, device=g.device)
        eye[..., idx, idx] = floor
        return g + eye

    def __call__(
        self,
        x0: Tensor,
        drift: DriftFn,
        diffusion: DiffFn,
        *,
        seed: Optional[int] = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        cfg = self.cfg
        device, dtype = x0.device, x0.dtype
        x = x0
        traj = [x0] if cfg.save_trajectory else None

        bridge: Optional[VirtualBrownianTree] = None
        if cfg.use_virtual_brownian:
            w0 = torch.zeros(x0.shape[0], cfg.noise_dim, device=device, dtype=dtype)
            bridge = VirtualBrownianTree(cfg.t0, cfg.t1, w0, seed=seed or 0)

        t = torch.full((x0.shape[0], 1), cfg.t0, device=device, dtype=dtype)
        for step in range(self.num_steps):
            t_curr = cfg.t0 + step * cfg.dt
            t_next = t_curr + cfg.dt
            f = drift(x, t)                                  # (B, d)
            g = self._project_diffusion(diffusion(x, t))     # (B, d, m)

            if bridge is not None:
                dw = bridge.increment(t_curr, t_next)        # (B, m)
            else:
                dw = torch.randn(x0.shape[0], cfg.noise_dim, device=device, dtype=dtype) * math.sqrt(cfg.dt)

            # x_{k+1} = x_k + f dt + g dW
            x = x + f * cfg.dt + torch.einsum("bij,bj->bi", g, dw)
            t = t + cfg.dt

            if cfg.save_trajectory:
                traj.append(x)

        if cfg.save_trajectory:
            return x, torch.stack(traj, dim=1)
        return x


def integrate_sde(
    x0: Tensor,
    drift: DriftFn,
    diffusion: DiffFn,
    *,
    cfg: Optional[EMConfig] = None,
    seed: Optional[int] = None,
) -> Tensor:
    cfg = cfg or EMConfig()
    return EulerMaruyama(cfg)(x0, drift, diffusion, seed=seed)
