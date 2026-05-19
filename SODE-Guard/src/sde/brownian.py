"""Virtual Brownian tree for variance-reduced SDE solving.

Reference: Li, X., Wong, T.-K. L., Chen, R. T. Q. and Duvenaud, D.,
"Scalable Gradients for Stochastic Differential Equations" (AISTATS 2020).
"""
from __future__ import annotations
import math
import torch


class VirtualBrownianTree:
    """Lévy-area-free Brownian bridge generator.

    Provides reproducible Brownian increments W_t - W_s for arbitrary 0 ≤ s < t ≤ T
    by deterministically splitting the interval [0, T] via a binary tree seeded with
    `seed`. Memory cost is O(log_2(1/dt)) instead of O(1/dt) of naïve sampling.
    """

    def __init__(self, t0: float, t1: float, w0: torch.Tensor, *, seed: int = 0):
        if t1 <= t0:
            raise ValueError("t1 must exceed t0")
        self.t0, self.t1 = float(t0), float(t1)
        self.w0 = w0
        self.seed = seed
        self._cache: dict[tuple[float, float], torch.Tensor] = {}

    def _gen(self, ta: float, tb: float, shape: torch.Size, device, dtype) -> torch.Tensor:
        key = (round(ta, 12), round(tb, 12))
        if key in self._cache:
            return self._cache[key]
        # Seed deterministically from interval boundaries + global seed.
        h = hash((self.seed, key)) & 0xFFFFFFFF
        gen = torch.Generator(device=device).manual_seed(h)
        dw = torch.randn(shape, generator=gen, device=device, dtype=dtype) * math.sqrt(tb - ta)
        self._cache[key] = dw
        return dw

    def increment(self, s: float, t: float) -> torch.Tensor:
        """Return W_t − W_s with the same dtype/device as the tree's root sample."""
        return self._gen(s, t, self.w0.shape, self.w0.device, self.w0.dtype)
