"""Gaussian-noise corruption (sanity-check baseline)."""
from __future__ import annotations
import torch


class GaussianNoise:
    def __init__(self, sigma: float = 0.01):
        self.sigma = float(sigma)

    def __call__(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return (x + self.sigma * torch.randn_like(x)).clamp(0.0, 1.0)
