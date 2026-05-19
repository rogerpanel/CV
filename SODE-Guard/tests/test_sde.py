"""Unit tests for the SDE integrator."""
import math
import pytest
import torch

from src.sde.integrator import EulerMaruyama, EMConfig
from src.sde.brownian import VirtualBrownianTree


def test_em_geometric_brownian():
    # dX = μ X dt + σ X dW with closed-form mean μ X₀
    torch.manual_seed(0)
    d, B = 1, 4096
    x0 = torch.ones(B, d)
    mu, sigma = 0.1, 0.2

    def drift(x, t): return mu * x
    def diffusion(x, t): return (sigma * x).unsqueeze(-1)   # (B, d, 1)

    em = EulerMaruyama(EMConfig(t0=0.0, t1=1.0, dt=0.01, noise_dim=1, ellipticity_floor=0.0))
    xT = em(x0, drift, diffusion)
    expected_mean = math.exp(mu * 1.0) * 1.0
    assert abs(xT.mean().item() - expected_mean) < 0.05


def test_virtual_brownian_reproducible():
    w0 = torch.zeros(2, 4)
    a = VirtualBrownianTree(0.0, 1.0, w0, seed=123).increment(0.0, 0.5)
    b = VirtualBrownianTree(0.0, 1.0, w0, seed=123).increment(0.0, 0.5)
    assert torch.equal(a, b)


def test_ellipticity_floor():
    em = EulerMaruyama(EMConfig(t0=0.0, t1=0.1, dt=0.05, noise_dim=4,
                                ellipticity_floor=1e-2))
    g = torch.zeros(8, 6, 4)
    g_proj = em._project_diffusion(g)
    # smallest singular value should be ≥ sqrt(1e-2)
    s = torch.linalg.svdvals(g_proj)
    assert s.min().item() >= math.sqrt(1e-2) - 1e-6
