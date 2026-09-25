"""Adaptive attacks on a toy classifier and the BEL estimator on a linear SDE."""
import math

import torch

from src.attacks import PGD
from src.attacks.eot_autoattack import apgd, eot_pgd, expected_margin_pgd, square_attack
from src.theory.bel import bel_gradient


def toy():
    torch.manual_seed(0)
    w = torch.randn(10, 3)
    x = torch.randn(64, 10)
    y = (x @ w).argmax(1)
    return (lambda z: z @ w), x, y


def acc(f, x, y):
    return (f(x).argmax(1) == y).float().mean().item()


def test_attacks_respect_budget_and_reduce_accuracy():
    f, x, y = toy()
    eps = 0.5
    for x_adv in (eot_pgd(f, x, y, eps=eps, steps=20, eot=1),
                  apgd(f, x, y, eps=eps, n_iter=20, loss="ce", eot=1),
                  apgd(f, x, y, eps=eps, n_iter=20, loss="dlr", eot=1),
                  expected_margin_pgd(f, x, y, eps=eps, steps=20),
                  square_attack(f, x, y, eps=eps, n_queries=300),
                  PGD(f, eps=eps, steps=20)(x, y)):
        assert (x_adv - x).abs().max() <= eps + 1e-5
        assert acc(f, x_adv, y) < acc(f, x, y)


def test_l2_pgd_budget():
    f, x, y = toy()
    x_adv = PGD(f, eps=0.7, steps=10, norm="l2")(x, y)
    assert ((x_adv - x).norm(dim=1) <= 0.7 + 1e-5).all()


def test_bel_matches_analytic_gradient_linear_sde():
    """dX = A X dt + σ dW (m = d): ∇ E||X_T||² = 2 (Mⁿ)ᵀ Mⁿ x for the EM scheme."""
    torch.manual_seed(0)
    d, dt, T, sigma = 2, 0.1, 1.0, 0.8
    A = torch.tensor([[-0.3, 0.2], [0.0, -0.5]], dtype=torch.float64)
    f = lambda z, t: z @ A.T
    g = lambda z, t: sigma * torch.eye(d, dtype=z.dtype).expand(z.shape[0], d, d)
    x0 = torch.tensor([[0.7, -0.4]], dtype=torch.float64)
    est = bel_gradient(f, g, x0, lambda z: z.pow(2).sum(-1), T=T, dt=dt, n_paths=200_000,
                       floor=1e-9, generator=torch.Generator().manual_seed(1))
    Mn = torch.linalg.matrix_power(torch.eye(d, dtype=torch.float64) + A * dt, int(round(T / dt)))
    exact = 2 * (Mn.T @ Mn @ x0.T).T
    assert torch.allclose(est, exact, atol=0.08), (est, exact)
