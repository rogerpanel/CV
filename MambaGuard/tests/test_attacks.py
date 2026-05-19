"""Tests for adversarial attacks."""
from __future__ import annotations

import pytest


class _ToyModel:
    """Minimal forward(batch_dict) -> logits wrapper around a Linear."""

    def __init__(self, in_dim: int = 16, n_classes: int = 3) -> None:
        import torch

        self.lin = torch.nn.Linear(in_dim, n_classes)
        self.training = False

    def __call__(self, batch):
        return self.lin(batch["p"])

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = bool(mode)
        return self

    def parameters(self):
        return self.lin.parameters()


def test_fgsm_within_epsilon_ball():
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.attacks import FGSM
    except Exception as exc:
        pytest.skip(f"FGSM unavailable: {exc}")

    torch.manual_seed(0)
    model = _ToyModel(16, 3)
    x = torch.randn(4, 16)
    y = torch.randint(0, 3, (4,))
    batch = {"p": x, "labels": y}
    eps = 0.05
    atk = FGSM(epsilon=eps)
    adv = atk(model, batch)
    delta = (adv["p"] - x).abs().max().item()
    assert delta <= eps + 1e-6


def test_pgd_within_epsilon_ball():
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.attacks import PGD
    except Exception as exc:
        pytest.skip(f"PGD unavailable: {exc}")

    torch.manual_seed(0)
    model = _ToyModel(16, 3)
    x = torch.randn(4, 16)
    y = torch.randint(0, 3, (4,))
    batch = {"p": x, "labels": y}
    eps = 0.05
    atk = PGD(epsilon=eps, alpha=0.01, steps=10)
    adv = atk(model, batch)
    delta = (adv["p"] - x).abs().max().item()
    assert delta <= eps + 1e-5
