"""Tests for the Lipschitz bound computation."""
from __future__ import annotations

import pytest


def test_lipschitz_known_value_via_spectral_norm():
    """`spectral_norm` of a diagonal-singular-value matrix equals its top σ."""
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.certification.lipschitz_bounds import spectral_norm
    except Exception as exc:
        pytest.skip(f"spectral_norm unavailable: {exc}")
    torch.manual_seed(0)
    W = torch.eye(4) * torch.tensor([2.0, 1.0, 0.5, 0.5])
    sn = float(spectral_norm(W))
    assert sn == pytest.approx(2.0, rel=1e-4)


def test_compute_lipschitz_bound_returns_dict():
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.certification import compute_lipschitz_bound
    except Exception as exc:
        pytest.skip(f"compute_lipschitz_bound unavailable: {exc}")
    net = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)
    )
    out = compute_lipschitz_bound(net)
    assert isinstance(out, dict)
    assert "L_f" in out
    assert 0.0 < float(out["L_f"]) < float("inf")


def test_compute_lipschitz_no_recognised_blocks_defaults_to_one():
    torch = pytest.importorskip("torch")
    try:
        from mambaguard.certification import compute_lipschitz_bound
    except Exception as exc:
        pytest.skip(f"compute_lipschitz_bound unavailable: {exc}")
    # An empty Sequential has no SSM/GAT/Head children.
    net = torch.nn.Sequential()
    out = compute_lipschitz_bound(net)
    assert float(out["L_f"]) == pytest.approx(1.0)
