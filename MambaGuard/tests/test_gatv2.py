"""Tests for the temporal GATv2 layer."""
from __future__ import annotations

import pytest


@pytest.fixture
def gat_layer():
    pytest.importorskip("torch")
    try:
        from mambaguard.models.gatv2_temporal import TemporalGATv2Layer
    except Exception as exc:
        pytest.skip(f"TemporalGATv2Layer unavailable: {exc}")
    return TemporalGATv2Layer(d_in=16, d_out=16, edge_dim=4, heads=2)


def test_output_shape(gat_layer):
    import torch

    x = torch.randn(5, 16)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    ea = torch.randn(4, 4)
    et = torch.tensor([0.0, 1.0, 2.0, 3.0])
    out = gat_layer(x, ei, edge_attr=ea, edge_time=et)
    assert out.shape[0] == 5
    # concat=False averages heads → d_out
    assert out.shape[-1] == 16


def test_forward_is_finite(gat_layer):
    import torch

    x = torch.randn(5, 16)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    ea = torch.randn(4, 4)
    et = torch.zeros(4)
    out = gat_layer(x, ei, edge_attr=ea, edge_time=et)
    assert torch.isfinite(out).all()


def test_lipschitz_bound_finite(gat_layer):
    L = float(gat_layer.lipschitz_bound())
    assert 0.0 < L < float("inf")
