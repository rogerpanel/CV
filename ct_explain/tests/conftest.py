"""Shared pytest fixtures.

Tests are intentionally CPU-only and deterministic. The synthetic-data
path of every loader / model is exercised so the suite runs offline.
"""
from __future__ import annotations

import pytest
import torch

from ct_explain.data.graph_builder import ContinuousTimeGraphBuilder
from ct_explain.models.ct_tgnn import CTTemporalGNN
from ct_explain.models.sde_tgnn import SDETemporalGNN
from ct_explain.utils.seed import set_global_seed


@pytest.fixture(autouse=True)
def _determinism():
    set_global_seed(0)
    yield


@pytest.fixture
def small_graph():
    b = ContinuousTimeGraphBuilder(default_node_feature_dim=8)
    for i in range(6):
        b.add_event(
            src_id=f"h{i % 3}",
            tgt_id=f"s{i % 2}",
            timestamp=float(i) / 10.0,
            edge_feature=torch.randn(8).tolist(),
            label=int(i % 2),
        )
    return b.build()


@pytest.fixture
def ct_tgnn_model():
    return CTTemporalGNN(
        node_feat_dim=8, edge_feat_dim=8, hidden_dim=16, num_classes=2,
        num_heads=2, time_dim=8, ode_solver="euler", adjoint=False,
        integration_window=0.5,
    )


@pytest.fixture
def sde_tgnn_model():
    return SDETemporalGNN(
        node_feat_dim=8, edge_feat_dim=8, hidden_dim=16, num_classes=2,
        num_heads=2, time_dim=8, ode_solver="euler", adjoint=False,
        integration_window=0.5, diffusion_scale=0.05,
    )
