"""Graph-builder + neighbor-sampling tests."""
from __future__ import annotations

import numpy as np
import torch

from fedloraguard.graph.builder import build_graph_from_records
from fedloraguard.graph.schema import AdapterRecord
from fedloraguard.federated.sampling import build_query_batch


def _toy_records(n: int = 6) -> list:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        out.append(AdapterRecord(
            adapter_id=f"a_{i}",
            base_model="llama2-7b" if i % 2 == 0 else "mistral-7b",
            contributor=f"c_{i % 3}",
            application="alpaca",
            rank=4,
            upload_ts=float(i),
            label=int(i % 2),
            weight_features=rng.normal(size=8).astype(np.float32),
            text_features=rng.normal(size=8).astype(np.float32),
            behavioral_features=rng.normal(size=4).astype(np.float32),
        ))
    return out


def test_graph_partitioning_preserves_total_adapters():
    g = build_graph_from_records(
        _toy_records(),
        feature_dims={"weight": 8, "text": 8, "behavioral": 4, "fused": 20},
    )
    parts = g.split_by_marketplace([0, 0, 1, 1, 2, 2])
    total = sum(p.num_nodes("adapter") for p in parts.values())
    assert total == 6


def test_query_batch_has_correct_shapes():
    g = build_graph_from_records(
        _toy_records(),
        feature_dims={"weight": 8, "text": 8, "behavioral": 4, "fused": 20},
    )
    batch = build_query_batch(g, batch_size=3)
    assert len(batch) == 3
    for b in batch:
        assert b["query_feat"].shape[0] == 20
        assert isinstance(b["neighbor_feats"], torch.Tensor)
