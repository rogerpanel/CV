"""Flower-runtime adapter tests.  Skipped if `flwr` is not installed."""
from __future__ import annotations

import pytest


def test_flower_strategy_is_constructible():
    pytest.importorskip("flwr")
    from fedloraguard.federated.runtime_flower import make_strategy
    from fedloraguard.utils import load_config
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "smoke.yaml")
    strategy = make_strategy(cfg)
    assert strategy is not None


def test_friendly_error_when_flwr_missing():
    import sys
    if "flwr" in sys.modules:
        pytest.skip("Flower is installed; the friendly-error test only matters in its absence.")
    from fedloraguard.federated import runtime_flower

    with pytest.raises(RuntimeError, match="flwr"):
        runtime_flower.make_strategy({"federated": {"num_clients": 1, "rounds": 1,
                                                     "sampling_rate": 1.0,
                                                     "clients_per_round": 1},
                                       "privacy": {"noise_multiplier": 1.0,
                                                   "target_delta": 1e-5}})
