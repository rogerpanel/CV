"""Tests for the Hedge / multiplicative-weights defender."""
from __future__ import annotations

import math

import pytest


def test_hedge_regret_within_theory():
    np = pytest.importorskip("numpy")
    try:
        from mambaguard.certification import HedgeDefender
    except Exception as exc:
        pytest.skip(f"HedgeDefender unavailable: {exc}")

    rng = np.random.default_rng(0)
    actions = [f"a{i}" for i in range(5)]
    T = 2000
    losses = rng.uniform(0.0, 1.0, size=(T, len(actions)))
    hedge = HedgeDefender(actions=actions, horizon=T, B=1.0)
    cum_loss = 0.0
    for t in range(T):
        a = hedge.sample(rng=np.random.default_rng(t))
        cum_loss += float(losses[t, a])
        hedge.update(losses[t])
    best_in_hindsight = losses.sum(axis=0).min()
    regret = cum_loss - best_in_hindsight
    # Theoretical Hedge regret bound: B·sqrt((T/2)·ln K).
    bound = math.sqrt(T * math.log(len(actions)) / 2.0)
    # 3x margin for stochastic slack.
    assert regret <= 3.0 * bound, f"regret {regret} exceeds 3x bound {bound}"


def test_hedge_distribution_is_simplex():
    np = pytest.importorskip("numpy")
    try:
        from mambaguard.certification import HedgeDefender
    except Exception as exc:
        pytest.skip(f"HedgeDefender unavailable: {exc}")
    h = HedgeDefender(actions=["a", "b", "c", "d"], horizon=10, B=1.0)
    p = np.asarray(h.distribution())
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
    assert (p >= 0.0).all()
