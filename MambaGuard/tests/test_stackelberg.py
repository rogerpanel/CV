"""Tests for the Stackelberg solver."""
from __future__ import annotations

import pytest


def _cvxpy_or_skip():
    try:
        import cvxpy  # noqa: F401
    except Exception:
        pytest.skip("cvxpy not installed")


def test_stackelberg_2x2_known_sse():
    """2x2 zero-sum (attacker payoff = -U): defender's SSE value matches
    the maximin of U."""
    np = pytest.importorskip("numpy")
    _cvxpy_or_skip()
    try:
        from mambaguard.certification import StackelbergSolver
    except Exception as exc:
        pytest.skip(f"StackelbergSolver unavailable: {exc}")

    # Defender row 0 dominates row 1 column-wise — attacker best-responds to
    # the lowest column for row 0 (col 1, value 0.6).
    U = np.array([[0.9, 0.6], [0.4, 0.3]], dtype=np.float64)
    sol = StackelbergSolver(
        defender_actions=["d0", "d1"],
        attacker_actions=["a0", "a1"],
        utility_matrix=U,
    ).solve()
    pi = np.asarray(sol.pi_D)
    val = float(sol.value)
    assert pi[0] == pytest.approx(1.0, abs=1e-3)
    assert pi[1] == pytest.approx(0.0, abs=1e-3)
    assert val == pytest.approx(0.6, abs=1e-3)


def test_stackelberg_simplex_constraint():
    np = pytest.importorskip("numpy")
    _cvxpy_or_skip()
    try:
        from mambaguard.certification import StackelbergSolver
    except Exception as exc:
        pytest.skip(f"StackelbergSolver unavailable: {exc}")
    U = np.array([[0.7, 0.4, 0.6], [0.5, 0.8, 0.3], [0.6, 0.5, 0.7]])
    sol = StackelbergSolver(
        defender_actions=["d0", "d1", "d2"],
        attacker_actions=["a0", "a1", "a2"],
        utility_matrix=U,
    ).solve()
    pi = np.asarray(sol.pi_D)
    assert pi.sum() == pytest.approx(1.0, abs=1e-4)
    assert (pi >= -1e-6).all()
