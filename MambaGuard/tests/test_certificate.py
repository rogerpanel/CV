"""Tests for the composed certificate arithmetic."""
from __future__ import annotations

import math

import pytest


def test_composed_certificate_known_arithmetic():
    """V*=1, L_f=2, eps=0.05, B=1, T=10000, |A_D|=5 →
    bound = 1 - 2*0.05 - sqrt(ln 5 / 20000)."""
    try:
        from mambaguard.certification import composed_certificate
    except Exception as exc:
        pytest.skip(f"composed_certificate unavailable: {exc}")

    bound = float(
        composed_certificate(
            V_star=1.0,
            L_f=2.0,
            epsilon=0.05,
            B=1.0,
            T=10_000,
            num_actions=5,
        )
    )
    expected = 1.0 - 0.10 - math.sqrt(math.log(5) / 20_000.0)
    assert bound == pytest.approx(expected, abs=1e-6)


def test_certificate_monotone_in_epsilon():
    try:
        from mambaguard.certification import composed_certificate
    except Exception as exc:
        pytest.skip(f"composed_certificate unavailable: {exc}")
    args = dict(V_star=0.9, L_f=1.5, B=1.0, T=5000, num_actions=4)
    b1 = float(composed_certificate(epsilon=0.01, **args))
    b2 = float(composed_certificate(epsilon=0.05, **args))
    b3 = float(composed_certificate(epsilon=0.10, **args))
    assert b1 > b2 > b3
