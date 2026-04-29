"""Sanity tests for the Monte Carlo health-risk engine."""
import numpy as np
import pytest

from cmhr.health_risk import (
    MonteCarloRiskAssessment,
    add_ingestion, hazard_quotient, cancer_risk,
    classify_cancer_risk,
)


def test_add_ingestion_unit():
    add = add_ingestion(C=1.0, IR=1.0, EF=365, ED=24, BW=70, AT=24 * 365)
    # ADD = (1 * 1 * 365 * 24) / (70 * 8760) ≈ 0.01428 mg/kg/day
    assert add == pytest.approx(1 / 70, rel=1e-3)


def test_hazard_quotient_lookup():
    hq = hazard_quotient(1.0, "As")
    # RfD(As) ingestion = 3e-4 → HQ = 1 / 3e-4 ≈ 3333.3
    assert hq == pytest.approx(1 / 3e-4)


def test_cancer_risk_classification():
    cr = cancer_risk(1e-5, "As")  # CSF = 1.5
    assert classify_cancer_risk(float(cr)) in {"Tolerable", "Unacceptable"}


def test_monte_carlo_repeatability():
    mc1 = MonteCarloRiskAssessment(receptor="children", n_iterations=2000, seed=42)
    mc2 = MonteCarloRiskAssessment(receptor="children", n_iterations=2000, seed=42)
    r1 = mc1.run_ingestion(0.01, "Cd", "water").samples
    r2 = mc2.run_ingestion(0.01, "Cd", "water").samples
    assert np.allclose(r1, r2)


def test_monte_carlo_percentiles_increase():
    mc = MonteCarloRiskAssessment(receptor="adults", n_iterations=5000, seed=1)
    res = mc.run_ingestion(0.05, "As", "water")
    p = res.percentiles()
    assert p[5] < p[50] < p[95]
