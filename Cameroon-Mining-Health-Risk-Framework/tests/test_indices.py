"""Sanity tests for the pollution-index implementations."""
import math

import pandas as pd
import pytest

from cmhr.pollution_indices import (
    geoaccumulation_index, classify_igeo,
    contamination_factor, classify_cf,
    pollution_load_index,
    ecological_risk_factor,
    enrichment_factor, classify_ef,
    nemerow_pollution_index,
)


def test_igeo_unpolluted():
    # Cn = Bn → ratio = 1/1.5 → log2 ≈ -0.585 → unpolluted
    assert geoaccumulation_index(5.0, 5.0) == pytest.approx(math.log2(1 / 1.5))


def test_igeo_class_extreme():
    classes = classify_igeo([-1, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    assert classes[0] == "Unpolluted"
    assert classes[-1] == "Extremely polluted"


def test_cf_and_pli():
    df = pd.DataFrame({"As_mg_kg": [10, 20], "Pb_mg_kg": [34, 51]})
    bg = {"As": 5.0, "Pb": 17.0}
    cf_as = contamination_factor(df["As_mg_kg"], bg["As"])
    assert cf_as.tolist() == [2.0, 4.0]
    pli = pollution_load_index(df, bg)
    assert pli.iloc[0] == pytest.approx((2.0 * 2.0) ** 0.5)


def test_ecological_risk_factor():
    er = ecological_risk_factor(measured=10.0, background=5.0, element="As")
    # Tr(As) = 10, CF = 2 → Er = 20
    assert er == pytest.approx(20.0)


def test_enrichment_factor_classes():
    ef = enrichment_factor(sample_element=10, sample_normaliser=80000,
                           background_element=5, background_normaliser=80000)
    assert ef == pytest.approx(2.0)
    assert classify_ef([ef])[0] == "Minor enrichment"


def test_cf_classes_boundary():
    classes = classify_cf([0.5, 1.0, 3.0, 6.0, 10.0])
    assert classes == ["Low", "Moderate", "Considerable", "Very high", "Very high"]


def test_nemerow_index():
    df = pd.DataFrame({"As": [0.02, 0.04], "Pb": [0.01, 0.02]})
    standards = {"As": 0.01, "Pb": 0.01}
    pi_n = nemerow_pollution_index(df, standards)
    # PI_As=2, PI_Pb=1, mean=1.5, max=2 → sqrt((1.5²+2²)/2) ≈ 1.768
    assert pi_n.iloc[0] == pytest.approx(((1.5 ** 2 + 2 ** 2) / 2) ** 0.5)
