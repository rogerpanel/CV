import numpy as np
import pandas as pd
import pytest

from mlmh.evaluation.bootstrap import subject_bootstrap_ci
from mlmh.evaluation.metrics import (
    binary_metrics,
    calibration_intercept,
    calibration_slope,
    expected_calibration_error,
    subject_level,
)


def _calibrated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_calibration_slope_intercept_near_identity_for_calibrated_probs():
    y, p = _calibrated()
    assert abs(calibration_slope(y, p) - 1.0) < 0.1
    assert abs(calibration_intercept(y, p)) < 0.1


def test_overconfident_probs_have_slope_below_one():
    y, p = _calibrated()
    logit = np.log(p / (1 - p)) * 3.0  # stretch -> overconfident
    q = 1 / (1 + np.exp(-logit))
    assert calibration_slope(y, q) < 0.6


def test_shifted_probs_have_nonzero_intercept():
    y, p = _calibrated()
    logit = np.log(p / (1 - p)) + 1.0
    q = 1 / (1 + np.exp(-logit))
    assert calibration_intercept(y, q) < -0.5


def test_ece_zero_for_perfect_and_large_for_bad():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == 1.0


def test_binary_metrics_keys_and_ranges():
    y, p = _calibrated(500)
    m = binary_metrics(y, p)
    for k in ("auroc", "accuracy", "macro_f1", "brier", "calibration_slope", "calibration_intercept", "ece"):
        assert k in m
    assert 0 <= m["auroc"] <= 1 and 0 <= m["brier"] <= 1


def test_subject_level_aggregation_means_probabilities():
    pred = pd.DataFrame({"subject_id": ["a", "a", "b"], "y": [1, 1, 0], "p": [0.2, 0.8, 0.3]})
    s = subject_level(pred).set_index("subject_id")
    assert s.loc["a", "p"] == pytest.approx(0.5)
    assert s.loc["b", "n_windows"] == 1


def test_subject_bootstrap_resamples_subjects_not_windows():
    """A metric that counts distinct subjects must vary under the bootstrap; window count is constant per subject draw."""
    rng = np.random.default_rng(0)
    pred = pd.DataFrame({"subject_id": np.repeat([f"s{i}" for i in range(20)], 5), "y": np.repeat(rng.integers(0, 2, 20), 5), "p": rng.uniform(size=100)})
    est, lo, hi = subject_bootstrap_ci(pred, lambda f: f["subject_id"].nunique(), n_boot=200, seed=0, method="percentile")
    assert est == 20 and lo < 20  # resampling with replacement drops some subjects
    est, lo, hi = subject_bootstrap_ci(pred, lambda f: binary_metrics(f["y"], f["p"])["auroc"], n_boot=200, seed=0)
    assert lo <= est <= hi
