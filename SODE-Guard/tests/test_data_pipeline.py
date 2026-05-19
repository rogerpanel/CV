"""Data pipeline sanity checks (no external downloads)."""
import numpy as np
import pandas as pd

from src.data.feature_engineering import FlowFeatureExtractor, ALL_FEATURES, standardize
from src.data.splits import stratified_split


def test_feature_extractor_fills_missing_columns():
    df = pd.DataFrame({"duration": [1.0, 2.0], "label": [0, 1]})
    X = FlowFeatureExtractor()(df.drop(columns=["label"]))
    assert X.shape == (2, 83)
    assert X.dtype == np.float32


def test_standardize_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=3.0, scale=2.0, size=(1024, 83)).astype(np.float32)
    Xn, mean, std = standardize(X)
    assert abs(Xn.mean()) < 1e-3
    assert abs(Xn.std() - 1.0) < 1e-2


def test_stratified_split_sizes():
    X = np.random.randn(1000, 83).astype(np.float32)
    y = np.random.randint(0, 5, size=1000)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = stratified_split(X, y)
    assert len(ytr) + len(yv) + len(yte) == 1000
    assert abs(len(ytr) / 1000 - 0.70) < 0.02
