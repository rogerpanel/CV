"""Sanity tests for spatial autocorrelation."""
import numpy as np

from cmhr.geospatial import global_morans_i, local_morans_i


def test_morans_i_clustered():
    """Strongly clustered values should produce a positive Moran's I."""
    rng = np.random.default_rng(1)
    coords = [(i * 0.01, j * 0.01) for i in range(10) for j in range(10)]
    grid = np.zeros((10, 10))
    grid[:5, :5] = 5
    grid[5:, 5:] = 5
    values = grid.ravel() + rng.normal(0, 0.1, 100)
    res = global_morans_i(values, coords, threshold_km=2.0, permutations=199)
    assert res.statistic > 0


def test_lisa_returns_classes():
    rng = np.random.default_rng(2)
    coords = [(i * 0.01, j * 0.01) for i in range(6) for j in range(6)]
    values = rng.normal(0, 1, 36)
    df = local_morans_i(values, coords, threshold_km=5.0, permutations=199)
    assert {"value", "z_score", "spatial_lag", "local_I", "p_value", "cluster"} <= set(df.columns)
    assert set(df["cluster"].unique()) <= {
        "High-High", "Low-Low", "High-Low", "Low-High", "Not significant",
    }
