"""Gaussian Process Regression wrapper.

Identical kernel structure to the manuscript: ``C * Matern(ν=1.5) +
WhiteKernel`` with ten hyperparameter restarts.
"""
from __future__ import annotations

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmhr.utils import load_config


def build_gaussian_process():
    cfg = load_config("hyperparameters")["gaussian_process"]
    kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel()
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=cfg["n_restarts_optimizer"],
        alpha=cfg["alpha"],
        normalize_y=cfg["normalize_y"],
        random_state=cfg["random_state"],
    )
    return make_pipeline(StandardScaler(), gp)
