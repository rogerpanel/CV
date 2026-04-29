"""Random Forest regressor / classifier (sklearn).

Reproduces the ``random_forest_geochem`` configuration in
``config/hyperparameters.yaml``.  For the Sentinel-2 LULC classifier (deployed
on Google Earth Engine), see :mod:`cmhr.remote_sensing.lulc_classifier`.
"""
from __future__ import annotations

from typing import Literal

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from cmhr.utils import load_config


def build_random_forest(task: Literal["regression", "classification"] = "regression"):
    """Return a fitted-ready estimator with manuscript hyperparameters."""
    cfg = load_config("hyperparameters")["random_forest_geochem"]
    cls = RandomForestRegressor if task == "regression" else RandomForestClassifier
    kwargs = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "max_features": cfg["max_features"],
        "bootstrap": cfg["bootstrap"],
        "random_state": cfg["random_state"],
        "n_jobs": cfg["n_jobs"],
    }
    if task == "regression":
        kwargs["oob_score"] = cfg["oob_score"]
    return cls(**kwargs)
