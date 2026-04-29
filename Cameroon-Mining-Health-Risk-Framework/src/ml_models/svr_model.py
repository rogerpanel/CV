"""Support Vector Regression wrapper (sklearn)."""
from __future__ import annotations

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from cmhr.utils import load_config


def build_svr():
    cfg = load_config("hyperparameters")["svr"]
    return make_pipeline(
        StandardScaler(),
        SVR(kernel=cfg["kernel"], C=cfg["C"], epsilon=cfg["epsilon"], gamma=cfg["gamma"]),
    )
