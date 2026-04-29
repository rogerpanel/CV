"""Stacking ensemble (RF + XGB + LGBM + SVR → Ridge meta-learner)."""
from __future__ import annotations

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from cmhr.utils import load_config
from .gaussian_process import build_gaussian_process
from .lightgbm_model import build_lightgbm
from .random_forest import build_random_forest
from .svr_model import build_svr
from .xgboost_model import build_xgboost


def _build_base(name: str):
    return {
        "random_forest_geochem": lambda: build_random_forest("regression"),
        "xgboost":              lambda: build_xgboost("regression"),
        "lightgbm":             lambda: build_lightgbm("regression"),
        "svr":                  build_svr,
        "gaussian_process":     build_gaussian_process,
    }[name]()


def build_stacking_ensemble():
    cfg = load_config("hyperparameters")["stacking"]
    estimators = [(name, _build_base(name)) for name in cfg["base_models"]]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=cfg["meta_learner_alpha"]),
        cv=cfg["cv_folds"],
        passthrough=cfg.get("passthrough", False),
        n_jobs=-1,
    )
