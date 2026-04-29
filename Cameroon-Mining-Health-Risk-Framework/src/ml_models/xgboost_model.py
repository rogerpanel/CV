"""XGBoost regressor / classifier wrapper."""
from __future__ import annotations

from typing import Literal

from cmhr.utils import load_config


def build_xgboost(task: Literal["regression", "classification"] = "regression"):
    """Return an XGBoost estimator configured with manuscript hyperparameters."""
    try:
        import xgboost as xgb
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("xgboost is required: pip install xgboost") from err

    cfg = load_config("hyperparameters")["xgboost"]
    common = dict(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        gamma=cfg["gamma"],
        early_stopping_rounds=cfg.get("early_stopping_rounds"),
        tree_method=cfg.get("tree_method", "hist"),
        random_state=cfg["random_state"],
    )
    if task == "regression":
        common["objective"] = cfg.get("objective", "reg:squarederror")
        return xgb.XGBRegressor(**common)
    common["objective"] = "multi:softprob"
    return xgb.XGBClassifier(**common)
