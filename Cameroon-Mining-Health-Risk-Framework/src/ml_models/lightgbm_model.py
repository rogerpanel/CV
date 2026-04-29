"""LightGBM wrapper."""
from __future__ import annotations

from typing import Literal

from cmhr.utils import load_config


def build_lightgbm(task: Literal["regression", "classification"] = "regression"):
    try:
        import lightgbm as lgb
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("lightgbm is required: pip install lightgbm") from err

    cfg = load_config("hyperparameters")["lightgbm"]
    kwargs = dict(
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        max_depth=cfg["max_depth"],
        min_child_samples=cfg["min_child_samples"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        random_state=cfg["random_state"],
    )
    return lgb.LGBMRegressor(**kwargs) if task == "regression" else lgb.LGBMClassifier(**kwargs)
