"""Machine-learning regressors / classifiers used to model heavy-metal
concentrations and predict pollution / risk indices from environmental
covariates."""
from .random_forest import build_random_forest
from .xgboost_model import build_xgboost
from .lightgbm_model import build_lightgbm
from .ann_regressor import build_ann
from .svr_model import build_svr
from .gaussian_process import build_gaussian_process
from .ensemble import build_stacking_ensemble
from .hyperparameter_tuning import grid_search, bayes_search
from .interpretability import shap_summary, permutation_importance_df

__all__ = [
    "build_random_forest",
    "build_xgboost",
    "build_lightgbm",
    "build_ann",
    "build_svr",
    "build_gaussian_process",
    "build_stacking_ensemble",
    "grid_search",
    "bayes_search",
    "shap_summary",
    "permutation_importance_df",
]
