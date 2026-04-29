"""Hyperparameter tuning helpers."""
from __future__ import annotations

from typing import Any, Mapping

from sklearn.model_selection import GridSearchCV


def grid_search(estimator, param_grid: Mapping[str, Any], X, y, cv: int = 5,
                scoring: str = "r2", n_jobs: int = -1):
    """Exhaustive grid search wrapper."""
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
        return_train_score=True,
    )
    gs.fit(X, y)
    return gs


def bayes_search(estimator, search_spaces, X, y, n_iter: int = 32, cv: int = 5,
                 scoring: str = "r2", n_jobs: int = -1, random_state: int = 42):
    """Bayesian hyperparameter optimisation (skopt)."""
    try:
        from skopt import BayesSearchCV
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("scikit-optimize is required: pip install scikit-optimize") from err
    bs = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs,
        random_state=random_state, verbose=0,
    )
    bs.fit(X, y)
    return bs
