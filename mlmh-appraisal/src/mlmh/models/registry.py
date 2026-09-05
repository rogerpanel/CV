"""Model registry.

The majority-class baseline and logistic regression exist so that the flexible
models have to earn their complexity.  All hyper-parameters are fixed a priori
(no per-fold tuning) so that the E1 comparison isolates the splitting design;
the values are the library defaults or conservative choices documented here.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def make_model(name: str, seed: int = 0, n_classes: int = 2):
    if name == "majority":
        return DummyClassifier(strategy="prior")
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "xgboost":
        from xgboost import XGBClassifier

        kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=0)
        if n_classes > 2:
            kw.update(objective="multi:softprob", num_class=n_classes)
        else:
            kw.update(objective="binary:logistic")
        return XGBClassifier(**kw)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=2000, early_stopping=False, random_state=seed)
    if name == "cnn1d":
        from .cnn import Conv1DClassifier

        return Conv1DClassifier(seed=seed, n_classes=n_classes)
    raise KeyError(f"unknown model {name!r}")


MODEL_INPUT = {
    "majority": "engineered",
    "logreg": "engineered",
    "rf": "engineered",
    "xgboost": "engineered",
    "mlp": "engineered",
    "cnn1d": "raw",
}

MODEL_LABELS = {
    "majority": "Majority class",
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "cnn1d": "1D-CNN",
}
