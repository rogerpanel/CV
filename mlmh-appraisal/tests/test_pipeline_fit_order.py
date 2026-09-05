"""Preprocessing must be fitted on the training fold only."""
import numpy as np
import pytest

from mlmh.models.pipelines import make_pipeline


def test_scaler_statistics_come_from_training_fold_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    X_test = rng.normal(loc=50.0, scale=1.0, size=(50, 5))  # wildly different distribution
    y_train = rng.integers(0, 2, 200)
    pipe = make_pipeline("logreg", seed=0)
    pipe.fit(X_train, y_train)
    scaler = pipe.named_steps["scale"]
    assert np.allclose(scaler.mean_, X_train.mean(axis=0), atol=1e-6)
    # If the test set had leaked into the scaler, the mean would be pulled towards 50.
    assert np.all(np.abs(scaler.mean_) < 1.0)
    pipe.predict_proba(X_test)  # must not refit


def test_imputer_statistics_come_from_training_fold_only():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(100, 3))
    X_train[:, 0] = 1.0
    X_test = np.full((10, 3), np.nan)
    pipe = make_pipeline("rf", seed=0).fit(X_train, rng.integers(0, 2, 100))
    imp = pipe.named_steps["impute"]
    assert imp.statistics_[0] == 1.0
    out = imp.transform(X_test)
    assert np.all(out[:, 0] == 1.0)


def test_smote_is_a_sampler_step_inside_pipeline():
    pipe = make_pipeline("logreg", seed=0, resample="smote")
    names = [n for n, _ in pipe.steps]
    assert names.index("smote") > names.index("scale") and names.index("smote") < names.index("clf")
    from imblearn.pipeline import Pipeline as ImbPipeline

    assert isinstance(pipe, ImbPipeline)


def test_unknown_resample_rejected():
    with pytest.raises(ValueError):
        make_pipeline("logreg", resample="oversample_everything")


def test_no_full_dataset_fit_transform_in_source():
    """Grep the package for the anti-pattern of fitting a transformer outside a Pipeline."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "mlmh"
    offenders = []
    for f in src.rglob("*.py"):
        text = f.read_text()
        if "fit_transform(" in text and "Pipeline" not in text:
            offenders.append(f.name)
    assert not offenders, f"fit_transform used outside a Pipeline in {offenders}"
