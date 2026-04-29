"""Multi-layer perceptron regressor.

A Keras / TensorFlow implementation that mirrors the ``ann`` block of
``config/hyperparameters.yaml``.  When TensorFlow is unavailable the function
falls back to ``sklearn.neural_network.MLPRegressor`` with comparable
hyperparameters so the rest of the framework still runs.
"""
from __future__ import annotations

from cmhr.utils import load_config


def build_ann(input_dim: int):
    """Construct the ANN regressor.

    Parameters
    ----------
    input_dim : int
        Number of input features (e.g. spectral bands + indices + soil props).
    """
    cfg = load_config("hyperparameters")["ann"]
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers, optimizers
    except ImportError:                                              # pragma: no cover
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=tuple(cfg["hidden_layers"]),
            activation=cfg["activation"],
            solver="adam",
            alpha=cfg["l2_regularisation"],
            learning_rate_init=cfg["learning_rate"],
            max_iter=cfg["epochs"],
            random_state=cfg["random_state"],
        )

    tf.random.set_seed(cfg["random_state"])
    reg = regularizers.l2(cfg["l2_regularisation"])

    model = models.Sequential(name="cmhr_ann")
    model.add(layers.Input(shape=(input_dim,)))
    for units in cfg["hidden_layers"]:
        model.add(layers.Dense(units, activation=cfg["activation"], kernel_regularizer=reg))
        model.add(layers.Dropout(cfg["dropout_rate"]))
    model.add(layers.Dense(1, activation=cfg["output_activation"]))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss=cfg["loss"],
        metrics=["mae", "RootMeanSquaredError"],
    )
    return model
