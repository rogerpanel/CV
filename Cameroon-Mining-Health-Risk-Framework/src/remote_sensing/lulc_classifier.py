"""Random-Forest land-cover classifier (7 classes).

Reproduces the GEE configuration from Anaedevha et al. (2026):
500 trees, ``√n`` variables per split, minimum-leaf-population = 5,
bag fraction = 0.7.
"""
from __future__ import annotations

from cmhr.utils import load_config


def build_gee_random_forest():
    """Construct an ``ee.Classifier.smileRandomForest`` with the manuscript hyperparameters."""
    try:
        import ee
    except ImportError as err:                                       # pragma: no cover
        raise ImportError(
            "earthengine-api is required: pip install earthengine-api"
        ) from err
    cfg = load_config("hyperparameters")["random_forest_lulc"]
    return ee.Classifier.smileRandomForest(
        numberOfTrees=cfg["n_trees"],
        variablesPerSplit=None,                            # GEE → defaults to √n
        minLeafPopulation=cfg["min_leaf_population"],
        bagFraction=cfg["bag_fraction"],
        outOfBagMode=cfg["out_of_bag_mode"],
        seed=cfg["seed"],
    )


def train_lulc_classifier(composite, training_collection, label: str = "class"):
    """Train the GEE RF on a labelled feature collection.

    ``composite`` should already contain the 10 Sentinel-2 bands plus the seven
    spectral indices (see ``cmhr.remote_sensing.spectral_indices.all_indices``).
    """
    try:
        import ee
    except ImportError as err:                                       # pragma: no cover
        raise ImportError("earthengine-api is required") from err

    cfg = load_config("hyperparameters")["random_forest_lulc"]
    bands = cfg["feature_set"]["sentinel2_bands"] + cfg["feature_set"]["spectral_indices"]

    training = composite.select(bands).sampleRegions(
        collection=training_collection,
        properties=[label],
        scale=10,
        tileScale=4,
    )
    classifier = build_gee_random_forest().train(
        features=training,
        classProperty=label,
        inputProperties=bands,
    )
    return classifier
