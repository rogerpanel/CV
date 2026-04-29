"""DHS-based spatial epidemiology helpers."""
from .proximity_analysis import classify_proximity, distance_to_mining_km
from .dhs_processing import load_dhs_clusters, prepare_outcome_table
from .logistic_regression import adjusted_logit, prevalence_table

__all__ = [
    "classify_proximity",
    "distance_to_mining_km",
    "load_dhs_clusters",
    "prepare_outcome_table",
    "adjusted_logit",
    "prevalence_table",
]
