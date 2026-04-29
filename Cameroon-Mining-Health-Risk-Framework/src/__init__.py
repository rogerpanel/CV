"""cmhr — Cameroon Mining Health Risk reproducibility framework.

Top-level convenience re-exports so user code can simply::

    from cmhr.pollution_indices import geoaccumulation_index
    from cmhr.health_risk import MonteCarloRiskAssessment
"""
from importlib import import_module as _imp

__version__ = "1.0.0"

# Lazy submodule registration — keeps optional GEE/torch deps optional.
__all__ = [
    "pollution_indices",
    "water_quality",
    "health_risk",
    "remote_sensing",
    "ml_models",
    "geospatial",
    "species_distribution",
    "epidemiology",
    "visualization",
    "utils",
]

for _sub in __all__:
    try:
        globals()[_sub] = _imp(f"cmhr.{_sub}")
    except Exception:               # pragma: no cover - optional deps may be missing
        pass
