"""Water quality indices.

Three drinking-water indices commonly applied in mining-affected catchments:
WQI (NSF / Tiwari & Mishra), HPI (Mohan et al. 1996) and the contamination
degree of Edet & Offiong (2002).
"""
from .wqi import water_quality_index
from .hpi import heavy_metal_pollution_index
from .contamination_degree import contamination_degree

__all__ = [
    "water_quality_index",
    "heavy_metal_pollution_index",
    "contamination_degree",
]
