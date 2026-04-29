"""Geochemical pollution indices.

Implements the manuscript suite (Igeo, CF, PLI, RI) plus well-cited extensions
(Enrichment Factor, Nemerow PI, Modified Contamination Degree) so the same
codebase can be applied to data from any artisanal-mining landscape.
"""
from .geoaccumulation import geoaccumulation_index, classify_igeo
from .contamination_factor import contamination_factor, classify_cf
from .pollution_load_index import pollution_load_index
from .ecological_risk import ecological_risk_factor, ecological_risk_index, classify_ri
from .enrichment_factor import enrichment_factor, classify_ef
from .nemerow_index import nemerow_pollution_index, classify_nemerow
from .modified_indices import (
    modified_contamination_degree,
    pollution_index_single,
)

__all__ = [
    "geoaccumulation_index",
    "classify_igeo",
    "contamination_factor",
    "classify_cf",
    "pollution_load_index",
    "ecological_risk_factor",
    "ecological_risk_index",
    "classify_ri",
    "enrichment_factor",
    "classify_ef",
    "nemerow_pollution_index",
    "classify_nemerow",
    "modified_contamination_degree",
    "pollution_index_single",
]
