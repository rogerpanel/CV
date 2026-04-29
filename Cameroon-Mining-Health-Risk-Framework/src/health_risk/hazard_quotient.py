"""Non-carcinogenic risk: Hazard Quotient and Hazard Index."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from cmhr.utils import load_config


def _rfd_lookup(element: str, pathway: str) -> float:
    """Return the reference dose (mg/kg/day) for an element/pathway."""
    cfg = load_config("reference_doses")
    table = {
        "ingestion": cfg["reference_dose_oral"],
        "oral":      cfg["reference_dose_oral"],
        "dermal":    cfg["reference_dose_dermal"],
        "inhalation": cfg["reference_dose_inhalation"],
    }[pathway]
    if element not in table:
        raise KeyError(f"No RfD configured for {element!r} via {pathway} pathway")
    return float(table[element])


def hazard_quotient(add, element: str, pathway: str = "ingestion"):
    """Compute :math:`HQ = ADD / RfD`."""
    rfd = _rfd_lookup(element, pathway)
    return np.asarray(add, dtype=float) / rfd


def hazard_index(hqs: Iterable[float] | Mapping[str, float]) -> float:
    """Sum of HQs across elements / pathways.

    Both an iterable of scalars/arrays and a mapping ``{element: HQ}`` are
    accepted; arrays are broadcast and summed elementwise.
    """
    if isinstance(hqs, Mapping):
        arr = np.array(list(hqs.values()), dtype=object)
    else:
        arr = np.array(list(hqs), dtype=object)
    if arr.dtype == object and any(isinstance(x, (np.ndarray, pd.Series)) for x in arr):
        stacked = np.vstack([np.asarray(x, dtype=float) for x in arr])
        return stacked.sum(axis=0)
    return float(np.asarray(arr, dtype=float).sum())
