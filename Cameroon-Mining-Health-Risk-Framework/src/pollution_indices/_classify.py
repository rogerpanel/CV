"""Internal helper for threshold-based classification."""
from __future__ import annotations

from typing import Iterable, List, Mapping

import numpy as np
import pandas as pd


def classify(values, ranges: Iterable[Mapping]) -> List[str]:
    """Return the textual class label for each value using ``[min, max)`` rules.

    ``ranges`` is the list-of-mappings produced by the YAML configs in
    ``config/thresholds.yaml``.
    """
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    labels = np.empty(arr.shape, dtype=object)
    labels[:] = "unclassified"
    for rule in ranges:
        lo = float(rule["min"])
        hi = float(rule["max"])
        label = rule.get("label", str(rule.get("class", "")))
        mask = (arr >= lo) & (arr < hi)
        labels[mask] = label
    if isinstance(values, pd.Series):
        return pd.Series(labels, index=values.index, name=f"{values.name}_class")
    return labels.tolist() if labels.size > 1 else labels.item()
