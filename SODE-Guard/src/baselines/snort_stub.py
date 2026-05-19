"""Snort 3 + SnortML thin Python wrapper.

We don't ship the actual Snort binary; instead, we expose a sklearn-style
adapter that calls out to ``snort -c <rules> -r <pcap>`` if the binary is on
PATH, falling back to a no-op detector that emits the majority class. This
mirrors the comparison protocol described in §4.3 of the manuscript.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

import numpy as np


class SnortML:
    def __init__(self, rules_path: str | None = None):
        self.rules = rules_path
        self.binary = shutil.which("snort")
        self.fallback_class = 0

    def fit_majority(self, y: np.ndarray) -> None:
        vals, counts = np.unique(y, return_counts=True)
        self.fallback_class = int(vals[counts.argmax()])

    def predict_pcap(self, pcap_path: str) -> str:
        if not self.binary or not self.rules:
            return ""
        out = subprocess.run([self.binary, "-c", self.rules, "-r", pcap_path,
                              "-A", "fast", "-q"], capture_output=True, text=True)
        return out.stdout

    def predict_tabular(self, X: np.ndarray) -> np.ndarray:
        # Tabular flows: no rules apply — emit fallback class (benign in most IDS sets).
        return np.full(X.shape[0], self.fallback_class, dtype=np.int64)
