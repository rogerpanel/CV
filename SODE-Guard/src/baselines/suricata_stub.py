"""Suricata 7 wrapper (same pattern as the Snort stub)."""
from __future__ import annotations
import shutil
import subprocess
import numpy as np


class Suricata7:
    def __init__(self, rules_path: str | None = None,
                 config_path: str | None = None):
        self.rules = rules_path
        self.config = config_path
        self.binary = shutil.which("suricata")
        self.fallback_class = 0

    def fit_majority(self, y: np.ndarray) -> None:
        self.fallback_class = int(np.bincount(y).argmax())

    def predict_pcap(self, pcap_path: str) -> str:
        if not self.binary:
            return ""
        cmd = [self.binary, "-r", pcap_path, "-l", "/tmp/suricata-out", "-k", "none"]
        if self.config: cmd += ["-c", self.config]
        if self.rules:  cmd += ["-S", self.rules]
        out = subprocess.run(cmd, capture_output=True, text=True)
        return out.stdout

    def predict_tabular(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.fallback_class, dtype=np.int64)
