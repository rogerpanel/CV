"""Configuration loading and validation for UAV-EW-Bench-2026."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "config"))


@dataclass
class Defense:
    id: str
    label: str
    completion_anchors: List[float]
    description: str = ""
    reference: str = ""


@dataclass
class BenchmarkConfig:
    name: str
    version: str
    total_flights: int
    missions: List[str]
    receivers: List[str]
    js_min_db: float
    js_max_db: float
    js_levels: int
    js_report_points_db: List[float]
    repeats_per_point: int
    seeds: List[int]
    confidence: float
    labelling_standard: str
    regulatory_threshold: float
    js_anchor_points_db: List[float]
    defenses: List[Defense]
    stratified_enabled: bool = False
    receiver_logit_delta: dict = field(default_factory=dict)
    mission_logit_delta: dict = field(default_factory=dict)

    def js_grid(self) -> List[float]:
        """Full J/S sweep grid (js_levels points, inclusive endpoints)."""
        n = self.js_levels
        if n < 2:
            return [self.js_min_db]
        step = (self.js_max_db - self.js_min_db) / (n - 1)
        return [round(self.js_min_db + i * step, 4) for i in range(n)]


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(
    benchmark_path: str | None = None,
    defenses_path: str | None = None,
) -> BenchmarkConfig:
    """Load and validate benchmark + defence configuration."""
    benchmark_path = benchmark_path or os.path.join(_CONFIG_DIR, "benchmark.yaml")
    defenses_path = defenses_path or os.path.join(_CONFIG_DIR, "defenses.yaml")

    b = _load_yaml(benchmark_path)
    d = _load_yaml(defenses_path)

    anchors = d["js_anchor_points_db"]
    defenses = []
    for item in d["defenses"]:
        ca = item["completion_anchors"]
        if len(ca) != len(anchors):
            raise ValueError(
                f"defence '{item['id']}' has {len(ca)} anchors, "
                f"expected {len(anchors)} to match js_anchor_points_db"
            )
        if any(not (0.0 <= v <= 1.0) for v in ca):
            raise ValueError(f"defence '{item['id']}' anchors must lie in [0,1]")
        defenses.append(
            Defense(
                id=item["id"],
                label=item["label"],
                completion_anchors=[float(v) for v in ca],
                description=item.get("description", "").strip(),
                reference=item.get("reference", ""),
            )
        )

    strat = b.get("stratified_effects", {}) or {}

    cfg = BenchmarkConfig(
        name=b["benchmark"]["name"],
        version=str(b["benchmark"]["version"]),
        total_flights=int(b["corpus"]["total_flights"]),
        missions=list(b["corpus"]["missions"]),
        receivers=list(b["corpus"]["receivers"]),
        js_min_db=float(b["ew_sweep"]["js_min_db"]),
        js_max_db=float(b["ew_sweep"]["js_max_db"]),
        js_levels=int(b["ew_sweep"]["js_levels"]),
        js_report_points_db=[float(x) for x in b["ew_sweep"]["js_report_points_db"]],
        repeats_per_point=int(b["sampling"]["repeats_per_point"]),
        seeds=[int(s) for s in b["sampling"]["seeds"]],
        confidence=float(b["sampling"]["confidence"]),
        labelling_standard=str(b["labelling"]["standard"]),
        regulatory_threshold=float(b["labelling"]["regulatory_threshold"]),
        js_anchor_points_db=[float(x) for x in anchors],
        defenses=defenses,
        stratified_enabled=bool(strat.get("enabled", False)),
        receiver_logit_delta=dict(strat.get("receiver_logit_delta", {})),
        mission_logit_delta=dict(strat.get("mission_logit_delta", {})),
    )
    return cfg
