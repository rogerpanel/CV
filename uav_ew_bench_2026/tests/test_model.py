"""Sanity tests for UAV-EW-Bench-2026 core model and analysis."""

import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from uavbench.analysis import wilson_interval
from uavbench.config import load_config
from uavbench.model import CompletionModel


def test_anchors_reproduced_exactly():
    """Curve passes through the calibrated anchor points."""
    cfg = load_config()
    model = CompletionModel(cfg)
    for d in cfg.defenses:
        for js, y in zip(cfg.js_anchor_points_db, d.completion_anchors):
            assert abs(model.base_probability(d.id, js) - y) < 1e-9


def test_curves_monotone_nonincreasing():
    """Completion must not increase with jamming intensity."""
    cfg = load_config()
    model = CompletionModel(cfg)
    grid = cfg.js_grid()
    for d in cfg.defenses:
        vals = [model.base_probability(d.id, js) for js in grid]
        for a, b in zip(vals, vals[1:]):
            assert b <= a + 1e-9


def test_ordering_of_defenses_at_20db():
    """Ours >= Seq2Seq >= CAF-CNN >= No-Def at the typical 20 dB point."""
    cfg = load_config()
    model = CompletionModel(cfg)
    p = {d.id: model.base_probability(d.id, 20.0) for d in cfg.defenses}
    assert p["ours_m1m4m6m7"] >= p["seq2seq_tr"] >= p["caf_cnn"] >= p["no_def"]


def test_crossings_match_plotted_anchors():
    """0.90 crossings derived from the figure's plotted anchor points.

    These are the *ground-truth* crossings implied by the coordinates in
    fig:uav_mission_completion (monotone PCHIP through the anchors):
        No-Def ~7.7, CAF-CNN ~11.5, Seq2Seq ~14.3, Ours ~25.0 dB.
    (The chapter prose rounds these to approx 7.5 / 13 / 18 / 27 dB; see
    docs/RECONCILIATION.md.)
    """
    cfg = load_config()
    model = CompletionModel(cfg)
    expected = {
        "no_def": 7.7,
        "caf_cnn": 11.5,
        "seq2seq_tr": 14.3,
        "ours_m1m4m6m7": 25.0,
    }
    for did, exp in expected.items():
        got = model.crossing_js(did, cfg.regulatory_threshold)
        assert abs(got - exp) <= 0.6, f"{did}: {got:.2f} vs {exp}"


def test_wilson_interval_basic():
    lo, hi = wilson_interval(180, 200, 0.95)
    assert 0.0 <= lo < 0.90 < hi <= 1.0
    lo0, hi0 = wilson_interval(200, 200, 0.95)
    assert hi0 == 1.0 and lo0 < 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
