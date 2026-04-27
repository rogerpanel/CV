"""Visualization-module tests.  Run with `pytest tests/test_viz.py`."""
from __future__ import annotations

import os

import pytest

# Headless matplotlib so the tests don't try to open a display.
os.environ.setdefault("MPLBACKEND", "Agg")


def test_spectral_signature_plot_runs(tmp_path):
    pytest.importorskip("matplotlib")
    from fedloraguard.viz import spectral_signature_plot

    fig = spectral_signature_plot(
        benign_sigmas=[1.85, 1.42, 1.18, 0.97, 0.81, 0.65, 0.51, 0.39],
        backdoored_sigmas=[3.42, 1.36, 1.13, 0.93, 0.78, 0.62, 0.49, 0.37],
        output_path=str(tmp_path / "fig3"),
    )
    assert fig is not None
    assert (tmp_path / "fig3.pdf").exists() and (tmp_path / "fig3.png").exists()


def test_pareto_plot_runs(tmp_path):
    pytest.importorskip("matplotlib")
    from fedloraguard.viz import privacy_utility_pareto_plot

    fig = privacy_utility_pareto_plot([
        {"label": "FedLoRAGuard", "epsilon_T": 5.0, "macro_f1": 96.4},
        {"label": "DP-FedAvg", "epsilon_T": 5.0, "macro_f1": 85.7},
    ], output_path=str(tmp_path / "fig4"))
    assert fig is not None


def test_radar_plot_runs(tmp_path):
    pytest.importorskip("matplotlib")
    from fedloraguard.viz import achievement_radar_plot

    fig = achievement_radar_plot(
        {"FedLoRAGuard": [96.4, 98.4, 80, 75, 96.6, 30, 99]},
        output_path=str(tmp_path / "fig5"),
    )
    assert fig is not None
