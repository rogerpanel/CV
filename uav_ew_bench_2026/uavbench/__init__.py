"""UAV-EW-Bench-2026 — adversarially-robust UAV navigation benchmark.

A reproducible benchmark measuring UAV mission-completion under
electronic-warfare jamming (jamming-to-signal ratio sweep) and a combined
adversarial contour (GNSS spoofing + PGD visual + BIM DRL-policy attack),
for four defence configurations.

Two execution backends share one interface:

* ``sim-lite``  — analytical Monte-Carlo over a calibrated link-budget /
  detection-probability model.  Runs anywhere, reproduces the dissertation
  figure exactly, and is what generates the published benchmark artifact.
* ``airsim``    — drives real AirSim / PX4 SITL 3-D flights on a GPU
  workstation.  Provided for full-fidelity regeneration; expected to
  reproduce the same aggregate curves.
"""

__version__ = "1.0.0"
__benchmark__ = "UAV-EW-Bench-2026"
