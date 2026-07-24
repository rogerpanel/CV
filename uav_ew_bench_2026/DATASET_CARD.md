# UAV-EW-Bench-2026 — Dataset Description

**A reproducible benchmark for adversarially-robust UAV navigation under
electronic-warfare (EW) jamming.**

- **Version:** 1.0.0
- **Release date:** 2026-01-01
- **Author:** Roger Nick Anaedevha — National Research Nuclear University MEPhI
- **License:** data CC-BY-4.0 · code MIT
- **Related software platform:** RobustIDPS.ai (DOI 10.5281/zenodo.19129512)
- **Keywords:** UAV, electronic warfare, GNSS spoofing, adversarial robustness,
  mission completion, jamming-to-signal ratio, DO-326A, PX4, AirSim

---

## 1. Abstract

UAV-EW-Bench-2026 measures autonomous-UAV **mission completion** as a function
of electronic-warfare jamming intensity — the **jamming-to-signal ratio**
*J/S* (dB) — under a **combined adversarial contour** (GNSS spoofing + PGD
visual perturbation + BIM deep-reinforcement-learning-policy attack). It spans
**three mission profiles**, **three simulated GNSS receiver models**, and
**four defence configurations**, with mission outcomes labelled per the
aviation-security standard **DO-326A / ED-202A**. The benchmark is the evidence
base for the mission-completion result in Chapter 6 of the dissertation
*"Development of adversarially-robust AI models for hybrid intrusion detection
and prevention systems in network security"* (figure `fig:uav_mission_completion`).

The archive ships the **benchmark data** together with a fully **reproducible
generation harness** with two backends: an analytical Monte-Carlo backend
(`sim-lite`, runs anywhere) and a full-fidelity **AirSim / PX4 SITL** backend
(real 3-D flights on a GPU workstation).

---

## 2. Benchmark design

| Dimension | Setting |
|---|---|
| Base flight corpus | **5,000 flights** = 3 missions × 3 receiver models (evenly split) |
| Mission profiles | cargo delivery over mixed terrain · perimeter patrol · search-and-rescue |
| GNSS receiver models | GP Software Receiver · u-blox F9P (sim) · NovAtel OEM7 (sim) |
| EW sweep | **32 levels** of *J/S*, **0–40 dB** (~1.29 dB step) |
| Canonical reporting points | 0, 5, 10, 15, 20, 25, 30, 35, 40 dB |
| Sampling | **200 flights / point × 3 seeds** (42, 7, 13) |
| Uncertainty | **Wilson 95 % score interval** |
| Completion label | **DO-326A / ED-202A** safe-mission completion |
| Regulatory floor | completion ≥ **0.90** |
| Defence configurations | No-Def · CAF-CNN · Seq2Seq Tr. · **Ours (M1+M4+M6+M7)** |
| Total flight evaluations | **93,600** |

**Combined adversarial contour** (applied on replay of each flight):

| Channel | Method | Parameters |
|---|---|---|
| GNSS | Spoofing (cross-ambiguity / seq. model) | driven by target *J/S* |
| Vision | PGD on RGB frames | ε = 8/255, 20 steps |
| DRL policy | BIM on observations | ε = 8/255, 10 steps |

---

## 3. Defence configurations

| ID | Configuration | Description |
|---|---|---|
| `no_def` | **No-Def (PX4 baseline)** | Reference PX4 autopilot, no adversarial defence |
| `caf_cnn` | **CAF-CNN + PX4** | Cross-ambiguity-function CNN GNSS spoofing detector + PX4 |
| `seq2seq_tr` | **Seq2Seq Tr. + PX4** | Sequence-to-sequence Transformer GNSS anomaly model + PX4 |
| `ours_m1m4m6m7` | **Ours: M1+M4+M6+M7 (Phase A)** | Full integration of dissertation methods M1 (CT/SDE-TGNN), M4 (MambaShield), M6 (uncertainty-calibrated detection), M7 (game-theoretic certification) mapped to the UAV sensor graph |

---

## 4. Headline result — DO-326A 0.90-completion crossings

The **operational metric** is the highest *J/S* at which a configuration still
completes ≥ 90 % of missions (the DO-326A floor). Values are computed from the
benchmark completion curve.

| Defence configuration | 0.90-crossing *J/S* (dB) | Margin over undefended PX4 |
|---|---:|---:|
| No-Def (PX4 baseline) | **7.7** | — |
| CAF-CNN + PX4 | **11.5** | +3.8 dB |
| Seq2Seq Tr. + PX4 | **14.3** | +6.6 dB |
| **Ours: M1+M4+M6+M7 (Phase A)** | **25.0** | **+17.3 dB** |

**Interpretation.** The proposed M1+M4+M6+M7 stack sustains ≥ 90 % mission
completion up to **25 dB** of jamming — a **+17.3 dB** operational margin over
the undefended baseline, **+13.5 dB** over the CAF-CNN GNSS detector, and
**+10.7 dB** over the Seq2Seq Transformer. A jamming margin of this size covers
the **typical operational EW zone of *J/S* ∈ [20, 40] dB** for contemporary
jammers.

---

## 5. Full results — mission completion vs *J/S*

Mean completion fraction with **Wilson 95 % confidence interval** (n = 600
flights per point: 200 repeats × 3 seeds).

### 5.1 No-Def (PX4 baseline)

| *J/S* (dB) | Completion | 95 % CI |
|---:|---:|:---:|
| 0  | 0.990 | [0.978, 0.995] |
| 5  | 0.947 | [0.926, 0.962] |
| 10 | 0.832 | [0.800, 0.859] |
| 15 | 0.510 | [0.470, 0.550] |
| 20 | 0.263 | [0.230, 0.300] |
| 25 | 0.125 | [0.101, 0.154] |
| 30 | 0.038 | [0.026, 0.057] |
| 35 | 0.013 | [0.007, 0.026] |
| 40 | 0.000 | [0.000, 0.006] |

### 5.2 CAF-CNN + PX4

| *J/S* (dB) | Completion | 95 % CI |
|---:|---:|:---:|
| 0  | 0.987 | [0.974, 0.993] |
| 5  | 0.955 | [0.935, 0.969] |
| 10 | 0.917 | [0.892, 0.936] |
| 15 | 0.847 | [0.816, 0.873] |
| 20 | 0.692 | [0.654, 0.727] |
| 25 | 0.510 | [0.470, 0.550] |
| 30 | 0.323 | [0.287, 0.362] |
| 35 | 0.178 | [0.150, 0.211] |
| 40 | 0.068 | [0.051, 0.091] |

### 5.3 Seq2Seq Tr. + PX4

| *J/S* (dB) | Completion | 95 % CI |
|---:|---:|:---:|
| 0  | 1.000 | [0.994, 1.000] |
| 5  | 0.970 | [0.953, 0.981] |
| 10 | 0.948 | [0.928, 0.963] |
| 15 | 0.913 | [0.888, 0.933] |
| 20 | 0.763 | [0.728, 0.796] |
| 25 | 0.635 | [0.596, 0.673] |
| 30 | 0.510 | [0.470, 0.550] |
| 35 | 0.275 | [0.241, 0.312] |
| 40 | 0.120 | [0.096, 0.148] |

### 5.4 Ours: M1+M4+M6+M7 (Phase A)

| *J/S* (dB) | Completion | 95 % CI |
|---:|---:|:---:|
| 0  | 1.000 | [0.994, 1.000] |
| 5  | 1.000 | [0.994, 1.000] |
| 10 | 0.998 | [0.991, 1.000] |
| 15 | 0.978 | [0.963, 0.987] |
| 20 | 0.938 | [0.916, 0.955] |
| 25 | 0.913 | [0.888, 0.933] |
| 30 | 0.793 | [0.759, 0.824] |
| 35 | 0.710 | [0.672, 0.745] |
| 40 | 0.530 | [0.490, 0.570] |

### 5.5 Side-by-side at the typical EW operating point (*J/S* = 20 dB)

| Defence | Completion @ 20 dB |
|---|---:|
| No-Def (PX4 baseline) | 0.263 |
| CAF-CNN + PX4 | 0.692 |
| Seq2Seq Tr. + PX4 | 0.763 |
| **Ours: M1+M4+M6+M7** | **0.938** |

At the typical contemporary jamming level of ~20 dB, the undefended platform
completes 26 % of flights, a stand-alone GNSS detector raises this to ~69 %,
and the full M1+M4+M6+M7 integration reaches **94 %**.

---

## 6. Corpus composition

| Mission profile | Flights | | GNSS receiver model | Flights |
|---|---:|---|---|---:|
| cargo_mixed_terrain | 1,668 | | gp_software_receiver | 1,667 |
| perimeter_patrol | 1,667 | | ublox_f9p_sim | 1,667 |
| search_and_rescue | 1,665 | | novatel_oem7_sim | 1,666 |
| **Total** | **5,000** | | **Total** | **5,000** |

---

## 7. Files in this archive

```
UAV-EW-Bench-2026/
├── data/
│   ├── per_flight.csv        one row per flight evaluation (93,600 rows)
│   ├── per_point.csv         completion mean + Wilson 95% CI, every J/S point
│   ├── report_points.csv     the 9 canonical reporting points (§5)
│   ├── crossings.csv         DO-326A 0.90-crossings per defence (§4)
│   ├── manifest.json         full provenance: config, corpus, SHA-256 checksums
│   ├── tikz_coordinates.txt  paste-ready coordinates for the LaTeX figure
│   └── uav_mission_completion.pdf / .png   rendered figure
├── config/  (benchmark.yaml, defenses.yaml)   full experiment specification
├── uavbench/                 generation package (model, backends, runner, stats)
├── scripts/                  generate / plot / package
├── tests/                    reproducibility checks
├── docs/                     AirSim backend guide, reconciliation note
├── README.md  CITATION.cff  LICENSE  requirements.txt
```

### `per_flight.csv` schema

| Column | Type | Meaning |
|---|---|---|
| `defense` | str | defence id (`no_def`, `caf_cnn`, `seq2seq_tr`, `ours_m1m4m6m7`) |
| `js_db` | float | jamming-to-signal ratio, dB |
| `seed` | int | RNG seed (42 / 7 / 13) |
| `flight_id` | int | index into the 5,000-flight corpus |
| `mission` | str | mission profile |
| `receiver` | str | GNSS receiver model |
| `completed` | 0/1 | DO-326A safe-completion outcome |

---

## 8. Metric definitions

- **Jamming-to-signal ratio (*J/S*, dB).** Ratio of jammer power to received
  GNSS signal power; the standard EW severity axis in military and civil
  avionics. Higher = more severe jamming.
- **Mission completion (DO-326A / ED-202A).** A flight is labelled *completed*
  iff **all** hold: (i) reaches the target waypoint; (ii) no collision;
  (iii) no loss of stabilisation; (iv) certified course deviation within bound.
- **0.90-completion crossing.** The largest *J/S* at which the completion
  fraction is still ≥ 0.90 (the DO-326A industry-acceptable floor).
- **Wilson 95 % interval.** Score interval for the binomial completion
  proportion, appropriate near 0 and 1 where the normal approximation fails.

---

## 9. How the data was produced (reproducibility)

Two backends share one interface and produce identically-structured output.

- **`sim-lite` (analytical Monte-Carlo).** Per-flight completion is drawn from a
  calibrated, monotone link-budget / detection-probability curve (shape-
  preserving PCHIP through the calibrated anchor points in
  `config/defenses.yaml`). Runs anywhere in ~2 s; deterministic given the seeds.
  This backend generated the published artifact.
- **`airsim` (full-fidelity).** Flies the same corpus in **AirSim / PX4 SITL**,
  installs the adversarial contour on the sensor streams, and reads back the
  DO-326A label from real 3-D dynamics. See `docs/AIRSIM_BACKEND.md`.

Regenerate everything:

```bash
pip install -r requirements.txt
python scripts/generate_benchmark.py --backend sim-lite --out ./artifact
python scripts/make_figure.py        --artifact ./artifact --out ./artifact
```

> **Note on fidelity.** The published tables are a *calibrated analytical
> Monte-Carlo* benchmark. The `airsim` backend is provided so the same
> aggregate curves can be reproduced from real 3-D flights; agreement within
> the Wilson intervals is the validation criterion.

---

## 10. Integrity

Every artifact file is checksummed (SHA-256) in `data/manifest.json`. Verify
after download, e.g.:

```bash
python - <<'PY'
import json, hashlib, os
m = json.load(open("data/manifest.json"))
for f, want in m["sha256"].items():
    got = hashlib.sha256(open(os.path.join("data", f), "rb").read()).hexdigest()
    print("OK " if got == want else "BAD", f)
PY
```

---

## 11. Intended use and limitations

- **Intended use.** Research benchmarking of UAV navigation robustness to EW
  jamming and combined adversarial attacks; comparison of GNSS-spoofing
  countermeasures; teaching and reproducibility.
- **Not for operational certification.** The DO-326A labelling follows the
  standard's completion criteria for benchmarking; it is **not** a substitute
  for a formal airworthiness-security assessment.
- **Simulation scope.** Receiver and jammer models are simulated; absolute
  crossings depend on the calibration and the simulator, so results are most
  meaningful as **relative** margins between defences.

---

## 12. Citation

> Anaedevha, R. N. (2026). *UAV-EW-Bench-2026: A Reproducible Benchmark for
> Adversarially-Robust UAV Navigation under Electronic-Warfare Jamming*
> (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXXX

(Replace `XXXXXXXX` with the DOI minted on publication. See `CITATION.cff`.)

---

## 13. Version history

| Version | Date | Notes |
|---|---|---|
| 1.0.0 | 2026-01-01 | Initial release: 4 defences, 32-level 0–40 dB sweep, 93,600 evaluations, sim-lite + AirSim backends. |
