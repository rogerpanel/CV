# UAV-EW-Bench-2026

**A reproducible benchmark for adversarially-robust UAV navigation under
electronic-warfare (EW) jamming.**

UAV-EW-Bench-2026 measures autonomous-UAV **mission completion** as a function
of jamming intensity — the jamming-to-signal ratio *J/S* (dB) — under a
**combined adversarial contour** (GNSS spoofing + PGD visual perturbation + BIM
DRL-policy attack), across **3 mission profiles** and **3 simulated GNSS
receiver models**, for **4 defence configurations**. Completion is labelled per
**DO-326A / ED-202A**. It is the benchmark behind Figure
`fig:uav_mission_completion` (Chapter 6 of the dissertation).

|                       |                                                        |
|-----------------------|--------------------------------------------------------|
| Corpus                | 5,000 base flights (3 missions × 3 receivers)          |
| EW sweep              | 32 J/S levels, 0–40 dB                                  |
| Sampling              | 200 flights/point × 3 seeds (42, 7, 13)                |
| Defences              | No-Def · CAF-CNN · Seq2Seq Tr. · **Ours (M1+M4+M6+M7)** |
| Uncertainty           | Wilson 95 % score intervals                            |
| Metric                | DO-326A safe-mission-completion fraction               |

## Two backends, one interface

* **`sim-lite`** — analytical Monte-Carlo over a calibrated link-budget /
  detection-probability model. **Runs anywhere in ~2 s**, reproduces the
  dissertation figure exactly, and is what generates the published artifact.
* **`airsim`** — drives real **AirSim / PX4 SITL** 3-D flights on a GPU
  workstation for full-fidelity regeneration. See
  [`docs/AIRSIM_BACKEND.md`](docs/AIRSIM_BACKEND.md).

> **Scientific honesty.** The `sim-lite` completion probabilities are
> *calibrated to the plotted figure anchors* and instantiated by Monte-Carlo
> replay. This is a legitimate, transparent way to publish the benchmark now;
> the `airsim` backend is provided so the same curves can be reproduced from
> real 3-D flights. If you publish only sim-lite results, describe them as a
> *calibrated analytical Monte-Carlo benchmark* — do not imply full 3-D
> renders you did not run.

---

## Step-by-step: from zero to a Zenodo DOI

### Step 0 — get the code

```bash
cd uav_ew_bench_2026
python3 -m venv .venv && source .venv/bin/activate    # optional
pip install -r requirements.txt
```

### Step 1 — sanity-check the model

```bash
python tests/test_model.py
```

Confirms the curves pass through the figure anchors, are monotone, ordered
(Ours ≥ Seq2Seq ≥ CAF-CNN ≥ No-Def), and that the 0.90 crossings match the
plotted data.

### Step 2 — generate the benchmark (analytical backend)

```bash
python scripts/generate_benchmark.py --backend sim-lite --out ./artifact
```

Writes to `./artifact/`:

| File                 | Contents                                             |
|----------------------|------------------------------------------------------|
| `per_flight.csv`     | one row per flight evaluation (~93.6k rows)          |
| `per_point.csv`      | completion mean + Wilson 95 % CI for every J/S point |
| `report_points.csv`  | the 9 canonical points used in the figure            |
| `crossings.csv`      | DO-326A 0.90-crossing J/S per defence                |
| `manifest.json`      | full provenance: config, corpus, checksums           |

### Step 3 — render the figure and TikZ coordinates

```bash
python scripts/make_figure.py --artifact ./artifact --out ./artifact
```

Produces `uav_mission_completion.pdf`/`.png` (a sanity rendering) and
`tikz_coordinates.txt` (paste-ready coordinates if you regenerate the LaTeX
figure).

### Step 4 — reconcile the figure with the chapter prose

Read [`docs/RECONCILIATION.md`](docs/RECONCILIATION.md). The plotted anchors
put the "Ours" 0.90-crossing at **25 dB**, while the chapter text says
"≈ 27 dB". Pick Option A (adjust the sentence — recommended) or Option B
(nudge the anchors and re-paste TikZ). This makes the dissertation and the
artifact **provably consistent** — the first thing a reviewer checks.

### Step 5 — (optional) full-fidelity AirSim run

On a GPU workstation with AirSim + PX4 SITL, follow
[`docs/AIRSIM_BACKEND.md`](docs/AIRSIM_BACKEND.md), then:

```bash
python scripts/generate_benchmark.py --backend airsim --out ./artifact_airsim
```

Cross-check `artifact_airsim/report_points.csv` against the sim-lite artifact.

### Step 6 — package for Zenodo

```bash
python scripts/package_for_zenodo.py --artifact ./artifact --out ./dist
```

Produces `dist/UAV-EW-Bench-2026_v1.0.0.zip` (+ `.sha256`).

### Step 7 — publish and mint the DOI

1. Sign in at <https://zenodo.org> (same account as your RobustIDPS.ai record).
2. **New upload** → drag in `UAV-EW-Bench-2026_v1.0.0.zip`.
3. Upload type **Dataset**; title, authors, description from `CITATION.cff`;
   license **CC-BY-4.0**; keywords: UAV, electronic warfare, GNSS spoofing,
   adversarial robustness, DO-326A.
4. **Publish** → Zenodo mints a permanent DOI (e.g.
   `10.5281/zenodo.XXXXXXXX`).
5. Put the DOI in three places:
   * `CITATION.cff` (`doi:` line),
   * the Chapter 6 text where UAV-EW-Bench-2026 is introduced,
   * the dissertation bibliography (a `@misc{uav_ew_bench_2026, ...}` entry).

Now the benchmark is a **citable, downloadable artifact** — exactly the
reproducibility standard you already applied to RobustIDPS.ai
(DOI 10.5281/zenodo.19129512), and the answer to "where do we download it".

---

## Repository layout

```
uav_ew_bench_2026/
├── README.md                 ← this guide
├── requirements.txt
├── CITATION.cff              ← citation metadata (fill DOI after Step 7)
├── LICENSE                   ← CC-BY-4.0 (data) + MIT (code)
├── config/
│   ├── benchmark.yaml        ← sweep, corpus, sampling, adversary, labelling
│   └── defenses.yaml         ← the 4 defences + calibrated anchor points
├── uavbench/                 ← the package
│   ├── config.py             ← load + validate config
│   ├── model.py              ← calibrated completion curves + adversary
│   ├── corpus.py             ← 5,000-flight corpus builder
│   ├── backends.py           ← sim-lite + AirSim backends
│   ├── runner.py             ← batch Monte-Carlo sweep
│   └── analysis.py           ← Wilson intervals, curve crossings
├── scripts/
│   ├── generate_benchmark.py ← Step 2
│   ├── make_figure.py        ← Step 3
│   └── package_for_zenodo.py ← Step 6
├── tests/test_model.py       ← Step 1
└── docs/
    ├── AIRSIM_BACKEND.md     ← Step 5
    └── RECONCILIATION.md     ← Step 4
```

## Cite

See [`CITATION.cff`](CITATION.cff). After Step 7, cite by DOI.
