# Cameroon-Mining-Health-Risk-Framework

A reproducible computational framework for **Environmental Quality Assessment and Health
Risk Modeling in Artisanal Mining Landscapes of East Cameroon**, establishing a
transferable methodology for gemstone (gold, sapphire) mining impact evaluation.

This repository accompanies the manuscript:

> **Anaedevha, R. N., Delabrousse, N. M., & Kapralova, D. O. (2026).**
> *Environmental Quality Assessment and Health Risk Modeling in Artisanal Mining
> Landscapes of East Cameroon: Establishing a Transferable Framework for Gemstone
> Mining Impact Evaluation.* Environmental Science and Pollution Research.
> Source manuscript repository: https://github.com/rogerpanel/Cameroon-gold-mining-assessment

The codebase implements every analytical method described in the manuscript and a small
set of closely-related extensions (Enrichment Factor, Nemerow PI, Heavy-metal Pollution
Index, kriging/IDW interpolation, gradient-boosted ML, SHAP interpretability) so the
same pipeline can be applied to other artisanal mining landscapes (sapphire, cobalt,
coltan) with minimal change.

---

## 1. Scope

| Component | Module | Methods |
|-----------|--------|---------|
| Geochemical pollution | `src/pollution_indices/` | `I_geo`, `CF`, `PLI`, `RI`, `EF`, `Nemerow PI`, `mCdeg` |
| Water quality | `src/water_quality/` | WQI, HPI, Contamination Degree, SPI |
| Health risk (USEPA 2011) | `src/health_risk/` | ADD (ingestion / dermal / inhalation), HQ, HI, ILCR, Monte Carlo, Sobol' |
| Remote sensing | `src/remote_sensing/` | Spectral indices, GEE pipeline, RF land-cover classifier, Kappa accuracy |
| Spatial statistics | `src/geospatial/` | Global Moran's I, LISA, kriging, IDW, hotspot detection |
| Species distribution | `src/species_distribution/` | MaxEnt wrapper, ensemble SDM, mining-overlap analysis |
| Epidemiology | `src/epidemiology/` | DHS proximity classification, adjusted logistic regression |
| ML modelling | `src/ml_models/` | RF, XGBoost, LightGBM, ANN, SVR, GP, stacking ensemble, SHAP |
| Visualization | `src/visualization/` | Maps, plots, automated PDF reports |

All hyperparameters, exposure factors, reference doses, and toxic-response factors are
externalised in YAML files under [`config/`](config/) so they can be audited and changed
without touching code.

## 2. Study area and datasets

The framework was developed for three principal gold mining districts of Cameroon's
East Region — Bétaré-Oya (5°36'N, 14°05'E), Batouri/Kambélé/Pater (4°26'N, 14°22'E),
and Ngoura (5°02'N, 13°25'E) — covering approximately 9,914 km².

| Dataset | Source | Access | Notes |
|---------|--------|--------|-------|
| Landsat 5 TM (2000) | USGS Collection 2 L2 | https://earthexplorer.usgs.gov | 30 m |
| Landsat 8/9 OLI (2013–2024) | Google Earth Engine | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` | 30 m |
| Sentinel-2 MSI L2A (2015–2024) | Copernicus / GEE | `COPERNICUS/S2_SR_HARMONIZED` | 10–20 m |
| Geochemical samples (n = 183) | 6 peer-reviewed studies (2017–2024) | See `data/references/geochemistry.bib` | water=109, soil=42, sediment=32 |
| Cameroon DHS 2018 | DHS Program | https://dhsprogram.com/data/ | 54 East-region clusters |
| GBIF biodiversity | GBIF.org | https://www.gbif.org | 2,847 quality-controlled records |
| WorldClim 2.1 | WorldClim | https://www.worldclim.org/data/worldclim21.html | 19 bioclim, 30 arc-sec |
| SRTM elevation | NASA / USGS | `USGS/SRTMGL1_003` (GEE) | 30 m |
| ESA WorldCover | ESA | `ESA/WorldCover/v200` (GEE) | 10 m |
| WDPA protected areas | UNEP-WCMC | https://www.protectedplanet.net | Dja Faunal Reserve |
| SoilGrids | ISRIC | https://soilgrids.org | 250 m |
| CHIRPS rainfall | CHC / UCSB | `UCSB-CHG/CHIRPS/DAILY` (GEE) | 0.05° |
| MODIS NDVI | NASA LP DAAC | `MODIS/061/MOD13Q1` (GEE) | 250 m |
| Regional baseline | Mimba et al. 2018 | DOI in `data/references/` | Used as `B_n` |

The folder [`data/`](data/) contains CSV templates, dataset documentation, and a
download script (`scripts/download_datasets.py`).

## 3. Installation

```bash
git clone https://github.com/rogerpanel/CV.git
cd CV/Cameroon-Mining-Health-Risk-Framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core dependencies
pip install -e .                          # editable install of the `cmhr` package
# optional GEE access (requires a Google Earth Engine account)
earthengine authenticate
```

Conda users can instead run `conda env create -f environment.yml`.

## 4. Quick start

```python
import pandas as pd
from cmhr.pollution_indices import geoaccumulation_index, pollution_load_index
from cmhr.health_risk import MonteCarloRiskAssessment

df = pd.read_csv("data/geochemical/sample_template.csv")
backgrounds = {"As": 5.0, "Cd": 0.3, "Hg": 0.05, "Pb": 17.0}   # Mimba et al. 2018

df["Igeo_As"] = geoaccumulation_index(df["As_mg_kg"], backgrounds["As"])
print("PLI:", pollution_load_index(df, backgrounds))

mc = MonteCarloRiskAssessment(n_iterations=10_000, receptor="children")
risk = mc.run(concentration_mg_per_l=df["As_mg_l"].mean(), element="As")
print(risk.summary())
```

For end-to-end runs see the notebooks under [`notebooks/`](notebooks/) and the CLI:

```bash
python -m scripts.run_full_pipeline --config config/study_area.yaml
```

## 5. Reproducibility

* All hyperparameters and thresholds are stored in [`config/`](config/) — change one
  YAML key, re-run the pipeline, and reproduce the manuscript tables.
* Monte Carlo simulations use a deterministic seed (`config/exposure_parameters.yaml :
  random_seed`) so 10,000-iteration runs are bit-for-bit reproducible.
* Each notebook records the package versions in its first cell.
* The dataset manifest in [`data/README.md`](data/README.md) lists DOIs / GEE asset IDs
  for every input.

## 6. Citation

Please cite both the manuscript and this repository (see `CITATION.cff`).

## 7. License

MIT — see [`LICENSE`](LICENSE).
