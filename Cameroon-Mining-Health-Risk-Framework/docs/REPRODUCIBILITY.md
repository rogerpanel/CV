# Reproducibility checklist

The framework is engineered so that every figure and table in
Anaedevha et al. (2026) can be re-derived from the bundled configuration plus
the open datasets listed below.  Following this checklist guarantees the
manuscript's numerical results within Monte Carlo sampling error.

## A. Software environment

```bash
git clone https://github.com/rogerpanel/CV.git
cd CV/Cameroon-Mining-Health-Risk-Framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
earthengine authenticate            # only if running the GEE notebook
```

Supported Python versions: 3.10–3.12.  Pinned major dependency versions
appear in `requirements.txt` and `environment.yml`.

## B. Random seeds

| Component | Seed | File |
|-----------|------|------|
| Monte Carlo | 20260429 | `config/exposure_parameters.yaml` |
| Random Forest (geochem) | 42 | `config/hyperparameters.yaml` |
| Random Forest (LULC, GEE) | 42 | same |
| MaxEnt | 42 | same |
| Moran's I / LISA permutations | 42 | hard-coded in `geospatial.morans_i.global_morans_i` |
| Validation split | 7 | `config/hyperparameters.yaml :: random_forest_lulc.validation.seed` |

## C. Datasets

| Source | Access | Stewardship |
|--------|--------|-------------|
| Sentinel-2 SR (`COPERNICUS/S2_SR_HARMONIZED`) | GEE | ESA / Copernicus |
| Landsat 5/8/9 SR (`LANDSAT/LT05/...`, `LC08`, `LC09`) | GEE / EarthExplorer | USGS |
| WorldClim 2.1 | https://worldclim.org | CC-BY 4.0 |
| SRTM v3 (`USGS/SRTMGL1_003`) | GEE / NASA | Public domain |
| ESA WorldCover v200 | https://esa-worldcover.org | CC-BY 4.0 |
| GBIF | https://gbif.org | CC0 / CC-BY |
| WDPA | https://protectedplanet.net | UNEP-WCMC license |
| SoilGrids 2.0 | https://soilgrids.org | CC-BY 4.0 |
| CHIRPS daily (`UCSB-CHG/CHIRPS/DAILY`) | GEE | UC Santa Barbara |
| MODIS NDVI (`MODIS/061/MOD13Q1`) | GEE | NASA LP DAAC |
| **Cameroon DHS 2018** | https://dhsprogram.com/data/ | Restricted (registration required) |
| Heavy-metal compilation | Six peer-reviewed studies (see `data/references/geochemistry.bib`) | Authors |

## D. Running the principal pipeline

```bash
python -m scripts.run_full_pipeline \
    --geochem data/geochemical/sample_template.csv \
    --output  outputs/

# Notebook companions
jupyter lab notebooks/01_full_workflow_demo.ipynb
```

Outputs are written to:

* `outputs/pollution_indices.csv` — Igeo, CF, EF, PLI, RI, mCdeg, Nemerow PI
* `outputs/health_risk.csv` — HQ/HI/ILCR Monte Carlo summaries
* `outputs/spatial_autocorrelation.csv` — LISA + Moran's I
* `outputs/summary.json` — overall metrics

## E. Verification

* `pytest tests/` — unit tests covering pollution indices, MC reproducibility
  and Moran's I.
* The 10-iteration MaxEnt ensemble produces AUC ± 0.04 compared with the
  manuscript's reported 0.87 ± 0.04 across the five IUCN species.
* Children Cd HQ at Kambélé (water ingestion) reproduces the median 5.59 and
  95-percentile 8.74 reported in §3.3 within Monte Carlo noise (≈ ±2%).

## F. Known caveats

* The DHS 2018 dataset cannot be redistributed; users must register and
  download independently before running `notebooks/05_dhs_proximity.ipynb`.
* MaxEnt's *full* Phillips implementation (regularisation paths, λ-search)
  is delegated to the optional `elapid` backend; the default sklearn backend
  approximates it with engineered features.
* Earth Engine quotas may rate-limit batch exports; the framework retries
  with exponential back-off in `scripts/gee_export.py`.
