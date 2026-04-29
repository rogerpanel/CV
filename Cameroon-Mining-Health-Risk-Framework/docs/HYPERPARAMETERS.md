# Hyperparameter and threshold reference

Every adjustable number in the framework is centralised in YAML files under
`config/`.  This document lists what each key controls and where it appears in
Anaedevha et al. (2026).

## `config/study_area.yaml`

| Key | Manuscript reference | Notes |
|-----|----------------------|-------|
| `districts.*.centroid` | §2.1 Table 1 | Bétaré-Oya (5.60°N, 14.083°E); Batouri (4.433°N, 14.367°E); Ngoura (5.033°N, 13.417°E) |
| `total_area_km2` | §2.1 | 9,914 km² |
| `bounding_box` | §3.4 | GBIF query window (3.5–6.5°N, 12–16°E) |
| `mining_proximity_classes_km` | §2.5 | DHS proximity strata 0–10/10–25/>25 km |
| `protected_areas.dja_faunal_reserve` | §3.4 | 526,000 ha; buffers 10/25/50/80 km |

## `config/thresholds.yaml`

| Key | Source | Used in |
|-----|--------|---------|
| `geoaccumulation_index` | Müller (1969) | `pollution_indices.classify_igeo` |
| `contamination_factor` | Håkanson (1980) | `pollution_indices.classify_cf` |
| `ecological_risk_index` | Håkanson (1980) | `pollution_indices.classify_ri` |
| `enrichment_factor` | Sutherland (2000) | `pollution_indices.classify_ef` |
| `nemerow_pollution_index` | Cheng et al. (2007) | `pollution_indices.classify_nemerow` |
| `cancer_risk` | USEPA (2011) | `health_risk.classify_cancer_risk` |
| `cohen_kappa` | Landis & Koch (1977) | `remote_sensing.classify_kappa` |

## `config/reference_doses.yaml`

Holds RfD/CSF/Tᵣ tables for ingestion, dermal and inhalation pathways together
with the East-Cameroon regional background (Mimba et al. 2018) and WHO
drinking-water guidelines.

| Section | Source |
|---------|--------|
| `toxic_response_factor` | Håkanson (1980); Pejman et al. (2015) |
| `reference_dose_*` | USEPA IRIS / Exposure Factors Handbook (2011) |
| `cancer_slope_factor_*` | USEPA IRIS |
| `regional_background` | Mimba et al. (2018) |
| `who_water_standards` | WHO Drinking-Water Guidelines (2017, 2022) |

## `config/exposure_parameters.yaml`

Reproduces Table 3 of the manuscript.

| Parameter | Adults | Children |
|-----------|--------|----------|
| BW | 𝒩(70, 10) | 𝒩(15, 3) |
| IR_water | Tri(1.0, 2.5, 3.5) | Tri(0.5, 0.78, 1.5) |
| IR_soil | LN(100, 1.5) | LN(200, 1.8) |
| EF | U(300, 365) | U(300, 365) |
| ED | 𝒩(24, 5) | 6 (deterministic) |
| SA | 𝒩(5700, 900) | 𝒩(2800, 500) |

`n_iterations = 10000`, `random_seed = 20260429`, percentiles 5/50/95.
The Sobol' analysis uses `n_base_samples = 2048` and 1000 bootstrap resamples.

## `config/hyperparameters.yaml`

| Block | Manuscript counterpart |
|-------|------------------------|
| `random_forest_lulc` | GEE LULC RF (500 trees, √n, leaf=5, bag=0.7) |
| `random_forest_geochem` | sklearn RF for tabular geochemistry |
| `xgboost`, `lightgbm`, `svr`, `gaussian_process` | Extensions for benchmarking |
| `ann` | Keras MLP regressor (3-layer 128→64→32, ReLU, Adam) |
| `stacking` | RF + XGB + LGBM + SVR → Ridge meta-learner |
| `maxent` | Phillips et al. (2006), 10 replicates, β=1, AUC>0.7, TSS>0.4 |
| `spatial_autocorrelation` | 999 permutations, queen / 5 km distance band |
| `dhs_logistic` | Adjusted for age, household education, wealth quintile |
