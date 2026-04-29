# Methodology overview

A condensed, citable description of every analytical step in
Anaedevha et al. (2026), with pointers to the corresponding source modules.

## 1. Land-use / land-cover change

| Step | Module / Function | Reference |
|------|-------------------|-----------|
| Cloud masking (Sentinel-2 SCL ≤ 7 + cloud probability < 20%) | `remote_sensing.cloud_mask_sentinel2` | Copernicus L2A handbook |
| Annual dry-season median composite (Dec–Feb) | `remote_sensing.annual_composite` | Manuscript §2.2 |
| Spectral indices (NDVI, EVI, BSI, NDWI, MNDWI, SAVI, NDMI) | `remote_sensing.spectral_indices` | Huete 2002; Rikimaru 2002; Xu 2006 |
| Random Forest classifier (500 trees, √n features, leaf=5, bag=0.7) | `remote_sensing.lulc_classifier.build_gee_random_forest` | Breiman 2001 |
| Stratified validation (n = 700, 100/class) + Cohen's κ | `remote_sensing.accuracy_assessment` | Landis & Koch 1977 |
| Change detection from→to matrix | `remote_sensing.change_detection.lulc_change_matrix` | Pontius & Millones 2011 |

## 2. Geochemical pollution indices

| Index | Formula | Module |
|-------|---------|--------|
| Geoaccumulation | $I_{geo}=\log_2(C_n/1.5 B_n)$ | `pollution_indices.geoaccumulation` |
| Contamination factor | $CF_i=C_i/C_{ref,i}$ | `pollution_indices.contamination_factor` |
| Pollution Load Index | $\bigl(\prod CF_i\bigr)^{1/n}$ | `pollution_indices.pollution_load_index` |
| Ecological Risk | $RI=\sum T^i_r CF_i$ | `pollution_indices.ecological_risk` |
| Enrichment Factor | $(C_{el}/C_{norm})_{sample}/(.)_{bg}$ | `pollution_indices.enrichment_factor` |
| Nemerow PI | $\sqrt{(\overline{PI}^2+PI_{max}^2)/2}$ | `pollution_indices.nemerow_index` |
| mC-deg | $\frac{1}{n}\sum CF_i$ | `pollution_indices.modified_indices` |
| WQI / HPI / C-deg | weighted-arithmetic | `water_quality.*` |

Background concentrations are taken from Mimba et al. (2018) and stored in
`config/reference_doses.yaml :: regional_background`.

## 3. Probabilistic health-risk assessment (USEPA 2011)

* **Three exposure pathways** — ingestion, dermal, inhalation —
  implemented in `health_risk.add_calculation`.
* **Exposure factors** for adults and children (`exposure_parameters.yaml`)
  are sampled by `health_risk._distributions.sample` for 10,000 iterations
  with a deterministic seed (20260429).
* **Reference doses (RfD)** and **cancer-slope factors (CSF)** are loaded
  from `reference_doses.yaml`; the `MonteCarloRiskAssessment` class composes
  HQ, HI and ILCR distributions and reports the 5/50/95-percentile values.
* **Sensitivity analysis** —
  * Spearman rank correlation (`health_risk.sensitivity_analysis.spearman_sensitivity`),
  * Saltelli-scheme Sobol' indices (`health_risk.sensitivity_analysis.sobol_sensitivity`).

## 4. Spatial epidemiology (DHS 2018)

* Cluster proximity to mining footprints classified into proximate / intermediate /
  distal (`epidemiology.proximity_analysis`).
* Outcome prevalence per class plus **adjusted logistic regression**
  with covariates *age*, *household education*, *wealth quintile*
  (`epidemiology.logistic_regression.adjusted_logit`).

## 5. Species distribution modelling

* MaxEnt regression with linear/quadratic/hinge features
  (`species_distribution.maxent_wrapper.MaxEntModel`).
* Multicollinearity screening at \|r\| ≤ 0.8 via
  `species_distribution.env_predictors.screen_multicollinearity`.
* 10-replicate bootstrap ensemble in
  `species_distribution.ensemble_sdm.EnsembleSDM`.
* Model acceptance: AUC > 0.7 and TSS > 0.4 (manuscript §3.4).

## 6. Spatial autocorrelation

* Global Moran's I (`geospatial.morans_i.global_morans_i`) with 999 permutations.
* LISA (`geospatial.lisa.local_morans_i`) for HH/LL/HL/LH classification.
* Getis-Ord G\* (`geospatial.hotspot_detection.getis_ord_g_star`) as supplement.
* Interpolation of sparse samples via ordinary kriging (`geospatial.kriging`)
  or IDW (`geospatial.idw`).

## 7. Reproducibility hooks

* All thresholds → `config/thresholds.yaml`
* All hyperparameters → `config/hyperparameters.yaml`
* All exposure factors → `config/exposure_parameters.yaml`
* All RfD / CSF / Tᵣ / background → `config/reference_doses.yaml`
* Study area / proximity classes → `config/study_area.yaml`

Edit one file, re-run the pipeline, and reproduce the manuscript tables.
