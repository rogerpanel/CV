# Dataset documentation

Detailed metadata, access instructions and citation requirements for every
input used by the framework.

## 1. Remote sensing imagery

### 1.1 Landsat 5 TM (Collection 2 Level-2)

* **Asset ID (GEE):** `LANDSAT/LT05/C02/T1_L2`
* **Coverage:** 1984–2012 (used for the year-2000 baseline)
* **Resolution:** 30 m
* **Bands used:** SR_B1–SR_B5, SR_B7
* **Scaling:** SR = DN × 2.75 × 10⁻⁵ − 0.2

### 1.2 Landsat 8/9 OLI (Collection 2 Level-2)

* **Asset IDs:** `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`
* **Period used:** 2013–2024
* **Scaling:** identical to Landsat 5

### 1.3 Sentinel-2 MSI L2A

* **Asset ID:** `COPERNICUS/S2_SR_HARMONIZED`
* **Period used:** 2015–2024
* **Resolution:** 10 / 20 m
* **Cloud filter:** SCL ∈ {4,5,6,7} & MSK_CLDPRB < 20 & scene CLOUDY_PIXEL_PERCENTAGE < 30%

## 2. Auxiliary geospatial layers

| Variable | Source | Asset ID / URL |
|----------|--------|----------------|
| Elevation | SRTMv3 | `USGS/SRTMGL1_003` |
| Land cover | ESA WorldCover v200 | `ESA/WorldCover/v200` |
| Rainfall | CHIRPS | `UCSB-CHG/CHIRPS/DAILY` |
| NDVI time-series | MODIS | `MODIS/061/MOD13Q1` |
| Soil properties | SoilGrids 2.0 | https://soilgrids.org |
| Bioclim (19 vars) | WorldClim 2.1 | https://worldclim.org/data/worldclim21.html |
| Protected areas | WDPA | https://www.protectedplanet.net |

## 3. Geochemical compilation

183 georeferenced samples (water n=109; soil n=42; sediment n=32) extracted
from six peer-reviewed studies.  Quality criteria:

* AAS calibration R² ≥ 0.995, or
* ICP-MS with standard QC, or
* EDXRF certified protocol.

Samples lacking coordinates, depth or date are excluded.  Below-MDL values
substituted with MDL/2 per USEPA guidance.

| Citation | Records | Matrices | Districts |
|----------|--------:|----------|-----------|
| Rakotondrabe et al. 2017 | 32 | water | Bétaré-Oya |
| Rakotondrabe et al. 2018 | 28 | sediment | Bétaré-Oya |
| Dallou et al. 2018 | 24 | soil | Pawara |
| Fodoué et al. 2022 | 32 | soil | Kambélé |
| Edjengte Doumo et al. 2023 | 21 | water/sed | Ngoura |
| Ngoa Manga et al. 2024 | 26 | water | Bindiba |
| Fonshiynwa et al. 2024 | 20 | sediment | Lom Basin |

Regional background values (`Bₙ`) come from **Mimba et al. 2018**
(`data/geochemical/regional_background_Mimba_2018.csv`).

## 4. DHS 2018 (Cameroon)

* **Source:** DHS Program (https://dhsprogram.com/data/)
* **Files used:** `CMGE71FL.zip`, `CMHR71DT.ZIP`, `CMKR71DT.ZIP`, `CMIR71DT.ZIP`
* **East-region clusters:** 54 (12 proximate, 18 intermediate, 24 distal)
* **GPS displacement:** 2 km (urban) / 5 km (rural; 1% up to 10 km)
* **Outcomes used:**
  * Acute respiratory infection (children <5)
  * Diarrhoea (children <5)
  * Anaemia (women 15–49)
  * Stunting / wasting (children <5)

## 5. GBIF biodiversity

* **Search terms:** five IUCN species; bbox lat 3.5–6.5 lon 12–16
* **Cleaning:**
  * remove records with `coordinateUncertaintyInMeters > 5000`
  * flag institutional coordinates (e.g. country / capital city centroids)
  * keep only observations from 2000–2024
* **Final dataset:** 2,847 records

## 6. Reference doses & cancer slope factors

All RfD/CSF values used in the Monte Carlo engine originate from the USEPA
Integrated Risk Information System (IRIS) and the *Exposure Factors Handbook
(2011 final edition)*.  Dermal RfDs follow the RAGS Part E adjustment.

## 7. Citation requirements

When publishing results derived from this framework, please cite:

1. Anaedevha, R. N., Delabrousse, N. M., & Kapralova, D. O. (2026). *Environmental
   Quality Assessment and Health Risk Modeling in Artisanal Mining Landscapes
   of East Cameroon.* Environ Sci Pollut Res.
2. The relevant data-source citations from `data/references/datasets.bib` and
   `data/references/geochemistry.bib`.
3. The software (`CITATION.cff`).
