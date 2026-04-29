# Data manifest

This folder collects **dataset templates, references and download scripts**.
The framework is designed to run end-to-end on freely available secondary
data; no primary-field samples are required.

## 1. Geochemical samples (`geochemical/`)

| File | Description |
|------|-------------|
| `sample_template.csv` | Wide-format template with all canonical columns expected by `cmhr.utils.load_geochem_table`. |
| `geochemical_compilation_2017_2024.csv` | Synthesised compilation (n=183) from six peer-reviewed studies — see `data/references/geochemistry.bib`. Replace with the actual harmonised values once permission to redistribute is obtained. |
| `regional_background_Mimba_2018.csv` | Regional baseline values used as `Bₙ` in `Igeo` and `CF`. |

### Source studies (East Cameroon, 2017–2024)

* Rakotondrabe, F. et al. 2017. *Sci Total Environ* 610–611, 831–844.
* Rakotondrabe, F. et al. 2018. *Environ Pollut* 240, 213–223.
* Dallou, G. B. et al. 2018. *Cameroon J Exp Biol* 12 (1).
* Fodoué, Y. et al. 2022. *J Sediment Environ* 7, 1–14.
* Edjengte Doumo, E. P. et al. 2023. *Heliyon* 9, e16037.
* Ngoa Manga, V. et al. 2024. *Environ Earth Sci* 83, 12.
* Fonshiynwa, F. et al. 2024. *Geosci Front* 15, 101742.

## 2. Remote sensing (`remote_sensing/`)

Only **GEE asset IDs** are listed here.  The actual imagery is fetched on the
fly with `scripts/gee_export.py`.

| Asset | Period | Resolution |
|-------|--------|------------|
| `LANDSAT/LT05/C02/T1_L2` | 2000 (baseline) | 30 m |
| `LANDSAT/LC08/C02/T1_L2` | 2013–2024 | 30 m |
| `LANDSAT/LC09/C02/T1_L2` | 2022–2024 | 30 m |
| `COPERNICUS/S2_SR_HARMONIZED` | 2015–2024 | 10–20 m |
| `USGS/SRTMGL1_003` | 2000 SRTM v3 | 30 m |
| `ESA/WorldCover/v200` | 2021 | 10 m |
| `UCSB-CHG/CHIRPS/DAILY` | 1981–present | 0.05° |
| `MODIS/061/MOD13Q1` | 2000–present | 250 m |

## 3. Health (`health/`)

The Cameroon Demographic and Health Survey (DHS) 2018 cluster shapefile and
recodes are obtained from the DHS Program after registration:
<https://dhsprogram.com/data/>.

Place the following inside `health/dhs_2018/`:

* `CMGE71FL.zip` — geographic displacement points (cluster locations)
* `CMHR71DT.ZIP` — household recode
* `CMKR71DT.ZIP` — children's recode
* `CMIR71DT.ZIP` — individual women's recode

## 4. Biodiversity (`biodiversity/`)

`scripts/download_datasets.py --gbif` queries GBIF using the bounding box
`(lat 3.5–6.5, lon 12–16)` and the manuscript species list, applies coordinate
QC, and writes `gbif_cameroon_east.csv`.

| Species | IUCN | Records (after QC) |
|---------|------|--------------------|
| *Loxodonta cyclotis* | EN | varies |
| *Gorilla gorilla gorilla* | CR | varies |
| *Pan troglodytes troglodytes* | EN | varies |
| *Panthera pardus* | VU | varies |
| *Osteolaemus tetraspis* | VU | varies |

## 5. References (`references/`)

* `geochemistry.bib` — 6 East-Cameroon studies (heavy metals).
* `methods.bib` — original method publications (Müller 1969, Håkanson 1980,
  USEPA 2011, Phillips 2006, etc.).
* `datasets.bib` — DHS, GBIF, WorldClim, SRTM, ESA WorldCover, SoilGrids.

Cite both the manuscript and this software (`CITATION.cff`) when reusing the
datasets.
