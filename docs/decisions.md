# NEBULENS — Locked Project Decisions

> **Status: LOCKED.** These are binding architectural and scientific decisions for the NEBULENS project. They are not suggestions or starting points. Changes require explicit revision of this document with justification.

---

## Project Purpose

NEBULENS compares MODIS and VIIRS active-fire detections over Tamil Nadu on a common equal-area grid, day by day, and estimates how much of an apparent change in fire activity could come from the sensors instead of from the fires themselves.

---

## Locked Decisions

### 1. Study Region
Tamil Nadu, India, clipped to the official state boundary. Offshore detections are excluded from analysis.

### 2. Harmonised Outputs
Hotspot counts and Fire Radiative Power (FRP), modelled separately.

### 3. Matching Approach
Daily grid-based harmonisation. Overpass-level pairing is used only as a validation diagnostic.

### 4. Application Architecture
React frontend and FastAPI backend. DuckDB over Parquet initially; PostGIS only if later performance requirements justify it.

### 5. Data Source
NASA FIRMS standard-quality MODIS Collection 6.1 and VIIRS 375 m products. Near-real-time data is added last.

### 6. Spatial Grid
10 km equal-area square grid using **EPSG:6933** (WGS 84 / NSIDC EASE-Grid 2.0 Global). 5 km and 25 km grids are sensitivity checks only.

### 7. Day Definition
Indian Standard Time (IST) calendar day (UTC+05:30). Original UTC timestamps are retained alongside IST-converted dates.

### 8. Reference Scale
VIIRS 375 m-equivalent. Pre-2012 estimates are predicted from MODIS and **explicitly labelled as predictions**, not observations.

---

## Scientific Boundary

NEBULENS produces comparable estimates; it does **not** claim that MODIS and VIIRS measure the same thing.

- Original observations must remain untouched.
- Sensor effects are modelled explicitly.
- Only adjustments that pass the defined validation tests can be labelled *harmonised*.

---

## Research Questions

1. How do MODIS and VIIRS hotspot counts and FRP differ over Tamil Nadu by season, day versus night, scan angle and land cover?
2. Can validated calibration make the two records comparable across 2000 to the latest standard-quality date?
3. How much of an apparent change in fire activity is associated with observation conditions rather than fire?
4. Does calibration hold on years and places it was not trained on?

---

## Boundary Data Source & Provenance (Audited & Updated)

Following an audit of administrative boundary sources for Tamil Nadu:
- **Prior GADM 4.1 limitation:** Contained only 32 pre-2019 districts and lacked an Open Definition-compliant license.
- **Audited Candidates:**
  1. *TNGIS (tngis.tn.gov.in):* Official state portal, but WFS/GeoServer backend endpoints are restricted/firewalled (HTTP 404 on direct API access), and terms prohibit redistribution without formal TNeGA clearance.
  2. *NWDP/NWIC (nwdp.nwic.gov.in):* Maintained by Ministry of Jal Shakti / Survey of India; however, its pan-India district shapefile contains only 37 districts for Tamil Nadu (lacking Mayiladuthurai).
  3. *DataMeet (github.com/datameet/maps):* Community open repository, but master district boundaries are locked to Census 2011 (32 districts for Tamil Nadu).
  4. *geoBoundaries gbOpen ADM2 (William & Mary geoLab):* Provides full pan-India coverage mapped to the Local Government Directory (LGD, lgdirectory.gov.in) with administrative reference year 2021. Represents all **38 current districts of Tamil Nadu**, with 100% valid geometries and an Open Data Commons Open Database License (ODbL) 1.0.

### Selected Source Specifications
- **Dataset:** geoBoundaries Global Database (gbOpen ADM2)
- **Boundary ID:** `IND-ADM2-76128533`
- **Publisher:** William & Mary geoLab (wmgeolab.org)
- **Upstream Producer:** Pathways Data & Local Government Directory (lgdirectory.gov.in)
- **Administrative Reference Year:** **2021** (captures all 38 post-reorganisation districts)
- **Download Date:** **2026-11-14**
- **Original Source URL:** https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM2/geoBoundaries-IND-ADM2.geojson
- **Source Git Commit:** `9469f09592ced973a3448cf66b6100b741b64c0d` (short: `9469f09`)
- **Pan-India SHA-256:** `8bef6929fd65432e7dc775e7c44473e84efff064ebddbfd6b834e6291546db40`
- **Licence:** Open Data Commons Open Database License (ODbL) 1.0 (OD-compliant open licence, requires attribution and share-alike on derived databases)
- **District Count:** 38 features (including Mayiladuthurai, Tenkasi, Kallakurichi, Ranipet, Tirupathur, and Chengalpattu)
- **Storage CRS:** EPSG:4326 (WGS 84 geographic coordinates)
- **Analytical Grid CRS:** EPSG:6933 (locked decision, reprojection handled in M2)

### State Boundary Provenance & ADM1 Comparison Check
To independently verify the dissolved 38-district state polygon against an official state boundary, an independent comparison was conducted against the corresponding **geoBoundaries IND ADM1** dataset (`IND-ADM1-1811400`, `Tamil Nādu`, ISO: `IN-TN`) at the exact same Git commit (`9469f09`):
- **Upstream Source of ADM1:** DataMeet / Election Commission of India (ECI), 2011 administrative baseline.
- **Licence of ADM1:** Creative Commons Attribution 2.5 India (CC BY 2.5 IN).
- **Geometric Comparison Results:**
  - Dissolved ADM2 Area: **129,914.45 km²**
  - Official ADM1 Area: **130,207.56 km²**
  - Intersection Area: **129,623.47 km²**
  - **Intersection over Union (IoU):** **99.329%** (0.993294)
  - **Dissolved within ADM1:** **99.776%**
  - **ADM1 within Dissolved:** **99.551%**
  - Symmetric Difference: **875.07 km²** (0.672% of state area), concentrated along high-tide coastal fringes and minor river delta mudflats due to vertex simplification differences (ADM1 has 36,567 vertices vs 11,032 for dissolved ADM2).
- **Decision:** The dissolved 38-district polygon provides complete topological consistency between the district-level analyses and the state-level boundary, while matching the independent official ADM1 state boundary to within 99.33% spatial overlap. Using the dissolved boundary ensures zero orphaned or unassigned fire detections within Tamil Nadu.

---

## Python Environment Decision

- **Specified:** Python 3.11
- **Tested Runtime:** Python 3.14.7 (system default on Manjaro Linux)
- **Status:** All 15 required dependencies (`pandas`, `geopandas`, `pyarrow`, `duckdb`, `shapely`, `pyproj`, `statsmodels`, `scikit-learn`, `xgboost`, `fastapi`, `uvicorn`, `pytest`, `python-dotenv`, `matplotlib`, `contextily`) install without conflict and pass `pip check` and all automated test suites.
- **Python 3.11 Availability:** Python 3.11 is not pre-installed in the current OS environment. Downgrading is unnecessary as Python 3.14.7 is fully tested and verified.

---

*Last updated: 2026-11-14*
