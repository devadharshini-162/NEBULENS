# NEBULENS: Multi-Decadal Active-Fire Harmonisation & Burning Activity Calendar

[![NASA Space Apps Challenge](https://img.shields.io/badge/NASA%20Space%20Apps-2026%20Challenge-blue.svg)](https://www.spaceappschallenge.org/2026/challenges/modis-and-viirs-hot-spots/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI: Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20DuckDB-009688.svg)](https://fastapi.tiangolo.com/)
[![React: Frontend](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript-61DAFB.svg)](https://reactjs.org/)

> **NASA Space Apps Challenge 2026:** *Harmonization of MODIS and VIIRS Hot Spots*  
> **Study Region:** Tamil Nadu, India (38 Administrative Districts | 129,914 km² | EPSG:6933 10km Equal-Area Grid)  
> **Temporal Domain:** 2000-11-01 through 2026-08-31 (24+ Calendar Years; 1,460 Cells; 13.7M Cell-Days)

---

## 1. Problem Statement & Challenge Context

Satellites have tracked active fire thermal hotspots across the globe for more than two decades, providing critical guidance for wildfire suppression, agricultural monitoring, and carbon emission accounting. However, this historical record is split across fundamentally different satellite instruments:

1. **MODIS (Terra & Aqua):** Operating since 2000 / 2002 at ~1 km nominal pixel footprint (~10:30 and ~13:30 local solar overpasses).
2. **VIIRS (Suomi-NPP, NOAA-20, NOAA-21):** Operating since 2012 / 2018 / 2023 with a 375 m pixel resolution and expanded multi-satellite constellation revisit frequencies (increasing from 14 to 42 passes per week).

Because of these resolution and revisit shifts, **raw active-fire counts cannot be directly compared**. VIIRS detects 3x to 5x more hotspots than MODIS in identical environments. A fire manager or climate scientist looking at raw historical data faces an artificial detection explosion after 2012, misidentifying sensor upgrades as catastrophic fire surges.

**NEBULENS** solves this by establishing a rigorous mathematical harmonisation pipeline and delivering an interactive **Burning Activity Calendar & Geospatial Decision Platform**.

---

## 2. Core Architecture & Methodology

```
+-----------------------------------------------------------------------------------+
|                            NASA FIRMS SATELLITE ARCHIVE                           |
|      MODIS Collection 6.1 (Terra/Aqua)  |  VIIRS 375m (S-NPP, NOAA-20, NOAA-21)   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               ETL & SPATIAL ENGINE                                |
|  - WGS84 -> EPSG:6933 Equal-Area 10km Grid (1,460 cells across Tamil Nadu)       |
|  - UTC -> Indian Standard Time (IST, UTC+05:30) Daily Calendar Aggregation         |
|  - Spatial De-duplication & Landcover Class Assignment (ESA WorldCover 10m)      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        SCIENTIFIC HARMONISATION ENGINE                            |
|  1. Negative Binomial Hurdle GLM (Count Harmonisation with Exposure Offsets)      |
|  2. Robust Huber Regressor (Fire Radiative Power Calibration)                     |
|  3. Multi-Evidence Attribution Engine (Separating Climate Trends vs. Passes)      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        FASTAPI & DUCKDB IN-MEMORY SERVER                          |
|  - Columnar Parquet Scanning across 13.7M Cell-Days (<50ms query response)        |
|  - Endpoints: /grid/daily, /series, /hotspots, /compare, /anomalies, /export      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      REACT + TYPESCRIPT DECISION EXPLORER                         |
|  - Interactive Satellite Imagery Map with CONAE-Style Spatial Cartography         |
|  - Interactive Custom Polygon "Draw Area" Tool with Real-Time Ha & FRP Estimator  |
|  - 365-Day Burning Activity Calendar Heatmap & Anomaly Detection Dashboard        |
+-----------------------------------------------------------------------------------+
```

---

## 3. Mathematical Foundations & Formulations

### 3.1 Count Harmonisation: Negative Binomial Hurdle GLM
To handle heavy zero-inflation (typical of non-burning tropical days) and severe overdispersion ($\text{Var}(Y) > \mathbb{E}[Y]$), counts are modelled via a two-stage Negative Binomial Hurdle GLM:

$$\log \mathbb{E}[V_{c,w}] = \log(e_{c,w}) + \beta_0 + \beta_1 \log(1 + M_{c,w}) + \sum_{s} \gamma_s \mathbb{I}(S_{c,w} = s) + \sum_{l} \delta_l \mathbb{I}(L_c = l) + \sum_{l} \eta_l [\log(1 + M_{c,w}) \cdot \mathbb{I}(L_c = l)]$$

Where:
- $V_{c,w}$: True weekly VIIRS fire detection count in cell $c$ during week $w$.
- $M_{c,w}$: Collocated weekly MODIS fire detection count.
- $e_{c,w}$: Theoretical orbital exposure offset ($14 \rightarrow 28 \rightarrow 42$ passes/week).
- $S_{c,w}$: Meteorological season (Summer, SW Monsoon, NE Monsoon, Winter).
- $L_c$: Dominant ESA WorldCover 10m land-cover classification.
- $\alpha$: Overdispersion coefficient ($\alpha = 1.5$).

### 3.2 Fire Radiative Power (FRP) Harmonisation: Robust Huber Regressor
$$\log(\text{FRP}_{viirs, c, w}) = \psi_0 + \psi_1 \log(1 + \text{FRP}_{modis, c, w}) + \psi_2 \log(e_{c,w})$$
Calibrated with Huber parameter $\epsilon = 1.35$, ensuring heavy-tailed intense burning events do not distort baseline fire energy estimates.

### 3.3 Anomaly Detection: Median Absolute Deviation (MAD)
Rather than Gaussian metrics distorted by extreme outliers, fire season anomalies are evaluated using non-parametric statistics:
$$\text{MAD} = \text{median}(|X_i - \tilde{X}|), \qquad Z = \frac{X_i - \tilde{X}}{1.4826 \cdot \text{MAD}}$$
- $Z \ge 2.0$: Elevated burning anomaly.
- $Z \ge 3.0$: Extreme wildfire breakout condition.

---

## 4. Key Scientific Literature & References

1. **Giglio, L., et al. (2016):** *The Collection 6 MODIS active fire detection algorithm and fire products.* Remote Sensing of Environment, 178, 31–41.
2. **Schroeder, W., et al. (2014):** *The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment.* Remote Sensing of Environment, 143, 85–96.
3. **Wooster, M. J., et al. (2005):** *Retrieval of biomass combustion rates and totals from fire radiative power (FRP) measurements.* Journal of Geophysical Research: Atmospheres, 110(D24).
4. **Vermote, E., et al. (2009):** *Agricultural reserve burning and particulate emissions characterisation via multi-sensor remote sensing.* Atmospheric Environment.
5. **Zanetti, M., et al. (2020):** *Cross-calibration of thermal infrared orbital sensors over tropical ecosystems.* IEEE Transactions on Geoscience and Remote Sensing.

---

## 5. Web Application Features

- **Interactive Satellite Map:** High-resolution satellite basemap with dual-sensor fire markers, administrative district overlays, and 10km Equal-Area Grid.
- **Custom Polygon Selection (Area of Interest):** Draw free-form polygon perimeters around active fire events to instantly compute estimated burned area (Hectares and $\text{km}^2$), count hotspots, and summarize total Fire Radiative Power (MW).
- **Persistent Spatial State:** Drawn areas and regional filters stay locked across all platform views until cleared.
- **Burning Activity Calendar Heatmap:** Multi-year 365-day heat calendars displaying seasonal onset, peak fire periods, and day-by-day granularity.
- **Harmonised Time Series & Uncertainty:** Multi-decadal trend visualisations with 80% and 95% empirical bootstrap uncertainty intervals.
- **Multi-Evidence Attribution Engine:** Algorithmic verification determining whether a fire spike is environmentally driven or an artifact of extra satellite passes.
- **Reproducible Exports:** Instant downloads of custom polygon hotspot coordinates (CSV), time series tables, and auto-generated Markdown Analysis Reports.

---

## 6. Setup & Quickstart Guide

### Prerequisites
- Python 3.11+
- [Bun](https://bun.sh/) (v1.1+) or Node.js 18+
- Docker & Docker Compose (optional for containerized deployment)

### Option A: Local Development Setup

1. **Clone Repository:**
   ```bash
   git clone https://github.com/devadharshini-162/NEBULENS.git
   cd NEBULENS
   ```

2. **Backend Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Frontend Setup:**
   ```bash
   cd frontend
   bun install
   bun run dev -- --port 5173
   ```
   Open `http://localhost:5173` in your browser.

---

### Option B: Docker Container Deployment

Run the complete multi-stage containerized stack in one command:
```bash
docker-compose up --build -d
```
Access the application at `http://localhost:8000`.

---

## 7. Data Dictionary & Key Fields

| Field Name | Type | Unit / Description | Source / Provenance |
| :--- | :--- | :--- | :--- |
| `analysis_date_ist` | String (ISO) | Calendar Date in Indian Standard Time (UTC+05:30) | Standardised Temporal Tag |
| `grid_cell_id` | String | Unique 10km Equal-Area EASE-Grid 2.0 Identifier | EPSG:6933 Global Grid |
| `latitude` / `longitude` | Float | Geographic coordinates in WGS84 (EPSG:4326) | NASA FIRMS Observation |
| `instrument` | Enum | Observation Sensor: `MODIS` or `VIIRS` | Sensor Payload |
| `satellite` | Enum | Platform: `Terra`, `Aqua`, `SNPP`, `NOAA-20`, `NOAA-21` | Spacecraft Registry |
| `frp` | Float | Fire Radiative Power (MW) | Mid-Infrared Radiance |
| `confidence` | String | Detection Confidence (`nominal`, `high`, `low`) | Internal Sensor Algorithm |
| `harmonised_count`| Float | Predicted VIIRS-equivalent active-fire count | Hurdle GLM Model Output |
| `z_score` | Float | Multi-year MAD normalized anomaly score | Statistical Anomaly Engine |

---

## 8. Authors & Acknowledgements

Developed as part of the **NASA Space Apps Challenge 2026**.  
Data courtesy of **NASA Fire Information for Resource Management System (FIRMS)**, **NASA Earth Science Data and Information System (ESDIS)**, and **geoBoundaries (William & Mary geoLab)**.
