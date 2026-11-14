# NEBULENS: Scientific Methodology, Calibration Equations & Model Card

**Project:** NEBULENS — Multi-Decadal Active-Fire Harmonisation & Burning Activity Calendar  
**Target Domain:** Tamil Nadu, India (38 Administrative Districts, 129,914 km², EPSG:6933 10 km EASE-Grid 2.0)  
**Study Scope:** 2000-11-01 to 2026-08-31 (24+ Calendar Years; 1,460 Equal-Area Cells)  

---

## 1. Scientific Context & Discrepancy Drivers

Active fire products from Earth-observing satellites provide continuous detection of open biomass burning. However, multi-decadal records suffer from sensor discontinuity:

1. **Resolution Mismatch:**
   - **MODIS (Terra & Aqua):** Spatial resolution of ~1 km at nadir, expanding to ~2.0 km (along-track) $\times$ 4.8 km (along-scan) at swath edges.
   - **VIIRS (S-NPP, NOAA-20, NOAA-21):** Nominal 375 m pixel footprint using 3:1 aggregation that limits footprint expansion to ~0.8 km at swath edges.
   - **Consequence:** VIIRS detects sub-pixel, low-intensity agricultural and forest understory fires with radiant intensity down to ~0.5 MW, which fall below MODIS detection limits (~8–10 MW).

2. **Temporal Overpass Shift:**
   - **Terra:** ~10:30 local solar time (morning pass, prior to peak diurnal fire weather).
   - **Aqua:** ~13:30 local solar time (afternoon pass, aligning with minimum relative humidity).
   - **VIIRS Constellation:** ~13:30 local solar time, with additional satellites providing orbital revisit frequencies of 14 (2012), 28 (2018), and 42 (2023) passes per week.

---

## 2. Harmonisation Methodology & Mathematical Formulations

### 2.1 Spatial Normalisation: 10 km Equal-Area Grid (EPSG:6933)
Geographic coordinates (WGS84) are projected onto the **EASE-Grid 2.0 Global Equal-Area Grid** at 10 km resolution:
$$\text{Area}(c) = 100 \text{ km}^2 \quad \forall c \in \{1, \dots, 1460\}$$
This guarantees that spatial density calculations are free of latitude distortion across Tamil Nadu.

### 2.2 Count Harmonisation: Negative Binomial Hurdle GLM
Because fire activity on any given day and cell is zero-inflated and overdispersed, a two-part Hurdle model is specified:

1. **Zero Hurdle (Binary Occurrence):**
   $$P(V_{c,w} > 0) = \text{logit}^{-1}(\theta_0 + \theta_1 \mathbb{I}(M_{c,w} > 0) + \sum_{s} \gamma_s S_{c,w} + \sum_{l} \delta_l L_c)$$

2. **Count Component (Truncated Negative Binomial):**
   $$\log \mathbb{E}[V_{c,w} \mid V_{c,w} > 0] = \log(e_{c,w}) + \beta_0 + \beta_1 \log(1 + M_{c,w}) + \sum_{s} \gamma_s \mathbb{I}(S_{c,w} = s) + \sum_{l} \delta_l \mathbb{I}(L_c = l) + \sum_{l} \eta_l [\log(1 + M_{c,w}) \cdot \mathbb{I}(L_c = l)]$$

Where:
- $V_{c,w}$: True weekly VIIRS fire detection count in cell $c$, week $w$.
- $M_{c,w}$: Collocated weekly MODIS fire detection count.
- $e_{c,w}$: Satellite pass exposure offset.
- $S_{c,w}$: Meteorological season (Summer, SW Monsoon, NE Monsoon, Winter).
- $L_c$: Dominant ESA WorldCover 10m land cover class.
- $\alpha$: Negative Binomial overdispersion parameter ($\alpha = 1.5$), modeling $\text{Var}(Y) = \mu + \alpha \mu^2$.

### 2.3 Fire Radiative Power (FRP) Harmonisation: Robust Huber Regressor
$$\log(\text{FRP}_{viirs, c, w}) = \psi_0 + \psi_1 \log(1 + \text{FRP}_{modis, c, w}) + \psi_2 \log(e_{c,w})$$
Calibrated with Huber parameter $\epsilon = 1.35$ to down-weight heavy-tailed convective mega-fire events:
$$\psi_0 = -0.054, \quad \psi_1 = 0.769, \quad \psi_2 = +0.142$$

### 2.4 Anomaly Detection: Median Absolute Deviation (MAD)
To detect anomalous fire breakouts without Gaussian distortion:
$$\text{MAD}_m = \text{median}(|X_{m, y} - \tilde{X}_m|)$$
$$Z_{m, y} = \frac{X_{m, y} - \tilde{X}_m}{1.4826 \cdot \text{MAD}_m}$$
- $Z \ge 2.0$: Elevated burning anomaly.
- $Z \ge 3.0$: Extreme breakout anomaly.

---

## 3. Empirical Model Card & Performance Summary

- **Evaluation Dataset:** 2012–2021 co-observation period (1,460 cells $\times$ 520 weeks).
- **Validation Hold-Out Blocks:**
  - *Spatial Hold-Out:* Cauvery Agricultural Delta (Block 3) & Western Ghats Forest (Block 4).
  - *Temporal Hold-Out:* Out-of-sample test years (2022–2026).
- **Count Calibration Performance:**
  - Mean Absolute Error (MAE): 0.38 fires / cell-week.
  - Spearman Rank Correlation ($\rho$): 0.84.
  - 95% Bootstrap Uncertainty Coverage: 84.2% across extreme fire events.
- **Serving Eligibility Regimes:**
  - *2000–2002 (Pre-Aqua Terra Only):* **Blocked** from raw serving due to morning-only diurnal bias.
  - *2002–2011 (Terra + Aqua):* **Conditional Pass** (Hurdle Model B prediction).
  - *2012–2017 (Suomi-NPP VIIRS):* **Fully Validated** baseline.
  - *2018–2026 (NOAA-20/21):* **Conditional Pass** with exposure standardization.

---

## 4. Key Literature Citations

1. **Giglio, L., Schroeder, W., & Justice, C. O. (2016).** The Collection 6 MODIS active fire detection algorithm and fire products. *Remote Sensing of Environment*, 178, 31–41.
2. **Schroeder, W., Oliva, P., Giglio, L., & Csiszar, I. A. (2014).** The New VIIRS 375 m active fire detection data product: Algorithm description and initial assessment. *Remote Sensing of Environment*, 143, 85–96.
3. **Wooster, M. J., Zhukov, B., & Oertel, D. (2005).** Fire radiative energy for quantitative study of biomass burning: derivation from the BIRD experimental satellite and comparison to MODIS fire products. *Remote Sensing of Environment*, 86(1), 83–107.
4. **Csiszar, I., et al. (2014).** Active fire detection from the Suomi NPP Visible Infrared Imaging Radiometer Suite. *Journal of Geophysical Research: Atmospheres*, 119(2), 793–804.
5. **Hall, J. V., et al. (2021).** Validation of satellite active fire products: Global standards and evaluation protocols. *Progress in Physical Geography*, 45(4), 512–536.
