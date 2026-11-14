#!/usr/bin/env python3
"""
pipeline/m4_8_generate_harmonised_daily.py
==========================================
NEBULENS — Milestone M4.8: Harmonised Daily Dataset Generation (2000–2026).

Produces the definitive, publication-ready harmonised daily active fire dataset
across Tamil Nadu (1,460 cells at 10 km resolution, 2000–2026):
1. Preserves the exact five required fields from methodology:
   - observed_modis (modis_count_observed, modis_frp_observed)
   - observed_viirs (viirs_count_observed, viirs_frp_observed)
   - expected_sensor_specific_value (expected_viirs_count, expected_viirs_frp)
   - harmonised_estimate (harmonised_count, harmonised_frp)
   - uncertainty (count_lower_80, count_upper_80, count_lower_95, count_upper_95,
                  frp_lower_80, frp_upper_80, frp_lower_95, frp_upper_95)
   - Plus required metadata: grid_cell_id, analysis_date_ist, h_source, model_version

2. Implementation rules:
   - Uses promoted Step B Negative Binomial GLM and Model B FRP model
   - Rigorous overpass-weighted temporal downscaling from cell-weeks to daily:
     w_d = n_det_modis + 0.1 * (n_terra + n_aqua)
     p_d = w_d / sum(w_d) if sum(w_d) > 0 else (1.0 / 7.0)
     Strictly conserves weekly totals: sum_d(V_d) == V_w
   - Preserves source provenance:
     VIIRS era: h_source = 'observed_viirs' (direct observation)
     Pre-VIIRS era: h_source = 'calibrated_from_modis' (calibrated harmonisation)
   - Partitioned by year: data/parquet/harmonised_daily/year={year}/harmonised_daily_{year}.parquet

Generates:
- Partitioned dataset: data/parquet/harmonised_daily/year=YYYY/harmonised_daily_YYYY.parquet (2000–2026)
- Summary metadata: data/analysis/m4/m4_8_harmonised_summary.json
- Publication figure: docs/figures/m4_8_harmonised_timeseries.png
"""

import sys
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GRID_DAILY_DIR = PROJECT_ROOT / "data" / "parquet" / "grid_daily"
OUTPUT_PARQUET_DIR = PROJECT_ROOT / "data" / "parquet" / "harmonised_daily"
OUTPUT_ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis" / "m4"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
LANDCOVER_PATH = PROJECT_ROOT / "data" / "analysis" / "m3" / "m3_5_grid_landcover.parquet"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4_calibration_config.json"
BOOTSTRAP_MODELS_PATH = OUTPUT_ANALYSIS_DIR / "m4_7_bootstrap_models.json"
NB_COEFF_PATH = OUTPUT_ANALYSIS_DIR / "m4_3_nb_coefficients.json"
FRP_CELL_PATH = OUTPUT_ANALYSIS_DIR / "m4_5_frp_cell_model.json"

OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "m4_step_b_nb_glm_v1.0.0"


def get_orbital_exposure_daily(date_val: pd.Timestamp) -> float:
    """Return theoretical daily overpass exposure based on satellite constellation epoch."""
    d_str = date_val.strftime("%Y-%m-%d")
    if d_str < "2002-05-04":
        return 1.0  # Terra only
    elif d_str < "2012-01-20":
        return 2.0  # Terra + Aqua
    elif d_str < "2017-11-29":
        return 3.0  # Terra + Aqua + Suomi-NPP
    elif d_str < "2023-01-26":
        return 4.0  # Terra + Aqua + SNPP + NOAA-20
    else:
        return 5.0  # Terra + Aqua + SNPP + NOAA-20 + NOAA-21


def get_season(month: int) -> str:
    """Standard meteorological season definition for Tamil Nadu."""
    if month in (1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "SW_Monsoon"
    else:
        return "NE_Monsoon"


def main():
    print("=== Milestone M4.8: Harmonised Daily Dataset Generation (2000–2026) ===")
    t0 = time.time()

    # 1. Load models and parameters
    print("Loading model parameters and bootstrap ensemble...")
    with open(NB_COEFF_PATH) as f:
        nb_coeffs = json.load(f)["coefficients"]

    with open(FRP_CELL_PATH) as f:
        frp_coeffs = json.load(f)["coefficients"]

    with open(BOOTSTRAP_MODELS_PATH) as f:
        boot_dict = json.load(f)
    boot_count_params = np.array(boot_dict["count_params_all"])  # (100, 13)
    boot_frp_params = np.array(boot_dict["frp_params_all"])      # (100, 3)

    # 2. Load landcover classification
    print(f"Loading grid landcover from {LANDCOVER_PATH}...")
    lc_df = pd.read_parquet(LANDCOVER_PATH)[["grid_cell_id", "landcover_class"]].copy()
    lc_df["lc_clean"] = lc_df["landcover_class"].str.replace(r"[^a-zA-Z0-9]", "_", regex=True)
    cell_lc_map = dict(zip(lc_df["grid_cell_id"], lc_df["lc_clean"]))

    years = sorted([int(p.name.split("=")[1]) for p in GRID_DAILY_DIR.glob("year=*")])
    print(f"Discovered {len(years)} years of daily gridded data: {years[0]} to {years[-1]}")

    yearly_summaries = []
    total_records = 0
    total_modis_count = 0
    total_viirs_count = 0
    total_harmonised_count = 0.0

    # Process year by year for strict memory efficiency
    for yr in years:
        yr_t0 = time.time()
        file_path = GRID_DAILY_DIR / f"year={yr}" / f"grid_daily_{yr}.parquet"
        if not file_path.exists():
            continue

        df = pd.read_parquet(file_path)
        df["date"] = pd.to_datetime(df["analysis_date_ist"])
        df["month"] = df["date"].dt.month
        df["season"] = df["month"].apply(get_season)
        df["lc_clean"] = df["grid_cell_id"].map(cell_lc_map).fillna("Cropland___Mosaic__Inland___Dryland_")

        # ISO calendar week for temporal grouping
        df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
        df["exposure_daily"] = df["date"].apply(get_orbital_exposure_daily)

        # Weekly aggregation per cell
        weekly_group = df.groupby(["grid_cell_id", "iso_week"], as_index=False).agg(
            n_det_modis_w=("n_det_modis", "sum"),
            frp_sum_modis_w=("frp_sum_modis", "sum"),
            exposure_w=("exposure_daily", "sum"),
            season=("season", "first"),
            lc_clean=("lc_clean", "first")
        )

        # Predict weekly count rate (GLM Step B)
        # 13 design matrix features
        X_cols = [
            np.ones(len(weekly_group), dtype=np.float64),
            (weekly_group["season"] == "NE_Monsoon").astype(np.float64).values,
            (weekly_group["season"] == "SW_Monsoon").astype(np.float64).values,
            (weekly_group["season"] == "Winter").astype(np.float64).values,
            (weekly_group["lc_clean"] == "Cropland__Intensive___Delta_").astype(np.float64).values,
            (weekly_group["lc_clean"] == "Forest___Woodland").astype(np.float64).values,
            (weekly_group["lc_clean"] == "Shrubland___Savanna__Southern_Semi_Arid_").astype(np.float64).values,
            (weekly_group["lc_clean"] == "Urban___Developed_Corridor").astype(np.float64).values,
            np.log1p(weekly_group["n_det_modis_w"].values),
            (np.log1p(weekly_group["n_det_modis_w"].values) * (weekly_group["lc_clean"] == "Cropland___Mosaic__Inland___Dryland_")).astype(np.float64),
            (np.log1p(weekly_group["n_det_modis_w"].values) * (weekly_group["lc_clean"] == "Forest___Woodland")).astype(np.float64),
            (np.log1p(weekly_group["n_det_modis_w"].values) * (weekly_group["lc_clean"] == "Shrubland___Savanna__Southern_Semi_Arid_")).astype(np.float64),
            (np.log1p(weekly_group["n_det_modis_w"].values) * (weekly_group["lc_clean"] == "Urban___Developed_Corridor")).astype(np.float64)
        ]
        X_w = np.column_stack(X_cols)
        off_w = np.log(weekly_group["exposure_w"].values)

        # Vectorized prediction across bootstrap replicates for weekly bounds
        eta_w_matrix = off_w[:, None] + np.dot(X_w, boot_count_params.T)
        mu_w_matrix = np.exp(np.clip(eta_w_matrix, -20.0, 15.0))
        mu_w_median = np.median(mu_w_matrix, axis=1)

        # Weekly Negative Binomial interval bounds computed on ensemble median rate
        alpha = 1.5
        n_param = 1.0 / alpha
        p_param_median = n_param / (n_param + mu_w_median)

        lower_80_w = stats.nbinom.ppf(0.10, n_param, p_param_median)
        upper_80_w = stats.nbinom.ppf(0.90, n_param, p_param_median)
        lower_95_w = stats.nbinom.ppf(0.025, n_param, p_param_median)
        upper_95_w = stats.nbinom.ppf(0.975, n_param, p_param_median)

        # Predicted weekly FRP (Model B)
        X_frp_w = np.column_stack([
            np.ones(len(weekly_group)),
            np.log1p(weekly_group["frp_sum_modis_w"].values),
            off_w
        ])
        frp_pred_w = np.exp(np.dot(X_frp_w, boot_frp_params.T))  # (N_w, 100)
        frp_w_median = np.where(weekly_group["n_det_modis_w"].values > 0, np.median(frp_pred_w, axis=1), 0.0)
        frp_lower_80_w = np.where(weekly_group["n_det_modis_w"].values > 0, np.percentile(frp_pred_w, 10.0, axis=1), 0.0)
        frp_upper_80_w = np.where(weekly_group["n_det_modis_w"].values > 0, np.percentile(frp_pred_w, 90.0, axis=1), 0.0)
        frp_lower_95_w = np.where(weekly_group["n_det_modis_w"].values > 0, np.percentile(frp_pred_w, 2.5, axis=1), 0.0)
        frp_upper_95_w = np.where(weekly_group["n_det_modis_w"].values > 0, np.percentile(frp_pred_w, 97.5, axis=1), 0.0)

        # Assign predictions back to weekly_group
        weekly_group["pred_count_w"] = mu_w_median
        weekly_group["c_lower_80_w"] = np.nan_to_num(lower_80_w, nan=0.0)
        weekly_group["c_upper_80_w"] = np.nan_to_num(upper_80_w, nan=0.0)
        weekly_group["c_lower_95_w"] = np.nan_to_num(lower_95_w, nan=0.0)
        weekly_group["c_upper_95_w"] = np.nan_to_num(upper_95_w, nan=0.0)

        weekly_group["pred_frp_w"] = frp_w_median
        weekly_group["f_lower_80_w"] = np.nan_to_num(frp_lower_80_w, nan=0.0)
        weekly_group["f_upper_80_w"] = np.nan_to_num(frp_upper_80_w, nan=0.0)
        weekly_group["f_lower_95_w"] = np.nan_to_num(frp_lower_95_w, nan=0.0)
        weekly_group["f_upper_95_w"] = np.nan_to_num(frp_upper_95_w, nan=0.0)

        # Merge weekly targets onto daily records
        df = df.merge(
            weekly_group[[
                "grid_cell_id", "iso_week",
                "pred_count_w", "c_lower_80_w", "c_upper_80_w", "c_lower_95_w", "c_upper_95_w",
                "pred_frp_w", "f_lower_80_w", "f_upper_80_w", "f_lower_95_w", "f_upper_95_w"
            ]],
            on=["grid_cell_id", "iso_week"],
            how="left"
        )

        # Compute daily downscaling weight: w_d = n_det_modis + 0.1 * (n_terra + n_aqua)
        df["w_d"] = df["n_det_modis"] + 0.1 * (df["n_terra"] + df["n_aqua"])
        w_sum = df.groupby(["grid_cell_id", "iso_week"])["w_d"].transform("sum")
        w_cnt = df.groupby(["grid_cell_id", "iso_week"])["w_d"].transform("count")

        # If w_sum > 0, allocate proportional to w_d; otherwise allocate uniformly 1/count
        df["p_d"] = np.where(w_sum > 0, df["w_d"] / w_sum, 1.0 / np.maximum(w_cnt, 1))

        # Expected sensor-specific values
        expected_viirs_count = df["p_d"] * df["pred_count_w"]
        expected_viirs_frp = df["p_d"] * df["pred_frp_w"]

        # Temporal downscaled uncertainty intervals
        c_lower_80 = df["p_d"] * df["c_lower_80_w"]
        c_upper_80 = df["p_d"] * df["c_upper_80_w"]
        c_lower_95 = df["p_d"] * df["c_lower_95_w"]
        c_upper_95 = df["p_d"] * df["c_upper_95_w"]

        f_lower_80 = df["p_d"] * df["f_lower_80_w"]
        f_upper_80 = df["p_d"] * df["f_upper_80_w"]
        f_lower_95 = df["p_d"] * df["f_lower_95_w"]
        f_upper_95 = df["p_d"] * df["f_upper_95_w"]

        # Harmonisation decision based on era and sensor presence
        # VIIRS era begins 2012-01-20
        is_viirs_era = df["analysis_date_ist"] >= "2012-01-20"

        harmonised_count = np.where(is_viirs_era, df["n_det_viirs"].values.astype(float), expected_viirs_count)
        harmonised_frp = np.where(is_viirs_era, df["frp_sum_viirs"].values.astype(float), expected_viirs_frp)
        h_source = np.where(is_viirs_era, "observed_viirs", "calibrated_from_modis")

        # Build clean harmonised daily DataFrame with exact 5 required fields
        harm_df = pd.DataFrame({
            "grid_cell_id": df["grid_cell_id"],
            "analysis_date_ist": df["analysis_date_ist"],

            # 1. Observed MODIS
            "modis_count_observed": df["n_det_modis"].astype(int),
            "modis_frp_observed": np.round(df["frp_sum_modis"], 2),
            "observed_modis": df["n_det_modis"].astype(int),

            # 2. Observed VIIRS
            "viirs_count_observed": df["n_det_viirs"].astype(int),
            "viirs_frp_observed": np.round(df["frp_sum_viirs"], 2),
            "observed_viirs": df["n_det_viirs"].astype(int),

            # 3. Expected sensor-specific value
            "expected_viirs_count": np.round(expected_viirs_count, 3),
            "expected_viirs_frp": np.round(expected_viirs_frp, 2),
            "expected_sensor_specific_value": np.round(expected_viirs_count, 3),

            # 4. Harmonised estimate
            "harmonised_count": np.round(harmonised_count, 3),
            "harmonised_frp": np.round(harmonised_frp, 2),
            "harmonised_estimate": np.round(harmonised_count, 3),

            # 5. Uncertainty
            "count_lower_80": np.round(c_lower_80, 3),
            "count_upper_80": np.round(c_upper_80, 3),
            "count_lower_95": np.round(c_lower_95, 3),
            "count_upper_95": np.round(c_upper_95, 3),
            "frp_lower_80": np.round(f_lower_80, 2),
            "frp_upper_80": np.round(f_upper_80, 2),
            "frp_lower_95": np.round(f_lower_95, 2),
            "frp_upper_95": np.round(f_upper_95, 2),
            "uncertainty": np.round((c_upper_95 - c_lower_95) / 2.0, 3),

            # Provenance metadata
            "h_source": h_source,
            "model_version": MODEL_VERSION
        })

        # Save partitioned Parquet
        yr_out_dir = OUTPUT_PARQUET_DIR / f"year={yr}"
        yr_out_dir.mkdir(parents=True, exist_ok=True)
        yr_out_path = yr_out_dir / f"harmonised_daily_{yr}.parquet"
        harm_df.to_parquet(yr_out_path, index=False)

        # Yearly stats
        m_cnt = int(harm_df["modis_count_observed"].sum())
        v_cnt = int(harm_df["viirs_count_observed"].sum())
        h_cnt = float(harm_df["harmonised_count"].sum())
        n_rows = len(harm_df)

        total_records += n_rows
        total_modis_count += m_cnt
        total_viirs_count += v_cnt
        total_harmonised_count += h_cnt

        yearly_summaries.append({
            "year": yr,
            "records": n_rows,
            "modis_detections": m_cnt,
            "viirs_detections": v_cnt,
            "harmonised_detections": round(h_cnt, 1),
            "h_source": "observed_viirs" if yr >= 2012 else "calibrated_from_modis"
        })

        yr_elapsed = time.time() - yr_t0
        print(f"  Year {yr}: {n_rows:,} rows processed in {yr_elapsed:.2f}s | MODIS={m_cnt:,} | VIIRS={v_cnt:,} | Harmonised={h_cnt:,.1f}")

        del df, harm_df, weekly_group
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nAll {len(years)} years generated in {elapsed:.1f}s!")
    print(f"Total dataset records: {total_records:,}")
    print(f"Total MODIS Detections: {total_modis_count:,}")
    print(f"Total VIIRS Detections: {total_viirs_count:,}")
    print(f"Total Harmonised Count: {total_harmonised_count:,.1f}")

    # Save summary metadata
    summary_meta = {
        "dataset_name": "harmonised_daily",
        "description": "Harmonised daily active fire dataset for Tamil Nadu (2000–2026)",
        "spatial_resolution": "10 km EPSG:6933",
        "n_grid_cells": 1460,
        "temporal_range": f"{years[0]}-01-01 to {years[-1]}-12-31",
        "total_cell_days": total_records,
        "model_version": MODEL_VERSION,
        "count_model": "Negative Binomial GLM (Step B) with exposure offset and landcover interactions",
        "frp_model": "Huber Robust Log-Linear (Model B)",
        "uncertainty_method": "Year-Block Bootstrap (B=100 replicates, 80% and 95% intervals)",
        "downscaling_method": "Overpass-weighted daily allocation (strictly conserving weekly totals)",
        "required_fields_status": {
            "observed_modis": "Present (modis_count_observed, modis_frp_observed)",
            "observed_viirs": "Present (viirs_count_observed, viirs_frp_observed)",
            "expected_sensor_specific_value": "Present (expected_viirs_count, expected_viirs_frp)",
            "harmonised_estimate": "Present (harmonised_count, harmonised_frp)",
            "uncertainty": "Present (80% and 95% lower/upper bounds for count and FRP)"
        },
        "totals": {
            "modis_detections": total_modis_count,
            "viirs_detections": total_viirs_count,
            "harmonised_detections": round(total_harmonised_count, 1)
        },
        "yearly_breakdown": yearly_summaries
    }

    summary_path = OUTPUT_ANALYSIS_DIR / "m4_8_harmonised_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_meta, f, indent=2)
    print(f"Saved harmonisation summary to {summary_path}")

    # Generate Publication Timeseries Figure
    print("Generating publication figure: docs/figures/m4_8_harmonised_timeseries.png...")
    sum_df = pd.DataFrame(yearly_summaries)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Top Panel: 26-year Harmonised Time Series
    ax1 = axes[0]
    ax1.plot(sum_df["year"], sum_df["modis_detections"], marker="o", color="#e41a1c", linewidth=1.5, label="Raw MODIS (Terra + Aqua)")
    ax1.plot(sum_df[sum_df["year"] >= 2012]["year"], sum_df[sum_df["year"] >= 2012]["viirs_detections"], marker="s", color="#377eb8", linewidth=1.5, label="Raw VIIRS (SNPP + NOAA-20 + NOAA-21)")
    ax1.plot(sum_df["year"], sum_df["harmonised_detections"], marker="^", color="#4daf4a", linewidth=2.0, linestyle="--", label="Harmonised Estimate (NEBULENS M4)")

    # Transition line at 2012
    ax1.axvline(2012, color="gray", linestyle=":", linewidth=1.5)
    ax1.text(2012.2, ax1.get_ylim()[1] * 0.9, "VIIRS Era Begins (2012)", fontsize=10, fontstyle="italic", color="gray")

    ax1.set_title("A. 26-Year Active Fire Trajectory in Tamil Nadu (2000–2026): Raw vs Harmonised", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Annual Fire Detection Total", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=10)

    # Bottom Panel: Ratio of Harmonised to Raw MODIS
    ax2 = axes[1]
    ratio = sum_df["harmonised_detections"] / np.maximum(sum_df["modis_detections"], 1)
    ax2.bar(sum_df["year"], ratio, color="#984ea3", alpha=0.75, edgecolor="black")
    ax2.axhline(1.0, color="black", linestyle="-", linewidth=1.0)
    ax2.axhline(np.mean(ratio), color="red", linestyle="--", linewidth=1.2, label=f"Mean Calibration Ratio ({np.mean(ratio):.2f}x)")
    ax2.set_title("B. Annual Effective Harmonisation Scaling Factor (Harmonised / MODIS)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Scaling Ratio", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_8_harmonised_timeseries.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved harmonisation timeseries figure to {fig_path}")

    print("=== M4.8 Complete! ===")


if __name__ == "__main__":
    main()
