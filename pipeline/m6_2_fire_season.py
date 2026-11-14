#!/usr/bin/env python3
"""
pipeline/m6_2_fire_season.py
============================
NEBULENS — Milestone M6.2: Regional Fire-Season Metrics & Temporal Characterization.

Computes annual fire season metrics at State, District, and Grid-cell levels:
- Active-fire days (number of days with >= 1 detection)
- Core fire season start (10th percentile cumulative detection date)
- Core fire season end (90th percentile cumulative detection date)
- Fire season duration (days between 10th and 90th percentiles)
- Peak activity date and peak magnitude (maximum single-day count and FRP)
- Total annual FRP, mean FRP per active day, and median FRP per active day
- Separation of observed vs model predictions and Gate B restriction provenance.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_fire_season_from_daily(
    daily_series: pd.DataFrame,
    date_col: str = "analysis_date_ist",
    count_col: str = "count",
    frp_col: Optional[str] = "frp",
    active_threshold: float = 1.0,
    min_active_days_for_season: int = 5
) -> Dict[str, Any]:
    """
    Computes fire season metrics from a time series of daily observations.
    
    Parameters:
    - daily_series: DataFrame containing daily records for a single year
    - date_col: Name of date string column (YYYY-MM-DD)
    - count_col: Name of fire count column
    - frp_col: Optional name of FRP column
    - active_threshold: Threshold to qualify as an active fire day (>= 1.0)
    - min_active_days_for_season: Minimum active days required to compute season bounds
    """
    df = daily_series.sort_values(date_col).copy()
    counts = df[count_col].fillna(0.0).values
    dates = df[date_col].values
    
    total_count = float(np.sum(counts))
    active_mask = (counts >= active_threshold)
    n_active_days = int(np.sum(active_mask))
    
    frp_vals = df[frp_col].fillna(0.0).values if (frp_col and frp_col in df.columns) else None
    total_frp = float(np.sum(frp_vals)) if frp_vals is not None else None
    
    # FRP intensity on active days
    if frp_vals is not None and n_active_days > 0:
        active_frp = frp_vals[active_mask]
        mean_active_frp = float(np.mean(active_frp))
        median_active_frp = float(np.median(active_frp))
    else:
        mean_active_frp = 0.0 if n_active_days == 0 and frp_vals is not None else None
        median_active_frp = 0.0 if n_active_days == 0 and frp_vals is not None else None

    # Handle zero fire case
    if total_count < 1e-6 or n_active_days == 0:
        return {
            "total_detections": 0.0,
            "active_fire_days": 0,
            "season_start_date": None,
            "season_end_date": None,
            "season_duration_days": 0,
            "first_active_date": None,
            "last_active_date": None,
            "peak_date": None,
            "peak_count": 0.0,
            "peak_frp": 0.0 if frp_vals is not None else None,
            "total_frp": total_frp,
            "mean_active_frp": mean_active_frp,
            "median_active_frp": median_active_frp,
            "season_status": "no_fire_activity"
        }

    # Peak timing & magnitude
    peak_idx = int(np.argmax(counts))
    peak_date = str(dates[peak_idx])
    peak_count = float(counts[peak_idx])
    peak_frp = float(frp_vals[peak_idx]) if frp_vals is not None else None
    
    # Active day bounding dates
    active_indices = np.where(active_mask)[0]
    first_active_date = str(dates[active_indices[0]])
    last_active_date = str(dates[active_indices[-1]])

    # Sparse activity case: Insufficient active days to define a continuous season
    if n_active_days < min_active_days_for_season:
        return {
            "total_detections": total_count,
            "active_fire_days": n_active_days,
            "season_start_date": first_active_date,
            "season_end_date": last_active_date,
            "season_duration_days": int((pd.to_datetime(last_active_date) - pd.to_datetime(first_active_date)).days) + 1,
            "first_active_date": first_active_date,
            "last_active_date": last_active_date,
            "peak_date": peak_date,
            "peak_count": peak_count,
            "peak_frp": peak_frp,
            "total_frp": total_frp,
            "mean_active_frp": mean_active_frp,
            "median_active_frp": median_active_frp,
            "season_status": "sparse_activity"
        }

    # Core fire season definition (10th and 90th percentiles of cumulative count)
    cum_counts = np.cumsum(counts)
    p10_idx = int(np.searchsorted(cum_counts, 0.10 * total_count))
    p90_idx = int(np.searchsorted(cum_counts, 0.90 * total_count))
    p10_idx = min(p10_idx, len(dates) - 1)
    p90_idx = min(p90_idx, len(dates) - 1)
    
    season_start = str(dates[p10_idx])
    season_end = str(dates[p90_idx])
    duration = int((pd.to_datetime(season_end) - pd.to_datetime(season_start)).days) + 1
    
    return {
        "total_detections": total_count,
        "active_fire_days": n_active_days,
        "season_start_date": season_start,
        "season_end_date": season_end,
        "season_duration_days": max(1, duration),
        "first_active_date": first_active_date,
        "last_active_date": last_active_date,
        "peak_date": peak_date,
        "peak_count": peak_count,
        "peak_frp": peak_frp,
        "total_frp": total_frp,
        "mean_active_frp": mean_active_frp,
        "median_active_frp": median_active_frp,
        "season_status": "active_season"
    }


def determine_provenance_and_validation(
    year: int,
    stream_type: str  # 'observed_modis', 'observed_viirs', 'harmonised'
) -> Tuple[str, str]:
    """
    Assigns provenance and validation readiness label per M5 Gate B decisions.
    """
    if stream_type == "observed_modis":
        if year <= 2002:
            return "modis_terra_aqua_transition", "conditional_pass_diurnal_caveat"
        return "modis_observed_c61", "fully_validated"
    elif stream_type == "observed_viirs":
        if year < 2012:
            return "not_applicable", "sensor_not_in_orbit"
        elif year < 2018:
            return "viirs_snpp_observed", "fully_validated"
        elif year < 2023:
            return "viirs_snpp_noaa20_observed", "conditional_pass_opportunity_expansion"
        else:
            return "viirs_snpp_noaa20_noaa21_observed", "conditional_pass_opportunity_expansion"
    else:  # harmonised
        if year < 2002:
            return "modis_model_b_counterfactual_terra_only", "not_validated_terra_only_pre_2002"
        elif year == 2002:
            return "modis_model_b_counterfactual_hybrid", "conditional_pass_transition_year"
        elif year < 2012:
            return "modis_model_b_counterfactual_terra_aqua", "conditional_pass_gate_b"
        else:
            return "viirs_observed_harmonised_stream", "conditional_pass_gate_b"


def run_fire_season_pipeline(years: Optional[List[int]] = None) -> Dict[str, pd.DataFrame]:
    """
    Runs the complete regional fire season metrics computation pipeline
    across all available years (2000–2026).
    """
    print("--- Running M6.2 Fire Season Metrics Pipeline ---")
    
    # Load cell to district mapping
    grid_meta_path = PROJECT_ROOT / "data" / "analysis" / "m3" / "m3_5_grid_landcover.parquet"
    if not grid_meta_path.exists():
        raise FileNotFoundError(f"Missing grid metadata at {grid_meta_path}")
    grid_meta = pd.read_parquet(grid_meta_path, columns=["grid_cell_id", "district_name"])
    cell_to_dist = dict(zip(grid_meta["grid_cell_id"], grid_meta["district_name"]))
    
    harm_dir = PROJECT_ROOT / "data" / "parquet" / "harmonised_daily"
    avail_years = sorted([int(p.name.split("=")[-1]) for p in harm_dir.glob("year=*")])
    if years:
        target_years = [y for y in years if y in avail_years]
    else:
        target_years = avail_years
        
    print(f"Target years: {len(target_years)} years ({target_years[0]} to {target_years[-1]})")
    
    state_records = []
    district_records = []
    
    for y in target_years:
        fpath = harm_dir / f"year={y}" / f"harmonised_daily_{y}.parquet"
        cols = ["grid_cell_id", "analysis_date_ist", "modis_count_observed", "modis_frp_observed",
                "viirs_count_observed", "viirs_frp_observed", "harmonised_count", "harmonised_frp"]
        df_y = pd.read_parquet(fpath, columns=cols)
        df_y["district_name"] = df_y["grid_cell_id"].map(cell_to_dist).fillna("Unknown")
        
        # 1. State-level metrics for 3 streams: Observed MODIS, Observed VIIRS (post-2012), Harmonised
        # State Daily Aggregation
        state_daily = df_y.groupby("analysis_date_ist").agg({
            "modis_count_observed": "sum",
            "modis_frp_observed": "sum",
            "viirs_count_observed": "sum",
            "viirs_frp_observed": "sum",
            "harmonised_count": "sum",
            "harmonised_frp": "sum"
        }).reset_index()
        
        # Stream 1: Observed MODIS
        m_res = compute_fire_season_from_daily(state_daily, count_col="modis_count_observed", frp_col="modis_frp_observed")
        prov_m, val_m = determine_provenance_and_validation(y, "observed_modis")
        state_records.append({
            "year": y, "entity_type": "state", "entity_name": "Tamil Nadu", "stream": "observed_modis",
            "provenance": prov_m, "validation_status": val_m, **m_res
        })
        
        # Stream 2: Observed VIIRS (only >= 2012)
        if y >= 2012:
            v_res = compute_fire_season_from_daily(state_daily, count_col="viirs_count_observed", frp_col="viirs_frp_observed")
            prov_v, val_v = determine_provenance_and_validation(y, "observed_viirs")
            state_records.append({
                "year": y, "entity_type": "state", "entity_name": "Tamil Nadu", "stream": "observed_viirs",
                "provenance": prov_v, "validation_status": val_v, **v_res
            })
            
        # Stream 3: Harmonised
        h_res = compute_fire_season_from_daily(state_daily, count_col="harmonised_count", frp_col="harmonised_frp")
        prov_h, val_h = determine_provenance_and_validation(y, "harmonised")
        state_records.append({
            "year": y, "entity_type": "state", "entity_name": "Tamil Nadu", "stream": "harmonised",
            "provenance": prov_h, "validation_status": val_h, **h_res
        })
        
        # 2. District-level metrics
        dist_daily = df_y.groupby(["district_name", "analysis_date_ist"]).agg({
            "modis_count_observed": "sum",
            "modis_frp_observed": "sum",
            "viirs_count_observed": "sum",
            "viirs_frp_observed": "sum",
            "harmonised_count": "sum",
            "harmonised_frp": "sum"
        }).reset_index()
        
        for dname, d_grp in dist_daily.groupby("district_name"):
            # District Harmonised stream
            dh_res = compute_fire_season_from_daily(d_grp, count_col="harmonised_count", frp_col="harmonised_frp")
            district_records.append({
                "year": y, "entity_type": "district", "entity_name": dname, "stream": "harmonised",
                "provenance": prov_h, "validation_status": val_h, **dh_res
            })
            # District Observed VIIRS (>= 2012)
            if y >= 2012:
                dv_res = compute_fire_season_from_daily(d_grp, count_col="viirs_count_observed", frp_col="viirs_frp_observed")
                district_records.append({
                    "year": y, "entity_type": "district", "entity_name": dname, "stream": "observed_viirs",
                    "provenance": prov_v, "validation_status": val_v, **dv_res
                })
            # District Observed MODIS
            dm_res = compute_fire_season_from_daily(d_grp, count_col="modis_count_observed", frp_col="modis_frp_observed")
            district_records.append({
                "year": y, "entity_type": "district", "entity_name": dname, "stream": "observed_modis",
                "provenance": prov_m, "validation_status": val_m, **dm_res
            })

    df_state = pd.DataFrame(state_records)
    df_district = pd.DataFrame(district_records)
    
    # Save outputs
    out_state_file = OUTPUT_DIR / "m6_2_state_season_metrics.parquet"
    out_dist_file = OUTPUT_DIR / "m6_2_district_season_metrics.parquet"
    
    df_state.to_parquet(out_state_file, index=False)
    df_district.to_parquet(out_dist_file, index=False)
    
    print(f"Saved state metrics ({len(df_state)} rows) -> {out_state_file}")
    print(f"Saved district metrics ({len(df_district)} rows) -> {out_dist_file}")
    
    return {"state": df_state, "district": df_district}


if __name__ == "__main__":
    run_fire_season_pipeline()
