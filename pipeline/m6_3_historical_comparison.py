#!/usr/bin/env python3
"""
pipeline/m6_3_historical_comparison.py
======================================
NEBULENS — Milestone M6.3: Same-Season Historical Comparison Across Years.

Implements rigorous historical seasonal comparisons per Section 4.6:
- Compares equivalent seasonal windows (calendar months / weeks) across years
- Separates observed VIIRS from counterfactual model predictions
- Controls for satellite orbital exposure inflation (14 vs 28 vs 42 passes/week)
- Excludes pre-July 2002 Terra-only uncalibrated data from automated baselines
- Enforces minimum historical sample requirements (N >= 3)
- Produces standardized machine-readable comparison tables with full provenance.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.m6_1_anomaly_engine import (
    compute_robust_z_score,
    evaluate_series_anomalies,
    DEFAULT_ANOMALY_THRESHOLD,
    REFERENCE_ORBITAL_EXPOSURE
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_monthly_entity_panel(
    harmonised_dir: Path,
    years: List[int],
    grid_meta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds a unified monthly aggregated panel from daily harmonised tables.
    Monthly aggregation avoids treating daily disaggregated allocations as independent daily observations.
    """
    cell_to_dist = dict(zip(grid_meta_df["grid_cell_id"], grid_meta_df["district_name"]))
    records = []
    
    for y in sorted(years):
        fpath = harmonised_dir / f"year={y}" / f"harmonised_daily_{y}.parquet"
        if not fpath.exists():
            continue
        cols = ["grid_cell_id", "analysis_date_ist", "modis_count_observed",
                "viirs_count_observed", "harmonised_count", "harmonised_frp", "h_source"]
        df_y = pd.read_parquet(fpath, columns=cols)
        df_y["district_name"] = df_y["grid_cell_id"].map(cell_to_dist).fillna("Unknown")
        df_y["month"] = df_y["analysis_date_ist"].str[5:7].astype(int)
        df_y["year"] = y
        
        # Exposure pass metadata
        # Pre-2018: 14.0, 2018-2022: 28.0, 2023-2026: 42.0 (for VIIRS era >= 2012)
        if y < 2018:
            exp_val = 14.0
        elif y < 2023:
            exp_val = 28.0
        else:
            exp_val = 42.0
            
        # District monthly aggregate
        dist_agg = df_y.groupby(["district_name", "month"]).agg({
            "modis_count_observed": "sum",
            "viirs_count_observed": "sum",
            "harmonised_count": "sum",
            "harmonised_frp": "sum"
        }).reset_index()
        
        dist_agg["year"] = y
        dist_agg["orbital_exposure"] = exp_val
        dist_agg["h_source"] = "observed_viirs" if y >= 2012 else "modis_model_b"
        records.append(dist_agg)
        
    return pd.concat(records, ignore_index=True)


def run_historical_comparison_pipeline(
    target_years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Runs the same-season comparison pipeline across all districts and months.
    """
    print("--- Running M6.3 Same-Season Historical Comparison Pipeline ---")
    harm_dir = PROJECT_ROOT / "data" / "parquet" / "harmonised_daily"
    avail_years = sorted([int(p.name.split("=")[-1]) for p in harm_dir.glob("year=*")])
    
    # Load grid metadata
    meta_path = PROJECT_ROOT / "data" / "analysis" / "m3" / "m3_5_grid_landcover.parquet"
    grid_meta = pd.read_parquet(meta_path, columns=["grid_cell_id", "district_name"])
    
    print("Building monthly district panel across all years (2000–2026)...")
    panel_df = build_monthly_entity_panel(harm_dir, avail_years, grid_meta)
    
    # State-level aggregation
    state_panel = panel_df.groupby(["year", "month"]).agg({
        "modis_count_observed": "sum",
        "viirs_count_observed": "sum",
        "harmonised_count": "sum",
        "harmonised_frp": "sum",
        "orbital_exposure": "first",
        "h_source": "first"
    }).reset_index()
    state_panel["district_name"] = "Tamil Nadu (Statewide)"
    
    combined_panel = pd.concat([panel_df, state_panel], ignore_index=True)
    
    eval_years = target_years if target_years else [y for y in avail_years if y >= 2012]
    all_results = []
    
    for ty in eval_years:
        print(f"Evaluating same-season comparisons for target year {ty} against historical baseline...")
        res_df = evaluate_series_anomalies(
            df=combined_panel,
            entity_col="district_name",
            time_window_col="month",
            year_col="year",
            value_col="harmonised_count",
            target_year=ty,
            flag_threshold=DEFAULT_ANOMALY_THRESHOLD,
            min_years=3,
            exclude_pre_2002=True,
            exposure_col="orbital_exposure"
        )
        
        # Add metadata & provenance
        res_df["target_h_source"] = "observed_viirs" if ty >= 2012 else "modis_model_b"
        res_df["validation_status"] = "conditional_pass_gate_b" if ty < 2018 else "conditional_pass_opportunity_expansion"
        res_df["aggregation_level"] = np.where(res_df["district_name"].str.contains("Statewide"), "state", "district")
        all_results.append(res_df)
        
    final_comparisons = pd.concat(all_results, ignore_index=True)
    out_file = OUTPUT_DIR / "m6_3_historical_comparison.parquet"
    final_comparisons.to_parquet(out_file, index=False)
    print(f"Saved M6.3 historical comparison ({len(final_comparisons)} rows) -> {out_file}")
    
    return final_comparisons


if __name__ == "__main__":
    run_historical_comparison_pipeline()
