#!/usr/bin/env python3
"""
pipeline/m4_prepare_cell_weeks.py
=================================
NEBULENS — Milestone M4 Data Preparation.

Aggregates daily 10 km grid data into 10 km cell-weeks across the sensor overlap
period (2012-01-20 to 2026-08-31).

Joins:
- Frozen temporal splits (Train: 2012–2018, Val: 2019–2021, Test: 2022–2026)
- Frozen spatial blocks (Blocks 1–5 in EPSG:6933)
- Agro-ecological land-cover classes from M3.5
- Theoretical weekly orbital pass opportunities (exposure offset e_{c,w})
- Agro-climatic season

Generates:
- data/analysis/m4/m4_cell_weeks_dataset.parquet
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GRID_DAILY_DIR = PROJECT_ROOT / "data" / "parquet" / "grid_daily"
GRID_LANDCOVER_PATH = PROJECT_ROOT / "data" / "analysis" / "m3" / "m3_5_grid_landcover.parquet"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4_calibration_config.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_season(month: int) -> str:
    if month in [1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "SW_Monsoon"
    else:
        return "NE_Monsoon"


def get_orbital_exposure(date_str: str) -> float:
    """Theoretical weekly orbital view opportunities per cell."""
    if date_str < "2018-04-01":
        return 14.0  # Suomi-NPP solo (~2 passes/day * 7)
    elif date_str < "2023-03-01":
        return 28.0  # S-NPP + NOAA-20 (~4 passes/day * 7)
    else:
        return 42.0  # S-NPP + NOAA-20 + NOAA-21 (~6 passes/day * 7)


def assign_spatial_block(x: float, y: float) -> str:
    if y < 1220000:
        return "Block_5_South"
    elif y >= 1450000:
        if x < 7550000:
            return "Block_2_NorthWest"
        else:
            return "Block_1_NorthEast"
    else:
        if x < 7550000:
            return "Block_4_CentralWest"
        else:
            return "Block_3_CentralEast_Delta"


def main():
    print("=== Preparing Unified 10 km Cell-Week Dataset for M4 Calibration ===")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # 1. Load Grid Land Cover
    grid_lc = gpd.read_parquet(GRID_LANDCOVER_PATH)
    lc_map = dict(zip(grid_lc["grid_cell_id"], grid_lc["landcover_class"]))

    # Spatial block map
    centroids = grid_lc.geometry.centroid
    block_map = {
        row["grid_cell_id"]: assign_spatial_block(c.x, c.y)
        for (_, row), c in zip(grid_lc.iterrows(), centroids)
    }

    # 2. Iterate over grid_daily partitions (2012–2026)
    dfs = []
    years = list(range(2012, 2027))

    for y in years:
        f = GRID_DAILY_DIR / f"year={y}" / f"grid_daily_{y}.parquet"
        if not f.is_file():
            continue
        print(f"Loading {f.name}...")
        df = pd.read_parquet(
            f,
            columns=[
                "grid_cell_id", "analysis_date_ist",
                "n_det_modis", "n_unique_modis", "frp_sum_modis", "frp_max_modis",
                "n_det_viirs", "n_unique_viirs", "frp_sum_viirs", "frp_max_viirs"
            ]
        )
        # Filter to overlap start (2012-01-20)
        df = df[df["analysis_date_ist"] >= "2012-01-20"]
        if df.empty:
            continue

        df["date"] = pd.to_datetime(df["analysis_date_ist"])
        df["iso_week"] = df["date"].dt.to_period("W").astype(str)

        agg = df.groupby(["grid_cell_id", "iso_week"], as_index=False).agg({
            "analysis_date_ist": ["min", "max"],
            "n_det_modis": "sum",
            "n_unique_modis": "sum",
            "frp_sum_modis": "sum",
            "frp_max_modis": "max",
            "n_det_viirs": "sum",
            "n_unique_viirs": "sum",
            "frp_sum_viirs": "sum",
            "frp_max_viirs": "max",
        })
        agg.columns = [
            "grid_cell_id", "iso_week", "start_date", "end_date",
            "n_det_modis", "n_unique_modis", "frp_sum_modis", "frp_max_modis",
            "n_det_viirs", "n_unique_viirs", "frp_sum_viirs", "frp_max_viirs"
        ]
        dfs.append(agg)

    all_weeks = pd.concat(dfs, ignore_index=True)

    # 3. Add derived features and metadata
    all_weeks["mid_date"] = pd.to_datetime(all_weeks["start_date"]) + pd.Timedelta(days=3)
    all_weeks["mid_date_str"] = all_weeks["mid_date"].dt.strftime("%Y-%m-%d")
    all_weeks["year"] = all_weeks["mid_date"].dt.year
    all_weeks["month"] = all_weeks["mid_date"].dt.month
    all_weeks["season"] = all_weeks["month"].map(get_season)

    # Assign temporal split
    def assign_split(date_str):
        if date_str <= config["temporal_splits"]["train"]["end"]:
            return "train"
        elif date_str <= config["temporal_splits"]["validation"]["end"]:
            return "validation"
        else:
            return "test"

    all_weeks["split"] = all_weeks["mid_date_str"].map(assign_split)

    # Assign spatial block & landcover
    all_weeks["spatial_block"] = all_weeks["grid_cell_id"].map(block_map)
    all_weeks["landcover_class"] = all_weeks["grid_cell_id"].map(lc_map)

    # Assign exposure
    all_weeks["orbital_exposure_week"] = all_weeks["mid_date_str"].map(get_orbital_exposure)
    all_weeks["log_exposure"] = np.log(all_weeks["orbital_exposure_week"])
    all_weeks["log_modis1"] = np.log1p(all_weeks["n_det_modis"])

    out_file = OUTPUT_DIR / "m4_cell_weeks_dataset.parquet"
    all_weeks.to_parquet(out_file, index=False)
    print(f"\n✓ Saved 10 km cell-weeks dataset to {out_file} ({len(all_weeks):,} rows).")
    print("Split breakdown:")
    print(all_weeks["split"].value_counts().to_string())
    print("\nSpatial block breakdown:")
    print(all_weeks["spatial_block"].value_counts().to_string())
    print("=== Cell-Week Dataset Preparation Complete! ===")


if __name__ == "__main__":
    main()
