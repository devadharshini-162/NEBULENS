#!/usr/bin/env python3
"""
pipeline/aggregate_grid_daily.py
================================
NEBULENS — Milestone M2.8: Daily Grid Aggregation Engine.

Aggregates standardised observations to the 10 km EPSG:6933 analytical grid
for every calendar day (Indian Standard Time, Asia/Kolkata) across the full
study period (2000-11-01 to 2026-08-31).

Implements Section 4.2 of the Methodology Specification:
- Every cell-day receives a complete set of measures for both sensor families (MODIS and VIIRS):
    1. n_det (raw detections in the cell that day)
    2. n_unique (detections after collapsing each duplicate_group)
    3. frp_sum (total FRP from detections reporting FRP; 0 on zero days)
    4. frp_mean, frp_max (FRP extremes; NaN on zero days where FRP is undefined)
    5. active_fire_day (1 if total n_det > 0, else 0)
    6. Day and night split (n_det_day, n_det_night, frp_sum_day, frp_sum_night)
    7. Platform split:
         MODIS: n_terra, n_aqua
         VIIRS: n_snpp, n_noaa20, n_noaa21
    8. Scan-angle bins (proxy for optical distortion and pixel enlargement):
         MODIS: scan_bin_near (<=1.1), scan_bin_mid (1.1-1.5), scan_bin_far (>1.5)
         VIIRS: scan_bin_near (<=0.4), scan_bin_mid (0.4-0.6), scan_bin_far (>0.6)
    9. overpass_times: sorted list/string of distinct acquisition times seen in the cell that day
- Zero rows are retained for every cell-day (1,460 cells × ~9,435 calendar days = ~13.78M cell-days).
- Stored partitioned by year under:
    data/parquet/grid_daily/year={year}/grid_daily_{year}.parquet
"""

import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PARQUET_OBS_DIR = PROJECT_ROOT / "data" / "parquet" / "observations"
GRID_10KM_FILE = PROJECT_ROOT / "data" / "boundaries" / "grids" / "grid_10km.parquet"
GRID_DAILY_DIR = PROJECT_ROOT / "data" / "parquet" / "grid_daily"

START_DATE_GLOBAL = "2000-11-01"
END_DATE_GLOBAL = "2026-08-31"


def get_grid_cell_ids() -> list[str]:
    """Load sorted 10km grid cell IDs intersecting Tamil Nadu."""
    g10 = gpd.read_parquet(GRID_10KM_FILE)
    return sorted(g10["grid_cell_id"].unique())


def get_scan_bin_modis(scan_val: float) -> str:
    """MODIS scan angle bins: near (<=1.1 km), mid (1.1-1.5 km), far (>1.5 km)."""
    if scan_val <= 1.1:
        return "near"
    elif scan_val <= 1.5:
        return "mid"
    else:
        return "far"


def get_scan_bin_viirs(scan_val: float) -> str:
    """VIIRS 375m scan angle bins: near (<=0.4 km), mid (0.4-0.6 km), far (>0.6 km)."""
    if scan_val <= 0.4:
        return "near"
    elif scan_val <= 0.6:
        return "mid"
    else:
        return "far"


def aggregate_year(year: int, cell_ids: list[str]) -> Path:
    """Aggregate active fires for one calendar year and emit complete zero-filled Parquet."""
    out_dir = GRID_DAILY_DIR / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"grid_daily_{year}.parquet"

    # Determine date range for this year
    y_start = max(f"{year}-01-01", START_DATE_GLOBAL)
    y_end = min(f"{year}-12-31", END_DATE_GLOBAL)
    dates = pd.date_range(y_start, y_end).strftime("%Y-%m-%d").tolist()

    # Load in-TN observations whose analysis_date_ist falls in this year
    # Note: observations are partitioned by raw acquisition year, but night passes on Dec 31 UTC
    # (after 18:30 UTC) cross into Jan 01 IST of the next year. We include adjacent years to capture boundary crossings.
    candidate_years = [year - 1, year, year + 1]
    dfs = []
    for cy in candidate_years:
        for f in PARQUET_OBS_DIR.glob(f"source=*/year={cy}/*.parquet"):
            df = pd.read_parquet(
                f,
                columns=[
                    "source", "satellite", "grid_cell_id", "analysis_date_ist",
                    "in_tn", "frp", "daynight", "scan", "acq_time", "duplicate_group"
                ]
            )
            df_tn = df[df["in_tn"] & (df["analysis_date_ist"].between(y_start, y_end))]
            if not df_tn.empty:
                dfs.append(df_tn)

    combined_obs = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Create full base cartesian index (cell_id × date)
    cartesian_cells = np.repeat(cell_ids, len(dates))
    cartesian_dates = np.tile(dates, len(cell_ids))
    base_df = pd.DataFrame({
        "grid_cell_id": cartesian_cells,
        "analysis_date_ist": cartesian_dates,
    })

    if combined_obs.empty:
        # Zero year (e.g. early 2000 prior to observations)
        _populate_zero_table(base_df)
        base_df.to_parquet(out_file, index=False, compression="zstd")
        return out_file

    # Classify sensor families and scan bins
    combined_obs["is_modis"] = combined_obs["source"].str.contains("MODIS")
    
    # MODIS scan bins
    m_mask = combined_obs["is_modis"]
    combined_obs["scan_bin"] = "mid"
    combined_obs.loc[m_mask & (combined_obs["scan"] <= 1.1), "scan_bin"] = "near"
    combined_obs.loc[m_mask & (combined_obs["scan"] > 1.5), "scan_bin"] = "far"
    
    # VIIRS scan bins
    v_mask = ~combined_obs["is_modis"]
    combined_obs.loc[v_mask & (combined_obs["scan"] <= 0.4), "scan_bin"] = "near"
    combined_obs.loc[v_mask & (combined_obs["scan"] > 0.6), "scan_bin"] = "far"

    # Aggregations for MODIS
    modis_obs = combined_obs[m_mask]
    modis_agg = _compute_sensor_family_aggregations(modis_obs, "modis")

    # Aggregations for VIIRS
    viirs_obs = combined_obs[v_mask]
    viirs_agg = _compute_sensor_family_aggregations(viirs_obs, "viirs")

    # Merge aggregated measures with base cartesian table
    res_df = base_df.merge(modis_agg, on=["grid_cell_id", "analysis_date_ist"], how="left")
    res_df = res_df.merge(viirs_agg, on=["grid_cell_id", "analysis_date_ist"], how="left")

    # Fill zeros for counts/sums
    _fill_missing_and_types(res_df)

    # Combined active_fire_day indicator
    res_df["active_fire_day"] = ((res_df["n_det_modis"] > 0) | (res_df["n_det_viirs"] > 0)).astype(np.int8)

    # Reorder columns logically
    cols = [
        "grid_cell_id", "analysis_date_ist", "active_fire_day",
        # MODIS measures
        "n_det_modis", "n_unique_modis", "frp_sum_modis", "frp_mean_modis", "frp_max_modis",
        "n_det_day_modis", "n_det_night_modis", "frp_sum_day_modis", "frp_sum_night_modis",
        "n_terra", "n_aqua",
        "scan_near_modis", "scan_mid_modis", "scan_far_modis",
        "overpass_times_modis",
        # VIIRS measures
        "n_det_viirs", "n_unique_viirs", "frp_sum_viirs", "frp_mean_viirs", "frp_max_viirs",
        "n_det_day_viirs", "n_det_night_viirs", "frp_sum_day_viirs", "frp_sum_night_viirs",
        "n_snpp", "n_noaa20", "n_noaa21",
        "scan_near_viirs", "scan_mid_viirs", "scan_far_viirs",
        "overpass_times_viirs",
    ]
    res_df = res_df[cols]

    res_df.to_parquet(out_file, index=False, compression="zstd")
    return out_file


def _compute_sensor_family_aggregations(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute all Section 4.2 measures for a sensor family."""
    expected_cols = [
        "grid_cell_id", "analysis_date_ist",
        f"n_det_{prefix}", f"n_unique_{prefix}",
        f"frp_sum_{prefix}", f"frp_mean_{prefix}", f"frp_max_{prefix}",
        f"n_det_day_{prefix}", f"n_det_night_{prefix}",
        f"frp_sum_day_{prefix}", f"frp_sum_night_{prefix}",
        f"scan_near_{prefix}", f"scan_mid_{prefix}", f"scan_far_{prefix}",
        f"overpass_times_{prefix}"
    ]
    if prefix == "modis":
        expected_cols.extend(["n_terra", "n_aqua"])
    else:
        expected_cols.extend(["n_snpp", "n_noaa20", "n_noaa21"])

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    # Groupby key
    group_cols = ["grid_cell_id", "analysis_date_ist"]
    grouped = df.groupby(group_cols)

    # 1. Base counts and FRP stats
    # Filter valid FRP for stats
    agg_dict = {
        "frp": ["sum", "mean", "max"],
    }
    base_stats = grouped.agg(agg_dict)
    base_stats.columns = [f"frp_{stat}_{prefix}" for stat in ["sum", "mean", "max"]]
    base_stats = base_stats.reset_index()

    # n_det
    n_det_s = grouped.size().rename(f"n_det_{prefix}")
    base_stats = base_stats.merge(n_det_s, on=group_cols)

    # n_unique: unique duplicate_groups
    n_unique_s = df.groupby(group_cols)["duplicate_group"].nunique().rename(f"n_unique_{prefix}")
    base_stats = base_stats.merge(n_unique_s, on=group_cols)

    # Day / Night splits
    day_df = df[df["daynight"] == "D"]
    night_df = df[df["daynight"] == "N"]

    n_day = day_df.groupby(group_cols).size().rename(f"n_det_day_{prefix}")
    n_night = night_df.groupby(group_cols).size().rename(f"n_det_night_{prefix}")
    frp_day = day_df.groupby(group_cols)["frp"].sum().rename(f"frp_sum_day_{prefix}")
    frp_night = night_df.groupby(group_cols)["frp"].sum().rename(f"frp_sum_night_{prefix}")

    for s in [n_day, n_night, frp_day, frp_night]:
        base_stats = base_stats.merge(s, on=group_cols, how="left")

    # Platform counts
    if prefix == "modis":
        n_terra = df[df["satellite"] == "Terra"].groupby(group_cols).size().rename("n_terra")
        n_aqua = df[df["satellite"] == "Aqua"].groupby(group_cols).size().rename("n_aqua")
        base_stats = base_stats.merge(n_terra, on=group_cols, how="left")
        base_stats = base_stats.merge(n_aqua, on=group_cols, how="left")
    else:
        n_snpp = df[df["satellite"] == "N"].groupby(group_cols).size().rename("n_snpp")
        n_noaa20 = df[df["satellite"] == "N20"].groupby(group_cols).size().rename("n_noaa20")
        n_noaa21 = df[df["satellite"] == "N21"].groupby(group_cols).size().rename("n_noaa21")
        base_stats = base_stats.merge(n_snpp, on=group_cols, how="left")
        base_stats = base_stats.merge(n_noaa20, on=group_cols, how="left")
        base_stats = base_stats.merge(n_noaa21, on=group_cols, how="left")

    # Scan-angle bins
    s_near = df[df["scan_bin"] == "near"].groupby(group_cols).size().rename(f"scan_near_{prefix}")
    s_mid = df[df["scan_bin"] == "mid"].groupby(group_cols).size().rename(f"scan_mid_{prefix}")
    s_far = df[df["scan_bin"] == "far"].groupby(group_cols).size().rename(f"scan_far_{prefix}")
    for s in [s_near, s_mid, s_far]:
        base_stats = base_stats.merge(s, on=group_cols, how="left")

    # overpass_times: distinct sorted acquisition times as comma-separated string
    def _collect_times(s):
        sorted_times = sorted(set(s.astype(str).str.zfill(4)))
        return ";".join(sorted_times)

    overpass_s = df.groupby(group_cols)["acq_time"].apply(_collect_times).rename(f"overpass_times_{prefix}")
    base_stats = base_stats.merge(overpass_s, on=group_cols, how="left")

    return base_stats


def _fill_missing_and_types(df: pd.DataFrame) -> None:
    """Fill zeros for count/sum metrics, preserving NaN for mean/max on zero cell-days."""
    count_cols = [
        "n_det_modis", "n_unique_modis", "frp_sum_modis",
        "n_det_day_modis", "n_det_night_modis", "frp_sum_day_modis", "frp_sum_night_modis",
        "n_terra", "n_aqua",
        "scan_near_modis", "scan_mid_modis", "scan_far_modis",
        "n_det_viirs", "n_unique_viirs", "frp_sum_viirs",
        "n_det_day_viirs", "n_det_night_viirs", "frp_sum_day_viirs", "frp_sum_night_viirs",
        "n_snpp", "n_noaa20", "n_noaa21",
        "scan_near_viirs", "scan_mid_viirs", "scan_far_viirs",
    ]
    for c in count_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(np.int32 if not c.startswith("frp_sum") else np.float32)

    # String overpass columns
    for sc in ["overpass_times_modis", "overpass_times_viirs"]:
        if sc in df.columns:
            df[sc] = df[sc].fillna("")


def _populate_zero_table(df: pd.DataFrame) -> None:
    """Populate an all-zero structure for years without observations."""
    df["active_fire_day"] = np.int8(0)
    for p in ["modis", "viirs"]:
        df[f"n_det_{p}"] = np.int32(0)
        df[f"n_unique_{p}"] = np.int32(0)
        df[f"frp_sum_{p}"] = np.float32(0.0)
        df[f"frp_mean_{p}"] = np.float32(np.nan)
        df[f"frp_max_{p}"] = np.float32(np.nan)
        df[f"n_det_day_{p}"] = np.int32(0)
        df[f"n_det_night_{p}"] = np.int32(0)
        df[f"frp_sum_day_{p}"] = np.float32(0.0)
        df[f"frp_sum_night_{p}"] = np.float32(0.0)
        df[f"scan_near_{p}"] = np.int32(0)
        df[f"scan_mid_{p}"] = np.int32(0)
        df[f"scan_far_{p}"] = np.int32(0)
        df[f"overpass_times_{p}"] = ""

    df["n_terra"] = np.int32(0)
    df["n_aqua"] = np.int32(0)
    df["n_snpp"] = np.int32(0)
    df["n_noaa20"] = np.int32(0)
    df["n_noaa21"] = np.int32(0)


def main() -> None:
    print("=" * 60)
    print("NEBULENS — M2.8 Daily Grid Aggregation Pipeline")
    print("=" * 60)

    cell_ids = get_grid_cell_ids()
    print(f"Loaded {len(cell_ids)} 10 km grid cells intersecting Tamil Nadu.")

    years = list(range(2000, 2027))
    total_cell_days = 0

    GRID_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    for y in years:
        out_f = aggregate_year(y, cell_ids)
        # Check rows
        df_y = pd.read_parquet(out_f, columns=["active_fire_day", "n_det_modis", "n_det_viirs"])
        n_rows = len(df_y)
        total_cell_days += n_rows
        active_days = df_y["active_fire_day"].sum()
        m_tot = df_y["n_det_modis"].sum()
        v_tot = df_y["n_det_viirs"].sum()
        print(f"  ✓ {y}: {n_rows:,} cell-days | {active_days:,} active | MODIS={m_tot:,}, VIIRS={v_tot:,} → {out_f.relative_to(PROJECT_ROOT)}")
        del df_y
        gc.collect()

    print("\n" + "=" * 60)
    print("M2.8 Aggregation Complete:")
    print(f"  Total cell-days generated: {total_cell_days:,}")
    print(f"  Output directory: {GRID_DAILY_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
