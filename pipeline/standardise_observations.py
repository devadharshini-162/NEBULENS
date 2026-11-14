#!/usr/bin/env python3
"""
pipeline/standardise_observations.py
====================================
NEBULENS — Milestone M2: Data Standardisation, Grid Generation & Parquet Engine.

Implements M2.1–M2.7:
- Generates EPSG:6933 analytical grids at 10 km (primary), 5 km, and 25 km.
- Standardises all 626 raw NASA FIRMS CSVs into a unified Parquet dataset:
    data/parquet/observations/source={source}/year={year}/part-0.parquet
- Preserves every original sensor attribute and all 511,459 observations without loss.
- Parses UTC timestamps and computes Indian Standard Time (IST) analysis dates.
- Computes spatial inclusion within Tamil Nadu (in_tn, district_name).
- Maps FIRMS confidence to standard categorical levels while preserving original values.
- Implements type_flag according to official NASA FIRMS definitions.
- Sets persistent_flag as provisional (unclassified / False) without inventing thresholds.
- Assigns spatial grid cells for 10 km, 5 km, and 25 km resolutions in EPSG:6933.
- Defines stable observation_id and duplicate_group.
"""

import os
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import zoneinfo
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import pyproj

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
RAW_FIRMS_DIR = PROJECT_ROOT / "data" / "raw" / "firms"
MANIFEST_FILE = RAW_FIRMS_DIR / "manifest.csv"
BOUNDARIES_DIR = PROJECT_ROOT / "data" / "boundaries"
PARQUET_OBS_DIR = PROJECT_ROOT / "data" / "parquet" / "observations"
GRID_DIR = PROJECT_ROOT / "data" / "boundaries" / "grids"

STATE_BOUNDARY_FILE = BOUNDARIES_DIR / "tamil_nadu_state.parquet"
DISTRICTS_BOUNDARY_FILE = BOUNDARIES_DIR / "tamil_nadu_districts.parquet"

# Timezone
IST_ZONE = zoneinfo.ZoneInfo("Asia/Kolkata")
UTC_ZONE = timezone.utc

# Grid resolutions (metres in EPSG:6933)
RESOLUTIONS = {
    "10km": 10000,
    "5km": 5000,
    "25km": 25000,
}


def generate_grids() -> dict[str, gpd.GeoDataFrame]:
    """
    Generate aligned EPSG:6933 equal-area grids at 10km, 5km, and 25km.
    Saves them to data/boundaries/grids/grid_{res}.parquet
    Returns dict of GeoDataFrames.
    """
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    state = gpd.read_parquet(STATE_BOUNDARY_FILE)
    state_6933 = state.to_crs("EPSG:6933")
    state_geom = state_6933.geometry.iloc[0]
    bounds = state_6933.total_bounds  # [minx, miny, maxx, maxy]

    grid_gdfs = {}
    for name, res in RESOLUTIONS.items():
        out_path = GRID_DIR / f"grid_{name}.parquet"
        
        # Grid boundaries snapped to resolution multiples
        minx = np.floor(bounds[0] / res) * res
        maxx = np.ceil(bounds[2] / res) * res
        miny = np.floor(bounds[1] / res) * res
        maxy = np.ceil(bounds[3] / res) * res

        xs = np.arange(minx, maxx, res)
        ys = np.arange(miny, maxy, res)

        records = []
        for x in xs:
            for y in ys:
                b = box(x, y, x + res, y + res)
                intersects = state_geom.intersects(b)
                if intersects:
                    cell_id = f"E6933_{name}_{int(x)}_{int(y)}"
                    records.append({
                        "grid_cell_id": cell_id,
                        "res_km": res // 1000,
                        "origin_x": x,
                        "origin_y": y,
                        "geometry": b,
                    })

        gdf = gpd.GeoDataFrame(records, crs="EPSG:6933")
        gdf.to_parquet(out_path, index=False)
        grid_gdfs[name] = gdf
        print(f"  ✓ Saved grid_{name}.parquet: {len(gdf)} intersecting cells → {out_path.relative_to(PROJECT_ROOT)}")

    return grid_gdfs


def get_cell_id_array(xs: np.ndarray, ys: np.ndarray, res_name: str, res: int) -> np.ndarray:
    """Compute vectorised stable grid_cell_id from EPSG:6933 coordinates."""
    cell_xs = np.floor(xs / res) * res
    cell_ys = np.floor(ys / res) * res
    # Format strings efficiently
    return np.array([f"E6933_{res_name}_{int(cx)}_{int(cy)}" for cx, cy in zip(cell_xs, cell_ys)])


def map_modis_confidence(conf_val: int | float | str) -> str:
    """
    Official NASA FIRMS MODIS Confidence:
    Continuous percentage 0-100%.
    Low: 0-29%
    Nominal: 30-79%
    High: 80-100%
    """
    try:
        val = float(conf_val)
        if val >= 80:
            return "high"
        elif val >= 30:
            return "nominal"
        else:
            return "low"
    except Exception:
        return "unclassified"


def map_viirs_confidence(conf_val: str) -> str:
    """
    Official NASA FIRMS VIIRS Confidence:
    l: low
    n: nominal
    h: high
    """
    s = str(conf_val).strip().lower()
    if s == "h":
        return "high"
    elif s == "n":
        return "nominal"
    elif s == "l":
        return "low"
    else:
        return "unclassified"


def parse_acq_datetime(acq_date_series: pd.Series, acq_time_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Parse acq_date ('YYYY-MM-DD') and acq_time (integer/string 'HHMM' or 'HMM') into:
    1. acq_datetime_utc (datetime64[ns, UTC])
    2. analysis_date_ist (object or datetime64[ns] date 'YYYY-MM-DD' in Asia/Kolkata)
    """
    # Format acq_time to 4-digit HHMM string
    times_str = acq_time_series.astype(str).str.zfill(4)
    # Combine strings: 'YYYY-MM-DD HH:MM:00'
    combined = acq_date_series.astype(str) + " " + times_str.str.slice(0, 2) + ":" + times_str.str.slice(2, 4) + ":00"
    
    dt_utc = pd.to_datetime(combined, format="%Y-%m-%d %H:%M:%S", errors="coerce").dt.tz_localize("UTC")
    # Convert to IST (UTC+05:30)
    dt_ist = dt_utc.dt.tz_convert(IST_ZONE)
    date_ist = dt_ist.dt.strftime("%Y-%m-%d")
    
    return dt_utc, date_ist


def process_observations() -> None:
    """Process all 626 raw CSVs and save partitioned Parquet."""
    print("=" * 60)
    print("NEBULENS M2 — Standardisation & Grid Assignment")
    print("=" * 60)

    # 1. Grids
    print("\n1. Generating EPSG:6933 analytical grids...")
    generate_grids()

    # 2. Load administrative boundaries for point-in-polygon
    print("\n2. Loading Tamil Nadu administrative boundary...")
    state_gdf = gpd.read_parquet(STATE_BOUNDARY_FILE)
    districts_gdf = gpd.read_parquet(DISTRICTS_BOUNDARY_FILE)
    state_geom = state_gdf.geometry.iloc[0]

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

    # 3. Read manifest to iterate by source and year
    manifest = pd.read_csv(MANIFEST_FILE)
    sources = manifest["source"].unique()

    total_processed_rows = 0
    total_in_tn_rows = 0

    PARQUET_OBS_DIR.mkdir(parents=True, exist_ok=True)

    for source in sources:
        src_manifest = manifest[manifest["source"] == source]
        years = sorted(src_manifest["year"].unique())
        print(f"\nProcessing {source} across {len(years)} years...")

        is_modis = "MODIS" in source

        for year in years:
            year_rows = src_manifest[src_manifest["year"] == year]
            dfs = []
            for _, r in year_rows.iterrows():
                fpath = PROJECT_ROOT / r["file_path"]
                cdf = pd.read_csv(fpath)
                if len(cdf) > 0:
                    dfs.append(cdf)

            if not dfs:
                continue

            df = pd.concat(dfs, ignore_index=True)
            n_raw = len(df)
            total_processed_rows += n_raw

            # Coordinates & Point geometries
            lats = df["latitude"].values
            lons = df["longitude"].values

            # Vectorised reprojection to EPSG:6933
            xs_6933, ys_6933 = transformer.transform(lons, lats)
            df["x_6933"] = xs_6933
            df["y_6933"] = ys_6933

            # Grid cell assignments
            df["grid_cell_id_10km"] = get_cell_id_array(xs_6933, ys_6933, "10km", 10000)
            df["grid_cell_id_5km"] = get_cell_id_array(xs_6933, ys_6933, "5km", 5000)
            df["grid_cell_id_25km"] = get_cell_id_array(xs_6933, ys_6933, "25km", 25000)
            # Default analytical grid
            df["grid_cell_id"] = df["grid_cell_id_10km"]

            # Timestamps
            dt_utc, date_ist = parse_acq_datetime(df["acq_date"], df["acq_time"])
            df["acq_datetime_utc"] = dt_utc
            df["analysis_date_ist"] = date_ist

            # Spatial inclusion in Tamil Nadu
            # Using spatial index / contains on points
            points = gpd.points_from_xy(lons, lats, crs="EPSG:4326")
            obs_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
            
            # Point in polygon with districts
            # Spatial join with districts to get district_name and in_tn
            joined = gpd.sjoin(obs_gdf, districts_gdf[["shapeName", "geometry"]], how="left", predicate="intersects")
            # Drop any duplicates from boundary vertex sharing
            joined = joined[~joined.index.duplicated(keep="first")]
            
            df["district_name"] = joined["shapeName"].values
            df["in_tn"] = df["district_name"].notna()
            total_in_tn_rows += df["in_tn"].sum()

            # Confidence classification
            if is_modis:
                df["confidence_raw"] = df["confidence"].astype(str)
                df["confidence_class"] = df["confidence"].apply(map_modis_confidence)
            else:
                df["confidence_raw"] = df["confidence"].astype(str)
                df["confidence_class"] = df["confidence"].apply(map_viirs_confidence)

            # Type flag according to official NASA FIRMS definitions:
            # 0: vegetation, 1: volcano, 2: other static land, 3: offshore
            type_mapping = {
                0: "presumed_vegetation",
                1: "active_volcano",
                2: "static_land_source",
                3: "offshore",
            }
            df["type_flag"] = df["type"].map(type_mapping).fillna("unclassified")

            # Persistent flag (provisional: marked false until multi-temporal recurrence model is established)
            df["persistent_flag"] = False

            # Sensor identity standardization
            df["source"] = source
            df["year"] = int(year)

            # Duplicate group: overpass identifier linking spatial cluster of same satellite pass
            # Satellite overpass signature: satellite + acq_date + acq_time (hour/minute cluster)
            # Detections sharing the same overpass window and cell represent the same observation opportunity
            df["overpass_id"] = (
                df["source"].astype(str) + "_" +
                df["satellite"].astype(str) + "_" +
                df["acq_date"].astype(str) + "_" +
                df["acq_time"].astype(str).str.zfill(4)
            )
            df["duplicate_group"] = df["overpass_id"] + "_" + df["grid_cell_id"]

            # Unique observation ID: source + YYYY + row hash of primary attributes
            row_signatures = [
                f"{source}_{year}_{lat:.5f}_{lon:.5f}_{dt}_{sat}_{frp}"
                for lat, lon, dt, sat, frp in zip(
                    df["latitude"], df["longitude"], df["acq_time"], df["satellite"], df["frp"]
                )
            ]
            df["observation_id"] = [
                f"{source[:3]}_{year}_{hashlib.md5(sig.encode('utf-8')).hexdigest()[:12]}"
                for sig in row_signatures
            ]

            # Sensor-specific brightness columns preservation
            # Standardise common brightness column while preserving sensor-specific
            if is_modis:
                df["brightness_4um"] = df["brightness"]
                df["brightness_11um"] = df["bright_t31"]
                # Keep original raw columns as well
            else:
                df["brightness_4um"] = df["bright_ti4"]
                df["brightness_11um"] = df["bright_ti5"]

            # Output Parquet partition
            out_partition_dir = PARQUET_OBS_DIR / f"source={source}" / f"year={year}"
            out_partition_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_partition_dir / "part-0.parquet"

            df.to_parquet(out_file, index=False)
            print(f"  ✓ Saved {source} {year}: {n_raw} rows ({df['in_tn'].sum()} in TN) → {out_file.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 60)
    print("M2.1–M2.7 Standardisation Complete:")
    print(f"  Total raw observations processed: {total_processed_rows}")
    print(f"  Total observations inside Tamil Nadu: {total_in_tn_rows}")
    print(f"  Output Parquet directory: {PARQUET_OBS_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 60)


if __name__ == "__main__":
    process_observations()
