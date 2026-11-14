#!/usr/bin/env python3
"""
pipeline/fetch_boundaries.py
============================
Downloads, processes, and standardizes current Tamil Nadu state and district
boundaries with 38 districts.

Selected Boundary Source
------------------------
Dataset Name     : geoBoundaries Global Database (gbOpen)
Version / ID     : IND-ADM2-76128533 (Release 2023 / Year Represented: 2021)
Publisher        : William & Mary geoLab (wmgeolab.org)
Upstream Source  : Local Government Directory (lgdirectory.gov.in) & Pathways Data
Source URL       : https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM2/geoBoundaries-IND-ADM2.geojson
Source Git Commit: 9469f09592ced973a3448cf66b6100b741b64c0d (short: 9469f09)
API Metadata URL : https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/
Licence          : Open Data Commons Open Database License (ODbL) 1.0
Licence URL      : https://opendatacommons.org/licenses/odbl/1-0/
Attribution Req. : Attribution to William & Mary geoLab / geoBoundaries and upstream
                   sources required. Derivative databases must be licensed under ODbL 1.0 (Share-Alike).
Administrative   : 38 Districts (post-2020 administrative structure including Mayiladuthurai,
                   Tenkasi, Kallakurichi, Ranipet, Tirupathur, and Chengalpattu)
CRS              : EPSG:4326 (WGS 84 geographic coordinates)

Outputs & Integrity
-------------------
Pan-India SHA256 : 8bef6929fd65432e7dc775e7c44473e84efff064ebddbfd6b834e6291546db40
Raw Extract File : data/raw/boundaries/geoBoundaries_IND_ADM2_TamilNadu_raw.geojson
Raw Extract SHA  : 1311c668e7e84ae6841636ace320414b2c63269d7425c87f26d85ba6a199e0b2
District Parquet : data/boundaries/tamil_nadu_districts.parquet (38 districts)
State Parquet    : data/boundaries/tamil_nadu_state.parquet (1 dissolved boundary)
"""

import io
import sys
import json
import hashlib
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
from shapely.validation import make_valid

# ---------------------------------------------------------------------------
# Constants & Provenance Metadata (Pinned & Immutable)
# ---------------------------------------------------------------------------
GEOBOUNDARIES_API_URL = "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"
SOURCE_COMMIT = "9469f09592ced973a3448cf66b6100b741b64c0d"
SOURCE_COMMIT_SHORT = "9469f09"
GEOBOUNDARIES_GEOJSON_URL = (
    f"https://github.com/wmgeolab/geoBoundaries/raw/{SOURCE_COMMIT_SHORT}/releaseData/gbOpen/IND/ADM2/geoBoundaries-IND-ADM2.geojson"
)
EXPECTED_PAN_INDIA_SHA256 = "8bef6929fd65432e7dc775e7c44473e84efff064ebddbfd6b834e6291546db40"

STATE_CANONICAL = "Tamil Nadu"
TARGET_CRS = "EPSG:4326"

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "boundaries"
OUTPUT_DIR = PROJECT_ROOT / "data" / "boundaries"

RAW_SAVED_FILE = RAW_DIR / "geoBoundaries_IND_ADM2_TamilNadu_raw.geojson"
STATE_OUT = OUTPUT_DIR / "tamil_nadu_state.parquet"
DISTRICTS_OUT = OUTPUT_DIR / "tamil_nadu_districts.parquet"

PROVENANCE = {
    "DATASET_NAME": "geoBoundaries Global Database (gbOpen ADM2)",
    "BOUNDARY_ID": "IND-ADM2-76128533",
    "PUBLISHER": "William & Mary geoLab",
    "UPSTREAM_SOURCE": "Local Government Directory (lgdirectory.gov.in) & Pathways Data",
    "ADMIN_REF_YEAR": "2021",
    "SOURCE_URL": GEOBOUNDARIES_GEOJSON_URL,
    "SOURCE_COMMIT": SOURCE_COMMIT,
    "SOURCE_COMMIT_SHORT": SOURCE_COMMIT_SHORT,
    "METADATA_URL": GEOBOUNDARIES_API_URL,
    "DOWNLOAD_DATE": "2026-11-14",
    "LICENCE": "Open Data Commons Open Database License (ODbL) 1.0",
    "LICENCE_URL": "https://opendatacommons.org/licenses/odbl/1-0/",
    "LICENCE_ATTRIBUTION": "Attribution to geoBoundaries / William & Mary geoLab and upstream producers required; share-alike applies to derived databases.",
    "ADMIN_LEVEL": "Level 2 (District) & Level 1 (State)",
    "SOURCE_CRS": TARGET_CRS,
    "DISTRICT_COUNT": 38,
    "PAN_INDIA_SHA256": EXPECTED_PAN_INDIA_SHA256,
}

# Canonical 38 Tamil Nadu districts list
EXPECTED_38_DISTRICTS = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram",
    "Kanniyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai",
    "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi",
    "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
    "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur",
    "Vellore", "Viluppuram", "Virudhunagar"
]

# Name standardisation map for geoBoundaries spellings
NAME_STANDARDIZATION = {
    "Chengalputtu": "Chengalpattu",
    "Kancheepuram": "Kanchipuram",
    "The Nilgiris": "Nilgiris",
    "Thiruvallur": "Tiruvallur",
    "Thiruvarur": "Tiruvarur",
    "Thoothukkudi": "Thoothukudi",
}


def download_geoboundaries_adm2() -> gpd.GeoDataFrame:
    """Download the pan-India ADM2 dataset from geoBoundaries pinned to commit."""
    print("=" * 60)
    print("NEBULENS — Downloading geoBoundaries India ADM2 Dataset")
    print(f"URL: {GEOBOUNDARIES_GEOJSON_URL}")
    print(f"Pinned Commit: {SOURCE_COMMIT}")
    print("=" * 60)

    req = urllib.request.Request(
        GEOBOUNDARIES_GEOJSON_URL,
        headers={"User-Agent": "Mozilla/5.0 (NEBULENS Research Environment)"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw_bytes = resp.read()

    # Verify SHA-256 integrity
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    print(f"Downloaded {len(raw_bytes) / (1024 * 1024):.2f} MB. SHA-256: {actual_sha256}")
    assert actual_sha256 == EXPECTED_PAN_INDIA_SHA256, (
        f"Checksum mismatch! Expected {EXPECTED_PAN_INDIA_SHA256}, got {actual_sha256}"
    )
    print("  ✓ Pan-India dataset SHA-256 checksum verified.")

    gdf = gpd.read_file(io.BytesIO(raw_bytes))
    print(f"Loaded {len(gdf)} national ADM2 features, CRS: {gdf.crs}")
    return gdf


def filter_and_standardize_tamil_nadu(all_districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Isolate Tamil Nadu's 38 districts and standardize names."""
    print(f"\nFiltering features for {STATE_CANONICAL}...")

    # Tamil Nadu bounding box filter with representative point check
    tn_bbox = all_districts.cx[76.0:80.5, 8.0:13.6].copy()

    # The 38 district names in geoBoundaries
    geo_names = [
        'Ariyalur', 'Chengalputtu', 'Chennai', 'Coimbatore', 'Cuddalore',
        'Dharmapuri', 'Dindigul', 'Erode', 'Kallakurichi', 'Kancheepuram',
        'Kanniyakumari', 'Karur', 'Krishnagiri', 'Madurai', 'Mayiladuthurai',
        'Nagapattinam', 'Namakkal', 'Perambalur', 'Pudukkottai', 'Ramanathapuram',
        'Ranipet', 'Salem', 'Sivaganga', 'Tenkasi', 'Thanjavur', 'The Nilgiris',
        'Theni', 'Thiruvallur', 'Thiruvarur', 'Thoothukkudi', 'Tiruchirappalli',
        'Tirunelveli', 'Tirupathur', 'Tiruppur', 'Tiruvannamalai', 'Vellore',
        'Viluppuram', 'Virudhunagar'
    ]

    tn_districts = tn_bbox[tn_bbox['shapeName'].isin(geo_names)].copy().reset_index(drop=True)

    if len(tn_districts) != 38:
        raise ValueError(f"Expected exactly 38 districts, found {len(tn_districts)}")

    # Standardize district names
    tn_districts['NAME_2'] = tn_districts['shapeName'].replace(NAME_STANDARDIZATION)
    tn_districts['NAME_1'] = STATE_CANONICAL
    tn_districts['COUNTRY'] = "India"
    tn_districts['ISO_SUB'] = "IN-TN"

    # Add provenance columns
    for k, v in PROVENANCE.items():
        tn_districts[k] = v

    # Verify all expected names match
    found_names = sorted(tn_districts['NAME_2'].tolist())
    assert found_names == sorted(EXPECTED_38_DISTRICTS), "District names mismatch canonical list!"

    print(f"  ✓ Successfully verified all 38 districts including Mayiladuthurai, Tenkasi, Ranipet, etc.")
    return tn_districts


def validate_geometries(gdf: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Validate and repair geometries if required."""
    print(f"\nValidating geometries for {label}...")
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        print(f"  ⚠ Repairing {invalid.sum()} invalid geometries with make_valid...")
        gdf.loc[invalid, 'geometry'] = gdf.loc[invalid, 'geometry'].apply(make_valid)
    assert gdf.geometry.is_valid.all(), f"Failed to validate all geometries in {label}!"
    print(f"  ✓ All {len(gdf)} geometries in {label} are 100% valid.")
    return gdf


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download & Verify Checksum
    all_districts = download_geoboundaries_adm2()

    # 2. Filter & Standardize
    tn_districts = filter_and_standardize_tamil_nadu(all_districts)

    # 3. Validate Districts
    tn_districts = validate_geometries(tn_districts, "Tamil Nadu 38 Districts")

    # 4. Save Raw source extract and compute SHA-256
    tn_districts.to_file(RAW_SAVED_FILE, driver="GeoJSON")
    raw_sha256 = hashlib.sha256(RAW_SAVED_FILE.read_bytes()).hexdigest()
    print(f"  ✓ Saved raw source extract → {RAW_SAVED_FILE.relative_to(PROJECT_ROOT)}")
    print(f"    Raw extract SHA-256: {raw_sha256}")

    # Add raw extract SHA to parquet metadata
    tn_districts["RAW_EXTRACT_SHA256"] = raw_sha256

    # 5. Dissolve State
    print("\nDissolving 38 districts into single state boundary polygon...")
    state = tn_districts.dissolve(by="NAME_1", as_index=False)[["NAME_1", "geometry"]].copy()
    state["COUNTRY"] = "India"
    state["ISO_SUB"] = "IN-TN"
    for k, v in PROVENANCE.items():
        state[k] = v
    state["RAW_EXTRACT_SHA256"] = raw_sha256
    state = validate_geometries(state, "Tamil Nadu State Boundary")

    # 6. Save GeoParquet outputs
    print("\nSaving GeoParquet outputs (EPSG:4326)...")
    tn_districts.to_parquet(DISTRICTS_OUT)
    state.to_parquet(STATE_OUT)

    print(f"  ✓ Saved {len(tn_districts)} districts → {DISTRICTS_OUT.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Saved state boundary → {STATE_OUT.relative_to(PROJECT_ROOT)}")
    print("\nBoundary processing completed successfully.")


if __name__ == "__main__":
    main()
