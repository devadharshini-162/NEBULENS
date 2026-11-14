"""
tests/test_m0_foundation.py
===========================
Validation test suite for NEBULENS Milestone M0: Project Foundation.

Tests:
1. Required directories and project files exist.
2. Boundary files load as GeoParquet with valid CRS (EPSG:4326).
3. State boundary contains Tamil Nadu, is non-empty, and valid.
4. Exactly 38 current Tamil Nadu districts exist, with non-empty & valid geometries.
5. All 38 district names are unique and match the canonical administrative list.
6. All districts intersect or fall within the state boundary.
7. Boundary metadata and provenance are preserved.
8. Environment/configuration code does not expose the actual MAP_KEY.
"""

from pathlib import Path
import geopandas as gpd
import pytest
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical list of all 38 districts in Tamil Nadu
CANONICAL_38_DISTRICTS = [
    "Ariyalur", "Chengalpattu", "Chennai", "Coimbatore", "Cuddalore",
    "Dharmapuri", "Dindigul", "Erode", "Kallakurichi", "Kanchipuram",
    "Kanniyakumari", "Karur", "Krishnagiri", "Madurai", "Mayiladuthurai",
    "Nagapattinam", "Namakkal", "Nilgiris", "Perambalur", "Pudukkottai",
    "Ramanathapuram", "Ranipet", "Salem", "Sivaganga", "Tenkasi",
    "Thanjavur", "Theni", "Thoothukudi", "Tiruchirappalli", "Tirunelveli",
    "Tirupathur", "Tiruppur", "Tiruvallur", "Tiruvannamalai", "Tiruvarur",
    "Vellore", "Viluppuram", "Virudhunagar"
]


# ---------------------------------------------------------------------------
# Test 1: Project structure and required files
# ---------------------------------------------------------------------------
def test_directory_structure():
    """Verify all required top-level directories and files exist."""
    required_dirs = [
        "pipeline",
        "backend",
        "frontend",
        "data/raw",
        "data/parquet",
        "data/boundaries",
        "notebooks",
        "docs",
        "docs/figures",
        "configs",
        "tests",
    ]
    for d in required_dirs:
        dir_path = PROJECT_ROOT / d
        assert dir_path.is_dir(), f"Required directory missing: {d}"

    required_files = [
        ".gitignore",
        ".env.example",
        "README.md",
        "requirements.txt",
        "docs/decisions.md",
        "pipeline/__init__.py",
        "tests/__init__.py",
        "pipeline/check_environment.py",
        "pipeline/fetch_boundaries.py",
        "pipeline/plot_boundaries.py",
    ]
    for f in required_files:
        file_path = PROJECT_ROOT / f
        assert file_path.is_file(), f"Required file missing: {f}"


# ---------------------------------------------------------------------------
# Test 2: GeoParquet files loadable with valid CRS
# ---------------------------------------------------------------------------
def test_boundaries_loadable_and_crs():
    """Verify boundary files exist, load as GeoParquet, and possess valid CRS."""
    state_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_state.parquet"
    districts_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_districts.parquet"

    assert state_path.is_file(), f"State boundary file missing: {state_path}"
    assert districts_path.is_file(), f"District boundary file missing: {districts_path}"

    state_gdf = gpd.read_parquet(state_path)
    districts_gdf = gpd.read_parquet(districts_path)

    assert isinstance(state_gdf, gpd.GeoDataFrame)
    assert isinstance(districts_gdf, gpd.GeoDataFrame)

    assert state_gdf.crs is not None, "State boundary missing CRS"
    assert districts_gdf.crs is not None, "District boundaries missing CRS"
    assert state_gdf.crs.to_epsg() == 4326, f"Expected EPSG:4326 for state, got {state_gdf.crs}"
    assert districts_gdf.crs.to_epsg() == 4326, f"Expected EPSG:4326 for districts, got {districts_gdf.crs}"


# ---------------------------------------------------------------------------
# Test 3: State boundary properties
# ---------------------------------------------------------------------------
def test_state_boundary_validity_and_name():
    """Verify state boundary is non-empty, valid, and contains Tamil Nadu."""
    state_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_state.parquet"
    state_gdf = gpd.read_parquet(state_path)

    assert len(state_gdf) == 1, f"Expected 1 state feature, found {len(state_gdf)}"
    assert "NAME_1" in state_gdf.columns, "Column 'NAME_1' missing from state boundary"
    assert (state_gdf["NAME_1"] == "Tamil Nadu").all(), "State boundary does not identify 'Tamil Nadu'"

    # Geometries must be non-empty and valid
    assert state_gdf.geometry.notnull().all(), "Null geometry found in state boundary"
    assert not state_gdf.geometry.is_empty.any(), "Empty geometry found in state boundary"
    assert state_gdf.geometry.is_valid.all(), "Invalid geometry found in state boundary"


# ---------------------------------------------------------------------------
# Test 4: Current 38 district count and validity
# ---------------------------------------------------------------------------
def test_district_count_and_validity():
    """Verify all 38 current districts are present with valid geometries."""
    districts_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_districts.parquet"
    districts_gdf = gpd.read_parquet(districts_path)

    assert len(districts_gdf) == 38, f"Expected exactly 38 Tamil Nadu districts, found {len(districts_gdf)}"
    assert districts_gdf.geometry.notnull().all(), "Null geometry found in district boundaries"
    assert not districts_gdf.geometry.is_empty.any(), "Empty geometry found in district boundaries"
    assert districts_gdf.geometry.is_valid.all(), "Invalid geometry found in district boundaries"


# ---------------------------------------------------------------------------
# Test 5: Unique and complete district names
# ---------------------------------------------------------------------------
def test_district_names_uniqueness_and_completeness():
    """Verify all 38 canonical district names are present and unique."""
    districts_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_districts.parquet"
    districts_gdf = gpd.read_parquet(districts_path)

    assert "NAME_2" in districts_gdf.columns, "District name column 'NAME_2' missing"
    district_names = sorted(districts_gdf["NAME_2"].tolist())

    assert len(set(district_names)) == 38, "Duplicate district names detected"
    assert district_names == sorted(CANONICAL_38_DISTRICTS), (
        f"District names mismatch canonical list. Missing: {set(CANONICAL_38_DISTRICTS) - set(district_names)}, "
        f"Unexpected: {set(district_names) - set(CANONICAL_38_DISTRICTS)}"
    )

    # Specific check for newly carved districts
    new_districts = ["Mayiladuthurai", "Tenkasi", "Kallakurichi", "Ranipet", "Tirupathur", "Chengalpattu"]
    for nd in new_districts:
        assert nd in district_names, f"Recent district {nd} missing from dataset"


# ---------------------------------------------------------------------------
# Test 6: Spatial relationship between districts and state
# ---------------------------------------------------------------------------
def test_districts_spatial_relationship_with_state():
    """Verify all districts intersect or fall within the state boundary."""
    state_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_state.parquet"
    districts_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_districts.parquet"

    state_gdf = gpd.read_parquet(state_path)
    districts_gdf = gpd.read_parquet(districts_path)

    if hasattr(state_gdf, "union_all"):
        state_geom = state_gdf.union_all()
    else:
        state_geom = state_gdf.unary_union

    for idx, row in districts_gdf.iterrows():
        assert row.geometry.intersects(state_geom), (
            f"District {row.get('NAME_2', idx)} does not intersect the state boundary"
        )


# ---------------------------------------------------------------------------
# Test 7: Source metadata and provenance preservation
# ---------------------------------------------------------------------------
def test_provenance_metadata():
    """Verify dataset provenance attributes are preserved."""
    districts_path = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_districts.parquet"
    districts_gdf = gpd.read_parquet(districts_path)

    required_meta = [
        "DATASET_NAME",
        "PUBLISHER",
        "SOURCE_URL",
        "DOWNLOAD_DATE",
        "LICENCE",
        "ADMIN_REF_YEAR",
    ]
    for col in required_meta:
        assert col in districts_gdf.columns, f"Provenance column '{col}' missing from districts"
        assert districts_gdf[col].notnull().all(), f"Null value in provenance column '{col}'"

    assert (districts_gdf["ADMIN_REF_YEAR"] == "2021").all()
    assert (districts_gdf["LICENCE"] == "Open Data Commons Open Database License (ODbL) 1.0").all()


# ---------------------------------------------------------------------------
# Test 8: Secret leakage prevention
# ---------------------------------------------------------------------------
def test_no_map_key_exposure():
    """Verify .env is gitignored and .env.example only contains placeholders."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore missing"
    gitignore_content = gitignore_path.read_text()
    assert ".env" in gitignore_content, ".gitignore does not exclude .env"

    example_path = PROJECT_ROOT / ".env.example"
    assert example_path.is_file(), ".env.example missing"
    example_vals = dotenv_values(example_path)
    assert "FIRMS_MAP_KEY" in example_vals, "FIRMS_MAP_KEY missing from .env.example"
    assert example_vals["FIRMS_MAP_KEY"] == "your_firms_map_key_here", (
        "Placeholder key should not be a real secret"
    )

    import re
    hex_key_pattern = re.compile(r"FIRMS_MAP_KEY\s*=\s*['\"][a-fA-F0-9]{32}['\"]")
    for py_file in PROJECT_ROOT.glob("**/*.py"):
        if ".venv" in py_file.parts:
            continue
        text = py_file.read_text()
        assert not hex_key_pattern.search(text), f"Potential hardcoded MAP_KEY found in {py_file}"
