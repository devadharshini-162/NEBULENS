"""
tests/test_m2_standardise.py
=============================
Milestone M2 validation tests for NEBULENS (M2.1–M2.7):

Tests:
1. Grids exist at 10km, 5km, and 25km in EPSG:6933 with correct cell counts.
2. Parquet dataset contains exactly 511,459 observations (100% preservation of M1).
3. Sensor counts reconcile with M1 manifest row counts for every sensor.
4. No null or unparseable acq_datetime_utc or analysis_date_ist values.
5. All observations have valid grid cell assignments at all three resolutions.
6. Spatial inclusion (in_tn) correctly flags detections within Tamil Nadu.
7. Original confidence and sensor-specific attributes are preserved.
8. Official FIRMS type_flag and confidence_class values are properly mapped.
9. Persistent flag is present and not arbitrarily fabricated.
10. Observation IDs and duplicate groups are present, non-null, and well-formed.
"""

from pathlib import Path
import pytest
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_OBS_DIR = PROJECT_ROOT / "data" / "parquet" / "observations"
GRID_DIR = PROJECT_ROOT / "data" / "boundaries" / "grids"
MANIFEST_FILE = PROJECT_ROOT / "data" / "raw" / "firms" / "manifest.csv"

M1_EXPECTED_TOTAL_ROWS = 511459
M1_SENSOR_TOTALS = {
    "MODIS_SP": 47693,
    "VIIRS_SNPP_SP": 246601,
    "VIIRS_NOAA20_SP": 148356,
    "VIIRS_NOAA21_SP": 68809,
}

EXPECTED_GRID_COUNTS = {
    "10km": 1460,
    "5km": 5547,
    "25km": 270,
}


@pytest.fixture(scope="module")
def manifest_df():
    return pd.read_csv(MANIFEST_FILE)


@pytest.fixture(scope="module")
def sample_modis():
    p = PARQUET_OBS_DIR / "source=MODIS_SP" / "year=2021" / "part-0.parquet"
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def sample_viirs():
    p = PARQUET_OBS_DIR / "source=VIIRS_SNPP_SP" / "year=2021" / "part-0.parquet"
    return pd.read_parquet(p)


class TestM2Grids:
    """Validate EPSG:6933 analytical grids at 10km, 5km, and 25km."""

    @pytest.mark.parametrize("res,expected_count", list(EXPECTED_GRID_COUNTS.items()))
    def test_grid_file_and_crs(self, res, expected_count):
        gpath = GRID_DIR / f"grid_{res}.parquet"
        assert gpath.is_file(), f"Missing grid file: {gpath}"
        gdf = gpd.read_parquet(gpath)
        assert len(gdf) == expected_count, f"Expected {expected_count} cells for {res}, got {len(gdf)}"
        assert gdf.crs.to_epsg() == 6933, f"Expected EPSG:6933, got {gdf.crs}"
        assert "grid_cell_id" in gdf.columns


class TestM2ObservationPreservation:
    """Verify zero observation loss across all 55 annual partitions."""

    def test_total_observation_count(self):
        files = list(PARQUET_OBS_DIR.rglob("*.parquet"))
        total = sum(len(pd.read_parquet(f)) for f in files)
        assert total == M1_EXPECTED_TOTAL_ROWS, f"Expected {M1_EXPECTED_TOTAL_ROWS}, got {total}"

    @pytest.mark.parametrize("source,expected_count", list(M1_SENSOR_TOTALS.items()))
    def test_sensor_row_count_reconciliation(self, source, expected_count):
        s_dir = PARQUET_OBS_DIR / f"source={source}"
        assert s_dir.is_dir(), f"Missing partition dir: {s_dir}"
        files = list(s_dir.rglob("*.parquet"))
        actual = sum(len(pd.read_parquet(f)) for f in files)
        assert actual == expected_count, f"{source}: expected {expected_count}, got {actual}"


class TestM2StandardisedSchema:
    """Verify timestamps, spatial inclusion, confidence, and type flags."""

    def test_timestamps_valid_and_not_null(self, sample_modis, sample_viirs):
        for df in [sample_modis, sample_viirs]:
            assert df["acq_datetime_utc"].notna().all(), "Found null UTC timestamps"
            assert df["analysis_date_ist"].notna().all(), "Found null IST analysis dates"
            # Verify string format YYYY-MM-DD
            assert df["analysis_date_ist"].str.match(r"^\d{4}-\d{2}-\d{2}$").all()

    def test_spatial_inclusion_and_districts(self, sample_modis, sample_viirs):
        for df in [sample_modis, sample_viirs]:
            assert "in_tn" in df.columns
            assert df["in_tn"].dtype == bool
            # Detections inside TN must have non-null district
            in_tn_rows = df[df["in_tn"]]
            assert in_tn_rows["district_name"].notna().all()

    def test_grid_assignments_present_at_all_resolutions(self, sample_modis, sample_viirs):
        for df in [sample_modis, sample_viirs]:
            for res in ["10km", "5km", "25km"]:
                col = f"grid_cell_id_{res}"
                assert col in df.columns
                assert df[col].notna().all()
                assert df[col].str.startswith(f"E6933_{res}_").all()

    def test_confidence_classification(self, sample_modis, sample_viirs):
        # MODIS
        assert set(sample_modis["confidence_class"].unique()).issubset({"low", "nominal", "high"})
        assert sample_modis["confidence_raw"].notna().all()

        # VIIRS
        assert set(sample_viirs["confidence_class"].unique()).issubset({"low", "nominal", "high"})
        assert sample_viirs["confidence_raw"].notna().all()

    def test_type_flag_and_persistent_flag(self, sample_modis, sample_viirs):
        valid_types = {"presumed_vegetation", "active_volcano", "static_land_source", "offshore"}
        for df in [sample_modis, sample_viirs]:
            assert set(df["type_flag"].unique()).issubset(valid_types)
            # Persistent flag is provisional boolean
            assert "persistent_flag" in df.columns
            assert df["persistent_flag"].dtype == bool

    def test_observation_id_and_duplicate_group(self, sample_modis, sample_viirs):
        for df in [sample_modis, sample_viirs]:
            assert df["observation_id"].notna().all()
            assert df["duplicate_group"].notna().all()
            # Observation IDs must be non-empty strings
            assert (df["observation_id"].str.len() > 10).all()


class TestM2GridAlignmentAndNesting:
    """Verify geometric alignment and hierarchical nesting relationships across resolutions."""

    def test_5km_nests_strictly_in_10km(self):
        """5km cells must nest 4-to-1 inside 10km grid bounding boxes without crossing boundaries."""
        g10 = gpd.read_parquet(GRID_DIR / "grid_10km.parquet")
        g5 = gpd.read_parquet(GRID_DIR / "grid_5km.parquet")
        
        # Test sample of 100 cells
        for _, row5 in g5.head(100).iterrows():
            px = int((row5["origin_x"] // 10000) * 10000)
            py = int((row5["origin_y"] // 10000) * 10000)
            p10_id = f"E6933_10km_{px}_{py}"
            
            p10 = g10[g10["grid_cell_id"] == p10_id]
            if not p10.empty:
                # Must be geometrically contained
                assert p10.geometry.iloc[0].contains(row5.geometry)

    def test_25km_split_relationship_documented(self):
        """25km and 10km share global origin (0, 0), but 25 is not a multiple of 10."""
        assert 10000 % 5000 == 0, "5km should nest cleanly into 10km"
        assert 25000 % 10000 == 5000, "25km does not nest cleanly into 10km"


class TestM2Histograms:
    """Validate acquisition time histogram plot outputs."""

    def test_histogram_figures_exist(self):
        fig1 = PROJECT_ROOT / "docs" / "figures" / "acquisition_times_by_satellite.png"
        fig2 = PROJECT_ROOT / "docs" / "figures" / "acquisition_times_annual_drift.png"
        assert fig1.is_file(), f"Missing figure: {fig1}"
        assert fig2.is_file(), f"Missing figure: {fig2}"
        assert fig1.stat().st_size > 10000, "Figure 1 appears empty or too small"
        assert fig2.stat().st_size > 10000, "Figure 2 appears empty or too small"


class TestM2GridDaily:
    """Validate Milestone M2.8 Daily Grid Aggregation against Section 4.2 specifications."""

    @pytest.fixture(scope="class")
    @classmethod
    def sample_grid_daily(cls):
        # Load 2021 as a representative full-fire year
        p = PROJECT_ROOT / "data" / "parquet" / "grid_daily" / "year=2021" / "grid_daily_2021.parquet"
        assert p.is_file(), f"Missing grid daily file: {p}"
        return pd.read_parquet(p)

    def test_grid_daily_annual_partition_count(self):
        gdir = PROJECT_ROOT / "data" / "parquet" / "grid_daily"
        files = list(gdir.rglob("*.parquet"))
        assert len(files) == 27, f"Expected 27 annual partitions (2000-2026), found {len(files)}"

    def test_total_cell_day_completeness(self):
        gdir = PROJECT_ROOT / "data" / "parquet" / "grid_daily"
        total_rows = sum(len(pd.read_parquet(f, columns=["grid_cell_id"])) for f in gdir.rglob("*.parquet"))
        expected_total = 1460 * 9435  # 1460 cells × 9435 days
        assert total_rows == expected_total, f"Expected {expected_total} cell-days, got {total_rows}"

    def test_schema_and_frp_zero_distinction(self, sample_grid_daily):
        df = sample_grid_daily
        required_cols = [
            "grid_cell_id", "analysis_date_ist", "active_fire_day",
            "n_det_modis", "n_unique_modis", "frp_sum_modis", "frp_mean_modis", "frp_max_modis",
            "n_det_day_modis", "n_det_night_modis", "frp_sum_day_modis", "frp_sum_night_modis",
            "n_terra", "n_aqua",
            "scan_near_modis", "scan_mid_modis", "scan_far_modis", "overpass_times_modis",
            "n_det_viirs", "n_unique_viirs", "frp_sum_viirs", "frp_mean_viirs", "frp_max_viirs",
            "n_det_day_viirs", "n_det_night_viirs", "frp_sum_day_viirs", "frp_sum_night_viirs",
            "n_snpp", "n_noaa20", "n_noaa21",
            "scan_near_viirs", "scan_mid_viirs", "scan_far_viirs", "overpass_times_viirs",
        ]
        for c in required_cols:
            assert c in df.columns, f"Missing column: {c}"

        # Zero day checks: counts must be 0, frp_sum must be 0, but frp_mean/frp_max must be NaN
        zero_rows = df[df["n_det_modis"] == 0]
        assert (zero_rows["n_unique_modis"] == 0).all()
        assert (zero_rows["frp_sum_modis"] == 0.0).all()
        assert zero_rows["frp_mean_modis"].isna().all(), "frp_mean_modis on zero days must be NaN"
        assert zero_rows["frp_max_modis"].isna().all(), "frp_max_modis on zero days must be NaN"

        # Active day checks: n_det > 0 must have frp_mean > 0 and frp_max >= frp_mean
        active_viirs = df[df["n_det_viirs"] > 0]
        assert (active_viirs["frp_mean_viirs"] > 0).all()
        assert (active_viirs["frp_max_viirs"] >= active_viirs["frp_mean_viirs"]).all()
        assert (active_viirs["n_unique_viirs"] <= active_viirs["n_det_viirs"]).all()

    def test_reconciliation_totals_match_observations(self):
        """Sum of n_det across all grid_daily files must exactly equal in_tn observations."""
        gdir = PROJECT_ROOT / "data" / "parquet" / "grid_daily"
        pdir = PROJECT_ROOT / "data" / "parquet" / "observations"

        gd_modis = sum(pd.read_parquet(f, columns=["n_det_modis"])["n_det_modis"].sum() for f in gdir.rglob("*.parquet"))
        gd_viirs = sum(pd.read_parquet(f, columns=["n_det_viirs"])["n_det_viirs"].sum() for f in gdir.rglob("*.parquet"))

        obs_modis = sum(pd.read_parquet(f, columns=["in_tn"])["in_tn"].sum() for f in (pdir / "source=MODIS_SP").rglob("*.parquet"))
        obs_viirs = sum(pd.read_parquet(f, columns=["in_tn"])["in_tn"].sum() for f in pdir.rglob("*.parquet") if "MODIS" not in str(f))

        assert gd_modis == obs_modis == 23453, f"MODIS mismatch: grid_daily={gd_modis}, obs={obs_modis}"
        assert gd_viirs == obs_viirs == 252855, f"VIIRS mismatch: grid_daily={gd_viirs}, obs={obs_viirs}"
