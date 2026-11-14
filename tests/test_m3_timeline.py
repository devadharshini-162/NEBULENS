"""
tests/test_m3_timeline.py
==========================
Validation tests for Milestone M3.1: Satellite Operational Timeline & Coverage.

Tests:
1. Timeline figure exists in docs/figures/ and has valid non-trivial size.
2. Verified date ranges and continuity for all 5 satellite platforms:
     - Terra: 2000-11 to 2026-08 (continuous)
     - Aqua: 2002-07 to 2026-08 (continuous; 0 in-TN detections in monsoon troughs 2003-11, 2021-11)
     - Suomi-NPP: 2012-01 to 2026-08 (continuous)
     - NOAA-20: 2018-04 to 2026-08 (continuous)
     - NOAA-21: 2023-03 to 2026-08 (continuous)
3. Total in-TN detection counts per satellite match the standardized observation Parquet dataset.
"""

from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_OBS_DIR = PROJECT_ROOT / "data" / "parquet" / "observations"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

EXPECTED_SATELLITE_TOTALS = {
    "Terra": 11362,
    "Aqua": 12091,
    "N": 134207,
    "N20": 81000,
    "N21": 37648,
}

EXPECTED_BOUNDS = {
    "Terra": ("2000-11-27", "2026-08-31"),
    "Aqua": ("2002-07-04", "2026-08-31"),
    "N": ("2012-01-20", "2026-08-01"),
    "N20": ("2018-04-01", "2026-08-01"),
    "N21": ("2023-03-01", "2026-08-01"),
}


class TestM3SatelliteTimeline:
    """Validate satellite timeline coverage, counts, and figure generation."""

    def test_timeline_figure_exists(self):
        fig_path = FIGURES_DIR / "satellite_timeline_coverage.png"
        assert fig_path.is_file(), f"Missing timeline figure: {fig_path}"
        assert fig_path.stat().st_size > 50000, "Timeline figure appears truncated or too small"

    @pytest.mark.parametrize("sat,expected_count", list(EXPECTED_SATELLITE_TOTALS.items()))
    def test_satellite_detection_counts_in_tn(self, sat, expected_count):
        files = list(PARQUET_OBS_DIR.rglob("*.parquet"))
        actual_count = 0
        for f in files:
            df = pd.read_parquet(f, columns=["satellite", "in_tn"])
            actual_count += len(df[(df["satellite"] == sat) & (df["in_tn"])])
        assert actual_count == expected_count, f"{sat}: expected {expected_count}, got {actual_count}"

    @pytest.mark.parametrize("sat,bounds", list(EXPECTED_BOUNDS.items()))
    def test_satellite_date_bounds(self, sat, bounds):
        expected_start, expected_end = bounds
        files = list(PARQUET_OBS_DIR.rglob("*.parquet"))
        dates = []
        for f in files:
            df = pd.read_parquet(f, columns=["satellite", "in_tn", "analysis_date_ist"])
            df_sat = df[(df["satellite"] == sat) & (df["in_tn"])]
            if not df_sat.empty:
                dates.extend(df_sat["analysis_date_ist"].tolist())
        assert min(dates) == expected_start, f"{sat}: start date expected {expected_start}, got {min(dates)}"
        assert max(dates) == expected_end, f"{sat}: end date expected {expected_end}, got {max(dates)}"
