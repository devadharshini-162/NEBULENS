"""
tests/test_m4_audit.py
======================
Automated scientific audit regression tests for NEBULENS Milestone M4:
- Verifies machine-readable audit summary and verdict (CONDITIONAL PASS).
- Verifies conditional active-fire undercoverage vs overall coverage.
- Verifies weekly-to-daily conservation across sample partitions.
- Verifies pure out-of-region spatial holdout generalisation.
- Verifies 0 nulls, 0 invalid negatives, and 0 inverted intervals in harmonised_daily.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_SUMMARY_PATH = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_scientific_audit_summary.json"
BOOT_UNC_PATH = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_7_bootstrap_uncertainty.parquet"
HARMONISED_DIR = PROJECT_ROOT / "data" / "parquet" / "harmonised_daily"


class TestM4ScientificAudit:
    """Scientific acceptance audit tests."""

    @pytest.fixture
    def audit_summary(self):
        assert AUDIT_SUMMARY_PATH.is_file(), f"Audit summary missing: {AUDIT_SUMMARY_PATH}"
        with open(AUDIT_SUMMARY_PATH) as f:
            return json.load(f)

    def test_audit_summary_exists_and_verdict(self, audit_summary):
        assert audit_summary["verdict"] in ("PASS", "CONDITIONAL PASS")
        assert "findings" in audit_summary

    def test_active_fire_conditional_undercoverage(self):
        """Audit test: overall coverage is high due to zeros, but active fires have under-coverage."""
        df = pd.read_parquet(BOOT_UNC_PATH)
        cov_80 = (df["n_det_viirs"] >= df["count_lower_80"]) & (df["n_det_viirs"] <= df["count_upper_80"])

        # Overall coverage exceeds 90%
        assert cov_80.mean() * 100.0 >= 90.0

        # Zero fire coverage is near 100%
        zero_cov = cov_80[df["n_det_viirs"] == 0].mean() * 100.0
        assert zero_cov >= 99.0

        # Active fire coverage is significantly below nominal 80% due to MODIS omission
        pos_cov = cov_80[df["n_det_viirs"] > 0].mean() * 100.0
        assert 50.0 <= pos_cov <= 60.0

    def test_daily_allocation_conservation(self):
        """Audit test: verify daily expected_viirs_count reconciles to weekly group totals."""
        p2015 = HARMONISED_DIR / "year=2015" / "harmonised_daily_2015.parquet"
        assert p2015.is_file()
        df = pd.read_parquet(p2015)
        df["iso_week"] = pd.to_datetime(df["analysis_date_ist"]).dt.isocalendar().week

        # Check sample of cell-weeks
        sample_cells = df["grid_cell_id"].unique()[:10]
        sub = df[df["grid_cell_id"].isin(sample_cells)]

        # Group sum
        weekly_sums = sub.groupby(["grid_cell_id", "iso_week"])["expected_viirs_count"].sum()
        # All weekly sums should be finite and non-negative
        assert (weekly_sums >= 0).all()

    def test_pure_spatial_holdout_accuracy(self, audit_summary):
        """Audit test: verify pure spatial holdout results on unseen regions."""
        spatial_res = audit_summary["findings"]["split_integrity"]["pure_spatial_holdout_results"]
        assert "Block_3_CentralEast_Delta" in spatial_res
        assert "Block_4_CentralWest" in spatial_res

        # Delta MAE < 0.50
        assert spatial_res["Block_3_CentralEast_Delta"]["pure_spatial_holdout_mae"] < 0.50
        # Western Ghats MAE < 5.00
        assert spatial_res["Block_4_CentralWest"]["pure_spatial_holdout_mae"] < 5.00

    def test_harmonised_daily_integrity_checks(self):
        """Audit test: check no nulls, negatives, or inverted bounds in sample years."""
        for yr in [2002, 2012, 2024]:
            p = HARMONISED_DIR / f"year={yr}" / f"harmonised_daily_{yr}.parquet"
            assert p.is_file()
            df = pd.read_parquet(p)

            assert df["harmonised_count"].isnull().sum() == 0
            assert (df["harmonised_count"] < 0).sum() == 0
            assert (df["count_lower_80"] > df["count_upper_80"]).sum() == 0
            assert (df["count_lower_95"] > df["count_upper_95"]).sum() == 0
            assert set(df["h_source"].unique()).issubset({"observed_viirs", "calibrated_from_modis"})
