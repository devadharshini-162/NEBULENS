"""
tests/test_m3_analysis.py
=========================
Automated tests for NEBULENS Milestone M3 (M3.2–M3.7):
- M3.2: Detection comparison, correlations, seasonal and day/night splits.
- M3.3: FRP distributions, quantiles, and scan-angle effects.
- M3.4: Aqua–VIIRS pair linking, sensitivity, and Terra separation gap verification.
- M3.5: Land-cover assignment (1,460 grid cells, 276,308 observations) and stratified metrics.
- M3.6: Tier 1 coverage proxies, cloud source matrix, and orbital exposure timeline.
- M3.7: Resolution sparsity analysis, zero fractions, and resolution hierarchy.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M3_DIR = PROJECT_ROOT / "data" / "analysis" / "m3"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"


class TestM32DetectionComparison:
    """Tests for Milestone M3.2: MODIS vs VIIRS detection comparisons."""

    def test_m3_2_files_exist(self):
        assert (M3_DIR / "m3_2_daily_comparison.parquet").is_file()
        assert (M3_DIR / "m3_2_weekly_comparison.parquet").is_file()
        assert (M3_DIR / "m3_2_monthly_comparison.parquet").is_file()
        assert (M3_DIR / "m3_2_summary_metrics.csv").is_file()
        assert (FIGURES_DIR / "m3_2_detection_comparison.png").is_file()

    def test_daily_comparison_row_count_and_dates(self):
        df = pd.read_parquet(M3_DIR / "m3_2_daily_comparison.parquet")
        assert len(df) == 5338  # Days from 2012-01-20 to 2026-08-31
        assert df["analysis_date_ist"].min() == "2012-01-20"
        assert df["analysis_date_ist"].max() == "2026-08-31"

    def test_monthly_correlations_and_ratio(self):
        metrics = pd.read_csv(M3_DIR / "m3_2_summary_metrics.csv")
        monthly_row = metrics[metrics["comparison_level"] == "Monthly (Regional)"]
        assert not monthly_row.empty
        pearson_r = monthly_row["pearson_r"].values[0]
        spearman_rho = monthly_row["spearman_rho"].values[0]
        viirs_modis_ratio = monthly_row["overall_detection_ratio"].values[0]

        assert pearson_r >= 0.70, f"Expected Pearson r >= 0.70, got {pearson_r}"
        assert spearman_rho >= 0.70, f"Expected Spearman rho >= 0.70, got {spearman_rho}"
        assert 12.0 <= viirs_modis_ratio <= 20.0, f"Unexpected ratio {viirs_modis_ratio}"


class TestM33FRPScanComparison:
    """Tests for Milestone M3.3: FRP and scan-angle distributions."""

    def test_m3_3_files_exist(self):
        assert (M3_DIR / "m3_3_frp_summary_by_satellite.csv").is_file()
        assert (M3_DIR / "m3_3_frp_by_scan_bin.csv").is_file()
        assert (FIGURES_DIR / "m3_3_frp_scan_comparison.png").is_file()

    def test_frp_satellite_statistics(self):
        df = pd.read_csv(M3_DIR / "m3_3_frp_summary_by_satellite.csv")
        overall = df[df["group"] == "Overall"]
        assert len(overall) == 5  # Terra, Aqua, N, N20, N21

        modis_medians = overall[overall["platform_name"].str.contains("MODIS")]["frp_median"].values
        viirs_medians = overall[overall["platform_name"].str.contains("VIIRS")]["frp_median"].values

        assert all(m > 10.0 for m in modis_medians), "MODIS median FRP should exceed 10 MW"
        assert all(v < 6.0 for v in viirs_medians), "VIIRS median FRP should be under 6 MW"
        assert min(modis_medians) > max(viirs_medians), "MODIS median FRP should exceed VIIRS median FRP"

    def test_scan_bin_counts_reconcile(self):
        df = pd.read_csv(M3_DIR / "m3_3_frp_by_scan_bin.csv")
        for sensor in ["MODIS", "VIIRS"]:
            sub = df[df["sensor_family"] == sensor]
            assert len(sub) == 4  # Nadir, Mid, Far, Edge
            assert (sub["count"] > 0).all()


class TestM34PairLinkAnalysis:
    """Tests for Milestone M3.4: Aqua-VIIRS pair linking and Terra verification."""

    def test_m3_4_files_exist(self):
        assert (M3_DIR / "m3_4_aqua_viirs_pairs.parquet").is_file()
        assert (M3_DIR / "m3_4_pair_sensitivity.csv").is_file()
        assert (M3_DIR / "m3_4_terra_coincidence_report.csv").is_file()
        assert (FIGURES_DIR / "m3_4_pair_link_analysis.png").is_file()

    def test_aqua_viirs_pair_properties(self):
        pairs = pd.read_parquet(M3_DIR / "m3_4_aqua_viirs_pairs.parquet")
        assert len(pairs) >= 1500  # Baseline 1 km / 15 min yields 1,809 strictly validated pairs
        assert (pairs["distance_m"] <= 1000.0).all()
        assert (pairs["time_diff_sec"] <= 900.0).all()

    def test_terra_coincidence_minimum_gap(self):
        terra_rep = pd.read_csv(M3_DIR / "m3_4_terra_coincidence_report.csv")
        assert not terra_rep.empty
        min_seconds = terra_rep["minimum_time_difference_seconds"].values[0]
        min_hours = terra_rep["minimum_time_difference_hours"].values[0]
        verdict = terra_rep["verdict"].values[0]

        assert min_seconds >= 3600, f"Expected Terra min gap >= 3600s, got {min_seconds}"
        assert min_hours >= 1.0, f"Expected Terra min gap >= 1.0h, got {min_hours}"
        assert "NO COINCIDENT PASSES" in verdict


class TestM35LandCoverStratification:
    """Tests for Milestone M3.5: Land-cover assignment and stratification."""

    def test_m3_5_files_exist(self):
        assert (M3_DIR / "m3_5_grid_landcover.parquet").is_file()
        assert (M3_DIR / "m3_5_detection_landcover.parquet").is_file()
        assert (M3_DIR / "m3_5_modis_viirs_by_landcover.csv").is_file()
        assert (M3_DIR / "m3_5_frp_by_landcover.csv").is_file()
        assert (FIGURES_DIR / "m3_5_landcover_stratified.png").is_file()
        assert (REPORTS_DIR / "m3_5_landcover_evaluation.md").is_file()

    def test_grid_landcover_completeness(self):
        grid_df = pd.read_parquet(M3_DIR / "m3_5_grid_landcover.parquet")
        assert len(grid_df) == 1460, f"Expected 1460 cells, got {len(grid_df)}"
        assert grid_df["landcover_class"].isna().sum() == 0, "Grid cells have missing landcover classes"
        assert grid_df["district_name"].isna().sum() == 0, "Grid cells have missing districts"

    def test_detection_landcover_reconciliation(self):
        obs_df = pd.read_parquet(M3_DIR / "m3_5_detection_landcover.parquet")
        assert len(obs_df) == 276308, f"Expected 276,308 in-TN detections, got {len(obs_df)}"
        assert obs_df["landcover_class"].isna().sum() == 0, "Detections have missing landcover classes"

    def test_landcover_sample_sizes_and_ratios(self):
        lc_df = pd.read_csv(M3_DIR / "m3_5_modis_viirs_by_landcover.csv")
        assert (lc_df["sample_size_adequate"] == True).all()
        # Forest ratio should be lower than delta cropland ratio
        forest_ratio = lc_df[lc_df["landcover_class"] == "Forest & Woodland"]["viirs_to_modis_ratio"].values[0]
        crop_ratio = lc_df[lc_df["landcover_class"] == "Cropland (Intensive / Delta)"]["viirs_to_modis_ratio"].values[0]
        assert forest_ratio < crop_ratio, f"Forest ratio ({forest_ratio}) should be < Crop ratio ({crop_ratio})"


class TestM36CoverageProxies:
    """Tests for Milestone M3.6: Coverage proxies and cloud source evaluation."""

    def test_m3_6_files_exist(self):
        assert (M3_DIR / "m3_6_coverage_tier1_summary.csv").is_file()
        assert (M3_DIR / "m3_6_cloud_source_evaluation.csv").is_file()
        assert (M3_DIR / "m3_6_orbital_exposure_timeline.csv").is_file()
        assert (FIGURES_DIR / "m3_6_coverage_proxies.png").is_file()
        assert (REPORTS_DIR / "m3_6_coverage_and_clouds.md").is_file()

    def test_tier1_summary_coverage(self):
        t1 = pd.read_csv(M3_DIR / "m3_6_coverage_tier1_summary.csv")
        assert len(t1) == 27  # 2000 through 2026
        assert (t1["active_cell_days"] > 0).all()

    def test_cloud_evaluation_matrix(self):
        cloud_df = pd.read_csv(M3_DIR / "m3_6_cloud_source_evaluation.csv")
        assert len(cloud_df) >= 4
        era5 = cloud_df[cloud_df["candidate_source"].str.contains("ERA5")]
        assert not era5.empty
        assert "Recommended" in era5["recommendation"].values[0]

    def test_orbital_exposure_jump(self):
        exp_df = pd.read_csv(M3_DIR / "m3_6_orbital_exposure_timeline.csv")
        early_passes = exp_df[exp_df["epoch"] == "Terra Only"]["total_view_opportunities_per_day"].values[0]
        late_passes = exp_df[exp_df["epoch"].str.contains("NOAA-21")]["total_view_opportunities_per_day"].values[0]
        assert early_passes < 2.0
        assert late_passes > 9.0
        assert late_passes > early_passes * 4.5  # Over 4.5x exposure expansion


class TestM37ResolutionDecision:
    """Tests for Milestone M3.7: Resolution sparsity and zero fractions."""

    def test_m3_7_files_exist(self):
        assert (M3_DIR / "m3_7_resolution_analysis.csv").is_file()
        assert (FIGURES_DIR / "m3_7_sparsity_analysis.png").is_file()
        assert (REPORTS_DIR / "m3_7_resolution_decision.md").is_file()

    def test_resolution_table_completeness(self):
        res = pd.read_csv(M3_DIR / "m3_7_resolution_analysis.csv")
        assert len(res) == 12  # 2 periods * 3 grids * 2 temporal resolutions

        # 10 km daily zero fraction is between 98.0% and 99.5%
        res_10k_daily = res[(res["grid_resolution"].str.contains("10 km")) & (res["temporal_resolution"] == "Daily") & (res["period"].str.contains("Full"))]
        assert not res_10k_daily.empty
        z_10k = res_10k_daily["zero_detection_fraction_pct"].values[0]
        assert 98.0 <= z_10k <= 99.5, f"Expected 10 km daily zero pct ~98.9%, got {z_10k}"

        # Weekly zero fraction is strictly less than daily zero fraction for all grids
        for g in res["grid_resolution"].unique():
            d_val = res[(res["grid_resolution"] == g) & (res["temporal_resolution"] == "Daily") & (res["period"].str.contains("Full"))]["zero_detection_fraction_pct"].values[0]
            w_val = res[(res["grid_resolution"] == g) & (res["temporal_resolution"] == "Weekly") & (res["period"].str.contains("Full"))]["zero_detection_fraction_pct"].values[0]
            assert w_val < d_val, f"Weekly zero fraction ({w_val}) should be < daily ({d_val}) for {g}"


class TestM38FindingsAndGateA:
    """Tests for Milestone M3.8: Findings note and Gate A decision."""

    def test_m3_8_findings_note_exists_and_answers_questions(self):
        note_file = REPORTS_DIR / "m3_findings_note.md"
        assert note_file.is_file()
        content = note_file.read_text()

        # Check Gate A decision
        assert "Gate A Decision:" in content or "Gate A recommendation" in content.lower()
        assert "PASSES" in content

        # Check all 4 research questions are explicitly answered
        assert "Question 1:" in content
        assert "Question 2:" in content
        assert "Question 3:" in content
        assert "Question 4:" in content
