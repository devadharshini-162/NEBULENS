"""
tests/test_m5_validation.py
===========================
Automated test suite for Milestone M5: Validation Framework & Gate B Decision.

Verifies:
1. All M5 machine-readable artifacts and diagnostic figures exist and are valid.
2. Section 4.4 seven-test inventory is complete with explicit criteria and verdicts.
3. Overlap-period comparability and seasonal profile (correlation >= 0.85, peak month agreement).
4. Land-cover per-class evaluation with explicit agro-ecological proxy acknowledgment.
5. Separation of out-of-time block evaluation and pure spatial holdouts.
6. Sensor transition level-shift results (Jan 2012 jump reduction and July 2002 limitation).
7. Empirical uncertainty coverage (marginal vs active-fire undercoverage).
8. App readiness matrix explicitly controls go-live permissions and labels.
9. Gate B decision is pre-registered, documented, and awarded as CONDITIONAL PASS.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
M5_DATA_DIR = PROJECT_ROOT / "data" / "analysis" / "m5"
M5_FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def test_m5_artifacts_exist():
    """Verify all machine-readable CSVs, JSONs, and figures exist."""
    required_csvs = [
        "m5_1_test_inventory.csv",
        "m5_2_overlap_annual_comparison.csv",
        "m5_2_seasonal_profile.csv",
        "m5_3_landcover_metrics.csv",
        "m5_4_temporal_metrics.csv",
        "m5_4_pure_spatial_holdout.csv",
        "m5_5_sensor_transitions.csv",
        "m5_6_uncertainty_coverage.csv",
        "m5_app_readiness_matrix.csv"
    ]
    for csv_file in required_csvs:
        path = M5_DATA_DIR / csv_file
        assert path.exists(), f"Missing required CSV: {csv_file}"
        assert path.stat().st_size > 50, f"CSV {csv_file} is empty"

    required_jsons = [
        "m5_gate_b_decision.json",
        "m5_validation_summary.json",
        "m5_gate_b_final_audit.json"
    ]
    for json_file in required_jsons:
        path = M5_DATA_DIR / json_file
        assert path.exists(), f"Missing required JSON: {json_file}"
        with open(path) as f:
            data = json.load(f)
            assert len(data) > 0, f"JSON {json_file} is empty"

    # Also verify final audit report exists
    audit_report = PROJECT_ROOT / "docs" / "reports" / "m5_gate_b_final_audit.md"
    assert audit_report.exists(), "Missing docs/reports/m5_gate_b_final_audit.md"

    required_figures = [
        "m5_overlap_and_seasonal.png",
        "m5_landcover_validation.png",
        "m5_holdout_evaluation.png",
        "m5_sensor_transitions.png",
        "m5_uncertainty_coverage.png"
    ]
    for fig_file in required_figures:
        path = M5_FIGURES_DIR / fig_file
        assert path.exists(), f"Missing required figure: {fig_file}"
        assert path.stat().st_size > 10000, f"Figure {fig_file} is suspiciously small"


def test_m5_test_inventory():
    """Verify Section 4.4 seven-test inventory completeness."""
    df = pd.read_csv(M5_DATA_DIR / "m5_1_test_inventory.csv")
    assert len(df) == 7, f"Expected exactly 7 validation tests, found {len(df)}"
    assert set(df["test_number"]) == {1, 2, 3, 4, 5, 6, 7}
    for col in ["test_name", "question", "design", "pass_rule", "quantitative_result", "status", "verdict_rationale"]:
        assert col in df.columns
        assert df[col].notna().all()


def test_m5_overlap_and_seasonal():
    """Verify overlap-period stability and seasonal profile preservation."""
    df_annual = pd.read_csv(M5_DATA_DIR / "m5_2_overlap_annual_comparison.csv")
    assert len(df_annual) >= 14, "Expected >= 14 overlap years (2012-2025/2026)"
    assert (df_annual["raw_ratio_v_to_m"] > 5.0).all(), "Raw VIIRS/MODIS ratio should exceed 5x"
    assert (df_annual["pred_ratio_exp_to_v"] > 0.5).all()

    df_seasonal = pd.read_csv(M5_DATA_DIR / "m5_2_seasonal_profile.csv")
    assert len(df_seasonal) == 12, "Expected 12 calendar months"
    # Check Pearson r >= 0.85 and peak month is March (month 3)
    r = df_seasonal["pearson_r_exp_viirs"].iloc[0]
    peak_v = df_seasonal["peak_viirs_month"].iloc[0]
    peak_exp = df_seasonal["peak_expected_month"].iloc[0]
    assert r >= 0.85, f"Seasonal correlation {r:.3f} below 0.85 threshold"
    assert peak_v == 3, "Observed VIIRS peak month should be March (3)"
    assert peak_exp == 3, "Expected VIIRS peak month should match March (3)"


def test_m5_landcover_validation():
    """Verify per-class ecological validation and proxy acknowledgment."""
    df = pd.read_csv(M5_DATA_DIR / "m5_3_landcover_metrics.csv")
    assert len(df) == 5, "Expected 5 agro-ecological classes"
    assert df["is_agro_ecological_proxy"].all() is True or (df["is_agro_ecological_proxy"] == True).all(), \
        "All land-cover classes must be identified as agro-ecological proxies"
    
    # Forest & Woodland must be CONDITIONAL
    forest_row = df[df["landcover_class"] == "Forest & Woodland"].iloc[0]
    assert forest_row["validation_status"] == "CONDITIONAL", "Forest & Woodland must be CONDITIONAL"
    assert "power-law" in forest_row["scientific_note"].lower()


def test_m5_holdout_distinction():
    """Verify out-of-time evaluation and pure spatial holdouts are separate experiments."""
    df_temp = pd.read_csv(M5_DATA_DIR / "m5_4_temporal_metrics.csv")
    assert len(df_temp) == 3, "Expected train, validation, and test partitions"
    test_mae = df_temp.loc[df_temp["partition"] == "test", "mae"].values[0]
    val_mae = df_temp.loc[df_temp["partition"] == "validation", "mae"].values[0]
    assert test_mae <= 1.5 * val_mae, "Test MAE exceeds 1.5x validation MAE tolerance"

    df_sp = pd.read_csv(M5_DATA_DIR / "m5_4_pure_spatial_holdout.csv")
    assert len(df_sp) == 6, "Expected 2 holdout blocks x 3 time periods"
    blocks = set(df_sp["spatial_block"])
    assert blocks == {"Block_3_CentralEast_Delta", "Block_4_CentralWest"}
    
    # Delta should have low MAE, Western Ghats higher due to forest runaway
    delta_test = df_sp[(df_sp["spatial_block"] == "Block_3_CentralEast_Delta") & (df_sp["split_period"] == "test")].iloc[0]
    assert delta_test["mae_pure_spatial"] < 1.0


def test_m5_sensor_transitions():
    """Verify Jan 2012 seam reduction and July 2002 limitation."""
    df = pd.read_csv(M5_DATA_DIR / "m5_5_sensor_transitions.csv")
    assert len(df) == 4, "Expected 4 sensor transition seams"
    
    # Jan 2012 seam
    jan_row = df[df["transition"].str.contains("January 2012")].iloc[0]
    assert jan_row["verdict"] == "PASS"
    assert jan_row["harm_pct_shift"] < 50.0
    assert jan_row["raw_pct_shift"] > 1000.0

    # July 2002 seam
    july_row = df[df["transition"].str.contains("July 2002")].iloc[0]
    assert "FAIL" in july_row["verdict"]
    assert "diurnal bridge" in july_row["scientific_caveat"].lower()


def test_m5_uncertainty_coverage():
    """Verify empirical coverage reproduction and active-fire undercoverage."""
    df = pd.read_csv(M5_DATA_DIR / "m5_6_uncertainty_coverage.csv")
    
    overall = df[df["category"] == "Overall (Marginal)"].iloc[0]
    assert overall["coverage_80_pct"] >= 90.0, "Marginal 80% coverage should be >= 90%"
    assert overall["coverage_95_pct"] >= 95.0, "Marginal 95% coverage should be >= 95%"

    active = df[df["category"] == "Active-Fire (V >= 1)"].iloc[0]
    assert active["coverage_80_pct"] < 60.0, "Active-fire 80% coverage should show undercoverage (~54.6%)"
    assert active["coverage_95_pct"] < 80.0, "Active-fire 95% coverage should show undercoverage (~75.7%)"
    assert "FAIL" in active["evaluation_status"]


def test_m5_app_readiness_matrix():
    """Verify that App Readiness Matrix enforces specific go-live permissions."""
    df = pd.read_csv(M5_DATA_DIR / "m5_app_readiness_matrix.csv")
    assert len(df) >= 7, "Expected at least 7 output streams evaluated"
    statuses = set(df["readiness_status"])
    assert "APPROVED FOR RELEASE" in statuses
    assert "BLOCKED / UNVALIDATED" in statuses
    
    # Pre-2002 Terra-only must be blocked / unvalidated
    terra_row = df[df["output_stream"].str.contains("Terra-Only")].iloc[0]
    assert "BLOCKED" in terra_row["readiness_status"]
    assert "NOT VALIDATED" in terra_row["required_app_label"]


def test_m5_gate_b_decision():
    """Verify Gate B pre-registered decision is CONDITIONAL PASS."""
    with open(M5_DATA_DIR / "m5_gate_b_decision.json") as f:
        data = json.load(f)
    assert data["gate"] == "Gate B (Model Validation & Go-Live Authorization)"
    assert data["decision"] == "CONDITIONAL PASS"
    assert "APPROVED FOR MILESTONE M6" in data["next_milestone_readiness"]
    assert len(data["app_authorization"]["authorized_for_release"]) > 0
    assert len(data["app_authorization"]["restricted_or_flagged"]) > 0
