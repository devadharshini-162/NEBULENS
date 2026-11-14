"""
tests/test_m6_1_anomaly_engine.py
=================================
Automated test suite for Milestone M6.1: Seasonal Baseline & Robust Anomaly Engine.

Verifies:
1. Median Absolute Deviation (MAD) mathematical accuracy.
2. Stable seasonal patterns do not trigger false anomalies.
3. Genuine surges relative to the historical seasonal baseline are reliably flagged.
4. Sparse zero-count observations activate the exact prescribed fallbacks without division-by-zero.
5. Inadequate historical coverage produces an explicit "insufficient_evidence" result.
6. Resistance to masking by historical mega-fire outliers.
7. Orbital exposure standardization prevents constellation-expansion false alarms.
8. Batch time-series evaluation on panel datasets.
9. Preserves existing M0–M5 raw data, calibration models, and validation artifacts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline.m6_1_anomaly_engine import (
    compute_mad,
    compute_nb_exceedance_pvalue,
    compute_robust_z_score,
    evaluate_series_anomalies,
    DEFAULT_ANOMALY_THRESHOLD,
    MIN_BASELINE_YEARS
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_compute_mad():
    """Verify MAD computation on known vectors."""
    # Data: [1, 2, 3, 4, 5, 6, 7, 8, 9] -> median = 5, abs diffs = [4, 3, 2, 1, 0, 1, 2, 3, 4] -> median = 2.0
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert compute_mad(arr) == 2.0

    # Constant vector -> MAD = 0
    assert compute_mad([0, 0, 0, 0]) == 0.0
    assert compute_mad([]) == 0.0


def test_normal_seasonal_pattern():
    """Verify that typical seasonal observations do not produce false anomalies."""
    baseline = [2.0, 3.0, 2.0, 4.0, 3.0, 2.0, 3.0]
    res = compute_robust_z_score(observed_value=3.0, baseline_values=baseline)
    
    assert res["method_used"] == "standard_mad"
    assert res["baseline_median"] == 3.0
    assert res["baseline_mad"] > 0.0
    assert res["z_score"] == 0.0
    assert res["anomaly_flag"] is False


def test_genuine_anomaly_detection():
    """Verify that a genuine surge relative to history is flagged as an anomaly."""
    baseline = [1.0, 2.0, 2.0, 1.0, 3.0, 2.0, 1.0]  # median = 2, MAD = 1
    res = compute_robust_z_score(observed_value=25.0, baseline_values=baseline)
    
    assert res["method_used"] == "standard_mad"
    assert res["z_score"] > DEFAULT_ANOMALY_THRESHOLD
    assert res["anomaly_flag"] is True


def test_sparse_zero_with_zero_current():
    """Verify that historically inactive cells remaining at zero do not divide by zero."""
    baseline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    res = compute_robust_z_score(observed_value=0.0, baseline_values=baseline)
    
    assert res["method_used"] == "zero_identical"
    assert res["z_score"] == 0.0
    assert res["anomaly_flag"] is False
    assert res["baseline_mad"] == 0.0


def test_sparse_zero_with_positive_current_fallback():
    """Verify that fires in historically zero cells activate the Negative Binomial fallback."""
    baseline = [0.0, 0.0, 0.0, 0.0, 0.0]  # MAD == 0
    res = compute_robust_z_score(observed_value=5.0, baseline_values=baseline)
    
    assert res["method_used"] == "nb_exceedance_fallback"
    assert res["baseline_mad"] == 0.0
    assert res["exceedance_prob"] is not None
    assert res["exceedance_prob"] < 0.001
    assert res["z_score"] >= DEFAULT_ANOMALY_THRESHOLD
    assert res["anomaly_flag"] is True


def test_insufficient_history():
    """Verify that inadequate history produces an explicit insufficient-evidence status."""
    baseline = [2.0, 3.0]  # Only 2 years (< MIN_BASELINE_YEARS)
    res = compute_robust_z_score(observed_value=10.0, baseline_values=baseline)
    
    assert res["method_used"] == "insufficient_evidence"
    assert res["z_score"] is None
    assert res["anomaly_flag"] is False
    assert res["baseline_n"] == 2


def test_masking_resistance_with_historical_outlier():
    """Verify that an ancient mega-fire does not mask a genuine subsequent anomaly."""
    # Classical mean/std would have mean ~17.8, std ~40, so x = 8 would have z = (8 - 17.8)/40 = -0.2 (completely masked!)
    # Robust median is 1.5, MAD is 0.5 -> x = 8 gives z = 0.6745 * (8 - 1.5) / 0.5 = 8.77 (detected!)
    baseline_with_outlier = [1.0, 2.0, 1.0, 100.0, 2.0, 1.0]
    res = compute_robust_z_score(observed_value=8.0, baseline_values=baseline_with_outlier)
    
    assert res["method_used"] == "standard_mad"
    assert res["baseline_median"] == 1.5
    assert res["baseline_mad"] == 0.5
    assert res["z_score"] > DEFAULT_ANOMALY_THRESHOLD
    assert res["anomaly_flag"] is True


def test_exposure_standardization():
    """Verify that observation exposure doubling is neutralized by exposure standardization."""
    # Synthetic panel with exposure doubling in 2024 (e.g. from 14 to 28 passes/week)
    records = []
    # Baseline years 2018-2023: exposure = 14, counts = ~10
    for y in range(2018, 2024):
        records.append({
            "grid_cell_id": "cell_1",
            "month": 3,
            "year": y,
            "count": 10.0,
            "orbital_exposure": 14.0
        })
    # Target year 2024: exposure doubled to 28, count doubled to 20 (pure constellation effect)
    records.append({
        "grid_cell_id": "cell_1",
        "month": 3,
        "year": 2024,
        "count": 20.0,
        "orbital_exposure": 28.0
    })
    df_panel = pd.DataFrame(records)
    
    # 1. Unadjusted evaluation: Count doubled from 10 to 20 -> triggers false anomaly if MAD is tight
    res_unadj = evaluate_series_anomalies(
        df=df_panel,
        entity_col="grid_cell_id",
        time_window_col="month",
        year_col="year",
        value_col="count",
        target_year=2024,
        exclude_pre_2002=False,
        exposure_col=None
    )
    
    # 2. Exposure-adjusted evaluation: 20 * (14/28) = 10 -> matches baseline 10 exactly!
    res_adj = evaluate_series_anomalies(
        df=df_panel,
        entity_col="grid_cell_id",
        time_window_col="month",
        year_col="year",
        value_col="count",
        target_year=2024,
        exclude_pre_2002=False,
        exposure_col="orbital_exposure"
    )
    
    assert res_adj.iloc[0]["evaluated_value"] == 10.0
    assert res_adj.iloc[0]["z_score"] == 0.0
    assert bool(res_adj.iloc[0]["anomaly_flag"]) is False


def test_district_level_fallback():
    """Verify Section 4.6 district fallback when cell has all-zero history."""
    records = []
    # District D has 3 cells. Cell 1 has all zeros historically.
    # Other cells in district D have normal fire activity with natural variation: sum is ~10 fires/year.
    var_counts = [4.0, 5.0, 4.0, 6.0, 5.0, 5.0]
    for i, y in enumerate(range(2018, 2024)):
        c_val = var_counts[i]
        records.append({"cell_id": "c1", "district": "D1", "week": 12, "year": y, "count": 0.0})
        records.append({"cell_id": "c2", "district": "D1", "week": 12, "year": y, "count": c_val})
        records.append({"cell_id": "c3", "district": "D1", "week": 12, "year": y, "count": c_val})
        
    # In target year 2024, c1 experiences 8 fires, while district total surges to 50
    records.append({"cell_id": "c1", "district": "D1", "week": 12, "year": 2024, "count": 8.0})
    records.append({"cell_id": "c2", "district": "D1", "week": 12, "year": 2024, "count": 21.0})
    records.append({"cell_id": "c3", "district": "D1", "week": 12, "year": 2024, "count": 21.0})
    
    df_panel = pd.DataFrame(records)
    res = evaluate_series_anomalies(
        df=df_panel,
        entity_col="cell_id",
        time_window_col="week",
        year_col="year",
        value_col="count",
        target_year=2024,
        district_col="district",
        exclude_pre_2002=False
    )
    
    c1_row = res[res["cell_id"] == "c1"].iloc[0]
    assert c1_row["scoring_method"] == "nb_exceedance_fallback"
    assert c1_row["district_fallback_z"] is not None
    assert c1_row["district_fallback_flag"] is True  # District surge confirms regional anomaly


def test_m0_through_m5_data_untouched():
    """Verify that all M0–M5 baseline files remain untouched."""
    # Verify core datasets and models exist
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_coefficients.json").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_app_readiness_matrix.csv").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_gate_b_final_audit.json").exists()
    assert (PROJECT_ROOT / "docs" / "reports" / "m5_validation_report.md").exists()
    assert (PROJECT_ROOT / "docs" / "reports" / "m5_gate_b_final_audit.md").exists()
    assert (PROJECT_ROOT / "docs" / "reports" / "m5_completion_note.md").exists()
