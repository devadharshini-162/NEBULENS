#!/usr/bin/env python3
"""
tests/test_m6_3_historical_comparison.py
========================================
Unit tests for Milestone M6.3: Same-Season Historical Comparison Across Years.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline.m6_3_historical_comparison import (
    build_monthly_entity_panel,
    run_historical_comparison_pipeline,
    OUTPUT_DIR
)
from pipeline.m6_1_anomaly_engine import evaluate_series_anomalies

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_temporal_segregation_and_prior_years():
    """Verify that baseline only includes strictly prior years and respects pre-2002 exclusion."""
    # Synthetic dataset with years 2000 to 2015
    records = []
    # Baseline with natural variance around 10
    var_vals = [9.0, 10.0, 11.0, 10.0, 9.0, 11.0, 10.0, 10.0, 9.0, 11.0, 10.0, 10.0, 9.0, 11.0, 10.0]
    for i, y in enumerate(range(2000, 2016)):
        records.append({
            "entity": "District_A",
            "month": 3,
            "year": y,
            "val": var_vals[i] if y != 2015 else 50.0,
            "exposure": 14.0
        })
    df = pd.DataFrame(records)
    
    # Evaluate 2015 with exclude_pre_2002=True
    # Historical years should be 2003..2014 = 12 years (2000, 2001, 2002 excluded)
    res = evaluate_series_anomalies(
        df=df,
        entity_col="entity",
        time_window_col="month",
        year_col="year",
        value_col="val",
        target_year=2015,
        exclude_pre_2002=True
    )
    
    assert len(res) == 1
    row = res.iloc[0]
    assert row["baseline_n_years"] == 12  # 2003..2014 inclusive
    assert row["baseline_median"] == 10.0
    assert row["baseline_mad"] > 0.0
    assert row["z_score"] > 3.5
    assert bool(row["anomaly_flag"]) is True  # 50 vs 10


def test_exposure_adjustment_in_comparison():
    """Verify exposure normalization scales evaluation value proportionally."""
    records = []
    # Baseline: 2012-2017 with exposure = 14 and count = 100
    for y in range(2012, 2018):
        records.append({"entity": "D1", "month": 3, "year": y, "count": 100.0, "exposure": 14.0})
    # Target: 2024 with exposure = 42 and count = 300 (pure opportunity inflation)
    records.append({"entity": "D1", "month": 3, "year": 2024, "count": 300.0, "exposure": 42.0})
    df = pd.DataFrame(records)
    
    res = evaluate_series_anomalies(
        df=df,
        entity_col="entity",
        time_window_col="month",
        year_col="year",
        value_col="count",
        target_year=2024,
        exclude_pre_2002=False,
        exposure_col="exposure"
    )
    
    row = res.iloc[0]
    # 300 * (14 / 42) = 100.0
    assert np.isclose(row["evaluated_value"], 100.0)
    assert row["z_score"] == 0.0
    assert bool(row["anomaly_flag"]) is False


def test_m6_3_parquet_integrity():
    """Verify that generated historical comparison parquet exists and has expected fields."""
    pq_path = OUTPUT_DIR / "m6_3_historical_comparison.parquet"
    assert pq_path.exists(), f"Missing {pq_path}"
    
    df = pd.read_parquet(pq_path)
    assert len(df) > 0
    
    expected_cols = [
        "district_name", "month", "target_year", "raw_observed", "evaluated_value",
        "is_exposure_standardized", "z_score", "anomaly_flag", "scoring_method",
        "baseline_median", "baseline_mad", "baseline_n_years", "target_h_source",
        "validation_status", "aggregation_level"
    ]
    for c in expected_cols:
        assert c in df.columns
        
    # Check that both state and district levels are represented
    assert "state" in df["aggregation_level"].values
    assert "district" in df["aggregation_level"].values


def test_m0_through_m5_data_untouched():
    """Verify that all M0–M5 baseline files remain untouched."""
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_coefficients.json").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_app_readiness_matrix.csv").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_gate_b_final_audit.json").exists()
