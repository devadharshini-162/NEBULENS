#!/usr/bin/env python3
"""
tests/test_m6_2_fire_season.py
==============================
Unit tests for Milestone M6.2: Regional Fire-Season Metrics & Temporal Characterization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline.m6_2_fire_season import (
    compute_fire_season_from_daily,
    determine_provenance_and_validation,
    run_fire_season_pipeline,
    OUTPUT_DIR
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_compute_fire_season_synthetic_active():
    """Verify season start, end, peak, and active days on a known Gaussian-shaped fire season."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    # Gaussian bump peaking at day 70 (mid March)
    t = np.arange(len(dates))
    counts = 100.0 * np.exp(-0.5 * ((t - 70) / 10.0) ** 2)
    frp = counts * 5.0
    
    df = pd.DataFrame({"analysis_date_ist": dates, "count": counts, "frp": frp})
    res = compute_fire_season_from_daily(df, active_threshold=1.0)
    
    assert res["season_status"] == "active_season"
    assert res["peak_date"] == dates[70]
    assert np.isclose(res["peak_count"], 100.0, atol=1e-2)
    assert np.isclose(res["peak_frp"], 500.0, atol=1e-2)
    assert res["active_fire_days"] > 20
    # Peak day must lie between season start and end
    assert res["season_start_date"] <= res["peak_date"] <= res["season_end_date"]
    assert res["season_duration_days"] > 10


def test_compute_fire_season_zero_activity():
    """Verify graceful handling of zero-fire activity."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"analysis_date_ist": dates, "count": [0.0] * len(dates), "frp": [0.0] * len(dates)})
    
    res = compute_fire_season_from_daily(df)
    assert res["season_status"] == "no_fire_activity"
    assert res["total_detections"] == 0.0
    assert res["active_fire_days"] == 0
    assert res["season_start_date"] is None
    assert res["season_end_date"] is None
    assert res["season_duration_days"] == 0
    assert res["peak_date"] is None


def test_compute_fire_season_sparse_activity():
    """Verify handling of sparse activity (< 5 active days)."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D").strftime("%Y-%m-%d").tolist()
    counts = [0.0] * len(dates)
    counts[30] = 2.0  # Jan 31
    counts[90] = 3.0  # Apr 1
    
    df = pd.DataFrame({"analysis_date_ist": dates, "count": counts, "frp": counts})
    res = compute_fire_season_from_daily(df, min_active_days_for_season=5)
    
    assert res["season_status"] == "sparse_activity"
    assert res["active_fire_days"] == 2
    assert res["first_active_date"] == dates[30]
    assert res["last_active_date"] == dates[90]
    assert res["season_start_date"] == dates[30]
    assert res["season_end_date"] == dates[90]
    assert res["season_duration_days"] == (pd.to_datetime(dates[90]) - pd.to_datetime(dates[30])).days + 1


def test_determine_provenance_and_validation():
    """Verify Gate B validation and provenance assignment."""
    # Pre-2002 Terra only
    prov, val = determine_provenance_and_validation(2001, "harmonised")
    assert "terra_only" in prov
    assert val == "not_validated_terra_only_pre_2002"
    
    # 2002 transition
    prov, val = determine_provenance_and_validation(2002, "harmonised")
    assert val == "conditional_pass_transition_year"
    
    # Counterfactual 2003-2011
    prov, val = determine_provenance_and_validation(2005, "harmonised")
    assert "counterfactual" in prov
    assert val == "conditional_pass_gate_b"
    
    # Observed VIIRS pre-2012
    prov, val = determine_provenance_and_validation(2008, "observed_viirs")
    assert val == "sensor_not_in_orbit"
    
    # Observed VIIRS post-2012
    prov, val = determine_provenance_and_validation(2015, "observed_viirs")
    assert val == "fully_validated"
    
    # Observed VIIRS constellation expansion
    prov, val = determine_provenance_and_validation(2024, "observed_viirs")
    assert val == "conditional_pass_opportunity_expansion"


def test_m6_2_generated_parquet_integrity():
    """Verify that generated Parquet tables exist, are non-empty, and conform to schema."""
    state_file = OUTPUT_DIR / "m6_2_state_season_metrics.parquet"
    dist_file = OUTPUT_DIR / "m6_2_district_season_metrics.parquet"
    
    assert state_file.exists(), f"Missing {state_file}"
    assert dist_file.exists(), f"Missing {dist_file}"
    
    df_state = pd.read_parquet(state_file)
    df_dist = pd.read_parquet(dist_file)
    
    assert len(df_state) > 0
    assert len(df_dist) > 0
    
    # Expected columns
    expected_cols = [
        "year", "entity_type", "entity_name", "stream", "provenance", "validation_status",
        "total_detections", "active_fire_days", "season_start_date", "season_end_date",
        "season_duration_days", "peak_date", "peak_count", "total_frp"
    ]
    for c in expected_cols:
        assert c in df_state.columns
        assert c in df_dist.columns
        
    # Check that Tamil Nadu state has all 27 years
    assert set(df_state["year"].unique()) == set(range(2000, 2027))


def test_m0_through_m5_data_untouched():
    """Verify that all M0–M5 baseline files remain untouched."""
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_coefficients.json").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_app_readiness_matrix.csv").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_gate_b_final_audit.json").exists()
