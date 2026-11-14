#!/usr/bin/env python3
"""
tests/test_m6_6_attribution_synthetic.py
========================================
Synthetic unit tests for Milestone M6.6: Attribution Rules Engine.

Tests the 8 mandatory pre-registered synthetic scenarios from the methodology:
1. Detection counts increase, FRP stable, coverage increases -> likely_sensor_related_change
2. Detection counts increase, FRP increases -> consistent_with_increased_fire_activity
3. Detection counts increase but FRP and burned area unavailable -> insufficient_evidence / qualified
4. Detection counts and FRP disagree -> mixed_effects
5. Exposure increases while underlying activity remains constant -> no false fire-activity attribution
6. All relevant observations are missing -> insufficient_evidence
7. Zero-heavy historical distributions -> correct sparse-count handling
8. Genuine seasonal fire-activity increase -> correct historical anomaly interpretation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline.m6_5_attribution import evaluate_attribution_case
from pipeline.m6_1_anomaly_engine import compute_robust_z_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_scenario_1_count_up_frp_stable_coverage_up():
    """Scenario 1: Detections increase, FRP stable, coverage increases -> likely sensor-related."""
    res = evaluate_attribution_case(
        count_z=5.2,
        count_observed=250.0,
        count_baseline_median=50.0,
        frp_mean_observed=30.0,
        frp_mean_baseline=30.0,  # frp_ratio = 1.0 (stable)
        exposure_current=28.0,
        exposure_reference=14.0,  # exposure_ratio = 2.0 (doubled coverage)
        n_baseline_years=7
    )
    assert res["attribution_outcome"] == "likely_sensor_related_change"
    assert res["confidence"] == "high"
    assert "observation coverage expanded" in res["explanation"]


def test_scenario_2_count_up_frp_up_corroborated():
    """Scenario 2: Detections and FRP both increase under constant coverage -> consistent with fire activity."""
    res = evaluate_attribution_case(
        count_z=4.8,
        count_observed=200.0,
        count_baseline_median=50.0,
        frp_mean_observed=65.0,
        frp_mean_baseline=35.0,  # frp_ratio = 1.86 (elevated)
        exposure_current=14.0,
        exposure_reference=14.0,  # constant coverage
        burned_area_km2=25.0,
        burned_area_baseline=5.0,  # independent corroboration
        n_baseline_years=8
    )
    assert res["attribution_outcome"] == "consistent_with_increased_fire_activity"
    assert res["confidence"] == "high"
    assert "Corroborated by independent burned-area" in res["explanation"]


def test_scenario_3_count_up_frp_and_burned_area_unavailable():
    """Scenario 3: Detections increase but FRP and burned area are unavailable -> insufficient evidence."""
    res = evaluate_attribution_case(
        count_z=4.1,
        count_observed=120.0,
        count_baseline_median=40.0,
        frp_mean_observed=None,
        frp_mean_baseline=None,
        exposure_current=14.0,
        exposure_reference=14.0,
        burned_area_km2=None,
        n_baseline_years=5
    )
    assert res["attribution_outcome"] == "insufficient_evidence"
    assert res["confidence"] == "low"
    assert "absence of FRP and burned area" in res["explanation"]


def test_scenario_4_count_and_frp_disagree():
    """Scenario 4: Detections increase, FRP collapses with constant coverage -> mixed effects."""
    res = evaluate_attribution_case(
        count_z=4.5,
        count_observed=180.0,
        count_baseline_median=45.0,
        frp_mean_observed=10.0,
        frp_mean_baseline=40.0,  # frp_ratio = 0.25 (collapsed)
        exposure_current=14.0,
        exposure_reference=14.0,  # constant coverage
        n_baseline_years=6
    )
    assert res["attribution_outcome"] == "mixed_effects"
    assert res["confidence"] == "medium"
    assert "Conflicting evidence" in res["explanation"]


def test_scenario_5_exposure_increases_underlying_activity_constant():
    """Scenario 5: Exposure increases while underlying activity is constant -> no false fire activity attribution."""
    # Count doubles solely due to satellite pass doubling: standardized z is zero
    res = evaluate_attribution_case(
        count_z=0.0,  # Adjusted z is 0
        count_observed=100.0,
        count_baseline_median=50.0,
        frp_mean_observed=30.0,
        frp_mean_baseline=30.0,
        exposure_current=28.0,
        exposure_reference=14.0,
        n_baseline_years=6
    )
    assert res["attribution_outcome"] == "consistent_with_baseline_variation"
    assert "does not exceed robust anomaly threshold" in res["explanation"]


def test_scenario_6_missing_observations_insufficient_evidence():
    """Scenario 6: Baseline history < 3 years -> insufficient evidence."""
    res = evaluate_attribution_case(
        count_z=None,
        count_observed=100.0,
        count_baseline_median=None,
        frp_mean_observed=None,
        frp_mean_baseline=None,
        exposure_current=14.0,
        n_baseline_years=1  # Only 1 prior year
    )
    assert res["attribution_outcome"] == "insufficient_evidence"
    assert res["confidence"] == "low"
    assert "Insufficient historical baseline" in res["explanation"]


def test_scenario_7_zero_heavy_sparse_handling():
    """Scenario 7: Zero-heavy historical baseline correctly handled via NB exceedance."""
    # Historical counts are all zero (MAD = 0)
    score_res = compute_robust_z_score(
        observed_value=8.0,
        baseline_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        flag_threshold=3.5
    )
    assert score_res["method_used"] == "nb_exceedance_fallback"
    assert score_res["baseline_mad"] == 0.0
    assert score_res["z_score"] is not None
    # Feed into attribution
    res = evaluate_attribution_case(
        count_z=score_res["z_score"],
        count_observed=8.0,
        count_baseline_median=0.0,
        frp_mean_observed=45.0,
        frp_mean_baseline=20.0,
        exposure_current=14.0,
        n_baseline_years=6
    )
    assert res["attribution_outcome"] in ["consistent_with_increased_fire_activity", "mixed_effects", "consistent_with_baseline_variation"]


def test_scenario_8_genuine_seasonal_fire_surge():
    """Scenario 8: Genuine seasonal surge with elevated counts, elevated FRP, and constant exposure."""
    res = evaluate_attribution_case(
        count_z=6.5,
        count_observed=500.0,
        count_baseline_median=80.0,
        frp_mean_observed=80.0,
        frp_mean_baseline=40.0,  # frp_ratio = 2.0 (doubled intensity)
        exposure_current=14.0,
        exposure_reference=14.0,
        n_baseline_years=10
    )
    assert res["attribution_outcome"] == "consistent_with_increased_fire_activity"
    assert res["confidence"] in ["medium", "high"]
    assert "fire radiative power escalated" in res["explanation"]
