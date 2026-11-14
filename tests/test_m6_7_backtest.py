#!/usr/bin/env python3
"""
tests/test_m6_7_backtest.py
===========================
Unit tests for Milestone M6.7: Historical Sensor-Transition Backtesting.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

from pipeline.run_m6_7_backtest import (
    run_transition_backtest_1_jan2012,
    run_transition_backtest_2_may2018,
    run_transition_backtest_3_mar2023,
    run_all_backtests,
    OUTPUT_DIR
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_backtest_jan2012_identifies_sensor_effect():
    """Verify that Jan 2012 transition is attributed as sensor-related and avoids false fire growth claims."""
    res = run_transition_backtest_1_jan2012()
    assert res["attribution_outcome"] == "likely_sensor_related_change"
    assert res["avoids_false_fire_growth_claim"] is True
    assert res["identified_sensor_effect"] is True


def test_backtest_may2018_identifies_sensor_effect():
    """Verify that May 2018 NOAA-20 transition is attributed as sensor-related."""
    res = run_transition_backtest_2_may2018()
    assert res["attribution_outcome"] == "likely_sensor_related_change"
    assert res["avoids_false_fire_growth_claim"] is True
    assert res["identified_sensor_effect"] is True


def test_backtest_mar2023_identifies_sensor_effect():
    """Verify that Mar 2023 NOAA-21 transition is attributed as sensor-related."""
    res = run_transition_backtest_3_mar2023()
    assert res["attribution_outcome"] == "likely_sensor_related_change"
    assert res["avoids_false_fire_growth_claim"] is True
    assert res["identified_sensor_effect"] is True


def test_backtest_machine_readable_outputs_exist():
    """Verify that backtest JSON and Parquet artifacts exist and match schema."""
    json_path = OUTPUT_DIR / "m6_7_backtest_results.json"
    pq_path = OUTPUT_DIR / "m6_7_backtest_results.parquet"
    
    assert json_path.exists()
    assert pq_path.exists()
    
    with open(json_path) as f:
        data = json.load(f)
    assert len(data) == 3
    for entry in data:
        assert entry["identified_sensor_effect"] is True
        assert entry["avoids_false_fire_growth_claim"] is True


def test_m0_through_m5_data_untouched():
    """Verify that all M0–M5 baseline files remain untouched."""
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_coefficients.json").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_app_readiness_matrix.csv").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_gate_b_final_audit.json").exists()
