#!/usr/bin/env python3
"""
pipeline/run_m6_7_backtest.py
=============================
NEBULENS — Milestone M6.7: Historical Sensor-Transition Backtesting.

Backtests the completed attribution engine across the three major satellite transitions:
1. January 2012: MODIS (C6.1) to VIIRS (Suomi-NPP) resolution transition (1 km to 375 m).
2. May 2018: NOAA-20 constellation expansion (14 to 28 passes/week).
3. March 2023: NOAA-21 constellation expansion (28 to 42 passes/week).

Verification Objectives:
- Verifies that the engine reports 'likely_sensor_related_change' where available evidence indicates observation shifts.
- Verifies that the engine strictly avoids incorrectly claiming that sensor transitions represent real ground fire surges.
- Preserves unresolved ambiguities rather than forcing classifications.
- Outputs machine-readable results to data/analysis/m6/m6_7_backtest_results.json and .parquet.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.m6_5_attribution import evaluate_attribution_case
from pipeline.m6_1_anomaly_engine import compute_robust_z_score

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_transition_backtest_1_jan2012() -> Dict[str, Any]:
    """
    Backtest 1: January 2012 MODIS-to-VIIRS Transition.
    
    If unharmonised raw counts were naively compared, VIIRS (11,762 fires in 2012)
    vs MODIS (798 fires in 2011) would show a +1,374% jump.
    The attribution engine must identify the sensor change and flag it as sensor-related.
    """
    # Raw comparison (MODIS baseline vs VIIRS target)
    raw_modis_baseline = [798.0, 850.0, 920.0, 780.0, 810.0]  # prior years MODIS
    raw_viirs_obs_2012 = 11762.0
    
    # Anomaly score on uncalibrated jump
    z_raw = compute_robust_z_score(raw_viirs_obs_2012, raw_modis_baseline)["z_score"]
    
    # Evaluate attribution on raw transition
    att_raw = evaluate_attribution_case(
        count_z=z_raw,
        count_observed=raw_viirs_obs_2012,
        count_baseline_median=np.median(raw_modis_baseline),
        frp_mean_observed=35.0,  # VIIRS captures many smaller fires -> mean FRP drops/stable
        frp_mean_baseline=65.0,  # MODIS only captures larger fires -> mean FRP higher
        exposure_current=14.0,
        exposure_reference=14.0,
        n_baseline_years=len(raw_modis_baseline),
        is_known_transition=True  # Sensor transition window
    )
    
    return {
        "transition_id": "T1_JAN_2012",
        "description": "MODIS (1 km) to VIIRS Suomi-NPP (375 m) sensor transition",
        "transition_date": "2012-01-20",
        "raw_count_jump_pct": float((raw_viirs_obs_2012 - np.median(raw_modis_baseline)) / np.median(raw_modis_baseline) * 100.0),
        "raw_z_score": float(z_raw),
        "attribution_outcome": att_raw["attribution_outcome"],
        "confidence": att_raw["confidence"],
        "explanation": att_raw["explanation"],
        "avoids_false_fire_growth_claim": (att_raw["attribution_outcome"] != "consistent_with_increased_fire_activity"),
        "identified_sensor_effect": (att_raw["attribution_outcome"] == "likely_sensor_related_change")
    }


def run_transition_backtest_2_may2018() -> Dict[str, Any]:
    """
    Backtest 2: May 2018 NOAA-20 Constellation Expansion.
    
    Weekly orbital view opportunities doubled from 14 passes/week to 28 passes/week.
    Observed VIIRS increased from 7,142 (2017) to 10,387 (2018) and 25,097 (2019).
    """
    snpp_baseline = [7349.0, 8100.0, 7800.0, 7142.0, 7500.0]  # 2013-2017 baseline (exposure = 14)
    obs_2018 = 10387.0  # partial transition year
    obs_2019 = 25097.0  # full 2-satellite year (exposure = 28)
    
    z_raw_2019 = compute_robust_z_score(obs_2019, snpp_baseline)["z_score"]
    
    att_2019 = evaluate_attribution_case(
        count_z=z_raw_2019,
        count_observed=obs_2019,
        count_baseline_median=np.median(snpp_baseline),
        frp_mean_observed=38.0,
        frp_mean_baseline=36.0,  # frp_ratio = 1.05 (stable FRP)
        exposure_current=28.0,   # Doubled exposure
        exposure_reference=14.0,
        n_baseline_years=len(snpp_baseline),
        is_known_transition=True
    )
    
    return {
        "transition_id": "T2_MAY_2018",
        "description": "NOAA-20 VIIRS operational addition (exposure 14 to 28 passes/week)",
        "transition_date": "2018-05-01",
        "raw_count_jump_pct": float((obs_2019 - np.median(snpp_baseline)) / np.median(snpp_baseline) * 100.0),
        "raw_z_score": float(z_raw_2019),
        "attribution_outcome": att_2019["attribution_outcome"],
        "confidence": att_2019["confidence"],
        "explanation": att_2019["explanation"],
        "avoids_false_fire_growth_claim": (att_2019["attribution_outcome"] != "consistent_with_increased_fire_activity"),
        "identified_sensor_effect": (att_2019["attribution_outcome"] == "likely_sensor_related_change")
    }


def run_transition_backtest_3_mar2023() -> Dict[str, Any]:
    """
    Backtest 3: March 2023 NOAA-21 Constellation Expansion.
    
    Weekly orbital view opportunities expanded from 28 to 42 passes/week (3x over Suomi-NPP baseline).
    Observed VIIRS rose from 13,355 (2022) to 27,291 (2023) and 39,453 (2024).
    """
    baseline_2018_2022 = [10387.0, 25097.0, 15000.0, 16000.0, 13355.0]  # exposure = 28
    obs_2023 = 27291.0
    obs_2024 = 39453.0  # 3-satellite year (exposure = 42)
    
    z_raw_2024 = compute_robust_z_score(obs_2024, baseline_2018_2022)["z_score"]
    
    att_2024 = evaluate_attribution_case(
        count_z=z_raw_2024,
        count_observed=obs_2024,
        count_baseline_median=np.median(baseline_2018_2022),
        frp_mean_observed=36.0,
        frp_mean_baseline=37.0,  # FRP stable
        exposure_current=42.0,   # 42 passes/week vs 14 or 28
        exposure_reference=28.0, # 1.5x increase over 2-sat baseline
        n_baseline_years=len(baseline_2018_2022),
        is_known_transition=True
    )
    
    return {
        "transition_id": "T3_MAR_2023",
        "description": "NOAA-21 VIIRS operational addition (exposure 28 to 42 passes/week)",
        "transition_date": "2023-03-01",
        "raw_count_jump_pct": float((obs_2024 - np.median(baseline_2018_2022)) / np.median(baseline_2018_2022) * 100.0),
        "raw_z_score": float(z_raw_2024),
        "attribution_outcome": att_2024["attribution_outcome"],
        "confidence": att_2024["confidence"],
        "explanation": att_2024["explanation"],
        "avoids_false_fire_growth_claim": (att_2024["attribution_outcome"] != "consistent_with_increased_fire_activity"),
        "identified_sensor_effect": (att_2024["attribution_outcome"] == "likely_sensor_related_change")
    }


def run_all_backtests() -> List[Dict[str, Any]]:
    print("=== Running M6.7 Sensor-Transition Backtest Suite ===")
    t1 = run_transition_backtest_1_jan2012()
    t2 = run_transition_backtest_2_may2018()
    t3 = run_transition_backtest_3_mar2023()
    
    results = [t1, t2, t3]
    for r in results:
        print(f"[{r['transition_id']}] {r['description']}")
        print(f"  Outcome: {r['attribution_outcome']} (Confidence: {r['confidence']})")
        print(f"  Avoids false fire growth claim: {r['avoids_false_fire_growth_claim']}")
        print(f"  Identified sensor effect: {r['identified_sensor_effect']}")
        print(f"  Note: {r['explanation'][:100]}...\n")
        
    # Save machine readable outputs
    json_path = OUTPUT_DIR / "m6_7_backtest_results.json"
    pq_path = OUTPUT_DIR / "m6_7_backtest_results.parquet"
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    df = pd.DataFrame(results)
    df.to_parquet(pq_path, index=False)
    
    print(f"Saved machine-readable backtest results to {json_path} and {pq_path}")
    return results


if __name__ == "__main__":
    run_all_backtests()
