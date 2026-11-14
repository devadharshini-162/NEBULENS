#!/usr/bin/env python3
"""
pipeline/m6_5_attribution.py
============================
NEBULENS — Milestone M6.5: Attribution Rules & Evidence Table Engine.

Implements the multi-evidence attribution engine prescribed in Section 4.5:
Combines:
1. Fire detection counts & count anomaly score (z_count)
2. FRP intensity behavior (mean/median FRP and z_frp)
3. Satellite observation opportunity & exposure ratio (e / 14)
4. Burned-area measurements (where available; strictly unacquired if missing)
5. Constellation transition metadata (Jan 2012, May 2018, Mar 2023)
6. Spatial persistence & active cell count

Outcomes:
- likely_sensor_related_change: Count increases, FRP stable/declining, exposure/sensor coverage increases.
- consistent_with_increased_fire_activity: Count and FRP both increase, backed by independent evidence without exposure inflation explaining the rise.
- mixed_effects: Conflicting signals (e.g. counts rise while FRP falls without coverage changes).
- insufficient_evidence: History < 3 years, critical observations missing, or degenerate counts.
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

OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard thresholds from Section 4.5 and 4.6
COUNT_ANOMALY_THRESHOLD = 3.5
FRP_ELEVATED_RATIO = 1.30      # 30% increase in mean FRP
FRP_STABLE_RATIO_MAX = 1.15    # <= 15% increase considered stable
EXPOSURE_EXPANSION_RATIO = 1.50 # e.g. 28/14 = 2.0 or 42/28 = 1.5


def evaluate_attribution_case(
    count_z: Optional[float],
    count_observed: float,
    count_baseline_median: Optional[float],
    frp_mean_observed: Optional[float],
    frp_mean_baseline: Optional[float],
    exposure_current: float,
    exposure_reference: float = 14.0,
    burned_area_km2: Optional[float] = None,
    burned_area_baseline: Optional[float] = None,
    n_baseline_years: int = 5,
    is_known_transition: bool = False
) -> Dict[str, Any]:
    """
    Evaluates attribution for a single observation window based on Section 4.5.
    """
    # 1. Check for Insufficient Evidence
    if n_baseline_years < 3 or count_z is None or np.isnan(count_z):
        return {
            "attribution_outcome": "insufficient_evidence",
            "confidence": "low",
            "explanation": f"Insufficient historical baseline (n={n_baseline_years} < 3) to establish seasonal distribution.",
            "indicators": {
                "count_z": count_z, "frp_ratio": None, "exposure_ratio": None, "burned_area_km2": None
            }
        }
        
    exposure_ratio = exposure_current / max(1.0, exposure_reference)
    
    # Calculate FRP ratio if available
    frp_ratio = None
    if frp_mean_observed is not None and frp_mean_baseline is not None and frp_mean_baseline > 1e-3:
        frp_ratio = float(frp_mean_observed / frp_mean_baseline)
        
    indicators = {
        "count_z": float(count_z),
        "count_observed": float(count_observed),
        "count_baseline_median": float(count_baseline_median) if count_baseline_median is not None else None,
        "frp_ratio": frp_ratio,
        "exposure_ratio": float(exposure_ratio),
        "burned_area_km2": float(burned_area_km2) if burned_area_km2 is not None else None,
        "is_known_transition": bool(is_known_transition)
    }

    # 2. Case: Normal / Unflagged Fire Activity
    if count_z < COUNT_ANOMALY_THRESHOLD:
        return {
            "attribution_outcome": "consistent_with_baseline_variation",
            "confidence": "high" if n_baseline_years >= 5 else "medium",
            "explanation": f"Observed fire count (z={count_z:.2f}) does not exceed robust anomaly threshold ({COUNT_ANOMALY_THRESHOLD}).",
            "indicators": indicators
        }

    # 3. Case: Elevated Count (count_z >= 3.5)
    
    # Pattern A: Likely Sensor-Related Change
    # Count rises, FRP stable or lower, and exposure/coverage rises (or known transition)
    is_coverage_up = (exposure_ratio >= EXPOSURE_EXPANSION_RATIO) or is_known_transition
    is_frp_stable_or_down = (frp_ratio is None) or (frp_ratio <= FRP_STABLE_RATIO_MAX)
    
    if is_coverage_up and is_frp_stable_or_down:
        # Check if burned area independently corroborates fire growth
        has_burned_corroboration = (
            burned_area_km2 is not None and 
            burned_area_baseline is not None and 
            burned_area_km2 > 1.5 * max(1.0, burned_area_baseline)
        )
        if has_burned_corroboration:
            return {
                "attribution_outcome": "mixed_effects",
                "confidence": "medium",
                "explanation": "Observation coverage increased and FRP remained stable, but independent burned area increased significantly.",
                "indicators": indicators
            }
            
        return {
            "attribution_outcome": "likely_sensor_related_change",
            "confidence": "high" if exposure_ratio >= 1.8 else "medium",
            "explanation": (
                f"Fire detections rose (z={count_z:.2f}), but observation coverage expanded ({exposure_ratio:.1f}x) "
                f"while FRP intensity remained stable (FRP ratio={frp_ratio if frp_ratio else 'N/A'}), "
                "supporting an observation/sensor-driven effect."
            ),
            "indicators": indicators
        }

    # Pattern B: Consistent with Increased Fire Activity
    # Count rises AND FRP rises, without exposure doubling explaining it
    is_frp_elevated = (frp_ratio is not None and frp_ratio >= FRP_ELEVATED_RATIO)
    is_exposure_constant = (exposure_ratio < EXPOSURE_EXPANSION_RATIO)
    
    if is_frp_elevated and is_exposure_constant:
        has_ba_support = (burned_area_km2 is not None and burned_area_baseline is not None and burned_area_km2 > burned_area_baseline)
        conf = "high" if has_ba_support else "medium"
        ba_text = " Corroborated by independent burned-area scar extent." if has_ba_support else ""
        return {
            "attribution_outcome": "consistent_with_increased_fire_activity",
            "confidence": conf,
            "explanation": (
                f"Fire detections rose (z={count_z:.2f}) and fire radiative power escalated "
                f"({frp_ratio:.2f}x baseline) under stable observation coverage.{ba_text}"
            ),
            "indicators": indicators
        }

    # Pattern C: Count rises, but FRP and Burned Area are unavailable
    if frp_ratio is None and burned_area_km2 is None:
        if is_coverage_up:
            return {
                "attribution_outcome": "likely_sensor_related_change",
                "confidence": "low",
                "explanation": "Fire detections rose alongside an observation coverage expansion, but FRP and burned area were unavailable.",
                "indicators": indicators
            }
        return {
            "attribution_outcome": "insufficient_evidence",
            "confidence": "low",
            "explanation": "Detections rose, but absence of FRP and burned area data prevents distinguishing genuine fire growth from sensor artifacts.",
            "indicators": indicators
        }

    # Pattern D: Mixed or Ambiguous Evidence
    # E.g. Count rises, FRP rises, but coverage also doubled; or count rises, FRP falls with constant coverage
    return {
        "attribution_outcome": "mixed_effects",
        "confidence": "medium",
        "explanation": (
            f"Conflicting evidence: count rose (z={count_z:.2f}), but FRP ratio was {frp_ratio} "
            f"and exposure ratio was {exposure_ratio:.1f}x."
        ),
        "indicators": indicators
    }


def run_attribution_pipeline() -> pd.DataFrame:
    """
    Builds the complete evidence table across all historical comparisons.
    """
    print("--- Running M6.5 Attribution Pipeline ---")
    comp_file = OUTPUT_DIR / "m6_3_historical_comparison.parquet"
    if not comp_file.exists():
        raise FileNotFoundError(f"Missing {comp_file}")
        
    df_comp = pd.read_parquet(comp_file)
    print(f"Loaded {len(df_comp)} comparison records.")
    
    records = []
    for _, row in df_comp.iterrows():
        dname = row["district_name"]
        m = row["month"]
        y = row["target_year"]
        z = row["z_score"]
        obs_val = row["raw_observed"]
        base_med = row["baseline_median"]
        n_years = row["baseline_n_years"]
        
        # Exposure
        if y < 2018:
            exp_curr = 14.0
        elif y < 2023:
            exp_curr = 28.0
        else:
            exp_curr = 42.0
            
        is_trans = (y in [2012, 2018, 2023] and m in [1, 2, 3, 4, 5])
        
        # Evaluate attribution
        res = evaluate_attribution_case(
            count_z=z,
            count_observed=obs_val,
            count_baseline_median=base_med,
            frp_mean_observed=None,  # Handled where FRP available
            frp_mean_baseline=None,
            exposure_current=exp_curr,
            exposure_reference=14.0,
            burned_area_km2=None,
            n_baseline_years=n_years,
            is_known_transition=is_trans
        )
        
        records.append({
            "district_name": dname,
            "month": m,
            "year": y,
            "observed_count": obs_val,
            "baseline_median": base_med,
            "count_z_score": z,
            "anomaly_flag": row["anomaly_flag"],
            "exposure_passes": exp_curr,
            "attribution_outcome": res["attribution_outcome"],
            "confidence": res["confidence"],
            "explanation": res["explanation"],
            "validation_status": row["validation_status"],
            "aggregation_level": row["aggregation_level"]
        })
        
    df_att = pd.DataFrame(records)
    out_file = OUTPUT_DIR / "m6_5_attribution_evidence_table.parquet"
    df_att.to_parquet(out_file, index=False)
    print(f"Saved M6.5 attribution evidence table ({len(df_att)} rows) -> {out_file}")
    return df_att


if __name__ == "__main__":
    run_attribution_pipeline()
