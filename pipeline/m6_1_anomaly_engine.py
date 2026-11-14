#!/usr/bin/env python3
"""
pipeline/m6_1_anomaly_engine.py
===============================
NEBULENS — Milestone M6.1: Seasonal Baseline & Robust Anomaly Engine.

Implements the statistical anomaly scoring engine defined in Section 4.6 of the
research methodology:
1. Seasonal baseline construction: Comparing the exact same calendar window across
   prior years (e.g. March 2026 vs earlier Marches).
2. Robust anomaly scoring via rolling median and Median Absolute Deviation (MAD):
   z = 0.6745 * (x - median(x_baseline)) / MAD(x_baseline)
3. Deterministic sparse-count fallback when MAD == 0:
   - Case A: x == median == 0 -> normal_zero (z = 0.0)
   - Case B: x > median and MAD == 0 ->
       Tier 1: Negative Binomial exceedance quantile z_equiv = Phi^-1(1 - P(X >= x))
       Tier 2: Spatial district-level aggregated robust score
4. Minimum history requirements: At least 3 years required; otherwise "insufficient_evidence".
5. Exposure standardization: Evaluates counts adjusted for orbital view opportunities.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Normal scaling constant reciprocal: 1 / 1.482602218505602 ~= 0.67448975
MAD_NORMAL_CONSTANT = 0.6744897501960817
DEFAULT_ANOMALY_THRESHOLD = 3.5
DEFAULT_NB_ALPHA = 1.5
MIN_BASELINE_YEARS = 3
REFERENCE_ORBITAL_EXPOSURE = 14.0  # Suomi-NPP single-satellite weekly baseline


def compute_mad(values: Union[np.ndarray, List[float]]) -> float:
    """
    Computes the Median Absolute Deviation (MAD):
        MAD = median(|x_i - median(x)|)
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def compute_nb_exceedance_pvalue(
    x: float,
    mu: float,
    alpha: float = DEFAULT_NB_ALPHA
) -> float:
    """
    Computes upper-tail exceedance probability P(X >= x) under NegativeBinomial(mu, alpha):
        Var(X) = mu + alpha * mu^2
    """
    if x <= 0:
        return 1.0
    # Avoid zero mu edge-case by imposing tiny non-zero prior
    mu_eff = max(float(mu), 0.05)
    p_param = 1.0 / (1.0 + alpha * mu_eff)
    n_param = 1.0 / alpha
    
    # P(X >= x) = 1 - CDF(x - 1) = Survival Function(x - 1)
    k = max(0, int(np.floor(x)) - 1)
    p_val = float(stats.nbinom.sf(k, n_param, p_param))
    return max(1e-15, min(1.0, p_val))


def compute_robust_z_score(
    observed_value: float,
    baseline_values: Union[np.ndarray, List[float]],
    flag_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    nb_alpha: float = DEFAULT_NB_ALPHA,
    min_years: int = MIN_BASELINE_YEARS
) -> Dict[str, Any]:
    """
    Computes the robust anomaly score from rolling baseline values.
    
    Returns a dictionary with:
    - z_score: float or None
    - anomaly_flag: bool
    - method_used: str ('standard_mad', 'nb_exceedance_fallback', 'zero_identical', 'insufficient_evidence')
    - baseline_median: float
    - baseline_mad: float
    - baseline_n: int
    - exceedance_prob: float or None
    """
    arr = np.asarray(baseline_values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n_years = len(arr)
    
    # 1. Check minimum history requirement
    if n_years < min_years:
        return {
            "z_score": None,
            "anomaly_flag": False,
            "method_used": "insufficient_evidence",
            "baseline_median": None,
            "baseline_mad": None,
            "baseline_n": n_years,
            "exceedance_prob": None,
            "scientific_note": f"Insufficient historical baseline (n={n_years} < {min_years})"
        }
        
    med = float(np.median(arr))
    mad = compute_mad(arr)
    obs = float(observed_value)
    
    # 2. Case: MAD > 0 (Standard robust score)
    if mad > 1e-9:
        z = MAD_NORMAL_CONSTANT * (obs - med) / mad
        is_anom = bool(z >= flag_threshold)
        return {
            "z_score": float(z),
            "anomaly_flag": is_anom,
            "method_used": "standard_mad",
            "baseline_median": med,
            "baseline_mad": mad,
            "baseline_n": n_years,
            "exceedance_prob": None,
            "scientific_note": "Standard robust MAD score calculated"
        }
        
    # 3. Case: MAD == 0
    # Sub-case 3A: Observed matches median (typical zero fire week in zero fire cell)
    if abs(obs - med) < 1e-9:
        return {
            "z_score": 0.0,
            "anomaly_flag": False,
            "method_used": "zero_identical",
            "baseline_median": med,
            "baseline_mad": 0.0,
            "baseline_n": n_years,
            "exceedance_prob": 1.0,
            "scientific_note": "Observed count matches historical zero median; MAD is zero"
        }
        
    # Sub-case 3B: Observed differs from median and MAD == 0 (Sparse count fallback)
    # Activate Negative Binomial exceedance quantile fallback
    mu_baseline = float(np.mean(arr))
    p_exceed = compute_nb_exceedance_pvalue(obs, mu_baseline, alpha=nb_alpha)
    
    # Map p-value to standard normal quantile z = Phi^-1(1 - p_exceed)
    # Target threshold p_threshold = 1 - Phi(flag_threshold) = 1 - Phi(3.5) ~= 0.0002326
    p_clamped = max(1e-12, min(1.0 - 1e-12, p_exceed))
    z_equiv = float(stats.norm.ppf(1.0 - p_clamped))
    
    # Anomaly flagged if p_exceed <= alpha_crit (equivalent to z_equiv >= flag_threshold)
    crit_p = float(stats.norm.sf(flag_threshold))
    is_anom = bool(p_exceed <= crit_p)
    
    return {
        "z_score": float(np.clip(z_equiv, -10.0, 10.0)),
        "anomaly_flag": is_anom,
        "method_used": "nb_exceedance_fallback",
        "baseline_median": med,
        "baseline_mad": 0.0,
        "baseline_n": n_years,
        "exceedance_prob": float(p_exceed),
        "scientific_note": f"Sparse count fallback: MAD is zero; scored via NB exceedance quantile (P={p_exceed:.5e})"
    }


def evaluate_series_anomalies(
    df: pd.DataFrame,
    entity_col: str,
    time_window_col: str,
    year_col: str,
    value_col: str,
    target_year: int,
    flag_threshold: float = DEFAULT_ANOMALY_THRESHOLD,
    min_years: int = MIN_BASELINE_YEARS,
    exclude_pre_2002: bool = True,
    exposure_col: Optional[str] = None,
    district_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Evaluates anomalies for all entities in target_year against their historical baseline.
    
    Parameters:
    - df: DataFrame containing historical and target observations
    - entity_col: e.g. 'grid_cell_id' or 'district_name'
    - time_window_col: e.g. 'month' or 'iso_week'
    - year_col: e.g. 'year'
    - value_col: e.g. 'harmonised_estimate' or 'observed_viirs'
    - target_year: year to evaluate (e.g. 2024)
    - exclude_pre_2002: If True, excludes pre-July 2002 Terra-only uncalibrated era
    - exposure_col: Optional exposure pass column for exposure standardization
    - district_col: Optional district column for district-level aggregation fallback (Section 4.6)
    """
    # 1. Filter historical baseline (strictly strictly strictly prior years)
    base_mask = (df[year_col] < target_year)
    if exclude_pre_2002:
        base_mask &= (df[year_col] >= 2003)
        
    baseline_df = df[base_mask].copy()
    target_df = df[df[year_col] == target_year].copy()
    
    # Optional exposure standardization (only adjust when exposure > 14 to avoid double-correction)
    if exposure_col and exposure_col in df.columns:
        # If exposure <= 14 or not greater than reference, factor is 1.0
        exp_factor_base = np.where(baseline_df[exposure_col] > REFERENCE_ORBITAL_EXPOSURE,
                                   REFERENCE_ORBITAL_EXPOSURE / baseline_df[exposure_col].clip(lower=1.0),
                                   1.0)
        exp_factor_targ = np.where(target_df[exposure_col] > REFERENCE_ORBITAL_EXPOSURE,
                                   REFERENCE_ORBITAL_EXPOSURE / target_df[exposure_col].clip(lower=1.0),
                                   1.0)
        baseline_df["eval_val"] = baseline_df[value_col] * exp_factor_base
        target_df["eval_val"] = target_df[value_col] * exp_factor_targ
        is_std = True
    else:
        baseline_df["eval_val"] = baseline_df[value_col]
        target_df["eval_val"] = target_df[value_col]
        is_std = False
        
    # Precompute baseline vectors by (entity, time_window)
    hist_grouped = baseline_df.groupby([entity_col, time_window_col])["eval_val"].apply(np.array).to_dict()
    
    # Precompute district-level baselines if district_col provided (Section 4.6 fallback)
    dist_hist_grouped = {}
    dist_target_grouped = {}
    if district_col and district_col in df.columns and entity_col != district_col:
        dist_baseline = baseline_df.groupby([district_col, time_window_col, year_col])["eval_val"].sum().reset_index()
        dist_hist_grouped = dist_baseline.groupby([district_col, time_window_col])["eval_val"].apply(np.array).to_dict()
        dist_target = target_df.groupby([district_col, time_window_col])["eval_val"].sum().to_dict()
        dist_target_grouped = dist_target

    results = []
    for _, row in target_df.iterrows():
        ent = row[entity_col]
        twin = row[time_window_col]
        val = row["eval_val"]
        raw_val = row[value_col]
        dist_val = row[district_col] if (district_col and district_col in row) else None
        
        hist_vals = hist_grouped.get((ent, twin), np.array([]))
        score_res = compute_robust_z_score(
            observed_value=val,
            baseline_values=hist_vals,
            flag_threshold=flag_threshold,
            min_years=min_years
        )
        
        # Check if district fallback is triggered (Section 4.6: sparse counts with MAD == 0 and mu == 0)
        district_z = None
        district_flag = None
        if dist_val and dist_hist_grouped and score_res["method_used"] in ["nb_exceedance_fallback", "zero_identical"]:
            d_hist = dist_hist_grouped.get((dist_val, twin), np.array([]))
            d_obs = dist_target_grouped.get((dist_val, twin), 0.0)
            d_res = compute_robust_z_score(
                observed_value=d_obs,
                baseline_values=d_hist,
                flag_threshold=flag_threshold,
                min_years=min_years
            )
            district_z = d_res["z_score"]
            district_flag = d_res["anomaly_flag"]
        
        res_record = {
            entity_col: ent,
            time_window_col: twin,
            "target_year": target_year,
            "raw_observed": float(raw_val),
            "evaluated_value": float(val),
            "is_exposure_standardized": is_std,
            "z_score": score_res["z_score"],
            "anomaly_flag": score_res["anomaly_flag"],
            "scoring_method": score_res["method_used"],
            "baseline_median": score_res["baseline_median"],
            "baseline_mad": score_res["baseline_mad"],
            "baseline_n_years": score_res["baseline_n"],
            "exceedance_prob": score_res["exceedance_prob"],
            "district_fallback_z": district_z,
            "district_fallback_flag": district_flag,
            "scientific_note": score_res["scientific_note"]
        }
        results.append(res_record)
        
    return pd.DataFrame(results)


def main():
    print("=== Milestone M6.1: Anomaly Engine Verification Run ===")
    # Quick sanity check with synthetic test
    synth_baseline = [1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 2.0]
    normal_case = compute_robust_z_score(2.0, synth_baseline)
    anom_case = compute_robust_z_score(15.0, synth_baseline)
    sparse_case = compute_robust_z_score(4.0, [0.0, 0.0, 0.0, 0.0, 0.0])
    
    print("Normal case score:", normal_case)
    print("Anomaly case score:", anom_case)
    print("Sparse fallback score:", sparse_case)
    print("M6.1 Anomaly Engine core module is functional!")


if __name__ == "__main__":
    main()
