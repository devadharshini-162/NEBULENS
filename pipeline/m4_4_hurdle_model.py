#!/usr/bin/env python3
"""
pipeline/m4_4_hurdle_model.py
=============================
NEBULENS — Milestone M4.4: Hurdle Negative Binomial Count Model.

Evaluates and fits Step C of the model ladder:
1. Binary Logistic Regression Gate: P(V_{c,w} > 0)
2. Conditional Count Model: E[V_{c,w} | V_{c,w} > 0]
Combined Expected Value: E[V_{c,w}] = P(V > 0) * E[V | V > 0]

Triggered because Step B Negative Binomial GLM had a 4.88% zero-frequency misfit (threshold = 2.0%).

Evaluates promotion criteria:
- Does Step C reduce validation zero-misfit to <= 2.0%?
- Does Step C improve validation MAE by >= 3.0% over Step B?

Generates:
- data/analysis/m4/m4_4_hurdle_metrics.csv
- data/analysis/m4/m4_4_hurdle_coefficients.json
- docs/figures/m4_4_hurdle_comparison.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m4"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4_calibration_config.json"
NB_METRICS_PATH = OUTPUT_DIR / "m4_3_nb_metrics.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Milestone M4.4: Hurdle Negative Binomial Model ===")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    df = pd.read_parquet(DATASET_PATH)
    df["lc_clean"] = df["landcover_class"].str.replace("[^a-zA-Z0-9]", "_", regex=True)
    df["is_active"] = (df["n_det_viirs"] > 0).astype(int)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Part 1: Binary Logistic Gate
    print("Fitting Part 1: Binary Logistic Gate for P(V > 0)...")
    gate_formula = (
        "is_active ~ log_modis1 + log_exposure + "
        "C(season, Treatment(reference='Summer')) + "
        "C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))"
    )
    gate_model = smf.logit(formula=gate_formula, data=train_df).fit(disp=False)

    # Part 2: Truncated Count Model on Active Weeks (V >= 1)
    print("Fitting Part 2: Count Model on Active Weeks (V >= 1)...")
    train_active = train_df[train_df["is_active"] == 1].copy()
    count_formula = (
        "n_det_viirs ~ log_modis1 + "
        "C(season, Treatment(reference='Summer')) + "
        "C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_')) + "
        "log_modis1:C(lc_clean)"
    )
    count_model = smf.glm(
        formula=count_formula,
        data=train_active,
        offset=train_active["log_exposure"],
        family=sm.families.NegativeBinomial(alpha=1.0)
    ).fit(disp=False)

    # Fast Vectorized Predictions
    print("Computing Hurdle predictions across splits (vectorized)...")
    p_gate = gate_model.params
    gate_eta = p_gate["Intercept"] + p_gate["log_modis1"] * df["log_modis1"] + p_gate["log_exposure"] * df["log_exposure"]
    for s in ["NE_Monsoon", "SW_Monsoon", "Winter"]:
        term = f"C(season, Treatment(reference='Summer'))[T.{s}]"
        if term in p_gate:
            gate_eta += p_gate[term] * (df["season"] == s)
    for lc in df["lc_clean"].unique():
        term = f"C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.{lc}]"
        if term in p_gate:
            gate_eta += p_gate[term] * (df["lc_clean"] == lc)

    prob_active = 1.0 / (1.0 + np.exp(-np.clip(gate_eta, -25.0, 25.0)))

    # Count expectation
    p_cnt = count_model.params
    cnt_eta = p_cnt["Intercept"] + df["log_exposure"] + p_cnt["log_modis1"] * df["log_modis1"]
    for s in ["NE_Monsoon", "SW_Monsoon", "Winter"]:
        term = f"C(season, Treatment(reference='Summer'))[T.{s}]"
        if term in p_cnt:
            cnt_eta += p_cnt[term] * (df["season"] == s)
    for lc in df["lc_clean"].unique():
        term = f"C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.{lc}]"
        if term in p_cnt:
            cnt_eta += p_cnt[term] * (df["lc_clean"] == lc)
    for lc in df["lc_clean"].unique():
        term = f"log_modis1:C(lc_clean)[T.{lc}]"
        if term in p_cnt:
            cnt_eta += p_cnt[term] * df["log_modis1"] * (df["lc_clean"] == lc)

    mu_active = np.exp(np.clip(cnt_eta, -20.0, 15.0))
    # Hurdle prediction: E[V] = P(V > 0) * E[V | V > 0]
    df["pred_hurdle_count"] = prob_active * mu_active
    df["pred_prob_zero"] = 1.0 - prob_active

    train_df["pred_hurdle_count"] = df.loc[train_df.index, "pred_hurdle_count"]
    val_df["pred_hurdle_count"] = df.loc[val_df.index, "pred_hurdle_count"]
    test_df["pred_hurdle_count"] = df.loc[test_df.index, "pred_hurdle_count"]
    val_df["pred_prob_zero"] = df.loc[val_df.index, "pred_prob_zero"]

    # Compute Metrics
    metrics = []
    for split_name, s_df in [("Temporal_Train (2012-2018)", train_df),
                             ("Temporal_Validation (2019-2021)", val_df),
                             ("Temporal_Test (2022-2026 Held-Out)", test_df)]:
        y_true = s_df["n_det_viirs"].values
        y_pred = s_df["pred_hurdle_count"].values
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r, _ = stats.pearsonr(y_true, y_pred) if np.std(y_pred) > 0 else (0.0, 1.0)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))

        pred_zero_pct = float(np.mean(1.0 - prob_active.loc[s_df.index]) * 100.0)
        actual_zero_pct = float(np.mean(y_true == 0) * 100.0)
        zero_misfit_pct = float(abs(pred_zero_pct - actual_zero_pct))

        metrics.append({
            "model_step": "Step_C_Hurdle_Negative_Binomial",
            "evaluation_subset": split_name,
            "n_records": len(s_df),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "pearson_r": round(r, 4),
            "actual_zero_pct": round(actual_zero_pct, 2),
            "predicted_zero_pct": round(pred_zero_pct, 2),
            "zero_misfit_pct": round(zero_misfit_pct, 2)
        })

    hurdle_metrics_df = pd.DataFrame(metrics)
    hurdle_metrics_path = OUTPUT_DIR / "m4_4_hurdle_metrics.csv"
    hurdle_metrics_df.to_csv(hurdle_metrics_path, index=False)
    print(f"Saved hurdle metrics to {hurdle_metrics_path}")

    # Compare with Step B
    nb_metrics = pd.read_csv(NB_METRICS_PATH)
    nb_val = nb_metrics[nb_metrics["evaluation_subset"].str.contains("Validation")].iloc[0]
    hc_val = hurdle_metrics_df[hurdle_metrics_df["evaluation_subset"].str.contains("Validation")].iloc[0]

    val_mae_step_b = nb_val["mae"]
    val_mae_step_c = hc_val["mae"]
    mae_improvement_pct = ((val_mae_step_b - val_mae_step_c) / val_mae_step_b) * 100.0
    val_zero_misfit_step_c = hc_val["zero_misfit_pct"]

    print("\n=== Step B vs Step C Comparison on Validation Partition ===")
    print(f"Step B (Negative Binomial GLM): MAE = {val_mae_step_b:.4f}, Zero-Misfit = {nb_val['zero_misfit_pct']:.2f}%")
    print(f"Step C (Hurdle Negative Binomial): MAE = {val_mae_step_c:.4f}, Zero-Misfit = {val_zero_misfit_step_c:.2f}%")
    print(f"MAE Improvement = {mae_improvement_pct:.2f}% (Promotion Threshold = {config['promotion_rules']['hurdle_min_mae_improvement_pct']}%)")

    promoted = (val_zero_misfit_step_c <= config["promotion_rules"]["hurdle_zero_misfit_threshold_pct"]) and \
               (mae_improvement_pct >= config["promotion_rules"]["hurdle_min_mae_improvement_pct"])

    print(f"\nStep C Hurdle Promotion Decision: {'PROMOTED' if promoted else 'REJECTED (Retaining Step B)'}")

    # Save summary
    res_summary = {
        "step_B_val_mae": float(val_mae_step_b),
        "step_C_val_mae": float(val_mae_step_c),
        "mae_improvement_pct": float(mae_improvement_pct),
        "step_B_val_zero_misfit_pct": float(nb_val["zero_misfit_pct"]),
        "step_C_val_zero_misfit_pct": float(val_zero_misfit_step_c),
        "promoted_over_step_B": bool(promoted)
    }
    with open(OUTPUT_DIR / "m4_4_hurdle_coefficients.json", "w") as f:
        json.dump(res_summary, f, indent=2)

    # Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    ax1 = axes[0]
    ax1.bar(["Step B (NB GLM)", "Step C (Hurdle)"], [val_mae_step_b, val_mae_step_c], color=["#1f78b4", "#33a02c"], width=0.5)
    ax1.set_title("Validation MAE: Step B vs Step C", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    ax2 = axes[1]
    ax2.bar(["Step B (NB GLM)", "Step C (Hurdle)"], [nb_val["zero_misfit_pct"], val_zero_misfit_step_c], color=["#e31a1c", "#ff7f00"], width=0.5)
    ax2.axhline(config["promotion_rules"]["hurdle_zero_misfit_threshold_pct"], color="black", linestyle="--", label="Tolerance (2.0%)")
    ax2.set_title("Zero-Frequency Misfit (%): Step B vs Step C", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Absolute Misfit (%)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_4_hurdle_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved hurdle comparison figure to {fig_path}")
    print("=== M4.4 Complete! ===")


if __name__ == "__main__":
    main()
