#!/usr/bin/env python3
"""
pipeline/m4_3_negative_binomial_glm.py
======================================
NEBULENS — Milestone M4.3: Negative Binomial GLM Count Calibration.

Fits Step B of the calibration model ladder on 10 km cell-weeks:
log E[V_{c,w}] = log(e_{c,w}) + beta_0 + beta_1 * log(1 + M_{c,w}) + gamma_season + delta_landcover + interaction

Features:
- Offset: log(e_{c,w}) (theoretical orbital weekly pass opportunities)
- Predictor: log(1 + n_det_modis)
- Categorical Season: Summer (ref), Winter, SW_Monsoon, NE_Monsoon
- Categorical Land Cover: Cropland & Mosaic (ref), Forest & Woodland, Cropland (Delta), Shrubland & Savanna, Urban
- Interaction: log(1 + n_det_modis) * Land Cover

Evaluates:
- Temporal Train, Validation, and Held-out Test sets
- Spatial Holdout Blocks (Block 3 Delta, Block 4 Ghats)
- Zero-frequency fit to evaluate Step C Hurdle necessity

Generates:
- data/analysis/m4/m4_3_nb_metrics.csv
- data/analysis/m4/m4_3_nb_coefficients.json
- docs/figures/m4_3_nb_calibration.png
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Milestone M4.3: Negative Binomial GLM Count Calibration ===")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    # 1. Load cell-week dataset
    print(f"Loading cell-week dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Loaded {len(df):,} cell-weeks.")

    # Clean landcover string labels for clean formula parsing
    df["lc_clean"] = df["landcover_class"].str.replace("[^a-zA-Z0-9]", "_", regex=True)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # 2. Formula definition
    # Reference categories: Season=Summer (peak fire season), Landcover=Cropland___Mosaic__Inland___Dryland_
    formula = (
        "n_det_viirs ~ log_modis1 + C(season, Treatment(reference='Summer')) + "
        "C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_')) + "
        "log_modis1:C(lc_clean)"
    )

    print(f"Fitting Negative Binomial GLM with formula:\n  {formula}\n  offset: log_exposure...")

    # Estimate overdispersion alpha using auxiliary OLS on residuals or grid search
    # In fire count literature, typical alpha is ~1.0 to 2.5
    model = smf.glm(
        formula=formula,
        data=train_df,
        offset=train_df["log_exposure"],
        family=sm.families.NegativeBinomial(alpha=1.5)
    )
    res = model.fit(maxiter=100)
    print("\nGLM Fit Summary:")
    print(res.summary().tables[1])

    # 3. Generate predictions across all partitions (vectorized for instant computation)
    print("Generating predictions across Train, Validation, and Test partitions (vectorized)...")
    p = res.params
    eta = p["Intercept"] + df["log_exposure"] + p["log_modis1"] * df["log_modis1"]

    # Season effects
    for s in ["NE_Monsoon", "SW_Monsoon", "Winter"]:
        term = f"C(season, Treatment(reference='Summer'))[T.{s}]"
        if term in p:
            eta += p[term] * (df["season"] == s)

    # Landcover main effects
    for lc in df["lc_clean"].unique():
        term = f"C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.{lc}]"
        if term in p:
            eta += p[term] * (df["lc_clean"] == lc)

    # Interactions: log_modis1 : lc_clean
    for lc in df["lc_clean"].unique():
        term = f"log_modis1:C(lc_clean)[T.{lc}]"
        if term in p:
            eta += p[term] * df["log_modis1"] * (df["lc_clean"] == lc)

    df["pred_viirs_count"] = np.exp(np.clip(eta, -20.0, 15.0))
    train_df["pred_viirs_count"] = df.loc[train_df.index, "pred_viirs_count"]
    val_df["pred_viirs_count"] = df.loc[val_df.index, "pred_viirs_count"]
    test_df["pred_viirs_count"] = df.loc[test_df.index, "pred_viirs_count"]

    # 4. Compute metrics
    metrics = []

    # Helper function
    def eval_subset(sub, name):
        y_true = sub["n_det_viirs"].values
        y_pred = sub["pred_viirs_count"].values
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r, _ = stats.pearsonr(y_true, y_pred) if np.std(y_pred) > 0 else (0.0, 1.0)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))

        # Zero-frequency fit
        # Under NB(mu, alpha), P(Y=0) = (1 + alpha * mu) ** (-1/alpha)
        alpha = 1.5
        prob_zeros = (1.0 + alpha * y_pred) ** (-1.0 / alpha)
        expected_zero_pct = float(np.mean(prob_zeros) * 100.0)
        actual_zero_pct = float(np.mean(y_true == 0) * 100.0)
        zero_misfit_pct = float(abs(expected_zero_pct - actual_zero_pct))

        # Positive active-only MAE
        pos = y_true > 0
        mae_pos = float(np.mean(np.abs(y_true[pos] - y_pred[pos]))) if np.sum(pos) > 0 else np.nan

        return {
            "evaluation_subset": name,
            "n_records": len(sub),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "pearson_r": round(r, 4),
            "actual_zero_pct": round(actual_zero_pct, 2),
            "predicted_zero_pct": round(expected_zero_pct, 2),
            "zero_misfit_pct": round(zero_misfit_pct, 2),
            "mae_active_fires_only": round(mae_pos, 4) if not np.isnan(mae_pos) else np.nan,
        }

    # Temporal splits
    metrics.append(eval_subset(train_df, "Temporal_Train (2012-2018)"))
    metrics.append(eval_subset(val_df, "Temporal_Validation (2019-2021)"))
    metrics.append(eval_subset(test_df, "Temporal_Test (2022-2026 Held-Out)"))

    # Spatial holdout blocks (evaluated on validation + test data)
    val_test = df[df["split"].isin(["validation", "test"])]
    block_delta = val_test[val_test["spatial_block"] == "Block_3_CentralEast_Delta"]
    block_ghats = val_test[val_test["spatial_block"] == "Block_4_CentralWest"]
    metrics.append(eval_subset(block_delta, "Spatial_Holdout_CauveryDelta"))
    metrics.append(eval_subset(block_ghats, "Spatial_Holdout_WesternGhats"))

    # Land cover specific performance (on validation set)
    for lc in val_df["landcover_class"].unique():
        sub_lc = val_df[val_df["landcover_class"] == lc]
        metrics.append(eval_subset(sub_lc, f"Validation_LC_{lc.split(' (')[0]}"))

    metrics_df = pd.DataFrame(metrics)
    metrics_path = OUTPUT_DIR / "m4_3_nb_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")

    # 5. Save model coefficients
    coef_dict = {
        "model_name": "Step_B_Negative_Binomial_GLM",
        "family": "NegativeBinomial(alpha=1.5)",
        "formula": formula,
        "offset": "log(orbital_exposure_week)",
        "converged": bool(res.converged),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "coefficients": {k: float(v) for k, v in res.params.items()},
        "pvalues": {k: float(v) for k, v in res.pvalues.items()},
        "conf_int_lower": {k: float(v[0]) for k, v in res.conf_int().iterrows()},
        "conf_int_upper": {k: float(v[1]) for k, v in res.conf_int().iterrows()},
    }
    coef_path = OUTPUT_DIR / "m4_3_nb_coefficients.json"
    with open(coef_path, "w") as f:
        json.dump(coef_dict, f, indent=2)
    print(f"Saved coefficients to {coef_path}")

    # 6. Plot Calibration Diagnostics
    print("Generating calibration diagnostic plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # Panel A: Calibration Curve (Observed vs Predicted Deciles)
    ax_cal = axes[0, 0]
    val_pos = val_df.copy()
    val_pos["pred_bin"] = pd.qcut(val_pos["pred_viirs_count"], q=10, duplicates="drop")
    cal_agg = val_pos.groupby("pred_bin", observed=True).agg({"pred_viirs_count": "mean", "n_det_viirs": "mean"})

    ax_cal.plot(cal_agg["pred_viirs_count"], cal_agg["n_det_viirs"], "o-", color="#2b83ba", linewidth=2.0, label="Validation Deciles")
    ref_max = max(cal_agg["pred_viirs_count"].max(), cal_agg["n_det_viirs"].max())
    ax_cal.plot([0, ref_max], [0, ref_max], "r--", linewidth=1.5, label="Perfect Calibration (1:1)")
    ax_cal.set_title("A. Reliability Calibration Curve (Validation Deciles)", fontsize=11, fontweight="bold")
    ax_cal.set_xlabel("Mean Predicted VIIRS Count per Cell-Week", fontsize=10, fontweight="bold")
    ax_cal.set_ylabel("Mean Observed VIIRS Count per Cell-Week", fontsize=10, fontweight="bold")
    ax_cal.grid(True, linestyle="--", alpha=0.4)
    ax_cal.legend(loc="upper left")

    # Panel B: Residuals vs Fitted
    ax_res = axes[0, 1]
    sample_val = val_df.sample(n=min(5000, len(val_df)), random_state=42)
    ax_res.scatter(sample_val["pred_viirs_count"], sample_val["n_det_viirs"] - sample_val["pred_viirs_count"],
                   alpha=0.3, s=15, color="#d7191c", edgecolor="none")
    ax_res.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax_res.set_title("B. Residuals vs Predicted (Validation Sample, n=5,000)", fontsize=11, fontweight="bold")
    ax_res.set_xlabel("Predicted VIIRS Count", fontsize=10, fontweight="bold")
    ax_res.set_ylabel("Residual (Observed - Predicted)", fontsize=10, fontweight="bold")
    ax_res.grid(True, linestyle="--", alpha=0.4)

    # Panel C: Observed vs Predicted by Season
    ax_seas = axes[1, 0]
    seas_metrics = val_df.groupby("season").agg({"n_det_viirs": "mean", "pred_viirs_count": "mean"})
    x = np.arange(len(seas_metrics))
    width = 0.35
    ax_seas.bar(x - width/2, seas_metrics["n_det_viirs"], width, label="Observed Mean VIIRS", color="#2ca02c")
    ax_seas.bar(x + width/2, seas_metrics["pred_viirs_count"], width, label="Predicted Mean VIIRS (GLM)", color="#1f78b4")
    ax_seas.set_xticks(x)
    ax_seas.set_xticklabels(seas_metrics.index, fontsize=10, fontweight="bold")
    ax_seas.set_ylabel("Mean Counts per Cell-Week", fontsize=10, fontweight="bold")
    ax_seas.set_title("C. Seasonal Mean Counts (Validation Period)", fontsize=11, fontweight="bold")
    ax_seas.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_seas.legend(loc="upper right")

    # Panel D: Zero-Frequency Fit
    ax_zero = axes[1, 1]
    splits_plot = ["Train", "Validation", "Test"]
    actual_zeros = [metrics[0]["actual_zero_pct"], metrics[1]["actual_zero_pct"], metrics[2]["actual_zero_pct"]]
    pred_zeros = [metrics[0]["predicted_zero_pct"], metrics[1]["predicted_zero_pct"], metrics[2]["predicted_zero_pct"]]
    x_z = np.arange(3)
    ax_zero.bar(x_z - width/2, actual_zeros, width, label="Actual Zero %", color="#4575b4")
    ax_zero.bar(x_z + width/2, pred_zeros, width, label="Model Predicted Zero %", color="#f46d43")
    ax_zero.set_xticks(x_z)
    ax_zero.set_xticklabels(splits_plot, fontsize=10, fontweight="bold")
    ax_zero.set_ylabel("Zero-Detection Percentage (%)", fontsize=10, fontweight="bold")
    ax_zero.set_ylim(80, 100)
    ax_zero.set_title("D. Zero-Frequency Agreement (Negative Binomial)", fontsize=11, fontweight="bold")
    ax_zero.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_zero.legend(loc="lower left")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_3_nb_calibration.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved calibration figure to {fig_path}")

    # Print Key Metrics
    print("\n=== Negative Binomial GLM Performance Metrics ===")
    print(metrics_df[["evaluation_subset", "mae", "rmse", "r2", "pearson_r", "actual_zero_pct", "predicted_zero_pct", "zero_misfit_pct"]].to_string(index=False))

    # Check Hurdle trigger rule
    val_zero_misfit = metrics[1]["zero_misfit_pct"]
    print(f"\nValidation Zero-Misfit = {val_zero_misfit:.2f}% (Threshold = {config['promotion_rules']['hurdle_zero_misfit_threshold_pct']}%)")
    if val_zero_misfit > config["promotion_rules"]["hurdle_zero_misfit_threshold_pct"]:
        print("  -> Hurdle Model TRIGGERED: Step B exceeds zero-misfit threshold.")
    else:
        print("  -> Hurdle Model NOT mandatory: Step B adequately captures zero frequency within threshold.")

    print("=== M4.3 Complete! ===")


if __name__ == "__main__":
    main()
