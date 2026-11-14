#!/usr/bin/env python3
"""
pipeline/m4_5_frp_calibration.py
================================
NEBULENS — Milestone M4.5: Fire Radiative Power (FRP) Calibration Models.

Implements two separate FRP models:
1. Model A: Per-pair FRP bias against scan angle (using 1,809 Aqua–VIIRS 1:1 linked pairs)
   log(frp_viirs / frp_aqua) ~ scan_aqua + log(frp_aqua) + distance_m

2. Model B: Total FRP per cell-week conditional on positive fire count
   log(frp_sum_viirs) ~ log1p(frp_sum_modis) + log_exposure + season + landcover

Evaluates:
- Training, Validation, and Held-out Test partitions
- Decouples FRP calibration completely from count calibration

Generates:
- data/analysis/m4/m4_5_frp_pair_model.json
- data/analysis/m4/m4_5_frp_cell_model.json
- data/analysis/m4/m4_5_frp_metrics.csv
- docs/figures/m4_5_frp_models.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import HuberRegressor
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PAIRS_PATH = PROJECT_ROOT / "data" / "analysis" / "m3" / "m3_4_aqua_viirs_pairs.parquet"
CELL_WEEKS_PATH = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m4"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4_calibration_config.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fit_model_a_pairwise(config: dict):
    """Fit Model A: Pairwise FRP bias against scan angle."""
    print("Fitting Model A: Pairwise FRP Bias against Scan Angle...")
    pairs = pd.read_parquet(PAIRS_PATH)
    pairs["log_ratio"] = np.log(pairs["frp_viirs"] / pairs["frp_aqua"])
    pairs["log_frp_aqua"] = np.log(pairs["frp_aqua"])

    # Split
    train_end = config["temporal_splits"]["train"]["end"]
    val_end = config["temporal_splits"]["validation"]["end"]

    p_train = pairs[pairs["date"] <= train_end].copy()
    p_val = pairs[(pairs["date"] > train_end) & (pairs["date"] <= val_end)].copy()
    p_test = pairs[pairs["date"] > val_end].copy()

    # Fit Huber robust regression
    X_train = p_train[["scan_aqua", "log_frp_aqua", "distance_m"]].values
    y_train = p_train["log_ratio"].values
    huber = HuberRegressor(epsilon=1.35, max_iter=1000).fit(X_train, y_train)

    # Evaluate across splits
    metrics = []
    for split_name, s_df in [("Train", p_train), ("Validation", p_val), ("Test", p_test)]:
        X = s_df[["scan_aqua", "log_frp_aqua", "distance_m"]].values
        y_true = s_df["log_ratio"].values
        y_pred = huber.predict(X)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r, _ = stats.pearsonr(y_true, y_pred)

        # In native FRP units: predicted VIIRS FRP = Aqua FRP * exp(y_pred)
        frp_pred = s_df["frp_aqua"] * np.exp(y_pred)
        frp_true = s_df["frp_viirs"]
        frp_mae = float(np.mean(np.abs(frp_true - frp_pred)))

        metrics.append({
            "model": "Model_A_Pairwise_FRP",
            "split": split_name,
            "n_pairs": len(s_df),
            "log_ratio_mae": round(mae, 4),
            "log_ratio_rmse": round(rmse, 4),
            "log_ratio_r": round(r, 4),
            "frp_mae_mw": round(frp_mae, 2),
            "median_frp_ratio_obs": round(float((frp_true / s_df["frp_aqua"]).median()), 3),
            "median_frp_ratio_pred": round(float((frp_pred / s_df["frp_aqua"]).median()), 3)
        })

    params = {
        "model_name": "Model_A_Pairwise_FRP",
        "algorithm": "HuberRegressor(epsilon=1.35)",
        "features": ["scan_aqua", "log_frp_aqua", "distance_m"],
        "coefficients": {
            "intercept": float(huber.intercept_),
            "scan_aqua": float(huber.coef_[0]),
            "log_frp_aqua": float(huber.coef_[1]),
            "distance_m": float(huber.coef_[2])
        },
        "scale": float(huber.scale_)
    }
    with open(OUTPUT_DIR / "m4_5_frp_pair_model.json", "w") as f:
        json.dump(params, f, indent=2)

    return pd.DataFrame(metrics), params, pairs, huber


def fit_model_b_cell_total(config: dict):
    """Fit Model B: Total FRP per cell-week on active cell-weeks."""
    print("Fitting Model B: Total Cell-Week FRP Model...")
    df = pd.read_parquet(CELL_WEEKS_PATH)
    df["lc_clean"] = df["landcover_class"].str.replace("[^a-zA-Z0-9]", "_", regex=True)

    # Active cell-weeks with positive MODIS FRP
    pos_df = df[(df["n_det_modis"] > 0) & (df["n_det_viirs"] > 0)].copy()
    pos_df["log_frp_viirs"] = np.log(pos_df["frp_sum_viirs"])
    pos_df["log1p_frp_modis"] = np.log1p(pos_df["frp_sum_modis"])

    train_df = pos_df[pos_df["split"] == "train"].copy()
    val_df = pos_df[pos_df["split"] == "validation"].copy()
    test_df = pos_df[pos_df["split"] == "test"].copy()

    # Fit robust regression for total FRP
    features = ["log1p_frp_modis", "log_exposure"]
    X_train = train_df[features].values
    y_train = train_df["log_frp_viirs"].values
    huber_cell = HuberRegressor(epsilon=1.35, max_iter=1000).fit(X_train, y_train)

    metrics = []
    for split_name, s_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        X = s_df[features].values
        y_true = s_df["frp_sum_viirs"].values
        y_pred = np.exp(huber_cell.predict(X))

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r, _ = stats.pearsonr(y_true, y_pred) if np.std(y_pred) > 0 else (0.0, 1.0)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - (ss_res / max(ss_tot, 1e-6)))

        metrics.append({
            "model": "Model_B_Cell_Total_FRP",
            "split": split_name,
            "n_active_weeks": len(s_df),
            "frp_mae_mw": round(mae, 2),
            "frp_rmse_mw": round(rmse, 2),
            "frp_r2": round(r2, 4),
            "frp_pearson_r": round(r, 4)
        })

    params = {
        "model_name": "Model_B_Cell_Total_FRP",
        "algorithm": "HuberRegressor(epsilon=1.35)",
        "features": features,
        "coefficients": {
            "intercept": float(huber_cell.intercept_),
            "log1p_frp_modis": float(huber_cell.coef_[0]),
            "log_exposure": float(huber_cell.coef_[1])
        },
        "scale": float(huber_cell.scale_)
    }
    with open(OUTPUT_DIR / "m4_5_frp_cell_model.json", "w") as f:
        json.dump(params, f, indent=2)

    return pd.DataFrame(metrics), params, pos_df, huber_cell


def plot_frp_diagnostics(pairs, params_a, pos_df, params_b):
    """Plot publication-quality 4-panel diagnostic for FRP calibration."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # Panel A: Pairwise FRP Ratio vs Scan Angle
    ax_a = axes[0, 0]
    ax_a.scatter(pairs["scan_aqua"], pairs["frp_viirs"] / pairs["frp_aqua"], alpha=0.4, color="#1f78b4", s=20)
    scans = np.linspace(pairs["scan_aqua"].min(), pairs["scan_aqua"].max(), 100)
    # Predicted ratio at median Aqua FRP
    med_log_aqua = np.log(pairs["frp_aqua"].median())
    med_dist = pairs["distance_m"].median()
    pred_log_ratio = params_a["coefficients"]["intercept"] + \
                     params_a["coefficients"]["scan_aqua"] * scans + \
                     params_a["coefficients"]["log_frp_aqua"] * med_log_aqua + \
                     params_a["coefficients"]["distance_m"] * med_dist
    ax_a.plot(scans, np.exp(pred_log_ratio), "r-", linewidth=2.0, label="Model A Fit (Median FRP & Dist)")
    ax_a.set_yscale("log")
    ax_a.set_title("A. Pairwise FRP Ratio (VIIRS/Aqua) vs Aqua Scan Angle", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("MODIS Aqua Scan Angle (km)", fontsize=10, fontweight="bold")
    ax_a.set_ylabel("FRP Ratio (VIIRS / Aqua, log scale)", fontsize=10, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_a.legend(loc="upper right")

    # Panel B: Pairwise Predicted vs Observed VIIRS FRP
    ax_b = axes[0, 1]
    X_all = pairs[["scan_aqua", "log_frp_aqua", "distance_m"]].values
    log_ratio_pred = params_a["coefficients"]["intercept"] + \
                     params_a["coefficients"]["scan_aqua"] * X_all[:, 0] + \
                     params_a["coefficients"]["log_frp_aqua"] * X_all[:, 1] + \
                     params_a["coefficients"]["distance_m"] * X_all[:, 2]
    pred_viirs_pair = pairs["frp_aqua"] * np.exp(log_ratio_pred)
    ax_b.scatter(pairs["frp_viirs"], pred_viirs_pair, alpha=0.4, color="#33a02c", s=20)
    max_val = max(pairs["frp_viirs"].max(), pred_viirs_pair.max())
    ax_b.plot([0.5, max_val], [0.5, max_val], "k--", label="1:1 Equality")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_title("B. Model A: Observed vs Predicted VIIRS FRP (Pairs)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Observed VIIRS FRP (MW)", fontsize=10, fontweight="bold")
    ax_b.set_ylabel("Predicted VIIRS FRP (MW)", fontsize=10, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_b.legend(loc="upper left")

    # Panel C: Cell-Week Total FRP Fit
    ax_c = axes[1, 0]
    X_cell = pos_df[["log1p_frp_modis", "log_exposure"]].values
    pred_cell_frp = np.exp(params_b["coefficients"]["intercept"] +
                           params_b["coefficients"]["log1p_frp_modis"] * X_cell[:, 0] +
                           params_b["coefficients"]["log_exposure"] * X_cell[:, 1])
    ax_c.scatter(pos_df["frp_sum_modis"], pos_df["frp_sum_viirs"], alpha=0.3, color="#ff7f00", s=25, label="Observed Cell-Weeks")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    max_c_val = max(pos_df["frp_sum_viirs"].max(), pos_df["frp_sum_modis"].max())
    ax_c.plot([1, max_c_val], [1, max_c_val], "k--", label="1:1 Equality")
    ax_c.set_title("C. Model B: Total FRP per Cell-Week (MODIS vs VIIRS)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("MODIS Total FRP per Cell-Week (MW)", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("VIIRS Total FRP per Cell-Week (MW)", fontsize=10, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_c.legend(loc="upper left")

    # Panel D: Residual Distribution of Cell-Week FRP
    ax_d = axes[1, 1]
    res_frp = np.log(pos_df["frp_sum_viirs"]) - np.log(pred_cell_frp)
    ax_d.hist(res_frp, bins=40, color="#6a3d9a", alpha=0.7, edgecolor="black")
    ax_d.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax_d.set_title("D. Model B: Log-Scale Residual Distribution (Total FRP)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Log-Scale Residual log(V_obs) - log(V_pred)", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Cell-Week Count", fontsize=10, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_5_frp_models.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved FRP figure to {fig_path}")


def main():
    print("=== Milestone M4.5: FRP Calibration Models ===")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    metrics_a, params_a, pairs, model_a = fit_model_a_pairwise(config)
    metrics_b, params_b, pos_df, model_b = fit_model_b_cell_total(config)

    all_metrics = pd.concat([metrics_a, metrics_b], ignore_index=True)
    metrics_path = OUTPUT_DIR / "m4_5_frp_metrics.csv"
    all_metrics.to_csv(metrics_path, index=False)
    print(f"Saved combined FRP metrics to {metrics_path}")

    plot_frp_diagnostics(pairs, params_a, pos_df, params_b)

    print("\nFRP Model Metrics Summary:")
    print(all_metrics.to_string(index=False))
    print("=== M4.5 Complete! ===")


if __name__ == "__main__":
    main()
