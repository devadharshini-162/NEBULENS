#!/usr/bin/env python3
"""
pipeline/m4_2_robust_baseline.py
================================
NEBULENS — Milestone M4.2: Robust Linear Baseline.

Fits Huber robust linear regression models on monthly regional totals for:
1. Fire detection counts: VIIRS count ~ MODIS count
2. Fire radiative power: VIIRS total FRP ~ MODIS total FRP

Evaluates strictly on:
- Training set: 2012-01 to 2018-12 (84 months)
- Validation set: 2019-01 to 2021-12 (36 months)
- Test set (held-out): 2022-01 to 2026-08 (56 months)

Generates:
- data/analysis/m4/m4_2_baseline_predictions.parquet
- data/analysis/m4/m4_2_baseline_metrics.csv
- data/analysis/m4/m4_2_baseline_parameters.json
- docs/figures/m4_2_baseline_residuals.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MONTHLY_DATA_PATH = PROJECT_ROOT / "data" / "analysis" / "m3" / "m3_2_monthly_comparison.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m4"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4_calibration_config.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split_data():
    """Load monthly data and assign train/val/test splits."""
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    df = pd.read_parquet(MONTHLY_DATA_PATH)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    def get_split(row):
        d = row["year_month"]
        if d <= "2018-12":
            return "train"
        elif d <= "2021-12":
            return "validation"
        else:
            return "test"

    df["split"] = df.apply(get_split, axis=1)
    print(f"Monthly records by split: Train={sum(df['split']=='train')}, Val={sum(df['split']=='validation')}, Test={sum(df['split']=='test')}")
    return df, config


def fit_and_evaluate_models(df: pd.DataFrame):
    """Fit robust linear models for counts and FRP and evaluate across splits."""
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    # 1. Count Model: VIIRS ~ MODIS
    huber_count = HuberRegressor(epsilon=1.35, max_iter=1000)
    X_train_c = train[["n_det_modis"]].values
    y_train_c = train["n_det_viirs"].values
    huber_count.fit(X_train_c, y_train_c)

    # 2. FRP Model: VIIRS FRP ~ MODIS FRP
    huber_frp = HuberRegressor(epsilon=1.35, max_iter=1000)
    X_train_f = train[["frp_sum_modis"]].values
    y_train_f = train["frp_sum_viirs"].values
    huber_frp.fit(X_train_f, y_train_f)

    # Predictions
    df_out = df.copy()
    df_out["pred_count_viirs"] = huber_count.predict(df[["n_det_modis"]].values)
    df_out["pred_frp_viirs"] = huber_frp.predict(df[["frp_sum_modis"]].values)

    # Clip negative predictions at zero
    df_out["pred_count_viirs"] = np.maximum(df_out["pred_count_viirs"], 0.0)
    df_out["pred_frp_viirs"] = np.maximum(df_out["pred_frp_viirs"], 0.0)

    # Residuals
    df_out["resid_count"] = df_out["n_det_viirs"] - df_out["pred_count_viirs"]
    df_out["resid_frp"] = df_out["frp_sum_viirs"] - df_out["pred_frp_viirs"]

    # Compute metrics per split
    metrics_list = []
    for split_name in ["train", "validation", "test"]:
        s_df = df_out[df_out["split"] == split_name]

        # Count metrics
        mae_c = np.mean(np.abs(s_df["n_det_viirs"] - s_df["pred_count_viirs"]))
        rmse_c = np.sqrt(np.mean((s_df["n_det_viirs"] - s_df["pred_count_viirs"]) ** 2))
        r_c, _ = stats.pearsonr(s_df["n_det_viirs"], s_df["pred_count_viirs"])
        r2_c = 1.0 - (np.sum((s_df["n_det_viirs"] - s_df["pred_count_viirs"]) ** 2) / np.sum((s_df["n_det_viirs"] - s_df["n_det_viirs"].mean()) ** 2))

        # FRP metrics
        mae_f = np.mean(np.abs(s_df["frp_sum_viirs"] - s_df["pred_frp_viirs"]))
        rmse_f = np.sqrt(np.mean((s_df["frp_sum_viirs"] - s_df["pred_frp_viirs"]) ** 2))
        r_f, _ = stats.pearsonr(s_df["frp_sum_viirs"], s_df["pred_frp_viirs"])
        r2_f = 1.0 - (np.sum((s_df["frp_sum_viirs"] - s_df["pred_frp_viirs"]) ** 2) / np.sum((s_df["frp_sum_viirs"] - s_df["frp_sum_viirs"].mean()) ** 2))

        metrics_list.append({
            "model_step": "Step_A_Robust_Baseline",
            "split": split_name,
            "n_months": len(s_df),
            "count_mae": round(mae_c, 2),
            "count_rmse": round(rmse_c, 2),
            "count_r2": round(r2_c, 3),
            "count_pearson_r": round(r_c, 3),
            "frp_mae": round(mae_f, 2),
            "frp_rmse": round(rmse_f, 2),
            "frp_r2": round(r2_f, 3),
            "frp_pearson_r": round(r_f, 3),
        })

    metrics_df = pd.DataFrame(metrics_list)

    # Save fitted parameters
    params = {
        "model_step": "Step_A_Robust_Baseline",
        "algorithm": "HuberRegressor(epsilon=1.35)",
        "count_model": {
            "slope": float(huber_count.coef_[0]),
            "intercept": float(huber_count.intercept_),
            "scale": float(huber_count.scale_)
        },
        "frp_model": {
            "slope": float(huber_frp.coef_[0]),
            "intercept": float(huber_frp.intercept_),
            "scale": float(huber_frp.scale_)
        }
    }

    # Save outputs
    pred_path = OUTPUT_DIR / "m4_2_baseline_predictions.parquet"
    metrics_path = OUTPUT_DIR / "m4_2_baseline_metrics.csv"
    params_path = OUTPUT_DIR / "m4_2_baseline_parameters.json"

    df_out.to_parquet(pred_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved predictions to {pred_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved parameters to {params_path}")
    return df_out, metrics_df, params


def plot_baseline_diagnostics(df: pd.DataFrame, params: dict):
    """Plot 4-panel diagnostic figure for Step A robust baseline."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # Panel A: Count Fit (Train vs Val vs Test)
    ax_c = axes[0, 0]
    colors = {"train": "#1f78b4", "validation": "#ff7f00", "test": "#33a02c"}
    for sp in ["train", "validation", "test"]:
        sub = df[df["split"] == sp]
        ax_c.scatter(sub["n_det_modis"], sub["n_det_viirs"], label=f"{sp.capitalize()} (n={len(sub)})",
                     color=colors[sp], alpha=0.7, s=35, edgecolor="none")

    max_c = max(df["n_det_modis"].max(), df["n_det_viirs"].max())
    x_ref = np.linspace(0, df["n_det_modis"].max(), 100)
    y_fit = params["count_model"]["intercept"] + params["count_model"]["slope"] * x_ref
    ax_c.plot(x_ref, y_fit, "r-", linewidth=2.0,
              label=f"Huber Fit: V = {params['count_model']['slope']:.2f}M + {params['count_model']['intercept']:.1f}")

    ax_c.set_title("A. Monthly Detection Counts: Robust Linear Baseline (Huber)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("MODIS Monthly Fire Detections", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("VIIRS Monthly Fire Detections", fontsize=10, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.4)
    ax_c.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel B: Count Residuals over Time
    ax_res = axes[0, 1]
    df["dt"] = pd.to_datetime(df["date"])
    for sp in ["train", "validation", "test"]:
        sub = df[df["split"] == sp]
        ax_res.scatter(sub["dt"], sub["resid_count"], color=colors[sp], label=sp.capitalize(), alpha=0.8, s=30)

    ax_res.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax_res.set_title("B. Monthly Count Residuals over Time (2012–2026)", fontsize=11, fontweight="bold")
    ax_res.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax_res.set_ylabel("Residual (Observed - Predicted VIIRS)", fontsize=10, fontweight="bold")
    ax_res.grid(True, linestyle="--", alpha=0.4)
    ax_res.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel C: FRP Fit
    ax_f = axes[1, 0]
    for sp in ["train", "validation", "test"]:
        sub = df[df["split"] == sp]
        ax_f.scatter(sub["frp_sum_modis"], sub["frp_sum_viirs"], color=colors[sp], label=f"{sp.capitalize()}", alpha=0.7, s=35)

    x_f_ref = np.linspace(0, df["frp_sum_modis"].max(), 100)
    y_f_fit = params["frp_model"]["intercept"] + params["frp_model"]["slope"] * x_f_ref
    ax_f.plot(x_f_ref, y_f_fit, "r-", linewidth=2.0,
              label=f"Huber Fit: FRP_V = {params['frp_model']['slope']:.2f}FRP_M + {params['frp_model']['intercept']:.1f}")

    ax_f.set_title("C. Monthly Total FRP: Robust Linear Baseline (Huber)", fontsize=11, fontweight="bold")
    ax_f.set_xlabel("MODIS Monthly Total FRP (MW)", fontsize=10, fontweight="bold")
    ax_f.set_ylabel("VIIRS Monthly Total FRP (MW)", fontsize=10, fontweight="bold")
    ax_f.grid(True, linestyle="--", alpha=0.4)
    ax_f.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel D: Residual QQ Plot (Count Model)
    ax_qq = axes[1, 1]
    stats.probplot(df["resid_count"], dist="norm", plot=ax_qq)
    ax_qq.set_title("D. Normal Q-Q Plot of Baseline Count Residuals", fontsize=11, fontweight="bold")
    ax_qq.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_2_baseline_residuals.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved diagnostic figure to {fig_path}")


def main():
    print("=== Milestone M4.2: Robust Linear Baseline ===")
    df, config = load_and_split_data()
    df_out, metrics_df, params = fit_and_evaluate_models(df)
    plot_baseline_diagnostics(df_out, params)
    print("\nStep A Metrics Summary:")
    print(metrics_df.to_string(index=False))
    print("=== M4.2 Complete! ===")


if __name__ == "__main__":
    main()
