#!/usr/bin/env python3
"""
pipeline/m4_7_bootstrap_uncertainty.py
======================================
NEBULENS — Milestone M4.7: Year-Block Bootstrap Uncertainty Estimation.

High-performance, memory-efficient implementation:
- Resamples calendar years [2012, 2018] with replacement (B = 100 replicates)
- Maps year-level resampling to observation frequency weights
- Solves Step B Negative Binomial GLM using warm-started IRLS in vectorised NumPy (< 0.2s per rep)
- Solves Model B Huber FRP model using sample weights (< 0.01s per rep)
- Calculates empirical 80% (q_0.10, q_0.90) and 95% (q_0.025, q_0.975) intervals
- Evaluates out-of-sample coverage on held-out Test partition (2022–2026, 362,080 cell-weeks)
- Generates publication figure: docs/figures/m4_7_bootstrap_coverage.png
"""

import sys
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
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
NB_COEFF_PATH = OUTPUT_DIR / "m4_3_nb_coefficients.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_exact_design_matrix(df: pd.DataFrame):
    """Construct deterministic design matrix matching Step B GLM exactly."""
    df_clean = df.copy()
    df_clean["lc_clean"] = df_clean["landcover_class"].str.replace(r"[^a-zA-Z0-9]", "_", regex=True)

    col_names = [
        "Intercept",
        "season_NE_Monsoon",
        "season_SW_Monsoon",
        "season_Winter",
        "lc_Cropland__Intensive___Delta_",
        "lc_Forest___Woodland",
        "lc_Shrubland___Savanna__Southern_Semi_Arid_",
        "lc_Urban___Developed_Corridor",
        "log_modis1",
        "inter_Cropland___Mosaic__Inland___Dryland_",
        "inter_Forest___Woodland",
        "inter_Shrubland___Savanna__Southern_Semi_Arid_",
        "inter_Urban___Developed_Corridor"
    ]

    X_cols = [
        np.ones(len(df_clean), dtype=np.float64),
        (df_clean["season"] == "NE_Monsoon").astype(np.float64).values,
        (df_clean["season"] == "SW_Monsoon").astype(np.float64).values,
        (df_clean["season"] == "Winter").astype(np.float64).values,
        (df_clean["lc_clean"] == "Cropland__Intensive___Delta_").astype(np.float64).values,
        (df_clean["lc_clean"] == "Forest___Woodland").astype(np.float64).values,
        (df_clean["lc_clean"] == "Shrubland___Savanna__Southern_Semi_Arid_").astype(np.float64).values,
        (df_clean["lc_clean"] == "Urban___Developed_Corridor").astype(np.float64).values,
        df_clean["log_modis1"].values.astype(np.float64),
        (df_clean["log_modis1"] * (df_clean["lc_clean"] == "Cropland___Mosaic__Inland___Dryland_")).values.astype(np.float64),
        (df_clean["log_modis1"] * (df_clean["lc_clean"] == "Forest___Woodland")).values.astype(np.float64),
        (df_clean["log_modis1"] * (df_clean["lc_clean"] == "Shrubland___Savanna__Southern_Semi_Arid_")).values.astype(np.float64),
        (df_clean["log_modis1"] * (df_clean["lc_clean"] == "Urban___Developed_Corridor")).values.astype(np.float64)
    ]
    return np.column_stack(X_cols), col_names


def main():
    print("=== Milestone M4.7: Year-Block Bootstrap Uncertainty Estimation ===")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    b_config = config.get("bootstrap_uncertainty", {})
    n_replicates = b_config.get("n_replicates", 100)
    master_seed = b_config.get("random_seed", 42)

    print(f"Loading cell-weeks dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_years = np.sort(train_df["year"].unique())
    n_years = len(train_years)
    print(f"Training years for block bootstrap: {train_years} (Total {n_years} years, {len(train_df):,} cell-weeks)")

    # 1. Build design matrix
    print("Constructing exact GLM design matrices...")
    X_train, col_names = build_exact_design_matrix(train_df)
    y_train = train_df["n_det_viirs"].values.astype(np.float64)
    off_train = train_df["log_exposure"].values.astype(np.float64)
    years_train = train_df["year"].values

    # Load baseline beta for warm start
    with open(NB_COEFF_PATH) as f:
        nb_dict = json.load(f)["coefficients"]

    beta_init = np.array([
        nb_dict["Intercept"],
        nb_dict["C(season, Treatment(reference='Summer'))[T.NE_Monsoon]"],
        nb_dict["C(season, Treatment(reference='Summer'))[T.SW_Monsoon]"],
        nb_dict["C(season, Treatment(reference='Summer'))[T.Winter]"],
        nb_dict["C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.Cropland__Intensive___Delta_]"],
        nb_dict["C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.Forest___Woodland]"],
        nb_dict["C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.Shrubland___Savanna__Southern_Semi_Arid_]"],
        nb_dict["C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.Urban___Developed_Corridor]"],
        nb_dict["log_modis1"],
        nb_dict["log_modis1:C(lc_clean)[T.Cropland___Mosaic__Inland___Dryland_]"],
        nb_dict["log_modis1:C(lc_clean)[T.Forest___Woodland]"],
        nb_dict["log_modis1:C(lc_clean)[T.Shrubland___Savanna__Southern_Semi_Arid_]"],
        nb_dict["log_modis1:C(lc_clean)[T.Urban___Developed_Corridor]"]
    ], dtype=np.float64)
    print(f"Warm-start beta loaded: Intercept={beta_init[0]:.3f}, log_modis1={beta_init[8]:.3f}")

    # 2. FRP training setup
    pos_mask = (train_df["n_det_modis"].values > 0) & (train_df["n_det_viirs"].values > 0)
    train_frp_df = train_df[pos_mask].copy()
    train_frp_df["log1p_frp_modis"] = np.log1p(train_frp_df["frp_sum_modis"])
    train_frp_df["log_frp_viirs"] = np.log(train_frp_df["frp_sum_viirs"])
    frp_features = ["log1p_frp_modis", "log_exposure"]
    X_frp = train_frp_df[frp_features].values
    y_frp = train_frp_df["log_frp_viirs"].values
    years_frp = train_frp_df["year"].values

    train_year_mask = {yr: (years_train == yr) for yr in train_years}
    frp_year_mask = {yr: (years_frp == yr) for yr in train_years}

    # 3. High-Performance Bootstrap Loop
    print(f"\nRunning {n_replicates} block-bootstrap replicates using vectorised IRLS...")
    t0 = time.time()
    rng = np.random.RandomState(master_seed)

    count_params_list = []
    frp_params_list = []
    alpha = 1.5

    for b in range(n_replicates):
        resampled_years = rng.choice(train_years, size=n_years, replace=True)
        counts = {yr: int(np.sum(resampled_years == yr)) for yr in train_years}

        # Vectorized weight construction
        w_count = np.zeros(len(train_df), dtype=np.float64)
        for yr, c in counts.items():
            if c > 0:
                w_count[train_year_mask[yr]] = c

        # 3 Newton-Raphson steps from warm-start beta
        beta = beta_init.copy()
        for it in range(3):
            eta = off_train + np.dot(X_train, beta)
            eta = np.clip(eta, -20.0, 10.0)
            mu = np.exp(eta)
            W = (w_count * mu) / (1.0 + alpha * mu)
            z = (eta - off_train) + (y_train - mu) / np.maximum(mu, 1e-6)
            XtW = X_train.T * W
            H = np.dot(XtW, X_train) + 1e-5 * np.eye(13)
            g = np.dot(XtW, z)
            beta = np.linalg.solve(H, g)

        count_params_list.append(beta.tolist())

        # Fit FRP Huber with sample weights
        w_frp = np.zeros(len(train_frp_df), dtype=np.float64)
        for yr, c in counts.items():
            if c > 0:
                w_frp[frp_year_mask[yr]] = c

        huber = HuberRegressor(epsilon=1.35, max_iter=200).fit(X_frp, y_frp, sample_weight=w_frp)
        frp_params_list.append([float(huber.intercept_)] + [float(c) for c in huber.coef_])

        if (b + 1) % 25 == 0 or b == n_replicates - 1:
            elapsed = time.time() - t0
            print(f"  Replicate {b + 1}/{n_replicates} completed ({elapsed:.1f}s elapsed, {elapsed/(b+1):.3f}s/rep)...")

    elapsed_total = time.time() - t0
    print(f"\nAll {n_replicates} replicates finished in {elapsed_total:.2f}s!")

    count_params_arr = np.array(count_params_list)  # (100, 13)
    frp_params_arr = np.array(frp_params_list)      # (100, 3)

    # Free training structures
    del X_train, y_train, off_train, train_df, train_frp_df, w_count, w_frp
    gc.collect()

    # Save bootstrap models JSON
    models_dict = {
        "n_replicates": n_replicates,
        "master_seed": master_seed,
        "count_features": col_names,
        "count_params_mean": count_params_arr.mean(axis=0).tolist(),
        "count_params_std": count_params_arr.std(axis=0).tolist(),
        "frp_features": ["Intercept"] + frp_features,
        "frp_params_mean": frp_params_arr.mean(axis=0).tolist(),
        "frp_params_std": frp_params_arr.std(axis=0).tolist(),
        "count_params_all": count_params_arr.tolist(),
        "frp_params_all": frp_params_arr.tolist()
    }
    models_path = OUTPUT_DIR / "m4_7_bootstrap_models.json"
    with open(models_path, "w") as f:
        json.dump(models_dict, f, indent=2)
    print(f"Saved bootstrap models to {models_path}")

    # 4. Evaluate Test Set Coverage
    print("\nEvaluating empirical coverage on held-out Test set (2022–2026, 362,080 cell-weeks)...")
    X_test, _ = build_exact_design_matrix(test_df)
    off_test = test_df["log_exposure"].values.astype(np.float64)
    y_test = test_df["n_det_viirs"].values.astype(np.float64)

    print("Computing ensemble rate predictions...")
    eta_matrix = off_test[:, None] + np.dot(X_test, count_params_arr.T)
    mu_matrix = np.exp(np.clip(eta_matrix, -20.0, 15.0))  # (n_test, 100)
    mu_median = np.median(mu_matrix, axis=1)

    n_param = 1.0 / alpha
    p_param_matrix = n_param / (n_param + mu_matrix)

    print("Computing 80% and 95% interval bounds across replicates...")
    lower_80_mat = stats.nbinom.ppf(0.10, n_param, p_param_matrix)
    upper_80_mat = stats.nbinom.ppf(0.90, n_param, p_param_matrix)
    lower_95_mat = stats.nbinom.ppf(0.025, n_param, p_param_matrix)
    upper_95_mat = stats.nbinom.ppf(0.975, n_param, p_param_matrix)

    lower_80 = np.nan_to_num(np.median(lower_80_mat, axis=1), nan=0.0)
    upper_80 = np.nan_to_num(np.median(upper_80_mat, axis=1), nan=0.0)
    lower_95 = np.nan_to_num(np.median(lower_95_mat, axis=1), nan=0.0)
    upper_95 = np.nan_to_num(np.median(upper_95_mat, axis=1), nan=0.0)

    # Coverage metrics
    covered_80 = (y_test >= lower_80) & (y_test <= upper_80)
    covered_95 = (y_test >= lower_95) & (y_test <= upper_95)

    cov_80_all = float(np.mean(covered_80) * 100.0)
    cov_95_all = float(np.mean(covered_95) * 100.0)

    mask_zero = (y_test == 0)
    mask_pos = (y_test > 0)
    mask_high = (y_test >= 5)

    cov_80_zero = float(np.mean(covered_80[mask_zero]) * 100.0)
    cov_95_zero = float(np.mean(covered_95[mask_zero]) * 100.0)

    cov_80_pos = float(np.mean(covered_80[mask_pos]) * 100.0)
    cov_95_pos = float(np.mean(covered_95[mask_pos]) * 100.0)

    cov_80_high = float(np.mean(covered_80[mask_high]) * 100.0)
    cov_95_high = float(np.mean(covered_95[mask_high]) * 100.0)

    print(f"Overall Coverage: 80% Nominal = {cov_80_all:.2f}% | 95% Nominal = {cov_95_all:.2f}%")
    print(f"Zero Fire Cell-Weeks (N={np.sum(mask_zero):,}): 80% = {cov_80_zero:.2f}% | 95% = {cov_95_zero:.2f}%")
    print(f"Active Fire Cell-Weeks (N={np.sum(mask_pos):,}): 80% = {cov_80_pos:.2f}% | 95% = {cov_95_pos:.2f}%")
    print(f"High Fire Cell-Weeks >=5 (N={np.sum(mask_high):,}): 80% = {cov_80_high:.2f}% | 95% = {cov_95_high:.2f}%")

    coverage_df = pd.DataFrame([
        {"subset": "All Cell-Weeks (Test)", "n_records": len(y_test), "cov_80_nominal": 80.0, "cov_80_actual": round(cov_80_all, 2), "cov_95_nominal": 95.0, "cov_95_actual": round(cov_95_all, 2)},
        {"subset": "Zero-Fire (Test)", "n_records": int(np.sum(mask_zero)), "cov_80_nominal": 80.0, "cov_80_actual": round(cov_80_zero, 2), "cov_95_nominal": 95.0, "cov_95_actual": round(cov_95_zero, 2)},
        {"subset": "Active-Fire (V > 0)", "n_records": int(np.sum(mask_pos)), "cov_80_nominal": 80.0, "cov_80_actual": round(cov_80_pos, 2), "cov_95_nominal": 95.0, "cov_95_actual": round(cov_95_pos, 2)},
        {"subset": "High-Fire (V >= 5)", "n_records": int(np.sum(mask_high)), "cov_80_nominal": 80.0, "cov_80_actual": round(cov_80_high, 2), "cov_95_nominal": 95.0, "cov_95_actual": round(cov_95_high, 2)}
    ])
    coverage_path = OUTPUT_DIR / "m4_7_bootstrap_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Saved coverage metrics to {coverage_path}")

    # Summary dataset for test set
    test_summary = test_df[["grid_cell_id", "mid_date_str", "n_det_modis", "n_det_viirs", "frp_sum_modis", "frp_sum_viirs", "landcover_class", "season"]].copy()
    test_summary["pred_count_median"] = np.round(mu_median, 3)
    test_summary["count_lower_80"] = lower_80.astype(int)
    test_summary["count_upper_80"] = upper_80.astype(int)
    test_summary["count_lower_95"] = lower_95.astype(int)
    test_summary["count_upper_95"] = upper_95.astype(int)

    test_unc_path = OUTPUT_DIR / "m4_7_bootstrap_uncertainty.parquet"
    test_summary.to_parquet(test_unc_path, index=False)
    print(f"Saved sample test uncertainty dataset to {test_unc_path}")

    # 5. Publication Figure
    print("Generating publication figure: docs/figures/m4_7_bootstrap_coverage.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: GLM Parameters
    ax_a = axes[0, 0]
    param_indices = [col_names.index("Intercept"), col_names.index("log_modis1"), col_names.index("season_Winter"), col_names.index("season_SW_Monsoon")]
    param_labels = ["Intercept", "log(1+MODIS)", "Season: Winter", "Season: SW Monsoon"]
    box_data = [count_params_arr[:, idx] for idx in param_indices]
    bp = ax_a.boxplot(box_data, patch_artist=True, tick_labels=param_labels)
    colors = ["#4575b4", "#74add1", "#fdae61", "#f46d43"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_a.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_a.set_ylabel("Parameter Estimate Value", fontsize=10, fontweight="bold")
    ax_a.set_title("A. GLM Count Parameters Across 100 Year-Blocks", fontsize=11, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel B: FRP Parameters
    ax_b = axes[0, 1]
    frp_labels = ["FRP Intercept", "log(1+FRP MODIS)", "log(Exposure)"]
    bp_b = ax_b.boxplot([frp_params_arr[:, 0], frp_params_arr[:, 1], frp_params_arr[:, 2]], patch_artist=True, tick_labels=frp_labels)
    for patch, color in zip(bp_b["boxes"], ["#a6d96a", "#66bd63", "#1a9850"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax_b.set_ylabel("FRP Parameter Estimate", fontsize=10, fontweight="bold")
    ax_b.set_title("B. FRP Model Parameters Across 100 Year-Blocks", fontsize=11, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel C: Empirical vs Nominal Coverage
    ax_c = axes[1, 0]
    subsets = coverage_df["subset"]
    x = np.arange(len(subsets))
    w = 0.35
    ax_c.bar(x - w/2, coverage_df["cov_80_actual"], width=w, label="80% Nominal Target", color="#4575b4", alpha=0.85, edgecolor="black")
    ax_c.bar(x + w/2, coverage_df["cov_95_actual"], width=w, label="95% Nominal Target", color="#d73027", alpha=0.85, edgecolor="black")
    ax_c.axhline(80.0, color="#4575b4", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_c.axhline(95.0, color="#d73027", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(subsets, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax_c.set_ylabel("Empirical Test Coverage (%)", fontsize=10, fontweight="bold")
    ax_c.set_title("C. Out-of-Sample Coverage (Test Partition 2022–2026)", fontsize=11, fontweight="bold")
    ax_c.set_ylim(0, 105)
    ax_c.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_c.legend(loc="lower right")

    # Panel D: Regional Trajectory
    ax_d = axes[1, 1]
    test_summary["date"] = pd.to_datetime(test_summary["mid_date_str"])
    weekly_agg = test_summary.groupby("date").agg({
        "n_det_viirs": "sum",
        "pred_count_median": "sum",
        "count_lower_80": "sum",
        "count_upper_80": "sum",
        "count_lower_95": "sum",
        "count_upper_95": "sum"
    }).reset_index().sort_values("date")

    ax_d.fill_between(weekly_agg["date"], weekly_agg["count_lower_95"], weekly_agg["count_upper_95"], color="#fee090", alpha=0.6, label="95% Bootstrap Interval")
    ax_d.fill_between(weekly_agg["date"], weekly_agg["count_lower_80"], weekly_agg["count_upper_80"], color="#fdae61", alpha=0.6, label="80% Bootstrap Interval")
    ax_d.plot(weekly_agg["date"], weekly_agg["n_det_viirs"], color="black", linewidth=1.2, label="Observed VIIRS Total")
    ax_d.plot(weekly_agg["date"], weekly_agg["pred_count_median"], color="#d73027", linestyle="--", linewidth=1.2, label="Calibrated Median")
    ax_d.set_ylabel("Regional Weekly Fire Detections", fontsize=10, fontweight="bold")
    ax_d.set_title("D. Regional Trajectory with Uncertainty (Held-Out Test)", fontsize=11, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4)
    ax_d.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_7_bootstrap_coverage.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved bootstrap figure to {fig_path}")

    print("=== M4.7 Complete! ===")


if __name__ == "__main__":
    main()
