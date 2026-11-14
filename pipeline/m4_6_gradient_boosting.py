#!/usr/bin/env python3
"""
pipeline/m4_6_gradient_boosting.py
==================================
NEBULENS — Milestone M4.6: Gradient Boosting Comparison & Model Selection.

Fits Step D of the model ladder:
1. Count Model: HistGradientBoostingRegressor with Poisson deviance loss
2. FRP Model: HistGradientBoostingRegressor with squared error loss on log1p(FRP)

Compares against Step B (Negative Binomial GLM) on:
- Temporal Validation partition (2019–2021)
- Temporal Test partition (2022–2026 Held-out)
- Spatial Holdout Blocks (Block 3 Delta, Block 4 Ghats)

Applies frozen promotion criteria:
- Does Gradient Boosting improve held-out MAE by >= 5.0% on both temporal and spatial holdouts?
- If not, the simpler interpretable parametric model (Negative Binomial GLM) is selected.

Generates:
- data/analysis/m4/m4_6_gbm_metrics.csv
- data/analysis/m4/m4_6_model_comparison_decision.json
- docs/figures/m4_6_gbm_comparison.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
    print("=== Milestone M4.6: Gradient Boosting Comparison ===")
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    df = pd.read_parquet(DATASET_PATH)

    # Encode categorical features as integer categories
    df["season_cat"] = df["season"].astype("category").cat.codes
    df["lc_cat"] = df["landcover_class"].astype("category").cat.codes

    feature_cols = ["log_modis1", "log_exposure", "month", "season_cat", "lc_cat"]
    feature_names = ["Log(1+MODIS)", "Log(Exposure)", "Calendar Month", "Season", "Land Cover Class"]

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols].values
    y_train_c = train_df["n_det_viirs"].values
    y_train_f = np.log1p(train_df["frp_sum_viirs"].values)

    # 1. Fit Count Model (Poisson loss)
    print("Fitting Count Model: HistGradientBoostingRegressor(loss='poisson')...")
    gbm_count = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=100,
        learning_rate=0.08,
        min_samples_leaf=50,
        random_state=config["random_seed"]
    )
    gbm_count.fit(X_train, y_train_c)

    # 2. Fit FRP Model (Squared error on log1p(FRP))
    print("Fitting FRP Model: HistGradientBoostingRegressor on log1p(FRP)...")
    gbm_frp = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=100,
        learning_rate=0.08,
        min_samples_leaf=50,
        random_state=config["random_seed"]
    )
    gbm_frp.fit(X_train, y_train_f)

    # Predictions
    print("Generating Gradient Boosting predictions across all partitions...")
    df["gbm_pred_count"] = gbm_count.predict(df[feature_cols].values)
    df["gbm_pred_frp"] = np.expm1(np.clip(gbm_frp.predict(df[feature_cols].values), 0, 15))

    train_df["gbm_pred_count"] = df.loc[train_df.index, "gbm_pred_count"]
    val_df["gbm_pred_count"] = df.loc[val_df.index, "gbm_pred_count"]
    test_df["gbm_pred_count"] = df.loc[test_df.index, "gbm_pred_count"]

    train_df["gbm_pred_frp"] = df.loc[train_df.index, "gbm_pred_frp"]
    val_df["gbm_pred_frp"] = df.loc[val_df.index, "gbm_pred_frp"]
    test_df["gbm_pred_frp"] = df.loc[test_df.index, "gbm_pred_frp"]

    # 3. Evaluate Metrics
    metrics = []

    def eval_gbm(sub, name):
        y_c = sub["n_det_viirs"].values
        p_c = sub["gbm_pred_count"].values
        mae_c = float(np.mean(np.abs(y_c - p_c)))
        rmse_c = float(np.sqrt(np.mean((y_c - p_c) ** 2)))
        r_c, _ = stats.pearsonr(y_c, p_c) if np.std(p_c) > 0 else (0.0, 1.0)

        y_f = sub["frp_sum_viirs"].values
        p_f = sub["gbm_pred_frp"].values
        mae_f = float(np.mean(np.abs(y_f - p_f)))
        rmse_f = float(np.sqrt(np.mean((y_f - p_f) ** 2)))
        r_f, _ = stats.pearsonr(y_f, p_f) if np.std(p_f) > 0 else (0.0, 1.0)

        return {
            "model_step": "Step_D_Gradient_Boosting",
            "evaluation_subset": name,
            "n_records": len(sub),
            "count_mae": round(mae_c, 4),
            "count_rmse": round(rmse_c, 4),
            "count_pearson_r": round(r_c, 4),
            "frp_mae": round(mae_f, 4),
            "frp_rmse": round(rmse_f, 4),
            "frp_pearson_r": round(r_f, 4)
        }

    metrics.append(eval_gbm(train_df, "Temporal_Train (2012-2018)"))
    metrics.append(eval_gbm(val_df, "Temporal_Validation (2019-2021)"))
    metrics.append(eval_gbm(test_df, "Temporal_Test (2022-2026 Held-Out)"))

    # Spatial holdouts
    val_test = df[df["split"].isin(["validation", "test"])]
    block_delta = val_test[val_test["spatial_block"] == "Block_3_CentralEast_Delta"]
    block_ghats = val_test[val_test["spatial_block"] == "Block_4_CentralWest"]
    metrics.append(eval_gbm(block_delta, "Spatial_Holdout_CauveryDelta"))
    metrics.append(eval_gbm(block_ghats, "Spatial_Holdout_WesternGhats"))

    gbm_metrics_df = pd.DataFrame(metrics)
    gbm_metrics_path = OUTPUT_DIR / "m4_6_gbm_metrics.csv"
    gbm_metrics_df.to_csv(gbm_metrics_path, index=False)
    print(f"Saved GBM metrics to {gbm_metrics_path}")

    # 4. Compare with Step B (Negative Binomial GLM)
    nb_df = pd.read_csv(NB_METRICS_PATH)
    val_nb = nb_df[nb_df["evaluation_subset"].str.contains("Validation")].iloc[0]
    val_gbm = gbm_metrics_df[gbm_metrics_df["evaluation_subset"].str.contains("Validation")].iloc[0]

    test_nb = nb_df[nb_df["evaluation_subset"].str.contains("Test")].iloc[0]
    test_gbm = gbm_metrics_df[gbm_metrics_df["evaluation_subset"].str.contains("Test")].iloc[0]

    val_imp = ((val_nb["mae"] - val_gbm["count_mae"]) / val_nb["mae"]) * 100.0
    test_imp = ((test_nb["mae"] - test_gbm["count_mae"]) / test_nb["mae"]) * 100.0

    # Spatial holdout comparison
    sp_delta_nb = nb_df[nb_df["evaluation_subset"] == "Spatial_Holdout_CauveryDelta"].iloc[0]["mae"]
    sp_delta_gbm = gbm_metrics_df[gbm_metrics_df["evaluation_subset"] == "Spatial_Holdout_CauveryDelta"].iloc[0]["count_mae"]
    sp_delta_imp = ((sp_delta_nb - sp_delta_gbm) / sp_delta_nb) * 100.0

    sp_ghats_nb = nb_df[nb_df["evaluation_subset"] == "Spatial_Holdout_WesternGhats"].iloc[0]["mae"]
    sp_ghats_gbm = gbm_metrics_df[gbm_metrics_df["evaluation_subset"] == "Spatial_Holdout_WesternGhats"].iloc[0]["count_mae"]
    sp_ghats_imp = ((sp_ghats_nb - sp_ghats_gbm) / sp_ghats_nb) * 100.0

    print("\n=== Model Ladder Comparison: Step B (NB GLM) vs Step D (Gradient Boosting) ===")
    print(f"Validation Count MAE: NB GLM = {val_nb['mae']:.4f} | GBM = {val_gbm['count_mae']:.4f} (Improvement = {val_imp:.2f}%)")
    print(f"Test Count MAE:       NB GLM = {test_nb['mae']:.4f} | GBM = {test_gbm['count_mae']:.4f} (Improvement = {test_imp:.2f}%)")
    print(f"Spatial Delta MAE:    NB GLM = {sp_delta_nb:.4f} | GBM = {sp_delta_gbm:.4f} (Improvement = {sp_delta_imp:.2f}%)")
    print(f"Spatial Ghats MAE:    NB GLM = {sp_ghats_nb:.4f} | GBM = {sp_ghats_gbm:.4f} (Improvement = {sp_ghats_imp:.2f}%)")

    # Final Promotion Decision
    threshold = config["promotion_rules"]["gradient_boosting_min_mae_improvement_pct"]
    passes_threshold = (test_imp >= threshold) and (sp_delta_imp >= threshold) and (sp_ghats_imp >= threshold)

    print(f"\nRequired Improvement Threshold on All Held-Outs = {threshold}%")
    if passes_threshold:
        selected_model = "Step_D_Gradient_Boosting"
        decision_rationale = (
            f"Gradient Boosting exceeds the {threshold}% error improvement threshold on all temporal and spatial "
            f"holdouts (Test: {test_imp:.1f}%, Delta: {sp_delta_imp:.1f}%, Ghats: {sp_ghats_imp:.1f}%)."
        )
    else:
        selected_model = "Step_B_Negative_Binomial_GLM"
        decision_rationale = (
            f"Gradient Boosting failed to consistently beat Negative Binomial GLM by >= {threshold}% across all "
            f"spatial holdouts. In accordance with the simplicity rule, Step B (Negative Binomial GLM) is promoted."
        )

    print(f"Final Model Choice: {selected_model}")
    print(f"Rationale: {decision_rationale}")

    # Save Decision Summary
    decision_payload = {
        "selected_model": selected_model,
        "selected_model_step": "Step_B" if "Negative_Binomial" in selected_model else "Step_D",
        "rationale": decision_rationale,
        "comparisons": {
            "validation_mae": {"nb_glm": float(val_nb["mae"]), "gbm": float(val_gbm["count_mae"]), "improvement_pct": float(val_imp)},
            "test_mae": {"nb_glm": float(test_nb["mae"]), "gbm": float(test_gbm["count_mae"]), "improvement_pct": float(test_imp)},
            "spatial_delta_mae": {"nb_glm": float(sp_delta_nb), "gbm": float(sp_delta_gbm), "improvement_pct": float(sp_delta_imp)},
            "spatial_ghats_mae": {"nb_glm": float(sp_ghats_nb), "gbm": float(sp_ghats_gbm), "improvement_pct": float(sp_ghats_imp)},
        }
    }
    decision_path = OUTPUT_DIR / "m4_6_model_comparison_decision.json"
    with open(decision_path, "w") as f:
        json.dump(decision_payload, f, indent=2)
    print(f"Saved model decision to {decision_path}")

    # 5. Diagnostic Figures
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    # Panel A: MAE Comparison Bar Chart across Holdouts
    ax_bar = axes[0, 0]
    subsets = ["Validation (2019-21)", "Held-Out Test (2022-26)", "Spatial Delta", "Spatial Ghats"]
    nb_maes = [val_nb["mae"], test_nb["mae"], sp_delta_nb, sp_ghats_nb]
    gbm_maes = [val_gbm["count_mae"], test_gbm["count_mae"], sp_delta_gbm, sp_ghats_gbm]

    x = np.arange(len(subsets))
    w = 0.35
    ax_bar.bar(x - w/2, nb_maes, w, label="Step B (Negative Binomial GLM)", color="#1f78b4")
    ax_bar.bar(x + w/2, gbm_maes, w, label="Step D (Gradient Boosting)", color="#33a02c")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(subsets, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax_bar.set_ylabel("Count MAE (Cell-Weeks)", fontsize=10, fontweight="bold")
    ax_bar.set_title("A. Predictive Error (MAE): GLM vs Gradient Boosting", fontsize=11, fontweight="bold")
    ax_bar.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_bar.legend(loc="upper left")

    # Panel B: Permutation Feature Importance (GBM)
    ax_imp = axes[0, 1]
    # Compute permutation importance on a validation sample
    val_sample = val_df.sample(n=min(10000, len(val_df)), random_state=42)
    perm = permutation_importance(gbm_count, val_sample[feature_cols].values, val_sample["n_det_viirs"].values, n_repeats=5, random_state=42)
    y_pos = np.arange(len(feature_names))
    ax_imp.barh(y_pos, perm.importances_mean, color="#e31a1c", alpha=0.8, edgecolor="black")
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(feature_names, fontsize=10, fontweight="bold")
    ax_imp.set_xlabel("Mean Permutation Importance (Validation)", fontsize=10, fontweight="bold")
    ax_imp.set_title("B. Gradient Boosting Feature Importance", fontsize=11, fontweight="bold")
    ax_imp.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Compute GLM predictions for test_df for comparison plotting
    with open(OUTPUT_DIR / "m4_3_nb_coefficients.json") as f_nb:
        nb_params = json.load(f_nb)["coefficients"]
    
    df["lc_clean"] = df["landcover_class"].str.replace(r"[ /()-]", "_", regex=True)
    eta = nb_params["Intercept"] + df["log_exposure"] + nb_params["log_modis1"] * df["log_modis1"]
    for s in ["NE_Monsoon", "SW_Monsoon", "Winter"]:
        term = f"C(season, Treatment(reference='Summer'))[T.{s}]"
        if term in nb_params:
            eta += nb_params[term] * (df["season"] == s)
    for lc in df["lc_clean"].unique():
        term = f"C(lc_clean, Treatment(reference='Cropland___Mosaic__Inland___Dryland_'))[T.{lc}]"
        if term in nb_params:
            eta += nb_params[term] * (df["lc_clean"] == lc)
        inter_term = f"log_modis1:C(lc_clean)[T.{lc}]"
        if inter_term in nb_params:
            eta += nb_params[inter_term] * df["log_modis1"] * (df["lc_clean"] == lc)
    df["pred_viirs_count"] = np.exp(np.clip(eta, -20.0, 15.0))
    test_df["pred_viirs_count"] = df.loc[test_df.index, "pred_viirs_count"]

    # Panel C: Observed vs Predicted on Test Set (GLM)
    ax_c = axes[1, 0]
    test_sample = test_df.sample(n=min(5000, len(test_df)), random_state=42)
    ax_c.scatter(test_sample["pred_viirs_count"], test_sample["n_det_viirs"], alpha=0.3, color="#1f78b4", s=20)
    max_c = max(test_sample["n_det_viirs"].max(), test_sample["pred_viirs_count"].max())
    ax_c.plot([0, max_c], [0, max_c], "k--", label="1:1 Equality")
    ax_c.set_xscale("symlog")
    ax_c.set_yscale("symlog")
    ax_c.set_title("C. Step B (NB GLM): Observed vs Predicted (Test Set)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Predicted VIIRS Count (symlog)", fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Observed VIIRS Count (symlog)", fontsize=10, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_c.legend(loc="upper left")

    # Panel D: Observed vs Predicted on Test Set (GBM)
    ax_d = axes[1, 1]
    ax_d.scatter(test_sample["gbm_pred_count"], test_sample["n_det_viirs"], alpha=0.3, color="#33a02c", s=20)
    max_d = max(test_sample["n_det_viirs"].max(), test_sample["gbm_pred_count"].max())
    ax_d.plot([0, max_d], [0, max_d], "k--", label="1:1 Equality")
    ax_d.set_xscale("symlog")
    ax_d.set_yscale("symlog")
    ax_d.set_title("D. Step D (Gradient Boosting): Observed vs Predicted (Test Set)", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Predicted VIIRS Count (symlog)", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Observed VIIRS Count (symlog)", fontsize=10, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_d.legend(loc="upper left")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_6_gbm_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved comparison figure to {fig_path}")
    print("=== M4.6 Complete! ===")


if __name__ == "__main__":
    main()
