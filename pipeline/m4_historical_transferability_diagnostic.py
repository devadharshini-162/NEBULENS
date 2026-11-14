#!/usr/bin/env python3
"""
pipeline/m4_historical_transferability_diagnostic.py
===================================================
NEBULENS — Milestone M4: Historical Transferability Diagnostic (2000–2026).

Audits the transferability of the Step B Negative Binomial GLM calibration
from the training period (2012–2018) back to the historical era (2000–2011):
1. Investigates Terra-only era (2000-02-24 to 2002-05-03) vs Dual-MODIS era (2002–2011).
2. Investigates sensor composition level shifts around July 2002 and January 2012.
3. Quantifies power-law exponent runaway on high-MODIS forest cells (e.g. 2009 Dindigul fire).
4. Generates:
   - data/analysis/m4/m4_historical_transferability_diagnostic.csv
   - docs/figures/m4_historical_transferability.png
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HARMONISED_DIR = PROJECT_ROOT / "data" / "parquet" / "harmonised_daily"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m4"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
CONFIG_PATH = PROJECT_ROOT / "configs" / "m4_calibration_config.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_season(m: int) -> str:
    if m in (1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Summer"
    elif m in (6, 7, 8, 9):
        return "SW_Monsoon"
    else:
        return "NE_Monsoon"


def main():
    print("=== Running M4 Historical Transferability Diagnostic ===")
    t0 = time.time()

    years = range(2000, 2027)
    records = []

    for yr in years:
        file_path = HARMONISED_DIR / f"year={yr}" / f"harmonised_daily_{yr}.parquet"
        if not file_path.exists():
            continue

        df = pd.read_parquet(file_path)
        df["date"] = pd.to_datetime(df["analysis_date_ist"])
        df["month"] = df["date"].dt.month
        df["season"] = df["month"].apply(get_season)
        df["iso_week"] = df["date"].dt.isocalendar().week

        m_tot = int(df["modis_count_observed"].sum())
        v_tot = int(df["viirs_count_observed"].sum())
        h_tot = float(df["harmonised_count"].sum())
        exp_v_tot = float(df["expected_viirs_count"].sum())

        # Check maximum single cell-day and weekly totals
        max_daily_h = float(df["harmonised_count"].max())
        weekly_sums = df.groupby(["grid_cell_id", "iso_week"])["expected_viirs_count"].sum()
        max_weekly_h = float(weekly_sums.max())
        outlier_weeks_gt5k = int((weekly_sums > 5000).sum())

        # Era definition
        if yr <= 2001:
            epoch = "Terra_Only"
        elif yr == 2002:
            epoch = "Transition_Terra_to_Aqua"
        elif yr <= 2011:
            epoch = "Dual_MODIS"
        elif yr <= 2017:
            epoch = "VIIRS_Era_1_SNPP"
        elif yr <= 2022:
            epoch = "VIIRS_Era_2_NOAA20"
        else:
            epoch = "VIIRS_Era_3_NOAA21"

        row = {
            "year": yr,
            "epoch": epoch,
            "n_cell_days": len(df),
            "modis_detections": m_tot,
            "viirs_detections": v_tot,
            "harmonised_detections": round(h_tot, 1),
            "expected_viirs_detections": round(exp_v_tot, 1),
            "ratio_harm_to_modis": round(h_tot / max(m_tot, 1), 2),
            "max_daily_harmonised": round(max_daily_h, 1),
            "max_weekly_harmonised": round(max_weekly_h, 1),
            "outlier_weeks_gt5k": outlier_weeks_gt5k
        }

        # Seasonal breakdowns
        for s in ["Winter", "Summer", "SW_Monsoon", "NE_Monsoon"]:
            s_df = df[df["season"] == s]
            row[f"modis_{s}"] = int(s_df["modis_count_observed"].sum())
            row[f"harm_{s}"] = round(float(s_df["harmonised_count"].sum()), 1)

        records.append(row)
        print(f"  Year {yr} ({epoch}): MODIS={m_tot:,} | Harmonised={h_tot:,.1f} | Ratio={row['ratio_harm_to_modis']}x | MaxWeek={max_weekly_h:,.1f}")

    diag_df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "m4_historical_transferability_diagnostic.csv"
    diag_df.to_csv(csv_path, index=False)
    print(f"Saved diagnostic CSV to {csv_path}")

    # Generate Publication Diagnostic Figure
    print("Generating Figure: docs/figures/m4_historical_transferability.png...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel A: 26-year Trajectory (Log Scale to handle 2009 spike cleanly)
    ax_a = axes[0, 0]
    ax_a.plot(diag_df["year"], diag_df["modis_detections"], marker="o", color="#e41a1c", linewidth=1.5, label="Raw MODIS Total")
    ax_a.plot(diag_df[diag_df["year"] >= 2012]["year"], diag_df[diag_df["year"] >= 2012]["viirs_detections"], marker="s", color="#377eb8", linewidth=1.5, label="Observed VIIRS Total")
    ax_a.plot(diag_df["year"], diag_df["harmonised_detections"], marker="^", color="#4daf4a", linewidth=2.0, linestyle="--", label="Harmonised Total")
    ax_a.set_yscale("log")
    ax_a.axvline(2002.35, color="purple", linestyle=":", label="Aqua Launch (May 2002)")
    ax_a.axvline(2012.05, color="black", linestyle=":", label="VIIRS Launch (Jan 2012)")
    ax_a.annotate("2009 Dindigul Forest Fire\n(Power-Law Runaway Spike)", xy=(2009, 244081), xytext=(2003, 100000),
                  arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5), fontweight="bold", fontsize=9)
    ax_a.set_ylabel("Annual Detections (Log Scale)", fontsize=10, fontweight="bold")
    ax_a.set_title("A. 26-Year Trajectory & Sensor Epoch Transitions (Log Scale)", fontsize=11, fontweight="bold")
    ax_a.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_a.legend(loc="lower left", fontsize=8)

    # Panel B: Effective Annual Harmonisation Ratio
    ax_b = axes[0, 1]
    # Highlight normal vs runaway ratio
    colors = ["#e41a1c" if r > 30 else "#377eb8" for r in diag_df["ratio_harm_to_modis"]]
    ax_b.bar(diag_df["year"], diag_df["ratio_harm_to_modis"], color=colors, alpha=0.8, edgecolor="black")
    ax_b.axhline(12.5, color="green", linestyle="--", linewidth=1.5, label="Typical Scaling Ratio (~12.5x)")
    ax_b.axvline(2002.35, color="purple", linestyle=":")
    ax_b.axvline(2012.05, color="black", linestyle=":")
    ax_b.set_ylabel("Scaling Factor (Harmonised / MODIS)", fontsize=10, fontweight="bold")
    ax_b.set_title("B. Effective Annual Harmonisation Ratio (Red = Runaway Outliers)", fontsize=11, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_b.legend(loc="upper left")

    # Panel C: Seasonal Proportion Comparison Across Epochs
    ax_c = axes[1, 0]
    # Compare seasonal shares across 3 epochs
    epochs_to_compare = {
        "Terra-Only (2001)": diag_df[diag_df["year"] == 2001].iloc[0],
        "Dual-MODIS (2003-2011 Mean)": diag_df[(diag_df["year"] >= 2003) & (diag_df["year"] <= 2011)].mean(numeric_only=True),
        "VIIRS Era (2012-2020 Mean)": diag_df[(diag_df["year"] >= 2012) & (diag_df["year"] <= 2020)].mean(numeric_only=True)
    }
    seasons = ["Winter", "Summer", "SW_Monsoon", "NE_Monsoon"]
    x = np.arange(len(seasons))
    w = 0.25
    for i, (ep_name, ep_row) in enumerate(epochs_to_compare.items()):
        tot = sum([ep_row[f"harm_{s}"] for s in seasons])
        shares = [ep_row[f"harm_{s}"] / max(tot, 1) * 100.0 for s in seasons]
        ax_c.bar(x + (i - 1) * w, shares, width=w, label=ep_name, alpha=0.85, edgecolor="black")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(seasons, fontsize=10, fontweight="bold")
    ax_c.set_ylabel("Seasonal Fire Share (%)", fontsize=10, fontweight="bold")
    ax_c.set_title("C. Seasonal Profile Consistency Across Satellite Epochs", fontsize=11, fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax_c.legend(loc="upper right", fontsize=8)

    # Panel D: Power-Law Scaling Curve by Land Cover
    ax_d = axes[1, 1]
    with open(OUTPUT_DIR / "m4_3_nb_coefficients.json") as f:
        nb_params = json.load(f)["coefficients"]

    m_vals = np.linspace(0, 30, 100)
    log_m1 = np.log1p(m_vals)
    b_base = nb_params["log_modis1"]

    lcs = {
        "Cropland (Inland)": b_base + nb_params.get("log_modis1:C(lc_clean)[T.Cropland___Mosaic__Inland___Dryland_]", 0.0),
        "Forest & Woodland": b_base + nb_params.get("log_modis1:C(lc_clean)[T.Forest___Woodland]", 0.0),
        "Shrubland (Semi-Arid)": b_base + nb_params.get("log_modis1:C(lc_clean)[T.Shrubland___Savanna__Southern_Semi_Arid_]", 0.0),
        "Urban / Developed": b_base + nb_params.get("log_modis1:C(lc_clean)[T.Urban___Developed_Corridor]", 0.0)
    }
    for lc_name, slope in lcs.items():
        # Relative multiplier (1+M)^slope
        rel_mult = np.exp(slope * log_m1)
        ax_d.plot(m_vals, rel_mult, linewidth=2.0, label=f"{lc_name} (Slope = {slope:.2f})")

    ax_d.set_yscale("log")
    ax_d.set_xlabel("MODIS Weekly Detection Count ($M_{c,w}$)", fontsize=10, fontweight="bold")
    ax_d.set_ylabel("Relative Multiplier $(1 + M_{c,w})^{\\beta}$ [log scale]", fontsize=10, fontweight="bold")
    ax_d.set_title("D. Power-Law Exponent Scaling by Land Cover", fontsize=11, fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.4, which="both")
    ax_d.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "m4_historical_transferability.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved historical transferability figure to {fig_path}")

    elapsed = time.time() - t0
    print(f"Historical diagnostic complete in {elapsed:.1f}s!")


if __name__ == "__main__":
    main()
