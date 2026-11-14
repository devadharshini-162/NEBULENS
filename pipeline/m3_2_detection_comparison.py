#!/usr/bin/env python3
"""
pipeline/m3_2_detection_comparison.py
=====================================
NEBULENS — Milestone M3.2: MODIS versus VIIRS Detection Comparison.

Compares MODIS (Terra + Aqua) and VIIRS (S-NPP, NOAA-20, NOAA-21) active-fire
detection counts over their actual operational overlap period (2012-01-20 to 2026-08-31)
within Tamil Nadu (in_tn == True).

Produces:
1. Daily, weekly (ISO calendar week), and monthly aggregated count comparison tables.
2. Separate day vs night comparisons.
3. Seasonal breakdown:
     - Winter / Dry Post-Monsoon (Jan–Feb)
     - Summer / Peak Fire Season (Mar–May)
     - Southwest Monsoon (Jun–Sep)
     - Northeast Monsoon (Oct–Dec)
4. Ratio metrics:
     - VIIRS-to-MODIS count ratio (handling zero denominators: raw ratio, Laplace-smoothed +1 ratio)
5. Statistical summaries:
     - Pearson and Spearman correlations, R-squared, MAE, RMSE, median ratio.
6. Publication-quality figure:
     - docs/figures/m3_2_detection_comparison.png
7. Machine-readable data tables:
     - data/analysis/m3/m3_2_daily_comparison.parquet
     - data/analysis/m3/m3_2_weekly_comparison.parquet
     - data/analysis/m3/m3_2_monthly_comparison.parquet
     - data/analysis/m3/m3_2_summary_metrics.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GRID_DAILY_DIR = PROJECT_ROOT / "data" / "parquet" / "grid_daily"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m3"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

START_DATE_OVERLAP = "2012-01-20"
END_DATE_OVERLAP = "2026-08-31"


def get_season(month: int) -> str:
    """Map calendar month to Tamil Nadu agro-climatic season."""
    if month in [1, 2]:
        return "Winter / Dry (Jan-Feb)"
    elif month in [3, 4, 5]:
        return "Summer / Peak Fire (Mar-May)"
    elif month in [6, 7, 8, 9]:
        return "SW Monsoon (Jun-Sep)"
    else:
        return "NE Monsoon (Oct-Dec)"


def load_daily_regional_totals() -> pd.DataFrame:
    """Load grid_daily parquet partitions for overlap years (2012-2026) and aggregate regionally."""
    print("Loading daily grid aggregations across Tamil Nadu for overlap period (2012–2026)...")
    years = list(range(2012, 2027))
    dfs = []
    for y in years:
        f = GRID_DAILY_DIR / f"year={y}" / f"grid_daily_{y}.parquet"
        if not f.is_file():
            continue
        df = pd.read_parquet(
            f,
            columns=[
                "analysis_date_ist",
                "n_det_modis", "n_unique_modis", "frp_sum_modis",
                "n_det_day_modis", "n_det_night_modis",
                "n_det_viirs", "n_unique_viirs", "frp_sum_viirs",
                "n_det_day_viirs", "n_det_night_viirs",
            ]
        )
        dfs.append(df)

    full_gd = pd.concat(dfs, ignore_index=True)
    # Filter to exact overlap dates
    full_gd = full_gd[full_gd["analysis_date_ist"].between(START_DATE_OVERLAP, END_DATE_OVERLAP)]

    # Regional daily sum across all 1,460 cells
    daily = full_gd.groupby("analysis_date_ist").sum().reset_index()
    daily["date"] = pd.to_datetime(daily["analysis_date_ist"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["week"] = daily["date"].dt.isocalendar().week
    daily["year_week"] = daily["date"].dt.strftime("%G-W%V")
    daily["year_month"] = daily["date"].dt.strftime("%Y-%m")
    daily["season"] = daily["month"].apply(get_season)

    # Ratios
    # Zero-denominator handling:
    # 1. raw ratio where modis > 0
    # 2. smoothed ratio (viirs + 1) / (modis + 1)
    daily["ratio_raw"] = np.where(daily["n_det_modis"] > 0, daily["n_det_viirs"] / daily["n_det_modis"], np.nan)
    daily["ratio_smoothed"] = (daily["n_det_viirs"] + 1.0) / (daily["n_det_modis"] + 1.0)

    print(f"Aggregated {len(daily)} daily regional totals ({daily['date'].min().date()} to {daily['date'].max().date()}).")
    return daily


def compute_weekly_and_monthly(daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate daily totals into weekly and monthly series."""
    # Weekly aggregation
    weekly = daily_df.groupby("year_week").agg({
        "date": "min",
        "n_det_modis": "sum",
        "n_unique_modis": "sum",
        "frp_sum_modis": "sum",
        "n_det_day_modis": "sum",
        "n_det_night_modis": "sum",
        "n_det_viirs": "sum",
        "n_unique_viirs": "sum",
        "frp_sum_viirs": "sum",
        "n_det_day_viirs": "sum",
        "n_det_night_viirs": "sum",
    }).reset_index()
    weekly["month"] = weekly["date"].dt.month
    weekly["season"] = weekly["month"].apply(get_season)
    weekly["ratio_raw"] = np.where(weekly["n_det_modis"] > 0, weekly["n_det_viirs"] / weekly["n_det_modis"], np.nan)
    weekly["ratio_smoothed"] = (weekly["n_det_viirs"] + 1.0) / (weekly["n_det_modis"] + 1.0)

    # Monthly aggregation
    monthly = daily_df.groupby("year_month").agg({
        "date": "min",
        "n_det_modis": "sum",
        "n_unique_modis": "sum",
        "frp_sum_modis": "sum",
        "n_det_day_modis": "sum",
        "n_det_night_modis": "sum",
        "n_det_viirs": "sum",
        "n_unique_viirs": "sum",
        "frp_sum_viirs": "sum",
        "n_det_day_viirs": "sum",
        "n_det_night_viirs": "sum",
    }).reset_index()
    monthly["month"] = monthly["date"].dt.month
    monthly["season"] = monthly["month"].apply(get_season)
    monthly["ratio_raw"] = np.where(monthly["n_det_modis"] > 0, monthly["n_det_viirs"] / monthly["n_det_modis"], np.nan)
    monthly["ratio_smoothed"] = (monthly["n_det_viirs"] + 1.0) / (monthly["n_det_modis"] + 1.0)

    return weekly, monthly


def compute_metrics(x: pd.Series, y: pd.Series, name: str) -> dict:
    """Compute agreement and correlation statistics between MODIS and VIIRS series."""
    # Filter positive pairs for power-law / ratio analysis
    pos_mask = (x > 0) & (y > 0)
    pos_x = x[pos_mask]
    pos_y = y[pos_mask]

    r_pearson, p_pearson = stats.pearsonr(x, y) if len(x) > 2 else (np.nan, np.nan)
    r_spearman, p_spearman = stats.spearmanr(x, y) if len(x) > 2 else (np.nan, np.nan)
    r_squared = r_pearson ** 2 if not np.isnan(r_pearson) else np.nan

    ratios = pos_y / pos_x if len(pos_x) > 0 else pd.Series(dtype=float)

    return {
        "comparison_level": name,
        "n_periods": len(x),
        "n_both_active": int(pos_mask.sum()),
        "total_modis_detections": int(x.sum()),
        "total_viirs_detections": int(y.sum()),
        "overall_detection_ratio": float(y.sum() / x.sum()) if x.sum() > 0 else np.nan,
        "median_period_ratio": float(ratios.median()) if len(ratios) > 0 else np.nan,
        "mean_period_ratio": float(ratios.mean()) if len(ratios) > 0 else np.nan,
        "q25_ratio": float(ratios.quantile(0.25)) if len(ratios) > 0 else np.nan,
        "q75_ratio": float(ratios.quantile(0.75)) if len(ratios) > 0 else np.nan,
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "spearman_rho": float(r_spearman),
        "spearman_p": float(p_spearman),
        "r_squared": float(r_squared),
        "mae": float(np.mean(np.abs(y - x))),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
    }


def generate_plots(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame) -> Path:
    """Generate 4-panel publication-grade detection comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Panel A: Monthly Time Series Overlap (2012-2026)
    ax_a = axes[0, 0]
    ax_a.plot(monthly["date"], monthly["n_det_modis"], color="#d95f02", label="MODIS (Terra + Aqua)", linewidth=1.8, marker="o", markersize=3.5)
    ax_a.plot(monthly["date"], monthly["n_det_viirs"], color="#2ca02c", label="VIIRS (S-NPP, N20, N21)", linewidth=1.8, marker="s", markersize=3.5)
    ax_a.set_yscale("log")
    ax_a.set_title("A: Monthly Active Fire Detections in Tamil Nadu (2012–2026)", fontsize=11, weight="bold")
    ax_a.set_ylabel("Monthly Detections (log scale)", fontsize=10, weight="bold")
    ax_a.set_xlabel("Year", fontsize=10, weight="bold")
    ax_a.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_a.legend(loc="upper left", framealpha=0.9)

    # 2. Panel B: Weekly Scatter Plot with Linear & 1:1 Reference
    ax_b = axes[0, 1]
    # Filter active weeks
    active_weeks = weekly[(weekly["n_det_modis"] > 0) | (weekly["n_det_viirs"] > 0)]
    sc = ax_b.scatter(
        active_weeks["n_det_modis"], active_weeks["n_det_viirs"],
        c=active_weeks["month"], cmap="viridis", alpha=0.75, edgecolors="none", s=28
    )
    cbar = fig.colorbar(sc, ax=ax_b, label="Month of Year")
    cbar.set_ticks(range(1, 13, 2))
    
    # Reference line y = 10x
    max_val = max(active_weeks["n_det_modis"].max(), 100)
    ref_x = np.linspace(1, max_val, 100)
    ax_b.plot(ref_x, ref_x * 10, "k--", alpha=0.6, label="Reference y = 10x")
    ax_b.plot(ref_x, ref_x, "k:", alpha=0.4, label="Reference 1:1")
    
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_title("B: Weekly Count Relationship (MODIS vs VIIRS)", fontsize=11, weight="bold")
    ax_b.set_xlabel("MODIS Weekly Detections (log scale)", fontsize=10, weight="bold")
    ax_b.set_ylabel("VIIRS Weekly Detections (log scale)", fontsize=10, weight="bold")
    ax_b.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper left")

    # 3. Panel C: Seasonal Ratio Boxplots (Daily)
    ax_c = axes[1, 0]
    season_order = [
        "Winter / Dry (Jan-Feb)",
        "Summer / Peak Fire (Mar-May)",
        "SW Monsoon (Jun-Sep)",
        "NE Monsoon (Oct-Dec)"
    ]
    season_data = []
    season_labels = ["Winter\n(Jan-Feb)", "Summer\n(Mar-May)", "SW Monsoon\n(Jun-Sep)", "NE Monsoon\n(Oct-Dec)"]
    for s in season_order:
        s_ratios = daily[(daily["season"] == s) & (daily["n_det_modis"] > 0)]["ratio_raw"].dropna()
        season_data.append(s_ratios)

    ax_c.boxplot(season_data, tick_labels=season_labels, patch_artist=True,
                 boxprops=dict(facecolor="#c7e9c0", color="#2ca02c"),
                 medianprops=dict(color="#006d2c", linewidth=2),
                 flierprops=dict(marker=".", markersize=3, alpha=0.3))
    ax_c.set_yscale("log")
    ax_c.set_title("C: VIIRS-to-MODIS Daily Detection Ratio by Season", fontsize=11, weight="bold")
    ax_c.set_ylabel("Ratio: VIIRS / MODIS (log scale)", fontsize=10, weight="bold")
    ax_c.grid(True, linestyle=":", alpha=0.5, axis="y")

    # 4. Panel D: Day vs Night Comparison
    ax_d = axes[1, 1]
    # Aggregate monthly day and night
    ax_d.scatter(monthly["n_det_day_modis"], monthly["n_det_day_viirs"], color="#e6550d", alpha=0.8, label="Daytime Passes", s=35)
    ax_d.scatter(monthly["n_det_night_modis"], monthly["n_det_night_viirs"], color="#3182bd", alpha=0.8, label="Nighttime Passes", s=35, marker="^")
    
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_title("D: Day vs Night Monthly Relationship", fontsize=11, weight="bold")
    ax_d.set_xlabel("MODIS Monthly Detections (log scale)", fontsize=10, weight="bold")
    ax_d.set_ylabel("VIIRS Monthly Detections (log scale)", fontsize=10, weight="bold")
    ax_d.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_d.legend(loc="upper left")

    fig.suptitle("NEBULENS M3.2 — MODIS versus VIIRS Active-Fire Detection Comparison in Tamil Nadu (2012–2026)", fontsize=13, weight="bold", y=0.99)
    fig.tight_layout()
    fig_file = FIGURES_DIR / "m3_2_detection_comparison.png"
    fig.savefig(fig_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved comparison figure → {fig_file.relative_to(PROJECT_ROOT)}")
    return fig_file


def run_m3_2_pipeline() -> dict:
    """Execute M3.2 comparison and emit tables and metrics."""
    print("=" * 60)
    print("NEBULENS M3.2 — Detection Count Comparison Pipeline")
    print("=" * 60)

    daily = load_daily_regional_totals()
    weekly, monthly = compute_weekly_and_monthly(daily)

    # Save data tables
    p_daily = OUTPUT_DIR / "m3_2_daily_comparison.parquet"
    p_weekly = OUTPUT_DIR / "m3_2_weekly_comparison.parquet"
    p_monthly = OUTPUT_DIR / "m3_2_monthly_comparison.parquet"

    daily.to_parquet(p_daily, index=False)
    weekly.to_parquet(p_weekly, index=False)
    monthly.to_parquet(p_monthly, index=False)
    print(f"  ✓ Saved daily table ({len(daily)} rows) → {p_daily.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Saved weekly table ({len(weekly)} rows) → {p_weekly.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Saved monthly table ({len(monthly)} rows) → {p_monthly.relative_to(PROJECT_ROOT)}")

    # Compute metrics across time scales
    metrics_list = [
        compute_metrics(daily["n_det_modis"], daily["n_det_viirs"], "Daily (Regional)"),
        compute_metrics(weekly["n_det_modis"], weekly["n_det_viirs"], "Weekly (Regional)"),
        compute_metrics(monthly["n_det_modis"], monthly["n_det_viirs"], "Monthly (Regional)"),
        compute_metrics(monthly["n_det_day_modis"], monthly["n_det_day_viirs"], "Monthly Daytime Only"),
        compute_metrics(monthly["n_det_night_modis"], monthly["n_det_night_viirs"], "Monthly Nighttime Only"),
    ]

    # Seasonal metrics (monthly)
    for s in ["Winter / Dry (Jan-Feb)", "Summer / Peak Fire (Mar-May)", "SW Monsoon (Jun-Sep)", "NE Monsoon (Oct-Dec)"]:
        m_s = monthly[monthly["season"] == s]
        metrics_list.append(compute_metrics(m_s["n_det_modis"], m_s["n_det_viirs"], f"Monthly {s}"))

    metrics_df = pd.DataFrame(metrics_list)
    p_metrics = OUTPUT_DIR / "m3_2_summary_metrics.csv"
    metrics_df.to_csv(p_metrics, index=False)
    print(f"  ✓ Saved summary metrics table ({len(metrics_df)} rows) → {p_metrics.relative_to(PROJECT_ROOT)}")

    # Plot
    generate_plots(daily, weekly, monthly)

    print("\nM3.2 Summary Findings:")
    for _, row in metrics_df.head(3).iterrows():
        print(f"  [{row['comparison_level']}]: Pearson r = {row['pearson_r']:.3f} (R² = {row['r_squared']:.3f}), Spearman rho = {row['spearman_rho']:.3f}, Overall Ratio = {row['overall_detection_ratio']:.2f}x (Median = {row['median_period_ratio']:.2f}x)")

    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "metrics": metrics_df,
    }


if __name__ == "__main__":
    run_m3_2_pipeline()
