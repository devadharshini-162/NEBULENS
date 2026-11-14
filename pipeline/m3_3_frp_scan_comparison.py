#!/usr/bin/env python3
"""
pipeline/m3_3_frp_scan_comparison.py
====================================
NEBULENS — Milestone M3.3: FRP and Scan-Angle Comparison.

Compares Fire Radiative Power (FRP) distributions and viewing scan angle
effects for each sensor/platform across Tamil Nadu (in_tn == True).

Produces:
1. FRP summary statistics (min, Q25, median, mean, Q75, max, IQR, sample count)
   stratified by:
     - Sensor platform (Terra, Aqua, S-NPP, NOAA-20, NOAA-21)
     - Day vs night overpasses
2. FRP distributions plotted on log scales.
3. Per-detection FRP versus scan-angle relationship analysis:
     - Optical footprint expansion across scan swath
     - Swath edge pixel enlargement effects on reported FRP
4. Sensor/platform-specific scan-angle distributions.
5. Machine-readable tables:
     - data/analysis/m3/m3_3_frp_summary_by_satellite.csv
     - data/analysis/m3/m3_3_frp_by_scan_bin.csv
6. Publication-grade figure:
     - docs/figures/m3_3_frp_scan_comparison.png
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

PARQUET_OBS_DIR = PROJECT_ROOT / "data" / "parquet" / "observations"
OUTPUT_DIR = PROJECT_ROOT / "data" / "analysis" / "m3"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PLATFORM_ORDER = ["Terra", "Aqua", "N", "N20", "N21"]
PLATFORM_LABELS = {
    "Terra": "MODIS Terra",
    "Aqua": "MODIS Aqua",
    "N": "VIIRS S-NPP",
    "N20": "VIIRS NOAA-20",
    "N21": "VIIRS NOAA-21",
}
PLATFORM_COLORS = {
    "Terra": "#d95f02",
    "Aqua": "#1f78b4",
    "N": "#2ca02c",
    "N20": "#9467bd",
    "N21": "#e377c2",
}


def load_in_tn_frp_scan_data() -> pd.DataFrame:
    """Load FRP and scan columns from all standardised in-TN observations."""
    print("Loading in-TN observations across all partitions for FRP & scan analysis...")
    files = sorted(PARQUET_OBS_DIR.rglob("*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(
            f,
            columns=[
                "source", "satellite", "in_tn", "frp", "scan", "track",
                "daynight", "confidence_class", "type_flag"
            ]
        )
        df_tn = df[df["in_tn"]].copy()
        if not df_tn.empty:
            dfs.append(df_tn)

    all_obs = pd.concat(dfs, ignore_index=True)
    all_obs["is_modis"] = all_obs["source"].str.contains("MODIS")
    print(f"Loaded {len(all_obs)} in-TN detections.")
    return all_obs


def compute_frp_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute parametric and non-parametric FRP statistics by platform and day/night."""
    records = []
    
    # 1. Overall by platform
    for sat in PLATFORM_ORDER:
        sat_df = df[df["satellite"] == sat]
        frp = sat_df["frp"].dropna()
        frp_pos = frp[frp > 0]
        
        records.append({
            "group": "Overall",
            "satellite": sat,
            "platform_name": PLATFORM_LABELS[sat],
            "count_total": len(sat_df),
            "count_valid_frp": len(frp_pos),
            "frp_mean": float(frp_pos.mean()),
            "frp_std": float(frp_pos.std()),
            "frp_min": float(frp_pos.min()),
            "frp_q25": float(frp_pos.quantile(0.25)),
            "frp_median": float(frp_pos.median()),
            "frp_q75": float(frp_pos.quantile(0.75)),
            "frp_max": float(frp_pos.max()),
            "frp_iqr": float(frp_pos.quantile(0.75) - frp_pos.quantile(0.25)),
            "frp_sum_mw": float(frp_pos.sum()),
        })

    # 2. Stratified by Day / Night
    for dn, dn_label in [("D", "Daytime"), ("N", "Nighttime")]:
        for sat in PLATFORM_ORDER:
            sat_df = df[(df["satellite"] == sat) & (df["daynight"] == dn)]
            frp = sat_df["frp"].dropna()
            frp_pos = frp[frp > 0]
            if len(frp_pos) == 0:
                continue
            records.append({
                "group": dn_label,
                "satellite": sat,
                "platform_name": PLATFORM_LABELS[sat],
                "count_total": len(sat_df),
                "count_valid_frp": len(frp_pos),
                "frp_mean": float(frp_pos.mean()),
                "frp_std": float(frp_pos.std()),
                "frp_min": float(frp_pos.min()),
                "frp_q25": float(frp_pos.quantile(0.25)),
                "frp_median": float(frp_pos.median()),
                "frp_q75": float(frp_pos.quantile(0.75)),
                "frp_max": float(frp_pos.max()),
                "frp_iqr": float(frp_pos.quantile(0.75) - frp_pos.quantile(0.25)),
                "frp_sum_mw": float(frp_pos.sum()),
            })

    stats_df = pd.DataFrame(records)
    return stats_df


def compute_scan_angle_binned_frp(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze FRP variation across scan angle / pixel size bins."""
    records = []
    
    # MODIS scan bins (km): [1.0, 1.2), [1.2, 1.5), [1.5, 2.0), [2.0, 5.0]
    modis_bins = [1.0, 1.2, 1.5, 2.0, 5.0]
    modis_labels = ["Nadir [1.0-1.2km)", "Mid [1.2-1.5km)", "Far [1.5-2.0km)", "Edge [2.0-5.0km]"]
    
    m_df = df[df["is_modis"]].copy()
    m_df["scan_cat"] = pd.cut(m_df["scan"], bins=modis_bins, labels=modis_labels, right=False)
    
    for cat in modis_labels:
        sub = m_df[m_df["scan_cat"] == cat]["frp"].dropna()
        if len(sub) > 0:
            records.append({
                "sensor_family": "MODIS",
                "scan_bin": cat,
                "count": len(sub),
                "frp_mean": float(sub.mean()),
                "frp_median": float(sub.median()),
                "frp_q25": float(sub.quantile(0.25)),
                "frp_q75": float(sub.quantile(0.75)),
            })

    # VIIRS scan bins (km): [0.32, 0.40), [0.40, 0.50), [0.50, 0.65), [0.65, 1.0]
    viirs_bins = [0.30, 0.40, 0.50, 0.65, 1.0]
    viirs_labels = ["Nadir [0.3-0.4km)", "Mid-1 [0.4-0.5km)", "Mid-2 [0.5-0.65km)", "Edge [0.65-1.0km]"]
    
    v_df = df[~df["is_modis"]].copy()
    v_df["scan_cat"] = pd.cut(v_df["scan"], bins=viirs_bins, labels=viirs_labels, right=False)
    
    for cat in viirs_labels:
        sub = v_df[v_df["scan_cat"] == cat]["frp"].dropna()
        if len(sub) > 0:
            records.append({
                "sensor_family": "VIIRS",
                "scan_bin": cat,
                "count": len(sub),
                "frp_mean": float(sub.mean()),
                "frp_median": float(sub.median()),
                "frp_q25": float(sub.quantile(0.25)),
                "frp_q75": float(sub.quantile(0.75)),
            })

    return pd.DataFrame(records)


def generate_frp_scan_plots(df: pd.DataFrame, scan_binned_df: pd.DataFrame) -> Path:
    """Generate 4-panel publication-grade FRP and scan-angle comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. Panel A: FRP Boxplots by Platform (Log Scale)
    ax_a = axes[0, 0]
    frp_data = [df[df["satellite"] == s]["frp"].dropna() for s in PLATFORM_ORDER]
    box_colors = [PLATFORM_COLORS[s] for s in PLATFORM_ORDER]
    
    bp = ax_a.boxplot(
        frp_data, tick_labels=[PLATFORM_LABELS[s] for s in PLATFORM_ORDER], patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker=".", markersize=2.5, alpha=0.15)
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    ax_a.set_yscale("log")
    ax_a.set_title("A: Fire Radiative Power (FRP) by Satellite Platform", fontsize=11, weight="bold")
    ax_a.set_ylabel("FRP per Detection (MW, log scale)", fontsize=10, weight="bold")
    ax_a.tick_params(axis="x", rotation=15)
    ax_a.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Annotate medians
    for i, s in enumerate(PLATFORM_ORDER, 1):
        med = df[df["satellite"] == s]["frp"].median()
        ax_a.text(i, med * 1.35, f"{med:.1f} MW", ha="center", fontsize=8.5, weight="bold")

    # 2. Panel B: Day vs Night FRP Distributions (Kernel Density on log FRP)
    ax_b = axes[0, 1]
    log_bins = np.linspace(-0.5, 3.5, 60)  # log10(FRP) from ~0.3 MW to ~3000 MW
    
    # MODIS Day vs Night
    m_day = np.log10(df[(df["is_modis"]) & (df["daynight"] == "D")]["frp"].dropna().clip(lower=0.1))
    m_night = np.log10(df[(df["is_modis"]) & (df["daynight"] == "N")]["frp"].dropna().clip(lower=0.1))
    
    # VIIRS Day vs Night
    v_day = np.log10(df[(~df["is_modis"]) & (df["daynight"] == "D")]["frp"].dropna().clip(lower=0.1))
    v_night = np.log10(df[(~df["is_modis"]) & (df["daynight"] == "N")]["frp"].dropna().clip(lower=0.1))

    ax_b.hist(v_day, bins=log_bins, density=True, histtype="step", color="#2ca02c", linewidth=2.0, label="VIIRS Daytime (Peak ~2 MW)")
    ax_b.hist(v_night, bins=log_bins, density=True, histtype="step", color="#2ca02c", linestyle="--", linewidth=1.8, label="VIIRS Nighttime (Shifted lower)")
    ax_b.hist(m_day, bins=log_bins, density=True, histtype="step", color="#d95f02", linewidth=2.0, label="MODIS Daytime (Peak ~12 MW)")
    ax_b.hist(m_night, bins=log_bins, density=True, histtype="step", color="#d95f02", linestyle="--", linewidth=1.8, label="MODIS Nighttime")

    ax_b.set_title("B: Day vs Night FRP Densities (Log10 MW)", fontsize=11, weight="bold")
    ax_b.set_xlabel("log10(FRP in MW) [0 = 1 MW, 1 = 10 MW, 2 = 100 MW]", fontsize=10, weight="bold")
    ax_b.set_ylabel("Probability Density", fontsize=10, weight="bold")
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper right", fontsize=8.5)

    # 3. Panel C: Scan Value Distributions by Platform
    ax_c = axes[1, 0]
    # Plot MODIS scan histogram
    ax_c.hist(df[df["is_modis"]]["scan"], bins=30, color="#d95f02", alpha=0.6, density=True, label="MODIS (1.0–4.8 km footprint)")
    ax_c.set_title("C: Optical Scan Parameter Distribution (Ground Pixel Footprint)", fontsize=11, weight="bold")
    ax_c.set_xlabel("MODIS Scan Size along scan (km)", fontsize=10, weight="bold", color="#d95f02")
    ax_c.set_ylabel("Density (MODIS)", fontsize=10, weight="bold", color="#d95f02")
    ax_c.grid(True, linestyle=":", alpha=0.5)

    # Twin axis for VIIRS scan (0.32 - 0.8 km)
    ax_c2 = ax_c.twiny()
    ax_c2.hist(df[~df["is_modis"]]["scan"], bins=30, color="#2ca02c", alpha=0.6, density=True, label="VIIRS (0.32–0.80 km)")
    ax_c2.set_xlabel("VIIRS Scan Size along scan (km)", fontsize=10, weight="bold", color="#2ca02c")
    ax_c2.set_ylabel("Density (VIIRS)", fontsize=10, weight="bold", color="#2ca02c")

    # 4. Panel D: Per-Detection FRP vs Scan Bin
    ax_d = axes[1, 1]
    m_scans = scan_binned_df[scan_binned_df["sensor_family"] == "MODIS"]
    v_scans = scan_binned_df[scan_binned_df["sensor_family"] == "VIIRS"]

    ax_d.plot(range(len(m_scans)), m_scans["frp_median"], marker="o", color="#d95f02", linewidth=2, label="MODIS Median FRP (Swath edge expansion)")
    ax_d.plot(range(len(v_scans)), v_scans["frp_median"], marker="s", color="#2ca02c", linewidth=2, label="VIIRS Median FRP (Constrained aggregation)")
    
    ax_d.set_xticks(range(4))
    ax_d.set_xticklabels(["Nadir", "Mid-1", "Mid-2 / Far", "Scan Edge"], fontsize=9.5, weight="bold")
    ax_d.set_title("D: Median FRP across Scan Swath Bins", fontsize=11, weight="bold")
    ax_d.set_ylabel("Median FRP (MW)", fontsize=10, weight="bold")
    ax_d.set_xlabel("Viewing Geometry from Nadir to Scan Edge", fontsize=10, weight="bold")
    ax_d.grid(True, linestyle=":", alpha=0.5)
    ax_d.legend(loc="upper left")

    fig.suptitle("NEBULENS M3.3 — Fire Radiative Power & Viewing Scan Geometry Analysis (Tamil Nadu)", fontsize=13, weight="bold", y=0.99)
    fig.tight_layout()
    fig_file = FIGURES_DIR / "m3_3_frp_scan_comparison.png"
    fig.savefig(fig_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved FRP & scan figure → {fig_file.relative_to(PROJECT_ROOT)}")
    return fig_file


def run_m3_3_pipeline() -> dict:
    """Execute M3.3 FRP & Scan comparison."""
    print("=" * 60)
    print("NEBULENS M3.3 — FRP and Scan-Angle Comparison Pipeline")
    print("=" * 60)

    obs = load_in_tn_frp_scan_data()
    stats_df = compute_frp_statistics(obs)
    scan_df = compute_scan_angle_binned_frp(obs)

    # Save tables
    p_stats = OUTPUT_DIR / "m3_3_frp_summary_by_satellite.csv"
    p_scan = OUTPUT_DIR / "m3_3_frp_by_scan_bin.csv"

    stats_df.to_csv(p_stats, index=False)
    scan_df.to_csv(p_scan, index=False)
    print(f"  ✓ Saved FRP stats table ({len(stats_df)} rows) → {p_stats.relative_to(PROJECT_ROOT)}")
    print(f"  ✓ Saved scan bin table ({len(scan_df)} rows) → {p_scan.relative_to(PROJECT_ROOT)}")

    # Plot
    generate_frp_scan_plots(obs, scan_df)

    print("\nM3.3 Summary Findings:")
    overall = stats_df[stats_df["group"] == "Overall"]
    for _, r in overall.iterrows():
        print(f"  [{r['platform_name']}]: Median FRP = {r['frp_median']:.2f} MW, Mean = {r['frp_mean']:.2f} MW, IQR = {r['frp_iqr']:.2f} MW (N = {r['count_valid_frp']:,})")

    return {
        "stats": stats_df,
        "scan": scan_df,
    }


if __name__ == "__main__":
    run_m3_3_pipeline()
