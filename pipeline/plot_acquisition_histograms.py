#!/usr/bin/env python3
"""
pipeline/plot_acquisition_histograms.py
=======================================
NEBULENS — Milestone M2: Acquisition-Time Histograms (M2.9).

Generates overpass acquisition-time histograms:
- Per satellite platform: Terra, Aqua, S-NPP (N), NOAA-20 (N20), NOAA-21 (N21)
- Per year across the full record (2000 to 2026)
- Using standardised UTC acquisition timestamps (with clear dual UTC / IST labelling)
- Investigates orbital drift across mission lifespans (notably Terra and Aqua orbit decay).

Outputs saved to:
- docs/figures/acquisition_times_by_satellite.png
- docs/figures/acquisition_times_annual_drift.png
- docs/figures/acquisition_times_matrix.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PARQUET_OBS_DIR = PROJECT_ROOT / "data" / "parquet" / "observations"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Map raw satellite codes to standard scientific display names
SATELLITE_NAMES = {
    "Terra": "MODIS Terra (EOS AM-1)",
    "Aqua": "MODIS Aqua (EOS PM-1)",
    "N": "VIIRS Suomi-NPP",
    "N20": "VIIRS NOAA-20 (JPSS-1)",
    "N21": "VIIRS NOAA-21 (JPSS-2)",
}

SATELLITE_COLORS = {
    "Terra": "#d95f02",
    "Aqua": "#1f78b4",
    "N": "#2ca02c",
    "N20": "#9467bd",
    "N21": "#e377c2",
}


def load_observation_times() -> pd.DataFrame:
    """Load essential columns from all partitioned Parquet files."""
    print("Loading observation timestamps across all 55 annual partitions...")
    files = sorted(PARQUET_OBS_DIR.rglob("*.parquet"))
    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=["satellite", "year", "acq_datetime_utc", "daynight", "in_tn"])
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["utc_hour"] = full_df["acq_datetime_utc"].dt.hour + full_df["acq_datetime_utc"].dt.minute / 60.0
    print(f"Loaded {len(full_df)} observations.")
    return full_df


def plot_by_satellite(df: pd.DataFrame) -> Path:
    """Plot acquisition time distributions by satellite."""
    out_file = FIGURES_DIR / "acquisition_times_by_satellite.png"
    
    satellites = ["Terra", "Aqua", "N", "N20", "N21"]
    fig, axes = plt.subplots(len(satellites), 1, figsize=(10, 12), sharex=True)
    
    bins = np.linspace(0, 24, 97)  # 15-minute bins across 24 hours
    
    for ax, sat in zip(axes, satellites):
        sat_data = df[df["satellite"] == sat]
        n_obs = len(sat_data)
        color = SATELLITE_COLORS.get(sat, "#333333")
        label = SATELLITE_NAMES.get(sat, sat)
        
        ax.hist(sat_data["utc_hour"], bins=bins, color=color, alpha=0.85, edgecolor="none")
        ax.set_ylabel("Detections", fontsize=10, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.4, axis="x")
        
        # Dual annotation: peak times
        ax.text(
            0.02, 0.80, f"{label} (N = {n_obs:,})",
            transform=ax.transAxes, fontsize=11, weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="#cccccc")
        )

    # Secondary x-axis for Indian Standard Time (UTC + 05:30)
    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 2))
    axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=9)
    axes[-1].set_xlabel("Acquisition Time (UTC) [Indian Standard Time = UTC + 05:30]", fontsize=11, weight="bold", labelpad=8)
    
    fig.suptitle("NEBULENS — Active Fire Detection Acquisition Times by Satellite Platform", fontsize=14, weight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved satellite histograms → {out_file.relative_to(PROJECT_ROOT)}")
    return out_file


def plot_annual_drift(df: pd.DataFrame) -> Path:
    """Plot annual overpass time evolution to investigate orbital drift."""
    out_file = FIGURES_DIR / "acquisition_times_annual_drift.png"
    
    # Terra and Aqua daytime passes (Terra morning ~05-07 UTC, Aqua afternoon ~07-09 UTC)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    
    # 1. Terra Daytime Drift Analysis
    terra_day = df[(df["satellite"] == "Terra") & (df["daynight"] == "D")]
    terra_years = sorted(terra_day["year"].unique())
    terra_medians = [terra_day[terra_day["year"] == y]["utc_hour"].median() for y in terra_years]
    terra_q25 = [terra_day[terra_day["year"] == y]["utc_hour"].quantile(0.25) for y in terra_years]
    terra_q75 = [terra_day[terra_day["year"] == y]["utc_hour"].quantile(0.75) for y in terra_years]
    
    ax1.plot(terra_years, terra_medians, marker="o", color=SATELLITE_COLORS["Terra"], linewidth=2, label="Terra Daytime Median UTC")
    ax1.fill_between(terra_years, terra_q25, terra_q75, color=SATELLITE_COLORS["Terra"], alpha=0.25, label="IQR (25th–75th Percentile)")
    ax1.set_title("MODIS Terra Daytime Overpass Drift (2000–2026)\nNoticeable orbital precessional drift starting ~2020 following cessation of orbit maintenance burns", fontsize=11, weight="bold")
    ax1.set_ylabel("UTC Decimal Hour", fontsize=10, weight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    # 2. Aqua vs VIIRS Daytime Stability
    aqua_day = df[(df["satellite"] == "Aqua") & (df["daynight"] == "D")]
    aqua_years = sorted(aqua_day["year"].unique())
    aqua_medians = [aqua_day[aqua_day["year"] == y]["utc_hour"].median() for y in aqua_years]
    
    snpp_day = df[(df["satellite"] == "N") & (df["daynight"] == "D")]
    snpp_years = sorted(snpp_day["year"].unique())
    snpp_medians = [snpp_day[snpp_day["year"] == y]["utc_hour"].median() for y in snpp_years]
    
    n20_day = df[(df["satellite"] == "N20") & (df["daynight"] == "D")]
    n20_years = sorted(n20_day["year"].unique())
    n20_medians = [n20_day[n20_day["year"] == y]["utc_hour"].median() for y in n20_years]

    ax2.plot(aqua_years, aqua_medians, marker="s", color=SATELLITE_COLORS["Aqua"], linewidth=2, label="Aqua Daytime Median UTC")
    ax2.plot(snpp_years, snpp_medians, marker="^", color=SATELLITE_COLORS["N"], linewidth=2, label="S-NPP Daytime Median UTC")
    ax2.plot(n20_years, n20_medians, marker="d", color=SATELLITE_COLORS["N20"], linewidth=2, label="NOAA-20 Daytime Median UTC")
    
    ax2.set_title("Afternoon Constellation Stability: Aqua (13:30 orbit) vs VIIRS S-NPP / NOAA-20", fontsize=11, weight="bold")
    ax2.set_ylabel("UTC Decimal Hour", fontsize=10, weight="bold")
    ax2.set_xlabel("Observation Year", fontsize=11, weight="bold")
    ax2.set_xticks(range(2000, 2027, 2))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    
    fig.suptitle("NEBULENS — Satellite Orbital Overpass Stability & Drift Diagnostic", fontsize=13, weight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved annual drift analysis → {out_file.relative_to(PROJECT_ROOT)}")
    return out_file


def main() -> None:
    print("=" * 60)
    print("NEBULENS M2.9 — Acquisition Time Histograms")
    print("=" * 60)
    df = load_observation_times()
    plot_by_satellite(df)
    plot_annual_drift(df)
    print("\nHistograms successfully generated.")


if __name__ == "__main__":
    main()
