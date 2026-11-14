#!/usr/bin/env python3
"""
pipeline/plot_satellite_timeline.py
===================================
NEBULENS — Milestone M3.1: Satellite Operational Timeline & Coverage Diagnostic.

Generates a publication-quality timeline figure showing:
- Operational active-fire observation lifespans across the full 2000–2026 study period
- Individual satellite platforms:
    * MODIS Terra (EOS AM-1): November 2000 – August 2026
    * MODIS Aqua (EOS PM-1): July 2002 – August 2026
    * VIIRS Suomi-NPP: January 2012 – August 2026
    * VIIRS NOAA-20 (JPSS-1): April 2018 – August 2026
    * VIIRS NOAA-21 (JPSS-2): March 2023 – August 2026
- Monthly observation intensity / detection density over Tamil Nadu (in_tn == True).
- Distinguishes satellite sensor families, operational overlap epochs, and seasonal monsoon troughs.

Output saved to:
- docs/figures/satellite_timeline_coverage.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PARQUET_OBS_DIR = PROJECT_ROOT / "data" / "parquet" / "observations"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = FIGURES_DIR / "satellite_timeline_coverage.png"

PLATFORMS = [
    {
        "code": "Terra",
        "name": "MODIS Terra (EOS AM-1)",
        "agency": "NASA (Morning ~10:30 LT)",
        "family": "MODIS (1 km)",
        "color": "#d95f02",
        "start": "2000-11-27",
        "end": "2026-08-31",
        "y": 4,
    },
    {
        "code": "Aqua",
        "name": "MODIS Aqua (EOS PM-1)",
        "agency": "NASA (Afternoon ~13:30 LT)",
        "family": "MODIS (1 km)",
        "color": "#1f78b4",
        "start": "2002-07-04",
        "end": "2026-08-31",
        "y": 3,
    },
    {
        "code": "N",
        "name": "VIIRS Suomi-NPP",
        "agency": "NASA/NOAA (Afternoon ~13:30 LT)",
        "family": "VIIRS (375 m)",
        "color": "#2ca02c",
        "start": "2012-01-20",
        "end": "2026-08-01",
        "y": 2,
    },
    {
        "code": "N20",
        "name": "VIIRS NOAA-20 (JPSS-1)",
        "agency": "NOAA/NASA (Afternoon ~12:40 LT)",
        "family": "VIIRS (375 m)",
        "color": "#9467bd",
        "start": "2018-04-01",
        "end": "2026-08-01",
        "y": 1,
    },
    {
        "code": "N21",
        "name": "VIIRS NOAA-21 (JPSS-2)",
        "agency": "NOAA/NASA (Afternoon ~14:20 LT)",
        "family": "VIIRS (375 m)",
        "color": "#e377c2",
        "start": "2023-03-01",
        "end": "2026-08-01",
        "y": 0,
    },
]


def load_monthly_in_tn_counts() -> pd.DataFrame:
    """Load and aggregate monthly in-TN detections per satellite platform."""
    print("Loading in-TN observation timestamps across all partitions...")
    files = sorted(PARQUET_OBS_DIR.rglob("*.parquet"))
    records = []
    for f in files:
        df = pd.read_parquet(f, columns=["satellite", "analysis_date_ist", "in_tn"])
        df_tn = df[df["in_tn"]].copy()
        if not df_tn.empty:
            df_tn["year_month"] = df_tn["analysis_date_ist"].str.slice(0, 7)
            records.append(df_tn[["satellite", "year_month"]])

    full_tn = pd.concat(records, ignore_index=True)
    monthly = full_tn.groupby(["satellite", "year_month"]).size().reset_index(name="detection_count")
    monthly["date"] = pd.to_datetime(monthly["year_month"] + "-01")
    return monthly


def plot_satellite_timeline(monthly_df: pd.DataFrame) -> Path:
    """Generate multi-panel publication quality timeline plot."""
    print("Generating timeline figure...")
    fig, (ax_gantt, ax_density) = plt.subplots(
        2, 1, figsize=(13, 9), height_ratios=[1.2, 1.0], sharex=True, gridspec_kw={"hspace": 0.12}
    )

    # 1. Top Panel: Operational Lifespan & Sensor Eras (Gantt style)
    min_date = pd.to_datetime("2000-10-01")
    max_date = pd.to_datetime("2026-10-01")

    # Era Shading
    # Era 1: MODIS Solo Era (2000 to 2012)
    ax_gantt.axvspan(pd.to_datetime("2000-11-01"), pd.to_datetime("2012-01-20"), color="#fee8c8", alpha=0.35, label="Era 1: MODIS Solo Record")
    # Era 2: S-NPP Overlap Baseline (2012 to 2018)
    ax_gantt.axvspan(pd.to_datetime("2012-01-20"), pd.to_datetime("2018-04-01"), color="#e5f5e0", alpha=0.40, label="Era 2: Single-VIIRS Overlap (S-NPP)")
    # Era 3: Dual-VIIRS Constellation (2018 to 2023)
    ax_gantt.axvspan(pd.to_datetime("2018-04-01"), pd.to_datetime("2023-03-01"), color="#efedf5", alpha=0.40, label="Era 3: Dual-VIIRS (S-NPP + NOAA-20)")
    # Era 4: Triple-VIIRS Constellation (2023 to 2026)
    ax_gantt.axvspan(pd.to_datetime("2023-03-01"), pd.to_datetime("2026-08-31"), color="#fde0dd", alpha=0.40, label="Era 4: Triple-VIIRS Constellation")

    for plat in PLATFORMS:
        s_date = pd.to_datetime(plat["start"])
        e_date = pd.to_datetime(plat["end"])
        y = plat["y"]
        color = plat["color"]

        # Horizontal duration bar
        ax_gantt.hlines(
            y, s_date, e_date, color=color, linewidth=10, alpha=0.9, capstyle="round"
        )
        # End markers
        ax_gantt.scatter([s_date, e_date], [y, y], color=color, s=40, zorder=4)

        # Satellite text label
        label_text = f"{plat['name']}\n{plat['agency']} | Start: {plat['start']}"
        ax_gantt.text(
            s_date + pd.Timedelta(days=60), y + 0.22, label_text,
            fontsize=8.5, weight="bold", color="#1a1a2e", va="bottom"
        )

    ax_gantt.set_yticks([p["y"] for p in PLATFORMS])
    ax_gantt.set_yticklabels([p["family"] for p in PLATFORMS], fontsize=9.5, weight="bold")
    ax_gantt.set_ylim(-0.8, 4.8)
    ax_gantt.grid(True, linestyle=":", alpha=0.6, axis="x")
    ax_gantt.set_title("NEBULENS — Satellite Active Fire Observation Timeline over Tamil Nadu (2000–2026)", fontsize=13, weight="bold", pad=12)
    ax_gantt.legend(loc="upper left", fontsize=8.5, framealpha=0.9, title="Harmonisation Epochs", title_fontsize=9)

    # 2. Bottom Panel: Detection Intensity (Log monthly detections)
    for plat in PLATFORMS:
        sat_data = monthly_df[monthly_df["satellite"] == plat["code"]].sort_values("date")
        if not sat_data.empty:
            ax_density.plot(
                sat_data["date"], sat_data["detection_count"],
                color=plat["color"], label=plat["name"], linewidth=1.5, alpha=0.85
            )

    ax_density.set_yscale("log")
    ax_density.set_ylabel("Monthly Detections (in-TN, log scale)", fontsize=10, weight="bold")
    ax_density.set_xlabel("Observation Calendar Year", fontsize=11, weight="bold", labelpad=8)
    ax_density.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_density.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    ax_density.set_xlim(min_date, max_date)

    # Format x-axis dates
    ax_density.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_density.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax_density.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Attribution & notes
    fig.text(
        0.5, 0.01,
        "Source: NASA FIRMS Standard Processing (_SP) | Region: Tamil Nadu (38 Districts, 129,914 km²) | EPSG:6933 Equal-Area Grid Framework",
        ha="center", fontsize=7.5, color="#555555"
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved timeline figure → {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return OUTPUT_FILE


def main() -> None:
    print("=" * 60)
    print("NEBULENS M3.1 — Satellite Timeline & Coverage Diagnostic")
    print("=" * 60)
    monthly_df = load_monthly_in_tn_counts()
    plot_satellite_timeline(monthly_df)
    print("\nTimeline successfully generated.")


if __name__ == "__main__":
    main()
