#!/usr/bin/env python3
"""
pipeline/plot_boundaries.py
===========================
Plots Tamil Nadu state and district boundaries over a geographic basemap.

Loads the processed GeoParquet files from data/boundaries/ and generates
a figure saved to docs/figures/tamil_nadu_boundaries.png.

Requires internet access for the contextily basemap tile download.
If the basemap is unavailable (no network or tile server error) the script
will warn clearly and fall back to a plain white background — the boundary
data is still plotted correctly.

Usage
-----
    python pipeline/plot_boundaries.py

Output
------
    docs/figures/tamil_nadu_boundaries.png
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

STATE_FILE = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_state.parquet"
DISTRICTS_FILE = PROJECT_ROOT / "data" / "boundaries" / "tamil_nadu_districts.parquet"
OUTPUT_FILE = PROJECT_ROOT / "docs" / "figures" / "tamil_nadu_boundaries.png"

# Contextily requires Web Mercator (EPSG:3857) for basemap alignment
BASEMAP_CRS = "EPSG:3857"


def load_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load state and district GeoParquet files."""
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            f"State boundary not found: {STATE_FILE}\n"
            "Run 'python pipeline/fetch_boundaries.py' first."
        )
    if not DISTRICTS_FILE.exists():
        raise FileNotFoundError(
            f"District boundaries not found: {DISTRICTS_FILE}\n"
            "Run 'python pipeline/fetch_boundaries.py' first."
        )
    state = gpd.read_parquet(STATE_FILE)
    districts = gpd.read_parquet(DISTRICTS_FILE)
    print(f"  Loaded state    : {len(state)} feature(s), CRS: {state.crs}")
    print(f"  Loaded districts: {len(districts)} feature(s), CRS: {districts.crs}")
    return state, districts


def reproject_for_basemap(
    state: gpd.GeoDataFrame, districts: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reproject both layers to Web Mercator for contextily basemap."""
    state_wm = state.to_crs(BASEMAP_CRS)
    districts_wm = districts.to_crs(BASEMAP_CRS)
    return state_wm, districts_wm


def district_label_points(districts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return representative points for district labels."""
    pts = districts.copy()
    pts["geometry"] = districts.geometry.representative_point()
    return pts


def add_basemap(ax: plt.Axes) -> bool:
    """
    Attempt to add a contextily basemap with CartoDB Positron fallback.
    Returns True if successful, False if network is unavailable.
    """
    try:
        import contextily as ctx
        ctx.add_basemap(
            ax,
            crs=BASEMAP_CRS,
            source=ctx.providers.CartoDB.PositronNoLabels,
            attribution_size=6,
            alpha=0.6,
        )
        return True
    except Exception:
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=BASEMAP_CRS,
                source=ctx.providers.OpenStreetMap.Mapnik,
                attribution_size=6,
                alpha=0.5,
            )
            return True
        except Exception:
            # Fallback to a clean neutral cartographic background
            ax.set_facecolor("#edf2f7")
            return False


def plot_boundaries(
    state: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
    state_wm: gpd.GeoDataFrame,
    districts_wm: gpd.GeoDataFrame,
) -> plt.Figure:
    """Create the boundary figure."""
    fig, ax = plt.subplots(figsize=(8, 11))

    # --- Draw district fills (subtle) ---
    districts_wm.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#444444",
        linewidth=0.6,
        zorder=2,
    )

    # --- Draw state outline (thick) ---
    state_wm.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#C0392B",
        linewidth=2.0,
        zorder=3,
    )

    # --- Basemap ---
    basemap_ok = add_basemap(ax)

    # --- District labels ---
    name_col = "NAME_2" if "NAME_2" in districts.columns else None
    if name_col:
        pts_wm = district_label_points(districts_wm)
        pts_wm[name_col] = districts[name_col].values
        for _, row in pts_wm.iterrows():
            name = str(row[name_col]) if row[name_col] else ""
            ax.annotate(
                name,
                xy=(row.geometry.x, row.geometry.y),
                fontsize=5.5,
                ha="center",
                va="center",
                color="#1a1a2e",
                path_effects=[
                    pe.withStroke(linewidth=1.5, foreground="white")
                ],
                zorder=4,
            )

    # --- Axes formatting ---
    ax.set_axis_off()
    ax.set_title(
        "Tamil Nadu — State and District Boundaries",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    # --- Attribution ---
    basemap_note = (
        "Basemap: © CartoDB / © OpenStreetMap contributors" if basemap_ok else "No external basemap"
    )
    fig.text(
        0.5, 0.01,
        f"Boundary data: geoBoundaries 9469f09 (ODbL 1.0) | 38 Administrative Districts | {basemap_note}",
        ha="center",
        fontsize=6,
        color="#666666",
    )

    return fig


def main() -> None:
    print("=" * 60)
    print("NEBULENS — Tamil Nadu Boundary Plot")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading boundary files …")
    state, districts = load_boundaries()

    print("\nReprojecting to Web Mercator for basemap …")
    state_wm, districts_wm = reproject_for_basemap(state, districts)

    print("\nRendering figure …")
    fig = plot_boundaries(state, districts, state_wm, districts_wm)

    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n  ✓  Figure saved → {OUTPUT_FILE.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
