"""
generate_presentation.py
========================
Generates a 7-slide NASA Space Apps Challenge presentation deck for NEBULENS.
Uses high-resolution figures, dark NASA-themed styling, and structured content.
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PROJECT_ROOT = Path("/home/dharshu/Documents/NASA_SPACE_APPS/NEBULENS")
FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_PPTX = PROJECT_ROOT / "docs" / "NEBULENS_Presentation.pptx"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

DARK_BG = RGBColor(10, 15, 30)
CARD_BG = RGBColor(18, 24, 46)
TEXT_WHITE = RGBColor(248, 250, 252)
TEXT_MUTED = RGBColor(148, 163, 184)
ACCENT_BLUE = RGBColor(56, 189, 248)
ACCENT_ORANGE = RGBColor(249, 115, 22)
ACCENT_GREEN = RGBColor(34, 197, 94)
ACCENT_YELLOW = RGBColor(250, 204, 21)


def set_bg(slide):
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = DARK_BG
    bg_shape.line.fill.background()
    return bg_shape


def add_header(slide, title_text, category_text="NASA SPACE APPS CHALLENGE 2026"):
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_BLUE

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE


# -----------------------------------------------------------------------------
# SLIDE 1: Title & Team Slide
# -----------------------------------------------------------------------------
s1 = prs.slides.add_slide(blank_layout)
set_bg(s1)

tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.3), Inches(5.0))
tf1 = tb1.text_frame
tf1.word_wrap = True

p1 = tf1.paragraphs[0]
p1.text = "NASA SPACE APPS CHALLENGE 2026"
p1.font.size = Pt(14)
p1.font.bold = True
p1.font.color.rgb = ACCENT_BLUE

p2 = tf1.add_paragraph()
p2.text = "NEBULENS"
p2.font.size = Pt(54)
p2.font.bold = True
p2.font.color.rgb = TEXT_WHITE

p3 = tf1.add_paragraph()
p3.text = "Multi-Decadal Active-Fire Harmonisation & Burning Activity Calendar"
p3.font.size = Pt(22)
p3.font.color.rgb = ACCENT_ORANGE
p3.space_after = Pt(24)

p4 = tf1.add_paragraph()
p4.text = "Bridging 24+ Years of MODIS (1km) & VIIRS (375m) Hotspots Across Tamil Nadu, India"
p4.font.size = Pt(15)
p4.font.color.rgb = TEXT_MUTED
p4.space_after = Pt(36)

p5 = tf1.add_paragraph()
p5.text = "Challenge: Harmonization of MODIS and VIIRS Hot Spots\nDomain: 1,460 Equal-Area Cells | 38 Districts | 2000–2026 Record\nRepository: https://github.com/devadharshini-162/NEBULENS.git"
p5.font.size = Pt(13)
p5.font.color.rgb = ACCENT_YELLOW


# -----------------------------------------------------------------------------
# SLIDE 2: The Discrepancy Problem (With Timeline & Detection Gap Image)
# -----------------------------------------------------------------------------
s2 = prs.slides.add_slide(blank_layout)
set_bg(s2)
add_header(s2, "The Multi-Decadal Sensor Discontinuity: Why Raw Data Fails")

# Left Column: Problem Explanation
tb2 = s2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.4))
tf2 = tb2.text_frame
tf2.word_wrap = True

items_s2 = [
    ("The 4x Detection Explosion", "VIIRS (375 m) detects 3x to 5x more active fires than MODIS (1 km) in identical fire environments due to fine spatial resolution and sub-pixel detection down to ~0.5 MW."),
    ("The Diurnal Revisit Asymmetry", "Terra passes in the morning (~10:30 IST); Aqua and VIIRS pass in peak afternoon (~13:30 IST). The VIIRS constellation expanded from 14 passes/wk (2012) to 42 passes/wk (2023 with NOAA-20 & NOAA-21)."),
    ("The Policy Risk", "An unharmonised historical analysis creates an artificial surge after 2012, misattributing sensor upgrades to worsening climate crises or failed agricultural fire policies."),
    ("The Research Mandate", "Build a continuous, counterfactual VIIRS-equivalent baseline from 2000 to present with rigorous uncertainty quantification.")
]

for title, desc in items_s2:
    p = tf2.add_paragraph()
    p.text = f"• {title}: "
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = ACCENT_ORANGE
    run = p.add_run()
    run.text = desc
    run.font.bold = False
    run.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(12)

# Right Column: Visual Figure
fig_s2 = FIG_DIR / "satellite_timeline_coverage.png"
if fig_s2.exists():
    s2.shapes.add_picture(str(fig_s2), Inches(6.8), Inches(1.6), width=Inches(5.7))


# -----------------------------------------------------------------------------
# SLIDE 3: Mathematical Harmonisation Engine
# -----------------------------------------------------------------------------
s3 = prs.slides.add_slide(blank_layout)
set_bg(s3)
add_header(s3, "Mathematical Framework: Two-Part Hurdle GLM + Model B")

tb3 = s3.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.4))
tf3 = tb3.text_frame
tf3.word_wrap = True

items_s3 = [
    ("Spatial Equal-Area Normalisation", "Reprojected WGS84 coordinates to EPSG:6933 (10 km EASE-Grid 2.0). All 1,460 cells across Tamil Nadu are exactly 100 km2, eliminating latitude distortion."),
    ("Part 1: Logistic Fire Occurrence Model", "Logit P(Y > 0) = a0 + a1 ln(M_modis + 1) + a2 ClimFire + a3 Landcover. Predicts probability of active burning in the grid cell."),
    ("Part 2: Zero-Truncated Negative Binomial GLM", "ln E[Y | Y > 0] = b0 + b1 ln(M_modis) + ln(Exposure_w). Overdispersion parameter alpha = 1.04 accommodates extreme fire clustering without variance compression."),
    ("Part 3: Huber Robust FRP Calibration", "FRP_viirs = beta * FRP_modis^(gamma). Huber loss (epsilon=1.35) prevents high-intensity wildfires from skewing agricultural baseline calibrations.")
]

for title, desc in items_s3:
    p = tf3.add_paragraph()
    p.text = f"• {title}:\n  "
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = ACCENT_BLUE
    run = p.add_run()
    run.text = desc
    run.font.bold = False
    run.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(10)

fig_s3 = FIG_DIR / "m4_4_hurdle_comparison.png"
if fig_s3.exists():
    s3.shapes.add_picture(str(fig_s3), Inches(7.0), Inches(1.6), width=Inches(5.5))


# -----------------------------------------------------------------------------
# SLIDE 4: Empirical Validation & Benchmark Metrics
# -----------------------------------------------------------------------------
s4 = prs.slides.add_slide(blank_layout)
set_bg(s4)
add_header(s4, "Scientific Validation: Spatial Holdouts & Multi-Decadal Performance")

tb4 = s4.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.4))
tf4 = tb4.text_frame
tf4.word_wrap = True

items_s4 = [
    ("Strict Spatial & Temporal Holdouts", "Trained on overlapping Aqua-VIIRS observations (2012–2021) and evaluated on independent spatial blocks (MAE = 0.38 fires/cell-week, Spearman rho = 0.84)."),
    ("Serving Eligibility & Quality Gates", "Full harmonisation is served for typical burning regimes (M <= 5). Extreme catastrophic fire fronts (M > 5) are flagged with scientific caveats to avoid over-prediction."),
    ("1,000-Iteration Bootstrap Uncertainty", "Every harmonised estimate provides 80% and 95% non-parametric bootstrap confidence intervals, capturing both parameter and residual variance."),
    ("Pre-July 2002 Scientific Guardrail", "Pre-July 2002 Terra-only records lack Aqua afternoon overpass data. NEBULENS marks these as 'blocked' rather than fabricating false afternoon diurnal profiles.")
]

for title, desc in items_s4:
    p = tf4.add_paragraph()
    p.text = f"• {title}:\n  "
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = ACCENT_GREEN
    run = p.add_run()
    run.text = desc
    run.font.bold = False
    run.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(10)

fig_s4 = FIG_DIR / "m4_8_harmonised_timeseries.png"
if fig_s4.exists():
    s4.shapes.add_picture(str(fig_s4), Inches(6.8), Inches(1.6), width=Inches(5.7))


# -----------------------------------------------------------------------------
# SLIDE 5: The NEBULENS Geospatial Decision Platform
# -----------------------------------------------------------------------------
s5 = prs.slides.add_slide(blank_layout)
set_bg(s5)
add_header(s5, "The NEBULENS Interactive Decision Platform")

tb5 = s5.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.4))
tf5 = tb5.text_frame
tf5.word_wrap = True

items_s5 = [
    ("CONAE-Inspired Cartographic Interface", "High-resolution satellite imagery overlaid with EPSG:6933 10 km grid cells, active thermal hotspots (VIIRS cyan, MODIS orange), and district administrative boundaries."),
    ("Arbitrary Polygon Area-of-Interest Tool", "Users outline custom fire perimeters. The system calculates hectares burned, total FRP, and individual satellite detections within the drawn polygon in real time."),
    ("Dynamic Inset Detail View", "Automatically generates a secondary zoom frame isolating the active fire cluster with high-contrast false-color cartography."),
    ("District & Granule Filtering", "Instant access to all 38 Tamil Nadu districts or full statewide aggregates with custom date ranges spanning 2000 to 2026.")
]

for title, desc in items_s5:
    p = tf5.add_paragraph()
    p.text = f"• {title}:\n  "
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = ACCENT_YELLOW
    run = p.add_run()
    run.text = desc
    run.font.bold = False
    run.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(10)

# Visual: Boundaries figure as placeholder
fig_s5 = FIG_DIR / "tamil_nadu_boundaries_clean.png"
if fig_s5.exists():
    s5.shapes.add_picture(str(fig_s5), Inches(6.8), Inches(1.6), width=Inches(5.7))


# -----------------------------------------------------------------------------
# SLIDE 6: Operational Modules: Calendar, Anomalies & Attribution
# -----------------------------------------------------------------------------
s6 = prs.slides.add_slide(blank_layout)
set_bg(s6)
add_header(s6, "Operational Modules: Burning Calendar & Anomaly Attribution")

tb6 = s6.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.4))
tf6 = tb6.text_frame
tf6.word_wrap = True

items_s6 = [
    ("Burning Activity Calendar", "Monthly & seasonal burning climatology categorising normal fire cycles vs out-of-season activity across agricultural crop residue and forest fire regimes."),
    ("MAD Anomaly Detection Engine", "Rolling 3-week Median Absolute Deviation baselines flag statistically significant fire surges (> 3.0x MAD) while filtering sensor artifacts."),
    ("Multi-Evidence Attribution", "Cross-references fire timing against landcover classes, day/night ratios, and meteorological wind vectors to distinguish agricultural burning from forest wildfires."),
    ("Standardized Export Engine", "One-click export of calibrated time-series CSVs, anomaly records, and markdown-formatted scientific synthesis reports.")
]

for title, desc in items_s6:
    p = tf6.add_paragraph()
    p.text = f"• {title}:\n  "
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = ACCENT_ORANGE
    run = p.add_run()
    run.text = desc
    run.font.bold = False
    run.font.color.rgb = TEXT_MUTED
    p.space_after = Pt(10)

fig_s6 = FIG_DIR / "m3_5_landcover_stratified.png"
if fig_s6.exists():
    s6.shapes.add_picture(str(fig_s6), Inches(7.0), Inches(1.6), width=Inches(5.5))


# -----------------------------------------------------------------------------
# SLIDE 7: Open Science, Cloud Deployment & Real-World Impact
# -----------------------------------------------------------------------------
s7 = prs.slides.add_slide(blank_layout)
set_bg(s7)
add_header(s7, "Open Science, Architecture & Global Impact")

tb7 = s7.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(5.4))
tf7 = tb7.text_frame
tf7.word_wrap = True

p_sum = tf7.paragraphs[0]
p_sum.text = "A Complete, Reproducible, and Production-Grade Geospatial Architecture"
p_sum.font.size = Pt(18)
p_sum.font.bold = True
p_sum.font.color.rgb = ACCENT_BLUE
p_sum.space_after = Pt(16)

sections_s7 = [
    ("Cloud-Native Packaging", "Containerised via multi-stage Docker (Bun + Python 3.12-slim) with Render blueprint configuration for instant public deployment."),
    ("Embedded Scientific Integrity", "172 backend tests (100% pass) and 16 frontend tests verifying provenance immutability and mathematical boundary conditions."),
    ("Global Extensibility", "While demonstrated across 38 districts in Tamil Nadu, the 10 km EASE-Grid 2.0 framework and Hurdle GLM engine can be deployed for any wildfire-prone state worldwide."),
    ("Links & Artifacts", "Code: https://github.com/devadharshini-162/NEBULENS.git\nDocumentation: METHODOLOGY.md | Model Card | Data Dictionary\nChallenge: NASA Space Apps 2026")
]

for title, desc in sections_s7:
    p = tf7.add_paragraph()
    p.text = f"{title}: "
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = ACCENT_YELLOW
    run = p.add_run()
    run.text = desc
    run.font.bold = False
    run.font.color.rgb = TEXT_WHITE
    p.space_after = Pt(12)

prs.save(str(OUTPUT_PPTX))
print(f"Presentation saved successfully to: {OUTPUT_PPTX}")
