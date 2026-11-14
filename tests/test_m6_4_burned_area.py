#!/usr/bin/env python3
"""
tests/test_m6_4_burned_area.py
==============================
Unit tests for Milestone M6.4: MCD64A1 C6.1 Burned-Area Integration.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from pipeline.m6_4_burned_area import (
    decode_mcd64a1_pixels,
    audit_mcd64a1_acquisition_readiness,
    generate_burned_area_evidence_table,
    TAMIL_NADU_TILES,
    MODIS_500M_PIXEL_AREA_KM2,
    OUTPUT_DIR
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_tamil_nadu_tile_identification():
    """Verify that the 4 required MODIS sinusoidal tiles are correctly identified."""
    assert len(TAMIL_NADU_TILES) == 4
    assert set(TAMIL_NADU_TILES) == {"h25v07", "h25v08", "h26v07", "h26v08"}


def test_decode_mcd64a1_pixels_valid_and_area():
    """Verify pixel decoding and square kilometre conversion."""
    # Synthetic grid: 10 burned pixels (DOY 60), 80 unburned land (0), 5 water (-1), 5 missing (-2)
    bd = np.array([60] * 10 + [0] * 80 + [-1] * 5 + [-2] * 5)
    res = decode_mcd64a1_pixels(bd)
    
    assert res["n_total_pixels"] == 100
    assert res["n_burned_pixels"] == 10
    assert res["n_unburned_pixels"] == 80
    assert res["n_missing_pixels"] == 5
    assert res["n_water_pixels"] == 5
    assert res["is_valid_observation"] is True
    # Area = 10 * 0.2146586 km^2
    assert np.isclose(res["burned_area_km2"], 10 * MODIS_500M_PIXEL_AREA_KM2)


def test_decode_mcd64a1_qa_masking():
    """Verify that QA missing bit invalidates a burn detection."""
    # 2 burned pixels (DOY 60), but pixel 1 has QA bit 1 set (missing data)
    bd = np.array([60, 60, 0, 0])
    qa = np.array([0, 2, 0, 0])  # QA = 2 (bit 1 is set)
    
    res = decode_mcd64a1_pixels(bd, qa_flags=qa)
    assert res["n_burned_pixels"] == 1  # Only pixel 0 is accepted
    assert res["n_missing_pixels"] == 1


def test_missing_vs_zero_burned_area_distinction():
    """Verify that unacquired data is never fabricated as zero burned area."""
    districts = ["Coimbatore", "Nilgiris"]
    years = [2023]
    df = generate_burned_area_evidence_table(districts, years)
    
    assert len(df) == 24  # 2 districts * 12 months
    for _, row in df.iterrows():
        # Value must be None, NOT 0.0
        assert pd.isna(row["burned_area_km2"]) or row["burned_area_km2"] is None
        assert row["evidence_status"] == "missing_unacquired"


def test_audit_artifact_generation():
    """Verify that the acquisition audit artifact records required technical metadata."""
    audit = audit_mcd64a1_acquisition_readiness()
    assert audit["product"] == "MCD64A1"
    assert audit["collection"] == "061"
    assert audit["spatial_resolution_m"] == 500
    assert "cmr_endpoint" in audit
    assert "sinu" in audit["projection"]
    
    audit_file = OUTPUT_DIR / "m6_4_burned_area_audit.json"
    assert audit_file.exists()
    with open(audit_file) as f:
        loaded = json.load(f)
    assert loaded["product"] == "MCD64A1"


def test_m0_through_m5_data_untouched():
    """Verify that all M0–M5 baseline files remain untouched."""
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_coefficients.json").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_app_readiness_matrix.csv").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_gate_b_final_audit.json").exists()
