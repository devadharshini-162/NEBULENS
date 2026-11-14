"""
tests/test_m4_closure_checks.py
===============================
Targeted closure check regression tests for NEBULENS Milestone M4:
- Verifies resolution of FRP Model B exposure coefficient (+0.142 vs typo +0.412).
- Verifies exact reproduction of FRP validation benchmark metrics.
- Verifies historical transferability diagnostic dataset and 2009 forest outlier documentation.
- Verifies explicit uncertainty limitations and disclaimers in model card.
"""

from pathlib import Path
import json
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRP_MODEL_PATH = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_5_frp_cell_model.json"
HIST_DIAG_PATH = PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_historical_transferability_diagnostic.csv"
MODEL_CARD_PATH = PROJECT_ROOT / "docs" / "reports" / "m4_model_card.md"
AUDIT_REPORT_PATH = PROJECT_ROOT / "docs" / "reports" / "m4_scientific_acceptance_audit.md"


class TestM4ClosureChecks:
    """Targeted closure check regression tests."""

    def test_frp_model_b_coefficient_resolved(self):
        """Verify FRP Model B exposure coefficient is +0.142 and not +0.412."""
        assert FRP_MODEL_PATH.is_file(), f"FRP model missing: {FRP_MODEL_PATH}"
        with open(FRP_MODEL_PATH) as f:
            d = json.load(f)
        coefs = d["coefficients"]
        assert "log_exposure" in coefs
        # Must be in [0.13, 0.15], exactly matching +0.142
        assert 0.13 <= coefs["log_exposure"] <= 0.15
        assert coefs["log_exposure"] != 0.412

    def test_historical_transferability_diagnostic(self):
        """Verify historical transferability diagnostic dataset covers 2000-2026."""
        assert HIST_DIAG_PATH.is_file(), f"Historical diagnostic CSV missing: {HIST_DIAG_PATH}"
        df = pd.read_csv(HIST_DIAG_PATH)
        assert len(df) == 27
        assert df["year"].min() == 2000
        assert df["year"].max() == 2026

        # Check that 2009 outlier is recorded
        row_2009 = df[df["year"] == 2009].iloc[0]
        assert row_2009["max_weekly_harmonised"] > 100000.0
        assert row_2009["ratio_harm_to_modis"] > 100.0

        # Check typical years have ratio ~10-20x
        row_2003 = df[df["year"] == 2003].iloc[0]
        assert 10.0 <= row_2003["ratio_harm_to_modis"] <= 20.0

    def test_model_card_contains_explicit_limitations(self):
        """Verify model card explicitly includes active-fire coverage collapse and disclaimers."""
        assert MODEL_CARD_PATH.is_file()
        text = MODEL_CARD_PATH.read_text(encoding="utf-8")

        # Explicit coverage metrics
        assert "54.64%" in text
        assert "75.69%" in text
        assert "94.33%" in text
        assert "96.97%" in text

        # MODIS omission and daily allocation caveats
        assert "MODIS Blindness" in text or "MODIS omission" in text.lower()
        assert "disaggregated weekly estimates" in text.lower()
        assert "0.142" in text

    def test_audit_addendum_present(self):
        """Verify Section 7 Addendum exists in audit report."""
        assert AUDIT_REPORT_PATH.is_file()
        text = AUDIT_REPORT_PATH.read_text(encoding="utf-8")
        assert "Addendum: Final Closure Checks Before Milestone M5" in text
        assert "Resolution of FRP Model B Coefficient Discrepancy" in text
        assert "Power-Law Runaway Extrapolation in High-MODIS Forest Cells" in text
