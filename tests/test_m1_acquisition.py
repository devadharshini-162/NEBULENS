"""
tests/test_m1_acquisition.py
=============================
M1 — Data Acquisition & Raw Storage validation tests for NEBULENS.

These tests verify:
1. Required source directories exist under data/raw/firms/
2. Manifest exists and has the required schema columns
3. No missing months within verified SP date ranges (requires full download)
4. Each manifest file path exists on disk
5. SHA-256 of each file matches its manifest value
6. No NRT version strings appear in any SP raw CSV
7. VIIRS_SNPP_SP has no data before 2012-01-20
8. VIIRS_NOAA21_SP has no data before 2023-03-01
9. Sample month row counts are non-zero for peak fire months (March)

Tests 3, 4, 5, 6, 7, 8 are skipped automatically when the full download has not
been completed yet (fewer than 90 % of expected months are present).  Tests 1,
2, 4, 5, 9 run on any partial dataset.
"""

import csv
import hashlib
import os
from datetime import date
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FIRMS_DIR = PROJECT_ROOT / "data" / "raw" / "firms"
MANIFEST_FILE = RAW_FIRMS_DIR / "manifest.csv"

SOURCES_DATE_RANGES = {
    "MODIS_SP": ("2000-11-01", "2026-08-31"),
    "VIIRS_SNPP_SP": ("2012-01-20", "2026-07-31"),
    "VIIRS_NOAA20_SP": ("2018-04-01", "2026-07-31"),
    "VIIRS_NOAA21_SP": ("2023-03-01", "2026-07-31"),
}

REQUIRED_MANIFEST_COLUMNS = {
    "source",
    "year",
    "month",
    "file_path",
    "request_template",
    "retrieval_timestamp",
    "row_count",
    "sha256",
    "chunks_count",
    "quality_type",
}

# Expected total months per source (pre-computed from date ranges above)
EXPECTED_MONTH_COUNTS = {
    "MODIS_SP": 310,
    "VIIRS_SNPP_SP": 175,
    "VIIRS_NOAA20_SP": 100,
    "VIIRS_NOAA21_SP": 41,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _month_list(start_str: str, end_str: str) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples covering the full SP range."""
    sy, sm = int(start_str[:4]), int(start_str[5:7])
    ey, em = int(end_str[:4]), int(end_str[5:7])
    months = []
    cy, cm = sy, sm
    while (cy < ey) or (cy == ey and cm <= em):
        months.append((cy, cm))
        cm += 1
        if cm > 12:
            cm = 1
            cy += 1
    return months


def _load_manifest() -> list[dict]:
    """Read manifest.csv and return list of row dicts."""
    with open(MANIFEST_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _present_months(source: str) -> set[tuple[int, int]]:
    """Return set of (year, month) tuples that have a file on disk."""
    src_dir = RAW_FIRMS_DIR / source
    present = set()
    for path in src_dir.rglob("*.csv"):
        parts = path.stem.split("_")
        # filename: {SOURCE}_{YYYY}_{MM}.csv — last two parts are YYYY and MM
        try:
            yr = int(parts[-2])
            mo = int(parts[-1])
            present.add((yr, mo))
        except (ValueError, IndexError):
            pass
    return present


def _download_completeness(source: str) -> float:
    """Return fraction of expected months present on disk (0.0–1.0)."""
    start, end = SOURCES_DATE_RANGES[source]
    expected = set(_month_list(start, end))
    present = _present_months(source)
    return len(present & expected) / len(expected) if expected else 1.0


def _full_download_complete(threshold: float = 0.90) -> bool:
    """True when all sources have >= threshold fraction of expected months."""
    return all(_download_completeness(src) >= threshold for src in SOURCES_DATE_RANGES)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


FULL_DOWNLOAD = _full_download_complete()
requires_full_download = pytest.mark.skipif(
    not FULL_DOWNLOAD,
    reason="Full historical download not yet complete (< 90 % months present); skipping completeness tests."
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestM1DirectoryStructure:
    """Source directories must exist under data/raw/firms/."""

    def test_raw_firms_dir_exists(self):
        assert RAW_FIRMS_DIR.is_dir(), f"Missing: {RAW_FIRMS_DIR}"

    @pytest.mark.parametrize("source", list(SOURCES_DATE_RANGES))
    def test_source_directory_exists(self, source):
        src_dir = RAW_FIRMS_DIR / source
        assert src_dir.is_dir(), f"Missing source directory: {src_dir}"


class TestM1Manifest:
    """Manifest file must exist and have the correct schema."""

    def test_manifest_file_exists(self):
        assert MANIFEST_FILE.is_file(), f"Missing manifest: {MANIFEST_FILE}"

    def test_manifest_has_required_columns(self):
        rows = _load_manifest()
        assert rows, "Manifest is empty"
        columns = set(rows[0].keys())
        missing = REQUIRED_MANIFEST_COLUMNS - columns
        assert not missing, f"Manifest missing columns: {missing}"

    def test_manifest_quality_type_all_standard(self):
        """Every row must declare STANDARD_QUALITY."""
        rows = _load_manifest()
        bad = [r for r in rows if r.get("quality_type") != "STANDARD_QUALITY"]
        assert not bad, f"Non-standard-quality rows in manifest: {bad[:5]}"

    def test_manifest_no_negative_row_counts(self):
        rows = _load_manifest()
        bad = [r for r in rows if int(r["row_count"]) < 0]
        assert not bad, f"Rows with negative row_count: {bad[:5]}"


class TestM1FileIntegrity:
    """Files referenced in manifest must exist and match their checksums."""

    def test_manifest_files_exist_on_disk(self):
        rows = _load_manifest()
        missing = []
        for r in rows:
            p = PROJECT_ROOT / r["file_path"]
            if not p.is_file():
                missing.append(r["file_path"])
        assert not missing, f"Manifest references {len(missing)} missing files: {missing[:5]}"

    @requires_full_download
    def test_manifest_sha256_matches_files(self):
        """SHA-256 in manifest must match current file contents."""
        rows = _load_manifest()
        mismatches = []
        for r in rows:
            p = PROJECT_ROOT / r["file_path"]
            if not p.is_file():
                continue  # Already caught by previous test
            actual = _sha256_file(p)
            expected = r["sha256"]
            if actual != expected:
                mismatches.append(
                    f"{r['file_path']}: expected {expected[:12]}… got {actual[:12]}…"
                )
        assert not mismatches, f"SHA-256 mismatches: {mismatches[:5]}"


class TestM1NoNRTData:
    """Standard-quality files must not contain any NRT version strings."""

    @requires_full_download
    def test_no_nrt_in_sp_files(self):
        """Scan every raw CSV for NRT version strings."""
        import pandas as pd

        nrt_violations = []
        for source in SOURCES_DATE_RANGES:
            src_dir = RAW_FIRMS_DIR / source
            for csv_path in sorted(src_dir.rglob("*.csv")):
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                if "version" not in df.columns:
                    continue
                nrt_mask = df["version"].astype(str).str.contains("NRT", case=False, na=False)
                if nrt_mask.any():
                    nrt_violations.append(str(csv_path.relative_to(PROJECT_ROOT)))
        assert not nrt_violations, f"NRT records found in SP files: {nrt_violations[:5]}"


class TestM1DateRanges:
    """Sensor-specific date range constraints must be respected."""

    def test_viirs_snpp_no_data_before_start(self):
        """VIIRS_SNPP_SP starts 2012-01-20; no file should predate 2012-01."""
        src_dir = RAW_FIRMS_DIR / "VIIRS_SNPP_SP"
        early = []
        for csv_path in src_dir.rglob("*.csv"):
            parts = csv_path.stem.split("_")
            try:
                yr, mo = int(parts[-2]), int(parts[-1])
                if (yr, mo) < (2012, 1):
                    early.append(str(csv_path.name))
            except (ValueError, IndexError):
                pass
        assert not early, f"VIIRS_SNPP_SP files predate start: {early}"

    def test_viirs_noaa21_no_data_before_start(self):
        """VIIRS_NOAA21_SP starts 2023-03-01; no file should predate 2023-03."""
        src_dir = RAW_FIRMS_DIR / "VIIRS_NOAA21_SP"
        early = []
        for csv_path in src_dir.rglob("*.csv"):
            parts = csv_path.stem.split("_")
            try:
                yr, mo = int(parts[-2]), int(parts[-1])
                if (yr, mo) < (2023, 3):
                    early.append(str(csv_path.name))
            except (ValueError, IndexError):
                pass
        assert not early, f"VIIRS_NOAA21_SP files predate start: {early}"

    @requires_full_download
    def test_no_missing_months_modis_sp(self):
        start, end = SOURCES_DATE_RANGES["MODIS_SP"]
        expected = set(_month_list(start, end))
        present = _present_months("MODIS_SP")
        missing = sorted(expected - present)
        assert not missing, f"MODIS_SP missing {len(missing)} months: {missing[:10]}"

    @requires_full_download
    def test_no_missing_months_viirs_snpp_sp(self):
        start, end = SOURCES_DATE_RANGES["VIIRS_SNPP_SP"]
        expected = set(_month_list(start, end))
        present = _present_months("VIIRS_SNPP_SP")
        missing = sorted(expected - present)
        assert not missing, f"VIIRS_SNPP_SP missing {len(missing)} months: {missing[:10]}"

    @requires_full_download
    def test_no_missing_months_viirs_noaa20_sp(self):
        start, end = SOURCES_DATE_RANGES["VIIRS_NOAA20_SP"]
        expected = set(_month_list(start, end))
        present = _present_months("VIIRS_NOAA20_SP")
        missing = sorted(expected - present)
        assert not missing, f"VIIRS_NOAA20_SP missing {len(missing)} months: {missing[:10]}"

    @requires_full_download
    def test_no_missing_months_viirs_noaa21_sp(self):
        start, end = SOURCES_DATE_RANGES["VIIRS_NOAA21_SP"]
        expected = set(_month_list(start, end))
        present = _present_months("VIIRS_NOAA21_SP")
        missing = sorted(expected - present)
        assert not missing, f"VIIRS_NOAA21_SP missing {len(missing)} months: {missing[:10]}"


class TestM1SampleRowCounts:
    """Peak fire-season sample months must have non-zero row counts."""

    SAMPLE_MONTHS = {
        "MODIS_SP": (2021, 3),
        "VIIRS_SNPP_SP": (2021, 3),
        "VIIRS_NOAA20_SP": (2021, 3),
        "VIIRS_NOAA21_SP": (2024, 3),  # First available March
    }

    @pytest.mark.parametrize("source,ym", list(SAMPLE_MONTHS.items()))
    def test_sample_march_rows_nonzero(self, source, ym):
        import pandas as pd
        yr, mo = ym
        csv_path = RAW_FIRMS_DIR / source / str(yr) / f"{source}_{yr}_{mo:02d}.csv"
        assert csv_path.is_file(), f"Sample file missing: {csv_path}"
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"Sample file has 0 rows: {csv_path}"


class TestM1Reconciliation:
    """3-month reconciliation: manifest row_count matches actual file row count."""

    RECONCILE_MONTHS = [
        ("MODIS_SP", 2021, 3),
        ("VIIRS_SNPP_SP", 2021, 3),
        ("VIIRS_NOAA20_SP", 2021, 3),
    ]

    @pytest.mark.parametrize("source,yr,mo", RECONCILE_MONTHS)
    def test_manifest_rowcount_matches_file(self, source, yr, mo):
        """Row count in manifest must match actual rows in the CSV file."""
        import pandas as pd

        # Find matching manifest row
        rows = _load_manifest()
        matches = [
            r for r in rows
            if r["source"] == source and int(r["year"]) == yr and int(r["month"]) == mo
        ]
        assert matches, f"No manifest entry for {source} {yr}-{mo:02d}"
        manifest_row = matches[0]

        csv_path = PROJECT_ROOT / manifest_row["file_path"]
        assert csv_path.is_file(), f"File missing: {csv_path}"

        df = pd.read_csv(csv_path)
        actual_count = len(df)
        expected_count = int(manifest_row["row_count"])

        assert actual_count == expected_count, (
            f"{source} {yr}-{mo:02d}: manifest says {expected_count} rows, "
            f"file has {actual_count} rows"
        )
