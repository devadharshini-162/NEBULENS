#!/usr/bin/env python3
"""
pipeline/acquire_firms.py
=========================
NASA FIRMS Standard-Quality (SP) Active Fire Data Acquisition Engine for NEBULENS.

Downloads monthly active-fire detections for Tamil Nadu:
- Sources: MODIS_SP, VIIRS_SNPP_SP, VIIRS_NOAA20_SP, VIIRS_NOAA21_SP
- Query Method: 5-day chunk bounding-box requests via FIRMS Area CSV API
- Bounding Box: Tamil Nadu extent [76.23, 8.08, 80.35, 13.56]
- Output: Immutable raw CSV per source per month under data/raw/firms/{source}/{YYYY}/{source}_{YYYY}_{MM}.csv
- Manifest: Append-only manifest with request URL, retrieval timestamp, row count, and SHA-256 checksum.
- Integrity: Resumable, idempotent, rate-limit compliant, retries with exponential backoff.
- Guarantee: Standard-quality data only (NRT strictly prohibited).
"""

import io
import os
import sys
import time
import hashlib
import calendar
import argparse
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")
MAP_KEY = os.getenv("FIRMS_MAP_KEY")

RAW_FIRMS_DIR = PROJECT_ROOT / "data" / "raw" / "firms"
MANIFEST_FILE = RAW_FIRMS_DIR / "manifest.csv"

# Bounding box of Tamil Nadu state boundary (W,S,E,N)
TAMIL_NADU_BBOX = "76.23,8.08,80.35,13.56"

# Verified historical Standard Processing (SP) date ranges from FIRMS API
SOURCES_DATE_RANGES = {
    "MODIS_SP": ("2000-11-01", "2026-08-31"),
    "VIIRS_SNPP_SP": ("2012-01-20", "2026-07-31"),
    "VIIRS_NOAA20_SP": ("2018-04-01", "2026-07-31"),
    "VIIRS_NOAA21_SP": ("2023-03-01", "2026-07-31"),
}

BASE_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def get_month_ranges(start_date_str: str, end_date_str: str) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples between start and end date strings."""
    s_parts = [int(p) for p in start_date_str.split("-")[:2]]
    e_parts = [int(p) for p in end_date_str.split("-")[:2]]
    sy, sm = s_parts[0], s_parts[1]
    ey, em = e_parts[0], e_parts[1]
    months = []
    cy, cm = sy, sm
    while (cy < ey) or (cy == ey and cm <= em):
        months.append((cy, cm))
        cm += 1
        if cm > 12:
            cm = 1
            cy += 1
    return months


def get_five_day_chunks(year: int, month: int, min_date: str, max_date: str) -> list[tuple[str, int]]:
    """
    Split a month into chunks of at most 5 days, respecting min_date and max_date bounds.
    Returns list of (start_date_str, num_days) tuples.
    """
    first_day = 1
    _, last_day = calendar.monthrange(year, month)

    min_d = datetime.strptime(min_date, "%Y-%m-%d").date()
    max_d = datetime.strptime(max_date, "%Y-%m-%d").date()

    chunks = []
    curr_day = first_day
    while curr_day <= last_day:
        chunk_start = date(year, month, curr_day)
        if chunk_start > max_d:
            break

        # Calculate chunk length (at most 5 days)
        chunk_len = min(5, last_day - curr_day + 1)
        chunk_end = date(year, month, curr_day + chunk_len - 1)

        # Adjust for min_d and max_d
        if chunk_end < min_d:
            curr_day += chunk_len
            continue

        if chunk_start < min_d:
            # Shift start
            actual_start = min_d
            actual_len = (chunk_end - actual_start).days + 1
        else:
            actual_start = chunk_start
            actual_len = chunk_len

        if chunk_end > max_d:
            actual_len = (max_d - actual_start).days + 1

        if actual_len > 0:
            chunks.append((actual_start.strftime("%Y-%m-%d"), actual_len))

        curr_day += chunk_len

    return chunks


def fetch_chunk(source: str, start_date: str, day_range: int, max_retries: int = 5) -> str:
    """Fetch a single 5-day chunk from FIRMS API with retry logic."""
    if not MAP_KEY or MAP_KEY == "your_firms_map_key_here":
        raise ValueError("Valid FIRMS_MAP_KEY not found in .env file.")

    url = f"{BASE_API_URL}/{MAP_KEY}/{source}/{TAMIL_NADU_BBOX}/{day_range}/{start_date}"
    # Mask key for safe logging
    safe_url = f"{BASE_API_URL}/[REDACTED_MAP_KEY]/{source}/{TAMIL_NADU_BBOX}/{day_range}/{start_date}"

    headers = {"User-Agent": "NEBULENS-Research-M1 (NASA Space Apps)"}

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                text = resp.text.strip()
                # Check for API error response disguised as text
                if text.startswith("Invalid") or text.startswith("Error"):
                    raise RuntimeError(f"FIRMS API returned message: {text}")
                return text
            elif resp.status_code == 429:
                wait_time = 2 ** attempt + 5
                print(f"  [Rate Limited 429] Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
            elif resp.status_code in [500, 502, 503, 504]:
                wait_time = 2 ** attempt
                print(f"  [Server Error {resp.status_code}] Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except (requests.RequestException, RuntimeError) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Failed chunk {safe_url} after {max_retries} attempts: {exc}")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Exhausted retries for {safe_url}")


def init_manifest() -> None:
    """Initialize manifest CSV if it does not exist."""
    RAW_FIRMS_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_FILE.exists():
        headers = [
            "source",
            "year",
            "month",
            "file_path",
            "request_template",
            "retrieval_timestamp",
            "row_count",
            "sha256",
            "chunks_count",
            "quality_type"
        ]
        MANIFEST_FILE.write_text(",".join(headers) + "\n")


def is_month_downloaded(source: str, year: int, month: int) -> bool:
    """Check if the monthly raw file already exists and is logged in manifest."""
    month_file = RAW_FIRMS_DIR / source / str(year) / f"{source}_{year}_{month:02d}.csv"
    if not month_file.exists():
        return False
    if not MANIFEST_FILE.exists():
        return False
    # Check manifest
    try:
        df = pd.read_csv(MANIFEST_FILE)
        match = df[(df["source"] == source) & (df["year"] == year) & (df["month"] == month)]
        return len(match) > 0
    except Exception:
        return False


def log_manifest(
    source: str,
    year: int,
    month: int,
    file_path: Path,
    row_count: int,
    sha256: str,
    chunks_count: int
) -> None:
    """Log an acquired monthly file to manifest."""
    rel_path = file_path.relative_to(PROJECT_ROOT)
    req_template = f'"{BASE_API_URL}/[MAP_KEY]/{source}/{TAMIL_NADU_BBOX}/[DAY_RANGE]/[DATE]"'
    ts = datetime.now(timezone.utc).isoformat()

    record = (
        f"{source},{year},{month},{rel_path},{req_template},{ts},{row_count},{sha256},{chunks_count},STANDARD_QUALITY\n"
    )
    with open(MANIFEST_FILE, "a") as f:
        f.write(record)


def acquire_month(source: str, year: int, month: int, min_date: str, max_date: str, delay: float = 0.1) -> Path:
    """Acquire all 5-day chunks for a month, combine into a raw immutable CSV."""
    out_dir = RAW_FIRMS_DIR / source / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{source}_{year}_{month:02d}.csv"

    chunks = get_five_day_chunks(year, month, min_date, max_date)
    if not chunks:
        print(f"  [{source}] {year}-{month:02d} is outside date range ({min_date} to {max_date}).")
        return out_file

    all_dfs = []
    header_cols = None

    for start_date, num_days in chunks:
        csv_text = fetch_chunk(source, start_date, num_days)
        time.sleep(delay)  # Politeness interval

        lines = csv_text.splitlines()
        if len(lines) <= 1:
            # Only header returned, no fires detected
            if lines and header_cols is None:
                header_cols = lines[0].split(",")
            continue

        header_cols = lines[0].split(",")
        chunk_df = pd.read_csv(io.StringIO(csv_text))

        # Scientific check: verify NO NRT records exist in standard processing stream
        if "version" in chunk_df.columns:
            nrt_mask = chunk_df["version"].astype(str).str.contains("NRT", case=False, na=False)
            if nrt_mask.any():
                raise ValueError(f"CRITICAL: Found NRT records in {source} {start_date} chunk!")

        all_dfs.append(chunk_df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Deduplicate identical records if overlapping day chunks occurred
        combined_df = combined_df.drop_duplicates()
    else:
        # Create empty DataFrame with standard headers
        combined_df = pd.DataFrame(columns=header_cols if header_cols else ["latitude", "longitude"])

    # Write raw immutable CSV
    combined_df.to_csv(out_file, index=False)

    # Compute SHA-256
    file_bytes = out_file.read_bytes()
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    row_count = len(combined_df)

    # Record in manifest
    log_manifest(source, year, month, out_file, row_count, file_sha256, len(chunks))
    print(f"  ✓ Saved {source} {year}-{month:02d}: {row_count} rows → {out_file.relative_to(PROJECT_ROOT)} (SHA: {file_sha256[:10]}...)")
    return out_file


def acquire_source(source: str, sample_only: bool = False) -> None:
    """Download historical monthly series for a given sensor source."""
    if source not in SOURCES_DATE_RANGES:
        raise ValueError(f"Unknown source '{source}'. Available: {list(SOURCES_DATE_RANGES.keys())}")

    min_date, max_date = SOURCES_DATE_RANGES[source]
    months = get_month_ranges(min_date[:7], max_date[:7])

    print("\n" + "=" * 60)
    print(f"Acquiring {source} ({len(months)} months: {min_date} to {max_date})")
    print("=" * 60)

    if sample_only:
        # If running in sample mode, acquire 3 sample months
        sample_months = [(2021, 3), (2022, 11), (2024, 3)]
        months = [m for m in months if m in sample_months]
        print(f"  [Sample Mode] Limiting to {len(months)} sample months: {months}")

    downloaded = 0
    skipped = 0
    for y, m in months:
        if is_month_downloaded(source, y, m):
            skipped += 1
            continue
        acquire_month(source, y, m, min_date, max_date)
        downloaded += 1

    print(f"Completed {source}: {downloaded} newly acquired, {skipped} already present.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NEBULENS FIRMS Data Acquisition Engine")
    parser.add_argument("--source", choices=list(SOURCES_DATE_RANGES.keys()) + ["ALL"], default="ALL",
                        help="Sensor source to download (default: ALL)")
    parser.add_argument("--sample-only", action="store_true", help="Download only sample months for testing")
    parser.add_argument("--year", type=int, help="Single year to download")
    parser.add_argument("--month", type=int, help="Single month to download")
    args = parser.parse_args()

    init_manifest()

    if args.year and args.month:
        sources = [args.source] if args.source != "ALL" else list(SOURCES_DATE_RANGES.keys())
        for src in sources:
            min_date, max_date = SOURCES_DATE_RANGES[src]
            acquire_month(src, args.year, args.month, min_date, max_date)
        return

    if args.source == "ALL":
        for src in SOURCES_DATE_RANGES.keys():
            acquire_source(src, sample_only=args.sample_only)
    else:
        acquire_source(args.source, sample_only=args.sample_only)


if __name__ == "__main__":
    main()
