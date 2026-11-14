#!/usr/bin/env python3
"""
pipeline/check_environment.py
=============================
Verifies that all required packages for NEBULENS Milestone M0 import
cleanly and prints their exact pinned versions.
"""

import sys
from importlib.metadata import version

REQUIRED_PACKAGES = [
    ("pandas", "pandas"),
    ("geopandas", "geopandas"),
    ("pyarrow", "pyarrow"),
    ("duckdb", "duckdb"),
    ("shapely", "shapely"),
    ("pyproj", "pyproj"),
    ("statsmodels", "statsmodels"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pytest", "pytest"),
    ("python-dotenv", "dotenv"),
    ("matplotlib", "matplotlib"),
    ("contextily", "contextily"),
]

def main() -> int:
    print("=" * 60)
    print("NEBULENS — Environment Verification")
    print(f"Python: {sys.version}")
    print("=" * 60)
    failed = False
    for pkg_name, mod_name in REQUIRED_PACKAGES:
        try:
            mod = __import__(mod_name)
            ver = version(pkg_name)
            print(f"  ✓ {pkg_name:<16} : {ver}")
        except Exception as exc:
            print(f"  ✗ {pkg_name:<16} : FAILED ({exc})")
            failed = True

    print("=" * 60)
    if failed:
        print("ENVIRONMENT VERIFICATION FAILED")
        return 1
    print("ALL REQUIRED PACKAGES VERIFIED SUCCESSFULLY")
    return 0

if __name__ == "__main__":
    sys.exit(main())
