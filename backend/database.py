"""
backend/database.py
===================
Reusable DuckDB data access layer over Parquet datasets.
Executes parameterized queries with thread-safe connection pooling and automatic timing.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd
from backend.config import settings


class DuckDBLayer:
    """
    DuckDB query execution layer.
    Uses in-memory DuckDB connection configured to read Parquet files directly.
    """
    def __init__(self):
        self._con: Optional[duckdb.DuckDBPyConnection] = None

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(database=":memory:", read_only=False)
            self._con.execute(f"SET threads TO {settings.DUCKDB_THREADS};")
            self._con.execute(f"SET memory_limit = '{settings.DUCKDB_MEMORY_LIMIT}';")
        # Return a cursor for thread-safe concurrent execution
        return self._con.cursor()

    def execute_query(
        self,
        query: str,
        params: Optional[List[Any]] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Executes a parameterized query and returns (records, execution_time_ms).
        """
        start_t = time.perf_counter()
        cursor = self.get_connection()
        try:
            if params:
                result = cursor.execute(query, params)
            else:
                result = cursor.execute(query)
            
            df = result.df()
            duration_ms = (time.perf_counter() - start_t) * 1000.0
            
            # Clean up DataFrame (handle timestamps and NaN)
            records = df.replace({float("nan"): None}).to_dict(orient="records")
            return records, duration_ms
        finally:
            cursor.close()

    def check_readiness(self) -> Dict[str, Any]:
        """
        Verifies DuckDB engine responsiveness and existence of essential datasets.
        """
        checks = []
        all_ready = True
        
        # Test DuckDB
        try:
            cursor = self.get_connection()
            val = cursor.execute("SELECT 42 AS test").fetchone()[0]
            cursor.close()
            checks.append({
                "name": "DuckDB In-Memory Engine",
                "available": (val == 42),
                "path": ":memory:",
                "details": "Engine initialized and executing successfully"
            })
        except Exception as e:
            all_ready = False
            checks.append({
                "name": "DuckDB In-Memory Engine",
                "available": False,
                "path": ":memory:",
                "details": str(e)
            })

        # Dataset checks
        datasets_to_check = [
            ("Grid Daily Parquet", settings.PARQUET_DIR / "grid_daily"),
            ("Harmonised Daily Parquet", settings.PARQUET_DIR / "harmonised_daily"),
            ("Observations Parquet", settings.PARQUET_DIR / "observations"),
            ("Districts GeoParquet", settings.BOUNDARIES_DIR / "tamil_nadu_districts.parquet"),
            ("Grid 10km GeoParquet", settings.BOUNDARIES_DIR / "grids" / "grid_10km.parquet"),
            ("Landcover Grid", settings.ANALYSIS_DIR / "m3" / "m3_5_grid_landcover.parquet"),
            ("Season Metrics Parquet", settings.ANALYSIS_DIR / "m6" / "m6_2_state_season_metrics.parquet"),
            ("Historical Comparison", settings.ANALYSIS_DIR / "m6" / "m6_3_historical_comparison.parquet"),
            ("Attribution Table", settings.ANALYSIS_DIR / "m6" / "m6_5_attribution_evidence_table.parquet"),
        ]

        for name, p in datasets_to_check:
            exists = p.exists()
            if not exists:
                all_ready = False
            checks.append({
                "name": name,
                "available": exists,
                "path": str(p),
                "details": "Ready for scanning" if exists else "Missing required dataset file"
            })

        return {
            "ready": all_ready,
            "duckdb_connected": True,
            "checks": checks
        }


db_layer = DuckDBLayer()
