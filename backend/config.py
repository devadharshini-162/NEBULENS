"""
backend/config.py
=================
Central configuration and environment management for the NEBULENS API.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    PROJECT_NAME: str = "NEBULENS API"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Paths
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PARQUET_DIR: Path = DATA_DIR / "parquet"
    ANALYSIS_DIR: Path = DATA_DIR / "analysis"
    BOUNDARIES_DIR: Path = DATA_DIR / "boundaries"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "https://nebulens.app",
        "*"
    ]
    
    # DuckDB Execution settings
    DUCKDB_THREADS: int = int(os.getenv("DUCKDB_THREADS", "4"))
    DUCKDB_MEMORY_LIMIT: str = os.getenv("DUCKDB_MEMORY_LIMIT", "4GB")


settings = Settings()
