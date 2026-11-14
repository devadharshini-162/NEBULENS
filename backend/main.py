"""
backend/main.py
===============
FastAPI application entry point for the NEBULENS active-fire API.
Configures CORS, middleware, global error handling, and API routes.
"""

from datetime import datetime, timezone
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.database import db_layer
from backend.models.schemas import HealthResponse, ReadinessResponse
from backend.routers import meta, grid_daily, series, hotspots, analysis, export

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Scientific API for NASA MODIS and VIIRS active-fire harmonisation, anomaly detection, and attribution in Tamil Nadu, India.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS configuration for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Structured RFC-compliant error responses."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
            "path": request.url.path
        }
    )


# System Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns basic service health status and timestamp."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["System"])
def readiness_check():
    """Verifies DuckDB engine responsiveness and Parquet data accessibility."""
    readiness = db_layer.check_readiness()
    return ReadinessResponse(
        ready=readiness["ready"],
        duckdb_connected=readiness["duckdb_connected"],
        checks=readiness["checks"]
    )


# Register API v1 Routers
api_v1_prefix = settings.API_V1_PREFIX
app.include_router(meta.router, prefix=api_v1_prefix)
app.include_router(grid_daily.router, prefix=api_v1_prefix)
app.include_router(series.router, prefix=api_v1_prefix)
app.include_router(hotspots.router, prefix=api_v1_prefix)
app.include_router(analysis.router, prefix=api_v1_prefix)
app.include_router(export.router, prefix=api_v1_prefix)

# Serve Frontend static assets if available (production / Docker build)
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="static_assets")
    
    # Catch-all route to serve React's index.html for SPA routing
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        target_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.exists(target_path) and os.path.isfile(target_path):
            return FileResponse(target_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
