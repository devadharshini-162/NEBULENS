"""
backend/routers/meta.py
=======================
API endpoints for metadata, dataset availability, districts, and grids.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.config import settings
from backend.database import db_layer
from backend.models.schemas import (
    APIResponseEnvelope,
    AvailabilityResponse,
    DistrictInfo,
    GridCellInfo,
    ProvenanceModel,
    QueryMetadata,
)

router = APIRouter(prefix="", tags=["Metadata & Grid"])


@router.get("/meta/availability", response_model=APIResponseEnvelope[AvailabilityResponse])
def get_availability():
    """
    Returns sensor capabilities, temporal date coverage, data versions, and Gate B validation status matrix.
    """
    provenance = ProvenanceModel(
        sensor="MODIS and VIIRS Multi-Sensor",
        version=settings.VERSION,
        h_source="metadata_catalog",
        validated=True,
        validation_status="conditional_pass_gate_b",
        scientific_caveat="Pre-2012 values are model predictions; 2018+ and 2023+ contain constellation expansion."
    )
    
    data = AvailabilityResponse(
        study_region="Tamil Nadu, India",
        spatial_coverage_km2=129914.0,
        crs="EPSG:6933 (Equal-Area 10km Square Grid)",
        temporal_range={
            "start_date": "2000-11-01",
            "end_date": "2026-08-31",
            "modis_sp_start": "2000-11-01",
            "viirs_snpp_sp_start": "2012-01-20",
            "viirs_noaa20_sp_start": "2018-04-01",
            "viirs_noaa21_sp_start": "2023-03-01",
            "standard_quality_cutoff": "2026-08-31"
        },
        sensors=[
            {
                "sensor_name": "MODIS (Terra + Aqua)",
                "products": ["MOD14", "MYD14", "MODIS_SP Collection 6.1"],
                "nominal_resolution_m": 1000,
                "coverage_years": "2000-2026",
                "status": "standard_processing"
            },
            {
                "sensor_name": "VIIRS (Suomi-NPP, NOAA-20, NOAA-21)",
                "products": ["VNP14IMGTDL", "VJ114IMGTDL", "VJ214IMGTDL", "VIIRS 375m SP"],
                "nominal_resolution_m": 375,
                "coverage_years": "2012-2026",
                "status": "standard_processing"
            }
        ],
        validation_matrix={
            "2000-2002_terra_only": {
                "status": "not_validated_terra_only_pre_2002",
                "validated": False,
                "note": "Morning-only pass; lacks diurnal bridge; level shift fails."
            },
            "2002_transition": {
                "status": "conditional_pass_transition_year",
                "validated": False,
                "note": "Aqua launch in July 2002 creates intra-annual shift."
            },
            "2003-2011_modis_model_b": {
                "status": "conditional_pass_gate_b",
                "validated": False,
                "serving_eligible": True,
                "note": "Calibrated counterfactual predicting VIIRS-equivalent counts from Terra+Aqua for typical fire weeks (M <= 5). Extrapolation risks on forest mega-fires."
            },
            "2012-2017_viirs_snpp": {
                "status": "fully_validated",
                "validated": True,
                "serving_eligible": True,
                "note": "Direct VIIRS 375m observations at baseline 14 passes/week exposure."
            },
            "2018-2022_viirs_noaa20": {
                "status": "conditional_pass_opportunity_expansion",
                "validated": False,
                "serving_eligible": True,
                "note": "Constellation expanded to 28 passes/week. Exposure standardization required."
            },
            "2023-2026_viirs_noaa21": {
                "status": "conditional_pass_opportunity_expansion",
                "validated": False,
                "serving_eligible": True,
                "note": "Constellation expanded to 42 passes/week. Exposure standardization required."
            }
        },
        limitations=[
            "Daily allocated harmonised counts are disaggregated weekly estimates, not independent daily observations.",
            "MCD64A1 burned area raster data is currently unacquired (blocked by Earthdata OAuth).",
            "Constellation exposure (14, 28, 42 passes/week) represents theoretical opportunity, not cloud-adjusted actual views."
        ]
    )
    
    return APIResponseEnvelope(
        status="success",
        data=data,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=0.5,
            source_dataset="configs/project.toml + metadata catalog",
            row_count=1
        )
    )


@router.get("/meta/districts", response_model=APIResponseEnvelope[List[DistrictInfo]])
def get_districts():
    """
    Returns list of 38 official Tamil Nadu districts with grid cell count.
    """
    parquet_path = settings.ANALYSIS_DIR / "m3" / "m3_5_grid_landcover.parquet"
    query = f"""
        SELECT district_name, count(*) as cell_count_10km
        FROM '{parquet_path}'
        GROUP BY district_name
        ORDER BY district_name ASC
    """
    records, duration = db_layer.execute_query(query)
    
    district_list = [
        DistrictInfo(
            district_name=r["district_name"],
            district_code=None,
            cell_count_10km=r["cell_count_10km"]
        )
        for r in records
    ]
    
    provenance = ProvenanceModel(
        sensor="Administrative Boundaries",
        version=settings.VERSION,
        h_source="survey_of_india_clipped",
        validated=True,
        validation_status="fully_validated",
        scientific_caveat=None
    )
    
    return APIResponseEnvelope(
        status="success",
        data=district_list,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset=str(parquet_path),
            row_count=len(district_list)
        )
    )


@router.get("/grid/cells", response_model=APIResponseEnvelope[List[GridCellInfo]])
def get_grid_cells(
    resolution: str = Query("10km", description="Grid resolution: '10km' (primary) or '25km' (sensitivity)"),
    district: Optional[str] = Query(None, description="Optional filter by district name")
):
    """
    Returns spatial metadata for grid cells (EPSG:6933) with district and landcover class.
    """
    if resolution not in ["10km", "25km"]:
        raise HTTPException(status_code=400, detail="Resolution must be '10km' or '25km'")
        
    if resolution == "10km":
        parquet_path = settings.ANALYSIS_DIR / "m3" / "m3_5_grid_landcover.parquet"
        where_clause = ""
        params = []
        if district:
            where_clause = "WHERE district_name = ?"
            params.append(district)
            
        query = f"""
            SELECT grid_cell_id, res_km as resolution_km, origin_x, origin_y, district_name, landcover_class
            FROM '{parquet_path}'
            {where_clause}
            ORDER BY grid_cell_id ASC
        """
    else:  # 25km sensitivity check
        parquet_path = settings.BOUNDARIES_DIR / "grids" / "grid_25km.parquet"
        query = f"""
            SELECT grid_cell_id, res_km as resolution_km, origin_x, origin_y, NULL as district_name, NULL as landcover_class
            FROM '{parquet_path}'
            ORDER BY grid_cell_id ASC
        """
        params = []

    records, duration = db_layer.execute_query(query, params)
    
    cells = [
        GridCellInfo(
            grid_cell_id=r["grid_cell_id"],
            resolution_km=int(r["resolution_km"]),
            origin_x=float(r["origin_x"]),
            origin_y=float(r["origin_y"]),
            district_name=r.get("district_name"),
            landcover_class=r.get("landcover_class")
        )
        for r in records
    ]
    
    provenance = ProvenanceModel(
        sensor="Equal-Area Grid EPSG:6933",
        version=settings.VERSION,
        h_source="ease_grid_2",
        validated=True,
        validation_status="fully_validated",
        scientific_caveat="10km grid is the primary analysis unit; 25km is sensitivity check only."
    )
    
    return APIResponseEnvelope(
        status="success",
        data=cells,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset=str(parquet_path),
            row_count=len(cells)
        )
    )
