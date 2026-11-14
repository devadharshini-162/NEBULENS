"""
backend/routers/hotspots.py
===========================
Hotspot detection list and detail endpoints for granular investigation.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Path as FastPath
from backend.config import settings
from backend.database import db_layer
from backend.models.schemas import (
    APIResponseEnvelope,
    HotspotDetail,
    HotspotListResponse,
    ProvenanceModel,
    QueryMetadata,
)

router = APIRouter(prefix="", tags=["Hotspot Detections"])


@router.get("/hotspots", response_model=APIResponseEnvelope[HotspotListResponse])
def get_hotspots(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    district: Optional[str] = Query(None, description="Optional district name filter"),
    sensor: Optional[str] = Query(None, description="Optional sensor filter: 'MODIS' or 'VIIRS'"),
    confidence_class: Optional[str] = Query(None, description="Confidence filter: 'low', 'nominal', 'high'"),
    min_lon: Optional[float] = Query(None, description="Bounding box min longitude"),
    min_lat: Optional[float] = Query(None, description="Bounding box min latitude"),
    max_lon: Optional[float] = Query(None, description="Bounding box max longitude"),
    max_lat: Optional[float] = Query(None, description="Bounding box max latitude"),
    limit: int = Query(100, ge=1, le=1000, description="Page limit (max 1,000)"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Returns filterable, paginated individual hotspot detections from standard-quality observations.
    """
    where_clauses = ["analysis_date_ist >= ?", "analysis_date_ist <= ?"]
    params = [start_date, end_date]

    if district:
        where_clauses.append("district_name = ?")
        params.append(district)

    if sensor:
        if sensor.upper() == "MODIS":
            where_clauses.append("source = 'MODIS_SP'")
        elif sensor.upper() == "VIIRS":
            where_clauses.append("source LIKE 'VIIRS%'")

    if confidence_class:
        where_clauses.append("confidence_class = ?")
        params.append(confidence_class)

    if min_lon is not None and max_lon is not None and min_lat is not None and max_lat is not None:
        where_clauses.append("longitude >= ? AND longitude <= ? AND latitude >= ? AND latitude <= ?")
        params.extend([min_lon, max_lon, min_lat, max_lat])

    where_sql = " AND ".join(where_clauses)
    obs_pattern = str(settings.PARQUET_DIR / "observations" / "*" / "*" / "*.parquet")

    # Count total matching query
    count_query = f"""
        SELECT count(*) as total
        FROM read_parquet('{obs_pattern}', union_by_name=True)
        WHERE {where_sql}
    """
    count_res, _ = db_layer.execute_query(count_query, params)
    total_count = int(count_res[0]["total"]) if count_res else 0

    # Retrieve paginated items
    select_query = f"""
        SELECT observation_id, source, satellite, instrument,
               acq_date, acq_time, CAST(acq_datetime_utc AS VARCHAR) as acq_datetime_utc,
               analysis_date_ist, latitude, longitude, frp,
               confidence_raw as confidence, confidence_class, daynight,
               district_name, grid_cell_id_10km, duplicate_group
        FROM read_parquet('{obs_pattern}', union_by_name=True)
        WHERE {where_sql}
        ORDER BY analysis_date_ist DESC, acq_time DESC, observation_id ASC
        LIMIT {limit} OFFSET {offset}
    """
    records, duration = db_layer.execute_query(select_query, params)

    hotspots = [
        HotspotDetail(
            observation_id=str(r["observation_id"]),
            source=str(r["source"]),
            satellite=str(r["satellite"]),
            instrument=str(r["instrument"]),
            acq_date=str(r["acq_date"]),
            acq_time=int(r["acq_time"]) if r.get("acq_time") is not None else 0,
            acq_datetime_utc=str(r["acq_datetime_utc"]),
            analysis_date_ist=str(r["analysis_date_ist"]),
            latitude=float(r["latitude"]),
            longitude=float(r["longitude"]),
            frp=float(r["frp"]) if r.get("frp") is not None else 0.0,
            confidence=r.get("confidence"),
            confidence_class=r.get("confidence_class"),
            daynight=r.get("daynight"),
            district_name=r.get("district_name"),
            grid_cell_id_10km=r.get("grid_cell_id_10km"),
            duplicate_group=r.get("duplicate_group"),
            landcover_class=None
        )
        for r in records
    ]

    provenance = ProvenanceModel(
        sensor="NASA FIRMS Standard Processing",
        version=settings.VERSION,
        h_source="raw_satellite_observations",
        validated=True,
        validation_status="fully_validated",
        scientific_caveat="Individual thermal hotspot detections. Not every detection represents an independent fire event."
    )

    data_resp = HotspotListResponse(
        total_count=total_count,
        limit=limit,
        offset=offset,
        hotspots=hotspots
    )

    return APIResponseEnvelope(
        status="success",
        data=data_resp,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset="observations/*/*/*.parquet",
            row_count=len(hotspots)
        )
    )


@router.get("/hotspots/{observation_id}", response_model=APIResponseEnvelope[HotspotDetail])
def get_hotspot_detail(
    observation_id: str = FastPath(..., description="Unique FIRMS detection identifier")
):
    """
    Returns granular forensic detail for an individual detection, joining landcover and cell provenance.
    """
    obs_pattern = str(settings.PARQUET_DIR / "observations" / "*" / "*" / "*.parquet")
    landcover_path = settings.ANALYSIS_DIR / "m3" / "m3_5_grid_landcover.parquet"

    query = f"""
        SELECT o.observation_id, o.source, o.satellite, o.instrument,
               o.acq_date, o.acq_time, CAST(o.acq_datetime_utc AS VARCHAR) as acq_datetime_utc,
               o.analysis_date_ist, o.latitude, o.longitude, o.frp,
               o.confidence_raw as confidence, o.confidence_class, o.daynight,
               o.district_name, o.grid_cell_id_10km, o.duplicate_group,
               g.landcover_class
        FROM read_parquet('{obs_pattern}', union_by_name=True) o
        LEFT JOIN '{landcover_path}' g ON o.grid_cell_id_10km = g.grid_cell_id
        WHERE o.observation_id = ?
        LIMIT 1
    """
    records, duration = db_layer.execute_query(query, [observation_id])
    if not records:
        raise HTTPException(status_code=404, detail=f"Hotspot observation '{observation_id}' not found.")

    r = records[0]
    detail = HotspotDetail(
        observation_id=str(r["observation_id"]),
        source=str(r["source"]),
        satellite=str(r["satellite"]),
        instrument=str(r["instrument"]),
        acq_date=str(r["acq_date"]),
        acq_time=int(r["acq_time"]) if r.get("acq_time") is not None else 0,
        acq_datetime_utc=str(r["acq_datetime_utc"]),
        analysis_date_ist=str(r["analysis_date_ist"]),
        latitude=float(r["latitude"]),
        longitude=float(r["longitude"]),
        frp=float(r["frp"]) if r.get("frp") is not None else 0.0,
        confidence=r.get("confidence"),
        confidence_class=r.get("confidence_class"),
        daynight=r.get("daynight"),
        district_name=r.get("district_name"),
        grid_cell_id_10km=r.get("grid_cell_id_10km"),
        duplicate_group=r.get("duplicate_group"),
        landcover_class=r.get("landcover_class")
    )

    provenance = ProvenanceModel(
        sensor=str(r["source"]),
        version=settings.VERSION,
        h_source="firms_standard_processing",
        validated=True,
        validation_status="fully_validated",
        scientific_caveat="Thermal hotspot point measurement."
    )

    return APIResponseEnvelope(
        status="success",
        data=detail,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset="observations + m3_5_grid_landcover",
            row_count=1
        )
    )
