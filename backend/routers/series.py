"""
backend/routers/series.py
=========================
Analytical time series endpoint supporting state, district, and grid cell aggregations.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.config import settings
from backend.database import db_layer
from backend.models.schemas import (
    APIResponseEnvelope,
    ProvenanceModel,
    QueryMetadata,
    SeriesDataPoint,
)

router = APIRouter(prefix="", tags=["Analytical Time Series"])


@router.get("/series", response_model=APIResponseEnvelope[List[SeriesDataPoint]])
def get_series(
    entity_type: str = Query("state", description="Spatial entity: 'state', 'district', or 'cell'"),
    entity_id: str = Query("Tamil Nadu", description="Entity name or grid_cell_id"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    stream: str = Query("harmonised", description="Data stream: 'harmonised', 'observed_modis', 'observed_viirs'"),
    aggregation: str = Query("daily", description="Temporal aggregation: 'daily', 'monthly', 'annual'")
):
    """
    Returns time series of fire counts, FRP, and uncertainty intervals across a date range.
    """
    if entity_type not in ["state", "district", "cell"]:
        raise HTTPException(status_code=400, detail="entity_type must be 'state', 'district', or 'cell'")
    if stream not in ["harmonised", "observed_modis", "observed_viirs"]:
        raise HTTPException(status_code=400, detail="stream must be 'harmonised', 'observed_modis', or 'observed_viirs'")
    if aggregation not in ["daily", "monthly", "annual"]:
        raise HTTPException(status_code=400, detail="aggregation must be 'daily', 'monthly', or 'annual'")

    try:
        start_year = int(start_date.split("-")[0])
        end_year = int(end_date.split("-")[0])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start_date or end_date format.")

    years = list(range(start_year, end_year + 1))
    
    # Fast path: Use precomputed state season table if annual state query
    if entity_type == "state" and aggregation == "annual":
        parquet_path = settings.ANALYSIS_DIR / "m6" / "m6_2_state_season_metrics.parquet"
        query = f"""
            SELECT CAST(year AS VARCHAR) AS date,
                   total_detections AS count,
                   COALESCE(total_frp, 0.0) AS frp,
                   NULL AS count_lower_80, NULL AS count_upper_80,
                   NULL AS count_lower_95, NULL AS count_upper_95,
                   NULL AS exposure_passes,
                   provenance AS h_source,
                   CASE WHEN stream = 'harmonised' AND year < 2012 THEN true ELSE false END AS is_counterfactual_prediction
            FROM '{parquet_path}'
            WHERE stream = ? AND year >= ? AND year <= ?
            ORDER BY year ASC
        """
        records, duration = db_layer.execute_query(query, [stream, start_year, end_year])
        source_name = "m6_2_state_season_metrics.parquet"
    else:
        # Dynamic query over harmonised_daily partitions
        # Resolve spatial join if district
        if entity_type == "district":
            landcover_path = settings.ANALYSIS_DIR / "m3" / "m3_5_grid_landcover.parquet"
            spatial_filter = f"h.grid_cell_id IN (SELECT grid_cell_id FROM '{landcover_path}' WHERE district_name = ?)"
            spatial_param = [entity_id]
        elif entity_type == "cell":
            spatial_filter = "h.grid_cell_id = ?"
            spatial_param = [entity_id]
        else:
            spatial_filter = "1=1"
            spatial_param = []

        # Stream column selection
        if stream == "harmonised":
            count_col = "h.harmonised_count"
            frp_col = "h.harmonised_frp"
            uncertainty_select = """
                AVG(h.count_lower_80) AS count_lower_80,
                AVG(h.count_upper_80) AS count_upper_80,
                AVG(h.count_lower_95) AS count_lower_95,
                AVG(h.count_upper_95) AS count_upper_95,
            """
        elif stream == "observed_modis":
            count_col = "h.modis_count_observed"
            frp_col = "h.modis_frp_observed"
            uncertainty_select = "NULL AS count_lower_80, NULL AS count_upper_80, NULL AS count_lower_95, NULL AS count_upper_95,"
        else:  # observed_viirs
            count_col = "h.viirs_count_observed"
            frp_col = "h.viirs_frp_observed"
            uncertainty_select = "NULL AS count_lower_80, NULL AS count_upper_80, NULL AS count_lower_95, NULL AS count_upper_95,"

        # Temporal grouping
        if aggregation == "daily":
            time_expr = "h.analysis_date_ist"
            group_by = "h.analysis_date_ist, h.h_source"
        elif aggregation == "monthly":
            time_expr = "SUBSTRING(h.analysis_date_ist, 1, 7)"
            group_by = "SUBSTRING(h.analysis_date_ist, 1, 7), h.h_source"
        else:  # annual
            time_expr = "SUBSTRING(h.analysis_date_ist, 1, 4)"
            group_by = "SUBSTRING(h.analysis_date_ist, 1, 4), h.h_source"

        harm_pattern = str(settings.PARQUET_DIR / "harmonised_daily" / "*" / "*.parquet")
        query = f"""
            SELECT {time_expr} AS date,
                   SUM({count_col}) AS count,
                   COALESCE(SUM({frp_col}), 0.0) AS frp,
                   {uncertainty_select}
                   AVG(CASE 
                       WHEN CAST(SUBSTRING(h.analysis_date_ist, 1, 4) AS INT) < 2018 THEN 14.0
                       WHEN CAST(SUBSTRING(h.analysis_date_ist, 1, 4) AS INT) < 2023 THEN 28.0
                       ELSE 42.0 END) AS exposure_passes,
                   h.h_source,
                   CASE WHEN h.h_source = 'calibrated_from_modis' THEN true ELSE false END AS is_counterfactual_prediction
            FROM read_parquet('{harm_pattern}') h
            WHERE {spatial_filter}
              AND h.analysis_date_ist >= ?
              AND h.analysis_date_ist <= ?
            GROUP BY {group_by}
            ORDER BY date ASC
        """
        params = spatial_param + [start_date, end_date]
        records, duration = db_layer.execute_query(query, params)
        source_name = "harmonised_daily"

    points = [
        SeriesDataPoint(
            date=str(r["date"]),
            count=round(float(r["count"]), 3),
            frp=round(float(r["frp"]), 2),
            count_lower_80=round(float(r["count_lower_80"]), 3) if r.get("count_lower_80") is not None else None,
            count_upper_80=round(float(r["count_upper_80"]), 3) if r.get("count_upper_80") is not None else None,
            count_lower_95=round(float(r["count_lower_95"]), 3) if r.get("count_lower_95") is not None else None,
            count_upper_95=round(float(r["count_upper_95"]), 3) if r.get("count_upper_95") is not None else None,
            exposure_passes=round(float(r["exposure_passes"]), 1) if r.get("exposure_passes") is not None else None,
            h_source=r.get("h_source"),
            is_counterfactual_prediction=bool(r.get("is_counterfactual_prediction", False))
        )
        for r in records
    ]

    has_pre_2002 = any(p.date < "2002-07" for p in points)
    has_pre_2012 = any(p.date < "2012" for p in points)
    has_post_2018 = any(p.date >= "2018" for p in points)

    if stream == "harmonised":
        if has_pre_2002:
            val_status = "not_validated_terra_only_pre_2002"
            is_val = False
            serving_ok = False
            caveat = "Contains pre-July 2002 Terra-only estimates uncalibrated for diurnal cycle. Blocked from cross-era trend analysis."
        elif has_pre_2012 or has_post_2018:
            val_status = "conditional_pass_gate_b"
            is_val = False  # Conditional pass requires qualification
            serving_ok = True
            caveat = "CONDITIONAL PASS: Pre-2012 values are Model B counterfactuals valid for typical fire weeks. Constellation opportunity expansion applies to post-2018."
        else:
            val_status = "fully_validated"
            is_val = True
            serving_ok = True
            caveat = "Observed VIIRS Suomi-NPP baseline era."
    else:
        val_status = "fully_validated" if not has_pre_2002 else "conditional_pass_diurnal_caveat"
        is_val = not has_pre_2002
        serving_ok = True
        caveat = "Observed satellite detections."

    provenance = ProvenanceModel(
        sensor=stream.upper(),
        version=settings.VERSION,
        h_source="mixed" if (has_pre_2012 and any(p.date >= "2012" for p in points)) else (points[0].h_source if points else "none"),
        validated=is_val,
        serving_eligible=serving_ok,
        validation_status=val_status,
        scientific_caveat=caveat
    )

    return APIResponseEnvelope(
        status="success",
        data=points,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset=source_name,
            row_count=len(points)
        )
    )
