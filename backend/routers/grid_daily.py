"""
backend/routers/grid_daily.py
=============================
Gridded daily values endpoint for map visualization.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.config import settings
from backend.database import db_layer
from backend.models.schemas import (
    APIResponseEnvelope,
    GridDailyRecord,
    ProvenanceModel,
    QueryMetadata,
)

router = APIRouter(prefix="", tags=["Grid Daily Map"])


@router.get("/grid/daily", response_model=APIResponseEnvelope[List[GridDailyRecord]])
def get_grid_daily(
    date: str = Query(..., description="Target date in Indian Standard Time (YYYY-MM-DD)"),
    sensor: str = Query("harmonised", description="Stream view: 'harmonised', 'modis', or 'viirs'"),
    metric: str = Query("count", description="Metric to visualize: 'count' or 'frp'")
):
    """
    Returns gridded cell-level observations or harmonised estimates for map layers on a single day.
    """
    if sensor not in ["harmonised", "modis", "viirs"]:
        raise HTTPException(status_code=400, detail="Sensor must be 'harmonised', 'modis', or 'viirs'")
    if metric not in ["count", "frp"]:
        raise HTTPException(status_code=400, detail="Metric must be 'count' or 'frp'")

    # Extract year from date to prune partitions
    try:
        year_str = date.split("-")[0]
        year_val = int(year_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD")

    if sensor == "harmonised":
        parquet_path = settings.PARQUET_DIR / "harmonised_daily" / f"year={year_val}" / f"harmonised_daily_{year_val}.parquet"
        if not parquet_path.exists():
            raise HTTPException(status_code=404, detail=f"No harmonised data found for year {year_val}")

        query = f"""
            SELECT grid_cell_id, analysis_date_ist,
                   CASE WHEN harmonised_count >= 1.0 THEN 1 ELSE 0 END AS active_fire_day,
                   harmonised_count AS count_value,
                   harmonised_frp AS frp_value,
                   h_source
            FROM '{parquet_path}'
            WHERE analysis_date_ist = ?
        """
        source_name = f"harmonised_daily_{year_val}"
    else:
        # Raw sensor view from grid_daily
        parquet_path = settings.PARQUET_DIR / "grid_daily" / f"year={year_val}" / f"grid_daily_{year_val}.parquet"
        if not parquet_path.exists():
            raise HTTPException(status_code=404, detail=f"No grid_daily data found for year {year_val}")

        if sensor == "modis":
            count_col = "n_unique_modis"
            frp_col = "frp_sum_modis"
            h_source_val = "'observed_modis'"
        else:  # viirs
            count_col = "n_unique_viirs"
            frp_col = "frp_sum_viirs"
            h_source_val = "'observed_viirs'"

        query = f"""
            SELECT grid_cell_id, analysis_date_ist,
                   active_fire_day,
                   {count_col} AS count_value,
                   COALESCE({frp_col}, 0.0) AS frp_value,
                   {h_source_val} AS h_source
            FROM '{parquet_path}'
            WHERE analysis_date_ist = ?
        """
        source_name = f"grid_daily_{year_val}"

    records, duration = db_layer.execute_query(query, [date])
    
    # Validation status logic
    serving_ok = True
    if sensor == "harmonised":
        if year_val < 2002:
            val_status = "not_validated_terra_only_pre_2002"
            is_val = False
            serving_ok = False
            caveat = "BLOCKED from trend analysis: Pre-July 2002 Terra-only estimates lack diurnal bridge."
        elif year_val == 2002:
            val_status = "conditional_pass_transition_year"
            is_val = False
            serving_ok = True
            caveat = "RESTRICTED: Aqua launch in July 2002 introduces intra-annual structural level shift."
        elif year_val < 2012:
            val_status = "conditional_pass_gate_b"
            is_val = False  # Conditional pass is NOT blanket full validation
            serving_ok = True
            caveat = "CONDITIONAL PASS: Counterfactual VIIRS predictions valid for typical fire weeks (M <= 5). Western Ghats and mega-fire extrapolation risks apply."
        elif year_val < 2018:
            val_status = "fully_validated"
            is_val = True
            serving_ok = True
            caveat = "Observed VIIRS Suomi-NPP single-satellite baseline."
        else:
            val_status = "conditional_pass_opportunity_expansion"
            is_val = False  # Multi-satellite opportunity expansion requires exposure adjustment
            serving_ok = True
            caveat = "CONDITIONAL PASS: Multi-satellite constellation expansion (NOAA-20/21). Exposure standardization required."
    elif sensor == "modis":
        val_status = "fully_validated" if year_val > 2002 else "conditional_pass_diurnal_caveat"
        is_val = (year_val > 2002)
        serving_ok = True
        caveat = "Raw MODIS Collection 6.1 active fire detections."
    else:  # viirs
        if year_val < 2012:
            val_status = "sensor_not_in_orbit"
            is_val = False
            serving_ok = False
            caveat = "VIIRS instrument was not in orbit prior to 2012."
        elif year_val < 2018:
            val_status = "fully_validated"
            is_val = True
            serving_ok = True
            caveat = "Raw VIIRS 375m active fire detections (Suomi-NPP)."
        else:
            val_status = "conditional_pass_opportunity_expansion"
            is_val = False
            serving_ok = True
            caveat = "Raw VIIRS 375m active fire detections with constellation expansion."

    provenance = ProvenanceModel(
        sensor=sensor.upper(),
        version=settings.VERSION,
        h_source="calibrated_from_modis" if (sensor == "harmonised" and year_val < 2012) else ("observed_viirs" if sensor == "harmonised" else f"observed_{sensor}"),
        validated=is_val,
        serving_eligible=serving_ok,
        validation_status=val_status,
        scientific_caveat=caveat
    )

    data_items = [
        GridDailyRecord(
            grid_cell_id=r["grid_cell_id"],
            analysis_date_ist=str(r["analysis_date_ist"]),
            active_fire_day=int(r["active_fire_day"]),
            count_value=float(r["count_value"]),
            frp_value=float(r["frp_value"]),
            h_source=r.get("h_source")
        )
        for r in records
    ]

    return APIResponseEnvelope(
        status="success",
        data=data_items,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset=source_name,
            row_count=len(data_items)
        )
    )
