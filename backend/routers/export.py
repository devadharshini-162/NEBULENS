"""
backend/routers/export.py
=========================
Streaming CSV data export endpoint for reproducible analysis.
"""

import io
import csv
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from backend.config import settings
from backend.database import db_layer

router = APIRouter(prefix="", tags=["Data Export"])


@router.get("/export")
def export_data(
    query_type: str = Query(..., description="Export dataset: 'series', 'anomalies', 'attribution', 'season_metrics'"),
    district: Optional[str] = Query(None, description="Optional district filter"),
    year: Optional[int] = Query(None, description="Optional year filter")
):
    """
    Returns query results directly as a comma-separated values (CSV) file download.
    """
    if query_type == "series":
        harm_pattern = str(settings.PARQUET_DIR / "harmonised_daily" / "*" / "*.parquet")
        where_clauses = ["1=1"]
        params = []
        if year:
            where_clauses.append("SUBSTRING(analysis_date_ist, 1, 4) = ?")
            params.append(str(year))
        
        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT analysis_date_ist,
                   SUM(harmonised_count) as harmonised_count,
                   SUM(harmonised_frp) as harmonised_frp,
                   SUM(modis_count_observed) as modis_count_observed,
                   SUM(viirs_count_observed) as viirs_count_observed,
                   first(h_source) as h_source
            FROM read_parquet('{harm_pattern}')
            WHERE {where_sql}
            GROUP BY analysis_date_ist
            ORDER BY analysis_date_ist ASC
        """
        filename = f"nebulens_series_{year or 'all'}.csv"
        
    elif query_type == "anomalies":
        pq_path = settings.ANALYSIS_DIR / "m6" / "m6_3_historical_comparison.parquet"
        where_clauses = ["anomaly_flag = true"]
        params = []
        if district:
            where_clauses.append("district_name = ?")
            params.append(district)
        if year:
            where_clauses.append("target_year = ?")
            params.append(year)
            
        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT district_name, month, target_year, raw_observed, evaluated_value,
                   z_score, scoring_method, baseline_median, baseline_mad, validation_status
            FROM '{pq_path}'
            WHERE {where_sql}
            ORDER BY target_year DESC, month ASC
        """
        filename = f"nebulens_anomalies_{district or 'all'}_{year or 'all'}.csv"
        
    elif query_type == "attribution":
        pq_path = settings.ANALYSIS_DIR / "m6" / "m6_5_attribution_evidence_table.parquet"
        where_clauses = ["1=1"]
        params = []
        if district:
            where_clauses.append("district_name = ?")
            params.append(district)
        if year:
            where_clauses.append("year = ?")
            params.append(year)
            
        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT district_name, month, year, observed_count, baseline_median,
                   count_z_score, anomaly_flag, exposure_passes, attribution_outcome,
                   confidence, explanation, validation_status
            FROM '{pq_path}'
            WHERE {where_sql}
            ORDER BY year DESC, month ASC
        """
        filename = f"nebulens_attribution_{district or 'all'}_{year or 'all'}.csv"
        
    elif query_type == "season_metrics":
        pq_path = settings.ANALYSIS_DIR / "m6" / "m6_2_district_season_metrics.parquet"
        where_clauses = ["1=1"]
        params = []
        if district:
            where_clauses.append("entity_name = ?")
            params.append(district)
        if year:
            where_clauses.append("year = ?")
            params.append(year)
            
        where_sql = " AND ".join(where_clauses)
        query = f"""
            SELECT year, entity_name as district_name, stream, total_detections,
                   active_fire_days, season_start_date, season_end_date,
                   season_duration_days, peak_date, peak_count, total_frp, validation_status
            FROM '{pq_path}'
            WHERE {where_sql}
            ORDER BY year DESC, district_name ASC
        """
        filename = f"nebulens_season_metrics_{district or 'all'}_{year or 'all'}.csv"
    else:
        raise HTTPException(status_code=400, detail="query_type must be 'series', 'anomalies', 'attribution', or 'season_metrics'")

    records, _ = db_layer.execute_query(query, params)
    if not records:
        raise HTTPException(status_code=404, detail="No records matched export criteria.")

    # Convert to CSV text
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    csv_content = output.getvalue()

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
