"""
backend/routers/analysis.py
===========================
API endpoints for seasonal comparison, anomalies, attribution, and observation coverage.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.config import settings
from backend.database import db_layer
from backend.models.schemas import (
    APIResponseEnvelope,
    AttributionRecord,
    CoverageRecord,
    ProvenanceModel,
    QueryMetadata,
    SeasonalComparisonRecord,
)

router = APIRouter(prefix="", tags=["Analysis & Attribution"])


@router.get("/compare/seasonal", response_model=APIResponseEnvelope[List[SeasonalComparisonRecord]])
def get_seasonal_comparison(
    entity_name: str = Query("Tamil Nadu (Statewide)", description="District name or 'Tamil Nadu (Statewide)'"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Target calendar month (1-12)"),
    target_year: Optional[int] = Query(None, ge=2012, le=2026, description="Target evaluation year")
):
    """
    Returns same-season comparisons across years, evaluating the current season against its historical baseline.
    """
    parquet_path = settings.ANALYSIS_DIR / "m6" / "m6_3_historical_comparison.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=503, detail="Historical comparison dataset not available.")

    where_clauses = ["district_name = ?"]
    params = [entity_name]

    if month is not None:
        where_clauses.append("month = ?")
        params.append(month)
    if target_year is not None:
        where_clauses.append("target_year = ?")
        params.append(target_year)

    where_sql = " AND ".join(where_clauses)
    query = f"""
        SELECT district_name, month, target_year, raw_observed, evaluated_value,
               is_exposure_standardized, z_score, anomaly_flag, scoring_method,
               baseline_median, baseline_mad, baseline_n_years, target_h_source,
               validation_status
        FROM '{parquet_path}'
        WHERE {where_sql}
        ORDER BY target_year DESC, month ASC
    """
    records, duration = db_layer.execute_query(query, params)

    items = [
        SeasonalComparisonRecord(
            district_name=r["district_name"],
            month=int(r["month"]),
            target_year=int(r["target_year"]),
            raw_observed=float(r["raw_observed"]),
            evaluated_value=float(r["evaluated_value"]),
            is_exposure_standardized=bool(r["is_exposure_standardized"]),
            z_score=float(r["z_score"]) if r.get("z_score") is not None else None,
            anomaly_flag=bool(r["anomaly_flag"]),
            scoring_method=str(r["scoring_method"]),
            baseline_median=float(r["baseline_median"]) if r.get("baseline_median") is not None else None,
            baseline_mad=float(r["baseline_mad"]) if r.get("baseline_mad") is not None else None,
            baseline_n_years=int(r["baseline_n_years"]),
            target_h_source=str(r["target_h_source"]),
            validation_status=str(r["validation_status"])
        )
        for r in records
    ]

    provenance = ProvenanceModel(
        sensor="Seasonal Baseline Engine",
        version=settings.VERSION,
        h_source="same_season_prior_years",
        validated=True,
        validation_status="conditional_pass_gate_b",
        scientific_caveat="Compares identical calendar months against prior years (2003 to target_year - 1). Pre-2002 excluded."
    )

    return APIResponseEnvelope(
        status="success",
        data=items,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset=str(parquet_path),
            row_count=len(items)
        )
    )


@router.get("/anomalies", response_model=APIResponseEnvelope[List[SeasonalComparisonRecord]])
def get_anomalies(
    district: Optional[str] = Query(None, description="Optional district name filter"),
    year: Optional[int] = Query(None, description="Target evaluation year"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Target calendar month"),
    flagged_only: bool = Query(True, description="Filter strictly to flagged anomalies (z >= 3.5)")
):
    """
    Returns flagged seasonal anomalies evaluated with robust median/MAD and sparse-count Negative Binomial fallbacks.
    """
    parquet_path = settings.ANALYSIS_DIR / "m6" / "m6_3_historical_comparison.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=503, detail="Historical comparison dataset not available.")

    where_clauses = []
    params = []

    if flagged_only:
        where_clauses.append("anomaly_flag = true")
    if district:
        where_clauses.append("district_name = ?")
        params.append(district)
    if year:
        where_clauses.append("target_year = ?")
        params.append(year)
    if month:
        where_clauses.append("month = ?")
        params.append(month)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"""
        SELECT district_name, month, target_year, raw_observed, evaluated_value,
               is_exposure_standardized, z_score, anomaly_flag, scoring_method,
               baseline_median, baseline_mad, baseline_n_years, target_h_source,
               validation_status
        FROM '{parquet_path}'
        {where_sql}
        ORDER BY z_score DESC NULLS LAST
    """
    records, duration = db_layer.execute_query(query, params)

    items = [
        SeasonalComparisonRecord(
            district_name=r["district_name"],
            month=int(r["month"]),
            target_year=int(r["target_year"]),
            raw_observed=float(r["raw_observed"]),
            evaluated_value=float(r["evaluated_value"]),
            is_exposure_standardized=bool(r["is_exposure_standardized"]),
            z_score=float(r["z_score"]) if r.get("z_score") is not None else None,
            anomaly_flag=bool(r["anomaly_flag"]),
            scoring_method=str(r["scoring_method"]),
            baseline_median=float(r["baseline_median"]) if r.get("baseline_median") is not None else None,
            baseline_mad=float(r["baseline_mad"]) if r.get("baseline_mad") is not None else None,
            baseline_n_years=int(r["baseline_n_years"]),
            target_h_source=str(r["target_h_source"]),
            validation_status=str(r["validation_status"])
        )
        for r in records
    ]

    provenance = ProvenanceModel(
        sensor="Robust Anomaly Engine (Section 4.6)",
        version=settings.VERSION,
        h_source="robust_z_and_nb_exceedance",
        validated=True,
        validation_status="conditional_pass_gate_b",
        scientific_caveat="Threshold z >= 3.5. Sparse counts handled via Negative Binomial upper tail exceedance."
    )

    return APIResponseEnvelope(
        status="success",
        data=items,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset=str(parquet_path),
            row_count=len(items)
        )
    )


@router.get("/attribution", response_model=APIResponseEnvelope[List[AttributionRecord]])
def get_attribution(
    district: Optional[str] = Query(None, description="Optional district name filter"),
    year: Optional[int] = Query(None, description="Target evaluation year"),
    month: Optional[int] = Query(None, ge=1, le=12, description="Target calendar month")
):
    """
    Returns multi-evidence attribution assessments: sensor-related change, increased fire activity, mixed, or insufficient evidence.
    """
    parquet_path = settings.ANALYSIS_DIR / "m6" / "m6_5_attribution_evidence_table.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=503, detail="Attribution evidence table not available.")

    where_clauses = []
    params = []

    if district:
        where_clauses.append("district_name = ?")
        params.append(district)
    if year:
        where_clauses.append("year = ?")
        params.append(year)
    if month:
        where_clauses.append("month = ?")
        params.append(month)

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"""
        SELECT district_name, month, year, observed_count, baseline_median,
               count_z_score, anomaly_flag, exposure_passes, attribution_outcome,
               confidence, explanation, validation_status
        FROM '{parquet_path}'
        {where_sql}
        ORDER BY year DESC, month ASC
    """
    records, duration = db_layer.execute_query(query, params)

    items = [
        AttributionRecord(
            district_name=r["district_name"],
            month=int(r["month"]),
            year=int(r["year"]),
            observed_count=float(r["observed_count"]),
            baseline_median=float(r["baseline_median"]) if r.get("baseline_median") is not None else None,
            count_z_score=float(r["count_z_score"]) if r.get("count_z_score") is not None else None,
            anomaly_flag=bool(r["anomaly_flag"]),
            exposure_passes=float(r["exposure_passes"]),
            attribution_outcome=str(r["attribution_outcome"]),
            confidence=str(r["confidence"]),
            explanation=str(r["explanation"]),
            burned_area_km2=None,  # Stored unacquired
            burned_area_status="missing_unacquired",
            validation_status=str(r["validation_status"])
        )
        for r in records
    ]

    provenance = ProvenanceModel(
        sensor="Attribution Decision Matrix (Section 4.5)",
        version=settings.VERSION,
        h_source="multi_evidence_synthesis",
        validated=True,
        validation_status="conditional_pass_gate_b",
        scientific_caveat="Attribution reflects an evidence assessment, never a single claimed cause. Burned-area raster evidence is unacquired."
    )

    return APIResponseEnvelope(
        status="success",
        data=items,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=duration,
            source_dataset=str(parquet_path),
            row_count=len(items)
        )
    )


@router.get("/coverage", response_model=APIResponseEnvelope[List[CoverageRecord]])
def get_coverage(
    year: Optional[int] = Query(None, description="Optional filter by year")
):
    """
    Returns observation opportunity metadata, satellite pass baselines, and cloud cover proxies.
    """
    timeline = [
        {
            "period": "2000-11 to 2002-06",
            "orbital_exposure_passes": 7.0,
            "active_satellites": ["Terra (MODIS)"],
            "cloud_eval_proxy": "MOD06 / Terra cloud mask",
            "scientific_note": "Morning-only overpass (~10:30 IST); systematically under-detects peak afternoon burning."
        },
        {
            "period": "2002-07 to 2011-12",
            "orbital_exposure_passes": 14.0,
            "active_satellites": ["Terra (MODIS)", "Aqua (MODIS)"],
            "cloud_eval_proxy": "MOD06 / MYD06 cloud mask",
            "scientific_note": "Dual overpass (~10:30 and ~13:30 IST); standard reference era."
        },
        {
            "period": "2012-01 to 2018-04",
            "orbital_exposure_passes": 14.0,
            "active_satellites": ["Terra (MODIS)", "Aqua (MODIS)", "Suomi-NPP (VIIRS)"],
            "cloud_eval_proxy": "MODIS cloud + VIIRS cloud mask",
            "scientific_note": "VIIRS single-satellite baseline (~2 passes/day * 7 = 14 passes/week)."
        },
        {
            "period": "2018-05 to 2023-02",
            "orbital_exposure_passes": 28.0,
            "active_satellites": ["Terra", "Aqua", "Suomi-NPP", "NOAA-20 (VIIRS)"],
            "cloud_eval_proxy": "Constellation dual afternoon view",
            "scientific_note": "NOAA-20 addition doubled afternoon view opportunities to 28 passes/week."
        },
        {
            "period": "2023-03 to 2026-08",
            "orbital_exposure_passes": 42.0,
            "active_satellites": ["Terra", "Aqua", "Suomi-NPP", "NOAA-20", "NOAA-21 (VIIRS)"],
            "cloud_eval_proxy": "Constellation triple afternoon view",
            "scientific_note": "NOAA-21 addition increased view opportunities to 42 passes/week (3x over single satellite)."
        }
    ]

    records = [CoverageRecord(**item) for item in timeline]

    provenance = ProvenanceModel(
        sensor="Orbital Constellation Metadata",
        version=settings.VERSION,
        h_source="m3_6_orbital_exposure_timeline",
        validated=True,
        validation_status="fully_validated",
        scientific_caveat="Passes/week represent theoretical nominal observation opportunities, not cloud-clear detections."
    )

    return APIResponseEnvelope(
        status="success",
        data=records,
        provenance=provenance,
        query_metadata=QueryMetadata(
            execution_time_ms=0.4,
            source_dataset="m3_6_orbital_exposure_timeline",
            row_count=len(records)
        )
    )
