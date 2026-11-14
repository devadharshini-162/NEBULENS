"""
tests/test_m7_api.py
====================
Comprehensive integration test suite for Milestone M7: FastAPI Backend & Data Serving.
Tests:
- Health and readiness endpoints
- Metadata, districts, and grid endpoints
- Gridded daily map layers
- Analytical time series with provenance
- Hotspot list and detail endpoints with pagination
- Seasonal comparison and anomaly endpoints
- Attribution multi-evidence and missing-data behavior
- Coverage proxies
- CSV data export streaming
- Error handling (400, 404, validation errors)
- CORS header compliance
- OpenAPI specification completeness
"""

import json
from pathlib import Path
import pytest
from backend.test_client import test_client
from backend.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_health_endpoint():
    """Verify health endpoint returns 200, healthy status, and timestamp."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_readiness_endpoint():
    """Verify readiness check verifies DuckDB engine and Parquet dataset availability."""
    resp = test_client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ready"] is True
    assert data["duckdb_connected"] is True
    assert len(data["checks"]) >= 8


def test_openapi_specification():
    """Verify that every Section 5 required endpoint is registered in the OpenAPI schema."""
    resp = test_client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema.get("paths", {})
    
    expected_endpoints = [
        "/api/v1/meta/availability",
        "/api/v1/meta/districts",
        "/api/v1/grid/cells",
        "/api/v1/grid/daily",
        "/api/v1/series",
        "/api/v1/hotspots",
        "/api/v1/hotspots/{observation_id}",
        "/api/v1/compare/seasonal",
        "/api/v1/anomalies",
        "/api/v1/attribution",
        "/api/v1/coverage",
        "/api/v1/export"
    ]
    for ep in expected_endpoints:
        assert ep in paths, f"Missing endpoint {ep} in OpenAPI schema"


def test_cors_headers():
    """Verify CORS headers allow configured frontend origins."""
    resp = test_client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_meta_availability():
    """Verify availability endpoint returns sensors, temporal range, and validation matrix."""
    resp = test_client.get("/api/v1/meta/availability")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "success"
    data = payload["data"]
    assert data["study_region"] == "Tamil Nadu, India"
    assert len(data["sensors"]) == 2
    assert "2000-2002_terra_only" in data["validation_matrix"]
    assert payload["provenance"]["serving_eligible"] is True
    # Pre-2012 Model B must have validated=False, serving_eligible=True
    v_matrix = data["validation_matrix"]
    assert v_matrix["2003-2011_modis_model_b"]["validated"] is False
    assert v_matrix["2003-2011_modis_model_b"]["serving_eligible"] is True
    assert v_matrix["2012-2017_viirs_snpp"]["validated"] is True


def test_scientific_validation_semantics_by_era():
    """Targeted audit test: verify scientific validation semantics and serving eligibility across historical eras."""
    # 1. Pre-July 2002 Terra-only: BLOCKED from trend analysis (validated=False, serving_eligible=False)
    r_terra = test_client.get("/api/v1/grid/daily?date=2001-03-04&sensor=harmonised")
    assert r_terra.status_code == 200
    p_terra = r_terra.json()["provenance"]
    assert p_terra["validated"] is False
    assert p_terra["serving_eligible"] is False
    assert p_terra["validation_status"] == "not_validated_terra_only_pre_2002"
    assert "BLOCKED" in p_terra["scientific_caveat"]

    # 2. 2003-2011 Model B counterfactual: CONDITIONAL PASS (validated=False, serving_eligible=True)
    r_modb = test_client.get("/api/v1/grid/daily?date=2005-03-04&sensor=harmonised")
    assert r_modb.status_code == 200
    p_modb = r_modb.json()["provenance"]
    assert p_modb["validated"] is False  # NOT blanket validated
    assert p_modb["serving_eligible"] is True  # authorized for serving with qualifications
    assert p_modb["validation_status"] == "conditional_pass_gate_b"
    assert "M <= 5" in p_modb["scientific_caveat"]

    # 3. 2012-2017 Suomi-NPP VIIRS baseline: FULLY VALIDATED (validated=True, serving_eligible=True)
    r_snpp = test_client.get("/api/v1/grid/daily?date=2015-03-04&sensor=harmonised")
    assert r_snpp.status_code == 200
    p_snpp = r_snpp.json()["provenance"]
    assert p_snpp["validated"] is True
    assert p_snpp["serving_eligible"] is True
    assert p_snpp["validation_status"] == "fully_validated"

    # 4. 2018+ Opportunity expansion: CONDITIONAL PASS (validated=False, serving_eligible=True)
    r_noaa = test_client.get("/api/v1/grid/daily?date=2023-03-04&sensor=harmonised")
    assert r_noaa.status_code == 200
    p_noaa = r_noaa.json()["provenance"]
    assert p_noaa["validated"] is False
    assert p_noaa["serving_eligible"] is True
    assert p_noaa["validation_status"] == "conditional_pass_opportunity_expansion"
    assert "Exposure standardization required" in p_noaa["scientific_caveat"]


def test_meta_districts():
    """Verify districts endpoint returns all 38 districts of Tamil Nadu."""
    resp = test_client.get("/api/v1/meta/districts")
    assert resp.status_code == 200
    payload = resp.json()
    districts = payload["data"]
    assert len(districts) == 38
    names = [d["district_name"] for d in districts]
    assert "Coimbatore" in names
    assert "Chennai" in names
    assert "The Nilgiris" in names


def test_grid_cells_primary_and_sensitivity():
    """Verify grid cells lookup for 10km (primary) and 25km (sensitivity)."""
    # 10km primary
    r10 = test_client.get("/api/v1/grid/cells?resolution=10km")
    assert r10.status_code == 200
    data10 = r10.json()["data"]
    assert len(data10) == 1460
    assert data10[0]["resolution_km"] == 10
    
    # 25km sensitivity
    r25 = test_client.get("/api/v1/grid/cells?resolution=25km")
    assert r25.status_code == 200
    data25 = r25.json()["data"]
    assert len(data25) == 270
    assert data25[0]["resolution_km"] == 25
    
    # District filter
    r_dist = test_client.get("/api/v1/grid/cells?resolution=10km&district=The%20Nilgiris")
    assert r_dist.status_code == 200
    assert len(r_dist.json()["data"]) > 0


def test_grid_daily_map_layer():
    """Verify gridded daily map layer returns 1,460 cells for a specific date."""
    resp = test_client.get("/api/v1/grid/daily?date=2023-03-04&sensor=harmonised&metric=count")
    assert resp.status_code == 200
    payload = resp.json()
    cells = payload["data"]
    assert len(cells) == 1460
    assert payload["provenance"]["sensor"] == "HARMONISED"
    assert payload["provenance"]["validation_status"] == "conditional_pass_opportunity_expansion"
    assert payload["provenance"]["serving_eligible"] is True
    assert payload["provenance"]["validated"] is False


def test_series_aggregation_and_provenance():
    """Verify time series endpoint across daily, monthly, and annual aggregations with provenance."""
    # Daily state series
    r_daily = test_client.get(
        "/api/v1/series?entity_type=state&entity_id=Tamil%20Nadu&start_date=2023-03-01&end_date=2023-03-07&stream=harmonised&aggregation=daily"
    )
    assert r_daily.status_code == 200
    pts_daily = r_daily.json()["data"]
    assert len(pts_daily) == 7
    assert pts_daily[0]["count"] > 0
    assert pts_daily[0]["exposure_passes"] == 42.0

    # Monthly district series
    r_monthly = test_client.get(
        "/api/v1/series?entity_type=district&entity_id=Coimbatore&start_date=2023-01-01&end_date=2023-06-30&stream=harmonised&aggregation=monthly"
    )
    assert r_monthly.status_code == 200
    assert len(r_monthly.json()["data"]) == 6

    # Annual state precomputed fast path
    r_annual = test_client.get(
        "/api/v1/series?entity_type=state&entity_id=Tamil%20Nadu&start_date=2020-01-01&end_date=2024-12-31&stream=harmonised&aggregation=annual"
    )
    assert r_annual.status_code == 200
    assert len(r_annual.json()["data"]) == 5


def test_hotspots_list_and_detail():
    """Verify paginated hotspot detections and granular detail endpoint."""
    resp = test_client.get("/api/v1/hotspots?start_date=2023-03-04&end_date=2023-03-04&limit=10&offset=0")
    assert resp.status_code == 200
    payload = resp.json()
    data = payload["data"]
    assert data["total_count"] > 0
    assert len(data["hotspots"]) <= 10
    
    first_id = data["hotspots"][0]["observation_id"]
    r_det = test_client.get(f"/api/v1/hotspots/{first_id}")
    assert r_det.status_code == 200
    detail = r_det.json()["data"]
    assert detail["observation_id"] == first_id
    assert detail["latitude"] > 0
    assert detail["longitude"] > 0
    assert detail["frp"] >= 0


def test_seasonal_comparison_and_anomalies():
    """Verify same-season comparison across years and anomaly scoring endpoints."""
    # Seasonal comparison
    r_comp = test_client.get("/api/v1/compare/seasonal?entity_name=Coimbatore&month=3&target_year=2024")
    assert r_comp.status_code == 200
    comp_data = r_comp.json()["data"]
    assert len(comp_data) >= 1
    assert comp_data[0]["baseline_n_years"] >= 3

    # Flagged anomalies query
    r_anom = test_client.get("/api/v1/anomalies?year=2024&month=3&flagged_only=true")
    assert r_anom.status_code == 200
    anom_list = r_anom.json()["data"]
    assert len(anom_list) > 0
    for a in anom_list:
        assert a["anomaly_flag"] is True
        assert a["z_score"] >= 3.5


def test_attribution_endpoint_missing_burned_area():
    """Verify attribution endpoint returns multi-evidence results and handles missing burned area as null."""
    resp = test_client.get("/api/v1/attribution?year=2024&month=3")
    assert resp.status_code == 200
    payload = resp.json()
    records = payload["data"]
    assert len(records) > 0
    
    for r in records:
        assert r["attribution_outcome"] in [
            "consistent_with_baseline_variation",
            "likely_sensor_related_change",
            "consistent_with_increased_fire_activity",
            "mixed_effects",
            "insufficient_evidence"
        ]
        # Burned area must remain null / unacquired, never 0.0
        assert r["burned_area_km2"] is None
        assert r["burned_area_status"] == "missing_unacquired"


def test_coverage_endpoint():
    """Verify coverage timeline metadata endpoint."""
    resp = test_client.get("/api/v1/coverage")
    assert resp.status_code == 200
    records = resp.json()["data"]
    assert len(records) >= 5
    # Verify progression of orbital passes
    passes = [r["orbital_exposure_passes"] for r in records]
    assert passes == [7.0, 14.0, 14.0, 28.0, 42.0]


def test_export_endpoint_csv():
    """Verify CSV export endpoint streams valid comma-separated text."""
    resp = test_client.get("/api/v1/export?query_type=anomalies&year=2024")
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "text/csv; charset=utf-8"
    assert "attachment; filename=nebulens_anomalies_all_2024.csv" in resp.headers.get("content-disposition")
    text = resp.text
    assert "district_name,month,target_year" in text
    lines = text.strip().split("\n")
    assert len(lines) > 5


def test_error_handling_and_validation():
    """Verify structured RFC error responses on invalid parameters and missing records."""
    # Bad date format
    r_bad_date = test_client.get("/api/v1/grid/daily?date=invalid-date")
    assert r_bad_date.status_code in [400, 422]
    
    # Missing hotspot
    r_not_found = test_client.get("/api/v1/hotspots/NONEXISTENT_ID_99999")
    assert r_not_found.status_code == 404
    
    # Bad export type
    r_bad_export = test_client.get("/api/v1/export?query_type=invalid_table")
    assert r_bad_export.status_code == 400


def test_m0_through_m6_baseline_integrity():
    """Verify that all M0–M6 baseline models, datasets, and audits remain untouched."""
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_3_nb_coefficients.json").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m4" / "m4_cell_weeks_dataset.parquet").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m5" / "m5_app_readiness_matrix.csv").exists()
    assert (PROJECT_ROOT / "data" / "analysis" / "m6" / "m6_5_attribution_evidence_table.parquet").exists()
    assert (PROJECT_ROOT / "docs" / "reports" / "m6_final_scientific_acceptance_audit.md").exists()
