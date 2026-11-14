"""
backend/models/schemas.py
=========================
Pydantic schemas and standard response envelopes for NEBULENS API.
"""

from typing import Dict, List, Optional, Any, Generic, TypeVar, Union
from pydantic import BaseModel, Field

T = TypeVar("T")


class ProvenanceModel(BaseModel):
    sensor: Optional[str] = Field(None, description="Satellite sensor (e.g. MODIS, VIIRS, Harmonised)")
    version: str = Field("1.0.0", description="Data / pipeline version")
    h_source: Optional[str] = Field(None, description="Underlying stream: calibrated_from_modis, observed_viirs, or observed_modis")
    validated: bool = Field(False, description="Strict scientific validation flag (true ONLY for fully_validated populations)")
    serving_eligible: bool = Field(True, description="Whether data is authorized for API serving under specified constraints")
    validation_status: str = Field("conditional_pass_gate_b", description="Explicit Gate B acceptance status")
    scientific_caveat: Optional[str] = Field(None, description="Mandatory scientific caveats or release restrictions")


class QueryMetadata(BaseModel):
    execution_time_ms: float = Field(..., description="Query execution duration in milliseconds")
    source_dataset: str = Field(..., description="Authoritative Parquet dataset scanned")
    row_count: int = Field(..., description="Total rows in query response")


class APIResponseEnvelope(BaseModel, Generic[T]):
    status: str = Field("success", description="Status code: success or error")
    data: T
    provenance: Optional[ProvenanceModel] = None
    query_metadata: Optional[QueryMetadata] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: str


class ReadinessCheckItem(BaseModel):
    name: str
    available: bool
    path: str
    details: Optional[str] = None


class ReadinessResponse(BaseModel):
    ready: bool
    duckdb_connected: bool
    checks: List[ReadinessCheckItem]


class AvailabilityResponse(BaseModel):
    study_region: str
    spatial_coverage_km2: float
    crs: str
    temporal_range: Dict[str, str]
    sensors: List[Dict[str, Any]]
    validation_matrix: Dict[str, Any]
    limitations: List[str]


class DistrictInfo(BaseModel):
    district_name: str
    district_code: Optional[str] = None
    cell_count_10km: int


class GridCellInfo(BaseModel):
    grid_cell_id: str
    resolution_km: int
    origin_x: float
    origin_y: float
    district_name: Optional[str] = None
    landcover_class: Optional[str] = None


class GridDailyRecord(BaseModel):
    grid_cell_id: str
    analysis_date_ist: str
    active_fire_day: int
    count_value: float
    frp_value: float
    h_source: Optional[str] = None


class SeriesDataPoint(BaseModel):
    date: str
    count: float
    frp: float
    count_lower_80: Optional[float] = None
    count_upper_80: Optional[float] = None
    count_lower_95: Optional[float] = None
    count_upper_95: Optional[float] = None
    exposure_passes: Optional[float] = None
    h_source: Optional[str] = None
    is_counterfactual_prediction: bool = False


class HotspotDetail(BaseModel):
    observation_id: str
    source: str
    satellite: str
    instrument: str
    acq_date: str
    acq_time: int
    acq_datetime_utc: str
    analysis_date_ist: str
    latitude: float
    longitude: float
    frp: float
    confidence: Optional[Union[float, int, str]] = None
    confidence_class: Optional[str] = None
    daynight: Optional[str] = None
    district_name: Optional[str] = None
    grid_cell_id_10km: Optional[str] = None
    duplicate_group: Optional[str] = None
    landcover_class: Optional[str] = None


class HotspotListResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    hotspots: List[HotspotDetail]


class SeasonalComparisonRecord(BaseModel):
    district_name: str
    month: int
    target_year: int
    raw_observed: float
    evaluated_value: float
    is_exposure_standardized: bool
    z_score: Optional[float] = None
    anomaly_flag: bool
    scoring_method: str
    baseline_median: Optional[float] = None
    baseline_mad: Optional[float] = None
    baseline_n_years: int
    target_h_source: str
    validation_status: str


class AttributionRecord(BaseModel):
    district_name: str
    month: int
    year: int
    observed_count: float
    baseline_median: Optional[float] = None
    count_z_score: Optional[float] = None
    anomaly_flag: bool
    exposure_passes: float
    attribution_outcome: str
    confidence: str
    explanation: str
    burned_area_km2: Optional[float] = None
    burned_area_status: str = "missing_unacquired"
    validation_status: str


class CoverageRecord(BaseModel):
    period: str
    orbital_exposure_passes: float
    active_satellites: List[str]
    cloud_eval_proxy: Optional[str] = None
    scientific_note: str
