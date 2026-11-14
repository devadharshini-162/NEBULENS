/**
 * NEBULENS — Core TypeScript Type Definitions
 * Exact alignment with M7 FastAPI OpenAPI schemas and Gate B scientific provenance structures.
 */

// Provenance Metadata
export interface ProvenanceModel {
  sensor?: string;
  version: string;
  h_source?: string;
  validated: boolean;
  serving_eligible: boolean;
  validation_status: string;
  scientific_caveat?: string;
}

// Query Metadata
export interface QueryMetadata {
  execution_time_ms?: number;
  row_count?: number;
  page?: number;
  page_size?: number;
  total_pages?: number;
  source_dataset?: string;
}

// Standard API Response Envelope
export interface ApiResponse<T> {
  status: 'success' | 'error';
  data: T;
  provenance?: ProvenanceModel;
  query_metadata?: QueryMetadata;
  error?: string;
  detail?: any;
}

// Health & Readiness
export interface HealthResponse {
  status: string;
  timestamp: string;
  service: string;
  version: string;
}

export interface ReadinessResponse {
  status: string;
  database: string;
  datasets_ready: Record<string, boolean>;
  all_ready: boolean;
}

// Metadata: Availability
export interface SensorDetail {
  sensor_name: string;
  products: string[];
  nominal_resolution_m: number;
  coverage_years: string;
  status: string;
}

export interface ValidationRegime {
  status: string;
  validated: boolean;
  serving_eligible?: boolean;
  note: string;
}

export interface TemporalRange {
  start_date: string;
  end_date: string;
  modis_sp_start: string;
  viirs_snpp_sp_start: string;
  viirs_noaa20_sp_start: string;
  viirs_noaa21_sp_start: string;
  standard_quality_cutoff: string;
}

export interface AvailabilityData {
  study_region: string;
  spatial_coverage_km2: number;
  crs: string;
  temporal_range: TemporalRange;
  sensors: SensorDetail[];
  validation_matrix: Record<string, ValidationRegime>;
  limitations: string[];
}

// Metadata: District
export interface DistrictInfo {
  district_id?: string;
  district_name: string;
  state: string;
  cell_count: number;
  area_sqkm: number;
  centroid_lon: number;
  centroid_lat: number;
}

// Grid Cell Info
export interface GridCellInfo {
  grid_cell_id: string;
  res_km: number;
  centroid_lon: number;
  centroid_lat: number;
  district_name?: string;
}

// Daily Gridded Map Data
export interface DailyGridPoint {
  grid_cell_id: string;
  analysis_date_ist: string;
  active_fire_day: number;
  count_value: number;
  frp_value: number;
  h_source: string;
}

// Time Series Data Point
export interface SeriesDataPoint {
  date: string;
  count: number;
  frp: number;
  count_lower_80?: number;
  count_upper_80?: number;
  count_lower_95?: number;
  count_upper_95?: number;
  exposure_passes?: number;
  h_source: string;
  is_counterfactual_prediction?: boolean;
}

// Seasonal Comparison Summary
export interface SeasonalSummary {
  district_name: string;
  month: number;
  target_year: number;
  raw_observed: number;
  evaluated_value: number;
  is_exposure_standardized: boolean;
  z_score: number;
  anomaly_flag: boolean;
  scoring_method: string;
  baseline_median: number;
  baseline_mad: number;
  baseline_n_years: number;
  target_h_source: string;
  validation_status: string;
}

// Anomaly Record
export interface AnomalyRecord {
  district_name: string;
  month: number;
  target_year: number;
  raw_observed: number;
  evaluated_value: number;
  is_exposure_standardized: boolean;
  z_score: number;
  anomaly_flag: boolean;
  scoring_method: string;
  baseline_median: number;
  baseline_mad: number;
  baseline_n_years: number;
  target_h_source: string;
  validation_status: string;
}

// Multi-Evidence Attribution Record
export interface AttributionRecord {
  district_name: string;
  month: number;
  year: number;
  observed_count: number;
  baseline_median: number;
  count_z_score: number;
  anomaly_flag: boolean;
  exposure_passes: number;
  attribution_outcome:
    | 'likely_sensor_related_change'
    | 'consistent_with_increased_fire_activity'
    | 'mixed_effects'
    | 'insufficient_evidence';
  confidence: 'high' | 'medium' | 'low';
  explanation: string;
  burned_area_km2: number | null;
  burned_area_status: string;
  validation_status: string;
}

// Hotspot Record
export interface HotspotRecord {
  observation_id: string;
  source: string;
  satellite: string;
  instrument: string;
  acq_date: string;
  acq_time: number;
  acq_datetime_utc: string;
  analysis_date_ist: string;
  latitude: number;
  longitude: number;
  frp: number;
  confidence: string;
  confidence_class: string;
  daynight: string;
  district_name: string | null;
  grid_cell_id_10km: string;
  duplicate_group: string;
  landcover_class: string | null;
}

// Paginated Hotspots Container
export interface HotspotsPaginated {
  total_count: number;
  limit: number;
  offset: number;
  hotspots: HotspotRecord[];
}

// Coverage Record
export interface CoverageRecord {
  period: string;
  orbital_exposure_passes: number;
  active_satellites: string[];
  cloud_eval_proxy: string;
  scientific_note: string;
}

// Shared Application State Types
export type SupportedSensor = 'harmonised' | 'viirs' | 'modis' | 'all';
export type SupportedProduct = 'standard';
export type ConfidenceFilter = 'all' | 'nominal_high' | 'high_only';
export type TemporalAggregation = 'day' | 'week' | 'month';
export type DataInterpretation = 'observed' | 'harmonised' | 'counterfactual';
export type ActiveView =
  | 'overview'
  | 'map'
  | 'heatmap'
  | 'series'
  | 'compare'
  | 'hotspots'
  | 'anomalies'
  | 'attribution'
  | 'visualisation'
  | 'transparency';

export interface DateRange {
  startDate: string;
  endDate: string;
}

export interface RegionSelection {
  type: 'state' | 'district' | 'cell' | 'polygon';
  id: string;
  name: string;
  polygonCoords?: [number, number][]; // [lat, lng] array
}

export interface AppState {
  sensor: SupportedSensor;
  product: SupportedProduct;
  confidence: ConfidenceFilter;
  dateRange: DateRange;
  temporalAggregation: TemporalAggregation;
  region: RegionSelection;
  interpretation: DataInterpretation;
  view: ActiveView;
  selectedHotspotId?: string | null;
  selectedAnomaly?: AnomalyRecord | null;
  activeAnalysisDate: string; // Current single day for map/granule inspection
  customPolygon?: [number, number][] | null;
}
