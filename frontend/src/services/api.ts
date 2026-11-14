import type {
  ApiResponse,
  HealthResponse,
  ReadinessResponse,
  AvailabilityData,
  DistrictInfo,
  GridCellInfo,
  DailyGridPoint,
  SeriesDataPoint,
  SeasonalSummary,
  AnomalyRecord,
  AttributionRecord,
  HotspotRecord,
  HotspotsPaginated,
  CoverageRecord,
} from '../types';

const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE_URL) ||
  '/api/v1';

export class ApiError extends Error {
  status?: number;
  data?: unknown;

  constructor(message: string, status?: number, data?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<ApiResponse<T>> {
  const response = await fetch(url, {
    headers: {
      Accept: 'application/json',
      ...init?.headers,
    },
    ...init,
  });

  const contentType = response.headers.get('content-type');
  let data: any = null;
  if (contentType && contentType.includes('application/json')) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    const errorMsg =
      (typeof data === 'object' && data !== null && (data.error || data.detail)) ||
      `HTTP error ${response.status}: ${response.statusText}`;
    throw new ApiError(errorMsg, response.status, data);
  }

  return data as ApiResponse<T>;
}

export const apiService = {
  getBaseUrl(): string {
    return API_BASE_URL;
  },

  // Health and Readiness
  async getHealth(): Promise<HealthResponse> {
    const rootUrl = API_BASE_URL.replace(/\/api\/v1\/?$/, '');
    const res = await fetch(`${rootUrl}/health`);
    if (!res.ok) throw new ApiError(`Health check failed: ${res.statusText}`, res.status);
    return res.json();
  },

  async getReadiness(): Promise<ReadinessResponse> {
    const rootUrl = API_BASE_URL.replace(/\/api\/v1\/?$/, '');
    const res = await fetch(`${rootUrl}/ready`);
    if (!res.ok) throw new ApiError(`Readiness check failed: ${res.statusText}`, res.status);
    return res.json();
  },

  // Metadata Endpoints
  async getAvailability(): Promise<ApiResponse<AvailabilityData>> {
    return fetchJson<AvailabilityData>(`${API_BASE_URL}/meta/availability`);
  },

  async getDistricts(): Promise<ApiResponse<DistrictInfo[]>> {
    return fetchJson<DistrictInfo[]>(`${API_BASE_URL}/meta/districts`);
  },

  // Grid Endpoints
  async getGridCells(params?: { district_name?: string }): Promise<ApiResponse<GridCellInfo[]>> {
    const query = new URLSearchParams();
    if (params?.district_name) query.set('district_name', params.district_name);
    const qs = query.toString();
    return fetchJson<GridCellInfo[]>(`${API_BASE_URL}/grid/cells${qs ? `?${qs}` : ''}`);
  },

  async getGridDaily(params: {
    date: string;
    sensor?: string;
  }): Promise<ApiResponse<DailyGridPoint[]>> {
    const query = new URLSearchParams({ date: params.date });
    if (params.sensor && params.sensor !== 'all') query.set('sensor', params.sensor);
    return fetchJson<DailyGridPoint[]>(`${API_BASE_URL}/grid/daily?${query.toString()}`);
  },

  // Series Endpoint
  async getSeries(params: {
    start_date: string;
    end_date: string;
    sensor?: string;
    district_name?: string;
    grid_cell_id?: string;
  }): Promise<ApiResponse<SeriesDataPoint[]>> {
    const query = new URLSearchParams({
      start_date: params.start_date,
      end_date: params.end_date,
    });
    if (params.sensor && params.sensor !== 'all') query.set('sensor', params.sensor);
    if (params.district_name && params.district_name !== 'Tamil Nadu (Statewide)') {
      query.set('district_name', params.district_name);
    }
    if (params.grid_cell_id) query.set('grid_cell_id', params.grid_cell_id);
    return fetchJson<SeriesDataPoint[]>(`${API_BASE_URL}/series?${query.toString()}`);
  },

  // Seasonal Comparison
  async getSeasonalComparison(params?: {
    district_name?: string;
  }): Promise<ApiResponse<SeasonalSummary[]>> {
    const query = new URLSearchParams();
    if (params?.district_name && params.district_name !== 'Tamil Nadu (Statewide)') {
      query.set('district_name', params.district_name);
    }
    const qs = query.toString();
    return fetchJson<SeasonalSummary[]>(`${API_BASE_URL}/compare/seasonal${qs ? `?${qs}` : ''}`);
  },

  // Anomalies
  async getAnomalies(params?: {
    year?: number;
    district_name?: string;
    min_score?: number;
  }): Promise<ApiResponse<AnomalyRecord[]>> {
    const query = new URLSearchParams();
    if (params?.year) query.set('year', params.year.toString());
    if (params?.district_name && params.district_name !== 'Tamil Nadu (Statewide)') {
      query.set('district_name', params.district_name);
    }
    if (params?.min_score !== undefined) query.set('min_score', params.min_score.toString());
    const qs = query.toString();
    return fetchJson<AnomalyRecord[]>(`${API_BASE_URL}/anomalies${qs ? `?${qs}` : ''}`);
  },

  // Attribution
  async getAttribution(params?: {
    district_name?: string;
    outcome?: string;
  }): Promise<ApiResponse<AttributionRecord[]>> {
    const query = new URLSearchParams();
    if (params?.district_name && params.district_name !== 'Tamil Nadu (Statewide)') {
      query.set('district_name', params.district_name);
    }
    if (params?.outcome) query.set('outcome', params.outcome);
    const qs = query.toString();
    return fetchJson<AttributionRecord[]>(`${API_BASE_URL}/attribution${qs ? `?${qs}` : ''}`);
  },

  // Hotspots
  async getHotspots(params: {
    start_date: string;
    end_date: string;
    grid_cell_id?: string;
    district_name?: string;
    confidence?: string;
    limit?: number;
    offset?: number;
  }): Promise<ApiResponse<HotspotsPaginated>> {
    const query = new URLSearchParams({
      start_date: params.start_date,
      end_date: params.end_date,
    });
    if (params.grid_cell_id) query.set('grid_cell_id', params.grid_cell_id);
    if (params.district_name && params.district_name !== 'Tamil Nadu (Statewide)') {
      query.set('district_name', params.district_name);
    }
    if (params.confidence && params.confidence !== 'all') query.set('confidence', params.confidence);
    if (params.limit) query.set('limit', params.limit.toString());
    if (params.offset !== undefined) query.set('offset', params.offset.toString());
    return fetchJson<HotspotsPaginated>(`${API_BASE_URL}/hotspots?${query.toString()}`);
  },

  async getHotspotDetail(observationId: string): Promise<ApiResponse<HotspotRecord>> {
    return fetchJson<HotspotRecord>(`${API_BASE_URL}/hotspots/${observationId}`);
  },

  // Coverage
  async getCoverage(): Promise<ApiResponse<CoverageRecord[]>> {
    return fetchJson<CoverageRecord[]>(`${API_BASE_URL}/coverage`);
  },

  // Export URL helper
  getExportUrl(params: {
    query_type: 'series' | 'anomalies' | 'attribution' | 'season_metrics';
    start_date?: string;
    end_date?: string;
    district_name?: string;
  }): string {
    const query = new URLSearchParams({ query_type: params.query_type });
    if (params.start_date) query.set('start_date', params.start_date);
    if (params.end_date) query.set('end_date', params.end_date);
    if (params.district_name && params.district_name !== 'Tamil Nadu (Statewide)') {
      query.set('district_name', params.district_name);
    }
    return `${API_BASE_URL}/export?${query.toString()}`;
  },
};
