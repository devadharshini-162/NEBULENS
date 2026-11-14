import React, { createContext, useContext, useState, useMemo, ReactNode, useEffect } from 'react';
import type {
  AppState,
  SupportedSensor,
  SupportedProduct,
  ConfidenceFilter,
  TemporalAggregation,
  DataInterpretation,
  DateRange,
  RegionSelection,
  ActiveView,
  AnomalyRecord,
} from '../types';

interface AppContextValue {
  state: AppState;
  setSensor: (sensor: SupportedSensor) => void;
  setProduct: (product: SupportedProduct) => void;
  setConfidence: (confidence: ConfidenceFilter) => void;
  setDateRange: (range: DateRange) => void;
  setTemporalAggregation: (agg: TemporalAggregation) => void;
  setRegion: (region: RegionSelection) => void;
  setCustomPolygon: (polygon: [number, number][] | null) => void;
  clearCustomPolygon: () => void;
  setInterpretation: (interp: DataInterpretation) => void;
  setView: (view: ActiveView) => void;
  setActiveAnalysisDate: (date: string) => void;
  setSelectedHotspotId: (id: string | null) => void;
  setSelectedAnomaly: (anomaly: AnomalyRecord | null) => void;
  resetFilters: () => void;
}

const defaultState: AppState = {
  sensor: 'harmonised',
  product: 'standard',
  confidence: 'all',
  dateRange: {
    startDate: '2023-01-01',
    endDate: '2023-12-31',
  },
  temporalAggregation: 'month',
  region: {
    type: 'state',
    id: 'TN',
    name: 'Tamil Nadu (Statewide)',
  },
  interpretation: 'harmonised',
  view: 'map',
  activeAnalysisDate: '2023-03-04',
  selectedHotspotId: null,
  selectedAnomaly: null,
  customPolygon: null,
};

// URL query sync helpers
function getStateFromUrl(): Partial<AppState> {
  if (typeof window === 'undefined') return {};
  const params = new URLSearchParams(window.location.search);
  const patch: Partial<AppState> = {};

  const view = params.get('view') as ActiveView | null;
  if (view) patch.view = view;

  const sensor = params.get('sensor') as SupportedSensor | null;
  if (sensor) patch.sensor = sensor;

  const startDate = params.get('start');
  const endDate = params.get('end');
  if (startDate && endDate) {
    patch.dateRange = { startDate, endDate };
  }

  const regionType = params.get('regionType') as 'state' | 'district' | 'cell' | null;
  const regionName = params.get('regionName');
  const regionId = params.get('regionId');
  if (regionType && regionName) {
    patch.region = { type: regionType, id: regionId || regionName, name: regionName };
  }

  const date = params.get('date');
  if (date) patch.activeAnalysisDate = date;

  return patch;
}

function syncStateToUrl(state: AppState) {
  if (typeof window === 'undefined' || !window.history?.replaceState) return;
  const params = new URLSearchParams();
  params.set('view', state.view);
  params.set('sensor', state.sensor);
  params.set('start', state.dateRange.startDate);
  params.set('end', state.dateRange.endDate);
  params.set('regionType', state.region.type);
  params.set('regionName', state.region.name);
  params.set('regionId', state.region.id);
  params.set('date', state.activeAnalysisDate);

  const newUrl = `${window.location.pathname}?${params.toString()}`;
  window.history.replaceState(null, '', newUrl);
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode; initialState?: Partial<AppState> }> = ({
  children,
  initialState,
}) => {
  const [state, setState] = useState<AppState>(() => ({
    ...defaultState,
    ...getStateFromUrl(),
    ...initialState,
  }));

  // Sync to URL when key state dimensions change
  useEffect(() => {
    syncStateToUrl(state);
  }, [state]);

  const setSensor = (sensor: SupportedSensor) => {
    setState((prev) => ({
      ...prev,
      sensor,
      // If switching to modis pre-2012, interpretation adjusts accordingly
      interpretation: sensor === 'harmonised' ? 'harmonised' : 'observed',
    }));
  };

  const setProduct = (product: SupportedProduct) => setState((prev) => ({ ...prev, product }));
  const setConfidence = (confidence: ConfidenceFilter) => setState((prev) => ({ ...prev, confidence }));
  const setDateRange = (dateRange: DateRange) => setState((prev) => ({ ...prev, dateRange }));
  const setTemporalAggregation = (temporalAggregation: TemporalAggregation) =>
    setState((prev) => ({ ...prev, temporalAggregation }));
  const setRegion = (region: RegionSelection) => {
    setState((prev) => ({
      ...prev,
      region,
      // If setting a district or statewide, clear any custom polygon unless it's explicitly a polygon region
      customPolygon: region.type === 'polygon' ? prev.customPolygon : null,
    }));
  };

  const setCustomPolygon = (polygon: [number, number][] | null) => {
    setState((prev) => {
      if (!polygon || polygon.length < 3) {
        return {
          ...prev,
          customPolygon: null,
          region: {
            type: 'state',
            id: 'TN',
            name: 'Tamil Nadu (Statewide)',
          },
        };
      }
      return {
        ...prev,
        customPolygon: polygon,
        region: {
          type: 'polygon',
          id: 'custom_polygon',
          name: 'Custom Selected Area',
          polygonCoords: polygon,
        },
      };
    });
  };

  const clearCustomPolygon = () => {
    setState((prev) => ({
      ...prev,
      customPolygon: null,
      region: {
        type: 'state',
        id: 'TN',
        name: 'Tamil Nadu (Statewide)',
      },
    }));
  };

  const setInterpretation = (interpretation: DataInterpretation) =>
    setState((prev) => ({ ...prev, interpretation }));
  const setView = (view: ActiveView) => setState((prev) => ({ ...prev, view }));
  const setActiveAnalysisDate = (activeAnalysisDate: string) =>
    setState((prev) => ({ ...prev, activeAnalysisDate }));
  const setSelectedHotspotId = (selectedHotspotId: string | null) =>
    setState((prev) => ({ ...prev, selectedHotspotId }));
  const setSelectedAnomaly = (selectedAnomaly: AnomalyRecord | null) =>
    setState((prev) => ({ ...prev, selectedAnomaly }));

  const resetFilters = () => setState(defaultState);

  const value = useMemo(
    () => ({
      state,
      setSensor,
      setProduct,
      setConfidence,
      setDateRange,
      setTemporalAggregation,
      setRegion,
      setCustomPolygon,
      clearCustomPolygon,
      setInterpretation,
      setView,
      setActiveAnalysisDate,
      setSelectedHotspotId,
      setSelectedAnomaly,
      resetFilters,
    }),
    [state]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

export const useApp = (): AppContextValue => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};
