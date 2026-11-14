import React from 'react';
import { useApp } from '../context/AppContext';
import type { SupportedSensor, DataInterpretation } from '../types';
import { Flame, ShieldAlert, Satellite, Calendar, MapPin, SlidersHorizontal } from 'lucide-react';

export const Header: React.FC = () => {
  const { state, setSensor, setInterpretation, resetFilters } = useApp();

  const handleSensorChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSensor(e.target.value as SupportedSensor);
  };

  const handleInterpChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setInterpretation(e.target.value as DataInterpretation);
  };

  const getProvenanceBadge = () => {
    switch (state.sensor) {
      case 'harmonised':
        return {
          label: 'Harmonised (Hurdle GLM + Model B)',
          bg: '#14532d',
          color: '#86efac',
          border: '#22c55e',
          badge: 'GATE B CONDITIONAL',
          tooltip: 'Calibrated counterfactual predicting VIIRS-equivalent counts from Terra+Aqua for typical fire weeks (M <= 5).',
        };
      case 'viirs':
        return {
          label: 'VIIRS VNP14IMGTDL (Observed 375m)',
          bg: '#1e3a8a',
          color: '#93c5fd',
          border: '#3b82f6',
          badge: 'PRIMARY OBSERVED',
          tooltip: 'Direct satellite active-fire observations; constellation exposure varies by era (14/28/42 passes/wk).',
        };
      case 'modis':
        return {
          label: 'MODIS MCD14DL (Historical 1km)',
          bg: '#78350f',
          color: '#fde68a',
          border: '#f59e0b',
          badge: 'CALIBRATION BASE',
          tooltip: 'Historical reference base. Pre-July 2002 Terra-only data lack diurnal bridge and are blocked.',
        };
      default:
        return {
          label: 'Multi-Sensor Archive',
          bg: '#1e293b',
          color: '#cbd5e1',
          border: '#475569',
          badge: 'ALL STREAMS',
          tooltip: 'All satellite observations combined across MODIS and VIIRS.',
        };
    }
  };

  const badgeInfo = getProvenanceBadge();

  return (
    <header className="app-header">
      <div className="header-top">
        <div className="brand-section">
          <Flame className="brand-icon" size={28} />
          <div>
            <h1 className="brand-title">NEBULENS</h1>
            <span className="brand-subtitle">
              Satellite Wildfire Observation, Harmonisation & Anomaly Analysis Explorer
            </span>
          </div>
        </div>

        <div className="provenance-tag-container" title={badgeInfo.tooltip}>
          <div
            className="provenance-tag"
            style={{
              backgroundColor: badgeInfo.bg,
              color: badgeInfo.color,
              borderColor: badgeInfo.border,
            }}
          >
            <ShieldAlert size={16} />
            <span className="provenance-tag-text">{badgeInfo.label}</span>
            <span className="provenance-status-pill">{badgeInfo.badge}</span>
          </div>
        </div>
      </div>

      <div className="header-controls">
        <div className="control-group">
          <Satellite size={16} className="control-icon" />
          <label htmlFor="sensor-select" className="control-label">Sensor:</label>
          <select
            id="sensor-select"
            value={state.sensor}
            onChange={handleSensorChange}
            className="control-select"
          >
            <option value="harmonised">Harmonised (Long-term continuous)</option>
            <option value="viirs">VIIRS VNP14IMGTDL (2012–Present)</option>
            <option value="modis">MODIS MCD14DL (2000–Present)</option>
            <option value="all">All Combined Streams</option>
          </select>
        </div>

        <div className="control-group">
          <SlidersHorizontal size={16} className="control-icon" />
          <label htmlFor="interp-select" className="control-label">Interpretation:</label>
          <select
            id="interp-select"
            value={state.interpretation}
            onChange={handleInterpChange}
            className="control-select"
          >
            <option value="observed">Direct Satellite Observation</option>
            <option value="harmonised">Calibrated Harmonised Estimate</option>
            <option value="counterfactual">Counterfactual Prediction</option>
          </select>
        </div>

        <div className="control-group">
          <Calendar size={16} className="control-icon" />
          <span className="control-label">Range:</span>
          <span className="control-val">{state.dateRange.startDate} → {state.dateRange.endDate}</span>
        </div>

        <div className="control-group">
          <MapPin size={16} className="control-icon" />
          <span className="control-label">Region:</span>
          <span className="control-val">{state.region.name}</span>
        </div>

        <button
          type="button"
          className="btn-reset-filters"
          onClick={resetFilters}
          title="Reset all filters to defaults"
        >
          Reset Filters
        </button>
      </div>
    </header>
  );
};
