import React, { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { DistrictInfo, ConfidenceFilter, TemporalAggregation } from '../types';
import { Filter, Layers, Calendar, ChevronRight, ChevronLeft } from 'lucide-react';

export const FilterPanel: React.FC = () => {
  const {
    state,
    setConfidence,
    setTemporalAggregation,
    setRegion,
    setDateRange,
    setActiveAnalysisDate,
  } = useApp();

  const [districts, setDistricts] = useState<DistrictInfo[]>([]);

  useEffect(() => {
    apiService.getDistricts().then((res) => {
      if (res.data) setDistricts(res.data);
    }).catch(() => {});
  }, []);

  const handleDistrictChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val === 'ALL') {
      setRegion({
        type: 'state',
        id: 'TN',
        name: 'Tamil Nadu (Statewide)',
      });
    } else {
      setRegion({
        type: 'district',
        id: val,
        name: val,
      });
    }
  };

  const handleConfidenceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setConfidence(e.target.value as ConfidenceFilter);
  };

  const handleAggChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setTemporalAggregation(e.target.value as TemporalAggregation);
  };

  const handleStartDate = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDateRange({ ...state.dateRange, startDate: e.target.value });
  };

  const handleEndDate = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDateRange({ ...state.dateRange, endDate: e.target.value });
  };

  const shiftDay = (days: number) => {
    const curr = new Date(state.activeAnalysisDate);
    curr.setDate(curr.getDate() + days);
    const iso = curr.toISOString().split('T')[0];
    setActiveAnalysisDate(iso);
  };

  return (
    <div className="scientific-filter-panel">
      <div className="filter-header">
        <div className="filter-title">
          <Filter size={16} />
          <span>Scientific Query Filters</span>
        </div>
        <div className="filter-provenance-status">
          <span className="badge badge-outline">ERA: {state.dateRange.startDate.slice(0, 4)}</span>
        </div>
      </div>

      <div className="filter-grid">
        {/* District Selection */}
        <div className="filter-item">
          <label htmlFor="dist-select" className="filter-label">
            <Layers size={14} /> Region / District:
          </label>
          <select
            id="dist-select"
            className="filter-control"
            value={state.region.type === 'state' ? 'ALL' : state.region.type === 'polygon' ? 'POLYGON' : state.region.name}
            onChange={handleDistrictChange}
          >
            {state.region.type === 'polygon' && (
              <option value="POLYGON">📍 Custom Drawn Polygon Area (Active)</option>
            )}
            <option value="ALL">Tamil Nadu (All 38 Districts)</option>
            {districts.map((d) => (
              <option key={d.district_name} value={d.district_name}>
                {d.district_name} ({d.cell_count} cells)
              </option>
            ))}
          </select>
        </div>

        {/* Confidence Filter */}
        <div className="filter-item">
          <label htmlFor="conf-select" className="filter-label">FIRMS Confidence:</label>
          <select
            id="conf-select"
            className="filter-control"
            value={state.confidence}
            onChange={handleConfidenceChange}
          >
            <option value="all">All Detections (Low, Nominal, High)</option>
            <option value="nominal_high">Nominal & High Quality (≥ 30%)</option>
            <option value="high_only">High Quality Only (≥ 80%)</option>
          </select>
        </div>

        {/* Temporal Aggregation */}
        <div className="filter-item">
          <label htmlFor="agg-select" className="filter-label">Aggregation Unit:</label>
          <select
            id="agg-select"
            className="filter-control"
            value={state.temporalAggregation}
            onChange={handleAggChange}
          >
            <option value="day">Daily (Analysis Date IST)</option>
            <option value="week">Weekly (Epi-Week)</option>
            <option value="month">Monthly Profile</option>
          </select>
        </div>

        {/* Global Date Range */}
        <div className="filter-item date-range-item">
          <label className="filter-label">
            <Calendar size={14} /> Global Date Range:
          </label>
          <div className="date-inputs">
            <input
              type="date"
              className="filter-control date-control"
              value={state.dateRange.startDate}
              min="2000-11-01"
              max="2026-08-31"
              onChange={handleStartDate}
            />
            <span className="date-sep">to</span>
            <input
              type="date"
              className="filter-control date-control"
              value={state.dateRange.endDate}
              min="2000-11-01"
              max="2026-08-31"
              onChange={handleEndDate}
            />
          </div>
        </div>

        {/* Single Day Explorer Navigation */}
        <div className="filter-item active-day-item">
          <label htmlFor="active-day-input" className="filter-label">Daily Map & Granule Date:</label>
          <div className="active-day-controls">
            <button
              type="button"
              className="btn-date-step"
              onClick={() => shiftDay(-1)}
              title="Previous Day"
            >
              <ChevronLeft size={16} />
            </button>
            <input
              id="active-day-input"
              type="date"
              className="filter-control date-control active-day-input"
              value={state.activeAnalysisDate}
              min="2000-11-01"
              max="2026-08-31"
              onChange={(e) => setActiveAnalysisDate(e.target.value)}
            />
            <button
              type="button"
              className="btn-date-step"
              onClick={() => shiftDay(1)}
              title="Next Day"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Methodological Context Warning for Pre-2012 */}
      {state.dateRange.startDate < '2012-01-20' && (
        <div className="filter-caveat-notice">
          <strong>Methodological Notice (Pre-2012):</strong> Selected range covers the MODIS-only era.
          Harmonised outputs represent Hurdle GLM + Model B counterfactual predictions.
          {state.dateRange.startDate < '2002-07-04' && (
            <span className="text-danger">
              {' '}Dates prior to July 2002 rely on Terra-only passes and lack diurnal bridge calibration (not validated).
            </span>
          )}
        </div>
      )}
    </div>
  );
};
