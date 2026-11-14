import React, { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { AnomalyRecord } from '../types';
import {
  AlertTriangle,
  HelpCircle,
  Activity,
} from 'lucide-react';

export const AnomalyView: React.FC = () => {
  const { state, setSelectedAnomaly, setView } = useApp();

  const [anomalies, setAnomalies] = useState<AnomalyRecord[]>([]);
  const [selected, setSelected] = useState<AnomalyRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [minScore, setMinScore] = useState<number>(3.5);
  const [yearFilter, setYearFilter] = useState<number>(2023);

  useEffect(() => {
    setLoading(true);
    apiService
      .getAnomalies({
        year: yearFilter,
        district_name: state.region.type === 'district' ? state.region.name : undefined,
        min_score: minScore,
      })
      .then((res) => {
        setAnomalies(res.data || []);
        if (res.data?.length > 0) {
          setSelected(res.data[0]);
          setSelectedAnomaly(res.data[0]);
        } else {
          setSelected(null);
          setSelectedAnomaly(null);
        }
      })
      .catch((e) => console.error('Failed to load anomalies:', e))
      .finally(() => setLoading(false));
  }, [yearFilter, minScore, state.region]);

  const handleSelect = (item: AnomalyRecord) => {
    setSelected(item);
    setSelectedAnomaly(item);
  };

  const getMonthName = (m: number) => {
    const d = new Date(2023, m - 1, 1);
    return d.toLocaleString('en-US', { month: 'long' });
  };

  return (
    <div className="view-content anomalies-container">
      <div className="section-header">
        <div className="section-title-row">
          <div>
            <h2>Negative Binomial Anomaly Investigation Engine</h2>
            <p className="section-description">
              Robust MAD-scored anomalies ($z \ge 3.5$) with Negative Binomial fallbacks across 27 years of baseline records.
            </p>
          </div>

          <div className="anomaly-filters">
            <div className="filter-group">
              <label htmlFor="year-filter">Target Year:</label>
              <select
                id="year-filter"
                className="control-select"
                value={yearFilter}
                onChange={(e) => setYearFilter(Number(e.target.value))}
              >
                {Array.from({ length: 27 }, (_, i) => 2000 + i).map((yr) => (
                  <option key={yr} value={yr}>
                    {yr}
                  </option>
                ))}
              </select>
            </div>

            <div className="filter-group">
              <label htmlFor="score-filter">Min Score (z):</label>
              <select
                id="score-filter"
                className="control-select"
                value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
              >
                <option value={3.5}>z ≥ 3.5 (Standard Statistical Threshold)</option>
                <option value={5.0}>z ≥ 5.0 (Extreme Fire Anomaly)</option>
                <option value={10.0}>z ≥ 10.0 (Unprecedented Mega-Fire)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="anomalies-grid-layout">
        {/* Left: Flagged Anomalies List */}
        <div className="card anomalies-list-card">
          <div className="card-header">
            <AlertTriangle size={18} className="text-warning" />
            <h3>Flagged Anomalies ({anomalies.length} Events)</h3>
          </div>

          <div className="card-body">
            {loading ? (
              <div className="loading-placeholder">Computing robust MAD anomalies...</div>
            ) : anomalies.length === 0 ? (
              <div className="empty-chart">No anomalies met the z ≥ {minScore} threshold in {yearFilter} for {state.region.name}.</div>
            ) : (
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>District</th>
                      <th>Month</th>
                      <th>Observed</th>
                      <th>Baseline Median</th>
                      <th>Robust Score (z)</th>
                      <th>Validation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {anomalies.map((item, idx) => {
                      const isSelected = selected?.district_name === item.district_name && selected?.month === item.month;
                      return (
                        <tr
                          key={`${item.district_name}-${item.month}-${idx}`}
                          className={isSelected ? 'selected-row' : ''}
                          onClick={() => handleSelect(item)}
                        >
                          <td className="font-bold">{item.district_name}</td>
                          <td>{getMonthName(item.month)}</td>
                          <td>{item.raw_observed.toFixed(1)}</td>
                          <td>{item.baseline_median.toFixed(2)}</td>
                          <td>
                            <span className="badge badge-danger font-bold">
                              z = {item.z_score.toFixed(2)}
                            </span>
                          </td>
                          <td>
                            <span className="badge badge-outline small">
                              {item.validation_status}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Right: "Why was this flagged?" Scientific Evidence Panel */}
        <div className="card anomaly-evidence-card">
          <div className="card-header">
            <HelpCircle size={18} className="text-primary" />
            <h3>Why Was This Flagged? (Evidence & Attribution Context)</h3>
          </div>

          <div className="card-body">
            {selected ? (
              <div className="evidence-content">
                <div className="evidence-hero">
                  <h4>
                    {selected.district_name} — {getMonthName(selected.month)} {selected.target_year}
                  </h4>
                  <div className="evidence-z-score">
                    Anomaly Score: <strong>z = {selected.z_score.toFixed(2)}</strong> (Exceeds threshold 3.50)
                  </div>
                </div>

                <div className="evidence-metrics-grid">
                  <div className="metric-box">
                    <span className="metric-label">Evaluated Fire Count</span>
                    <span className="metric-val text-fire">{selected.evaluated_value.toFixed(1)}</span>
                    <span className="metric-sub">Raw: {selected.raw_observed.toFixed(1)}</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Seasonal Baseline Median</span>
                    <span className="metric-val">{selected.baseline_median.toFixed(2)}</span>
                    <span className="metric-sub">Across {selected.baseline_n_years} reference years</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Median Absolute Deviation (MAD)</span>
                    <span className="metric-val">{selected.baseline_mad.toFixed(3)}</span>
                    <span className="metric-sub">Method: {selected.scoring_method}</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Constellation Exposure Standardized</span>
                    <span className="metric-val">
                      {selected.is_exposure_standardized ? 'YES (Preserved)' : 'NO (Baseline)'}
                    </span>
                    <span className="metric-sub">Stream: {selected.target_h_source}</span>
                  </div>
                </div>

                {/* Structured Evidence Questions */}
                <div className="evidence-qa-section">
                  <div className="qa-item">
                    <div className="qa-q">1. What changed?</div>
                    <div className="qa-a">
                      Fire detections in {selected.district_name} reached {selected.raw_observed.toFixed(1)}, representing a{' '}
                      <strong>{(selected.evaluated_value / Math.max(0.1, selected.baseline_median)).toFixed(1)}×</strong> elevation over the seasonal expectation.
                    </div>
                  </div>

                  <div className="qa-item">
                    <div className="qa-q">2. Compared with which baseline?</div>
                    <div className="qa-a">
                      Calibrated historical baseline of {selected.baseline_n_years} years for month {selected.month} in {selected.district_name} (median = {selected.baseline_median.toFixed(2)}).
                    </div>
                  </div>

                  <div className="qa-item">
                    <div className="qa-q">3. What alternative explanation exists?</div>
                    <div className="qa-a">
                      {selected.is_exposure_standardized
                        ? 'Constellation exposure expanded during this era (NOAA-20/21 added 28 to 42 passes/wk). True fire escalation must be separated from satellite overpass opportunity.'
                        : 'Single-sensor or baseline exposure era. Elevation is less likely to be pure orbital expansion.'}
                    </div>
                  </div>

                  <div className="qa-item">
                    <div className="qa-q">4. Scientific Validation & Gate B Caveat</div>
                    <div className="qa-a text-warning">
                      Validation status: <strong>{selected.validation_status}</strong>.
                      Anomalies are research detection signals and must not be described as ground-verified wildfires without multi-evidence cross-checking.
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  className="btn-toolbar full-width mt-3"
                  onClick={() => setView('attribution')}
                >
                  <Activity size={16} /> Open Multi-Evidence Attribution Engine
                </button>
              </div>
            ) : (
              <div className="empty-detail">Select an anomaly event to inspect its baseline evidence.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
