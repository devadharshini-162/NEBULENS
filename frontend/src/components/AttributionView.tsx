import React, { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { AttributionRecord } from '../types';
import {
  Crosshair,
  ShieldAlert,
  Satellite,
  Filter,
} from 'lucide-react';

export const AttributionView: React.FC = () => {
  const { state } = useApp();
  const [attributionList, setAttributionList] = useState<AttributionRecord[]>([]);
  const [selectedRecord, setSelectedRecord] = useState<AttributionRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [outcomeFilter, setOutcomeFilter] = useState<string>('all');

  useEffect(() => {
    setLoading(true);
    apiService
      .getAttribution({
        district_name: state.region.type === 'district' ? state.region.name : undefined,
        outcome: outcomeFilter !== 'all' ? outcomeFilter : undefined,
      })
      .then((res) => {
        const sorted = (res.data || []).sort((a, b) => {
          // Prioritize flagged anomalies and sensor-driven changes first
          if (a.anomaly_flag !== b.anomaly_flag) return a.anomaly_flag ? -1 : 1;
          return b.count_z_score - a.count_z_score;
        });
        setAttributionList(sorted);
        if (sorted.length > 0) setSelectedRecord(sorted[0]);
        else setSelectedRecord(null);
      })
      .catch((e) => console.error('Failed to load attribution:', e))
      .finally(() => setLoading(false));
  }, [state.region, outcomeFilter]);

  const getOutcomeBadge = (outcome: string) => {
    switch (outcome) {
      case 'likely_sensor_related_change':
        return {
          label: 'Sensor / Exposure Driven',
          badgeClass: 'badge-warning',
          desc: 'Detection elevation coincides with satellite constellation expansion without corresponding FRP increase.',
        };
      case 'consistent_with_increased_fire_activity':
        return {
          label: 'True Fire Activity Escalation',
          badgeClass: 'badge-danger',
          desc: 'Significant elevation in both detection count and fire radiative power exceeding observational artifacts.',
        };
      case 'mixed_effects':
        return {
          label: 'Mixed Effects (Sensor + Fire)',
          badgeClass: 'badge-primary',
          desc: 'Both fire escalation and satellite opportunity changes contribute to the observed signal.',
        };
      case 'consistent_with_baseline_variation':
        return {
          label: 'Baseline Seasonal Variation',
          badgeClass: 'badge-outline',
          desc: 'Fire detections conform to normal expected historical variation for this month.',
        };
      default:
        return {
          label: 'Insufficient Evidence',
          badgeClass: 'badge-outline',
          desc: 'Sparse detections or contradictory evidence prevents definitive attribution.',
        };
    }
  };

  return (
    <div className="view-content attribution-container">
      <div className="section-header">
        <div className="section-title-row">
          <div>
            <h2>Multi-Evidence Anomaly Attribution Engine</h2>
            <p className="section-description">
              Classifying fire anomaly drivers using combined satellite FRP, constellation exposure passes, and burned-area evidence.
            </p>
          </div>

          <div className="attribution-filters">
            <div className="filter-group">
              <label htmlFor="outcome-select"><Filter size={14} /> Outcome:</label>
              <select
                id="outcome-select"
                className="control-select"
                value={outcomeFilter}
                onChange={(e) => setOutcomeFilter(e.target.value)}
              >
                <option value="all">All Outcomes</option>
                <option value="likely_sensor_related_change">Sensor / Exposure Driven</option>
                <option value="consistent_with_increased_fire_activity">True Fire Escalation</option>
                <option value="mixed_effects">Mixed Effects</option>
                <option value="insufficient_evidence">Insufficient Evidence</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <div className="attribution-grid-layout">
        {/* Table of Attribution Outcomes */}
        <div className="card attribution-table-card">
          <div className="card-header">
            <Crosshair size={18} />
            <h3>Attribution Classified Events ({attributionList.length} Records)</h3>
          </div>

          <div className="card-body">
            {loading ? (
              <div className="loading-placeholder">Evaluating multi-evidence decision trees...</div>
            ) : attributionList.length === 0 ? (
              <div className="empty-chart">No attribution records found for current filters.</div>
            ) : (
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>District</th>
                      <th>Period</th>
                      <th>Attribution Outcome</th>
                      <th>Confidence</th>
                      <th>Passes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {attributionList.slice(0, 50).map((item, idx) => {
                      const isSelected = selectedRecord?.district_name === item.district_name && selectedRecord?.month === item.month && selectedRecord?.year === item.year;
                      const badge = getOutcomeBadge(item.attribution_outcome);
                      return (
                        <tr
                          key={`${item.district_name}-${item.year}-${item.month}-${idx}`}
                          className={isSelected ? 'selected-row' : ''}
                          onClick={() => setSelectedRecord(item)}
                        >
                          <td className="font-bold">{item.district_name}</td>
                          <td>{item.year}-{item.month.toString().padStart(2, '0')}</td>
                          <td>
                            <span className={`badge ${badge.badgeClass}`}>
                              {badge.label}
                            </span>
                          </td>
                          <td className="font-semibold">{item.confidence.toUpperCase()}</td>
                          <td>{item.exposure_passes} passes/wk</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Detailed Decision Tree & Evidence Breakdown */}
        <div className="card attribution-detail-card">
          <div className="card-header">
            <Satellite size={18} />
            <h3>Scientific Evidence Synthesis</h3>
          </div>

          <div className="card-body">
            {selectedRecord ? (
              <div className="attribution-detail-content">
                <div className="attribution-hero">
                  <h4>{selectedRecord.district_name} — {selectedRecord.year}-{selectedRecord.month.toString().padStart(2, '0')}</h4>
                  <div className={`badge large ${getOutcomeBadge(selectedRecord.attribution_outcome).badgeClass}`}>
                    {getOutcomeBadge(selectedRecord.attribution_outcome).label}
                  </div>
                </div>

                <div className="attribution-explanation-box">
                  <strong>Automated Attribution Analysis:</strong>
                  <p>{selectedRecord.explanation}</p>
                </div>

                <div className="attribution-metrics-grid">
                  <div className="metric-box">
                    <span className="metric-label">Observed Fire Count</span>
                    <span className="metric-val">{selectedRecord.observed_count}</span>
                    <span className="metric-sub">Baseline Median: {selectedRecord.baseline_median.toFixed(2)}</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Robust Score</span>
                    <span className="metric-val text-fire">z = {selectedRecord.count_z_score.toFixed(2)}</span>
                    <span className="metric-sub">Anomaly Flag: {selectedRecord.anomaly_flag ? 'TRUE' : 'FALSE'}</span>
                  </div>

                  <div className="metric-box">
                    <span className="metric-label">Constellation Exposure</span>
                    <span className="metric-val">{selectedRecord.exposure_passes} passes/wk</span>
                    <span className="metric-sub">Theoretical passes</span>
                  </div>
                </div>

                <div className="attribution-gate-b-status" style={{ marginTop: '1rem' }}>
                  <ShieldAlert size={16} />
                  <span>Validation Status: <strong>{selectedRecord.validation_status.replace(/_/g, ' ')}</strong></span>
                </div>
              </div>
            ) : (
              <div className="empty-detail">Select an attribution record to review scientific drivers.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
