import React, { useEffect, useState, useRef } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { SeasonalSummary } from '../types';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { GitCompare, Download, Info } from 'lucide-react';
import { toPng } from 'html-to-image';

export const SeasonalCompareView: React.FC = () => {
  const { state } = useApp();
  const [seasonalData, setSeasonalData] = useState<SeasonalSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedMonth, setSelectedMonth] = useState<number>(3); // Default March (peak fire season)

  const compareExportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    apiService
      .getSeasonalComparison({
        district_name: state.region.type === 'district' ? state.region.name : undefined,
      })
      .then((res) => {
        setSeasonalData(res.data || []);
      })
      .catch((e) => console.error('Failed to load seasonal comparison:', e))
      .finally(() => setLoading(false));
  }, [state.region]);

  const monthNames = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  // Filter seasonal comparison data by chosen calendar month across all 27 historical years
  const filteredData = seasonalData
    .filter((d) => d.month === selectedMonth)
    .sort((a, b) => a.target_year - b.target_year);

  const exportPng = async () => {
    if (!compareExportRef.current) return;
    try {
      const dataUrl = await toPng(compareExportRef.current, { cacheBust: true });
      const link = document.createElement('a');
      link.download = `nebulens_seasonal_${state.region.id}_month_${selectedMonth}.png`;
      link.href = dataUrl;
      link.click();
    } catch (e) {
      console.error('PNG export failed:', e);
    }
  };

  return (
    <div className="view-content compare-container" ref={compareExportRef}>
      <div className="section-header">
        <div className="section-title-row">
          <div>
            <h2>Same-Season Multi-Year Fire Comparison (2000–2026)</h2>
            <p className="section-description">
              Comparing equivalent seasonal fire periods across 27 years for <strong>{state.region.name}</strong>.
            </p>
          </div>

          <div className="compare-actions">
            <div className="filter-group">
              <label htmlFor="month-select" className="filter-label">Comparison Month:</label>
              <select
                id="month-select"
                className="control-select font-bold"
                value={selectedMonth}
                onChange={(e) => setSelectedMonth(Number(e.target.value))}
              >
                {monthNames.map((name, idx) => (
                  <option key={name} value={idx + 1}>
                    Month {idx + 1}: {name}
                  </option>
                ))}
              </select>
            </div>

            <button type="button" className="btn-toolbar" onClick={exportPng}>
              <Download size={14} />
              <span>Export Chart PNG</span>
            </button>
          </div>
        </div>
      </div>

      {/* Seasonal Comparison Chart */}
      <div className="card chart-card">
        {loading ? (
          <div className="loading-placeholder">Loading seasonal comparison aggregates...</div>
        ) : filteredData.length === 0 ? (
          <div className="empty-chart">No seasonal records available for this month and geographic partition.</div>
        ) : (
          <div style={{ width: '100%', height: 420 }}>
            <ResponsiveContainer>
              <BarChart data={filteredData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="target_year" stroke="#94a3b8" />
                <YAxis
                  stroke="#38bdf8"
                  label={{ value: `${monthNames[selectedMonth - 1]} Fires (Count)`, angle: -90, position: 'insideLeft', fill: '#38bdf8' }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                  formatter={(value: any, name: any) => [
                    typeof value === 'number' ? value.toFixed(1) : value,
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ paddingTop: 10 }} />
                <Bar dataKey="evaluated_value" fill="#0284c7" name="Evaluated Value (Standardized)" />
                <Bar dataKey="raw_observed" fill="#f59e0b" name="Raw Observed Hotspots" />
                <Bar dataKey="baseline_median" fill="#10b981" name="Historical Baseline Median" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="chart-footer-provenance">
          <Info size={14} />
          <span>
            Note: Pre-2012 values represent Model B harmonised predictions calibrated against Terra+Aqua observations. Post-2012 values represent observed VIIRS 375m hotspots.
          </span>
        </div>
      </div>

      {/* Summary Table */}
      <div className="card full-width">
        <div className="card-header">
          <GitCompare size={18} />
          <h3>Detailed Seasonal Metrics by Year ({monthNames[selectedMonth - 1]})</h3>
        </div>
        <div className="card-body">
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Target Year</th>
                  <th>Month</th>
                  <th>Raw Observed</th>
                  <th>Evaluated Value</th>
                  <th>Baseline Median</th>
                  <th>Robust Score (z)</th>
                  <th>Exposure Standardized</th>
                  <th>Regime / Stream</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((row, idx) => (
                  <tr key={`${row.target_year}-${idx}`}>
                    <td className="font-bold">{row.target_year}</td>
                    <td>{monthNames[row.month - 1]}</td>
                    <td>{row.raw_observed.toFixed(1)}</td>
                    <td className="font-semibold text-primary">{row.evaluated_value.toFixed(1)}</td>
                    <td>{row.baseline_median.toFixed(2)}</td>
                    <td>
                      <span className={`badge ${row.anomaly_flag ? 'badge-danger' : 'badge-outline'}`}>
                        z = {row.z_score.toFixed(2)}
                      </span>
                    </td>
                    <td>{row.is_exposure_standardized ? 'YES' : 'NO'}</td>
                    <td>
                      <span className={`badge ${row.target_year < 2012 ? 'badge-warning' : 'badge-success'}`}>
                        {row.target_year < 2012 ? 'Model B (Pre-2012)' : 'VIIRS Observed'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
