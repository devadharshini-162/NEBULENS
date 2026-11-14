import React, { useEffect, useState, useRef } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { SeriesDataPoint } from '../types';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { Download, Info, ShieldAlert } from 'lucide-react';
import { toPng } from 'html-to-image';

export const SeriesView: React.FC = () => {
  const { state } = useApp();
  const [seriesData, setSeriesData] = useState<SeriesDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [showUncertainty, setShowUncertainty] = useState<boolean>(true);
  const [showFrp, setShowFrp] = useState<boolean>(true);

  const chartExportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setLoading(true);
    apiService
      .getSeries({
        start_date: state.dateRange.startDate,
        end_date: state.dateRange.endDate,
        sensor: state.sensor !== 'all' ? state.sensor : undefined,
        district_name: state.region.type === 'district' ? state.region.name : undefined,
        grid_cell_id: state.region.type === 'cell' ? state.region.id : undefined,
      })
      .then((res) => {
        setSeriesData(res.data || []);
      })
      .catch((e) => console.error('Failed to load series:', e))
      .finally(() => setLoading(false));
  }, [state.dateRange, state.sensor, state.region]);

  const exportPng = async () => {
    if (!chartExportRef.current) return;
    try {
      const dataUrl = await toPng(chartExportRef.current, { cacheBust: true });
      const link = document.createElement('a');
      link.download = `nebulens_series_${state.region.id}_${state.dateRange.startDate}_${state.dateRange.endDate}.png`;
      link.href = dataUrl;
      link.click();
    } catch (e) {
      console.error('PNG export failed:', e);
    }
  };

  const isPre2012 = state.dateRange.startDate < '2012-01-20';

  return (
    <div className="view-content series-container" ref={chartExportRef}>
      <div className="section-header">
        <div className="section-title-row">
          <div>
            <h2>Continuous Fire Time Series & Bootstrap Uncertainty</h2>
            <p className="section-description">
              Region: <strong>{state.region.name}</strong> | Range: <strong>{state.dateRange.startDate} to {state.dateRange.endDate}</strong> | Stream: <strong>{state.sensor.toUpperCase()}</strong>
            </p>
          </div>

          <div className="series-actions">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showUncertainty}
                onChange={(e) => setShowUncertainty(e.target.checked)}
              />
              <span>95% Uncertainty Bands</span>
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showFrp}
                onChange={(e) => setShowFrp(e.target.checked)}
              />
              <span>FRP (Secondary Axis)</span>
            </label>

            <button type="button" className="btn-toolbar" onClick={exportPng}>
              <Download size={14} />
              <span>Export Chart PNG</span>
            </button>
          </div>
        </div>
      </div>

      {isPre2012 && (
        <div className="scientific-banner warning">
          <ShieldAlert size={18} />
          <div>
            <strong>Scientific Gate B Caveat (Pre-2012):</strong>
            Continuous estimates prior to 2012 represent Negative Binomial Hurdle GLM + Model B counterfactual predictions.
            Extreme forest fire episodes (M &gt; 10) carry extrapolation risk; bootstrap uncertainty intervals reflect known empirical undercoverage.
          </div>
        </div>
      )}

      {/* Main Chart */}
      <div className="card chart-card">
        {loading ? (
          <div className="loading-placeholder">Loading time series records...</div>
        ) : seriesData.length === 0 ? (
          <div className="empty-chart">No fire activity observations or model estimates returned for this date window.</div>
        ) : (
          <div style={{ width: '100%', height: 460 }}>
            <ResponsiveContainer>
              <ComposedChart data={seriesData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <YAxis
                  yAxisId="left"
                  stroke="#38bdf8"
                  label={{ value: 'Fire Detections (Count)', angle: -90, position: 'insideLeft', fill: '#38bdf8' }}
                />
                {showFrp && (
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    stroke="#f59e0b"
                    label={{ value: 'Total FRP (MW)', angle: 90, position: 'insideRight', fill: '#f59e0b' }}
                  />
                )}
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                  formatter={(value: any, name: any) => [
                    typeof value === 'number' ? value.toFixed(2) : value,
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ paddingTop: 10 }} />

                {/* 95% Bootstrap Uncertainty Band */}
                {showUncertainty && (
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="count_upper_95"
                    stroke="none"
                    fill="#0284c7"
                    fillOpacity={0.18}
                    name="95% CI Upper"
                  />
                )}

                {/* Primary Count Series */}
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="count"
                  stroke="#38bdf8"
                  strokeWidth={2}
                  dot={false}
                  name="Harmonised / Observed Fire Count"
                />

                {/* Secondary FRP Bar */}
                {showFrp && (
                  <Bar
                    yAxisId="right"
                    dataKey="frp"
                    fill="#f59e0b"
                    opacity={0.4}
                    name="Fire Radiative Power (MW)"
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="chart-footer-provenance">
          <Info size={14} />
          <span>
            Source: Harmonised Daily Archives. Daily allocated counts derived from weekly models represent disaggregated estimates, not independently measured daily passes.
          </span>
        </div>
      </div>
    </div>
  );
};
