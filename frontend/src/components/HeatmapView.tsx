import React, { useEffect, useState, useMemo, useRef } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { SeriesDataPoint } from '../types';
import { ChevronLeft, ChevronRight, Info, Download } from 'lucide-react';
import { toPng } from 'html-to-image';

export const HeatmapView: React.FC = () => {
  const { state, setActiveAnalysisDate, setView } = useApp();
  const [selectedYear, setSelectedYear] = useState<number>(2023);
  const [metric, setMetric] = useState<'count' | 'frp'>('count');
  const [seriesData, setSeriesData] = useState<SeriesDataPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const heatmapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const startDate = `${selectedYear}-01-01`;
    const endDate = `${selectedYear}-12-31`;

    setLoading(true);
    apiService
      .getSeries({
        start_date: startDate,
        end_date: endDate,
        sensor: state.sensor !== 'all' ? state.sensor : undefined,
        district_name: state.region.type === 'district' ? state.region.name : undefined,
      })
      .then((res) => {
        setSeriesData(res.data || []);
      })
      .catch((e) => console.error('Failed to load series for heatmap:', e))
      .finally(() => setLoading(false));
  }, [selectedYear, state.sensor, state.region]);

  const dateValueMap = useMemo(() => {
    const map = new Map<string, { count: number; frp: number; h_source: string }>();
    seriesData.forEach((d) => {
      map.set(d.date, { count: d.count, frp: d.frp, h_source: d.h_source });
    });
    return map;
  }, [seriesData]);

  // Months array
  const months = [
    { num: 1, name: 'Jan', days: 31 },
    { num: 2, name: 'Feb', days: selectedYear % 4 === 0 ? 29 : 28 },
    { num: 3, name: 'Mar', days: 31 },
    { num: 4, name: 'Apr', days: 30 },
    { num: 5, name: 'May', days: 31 },
    { num: 6, name: 'Jun', days: 30 },
    { num: 7, name: 'Jul', days: 31 },
    { num: 8, name: 'Aug', days: 31 },
    { num: 9, name: 'Sep', days: 30 },
    { num: 10, name: 'Oct', days: 31 },
    { num: 11, name: 'Nov', days: 30 },
    { num: 12, name: 'Dec', days: 31 },
  ];

  const getColor = (val: number, isMissing: boolean) => {
    if (isMissing) return '#1e293b'; // missing/unrecorded
    if (val === 0) return '#0f172a'; // true zero
    if (metric === 'count') {
      if (val < 10) return '#064e3b';
      if (val < 30) return '#047857';
      if (val < 70) return '#eab308';
      if (val < 150) return '#f97316';
      return '#ef4444';
    } else {
      if (val < 20) return '#064e3b';
      if (val < 80) return '#047857';
      if (val < 200) return '#eab308';
      if (val < 500) return '#f97316';
      return '#ef4444';
    }
  };

  const handleCellClick = (dateStr: string) => {
    setActiveAnalysisDate(dateStr);
    setView('map');
  };

  const exportPng = async () => {
    if (!heatmapRef.current) return;
    try {
      const dataUrl = await toPng(heatmapRef.current, { cacheBust: true });
      const link = document.createElement('a');
      link.download = `nebulens_heatmap_${selectedYear}_${metric}.png`;
      link.href = dataUrl;
      link.click();
    } catch (e) {
      console.error('Heatmap export failed:', e);
    }
  };

  return (
    <div className="view-content heatmap-container" ref={heatmapRef}>
      <div className="section-header">
        <div className="section-title-row">
          <div>
            <h2>Historical Fire Activity Calendar Heatmap</h2>
            <p className="section-description">
              Daily temporal distribution of satellite fire detections for {state.region.name}. Click any day to inspect in Map view.
            </p>
          </div>

          <div className="heatmap-actions">
            {/* Year Selector */}
            <div className="year-selector">
              <button
                type="button"
                className="btn-toolbar"
                onClick={() => setSelectedYear((y) => Math.max(2000, y - 1))}
              >
                <ChevronLeft size={16} />
              </button>
              <select
                className="control-select font-bold"
                value={selectedYear}
                onChange={(e) => setSelectedYear(Number(e.target.value))}
              >
                {Array.from({ length: 27 }, (_, i) => 2000 + i).map((yr) => (
                  <option key={yr} value={yr}>
                    Year {yr}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="btn-toolbar"
                onClick={() => setSelectedYear((y) => Math.min(2026, y + 1))}
              >
                <ChevronRight size={16} />
              </button>
            </div>

            {/* Metric Toggle */}
            <div className="metric-toggle">
              <button
                type="button"
                className={`btn-toggle ${metric === 'count' ? 'active' : ''}`}
                onClick={() => setMetric('count')}
              >
                Fire Count
              </button>
              <button
                type="button"
                className={`btn-toggle ${metric === 'frp' ? 'active' : ''}`}
                onClick={() => setMetric('frp')}
              >
                Total FRP (MW)
              </button>
            </div>

            <button type="button" className="btn-toolbar" onClick={exportPng}>
              <Download size={14} />
              <span>Export PNG</span>
            </button>
          </div>
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="calendar-grid-card">
        {loading ? (
          <div className="loading-placeholder">Loading {selectedYear} seasonal fire time series...</div>
        ) : (
          <div className="calendar-grid">
            {months.map((m) => (
              <div key={m.num} className="month-column">
                <div className="month-name">{m.name}</div>
                <div className="days-stack">
                  {Array.from({ length: m.days }, (_, idx) => {
                    const dayNum = idx + 1;
                    const dayPad = dayNum.toString().padStart(2, '0');
                    const monthPad = m.num.toString().padStart(2, '0');
                    const dateStr = `${selectedYear}-${monthPad}-${dayPad}`;

                    const entry = dateValueMap.get(dateStr);
                    const isMissing = !entry;
                    const val = entry ? (metric === 'count' ? entry.count : entry.frp) : 0;
                    const bg = getColor(val, isMissing);

                    return (
                      <div
                        key={dayNum}
                        className="day-cell"
                        style={{ backgroundColor: bg }}
                        title={`${dateStr}: ${
                          isMissing
                            ? 'Missing / Unacquired'
                            : `${val.toFixed(1)} ${metric === 'count' ? 'fires' : 'MW'}`
                        } (${entry?.h_source || 'N/A'})`}
                        onClick={() => !isMissing && handleCellClick(dateStr)}
                      >
                        <span className="day-number">{dayNum}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Legend */}
        <div className="heatmap-legend">
          <div className="legend-scale-title">
            Metric: <strong>{metric === 'count' ? 'Fire Detection Count' : 'Total Fire Radiative Power (MW)'}</strong>
          </div>
          <div className="scale-boxes">
            <div className="scale-item"><span className="scale-color" style={{ background: '#1e293b' }}></span> Missing/Null</div>
            <div className="scale-item"><span className="scale-color" style={{ background: '#0f172a' }}></span> 0 (Zero)</div>
            <div className="scale-item"><span className="scale-color" style={{ background: '#064e3b' }}></span> Low</div>
            <div className="scale-item"><span className="scale-color" style={{ background: '#047857' }}></span> Moderate</div>
            <div className="scale-item"><span className="scale-color" style={{ background: '#eab308' }}></span> Elevated</div>
            <div className="scale-item"><span className="scale-color" style={{ background: '#f97316' }}></span> High</div>
            <div className="scale-item"><span className="scale-color" style={{ background: '#ef4444' }}></span> Extreme Peak</div>
          </div>
          <div className="legend-caveat">
            <Info size={12} />
            <span>Missing observations are explicitly displayed as distinct from zero fire activity. Pre-2012 values reflect Model B harmonised estimates.</span>
          </div>
        </div>
      </div>
    </div>
  );
};
