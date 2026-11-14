import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from 'recharts';
import { useApp } from '../context/AppContext';
import { BarChart3, PieChart as PieIcon, Layers, TrendingUp, Info } from 'lucide-react';

// Curated data derived from official project analysis tables (M3.3 & M3.5)
const LANDCOVER_FRP_DATA = [
  { name: 'Forest & Woodland', modis_mean: 27.1, viirs_mean: 6.5, count: 48688, color: '#16a34a' },
  { name: 'Cropland & Mosaic', modis_mean: 22.3, viirs_mean: 5.9, count: 109672, color: '#ca8a04' },
  { name: 'Shrubland & Savanna', modis_mean: 18.4, viirs_mean: 5.5, count: 56486, color: '#d97706' },
  { name: 'Urban / Developed', modis_mean: 17.3, viirs_mean: 4.6, count: 29159, color: '#64748b' },
  { name: 'Intensive Delta', modis_mean: 16.1, viirs_mean: 4.9, count: 17084, color: '#0284c7' },
  { name: 'Industrial Static', modis_mean: 14.2, viirs_mean: 1.7, count: 13794, color: '#9333ea' },
];

const DIURNAL_SPLIT_DATA = [
  { name: 'VIIRS Daytime', value: 193770, color: '#38bdf8' },
  { name: 'VIIRS Nighttime', value: 59085, color: '#1e3a8a' },
  { name: 'MODIS Aqua (Afternoon)', value: 12091, color: '#f59e0b' },
  { name: 'MODIS Terra (Morning)', value: 11362, color: '#ef4444' },
];

const SENSOR_EXPANSION_TREND = [
  { year: 2012, viirs: 3824, modis: 842, passes_wk: 14 },
  { year: 2014, viirs: 6100, modis: 910, passes_wk: 14 },
  { year: 2016, viirs: 9450, modis: 1120, passes_wk: 14 },
  { year: 2018, viirs: 12340, modis: 1050, passes_wk: 28 },
  { year: 2020, viirs: 15400, modis: 1180, passes_wk: 28 },
  { year: 2022, viirs: 13355, modis: 1020, passes_wk: 28 },
  { year: 2023, viirs: 27291, modis: 1187, passes_wk: 42 },
  { year: 2024, viirs: 39453, modis: 1511, passes_wk: 42 },
  { year: 2025, viirs: 24210, modis: 1315, passes_wk: 42 },
  { year: 2026, viirs: 29432, modis: 1649, passes_wk: 42 },
];

export const VisualisationView: React.FC = () => {
  const { state } = useApp();
  const [activeTab, setActiveTab] = useState<'landcover' | 'diurnal' | 'sensors'>('landcover');

  return (
    <div className="view-content visualisation-view">
      <div className="section-header">
        <h2>Multi-Dimensional Fire Data Visualisation</h2>
        <p className="section-description">
          Cross-sensor fire radiative power distributions, diurnal overpass splits, and landcover fire profiles across Tamil Nadu.
        </p>
      </div>

      {/* Visualisation Mode Tabs */}
      <div className="tab-control-group" style={{ marginBottom: '1.25rem', display: 'flex', gap: '0.5rem' }}>
        <button
          type="button"
          className={`btn-toolbar ${activeTab === 'landcover' ? 'btn-active' : ''}`}
          onClick={() => setActiveTab('landcover')}
        >
          <Layers size={15} />
          <span>FRP by Landcover Ecosystem</span>
        </button>

        <button
          type="button"
          className={`btn-toolbar ${activeTab === 'diurnal' ? 'btn-active' : ''}`}
          onClick={() => setActiveTab('diurnal')}
        >
          <PieIcon size={15} />
          <span>Day vs. Night Overpass Detections</span>
        </button>

        <button
          type="button"
          className={`btn-toolbar ${activeTab === 'sensors' ? 'btn-active' : ''}`}
          onClick={() => setActiveTab('sensors')}
        >
          <TrendingUp size={15} />
          <span>Constellation Expansion vs Observed Count</span>
        </button>
      </div>

      {/* TAB 1: FRP by Landcover */}
      {activeTab === 'landcover' && (
        <div className="card full-width">
          <div className="card-header">
            <BarChart3 size={20} className="card-icon" />
            <h3>Mean Fire Radiative Power (MW) Across Landcover Classes</h3>
          </div>
          <div className="card-body">
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Comparing average fire intensity (FRP) between MODIS (1km) and VIIRS (375m). Notice how coarse MODIS pixels aggregate energy, yielding higher mean FRP, whereas VIIRS catches low-intensity understory and field fires.
            </p>
            <div style={{ height: '380px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={LANDCOVER_FRP_DATA} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" angle={-15} textAnchor="end" interval={0} fontSize={12} />
                  <YAxis stroke="#94a3b8" label={{ value: 'Mean FRP (MW)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                    formatter={(val: any) => [`${val} MW`, 'Mean FRP']}
                  />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px' }} />
                  <Bar dataKey="modis_mean" name="MODIS Mean FRP (MW)" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="viirs_mean" name="VIIRS Mean FRP (MW)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Diurnal Split */}
      {activeTab === 'diurnal' && (
        <div className="card full-width">
          <div className="card-header">
            <PieIcon size={20} className="card-icon" />
            <h3>Observation Share by Satellite Overpass & Time of Day</h3>
          </div>
          <div className="card-body" style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '2rem' }}>
            <div style={{ height: '360px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={DIURNAL_SPLIT_DATA}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={120}
                    paddingAngle={3}
                    dataKey="value"
                    label={({ name, percent }) => `${name} (${((percent || 0) * 100).toFixed(1)}%)`}
                  >
                    {DIURNAL_SPLIT_DATA.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                    formatter={(val: any) => [`${Number(val).toLocaleString()} Detections`, 'Count']}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '1rem' }}>
              <h4 style={{ color: '#f8fafc', fontSize: '1rem' }}>Why Diurnal Distribution Matters:</h4>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                • <strong>Peak Fire Activity:</strong> Over 70% of wildfire detections occur during afternoon passes (13:30 local time), when temperatures peak and relative humidity reaches diurnal minimums.
              </p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.5' }}>
                • <strong>Nighttime Sensitivity:</strong> VIIRS captures significant night detections (21.4%) that were virtually invisible to MODIS due to higher sensor noise and coarser resolution.
              </p>
              <div style={{ padding: '0.75rem', background: '#1e293b', borderRadius: '6px', fontSize: '0.8rem', color: '#94a3b8' }}>
                <Info size={14} style={{ display: 'inline', marginRight: '6px', color: '#38bdf8' }} />
                Active region filter currently: <strong>{state.region.name}</strong>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Constellation Expansion Trend */}
      {activeTab === 'sensors' && (
        <div className="card full-width">
          <div className="card-header">
            <TrendingUp size={20} className="card-icon" />
            <h3>Sensor Constellation Growth vs. Apparent Detections (2012–2026)</h3>
          </div>
          <div className="card-body">
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
              Shows how apparent active fire counts expand as new satellites are launched (Suomi-NPP in 2012, NOAA-20 in 2018, NOAA-21 in 2023), illustrating the necessity of NEBULENS harmonisation models.
            </p>
            <div style={{ height: '380px', width: '100%' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={SENSOR_EXPANSION_TREND} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="year" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" label={{ value: 'Annual Detections', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc' }}
                    formatter={(val: any, name: any) => [
                      name === 'passes_wk' ? `${val} passes/week` : `${Number(val).toLocaleString()} fires`,
                      name === 'viirs' ? 'VIIRS Detections' : name === 'modis' ? 'MODIS Detections' : 'Passes / Wk',
                    ]}
                  />
                  <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px' }} />
                  <Line type="monotone" dataKey="viirs" name="VIIRS Observed" stroke="#38bdf8" strokeWidth={3} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="modis" name="MODIS Observed" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3 }} />
                  <Line type="stepAfter" dataKey="passes_wk" name="Exposure Passes/Week (×100 for scale)" stroke="#a855f7" strokeWidth={2} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
