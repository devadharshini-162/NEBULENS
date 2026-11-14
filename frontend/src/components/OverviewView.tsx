import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import type { AvailabilityData, HealthResponse } from '../types';
import { LoadingState, ErrorState } from './StatusStates';
import { Layers, Calendar, Activity, CheckCircle, ShieldCheck } from 'lucide-react';
import { useApp } from '../context/AppContext';

export const OverviewView: React.FC = () => {
  const { setView } = useApp();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [availability, setAvailability] = useState<AvailabilityData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<number | undefined>(undefined);

  const fetchDiagnostics = async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, a] = await Promise.all([
        apiService.getHealth(),
        apiService.getAvailability(),
      ]);
      setHealth(h);
      setAvailability(a.data);
    } catch (err: any) {
      setError(err?.message || 'Failed to connect to NEBULENS API');
      setStatus(err?.status);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  if (loading) {
    return <LoadingState message="Loading NEBULENS Platform Summary..." />;
  }

  if (error) {
    return (
      <ErrorState
        title="Backend Connectivity Error"
        message={`Unable to reach backend API at ${apiService.getBaseUrl()}. Error: ${error}`}
        status={status}
        onRetry={fetchDiagnostics}
      />
    );
  }

  return (
    <div className="view-content overview-view">
      <div className="section-header">
        <h2>Platform & Observation Summary</h2>
        <p className="section-description">
          Satellite observation coverage, harmonisation status, and operational datasets for Tamil Nadu.
        </p>
      </div>

      <div className="grid-cards">
        {/* Core System Operational Status */}
        <div className="card">
          <div className="card-header">
            <ShieldCheck size={20} className="card-icon" />
            <h3>Observation Engine</h3>
          </div>
          <div className="card-body">
            <div className="status-row">
              <span className="label">Status:</span>
              <span className="value badge-success">
                <CheckCircle size={14} /> OPERATIONAL
              </span>
            </div>
            <div className="status-row">
              <span className="label">Active Core:</span>
              <span className="value">{health?.service || 'NEBULENS Core'}</span>
            </div>
            <div className="status-row">
              <span className="label">Analysis Grid:</span>
              <span className="value">EPSG:6933 (10km Equal-Area)</span>
            </div>
            <div className="status-row">
              <span className="label">Target Province:</span>
              <span className="value font-bold">Tamil Nadu, India</span>
            </div>
          </div>
        </div>

        {/* Spatial & Temporal Coverage */}
        <div className="card">
          <div className="card-header">
            <Calendar size={20} className="card-icon" />
            <h3>Observation Coverage</h3>
          </div>
          <div className="card-body">
            <div className="status-row">
              <span className="label">Study Region:</span>
              <span className="value font-bold">{availability?.study_region || 'Tamil Nadu'}</span>
            </div>
            <div className="status-row">
              <span className="label">Time Span:</span>
              <span className="value font-bold">
                {availability?.temporal_range?.start_date || '2000-11-01'} → {availability?.temporal_range?.end_date || '2026-03-31'}
              </span>
            </div>
            <div className="status-row">
              <span className="label">Coverage Area:</span>
              <span className="value">{availability?.spatial_coverage_km2?.toLocaleString() || '130,058'} km²</span>
            </div>
            <div className="status-row">
              <span className="label">Standard Units:</span>
              <span className="value">Active Fire Count & FRP (MW)</span>
            </div>
          </div>
        </div>

        {/* Quick Launch Card */}
        <div className="card">
          <div className="card-header">
            <Activity size={20} className="card-icon" />
            <h3>Quick Launch</h3>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              Jump straight into live analysis tools:
            </p>
            <button
              className="btn-toolbar btn-active"
              style={{ justifyContent: 'center', padding: '0.5rem' }}
              onClick={() => setView('map')}
            >
              Open Satellite Map & Draw Area
            </button>
            <button
              className="btn-toolbar"
              style={{ justifyContent: 'center', padding: '0.5rem' }}
              onClick={() => setView('hotspots')}
            >
              Inspect Active FIRMS Hotspots
            </button>
            <button
              className="btn-toolbar"
              style={{ justifyContent: 'center', padding: '0.5rem' }}
              onClick={() => setView('series')}
            >
              View Multi-Decadal Time Series
            </button>
          </div>
        </div>

        {/* Constellation & Harmonisation Base */}
        <div className="card full-width">
          <div className="card-header">
            <Layers size={20} className="card-icon" />
            <h3>Satellite Constellation & Sensor Assets</h3>
          </div>
          <div className="card-body">
            <div className="sensor-grid">
              {availability?.sensors &&
                availability.sensors.map((sensor, idx) => (
                  <div key={idx} className="sensor-card">
                    <div className="sensor-title">
                      <h4>{sensor.sensor_name}</h4>
                      <span className="badge badge-success">{sensor.status}</span>
                    </div>
                    <div className="sensor-meta-list">
                      <div><strong>Resolution:</strong> {sensor.nominal_resolution_m}m</div>
                      <div><strong>Active Years:</strong> {sensor.coverage_years}</div>
                      <div><strong>Products:</strong> {sensor.products.join(', ')}</div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
