import React, { useEffect, useState } from 'react';
import { apiService } from '../services/api';
import type { CoverageRecord, AvailabilityData } from '../types';
import { ShieldCheck, AlertTriangle, Info, CheckCircle2, XCircle, Clock } from 'lucide-react';

export const TransparencyView: React.FC = () => {
  const [coverage, setCoverage] = useState<CoverageRecord[]>([]);
  const [availability, setAvailability] = useState<AvailabilityData | null>(null);
  const [_loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([apiService.getCoverage(), apiService.getAvailability()])
      .then(([covRes, availRes]) => {
        setCoverage(covRes.data || []);
        setAvailability(availRes.data || null);
      })
      .catch((e) => console.error('Failed to load transparency data:', e))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="view-content transparency-container">
      <div className="section-header">
        <h2>Scientific Transparency, Uncertainty &amp; Gate B Framework</h2>
        <p className="section-description">
          Methodological caveats, sensor transition boundaries, constellation exposure timeline, and Gate B serving conditions.
        </p>
      </div>

      <div className="grid-cards">
        {/* Constellation Exposure Timeline */}
        <div className="card full-width">
          <div className="card-header">
            <Clock size={20} className="card-icon" />
            <h3>Satellite Constellation Orbital Exposure Timeline (M3.6 &amp; M7)</h3>
          </div>
          <div className="card-body">
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Temporal Era</th>
                    <th>Nominal Orbital Passes / Wk</th>
                    <th>Active Satellite Platforms</th>
                    <th>Cloud Proxy Evaluation</th>
                    <th>Scientific Methodological Note</th>
                  </tr>
                </thead>
                <tbody>
                  {coverage.map((c, idx) => (
                    <tr key={idx}>
                      <td className="font-bold">{c.period}</td>
                      <td>
                        <span className="badge badge-primary font-bold">
                          {c.orbital_exposure_passes} passes/wk
                        </span>
                      </td>
                      <td>{c.active_satellites.join(', ')}</td>
                      <td className="code-text small">{c.cloud_eval_proxy}</td>
                      <td className="caveat-text">{c.scientific_note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="card-footer-notice">
              <Info size={14} />
              <span>
                <strong>Scientific Integrity Rule #6:</strong> Exposure passes represent theoretical orbital observation opportunity (multipliers: 14x, 28x, 42x), NOT measured or cloud-cleared overpasses.
              </span>
            </div>
          </div>
        </div>

        {/* Gate B Scientific Validation Matrix */}
        <div className="card full-width">
          <div className="card-header">
            <ShieldCheck size={20} className="card-icon" />
            <h3>Gate B Empirical Decision Matrix &amp; Serving Restrictions</h3>
          </div>
          <div className="card-body">
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Domain / Era Regime</th>
                    <th>Gate B Status</th>
                    <th>Serving Eligible</th>
                    <th>Scientific Validated</th>
                    <th>Serving Limitations &amp; Extrapolation Bounds</th>
                  </tr>
                </thead>
                <tbody>
                  {availability?.validation_matrix &&
                    Object.entries(availability.validation_matrix).map(([regime, meta]) => (
                      <tr key={regime}>
                        <td className="font-bold">{regime}</td>
                        <td>
                          <span
                            className={`badge ${
                              meta.status.includes('fully_validated')
                                ? 'badge-success'
                                : meta.status.includes('conditional_pass')
                                ? 'badge-warning'
                                : 'badge-danger'
                            }`}
                          >
                            {meta.status}
                          </span>
                        </td>
                        <td>
                          {meta.serving_eligible ? (
                            <span className="badge badge-success">
                              <CheckCircle2 size={12} /> ELIGIBLE
                            </span>
                          ) : (
                            <span className="badge badge-danger">
                              <XCircle size={12} /> BLOCKED
                            </span>
                          )}
                        </td>
                        <td>
                          {meta.validated ? (
                            <span className="badge badge-success">TRUE</span>
                          ) : (
                            <span className="badge badge-outline">FALSE</span>
                          )}
                        </td>
                        <td className="caveat-text">{meta.note}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Core Methodological Limitations Card */}
        <div className="card full-width">
          <div className="card-header">
            <AlertTriangle size={20} className="text-warning" />
            <h3>Authoritative Project Limitations (M0–M7)</h3>
          </div>
          <div className="card-body">
            <ul className="limitations-list">
              <li>
                <strong>Disaggregated Weekly Estimates:</strong> Daily harmonised values are allocated estimates derived from weekly Negative Binomial Hurdle GLM models; they must not be treated as independent daily satellite observations.
              </li>
              <li>
                <strong>Pre-July 2002 Terra-Only Block:</strong> Terra morning-only passes (~10:30 IST) systematically miss peak afternoon burning. Because the Aqua diurnal bridge is missing, pre-July 2002 data fail level shift and are strictly blocked from serving.
              </li>
              <li>
                <strong>Forest Mega-Fire Extrapolation Risk:</strong> Model B calibration was evaluated primarily on typical fire weeks ($M \le 5$). Weeks with extreme burning ($M &gt; 10$) in evergreen Western Ghats forest regimes carry known extrapolation uncertainty.
              </li>
              <li>
                <strong>Undercoverage of Bootstrap Uncertainty Intervals:</strong> Empirical validation at Gate B demonstrated empirical 95% bootstrap intervals cover ~84.2% of real overlap observations due to heavy-tailed active-fire distributions.
              </li>
              <li>
                <strong>Unacquired MCD64A1 Burned Area:</strong> NASA Earthdata OAuth blocks prevented automated HDF granule acquisition. Missing burned-area values are explicitly preserved as <code>null</code> with <code>missing_unacquired</code>; zero values are never assumed.
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
