import React, { useEffect, useState } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { HotspotRecord, HotspotsPaginated } from '../types';
import { ListFilter, MapPin, Eye, Flame, Info, ChevronLeft, ChevronRight, Download } from 'lucide-react';
import { isPointInPolygon } from '../utils/spatial';

export const HotspotView: React.FC = () => {
  const { state, setSelectedHotspotId, setView } = useApp();

  const [paginatedData, setPaginatedData] = useState<HotspotsPaginated | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<HotspotRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [page, setPage] = useState<number>(0);
  const pageSize = 15;

  const isPolygonActive = state.region.type === 'polygon' && !!state.customPolygon;

  const loadHotspots = async (offsetVal: number) => {
    setLoading(true);
    try {
      // If custom polygon is active, fetch broader list and filter client-side
      const limitToFetch = isPolygonActive ? 1000 : pageSize;
      const res = await apiService.getHotspots({
        start_date: state.activeAnalysisDate,
        end_date: state.activeAnalysisDate,
        district_name: state.region.type === 'district' ? state.region.name : undefined,
        grid_cell_id: state.region.type === 'cell' ? state.region.id : undefined,
        confidence: state.confidence !== 'all' ? state.confidence : undefined,
        limit: limitToFetch,
        offset: isPolygonActive ? 0 : offsetVal,
      });

      if (isPolygonActive && state.customPolygon) {
        const poly = state.customPolygon;
        const filtered = (res.data?.hotspots || []).filter(h => isPointInPolygon(h.latitude, h.longitude, poly));
        setPaginatedData({
          total_count: filtered.length,
          limit: pageSize,
          offset: offsetVal,
          hotspots: filtered.slice(offsetVal, offsetVal + pageSize),
        });
      } else {
        setPaginatedData(res.data);
      }
    } catch (e) {
      console.error('Failed to load hotspots list:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHotspots(page * pageSize);
  }, [state.activeAnalysisDate, state.region, state.confidence, state.customPolygon, page]);

  useEffect(() => {
    if (state.selectedHotspotId) {
      setDetailLoading(true);
      apiService
        .getHotspotDetail(state.selectedHotspotId)
        .then((res) => setSelectedDetail(res.data))
        .catch((e) => console.error('Failed to fetch hotspot detail:', e))
        .finally(() => setDetailLoading(false));
    }
  }, [state.selectedHotspotId]);

  const handleSelect = (spot: HotspotRecord) => {
    setSelectedHotspotId(spot.observation_id);
    setSelectedDetail(spot);
  };

  // CSV download for current hotspot view
  const downloadHotspotsCsv = () => {
    const list = paginatedData?.hotspots || [];
    if (list.length === 0) return;
    const headers = ['observation_id', 'latitude', 'longitude', 'acq_datetime_utc', 'instrument', 'satellite', 'confidence', 'confidence_class', 'frp'];
    const rows = list.map(h => [
      h.observation_id,
      h.latitude,
      h.longitude,
      h.acq_datetime_utc,
      h.instrument,
      h.satellite,
      h.confidence,
      h.confidence_class,
      h.frp || 0
    ]);
    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `nebulens_hotspots_${state.region.id}_${state.activeAnalysisDate}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="view-content hotspots-container">
      <div className="section-header">
        <h2>Satellite Hotspot Granule Inspector</h2>
        <p className="section-description">
          Active satellite fire detections for <strong>{state.activeAnalysisDate}</strong> | Region:{' '}
          <strong style={{ color: isPolygonActive ? '#facc15' : 'inherit' }}>
            {state.region.name}
          </strong>.
          {isPolygonActive && ' (Filtered strictly to custom drawn polygon area)'}
        </p>
      </div>

      <div className="hotspots-layout">
        {/* List Table */}
        <div className="card hotspots-list-card">
          <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ListFilter size={18} />
              <h3>Hotspots ({paginatedData?.total_count || 0} Detections)</h3>
            </div>
            {(paginatedData?.total_count || 0) > 0 && (
              <button
                type="button"
                className="btn-toolbar"
                onClick={downloadHotspotsCsv}
                title="Download this hotspot table as CSV"
              >
                <Download size={14} />
                <span>Export CSV</span>
              </button>
            )}
          </div>

          <div className="card-body">
            {loading ? (
              <div className="loading-placeholder">Loading satellite hotspot granules...</div>
            ) : paginatedData?.hotspots.length === 0 ? (
              <div className="empty-chart">No satellite hotspots detected for this day in the selected region.</div>
            ) : (
              <>
                <div className="table-wrapper">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Observation ID</th>
                        <th>Sensor</th>
                        <th>Lat, Lon</th>
                        <th>FRP (MW)</th>
                        <th>Confidence</th>
                        <th>Day/Night</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedData?.hotspots.map((spot) => {
                        const isSelected = spot.observation_id === selectedDetail?.observation_id;
                        return (
                          <tr
                            key={spot.observation_id}
                            className={isSelected ? 'selected-row' : ''}
                            onClick={() => handleSelect(spot)}
                          >
                            <td className="code-text font-bold">{spot.observation_id}</td>
                            <td>
                              <span className={`badge ${spot.instrument === 'VIIRS' ? 'badge-primary' : 'badge-warning'}`}>
                                {spot.instrument}
                              </span>
                            </td>
                            <td>{spot.latitude.toFixed(4)}, {spot.longitude.toFixed(4)}</td>
                            <td className="font-bold">{spot.frp.toFixed(2)}</td>
                            <td>
                              <span className="badge badge-outline">
                                {spot.confidence} ({spot.confidence_class})
                              </span>
                            </td>
                            <td>{spot.daynight === 'D' ? 'Day' : 'Night'}</td>
                            <td>
                              <button
                                type="button"
                                className="btn-table-action"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSelect(spot);
                                }}
                              >
                                <Eye size={12} /> Inspect
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div className="pagination-bar">
                  <span className="pagination-info">
                    Showing page {page + 1} of {Math.ceil((paginatedData?.total_count || 1) / pageSize)}
                  </span>
                  <div className="pagination-buttons">
                    <button
                      type="button"
                      className="btn-page"
                      disabled={page === 0}
                      onClick={() => setPage((p) => Math.max(0, p - 1))}
                    >
                      <ChevronLeft size={16} /> Prev
                    </button>
                    <button
                      type="button"
                      className="btn-page"
                      disabled={(page + 1) * pageSize >= (paginatedData?.total_count || 0)}
                      onClick={() => setPage((p) => p + 1)}
                    >
                      Next <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Hotspot Granule Detail Inspection Card */}
        <div className="card hotspot-detail-card">
          <div className="card-header">
            <Flame size={18} className="text-warning" />
            <h3>Granule Provenance & Detail</h3>
          </div>
          <div className="card-body">
            {detailLoading ? (
              <div className="loading-placeholder">Fetching granule metadata...</div>
            ) : selectedDetail ? (
              <div className="detail-content">
                <div className="detail-badge-group">
                  <span className="badge badge-primary">{selectedDetail.instrument}</span>
                  <span className="badge badge-outline">{selectedDetail.satellite}</span>
                  <span className="badge badge-success">{selectedDetail.confidence_class}</span>
                </div>

                <div className="detail-meta-list">
                  <div className="detail-row">
                    <span className="label">Observation ID:</span>
                    <span className="value code-text">{selectedDetail.observation_id}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Location (Lat, Lon):</span>
                    <span className="value">{selectedDetail.latitude.toFixed(5)}, {selectedDetail.longitude.toFixed(5)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Acquisition UTC:</span>
                    <span className="value">{selectedDetail.acq_datetime_utc}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Overpass Orbit:</span>
                    <span className="value">{selectedDetail.daynight === 'D' ? 'Daytime Overpass' : 'Nighttime Overpass'}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Fire Radiative Power:</span>
                    <span className="value font-bold text-warning">{selectedDetail.frp.toFixed(2)} MW</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">10km Grid Cell:</span>
                    <span className="value code-text">{selectedDetail.grid_cell_id_10km || '—'}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Landcover Class:</span>
                    <span className="value">{selectedDetail.landcover_class || '—'}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">District Name:</span>
                    <span className="value font-bold">{selectedDetail.district_name || '—'}</span>
                  </div>
                </div>

                <div className="detail-actions">
                  <button
                    type="button"
                    className="btn-toolbar btn-active"
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={() => setView('map')}
                  >
                    <MapPin size={14} />
                    <span>View on Geospatial Map</span>
                  </button>
                </div>

                <div className="detail-caveat">
                  <Info size={12} />
                  <span>
                    Observation represents an elevated brightness temperature detection from orbit. It does not constitute an in-situ confirmed fire report.
                  </span>
                </div>
              </div>
            ) : (
              <div className="empty-placeholder">Select a hotspot row to inspect its orbital geometry and provenance metadata.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
