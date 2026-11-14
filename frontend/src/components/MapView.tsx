import React, { useEffect, useState, useMemo, useRef, useCallback } from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import type { DailyGridPoint, HotspotRecord } from '../types';
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Download, Layers, Pencil, Trash2, Compass, FileSpreadsheet, ZoomIn } from 'lucide-react';
import { toPng } from 'html-to-image';
import { isPointInPolygon, calculatePolygonAreaKm2 } from '../utils/spatial';

// ─────────────────────────────────────────────────────────────────────────────
// Map Inset Auto-Focuser
// ─────────────────────────────────────────────────────────────────────────────
const InsetMapController: React.FC<{ bounds: L.LatLngBoundsExpression | null }> = ({ bounds }) => {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [15, 15], maxZoom: 14 });
    }
  }, [bounds, map]);
  return null;
};

// ─────────────────────────────────────────────────────────────────────────────
// Polygon draw controller — renders inside MapContainer via useMap()
// ─────────────────────────────────────────────────────────────────────────────
interface DrawControllerProps {
  drawMode: boolean;
  onPolygonComplete: (latlngs: [number, number][]) => void;
}

const DrawController: React.FC<DrawControllerProps> = ({ drawMode, onPolygonComplete }) => {
  const map = useMap();
  const pointsRef = useRef<[number, number][]>([]);
  const polylineRef = useRef<L.Polyline | null>(null);
  const markerGroupRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!drawMode) {
      polylineRef.current?.remove();
      markerGroupRef.current?.clearLayers();
      pointsRef.current = [];
      return;
    }

    map.getContainer().style.cursor = 'crosshair';

    if (!markerGroupRef.current) {
      markerGroupRef.current = L.layerGroup().addTo(map);
    }

    const handleClick = (e: L.LeafletMouseEvent) => {
      const pt: [number, number] = [e.latlng.lat, e.latlng.lng];
      pointsRef.current = [...pointsRef.current, pt];

      const dot = L.circleMarker(e.latlng, {
        radius: 5,
        color: '#facc15',
        fillColor: '#facc15',
        fillOpacity: 1,
        weight: 2,
      });
      markerGroupRef.current?.addLayer(dot);

      polylineRef.current?.remove();
      if (pointsRef.current.length > 1) {
        polylineRef.current = L.polyline(
          pointsRef.current.map(([lat, lng]) => L.latLng(lat, lng)),
          { color: '#facc15', weight: 2, dashArray: '6, 4' }
        ).addTo(map);
      }
    };

    const handleDblClick = (e: L.LeafletMouseEvent) => {
      L.DomEvent.stopPropagation(e);
      const pts = [...pointsRef.current];
      if (pts.length >= 3) {
        onPolygonComplete(pts);
      }
      polylineRef.current?.remove();
      markerGroupRef.current?.clearLayers();
      pointsRef.current = [];
      map.getContainer().style.cursor = '';
    };

    map.on('click', handleClick);
    map.on('dblclick', handleDblClick);

    return () => {
      map.off('click', handleClick);
      map.off('dblclick', handleDblClick);
      map.getContainer().style.cursor = '';
      polylineRef.current?.remove();
      markerGroupRef.current?.clearLayers();
      pointsRef.current = [];
    };
  }, [drawMode, map, onPolygonComplete]);

  return null;
};

// ─────────────────────────────────────────────────────────────────────────────
// Drawn polygon renderer
// ─────────────────────────────────────────────────────────────────────────────
interface DrawnPolygonLayerProps {
  polygon: [number, number][] | null;
}

const DrawnPolygonLayer: React.FC<DrawnPolygonLayerProps> = ({ polygon }) => {
  const map = useMap();
  const layerRef = useRef<L.Polygon | null>(null);

  useEffect(() => {
    layerRef.current?.remove();
    if (!polygon || polygon.length < 3) return;

    layerRef.current = L.polygon(polygon.map(([lat, lng]) => L.latLng(lat, lng)), {
      color: '#facc15',
      weight: 3,
      fillColor: '#dc2626',
      fillOpacity: 0.25,
      dashArray: '',
    }).addTo(map);

    return () => {
      layerRef.current?.remove();
    };
  }, [polygon, map]);

  return null;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main CONAE-Style MapView
// ─────────────────────────────────────────────────────────────────────────────
export const MapView: React.FC = () => {
  const {
    state,
    setRegion,
    setSelectedHotspotId,
    setView,
    setCustomPolygon,
    clearCustomPolygon,
  } = useApp();

  const [districtsGeo, setDistrictsGeo] = useState<any>(null);
  const [gridGeo, setGridGeo] = useState<any>(null);
  const [dailyData, setDailyData] = useState<DailyGridPoint[]>([]);
  const [hotspots, setHotspots] = useState<HotspotRecord[]>([]);
  const showGrid = true;
  const showDistricts = true;
  const showHotspots = true;
  const [basemap, setBasemap] = useState<'satellite' | 'street'>('satellite');
  const [drawMode, setDrawMode] = useState<boolean>(false);

  const mapExportRef = useRef<HTMLDivElement>(null);

  // Load static GeoJSON boundaries
  useEffect(() => {
    Promise.all([
      fetch('/geo/tamil_nadu_districts.geojson').then((r) => r.json()),
      fetch('/geo/grid_10km.geojson').then((r) => r.json()),
    ])
      .then(([dist, grid]) => {
        setDistrictsGeo(dist);
        setGridGeo(grid);
      })
      .catch((err) => console.error('Error loading GeoJSON:', err));
  }, []);

  // Fetch daily data
  const loadMapData = async () => {
    try {
      const [gridRes, spotsRes] = await Promise.all([
        apiService.getGridDaily({
          date: state.activeAnalysisDate,
          sensor: state.sensor !== 'all' ? state.sensor : undefined,
        }),
        apiService.getHotspots({
          start_date: state.activeAnalysisDate,
          end_date: state.activeAnalysisDate,
          confidence: state.confidence !== 'all' ? state.confidence : undefined,
          limit: 1000,
        }),
      ]);
      setDailyData(gridRes.data || []);
      setHotspots(spotsRes.data?.hotspots || []);
    } catch (err) {
      console.error('Failed to load daily map data:', err);
    }
  };

  useEffect(() => {
    loadMapData();
  }, [state.activeAnalysisDate, state.sensor, state.confidence]);

  const handlePolygonComplete = useCallback(
    (latlngs: [number, number][]) => {
      setCustomPolygon(latlngs);
      setDrawMode(false);
    },
    [setCustomPolygon]
  );

  const handleClearPolygon = () => {
    clearCustomPolygon();
    setDrawMode(false);
  };

  const activePolygon = state.customPolygon;

  // Filter hotspots to polygon
  const hotspotsInPolygon = useMemo(() => {
    if (!activePolygon || activePolygon.length < 3) return hotspots;
    return hotspots.filter((s) => isPointInPolygon(s.latitude, s.longitude, activePolygon));
  }, [hotspots, activePolygon]);

  // Selection stats
  const selectionStats = useMemo(() => {
    if (!activePolygon) return null;
    const pts = hotspotsInPolygon;
    const totalFrp = pts.reduce((acc, s) => acc + (s.frp || 0), 0);
    const viirs = pts.filter((s) => s.instrument === 'VIIRS').length;
    const modis = pts.filter((s) => s.instrument !== 'VIIRS').length;
    const areaKm2 = calculatePolygonAreaKm2(activePolygon);
    const areaHa = areaKm2 * 100;
    return { count: pts.length, totalFrp, viirs, modis, areaKm2, areaHa };
  }, [hotspotsInPolygon, activePolygon]);

  // Inset zoom bounds when polygon exists
  const insetBounds = useMemo(() => {
    if (!activePolygon || activePolygon.length < 3) return null;
    const lats = activePolygon.map((p) => p[0]);
    const lngs = activePolygon.map((p) => p[1]);
    const minLat = Math.min(...lats);
    const maxLat = Math.max(...lats);
    const minLng = Math.min(...lngs);
    const maxLng = Math.max(...lngs);
    return L.latLngBounds([minLat, minLng], [maxLat, maxLng]);
  }, [activePolygon]);

  const cellValueMap = useMemo(() => {
    const map = new Map<string, DailyGridPoint>();
    dailyData.forEach((d) => map.set(d.grid_cell_id, d));
    return map;
  }, [dailyData]);

  const getCellFillColor = (count: number) => {
    if (count <= 0) return 'transparent';
    if (count < 2) return '#fef08a';
    if (count < 5) return '#f97316';
    if (count < 10) return '#dc2626';
    return '#7f1d1d';
  };

  const gridStyle = (feature: any) => {
    const cellId = feature?.properties?.grid_cell_id;
    const isSelected = state.region.type === 'cell' && state.region.id === cellId;
    const point = cellValueMap.get(cellId);
    const count = point ? point.count_value : 0;
    return {
      fillColor: getCellFillColor(count),
      fillOpacity: count > 0 ? 0.75 : 0.05,
      weight: isSelected ? 2.5 : 0.4,
      color: isSelected ? '#38bdf8' : '#94a3b8',
      dashArray: count > 0 ? '' : '2, 2',
    };
  };

  const districtStyle = (feature: any) => {
    const distName = feature?.properties?.district_name;
    const isSelected = state.region.type === 'district' && state.region.name === distName;
    return {
      fillColor: 'transparent',
      weight: isSelected ? 3 : 1.2,
      color: isSelected ? '#facc15' : 'rgba(255,255,255,0.45)',
      opacity: 0.9,
    };
  };

  const onEachDistrict = (feature: any, layer: any) => {
    const distName = feature?.properties?.district_name;
    layer.on({
      click: () => setRegion({ type: 'district', id: distName, name: distName }),
    });
  };

  const onEachGridCell = (feature: any, layer: any) => {
    const cellId = feature?.properties?.grid_cell_id;
    const pt = cellValueMap.get(cellId);
    const count = pt ? pt.count_value : 0;
    const frp = pt ? pt.frp_value : 0;
    layer.bindTooltip(
      `<strong>Cell:</strong> ${cellId}<br/><strong>Fires:</strong> ${count.toFixed(1)}<br/><strong>FRP:</strong> ${frp.toFixed(1)} MW<br/><em>Click to inspect</em>`,
      { sticky: true }
    );
    layer.on({ click: () => setRegion({ type: 'cell', id: cellId, name: cellId }) });
  };

  const exportMapPng = async () => {
    if (!mapExportRef.current) return;
    try {
      const dataUrl = await toPng(mapExportRef.current, { cacheBust: true });
      const link = document.createElement('a');
      link.download = `nebulens_satellite_map_${state.activeAnalysisDate}.png`;
      link.href = dataUrl;
      link.click();
    } catch (e) {
      console.error('PNG export failed:', e);
    }
  };

  const downloadPolygonHotspotsCsv = () => {
    if (hotspotsInPolygon.length === 0) return;
    const headers = ['observation_id', 'latitude', 'longitude', 'acq_datetime_utc', 'instrument', 'satellite', 'confidence', 'confidence_class', 'frp'];
    const rows = hotspotsInPolygon.map(h => [
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
    link.setAttribute('download', `nebulens_selected_area_hotspots_${state.activeAnalysisDate}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const displayedHotspots = activePolygon ? hotspotsInPolygon : hotspots;

  return (
    <div className="view-content conae-map-container" ref={mapExportRef}>
      {/* Top CONAE Title Banner */}
      <div className="conae-header-bar">
        <div className="conae-title-block">
          <div className="conae-main-title">ACTIVE FIRE & SATELLITE BURNING OBSERVATION</div>
          <div className="conae-sub-title">
            STATE OF TAMIL NADU, INDIA — {state.region.name.toUpperCase()}
          </div>
        </div>
        <div className="conae-meta-block">
          <div className="conae-activation">OPERATIONAL MONITORING</div>
          <div className="conae-date-display">{state.activeAnalysisDate} (IST)</div>
        </div>
      </div>

      {/* Main Interactive Stage */}
      <div className="conae-stage-layout">
        {/* Central Map Canvas */}
        <div className="conae-map-frame">
          {/* North Arrow Floating Badge */}
          <div className="conae-compass-badge">
            <Compass size={24} color="#f8fafc" />
            <span>N</span>
          </div>

          {/* Quick Toolbar */}
          <div className="conae-quick-controls">
            <button
              type="button"
              className={`conae-tool-btn ${basemap === 'satellite' ? 'active' : ''}`}
              onClick={() => setBasemap(basemap === 'satellite' ? 'street' : 'satellite')}
              title="Toggle Basemap"
            >
              <Layers size={14} />
              <span>{basemap === 'satellite' ? 'Satellite Imagery' : 'Street Map'}</span>
            </button>

            <button
              type="button"
              className={`conae-tool-btn ${drawMode ? 'drawing' : ''}`}
              onClick={() => {
                if (drawMode) {
                  setDrawMode(false);
                } else {
                  handleClearPolygon();
                  setDrawMode(true);
                }
              }}
              title="Draw Polygon Area"
            >
              <Pencil size={14} />
              <span>{drawMode ? 'Drawing (Double-Click to Close)' : 'Draw Area of Interest'}</span>
            </button>

            {activePolygon && (
              <button
                type="button"
                className="conae-tool-btn danger"
                onClick={handleClearPolygon}
                title="Clear Area"
              >
                <Trash2 size={14} />
                <span>Reset Area</span>
              </button>
            )}

            <button
              type="button"
              className="conae-tool-btn"
              onClick={exportMapPng}
              title="Export High-Res PNG"
            >
              <Download size={14} />
              <span>Snapshot</span>
            </button>
          </div>

          {/* Dynamic "Potential Affected Area" Banner */}
          {activePolygon && selectionStats && (
            <div className="conae-burned-area-tag">
              <span className="conae-tag-arrow">&gt;</span>
              <span>
                <strong>Potential Affected Area:</strong> {selectionStats.areaHa.toFixed(0)} Ha ({selectionStats.areaKm2.toFixed(1)} km²)
              </span>
            </div>
          )}

          {/* Main Leaflet Container */}
          <MapContainer
            center={[11.1271, 78.6569]}
            zoom={7}
            style={{ height: '700px', width: '100%', background: '#020617' }}
          >
            {basemap === 'satellite' ? (
              <TileLayer
                key="esri-sat"
                attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              />
            ) : (
              <TileLayer
                key="osm-st"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
            )}

            <DrawController drawMode={drawMode} onPolygonComplete={handlePolygonComplete} />
            <DrawnPolygonLayer polygon={activePolygon || null} />

            {showGrid && gridGeo && (
              <GeoJSON
                key={`grid-${state.activeAnalysisDate}-${dailyData.length}`}
                data={gridGeo}
                style={gridStyle}
                onEachFeature={onEachGridCell}
              />
            )}

            {showDistricts && districtsGeo && (
              <GeoJSON
                key={`districts-${state.region.id}`}
                data={districtsGeo}
                style={districtStyle}
                onEachFeature={onEachDistrict}
              />
            )}

            {showHotspots &&
              displayedHotspots.map((spot) => (
                <CircleMarker
                  key={spot.observation_id}
                  center={[spot.latitude, spot.longitude]}
                  radius={Math.min(10, Math.max(3.5, Math.sqrt(spot.frp || 1) * 1.5))}
                  pathOptions={{
                    fillColor: spot.instrument === 'VIIRS' ? '#38bdf8' : '#f59e0b',
                    color: activePolygon ? '#facc15' : '#ffffff',
                    weight: activePolygon ? 1.8 : 1,
                    fillOpacity: 0.9,
                  }}
                  eventHandlers={{
                    click: () => {
                      setSelectedHotspotId(spot.observation_id);
                      setView('hotspots');
                    },
                  }}
                >
                  <Tooltip>
                    <div>
                      <strong>{spot.instrument} Hotspot</strong> ({spot.source})<br />
                      <strong>FRP:</strong> {spot.frp} MW<br />
                      <strong>Confidence:</strong> {spot.confidence} ({spot.confidence_class})<br />
                      <strong>Acquisition:</strong> {spot.acq_datetime_utc}<br />
                      <em>Click to inspect details</em>
                    </div>
                  </Tooltip>
                </CircleMarker>
              ))}
          </MapContainer>

          {/* Mini Detail Zoom Inset */}
          {activePolygon && insetBounds && (
            <div className="conae-inset-frame">
              <div className="conae-inset-header">
                <ZoomIn size={13} />
                <span>AFFECTED AREA DETAIL (INSET)</span>
              </div>
              <MapContainer
                bounds={insetBounds}
                zoomControl={false}
                attributionControl={false}
                style={{ height: '180px', width: '230px', background: '#000' }}
              >
                <TileLayer url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" />
                <InsetMapController bounds={insetBounds} />
                <DrawnPolygonLayer polygon={activePolygon} />
                {hotspotsInPolygon.map((spot) => (
                  <CircleMarker
                    key={`inset-${spot.observation_id}`}
                    center={[spot.latitude, spot.longitude]}
                    radius={4}
                    pathOptions={{
                      fillColor: '#facc15',
                      color: '#ef4444',
                      weight: 1.5,
                      fillOpacity: 1,
                    }}
                  />
                ))}
              </MapContainer>
            </div>
          )}
        </div>

        {/* Right-Hand Side Cartographic Legend & Metadata Panel */}
        <div className="conae-carto-panel">
          {/* Section 1: Overview Locator */}
          <div className="conae-panel-card">
            <div className="conae-card-title">LOCATION REFERENCE</div>
            <div className="conae-overview-box">
              <div className="conae-locator-badge">PROVINCE: TAMIL NADU</div>
              <div className="conae-locator-badge">CRS: WGS84 / EPSG:4326</div>
              <div className="conae-locator-badge">ANALYSIS GRID: EPSG:6933 10km</div>
            </div>
          </div>

          {/* Section 2: Symbology / Legend */}
          <div className="conae-panel-card">
            <div className="conae-card-title">LEGEND / SYMBOLOGY</div>
            <div className="conae-legend-list">
              <div className="conae-legend-row">
                <span className="dot viirs-dot"></span>
                <span>VIIRS 375m Hotspots (SNPP / NOAA)</span>
              </div>
              <div className="conae-legend-row">
                <span className="dot modis-dot"></span>
                <span>MODIS 1km Hotspots (Terra / Aqua)</span>
              </div>
              <div className="conae-legend-row">
                <span className="line poly-line"></span>
                <span>Drawn Area of Interest (Perimeter)</span>
              </div>
              <div className="conae-legend-row">
                <span className="line dist-line"></span>
                <span>District Administrative Boundary</span>
              </div>
            </div>
          </div>

          {/* Section 3: Observation Metadata */}
          <div className="conae-panel-card">
            <div className="conae-card-title">OBSERVATION METADATA</div>
            <div className="conae-text-meta">
              The map displays satellite active fire hotspots detected on{' '}
              <strong>{state.activeAnalysisDate}</strong> over high-resolution composite imagery.
              Active markers denote calibrated orbital infrared thermal anomalies.
            </div>
            {selectionStats ? (
              <div className="conae-stats-highlight">
                <div className="stat-line">
                  <span>Estimated Area:</span>
                  <strong>{selectionStats.areaHa.toFixed(0)} Ha</strong>
                </div>
                <div className="stat-line">
                  <span>Hotspot Detections:</span>
                  <strong style={{ color: '#facc15' }}>{selectionStats.count} fires</strong>
                </div>
                <div className="stat-line">
                  <span>Fire Radiative Power:</span>
                  <strong>{selectionStats.totalFrp.toFixed(1)} MW</strong>
                </div>
                <button
                  type="button"
                  className="btn-toolbar btn-active"
                  style={{ width: '100%', marginTop: '0.6rem', justifyContent: 'center' }}
                  onClick={downloadPolygonHotspotsCsv}
                >
                  <FileSpreadsheet size={14} />
                  <span>Download Hotspots CSV</span>
                </button>
              </div>
            ) : (
              <div className="conae-hint-box">
                Use the <strong>"Draw Area of Interest"</strong> tool to outline a perimeter over fire clusters to calculate affected area and detections.
              </div>
            )}
          </div>

          {/* Section 4: Data Sources */}
          <div className="conae-panel-card">
            <div className="conae-card-title">DATA SOURCES</div>
            <div className="conae-sensor-specs">
              <div><strong>Satellites:</strong> TERRA, AQUA, SNPP, NOAA-20, NOAA-21</div>
              <div><strong>Sensors:</strong> MODIS, VIIRS</div>
              <div><strong>Resolution:</strong> 375 m / 1,000 m</div>
              <div><strong>Frequency:</strong> 14 to 42 passes / week</div>
              <div><strong>Platform:</strong> NASA FIRMS / NEBULENS Core</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
