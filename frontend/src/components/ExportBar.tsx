import React from 'react';
import { useApp } from '../context/AppContext';
import { apiService } from '../services/api';
import { Download, FileSpreadsheet, FileText } from 'lucide-react';
import { calculatePolygonAreaKm2 } from '../utils/spatial';

export const ExportBar: React.FC = () => {
  const { state } = useApp();

  const handleCsvDownload = (queryType: 'series' | 'anomalies' | 'attribution' | 'season_metrics') => {
    const url = apiService.getExportUrl({
      query_type: queryType,
      start_date: state.dateRange.startDate,
      end_date: state.dateRange.endDate,
      district_name: state.region.type === 'district' ? state.region.name : undefined,
    });
    window.open(url, '_blank');
  };

  const handleGenerateSummaryReport = () => {
    const isPolygon = state.region.type === 'polygon' && !!state.customPolygon;
    const polyArea = isPolygon && state.customPolygon ? calculatePolygonAreaKm2(state.customPolygon) : 0;
    const reportTitle = `NEBULENS Wildfire Observation & Anomaly Report`;
    const reportDate = new Date().toISOString().replace('T', ' ').slice(0, 19);

    const markdownContent = `# ${reportTitle}
**Generated:** ${reportDate} (UTC)
**Application:** NEBULENS — Wildfire Observation, Harmonisation & Anomaly Analysis Platform
**Geographic Domain:** Tamil Nadu, India

---

## 1. Selected Analysis Scope & Spatial Filtering
- **Selected Region:** ${state.region.name} (${state.region.type.toUpperCase()})
${isPolygon ? `- **Custom Boundary Coordinates:** ${state.customPolygon?.length} vertices defined\n- **Approximate Area Covered:** ${polyArea.toFixed(1)} km²\n- **Spatial Mode:** Custom interactive polygon selection` : `- **Administrative Scope:** ${state.region.type === 'district' ? 'District-level filter' : 'Statewide 38-district aggregate'}`}
- **Active Analysis Date (IST):** ${state.activeAnalysisDate}
- **Global Time Range Filter:** ${state.dateRange.startDate} to ${state.dateRange.endDate}
- **Selected Sensor Stream:** ${state.sensor.toUpperCase()}
- **FIRMS Confidence Filter:** ${state.confidence}
- **Spatial Analysis Grid:** EPSG:6933 Equal-Area 10 km (1,460 cells across Tamil Nadu)

---

## 2. Multi-Decadal Sensor Constellation Context
- **Terra MODIS (10:30 IST):** 2000 – Present (1 km resolution)
- **Aqua MODIS (13:30 IST):** July 2002 – Present (1 km resolution, Diurnal peak fire bridge)
- **Suomi-NPP VIIRS:** 2012 – Present (375 m resolution, 14 passes/week baseline)
- **NOAA-20 VIIRS:** 2018 – Present (Expansion to 28 passes/week)
- **NOAA-21 VIIRS:** 2023 – Present (Expansion to 42 passes/week)

*Harmonisation Methodology Note:* When comparing post-2018 numbers with historical records, exposure standardization accounts for the 3x increase in orbital revisit frequency.

---

## 3. Authoritative Scientific Interpretation Framework
1. **Remote Sensing Detections:** Hotspot coordinates represent orbital sensor radiance anomalies (elevated brightness temperatures), not independently ground-verified emergency responses.
2. **Negative Binomial Hurdle Counterfactuals:** Pre-2012 harmonised counts are estimated via Hurdle GLM models calibrated on 2012–2021 co-observation periods.
3. **Attribution Categorisation:** High detection peaks undergo multi-evidence testing to distinguish true climate-driven biomass fires from sensor opportunity expansions.

---
*NEBULENS Platform — NASA Space Apps Challenge Prototype*
`;

    const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `nebulens_analysis_report_${state.region.id}_${state.activeAnalysisDate}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const isPolygon = state.region.type === 'polygon';

  return (
    <div className="app-export-bar">
      <div className="export-bar-info">
        <FileSpreadsheet size={18} className="text-primary" />
        <span>
          <strong>Data Export & Analysis Reports:</strong> Filtered to{' '}
          <span style={{ color: isPolygon ? '#facc15' : '#38bdf8', fontWeight: 600 }}>
            {state.region.name}
          </span>
          . Download calibrated datasets and comprehensive analysis summaries.
        </span>
      </div>

      <div className="export-buttons">
        <button
          type="button"
          className="btn-export btn-active"
          onClick={handleGenerateSummaryReport}
          title="Download Comprehensive Analysis Summary Report"
        >
          <FileText size={14} />
          <span>Analysis Report (.md)</span>
        </button>

        <button
          type="button"
          className="btn-export"
          onClick={() => handleCsvDownload('series')}
          title="Export Time Series CSV"
        >
          <Download size={14} />
          <span>Series CSV</span>
        </button>

        <button
          type="button"
          className="btn-export"
          onClick={() => handleCsvDownload('anomalies')}
          title="Export Anomaly Records CSV"
        >
          <Download size={14} />
          <span>Anomalies CSV</span>
        </button>

        <button
          type="button"
          className="btn-export"
          onClick={() => handleCsvDownload('attribution')}
          title="Export Multi-Evidence Attribution CSV"
        >
          <Download size={14} />
          <span>Attribution CSV</span>
        </button>

        <button
          type="button"
          className="btn-export"
          onClick={() => handleCsvDownload('season_metrics')}
          title="Export Seasonal Baseline Metrics CSV"
        >
          <Download size={14} />
          <span>Seasonal Metrics CSV</span>
        </button>
      </div>
    </div>
  );
};
