import React from 'react';
import { useApp } from './context/AppContext';
import { Header } from './components/Header';
import { Navigation } from './components/Navigation';
import { FilterPanel } from './components/FilterPanel';
import { ExportBar } from './components/ExportBar';
import { OverviewView } from './components/OverviewView';
import { MapView } from './components/MapView';
import { HeatmapView } from './components/HeatmapView';
import { SeriesView } from './components/SeriesView';
import { SeasonalCompareView } from './components/SeasonalCompareView';
import { HotspotView } from './components/HotspotView';
import { AnomalyView } from './components/AnomalyView';
import { AttributionView } from './components/AttributionView';
import { VisualisationView } from './components/VisualisationView';
import { TransparencyView } from './components/TransparencyView';
import './App.css';

export const AppContent: React.FC = () => {
  const { state } = useApp();

  const renderActiveView = () => {
    switch (state.view) {
      case 'overview':
        return <OverviewView />;
      case 'map':
        return <MapView />;
      case 'heatmap':
        return <HeatmapView />;
      case 'series':
        return <SeriesView />;
      case 'compare':
        return <SeasonalCompareView />;
      case 'hotspots':
        return <HotspotView />;
      case 'anomalies':
        return <AnomalyView />;
      case 'attribution':
        return <AttributionView />;
      case 'visualisation':
        return <VisualisationView />;
      case 'transparency':
        return <TransparencyView />;
      default:
        return <OverviewView />;
    }
  };

  return (
    <div className="app-shell">
      <Header />
      <div className="app-body">
        <Navigation />
        <main className="main-content" role="region" aria-label="Main Scientific Content Area">
          <FilterPanel />
          <ExportBar />
          <div className="active-view-wrapper">{renderActiveView()}</div>
        </main>
      </div>
    </div>
  );
};
