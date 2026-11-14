import React from 'react';
import { useApp } from '../context/AppContext';
import type { ActiveView } from '../types';
import {
  LayoutDashboard,
  Map,
  CalendarDays,
  LineChart,
  GitCompare,
  ListFilter,
  AlertTriangle,
  Crosshair,
  BarChart3,
} from 'lucide-react';

interface NavItem {
  id: ActiveView;
  label: string;
  icon: React.ReactNode;
  badge?: string;
}

export const Navigation: React.FC = () => {
  const { state, setView } = useApp();

  const navItems: NavItem[] = [
    { id: 'map', label: 'Interactive Satellite Map', icon: <Map size={18} /> },
    { id: 'heatmap', label: 'Burning Activity Calendar', icon: <CalendarDays size={18} />, badge: 'CALENDAR' },
    { id: 'hotspots', label: 'Hotspot Inspector', icon: <ListFilter size={18} /> },
    { id: 'series', label: 'Harmonised Time Series', icon: <LineChart size={18} /> },
    { id: 'compare', label: 'Seasonal Comparison', icon: <GitCompare size={18} /> },
    { id: 'anomalies', label: 'Anomaly Engine', icon: <AlertTriangle size={18} /> },
    { id: 'attribution', label: 'Multi-Evidence Attribution', icon: <Crosshair size={18} /> },
    { id: 'visualisation', label: 'Data Visualisation', icon: <BarChart3 size={18} /> },
    { id: 'overview', label: 'Platform Summary', icon: <LayoutDashboard size={18} /> },
  ];

  return (
    <nav className="app-nav" aria-label="Main Scientific Navigation">
      <div className="nav-header">EXPLORER MODULES</div>
      <ul className="nav-list">
        {navItems.map((item) => {
          const isActive = state.view === item.id;
          return (
            <li key={item.id} className="nav-item">
              <button
                type="button"
                className={`nav-button ${isActive ? 'active' : ''}`}
                onClick={() => setView(item.id)}
                aria-current={isActive ? 'page' : undefined}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
                {item.badge && <span className="nav-badge">{item.badge}</span>}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
};
