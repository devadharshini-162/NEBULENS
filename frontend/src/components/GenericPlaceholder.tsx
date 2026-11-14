import React from 'react';
import { useApp } from '../context/AppContext';
import { EmptyState } from './StatusStates';
import { Layers, HelpCircle } from 'lucide-react';

interface GenericPlaceholderProps {
  viewName: string;
  description: string;
  milestoneScheduled: string;
}

export const GenericPlaceholder: React.FC<GenericPlaceholderProps> = ({
  viewName,
  description,
  milestoneScheduled,
}) => {
  const { state, setView } = useApp();

  return (
    <div className="view-content placeholder-view">
      <div className="section-header">
        <h2>{viewName}</h2>
        <p className="section-description">{description}</p>
      </div>

      <div className="card placeholder-card">
        <EmptyState
          title={`${viewName} (Scheduled for ${milestoneScheduled})`}
          message={`This interface component is part of ${milestoneScheduled} and has been intentionally omitted in M8.1 per strict milestone boundaries. The backend endpoints and typed client service are fully prepared.`}
          actionLabel="Return to System Overview"
          onAction={() => setView('overview')}
        />
        <div className="placeholder-provenance-info">
          <div className="info-title">
            <Layers size={16} />
            <span>Active Filter State & Context</span>
          </div>
          <div className="info-grid">
            <div><strong>Sensor Stream:</strong> {state.sensor}</div>
            <div><strong>Selected Region:</strong> {state.region.name} ({state.region.type})</div>
            <div><strong>Date Window:</strong> {state.dateRange.startDate} to {state.dateRange.endDate}</div>
            <div><strong>Product Tier:</strong> {state.product}</div>
          </div>
          <div className="caveat-banner">
            <HelpCircle size={16} />
            <span>
              Pre-2012 harmonised queries are subjected to Gate B conditions: typical fire weeks (M ≤ 5) only;
              extreme forest fire extrapolation restricted; pre-July 2002 Terra-only data blocked.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
