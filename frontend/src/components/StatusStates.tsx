import React from 'react';
import { Loader2, AlertCircle, Inbox, RefreshCw } from 'lucide-react';

interface LoadingStateProps {
  message?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading scientific data and harmonisation parameters...',
}) => {
  return (
    <div className="status-container loading-container" role="status" aria-live="polite">
      <Loader2 className="spinner-icon" size={32} />
      <p className="status-message">{message}</p>
    </div>
  );
};

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  status?: number;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Data Retrieval Error',
  message,
  onRetry,
  status,
}) => {
  return (
    <div className="status-container error-container" role="alert">
      <AlertCircle className="error-icon" size={36} />
      <div className="error-content">
        <h3 className="error-title">
          {title} {status ? `(HTTP ${status})` : ''}
        </h3>
        <p className="error-message">{message}</p>
        {onRetry && (
          <button type="button" className="retry-button" onClick={onRetry}>
            <RefreshCw size={16} />
            <span>Retry Query</span>
          </button>
        )}
      </div>
    </div>
  );
};

interface EmptyStateProps {
  title?: string;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No Data Found',
  message,
  actionLabel,
  onAction,
}) => {
  return (
    <div className="status-container empty-container">
      <Inbox className="empty-icon" size={36} />
      <div className="empty-content">
        <h3 className="empty-title">{title}</h3>
        <p className="empty-message">{message}</p>
        {actionLabel && onAction && (
          <button type="button" className="empty-action-button" onClick={onAction}>
            {actionLabel}
          </button>
        )}
      </div>
    </div>
  );
};
