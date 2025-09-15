// Dashboard Components
export { DashboardLayout } from './DashboardLayout';
export { AlertManager } from './AlertManager';
export { UserPreferencesComponent } from './UserPreferences';
export { default as ComprehensiveCausalDashboard } from './ComprehensiveCausalDashboard';

// Re-export dashboard types
export type {
  DashboardConfig,
  DashboardWidget,
  WidgetConfig,
  WidgetType,
  UserPreferences,
  DashboardAlert,
  AlertLevel,
  AlertType,
  NotificationSettings,
  AccessibilitySettings,
  PerformanceSettings
} from '../../types/dashboard';