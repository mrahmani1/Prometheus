/**
 * Dashboard and widget type definitions for the Prometheus AI Platform
 */

import { Layout } from 'react-grid-layout';

// Core dashboard types
export interface DashboardConfig {
  id: string;
  name: string;
  description?: string;
  layout: Record<string, Layout[]>;
  widgets: string[];
  theme: 'light' | 'dark' | 'auto';
  autoRefresh: boolean;
  refreshInterval: number;
  permissions: DashboardPermissions;
  createdAt?: Date;
  lastModified?: Date;
  owner?: string;
  shared?: boolean;
  tags?: string[];
}

export interface DashboardPermissions {
  canEdit: boolean;
  canAdd: boolean;
  canDelete: boolean;
  canExport: boolean;
  canShare?: boolean;
  canDuplicate?: boolean;
}

// Widget configuration types
export interface DashboardWidget {
  id: string;
  type: WidgetType;
  title: string;
  config: WidgetConfig;
  position: WidgetPosition;
  permissions: WidgetPermissions;
  dataSource?: string;
  filters?: WidgetFilter[];
  customProperties?: Record<string, any>;
}

export type WidgetType = 
  | 'seismic-monitor'
  | 'volcanic-status'
  | 'geological-map'
  | 'data-chart'
  | 'alert-panel'
  | 'data-grid'
  | 'statistics'
  | '3d-visualization'
  | 'real-time-stream'
  | 'export-panel'
  | 'custom';

export interface WidgetConfig {
  refreshInterval: number;
  showHeader: boolean;
  showBorder: boolean;
  backgroundColor: string;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  maxW?: number;
  maxH?: number;
  static?: boolean;
  isDraggable?: boolean;
  isResizable?: boolean;
  customSettings?: Record<string, any>;
}

export interface WidgetPosition {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WidgetPermissions {
  canEdit: boolean;
  canDelete: boolean;
  canMove: boolean;
  canResize: boolean;
  canExport?: boolean;
  canShare?: boolean;
}

export interface WidgetFilter {
  field: string;
  operator: FilterOperator;
  value: any;
  enabled: boolean;
}

export type FilterOperator = 
  | 'equals'
  | 'not_equals'
  | 'greater_than'
  | 'less_than'
  | 'greater_equal'
  | 'less_equal'
  | 'contains'
  | 'not_contains'
  | 'starts_with'
  | 'ends_with'
  | 'in'
  | 'not_in'
  | 'between'
  | 'is_null'
  | 'is_not_null';

// User preferences and settings
export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  dateFormat: string;
  numberFormat: string;
  defaultDashboard?: string;
  autoSave: boolean;
  notifications: NotificationSettings;
  accessibility: AccessibilitySettings;
  performance: PerformanceSettings;
}

export interface NotificationSettings {
  enabled: boolean;
  sound: boolean;
  desktop: boolean;
  email: boolean;
  alertLevels: AlertLevel[];
  quietHours?: {
    enabled: boolean;
    start: string;
    end: string;
  };
}

export type AlertLevel = 'info' | 'warning' | 'error' | 'critical';

export interface AccessibilitySettings {
  highContrast: boolean;
  largeText: boolean;
  reducedMotion: boolean;
  screenReader: boolean;
  keyboardNavigation: boolean;
}

export interface PerformanceSettings {
  enableVirtualization: boolean;
  maxDataPoints: number;
  renderQuality: 'low' | 'medium' | 'high';
  enableWebGL: boolean;
  enableWorkers: boolean;
  cacheSize: number;
}

// Alert and notification types
export interface DashboardAlert {
  id: string;
  type: AlertType;
  level: AlertLevel;
  title: string;
  message: string;
  timestamp: Date;
  source: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: Date;
  resolved: boolean;
  resolvedBy?: string;
  resolvedAt?: Date;
  metadata?: Record<string, any>;
  actions?: AlertAction[];
}

export type AlertType = 
  | 'seismic_event'
  | 'volcanic_activity'
  | 'system_error'
  | 'data_quality'
  | 'connection_lost'
  | 'threshold_exceeded'
  | 'model_update'
  | 'maintenance'
  | 'security'
  | 'custom';

export interface AlertAction {
  id: string;
  label: string;
  action: () => void;
  primary?: boolean;
  destructive?: boolean;
}

// Real-time update types
export interface RealTimeUpdate {
  type: UpdateType;
  widgetId?: string;
  data: any;
  timestamp: Date;
  source: string;
}

export type UpdateType = 
  | 'data_update'
  | 'widget_update'
  | 'layout_update'
  | 'alert_update'
  | 'system_update';

// Export and sharing types
export interface ExportOptions {
  format: ExportFormat;
  includeData: boolean;
  includeLayout: boolean;
  includeSettings: boolean;
  dateRange?: {
    start: Date;
    end: Date;
  };
  filters?: WidgetFilter[];
  customOptions?: Record<string, any>;
}

export type ExportFormat = 
  | 'pdf'
  | 'png'
  | 'svg'
  | 'json'
  | 'csv'
  | 'excel'
  | 'html';

export interface ShareOptions {
  type: ShareType;
  permissions: SharePermissions;
  expiration?: Date;
  password?: string;
  allowDownload: boolean;
  allowPrint: boolean;
  watermark?: string;
}

export type ShareType = 
  | 'public_link'
  | 'private_link'
  | 'email'
  | 'embed'
  | 'api';

export interface SharePermissions {
  canView: boolean;
  canEdit: boolean;
  canComment: boolean;
  canShare: boolean;
}

// Widget data binding types
export interface DataBinding {
  sourceId: string;
  sourceType: DataSourceType;
  query?: string;
  filters?: WidgetFilter[];
  aggregation?: DataAggregation;
  refreshInterval: number;
  cacheEnabled: boolean;
  cacheDuration: number;
}

export type DataSourceType = 
  | 'geological'
  | 'seismic'
  | 'volcanic'
  | 'real_time_stream'
  | 'api'
  | 'database'
  | 'file'
  | 'custom';

export interface DataAggregation {
  groupBy?: string[];
  aggregateFields: AggregateField[];
  timeWindow?: TimeWindow;
  sortBy?: SortField[];
  limit?: number;
}

export interface AggregateField {
  field: string;
  function: AggregateFunction;
  alias?: string;
}

export type AggregateFunction = 
  | 'count'
  | 'sum'
  | 'avg'
  | 'min'
  | 'max'
  | 'median'
  | 'std'
  | 'var'
  | 'first'
  | 'last';

export interface TimeWindow {
  duration: number;
  unit: TimeUnit;
  sliding?: boolean;
}

export type TimeUnit = 
  | 'second'
  | 'minute'
  | 'hour'
  | 'day'
  | 'week'
  | 'month'
  | 'year';

export interface SortField {
  field: string;
  direction: 'asc' | 'desc';
}

// Dashboard template types
export interface DashboardTemplate {
  id: string;
  name: string;
  description: string;
  category: TemplateCategory;
  preview: string;
  config: DashboardConfig;
  widgets: DashboardWidget[];
  tags: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedSetupTime: number;
  requirements?: string[];
  author?: string;
  version?: string;
  changelog?: TemplateChange[];
}

export type TemplateCategory = 
  | 'monitoring'
  | 'analysis'
  | 'reporting'
  | 'research'
  | 'operations'
  | 'custom';

export interface TemplateChange {
  version: string;
  date: Date;
  changes: string[];
  breaking?: boolean;
}

// Widget library types
export interface WidgetLibrary {
  id: string;
  name: string;
  description: string;
  widgets: WidgetDefinition[];
  version: string;
  author: string;
  license: string;
  repository?: string;
  documentation?: string;
}

export interface WidgetDefinition {
  id: string;
  name: string;
  description: string;
  category: WidgetCategory;
  icon: string;
  component: React.ComponentType<any>;
  defaultConfig: WidgetConfig;
  configSchema?: any;
  dataRequirements?: DataRequirement[];
  dependencies?: string[];
  examples?: WidgetExample[];
}

export type WidgetCategory = 
  | 'visualization'
  | 'monitoring'
  | 'data'
  | 'analytics'
  | 'alerts'
  | 'controls'
  | 'custom';

export interface DataRequirement {
  type: DataSourceType;
  required: boolean;
  description: string;
  schema?: any;
}

export interface WidgetExample {
  name: string;
  description: string;
  config: WidgetConfig;
  sampleData?: any;
}

// Dashboard state management types
export interface DashboardState {
  currentDashboard?: DashboardConfig;
  widgets: Record<string, DashboardWidget>;
  layouts: Record<string, Layout[]>;
  alerts: DashboardAlert[];
  preferences: UserPreferences;
  isEditMode: boolean;
  isLoading: boolean;
  error?: string;
  lastSaved?: Date;
  hasUnsavedChanges: boolean;
}

export interface DashboardAction {
  type: DashboardActionType;
  payload?: any;
}

export type DashboardActionType = 
  | 'LOAD_DASHBOARD'
  | 'SAVE_DASHBOARD'
  | 'ADD_WIDGET'
  | 'UPDATE_WIDGET'
  | 'REMOVE_WIDGET'
  | 'UPDATE_LAYOUT'
  | 'SET_EDIT_MODE'
  | 'ADD_ALERT'
  | 'ACKNOWLEDGE_ALERT'
  | 'RESOLVE_ALERT'
  | 'UPDATE_PREFERENCES'
  | 'SET_LOADING'
  | 'SET_ERROR'
  | 'CLEAR_ERROR';

// Performance monitoring types
export interface PerformanceMetrics {
  renderTime: number;
  dataLoadTime: number;
  memoryUsage: number;
  cpuUsage: number;
  networkLatency: number;
  errorRate: number;
  userInteractions: number;
  timestamp: Date;
}

export interface PerformanceThresholds {
  maxRenderTime: number;
  maxDataLoadTime: number;
  maxMemoryUsage: number;
  maxCpuUsage: number;
  maxNetworkLatency: number;
  maxErrorRate: number;
}

// Collaboration types
export interface CollaborationSession {
  id: string;
  dashboardId: string;
  participants: Participant[];
  startTime: Date;
  endTime?: Date;
  changes: CollaborationChange[];
  conflicts: Conflict[];
}

export interface Participant {
  userId: string;
  name: string;
  avatar?: string;
  role: ParticipantRole;
  joinTime: Date;
  leaveTime?: Date;
  cursor?: CursorPosition;
}

export type ParticipantRole = 
  | 'owner'
  | 'editor'
  | 'viewer'
  | 'commenter';

export interface CursorPosition {
  x: number;
  y: number;
  widgetId?: string;
}

export interface CollaborationChange {
  id: string;
  userId: string;
  type: ChangeType;
  target: string;
  before: any;
  after: any;
  timestamp: Date;
  applied: boolean;
}

export type ChangeType = 
  | 'widget_add'
  | 'widget_remove'
  | 'widget_update'
  | 'layout_change'
  | 'config_update';

export interface Conflict {
  id: string;
  changes: CollaborationChange[];
  resolution?: ConflictResolution;
  resolvedBy?: string;
  resolvedAt?: Date;
}

export interface ConflictResolution {
  strategy: ResolutionStrategy;
  selectedChange?: string;
  mergedResult?: any;
}

export type ResolutionStrategy = 
  | 'accept_local'
  | 'accept_remote'
  | 'merge'
  | 'manual';