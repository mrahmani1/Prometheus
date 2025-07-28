// UI-specific types
export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: string;
  children?: NavigationItem[];
  requiredPermissions?: string[];
}

export interface LayoutConfig {
  sidebarOpen: boolean;
  theme: 'light' | 'dark';
  primaryColor: string;
  compactMode: boolean;
}

export interface NotificationMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  autoHide?: boolean;
  duration?: number;
}

export interface DataVisualizationConfig {
  type: '2D' | '3D' | '4D' | 'chart' | 'map';
  layers: VisualizationLayer[];
  viewport: {
    center: [number, number];
    zoom: number;
    rotation?: number;
  };
  timeRange?: [string, string];
  colorScheme: string;
  showLegend: boolean;
  showControls: boolean;
}

export interface VisualizationLayer {
  id: string;
  name: string;
  type: 'seismic' | 'deformation' | 'thermal' | 'geochemical' | 'topography' | 'geological';
  visible: boolean;
  opacity: number;
  dataSource: string;
  style: Record<string, any>;
}

export interface LoadingState {
  isLoading: boolean;
  message?: string;
  progress?: number;
}

export interface ErrorState {
  hasError: boolean;
  message?: string;
  details?: string;
  timestamp?: string;
}