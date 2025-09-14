/**
 * Enhanced Dashboard configuration and layout management slice for Comprehensive Causal Dashboard
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface DashboardNotification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: Date;
  autoHide?: boolean;
  duration?: number;
  read?: boolean;
}

interface DashboardConfig {
  layout: 'grid' | 'tabs' | 'split' | 'custom';
  theme: 'light' | 'dark' | 'auto';
  realTimeUpdates: boolean;
  autoRefresh: boolean;
  refreshInterval: number;
  notifications: boolean;
  soundAlerts: boolean;
  performanceMode: 'high' | 'balanced' | 'low';
  renderQuality: 'high' | 'medium' | 'low';
  animationsEnabled: boolean;
  collaborativeFeatures: boolean;
}

interface DashboardState {
  // Legacy layout system
  layouts: {
    [layoutId: string]: {
      id: string;
      name: string;
      widgets: Array<{
        id: string;
        type: string;
        position: { x: number; y: number };
        size: { width: number; height: number };
        config: Record<string, any>;
        minimized: boolean;
      }>;
      isDefault: boolean;
    };
  };
  activeLayout: string;
  editMode: boolean;
  gridSize: number;
  snapToGrid: boolean;
  
  // Enhanced causal dashboard state
  activeTab: string;
  drawerOpen: boolean;
  fullscreen: boolean;
  config: DashboardConfig;
  notifications: DashboardNotification[];
  isLoading: boolean;
  error: string | null;
  realTimeStatus: 'connected' | 'disconnected' | 'error';
  performanceMetrics: {
    fps: number;
    renderTime: number;
    nodeCount: number;
    edgeCount: number;
    memoryUsage: number;
  };
  collaborativeSession: {
    active: boolean;
    sessionId?: string;
    participants: Array<{
      id: string;
      name: string;
      avatar?: string;
      status: 'online' | 'away' | 'offline';
    }>;
    permissions: {
      canEdit: boolean;
      canAnnotate: boolean;
      canShare: boolean;
    };
  };
}

const initialState: DashboardState = {
  // Legacy layout system
  layouts: {
    default: {
      id: 'default',
      name: 'Default Layout',
      widgets: [],
      isDefault: true,
    },
  },
  activeLayout: 'default',
  editMode: false,
  gridSize: 20,
  snapToGrid: true,
  
  // Enhanced causal dashboard state
  activeTab: 'network',
  drawerOpen: true,
  fullscreen: false,
  config: {
    layout: 'tabs',
    theme: 'auto',
    realTimeUpdates: true,
    autoRefresh: true,
    refreshInterval: 30000, // 30 seconds
    notifications: true,
    soundAlerts: false,
    performanceMode: 'balanced',
    renderQuality: 'high',
    animationsEnabled: true,
    collaborativeFeatures: false,
  },
  notifications: [],
  isLoading: false,
  error: null,
  realTimeStatus: 'disconnected',
  performanceMetrics: {
    fps: 60,
    renderTime: 0,
    nodeCount: 0,
    edgeCount: 0,
    memoryUsage: 0,
  },
  collaborativeSession: {
    active: false,
    participants: [],
    permissions: {
      canEdit: true,
      canAnnotate: true,
      canShare: true,
    },
  },
};

const dashboardSlice = createSlice({
  name: 'dashboard',
  initialState,
  reducers: {
    // Legacy layout actions
    setActiveLayout: (state, action: PayloadAction<string>) => {
      state.activeLayout = action.payload;
    },
    setEditMode: (state, action: PayloadAction<boolean>) => {
      state.editMode = action.payload;
    },
    addWidget: (state, action: PayloadAction<DashboardState['layouts'][string]['widgets'][0]>) => {
      const layout = state.layouts[state.activeLayout];
      if (layout) {
        layout.widgets.push(action.payload);
      }
    },
    removeWidget: (state, action: PayloadAction<string>) => {
      const layout = state.layouts[state.activeLayout];
      if (layout) {
        layout.widgets = layout.widgets.filter(w => w.id !== action.payload);
      }
    },
    updateWidget: (state, action: PayloadAction<{ id: string; updates: Partial<DashboardState['layouts'][string]['widgets'][0]> }>) => {
      const layout = state.layouts[state.activeLayout];
      if (layout) {
        const widget = layout.widgets.find(w => w.id === action.payload.id);
        if (widget) {
          Object.assign(widget, action.payload.updates);
        }
      }
    },
    
    // Enhanced causal dashboard actions
    setActiveTab: (state, action: PayloadAction<string>) => {
      state.activeTab = action.payload;
    },
    setDrawerOpen: (state, action: PayloadAction<boolean>) => {
      state.drawerOpen = action.payload;
    },
    setFullscreen: (state, action: PayloadAction<boolean>) => {
      state.fullscreen = action.payload;
    },
    updateDashboardConfig: (state, action: PayloadAction<Partial<DashboardConfig>>) => {
      state.config = { ...state.config, ...action.payload };
    },
    addNotification: (state, action: PayloadAction<DashboardNotification>) => {
      state.notifications.push(action.payload);
      // Keep only the last 50 notifications to prevent memory issues
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(-50);
      }
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    markNotificationAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    clearAllNotifications: (state) => {
      state.notifications = [];
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setRealTimeStatus: (state, action: PayloadAction<'connected' | 'disconnected' | 'error'>) => {
      state.realTimeStatus = action.payload;
    },
    updatePerformanceMetrics: (state, action: PayloadAction<Partial<DashboardState['performanceMetrics']>>) => {
      state.performanceMetrics = { ...state.performanceMetrics, ...action.payload };
    },
    startCollaborativeSession: (state, action: PayloadAction<{ sessionId: string; participants: DashboardState['collaborativeSession']['participants'] }>) => {
      state.collaborativeSession.active = true;
      state.collaborativeSession.sessionId = action.payload.sessionId;
      state.collaborativeSession.participants = action.payload.participants;
    },
    endCollaborativeSession: (state) => {
      state.collaborativeSession.active = false;
      state.collaborativeSession.sessionId = undefined;
      state.collaborativeSession.participants = [];
    },
    updateCollaborativeParticipants: (state, action: PayloadAction<DashboardState['collaborativeSession']['participants']>) => {
      state.collaborativeSession.participants = action.payload;
    },
    updateCollaborativePermissions: (state, action: PayloadAction<Partial<DashboardState['collaborativeSession']['permissions']>>) => {
      state.collaborativeSession.permissions = { ...state.collaborativeSession.permissions, ...action.payload };
    },
  },
});

export const {
  // Legacy actions
  setActiveLayout,
  setEditMode,
  addWidget,
  removeWidget,
  updateWidget,
  
  // Enhanced causal dashboard actions
  setActiveTab,
  setDrawerOpen,
  setFullscreen,
  updateDashboardConfig,
  addNotification,
  removeNotification,
  markNotificationAsRead,
  clearAllNotifications,
  setLoading,
  setError,
  setRealTimeStatus,
  updatePerformanceMetrics,
  startCollaborativeSession,
  endCollaborativeSession,
  updateCollaborativeParticipants,
  updateCollaborativePermissions,
} = dashboardSlice.actions;

export default dashboardSlice.reducer;