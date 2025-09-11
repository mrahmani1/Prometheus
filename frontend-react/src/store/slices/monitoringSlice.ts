/**
 * Real-time monitoring slice for system health and geological events
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface MonitoringState {
  // System health
  systemHealth: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    services: Record<string, {
      status: 'healthy' | 'degraded' | 'unhealthy';
      responseTime: number;
      lastCheck: Date;
    }>;
    lastUpdate: Date | null;
  };
  
  // Real-time alerts
  alerts: {
    active: Array<{
      id: string;
      type: 'seismic' | 'volcanic' | 'deformation' | 'system';
      severity: 'low' | 'medium' | 'high' | 'critical';
      message: string;
      timestamp: Date;
      acknowledged: boolean;
      location?: [number, number];
    }>;
    history: Array<{
      id: string;
      type: string;
      severity: string;
      message: string;
      timestamp: Date;
      resolvedAt?: Date;
    }>;
  };
  
  // WebSocket connection status
  websocket: {
    connected: boolean;
    reconnecting: boolean;
    lastConnected: Date | null;
    messageCount: number;
    subscriptions: string[];
  };
  
  // Performance metrics
  performance: {
    cpu: number;
    memory: number;
    network: number;
    renderFps: number;
    apiLatency: number;
  };
}

const initialState: MonitoringState = {
  systemHealth: {
    status: 'healthy',
    services: {},
    lastUpdate: null,
  },
  alerts: {
    active: [],
    history: [],
  },
  websocket: {
    connected: false,
    reconnecting: false,
    lastConnected: null,
    messageCount: 0,
    subscriptions: [],
  },
  performance: {
    cpu: 0,
    memory: 0,
    network: 0,
    renderFps: 60,
    apiLatency: 0,
  },
};

const monitoringSlice = createSlice({
  name: 'monitoring',
  initialState,
  reducers: {
    updateSystemHealth: (state, action: PayloadAction<MonitoringState['systemHealth']>) => {
      state.systemHealth = action.payload;
    },
    addAlert: (state, action: PayloadAction<MonitoringState['alerts']['active'][0]>) => {
      state.alerts.active.push(action.payload);
    },
    acknowledgeAlert: (state, action: PayloadAction<string>) => {
      const alert = state.alerts.active.find(a => a.id === action.payload);
      if (alert) {
        alert.acknowledged = true;
      }
    },
    removeAlert: (state, action: PayloadAction<string>) => {
      const index = state.alerts.active.findIndex(a => a.id === action.payload);
      if (index !== -1) {
        const alert = state.alerts.active.splice(index, 1)[0];
        state.alerts.history.push({
          ...alert,
          resolvedAt: new Date(),
        });
      }
    },
    setWebSocketStatus: (state, action: PayloadAction<Partial<MonitoringState['websocket']>>) => {
      state.websocket = { ...state.websocket, ...action.payload };
    },
    updatePerformanceMetrics: (state, action: PayloadAction<Partial<MonitoringState['performance']>>) => {
      state.performance = { ...state.performance, ...action.payload };
    },
  },
});

export const {
  updateSystemHealth,
  addAlert,
  acknowledgeAlert,
  removeAlert,
  setWebSocketStatus,
  updatePerformanceMetrics,
} = monitoringSlice.actions;

export default monitoringSlice.reducer;