/**
 * Notification management slice
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface NotificationState {
  notifications: Array<{
    id: string;
    type: 'success' | 'info' | 'warning' | 'error';
    title: string;
    message: string;
    timestamp: Date;
    duration?: number;
    persistent: boolean;
    actions?: Array<{
      label: string;
      action: string;
    }>;
  }>;
  settings: {
    position: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
    maxNotifications: number;
    defaultDuration: number;
  };
}

const initialState: NotificationState = {
  notifications: [],
  settings: {
    position: 'top-right',
    maxNotifications: 5,
    defaultDuration: 5000,
  },
};

const notificationSlice = createSlice({
  name: 'notifications',
  initialState,
  reducers: {
    addNotification: (state, action: PayloadAction<Omit<NotificationState['notifications'][0], 'id' | 'timestamp'>>) => {
      const notification = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date(),
      };
      state.notifications.unshift(notification);
      
      // Limit number of notifications
      if (state.notifications.length > state.settings.maxNotifications) {
        state.notifications = state.notifications.slice(0, state.settings.maxNotifications);
      }
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearAllNotifications: state => {
      state.notifications = [];
    },
    updateSettings: (state, action: PayloadAction<Partial<NotificationState['settings']>>) => {
      state.settings = { ...state.settings, ...action.payload };
    },
  },
});

export const {
  addNotification,
  removeNotification,
  clearAllNotifications,
  updateSettings,
} = notificationSlice.actions;

export default notificationSlice.reducer;