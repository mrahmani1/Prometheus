import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { LayoutConfig, NotificationMessage, LoadingState, ErrorState } from '@/types/ui';

interface UIState {
  layout: LayoutConfig;
  notifications: NotificationMessage[];
  loading: LoadingState;
  error: ErrorState;
  activeWorkspace: string | null;
  sidebarCollapsed: boolean;
}

const initialState: UIState = {
  layout: {
    sidebarOpen: true,
    theme: 'light',
    primaryColor: '#1976d2',
    compactMode: false,
  },
  notifications: [],
  loading: {
    isLoading: false,
  },
  error: {
    hasError: false,
  },
  activeWorkspace: null,
  sidebarCollapsed: false,
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    // Layout actions
    toggleSidebar: (state) => {
      state.layout.sidebarOpen = !state.layout.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.layout.sidebarOpen = action.payload;
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.layout.theme = action.payload;
    },
    setPrimaryColor: (state, action: PayloadAction<string>) => {
      state.layout.primaryColor = action.payload;
    },
    setCompactMode: (state, action: PayloadAction<boolean>) => {
      state.layout.compactMode = action.payload;
    },
    updateLayoutConfig: (state, action: PayloadAction<Partial<LayoutConfig>>) => {
      state.layout = { ...state.layout, ...action.payload };
    },
    
    // Notification actions
    addNotification: (state, action: PayloadAction<Omit<NotificationMessage, 'id' | 'timestamp'>>) => {
      const notification: NotificationMessage = {
        ...action.payload,
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
      };
      state.notifications.push(notification);
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    
    // Loading actions
    setLoading: (state, action: PayloadAction<LoadingState>) => {
      state.loading = action.payload;
    },
    startLoading: (state, action: PayloadAction<{ message?: string; progress?: number }>) => {
      state.loading = {
        isLoading: true,
        message: action.payload.message,
        progress: action.payload.progress,
      };
    },
    stopLoading: (state) => {
      state.loading = {
        isLoading: false,
      };
    },
    updateLoadingProgress: (state, action: PayloadAction<number>) => {
      if (state.loading.isLoading) {
        state.loading.progress = action.payload;
      }
    },
    
    // Error actions
    setError: (state, action: PayloadAction<ErrorState>) => {
      state.error = action.payload;
    },
    showError: (state, action: PayloadAction<{ message: string; details?: string }>) => {
      state.error = {
        hasError: true,
        message: action.payload.message,
        details: action.payload.details,
        timestamp: new Date().toISOString(),
      };
    },
    clearError: (state) => {
      state.error = {
        hasError: false,
      };
    },
    
    // Workspace actions
    setActiveWorkspace: (state, action: PayloadAction<string | null>) => {
      state.activeWorkspace = action.payload;
    },
    
    // Sidebar actions
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
    },
    toggleSidebarCollapsed: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
  },
});

export const {
  toggleSidebar,
  setSidebarOpen,
  setTheme,
  setPrimaryColor,
  setCompactMode,
  updateLayoutConfig,
  addNotification,
  removeNotification,
  clearNotifications,
  setLoading,
  startLoading,
  stopLoading,
  updateLoadingProgress,
  setError,
  showError,
  clearError,
  setActiveWorkspace,
  setSidebarCollapsed,
  toggleSidebarCollapsed,
} = uiSlice.actions;

export default uiSlice.reducer;