/**
 * UI state management slice for theme, layout, and user interface preferences
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UIState {
  theme: 'light' | 'dark' | 'auto';
  sidebarOpen: boolean;
  sidebarCollapsed: boolean;
  language: string;
  timezone: string;
  units: {
    length: 'metric' | 'imperial';
    temperature: 'celsius' | 'fahrenheit' | 'kelvin';
    pressure: 'pascal' | 'bar' | 'psi';
    mass: 'kilogram' | 'pound';
  };
  loading: {
    global: boolean;
    components: Record<string, boolean>;
  };
  modals: {
    [key: string]: {
      open: boolean;
      data?: any;
    };
  };
  notifications: {
    enabled: boolean;
    position: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
    duration: number;
  };
  accessibility: {
    highContrast: boolean;
    reducedMotion: boolean;
    screenReader: boolean;
    fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  };
  layout: {
    headerHeight: number;
    sidebarWidth: number;
    footerHeight: number;
    contentPadding: number;
  };
  viewport: {
    width: number;
    height: number;
    isMobile: boolean;
    isTablet: boolean;
    isDesktop: boolean;
  };
  performance: {
    enableAnimations: boolean;
    enableTransitions: boolean;
    enableShadows: boolean;
    renderQuality: 'low' | 'medium' | 'high' | 'ultra';
  };
}

const initialState: UIState = {
  theme: 'auto',
  sidebarOpen: true,
  sidebarCollapsed: false,
  language: 'en',
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  units: {
    length: 'metric',
    temperature: 'celsius',
    pressure: 'pascal',
    mass: 'kilogram',
  },
  loading: {
    global: false,
    components: {},
  },
  modals: {},
  notifications: {
    enabled: true,
    position: 'top-right',
    duration: 5000,
  },
  accessibility: {
    highContrast: false,
    reducedMotion: false,
    screenReader: false,
    fontSize: 'medium',
  },
  layout: {
    headerHeight: 64,
    sidebarWidth: 280,
    footerHeight: 48,
    contentPadding: 24,
  },
  viewport: {
    width: window.innerWidth,
    height: window.innerHeight,
    isMobile: window.innerWidth < 768,
    isTablet: window.innerWidth >= 768 && window.innerWidth < 1024,
    isDesktop: window.innerWidth >= 1024,
  },
  performance: {
    enableAnimations: true,
    enableTransitions: true,
    enableShadows: true,
    renderQuality: 'high',
  },
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'auto'>) => {
      state.theme = action.payload;
    },
    toggleSidebar: state => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    toggleSidebarCollapsed: state => {
      state.sidebarCollapsed = !state.sidebarCollapsed;
    },
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload;
    },
    setLanguage: (state, action: PayloadAction<string>) => {
      state.language = action.payload;
    },
    setTimezone: (state, action: PayloadAction<string>) => {
      state.timezone = action.payload;
    },
    setUnits: (state, action: PayloadAction<Partial<UIState['units']>>) => {
      state.units = { ...state.units, ...action.payload };
    },
    setGlobalLoading: (state, action: PayloadAction<boolean>) => {
      state.loading.global = action.payload;
    },
    setComponentLoading: (state, action: PayloadAction<{ component: string; loading: boolean }>) => {
      state.loading.components[action.payload.component] = action.payload.loading;
    },
    openModal: (state, action: PayloadAction<{ id: string; data?: any }>) => {
      state.modals[action.payload.id] = {
        open: true,
        data: action.payload.data,
      };
    },
    closeModal: (state, action: PayloadAction<string>) => {
      if (state.modals[action.payload]) {
        state.modals[action.payload].open = false;
      }
    },
    closeAllModals: state => {
      Object.keys(state.modals).forEach(key => {
        state.modals[key].open = false;
      });
    },
    setNotificationSettings: (state, action: PayloadAction<Partial<UIState['notifications']>>) => {
      state.notifications = { ...state.notifications, ...action.payload };
    },
    setAccessibilitySettings: (state, action: PayloadAction<Partial<UIState['accessibility']>>) => {
      state.accessibility = { ...state.accessibility, ...action.payload };
    },
    setLayoutSettings: (state, action: PayloadAction<Partial<UIState['layout']>>) => {
      state.layout = { ...state.layout, ...action.payload };
    },
    updateViewport: (state, action: PayloadAction<{ width: number; height: number }>) => {
      state.viewport.width = action.payload.width;
      state.viewport.height = action.payload.height;
      state.viewport.isMobile = action.payload.width < 768;
      state.viewport.isTablet = action.payload.width >= 768 && action.payload.width < 1024;
      state.viewport.isDesktop = action.payload.width >= 1024;
    },
    setPerformanceSettings: (state, action: PayloadAction<Partial<UIState['performance']>>) => {
      state.performance = { ...state.performance, ...action.payload };
    },
    resetUISettings: state => {
      // Reset to defaults but preserve viewport and loading states
      const { viewport, loading } = state;
      return { ...initialState, viewport, loading };
    },
  },
});

export const {
  setTheme,
  toggleSidebar,
  setSidebarOpen,
  toggleSidebarCollapsed,
  setSidebarCollapsed,
  setLanguage,
  setTimezone,
  setUnits,
  setGlobalLoading,
  setComponentLoading,
  openModal,
  closeModal,
  closeAllModals,
  setNotificationSettings,
  setAccessibilitySettings,
  setLayoutSettings,
  updateViewport,
  setPerformanceSettings,
  resetUISettings,
} = uiSlice.actions;

export default uiSlice.reducer;