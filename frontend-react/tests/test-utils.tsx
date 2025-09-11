import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { configureStore } from '@reduxjs/toolkit';
import { theme } from '../src/theme';

// Mock store for testing
const createMockStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      // Add your reducers here
      app: (state = { loading: false, error: null }, action) => state,
      geological: (state = { data: [], loading: false }, action) => state,
      dashboard: (state = { widgets: [], layout: {} }, action) => state,
    },
    preloadedState: initialState,
  });
};

// Custom render function that includes providers
const AllTheProviders: React.FC<{ children: React.ReactNode; initialState?: any }> = ({ 
  children, 
  initialState = {} 
}) => {
  const store = createMockStore(initialState);
  
  return (
    <Provider store={store}>
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          {children}
        </ThemeProvider>
      </BrowserRouter>
    </Provider>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'> & { initialState?: any }
) => {
  const { initialState, ...renderOptions } = options || {};
  
  return render(ui, {
    wrapper: (props) => <AllTheProviders {...props} initialState={initialState} />,
    ...renderOptions,
  });
};

// Mock geological data for testing
export const mockGeologicalData = {
  geological: [
    {
      id: 'test-geo-1',
      timestamp: new Date('2023-01-01T00:00:00Z'),
      data_type: 'volcanic' as const,
      latitude: 37.7749,
      longitude: -122.4194,
      elevation: 100,
      quality_score: 0.95
    },
    {
      id: 'test-geo-2',
      timestamp: new Date('2023-01-01T01:00:00Z'),
      data_type: 'igneous' as const,
      latitude: 40.7128,
      longitude: -74.0060,
      elevation: 50,
      quality_score: 0.87
    }
  ],
  seismic: [
    {
      eventId: 'test-seismic-1',
      timestamp: new Date('2023-01-01T00:30:00Z'),
      location: { x: -122.4194, y: 37.7749, z: 10 },
      magnitude: 4.5,
      depth: 15,
      eventType: 'earthquake' as const,
      confidence: 0.9,
      waveformData: new Float32Array([0.1, 0.2, -0.1, 0.3, -0.2])
    }
  ],
  volcanic: [
    {
      volcanoId: 'test-volcano-1',
      timestamp: new Date('2023-01-01T00:00:00Z'),
      alertLevel: 'yellow' as const,
      magmaChamberPressure: 75,
      deformationRate: 2.5,
      gasEmissions: {
        co2: 400,
        so2: 50,
        h2s: 10,
        h2o: 1000,
        temperature: 85,
        flux: 150
      },
      thermalAnomalies: [],
      eruptionProbability: 0.15
    }
  ]
};

// Mock dashboard configuration
export const mockDashboardConfig = {
  id: 'test-dashboard',
  name: 'Test Dashboard',
  description: 'Test dashboard for unit tests',
  layout: {
    lg: [
      { i: 'widget-1', x: 0, y: 0, w: 6, h: 4 },
      { i: 'widget-2', x: 6, y: 0, w: 6, h: 4 }
    ]
  },
  widgets: ['widget-1', 'widget-2'],
  theme: 'light' as const,
  autoRefresh: false,
  refreshInterval: 30000,
  permissions: {
    canEdit: true,
    canAdd: true,
    canDelete: true,
    canExport: true
  }
};

// Mock user preferences
export const mockUserPreferences = {
  theme: 'light' as const,
  language: 'en',
  timezone: 'UTC',
  dateFormat: 'MM/dd/yyyy',
  numberFormat: 'en-US',
  autoSave: true,
  notifications: {
    enabled: true,
    sound: false,
    desktop: false,
    email: false,
    alertLevels: ['warning', 'error', 'critical'] as const
  },
  accessibility: {
    highContrast: false,
    largeText: false,
    reducedMotion: true,
    screenReader: false,
    keyboardNavigation: true
  },
  performance: {
    enableVirtualization: true,
    maxDataPoints: 1000,
    renderQuality: 'medium' as const,
    enableWebGL: false,
    enableWorkers: false,
    cacheSize: 50
  }
};

// Utility functions for testing
export const waitForElementToBeRemoved = async (element: HTMLElement) => {
  return new Promise<void>((resolve) => {
    const observer = new MutationObserver(() => {
      if (!document.contains(element)) {
        observer.disconnect();
        resolve();
      }
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  });
};

export const mockIntersectionObserver = () => {
  const mockIntersectionObserver = vi.fn();
  mockIntersectionObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null
  });
  
  window.IntersectionObserver = mockIntersectionObserver;
};

export const mockResizeObserver = () => {
  const mockResizeObserver = vi.fn();
  mockResizeObserver.mockReturnValue({
    observe: () => null,
    unobserve: () => null,
    disconnect: () => null
  });
  
  window.ResizeObserver = mockResizeObserver;
};

export const mockWebGL = () => {
  const mockWebGLContext = {
    createShader: vi.fn(() => ({})),
    shaderSource: vi.fn(),
    compileShader: vi.fn(),
    createProgram: vi.fn(() => ({})),
    attachShader: vi.fn(),
    linkProgram: vi.fn(),
    useProgram: vi.fn(),
    createBuffer: vi.fn(() => ({})),
    bindBuffer: vi.fn(),
    bufferData: vi.fn(),
    getAttribLocation: vi.fn(() => 0),
    getUniformLocation: vi.fn(() => ({})),
    enableVertexAttribArray: vi.fn(),
    vertexAttribPointer: vi.fn(),
    uniform1f: vi.fn(),
    uniformMatrix4fv: vi.fn(),
    drawArrays: vi.fn(),
    viewport: vi.fn(),
    clearColor: vi.fn(),
    clear: vi.fn(),
    enable: vi.fn(),
    depthFunc: vi.fn(),
    blendFunc: vi.fn()
  };
  
  HTMLCanvasElement.prototype.getContext = vi.fn((contextType) => {
    if (contextType === 'webgl' || contextType === 'webgl2') {
      return mockWebGLContext;
    }
    return null;
  });
};

// Re-export everything from React Testing Library
export * from '@testing-library/react';

// Override the default render with our custom render
export { customRender as render };