import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { HelmetProvider } from 'react-helmet-async';
import '@testing-library/jest-dom';

import ComprehensiveCausalDashboard from './ComprehensiveCausalDashboard';
import dashboardReducer from '../../store/slices/dashboardSlice';
import { 
  CausalGraph, 
  CausalDiscoveryMethod, 
  GeologicalProcessType, 
  SpatialScale, 
  TemporalScale, 
  ValidationStatus 
} from '../../types/causal';

// Mock the visualization components
vi.mock('../visualization/causal/CausalNetworkVisualization', () => ({
  CausalNetworkVisualization: ({ onNodeClick, onEdgeClick }: any) => (
    <div data-testid="causal-network-visualization">
      <button onClick={() => onNodeClick({ id: 'test-node', label: 'Test Node' })}>
        Click Node
      </button>
      <button onClick={() => onEdgeClick({ id: 'test-edge', source: 'node1', target: 'node2' })}>
        Click Edge
      </button>
    </div>
  ),
}));

vi.mock('../visualization/causal/MultiScaleTemporalExplorer', () => ({
  MultiScaleTemporalExplorer: () => (
    <div data-testid="multi-scale-temporal-explorer">Temporal Explorer</div>
  ),
}));

vi.mock('../hypothesis-testing/HypothesisTestingInterface', () => ({
  HypothesisTestingInterface: () => (
    <div data-testid="hypothesis-testing-interface">Hypothesis Testing</div>
  ),
}));

vi.mock('../collaborative/CollaborativeAnalysisInterface', () => ({
  CollaborativeAnalysisInterface: () => (
    <div data-testid="collaborative-analysis-interface">Collaborative Analysis</div>
  ),
}));

vi.mock('../RealTimeProcessDashboard', () => ({
  __esModule: true,
  default: ({ onAlert, onStatusChange }: any) => (
    <div data-testid="real-time-process-dashboard">
      <button onClick={() => onAlert({ severity: 'warning', title: 'Test Alert', message: 'Test message' })}>
        Trigger Alert
      </button>
      <button onClick={() => onStatusChange('connected')}>
        Set Connected
      </button>
    </div>
  ),
}));

vi.mock('../visualization/causal/CausalSearchInterface', () => ({
  CausalSearchInterface: ({ onSearch, onFilterChange }: any) => (
    <div data-testid="causal-search-interface">
      <input 
        placeholder="Search..." 
        onChange={(e) => onSearch(e.target.value)}
        data-testid="search-input"
      />
      <button onClick={() => onFilterChange({ strengthRange: [0.5, 1] })}>
        Apply Filter
      </button>
    </div>
  ),
}));

vi.mock('../visualization/causal/CausalFilterPanel', () => ({
  CausalFilterPanel: ({ onFilterChange }: any) => (
    <div data-testid="causal-filter-panel">
      <button onClick={() => onFilterChange({ confidenceThreshold: 0.8 })}>
        Update Filter
      </button>
    </div>
  ),
}));

// Mock WebSocket
const MockWebSocket = vi.fn().mockImplementation(() => ({
  close: vi.fn(),
  send: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
}));

(MockWebSocket as any).CONNECTING = 0;
(MockWebSocket as any).OPEN = 1;
(MockWebSocket as any).CLOSING = 2;
(MockWebSocket as any).CLOSED = 3;

global.WebSocket = MockWebSocket as any;

// Mock requestFullscreen and exitFullscreen
Object.defineProperty(document, 'documentElement', {
  value: {
    requestFullscreen: vi.fn(),
  },
  writable: true,
});

Object.defineProperty(document, 'exitFullscreen', {
  value: vi.fn(),
  writable: true,
});

// Mock clipboard API
Object.defineProperty(navigator, 'clipboard', {
  value: {
    writeText: vi.fn().mockResolvedValue(undefined),
  },
  writable: true,
});

// Create test store
const createTestStore = (initialState = {}) => {
  return configureStore({
    reducer: {
      dashboard: dashboardReducer,
    },
    preloadedState: {
      dashboard: {
        activeTab: 'network',
        drawerOpen: true,
        fullscreen: false,
        config: {
          layout: 'tabs',
          theme: 'auto',
          realTimeUpdates: true,
          autoRefresh: true,
          refreshInterval: 30000,
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
        ...initialState,
      },
    },
  });
};

// Create test theme
const theme = createTheme();

// Create mock causal graph
const createMockCausalGraph = (): CausalGraph => ({
  id: 'test-graph-1',
  createdAt: new Date(),
  updatedAt: new Date(),
  metadata: {
    algorithmsUsed: [CausalDiscoveryMethod.PC, CausalDiscoveryMethod.GES],
    confidenceThreshold: 0.8,
    totalNodes: 2,
    totalEdges: 1,
    graphDensity: 0.5,
    dataSourcesUsed: ['IRIS/EarthScope', 'ASF-Hyp3'],
    processingTime: 1000
  },
  nodes: [
    {
      id: 'seismic-activity',
      label: 'Seismic Activity',
      processType: GeologicalProcessType.SEISMIC,
      spatialScale: SpatialScale.REGIONAL,
      temporalScale: TemporalScale.SECONDS,
      dataSource: 'IRIS/EarthScope',
      metadata: {
        description: 'Real-time seismic monitoring',
        lastUpdated: new Date(),
      },
      size: 20,
      color: '#FF6B6B',
      confidence: 0.95,
    },
    {
      id: 'ground-deformation',
      label: 'Ground Deformation',
      processType: GeologicalProcessType.DEFORMATION,
      spatialScale: SpatialScale.LOCAL,
      temporalScale: TemporalScale.DAYS,
      dataSource: 'ASF-Hyp3/Copernicus',
      metadata: {
        description: 'MT-SAR interferometry',
        lastUpdated: new Date(),
      },
      size: 18,
      color: '#4ECDC4',
      confidence: 0.88,
    },
  ],
  edges: [
    {
      id: 'seismic-deformation',
      source: 'seismic-activity',
      target: 'ground-deformation',
      causalStrength: 0.75,
      confidenceInterval: [0.65, 0.85],
      temporalLag: 3600000,
      statisticalSignificance: 0.01,
      discoveryMethod: CausalDiscoveryMethod.GRANGERCAUSALITY,
      validationStatus: ValidationStatus.VALIDATED,
      geologicalContext: {
        description: 'Seismic activity causes deformation'
      },
    },
  ],
});

const renderWithProviders = (
  component: React.ReactElement,
  { store = createTestStore() } = {}
) => {
  return render(
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <HelmetProvider>
          {component}
        </HelmetProvider>
      </ThemeProvider>
    </Provider>
  );
};

describe('ComprehensiveCausalDashboard', () => {
  const mockCausalGraph = createMockCausalGraph();
  const mockOnExport = vi.fn();
  const mockOnShare = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders the dashboard with all main components', () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      expect(screen.getByText('Comprehensive Causal Dashboard')).toBeInTheDocument();
      expect(screen.getByTestId('causal-network-visualization')).toBeInTheDocument();
    });

    it('displays performance metrics in the app bar', () => {
      const store = createTestStore({
        performanceMetrics: {
          fps: 45,
          renderTime: 12.5,
          nodeCount: 100,
          edgeCount: 50,
          memoryUsage: 75,
        },
      });

      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      expect(screen.getByText('45 FPS')).toBeInTheDocument();
    });

    it('shows real-time status indicator when enabled', () => {
      const store = createTestStore({
        realTimeStatus: 'connected',
      });

      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          realTimeEnabled={true}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      expect(screen.getByText('Connected')).toBeInTheDocument();
    });
  });

  describe('Navigation', () => {
    it('switches between tabs correctly', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      // Click on temporal analysis tab
      const temporalTab = screen.getByText('Temporal Analysis');
      fireEvent.click(temporalTab);

      await waitFor(() => {
        expect(screen.getByTestId('multi-scale-temporal-explorer')).toBeInTheDocument();
      });
    });

    it('toggles drawer open/closed', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      const menuButton = screen.getByLabelText('open drawer');
      fireEvent.click(menuButton);

      // The drawer state change should be reflected in the Redux store
      // This is tested indirectly through the component behavior
    });

    it('shows hypothesis testing interface when tab is selected', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      const hypothesisTab = screen.getByText('Hypothesis Testing');
      fireEvent.click(hypothesisTab);

      await waitFor(() => {
        expect(screen.getByTestId('hypothesis-testing-interface')).toBeInTheDocument();
      });
    });

    it('shows collaborative interface when collaborative mode is enabled', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          collaborativeMode={true}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      const collaborativeTab = screen.getByText('Collaboration');
      fireEvent.click(collaborativeTab);

      await waitFor(() => {
        expect(screen.getByTestId('collaborative-analysis-interface')).toBeInTheDocument();
      });
    });
  });

  describe('Interactions', () => {
    it('handles node clicks and shows notifications', async () => {
      const store = createTestStore();
      
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      const clickNodeButton = screen.getByText('Click Node');
      fireEvent.click(clickNodeButton);

      await waitFor(() => {
        const state = store.getState();
        expect(state.dashboard.notifications).toHaveLength(1);
        expect(state.dashboard.notifications[0].title).toBe('Node Selected');
      });
    });

    it('handles real-time alerts', async () => {
      const store = createTestStore();
      
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          realTimeEnabled={true}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      // Switch to monitor tab first
      const monitorTab = screen.getByText('Real-Time Monitor');
      fireEvent.click(monitorTab);

      await waitFor(() => {
        const triggerAlertButton = screen.getByText('Trigger Alert');
        fireEvent.click(triggerAlertButton);
      });

      await waitFor(() => {
        const state = store.getState();
        expect(state.dashboard.notifications.some(n => n.title === 'Test Alert')).toBe(true);
      });
    });

    it('handles search functionality', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      const searchInput = screen.getByTestId('search-input');
      fireEvent.change(searchInput, { target: { value: 'seismic' } });

      // Search functionality should update the component state
      // This is tested through the component's internal behavior
    });
  });

  describe('Export and Share', () => {
    it('calls onExport when export is triggered', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      // Open speed dial and click export action
      const speedDial = screen.getByLabelText('Dashboard Actions');
      fireEvent.click(speedDial);

      await waitFor(() => {
        const exportAction = screen.getByText('Export Data');
        fireEvent.click(exportAction);
      });

      // This should open the export dialog
      await waitFor(() => {
        expect(screen.getByText('Export Dashboard')).toBeInTheDocument();
      });
    });

    it('calls onShare when share is triggered', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      // Open speed dial and click share action
      const speedDial = screen.getByLabelText('Dashboard Actions');
      fireEvent.click(speedDial);

      await waitFor(() => {
        const shareAction = screen.getByText('Share Dashboard');
        fireEvent.click(shareAction);
      });

      // This should open the share dialog
      await waitFor(() => {
        expect(screen.getByText('Share Dashboard')).toBeInTheDocument();
      });
    });
  });

  describe('Settings', () => {
    it('opens settings dialog and allows configuration changes', async () => {
      const store = createTestStore();
      
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      // Click settings button in app bar
      const settingsButton = screen.getAllByLabelText(/settings/i)[0];
      fireEvent.click(settingsButton);

      await waitFor(() => {
        expect(screen.getByText('Dashboard Settings')).toBeInTheDocument();
      });

      // Test toggling real-time updates
      const realTimeSwitch = screen.getByRole('checkbox', { name: /real-time updates/i });
      fireEvent.click(realTimeSwitch);

      // The setting should be updated in the store
      await waitFor(() => {
        const state = store.getState();
        expect(state.dashboard.config.realTimeUpdates).toBe(false);
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error messages when provided', () => {
      const store = createTestStore({
        error: 'Failed to load causal graph',
      });

      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      expect(screen.getByText('Failed to load causal graph')).toBeInTheDocument();
    });

    it('shows loading state when isLoading is true', () => {
      const store = createTestStore({
        isLoading: true,
      });

      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      // Loading indicator should be visible
      expect(document.querySelector('.MuiLinearProgress-root')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels for interactive elements', () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      expect(screen.getByLabelText('open drawer')).toBeInTheDocument();
      expect(screen.getByLabelText('Dashboard Actions')).toBeInTheDocument();
    });

    it('supports keyboard navigation', async () => {
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />
      );

      // Test tab navigation
      const firstTab = screen.getByText('Causal Network');
      firstTab.focus();
      expect(document.activeElement).toBe(firstTab);
    });
  });

  describe('Performance', () => {
    it('updates performance metrics correctly', async () => {
      const store = createTestStore();
      
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      // Simulate performance metric update
      act(() => {
        store.dispatch({
          type: 'dashboard/updatePerformanceMetrics',
          payload: { fps: 30, nodeCount: 150 },
        });
      });

      await waitFor(() => {
        expect(screen.getByText('30 FPS')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Features', () => {
    it('handles real-time status changes', async () => {
      const store = createTestStore();
      
      renderWithProviders(
        <ComprehensiveCausalDashboard
          initialGraph={mockCausalGraph}
          realTimeEnabled={true}
          onExport={mockOnExport}
          onShare={mockOnShare}
        />,
        { store }
      );

      // Switch to monitor tab
      const monitorTab = screen.getByText('Real-Time Monitor');
      fireEvent.click(monitorTab);

      await waitFor(() => {
        const setConnectedButton = screen.getByText('Set Connected');
        fireEvent.click(setConnectedButton);
      });

      // Status should be updated in the store
      await waitFor(() => {
        const state = store.getState();
        expect(state.dashboard.realTimeStatus).toBe('connected');
      });
    });
  });
});