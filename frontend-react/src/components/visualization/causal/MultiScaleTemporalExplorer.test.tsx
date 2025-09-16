import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { MultiScaleTemporalExplorer } from './MultiScaleTemporalExplorer';
import type {
  TemporalVisualizationConfig,
  TemporalEvolutionData, 
  TemporalCausalSnapshot
} from './MultiScaleTemporalExplorer';

// Mock Three.js Canvas component
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="three-canvas">{children}</div>
  ),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  PerspectiveCamera: () => <div data-testid="perspective-camera" />,
  Environment: () => <div data-testid="environment" />,
  Text: ({ children }: { children: React.ReactNode }) => <div data-testid="text">{children}</div>,
  Line: () => <div data-testid="line" />,
  Sphere: () => <div data-testid="sphere" />,
  Box: () => <div data-testid="box" />,
  Trail: () => <div data-testid="trail" />,
}));

// Mock D3
vi.mock('d3', () => ({
  scaleTime: () => ({
    domain: vi.fn().mockReturnThis(),
    range: vi.fn().mockReturnThis(),
    invert: vi.fn((x) => new Date(x * 1000)),
    ticks: vi.fn(() => [new Date('2023-01-01'), new Date('2023-06-01'), new Date('2023-12-01')]),
  }),
  forceSimulation: vi.fn(() => ({
    force: vi.fn().mockReturnThis(),
    on: vi.fn().mockReturnThis(),
    stop: vi.fn(),
  })),
  forceLink: vi.fn(() => ({
    id: vi.fn().mockReturnThis(),
  })),
  forceManyBody: vi.fn(() => ({
    strength: vi.fn().mockReturnThis(),
  })),
  forceCenter: vi.fn(),
  mean: vi.fn((arr: number[]) => arr.reduce((a: number, b: number) => a + b, 0) / arr.length),
  deviation: vi.fn(() => 1),
}));

// Create mock data
const createMockSnapshot = (timestamp: Date, nodeCount: number = 5, edgeCount: number = 8): TemporalCausalSnapshot => ({
  timestamp,
  graph: {
    nodes: Array.from({ length: nodeCount }, (_, i) => ({
      id: `node_${i}`,
      label: `Node ${i}`,
      processType: 'seismic' as const,
      position: { x: (i * 20) % 100, y: (i * 15) % 100 }, // Deterministic positioning
      size: 10 + (i % 10),
      color: '#2196f3',
      metadata: {
        description: `Test node ${i}`,
        dataSource: 'test',
        lastUpdated: timestamp,
        confidence: 0.8 + (i * 0.02),
        spatialScale: 'local' as const,
        temporalScale: 'day' as const,
      },
      properties: {},
    })),
    edges: Array.from({ length: edgeCount }, (_, i) => ({
      id: `edge_${i}`,
      source: `node_${i % nodeCount}`,
      target: `node_${(i + 1) % nodeCount}`,
      strength: (i * 0.1) % 1,
      confidence: [0.7, 0.9] as [number, number],
      temporalLag: (i * 100) % 1000, // Deterministic temporal lag based on index
      edgeType: 'direct' as const,
      metadata: {
        discoveryMethod: 'PC',
        statisticalSignificance: 0.01,
        validationStatus: 'validated' as const,
        geologicalContext: 'test context',
      },
      properties: {},
    })),
    metadata: {
      createdAt: timestamp,
      lastUpdated: timestamp,
      totalNodes: nodeCount,
      totalEdges: edgeCount,
      averageCausalStrength: 0.5,
      spatialExtent: { north: 90, south: -90, east: 180, west: -180 },
      temporalExtent: { start: timestamp, end: timestamp },
    },
  },
  metadata: {
    temporalScale: 'day' as const,
    confidence: 0.85,
    dataQuality: 0.9,
    processingLatency: 100,
  },
});

const createMockTemporalData = (): TemporalEvolutionData => {
  const startDate = new Date('2023-01-01');
  const snapshots = Array.from({ length: 10 }, (_, i) => {
    const timestamp = new Date(startDate.getTime() + i * 24 * 60 * 60 * 1000); // Daily snapshots
    return createMockSnapshot(timestamp);
  });

  return {
    snapshots,
    temporalRange: {
      start: snapshots[0].timestamp,
      end: snapshots[snapshots.length - 1].timestamp,
      resolution: 24 * 60 * 60 * 1000, // Daily resolution
    },
    scaleHierarchy: [
      {
        scale: 'day' as const,
        snapshots,
        aggregationMethod: 'mean' as const,
      },
    ],
  };
};



const defaultConfig: TemporalVisualizationConfig = {
  playbackSpeed: 1.0,
  showConfidenceBands: true,
  showTemporalLags: true,
  showPatternHighlights: true,
  enableSynchronization: true,
  viewMode: '4D',
  temporalFilters: {
    scaleRange: ['second', 'geological'],
    confidenceThreshold: 0.7,
    processTypes: [],
  },
  animationSettings: {
    duration: 2000,
    easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    interpolation: 'cubic',
  },
};

describe('MultiScaleTemporalExplorer', () => {
  let mockTemporalData: TemporalEvolutionData;
  let mockOnTimeChange: ReturnType<typeof vi.fn>;
  let mockOnScaleChange: ReturnType<typeof vi.fn>;
  let mockOnPatternDetected: ReturnType<typeof vi.fn>;
  let mockOnCausalEvolutionAnalysis: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    mockTemporalData = createMockTemporalData();
    mockOnTimeChange = vi.fn();
    mockOnScaleChange = vi.fn();
    mockOnPatternDetected = vi.fn();
    mockOnCausalEvolutionAnalysis = vi.fn();
  });

  it('renders without crashing', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByText('Multi-Scale Temporal Explorer')).toBeInTheDocument();
  });

  it('displays timeline controls', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    // Check for playback controls
    expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
  });

  it('displays temporal scale selector', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByLabelText('Temporal Scale')).toBeInTheDocument();
  });

  it('displays view mode selector', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByLabelText('View Mode')).toBeInTheDocument();
  });

  it('renders timeline visualization', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    // Timeline should be rendered as SVG
    const timeline = screen.getByRole('img', { hidden: true }); // SVG elements have img role
    expect(timeline).toBeInTheDocument();
  });

  it('renders 3D visualization when view mode is 3D or 4D', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        initialConfig={{ ...defaultConfig, viewMode: '3D' }}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByTestId('three-canvas')).toBeInTheDocument();
  });

  it('displays current state information', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByText('Current State')).toBeInTheDocument();
  });

  it('displays visualization settings controls', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByText('Visualization Settings')).toBeInTheDocument();
    expect(screen.getByLabelText('Confidence Bands')).toBeInTheDocument();
    expect(screen.getByLabelText('Temporal Lags')).toBeInTheDocument();
    expect(screen.getByLabelText('Pattern Highlights')).toBeInTheDocument();
  });

  it('handles playback control interactions', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    const playButton = screen.getByRole('button', { name: /play/i });
    fireEvent.click(playButton);

    // Should change to pause button when playing
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument();
    });
  });

  it('handles temporal scale changes', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    const scaleSelect = screen.getByLabelText('Temporal Scale');
    fireEvent.mouseDown(scaleSelect);
    
    const hourOption = screen.getByText('Hour');
    fireEvent.click(hourOption);

    expect(mockOnScaleChange).toHaveBeenCalledWith('hour');
  });

  it('handles view mode changes', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    const viewModeSelect = screen.getByLabelText('View Mode');
    fireEvent.mouseDown(viewModeSelect);
    
    const multiViewOption = screen.getByText('Multi-View');
    fireEvent.click(multiViewOption);

    // Should update the view mode
    await waitFor(() => {
      expect(screen.getByDisplayValue('Multi-View')).toBeInTheDocument();
    });
  });

  it('displays detected patterns', () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByText(/Detected Patterns/)).toBeInTheDocument();
  });

  it('handles configuration changes', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    const confidenceBandsSwitch = screen.getByLabelText('Confidence Bands');
    fireEvent.click(confidenceBandsSwitch);

    // Configuration should be updated
    expect(confidenceBandsSwitch).not.toBeChecked();
  });

  it('handles playback speed changes', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    const speedSlider = screen.getByRole('slider');
    fireEvent.change(speedSlider, { target: { value: '2' } });

    // Speed should be updated
    expect(speedSlider).toHaveValue('2');
  });

  it('displays evolution analysis when available', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    // Wait for analysis to complete
    await waitFor(() => {
      expect(screen.getByText('Evolution Analysis')).toBeInTheDocument();
    });
  });

  it('handles empty temporal data gracefully', () => {
    const emptyData: TemporalEvolutionData = {
      snapshots: [],
      temporalRange: {
        start: new Date(),
        end: new Date(),
        resolution: 1000,
      },
      scaleHierarchy: [],
    };

    render(
      <MultiScaleTemporalExplorer
        temporalData={emptyData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByText('Multi-Scale Temporal Explorer')).toBeInTheDocument();
  });

  it('filters snapshots based on temporal configuration', () => {
    const configWithFilters: Partial<TemporalVisualizationConfig> = {
      temporalFilters: {
        scaleRange: ['hour', 'day'],
        confidenceThreshold: 0.9,
        processTypes: ['seismic'],
      },
    };

    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        initialConfig={configWithFilters}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    // Component should render with filtered data
    expect(screen.getByText('Multi-Scale Temporal Explorer')).toBeInTheDocument();
  });

  it('calls pattern detection callback when patterns are detected', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    // Wait for pattern detection to complete
    await waitFor(() => {
      expect(mockOnPatternDetected).toHaveBeenCalled();
    }, { timeout: 3000 });
  });

  it('calls evolution analysis callback when analysis is complete', async () => {
    render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    // Wait for evolution analysis to complete
    await waitFor(() => {
      expect(mockOnCausalEvolutionAnalysis).toHaveBeenCalled();
    }, { timeout: 3000 });
  });

  it('handles window resize gracefully', () => {
    const { rerender } = render(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        width={800}
        height={600}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    // Resize the component
    rerender(
      <MultiScaleTemporalExplorer
        temporalData={mockTemporalData}
        width={1200}
        height={800}
        onTimeChange={mockOnTimeChange}
        onScaleChange={mockOnScaleChange}
        onPatternDetected={mockOnPatternDetected}
        onCausalEvolutionAnalysis={mockOnCausalEvolutionAnalysis}
      />
    );

    expect(screen.getByText('Multi-Scale Temporal Explorer')).toBeInTheDocument();
  });
});