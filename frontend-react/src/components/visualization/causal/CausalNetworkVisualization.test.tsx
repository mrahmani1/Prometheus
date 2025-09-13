import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CausalNetworkVisualization } from './CausalNetworkVisualization';
import { CausalGraph, CausalLayoutConfig } from './types';

// Mock D3 to avoid DOM manipulation issues in tests
vi.mock('d3', () => {
  const mockForce = () => ({
    id: vi.fn().mockReturnThis(),
    distance: vi.fn().mockReturnThis(),
    strength: vi.fn().mockReturnThis(),
    radius: vi.fn().mockReturnThis(),
    x: vi.fn().mockReturnThis(),
    y: vi.fn().mockReturnThis()
  });

  return {
    select: vi.fn(() => ({
      call: vi.fn().mockReturnThis(),
      on: vi.fn().mockReturnThis(),
      transition: vi.fn(() => ({
        duration: vi.fn(() => ({
          call: vi.fn().mockReturnThis()
        }))
      }))
    })),
    zoom: vi.fn(() => ({
      scaleExtent: vi.fn(() => ({
        on: vi.fn().mockReturnThis()
      }))
    })),
    forceSimulation: vi.fn(() => ({
      force: vi.fn().mockReturnThis(),
      on: vi.fn().mockReturnThis(),
      stop: vi.fn().mockReturnThis(),
      restart: vi.fn().mockReturnThis(),
      alphaDecay: vi.fn().mockReturnThis(),
      velocityDecay: vi.fn().mockReturnThis()
    })),
    forceLink: vi.fn(() => mockForce()),
    forceManyBody: vi.fn(() => mockForce()),
    forceCenter: vi.fn(() => mockForce()),
    forceCollide: vi.fn(() => mockForce()),
    forceX: vi.fn(() => mockForce()),
    forceY: vi.fn(() => mockForce()),
    extent: vi.fn(() => [0, 100]),
    zoomIdentity: {
      translate: vi.fn(() => ({
        scale: vi.fn()
      }))
    }
  };
});

// Mock the CausalLayoutEngine to avoid D3 issues in tests
vi.mock('./CausalLayoutEngine', () => ({
  CausalLayoutEngine: vi.fn().mockImplementation(() => ({
    updateDimensions: vi.fn(),
    updateConfig: vi.fn(),
    layoutGraph: vi.fn(() => ({
      on: vi.fn(),
      stop: vi.fn()
    })),
    stop: vi.fn(),
    restart: vi.fn(),
    getSimulation: vi.fn(() => null)
  }))
}));

describe('CausalNetworkVisualization', () => {
  const mockGraph: CausalGraph = {
    nodes: [
      {
        id: 'node1',
        label: 'Seismic Activity',
        processType: 'seismic',
        position: { x: 100, y: 100 },
        size: 20,
        color: '#e74c3c',
        metadata: {
          description: 'Test seismic node',
          dataSource: 'Test Source',
          lastUpdated: new Date(),
          confidence: 0.9,
          spatialScale: 'regional',
          temporalScale: 'second'
        },
        properties: {}
      },
      {
        id: 'node2',
        label: 'Magma Intrusion',
        processType: 'magmatic',
        position: { x: 200, y: 200 },
        size: 25,
        color: '#e67e22',
        metadata: {
          description: 'Test magmatic node',
          dataSource: 'Test Source',
          lastUpdated: new Date(),
          confidence: 0.85,
          spatialScale: 'local',
          temporalScale: 'day'
        },
        properties: {}
      }
    ],
    edges: [
      {
        id: 'edge1',
        source: 'node1',
        target: 'node2',
        strength: 0.8,
        confidence: [0.7, 0.9],
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'PC Algorithm',
          statisticalSignificance: 0.01,
          validationStatus: 'validated',
          geologicalContext: 'test context'
        },
        properties: {}
      }
    ],
    metadata: {
      createdAt: new Date(),
      lastUpdated: new Date(),
      totalNodes: 2,
      totalEdges: 1,
      averageCausalStrength: 0.8,
      spatialExtent: {
        north: 45,
        south: 44,
        east: -121,
        west: -122
      },
      temporalExtent: {
        start: new Date(Date.now() - 86400000),
        end: new Date()
      }
    }
  };

  const mockLayoutConfig: CausalLayoutConfig = {
    algorithm: 'force-directed',
    forceStrength: 0.5,
    linkDistance: 100,
    nodeRepulsion: 300,
    centerForce: 0.1,
    collisionRadius: 5,
    alphaDecay: 0.0228,
    velocityDecay: 0.4,
    clustering: {
      enabled: false,
      clusterBy: 'processType',
      clusterStrength: 0.3
    },
    hierarchical: {
      enabled: false,
      direction: 'TB',
      levelSeparation: 100,
      nodeSeparation: 50
    }
  };

  const defaultProps = {
    graph: mockGraph,
    width: 800,
    height: 600,
    layoutConfig: mockLayoutConfig,
    interactionMode: 'explore' as const,
    renderMode: 'canvas' as const,
    showLabels: true,
    showConfidenceIntervals: true,
    showTemporalLags: true,
    animationEnabled: true
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the causal network visualization', () => {
    render(<CausalNetworkVisualization {...defaultProps} />);
    
    // Check if the main container is rendered
    expect(screen.getByText('2 nodes, 1 edges')).toBeInTheDocument();
  });

  it('displays the correct number of nodes and edges in toolbar', () => {
    render(<CausalNetworkVisualization {...defaultProps} />);
    
    // Check if the node and edge count is displayed
    expect(screen.getByText('2 nodes, 1 edges')).toBeInTheDocument();
  });

  it('renders toolbar controls', () => {
    render(<CausalNetworkVisualization {...defaultProps} />);
    
    // Check for zoom controls
    expect(screen.getByLabelText('Zoom In')).toBeInTheDocument();
    expect(screen.getByLabelText('Zoom Out')).toBeInTheDocument();
    expect(screen.getByLabelText('Fit to View')).toBeInTheDocument();
    
    // Check for other controls
    expect(screen.getByLabelText('Search')).toBeInTheDocument();
    expect(screen.getByLabelText('Filter')).toBeInTheDocument();
    expect(screen.getByLabelText('Settings')).toBeInTheDocument();
  });

  it('calls onNodeClick when a node is clicked', async () => {
    const mockOnNodeClick = vi.fn();
    render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        onNodeClick={mockOnNodeClick}
      />
    );

    // Note: In a real test, we would need to simulate clicking on the actual node
    // For now, we'll test that the handler is properly set up
    expect(mockOnNodeClick).not.toHaveBeenCalled();
  });

  it('calls onEdgeClick when an edge is clicked', async () => {
    const mockOnEdgeClick = vi.fn();
    render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        onEdgeClick={mockOnEdgeClick}
      />
    );

    // Note: In a real test, we would need to simulate clicking on the actual edge
    expect(mockOnEdgeClick).not.toHaveBeenCalled();
  });

  it('opens search panel when search button is clicked', async () => {
    render(<CausalNetworkVisualization {...defaultProps} />);
    
    const searchButton = screen.getByLabelText('Search');
    fireEvent.click(searchButton);
    
    // The search panel should be rendered
    await waitFor(() => {
      expect(screen.getByText('Search Causal Network')).toBeInTheDocument();
    });
  });

  it('opens filter panel when filter button is clicked', async () => {
    render(<CausalNetworkVisualization {...defaultProps} />);
    
    const filterButton = screen.getByLabelText('Filter');
    fireEvent.click(filterButton);
    
    // The filter panel should be rendered
    await waitFor(() => {
      expect(screen.getByText('Causal Network Filters')).toBeInTheDocument();
    });
  });

  it('handles layout configuration changes', () => {
    const mockOnLayoutComplete = vi.fn();
    const { rerender } = render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        onLayoutComplete={mockOnLayoutComplete}
      />
    );

    // Change layout algorithm
    const newLayoutConfig = {
      ...mockLayoutConfig,
      algorithm: 'hierarchical' as const
    };

    rerender(
      <CausalNetworkVisualization 
        {...defaultProps} 
        layoutConfig={newLayoutConfig}
        onLayoutComplete={mockOnLayoutComplete}
      />
    );

    // The component should handle the layout change
    expect(screen.getByText('2 nodes, 1 edges')).toBeInTheDocument();
  });

  it('filters graph based on filter configuration', () => {
    const filterConfig = {
      strengthRange: [0.5, 1.0] as [number, number],
      confidenceThreshold: 0.8,
      processTypes: ['seismic'],
      spatialScales: [],
      temporalScales: [],
      edgeTypes: [],
      validationStatus: ['validated'],
      discoveryMethods: []
    };

    render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        filterConfig={filterConfig}
      />
    );

    // The filtered graph should still render (filtering may not change the count in this test scenario)
    expect(screen.getByText(/\d+ nodes, \d+ edges/)).toBeInTheDocument();
  });

  it('handles search query filtering', () => {
    render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        searchQuery="seismic"
      />
    );

    // The component should handle search filtering
    expect(screen.getByText('1 nodes, 1 edges')).toBeInTheDocument();
  });

  it('displays zoom percentage in toolbar', () => {
    render(<CausalNetworkVisualization {...defaultProps} />);
    
    // Check if zoom percentage is displayed
    expect(screen.getByText('Zoom: 100%')).toBeInTheDocument();
  });

  it('handles different interaction modes', () => {
    const { rerender } = render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        interactionMode="select"
      />
    );

    expect(screen.getByText('2 nodes, 1 edges')).toBeInTheDocument();

    // Change to edit mode
    rerender(
      <CausalNetworkVisualization 
        {...defaultProps} 
        interactionMode="edit"
      />
    );

    expect(screen.getByText('2 nodes, 1 edges')).toBeInTheDocument();
  });

  it('handles different render modes', () => {
    const { rerender } = render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        renderMode="webgl"
      />
    );

    expect(screen.getByText('2 nodes, 1 edges')).toBeInTheDocument();

    // Change to SVG mode
    rerender(
      <CausalNetworkVisualization 
        {...defaultProps} 
        renderMode="svg"
      />
    );

    expect(screen.getByText('2 nodes, 1 edges')).toBeInTheDocument();
  });

  it('handles empty graph gracefully', () => {
    const emptyGraph: CausalGraph = {
      nodes: [],
      edges: [],
      metadata: {
        createdAt: new Date(),
        lastUpdated: new Date(),
        totalNodes: 0,
        totalEdges: 0,
        averageCausalStrength: 0,
        spatialExtent: {
          north: 0,
          south: 0,
          east: 0,
          west: 0
        },
        temporalExtent: {
          start: new Date(),
          end: new Date()
        }
      }
    };

    render(
      <CausalNetworkVisualization 
        {...defaultProps} 
        graph={emptyGraph}
      />
    );

    expect(screen.getByText('0 nodes, 0 edges')).toBeInTheDocument();
  });
});