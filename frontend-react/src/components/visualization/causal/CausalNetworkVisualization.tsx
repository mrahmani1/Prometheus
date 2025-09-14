import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { Box, Paper, Toolbar, IconButton, Tooltip, Typography } from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  CenterFocusStrong,
  // Fullscreen, // Unused
  Settings,
  FilterList,
  Search,
  Timeline,
  Share
} from '@mui/icons-material';
import * as d3 from 'd3';
import { CausalLayoutEngine } from './CausalLayoutEngine';
import { CausalNodeRenderer } from './CausalNodeRenderer';
import { CausalEdgeRenderer } from './CausalEdgeRenderer';
import { CausalFilterPanel } from './CausalFilterPanel';
import { CausalSearchInterface } from './CausalSearchInterface';
import {
  // CausalGraph, // Unused
  CausalNode,
  CausalEdge,
  // CausalLayoutConfig, // Unused
  CausalVisualizationProps,
  // CausalNetworkState, // Unused
  // InteractionMode, // Unused
  // RenderMode // Unused
} from './types';

interface CausalNetworkVisualizationState {
  selectedNodes: string[];
  selectedEdges: string[];
  hoveredNode: string | null;
  hoveredEdge: string | null;
  draggedNode: string | null;
  viewport: {
    x: number;
    y: number;
    scale: number;
  };
  showFilterPanel: boolean;
  showSearchPanel: boolean;
  showSettings: boolean;
}

export const CausalNetworkVisualization: React.FC<CausalVisualizationProps> = ({
  graph,
  width,
  height,
  layoutConfig,
  onNodeClick,
  onEdgeClick,
  onNodeHover,
  onEdgeHover,
  onSelectionChange,
  onLayoutComplete,
  filterConfig,
  searchQuery,
  highlightedPaths = [],
  interactionMode = 'explore',
  // renderMode = 'canvas', // Unused
  showLabels = true,
  showConfidenceIntervals = true,
  showTemporalLags = true,
  // animationEnabled = true // Unused
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const layoutEngineRef = useRef<CausalLayoutEngine | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  const [state, setState] = useState<CausalNetworkVisualizationState>({
    selectedNodes: [],
    selectedEdges: [],
    hoveredNode: null,
    hoveredEdge: null,
    draggedNode: null,
    viewport: { x: 0, y: 0, scale: 1 },
    showFilterPanel: false,
    showSearchPanel: false,
    showSettings: false
  });

  // Filter graph based on filter configuration
  const filteredGraph = useMemo(() => {
    if (!filterConfig) return graph;

    const filteredNodes = graph.nodes.filter(node => {
      // Apply strength filter
      const nodeStrength = graph.edges
        .filter(edge => edge.source === node.id || edge.target === node.id)
        .reduce((sum, edge) => sum + edge.strength, 0) / 
        graph.edges.filter(edge => edge.source === node.id || edge.target === node.id).length;

      if (nodeStrength < filterConfig.strengthRange[0] || nodeStrength > filterConfig.strengthRange[1]) {
        return false;
      }

      // Apply process type filter
      if (filterConfig.processTypes.length > 0 && !filterConfig.processTypes.includes(node.processType)) {
        return false;
      }

      // Apply spatial scale filter
      if (filterConfig.spatialScales.length > 0 && !filterConfig.spatialScales.includes(node.metadata.spatialScale)) {
        return false;
      }

      // Apply temporal scale filter
      if (filterConfig.temporalScales.length > 0 && !filterConfig.temporalScales.includes(node.metadata.temporalScale)) {
        return false;
      }

      // Apply confidence filter
      if (node.metadata.confidence < filterConfig.confidenceThreshold) {
        return false;
      }

      return true;
    });

    const filteredNodeIds = new Set(filteredNodes.map(node => node.id));
    const filteredEdges = graph.edges.filter(edge => {
      // Only include edges between filtered nodes
      if (!filteredNodeIds.has(edge.source) || !filteredNodeIds.has(edge.target)) {
        return false;
      }

      // Apply edge type filter
      if (filterConfig.edgeTypes.length > 0 && !filterConfig.edgeTypes.includes(edge.edgeType)) {
        return false;
      }

      // Apply validation status filter
      if (filterConfig.validationStatus.length > 0 && !filterConfig.validationStatus.includes(edge.metadata.validationStatus)) {
        return false;
      }

      // Apply discovery method filter
      if (filterConfig.discoveryMethods.length > 0 && !filterConfig.discoveryMethods.includes(edge.metadata.discoveryMethod)) {
        return false;
      }

      return true;
    });

    return {
      ...graph,
      nodes: filteredNodes,
      edges: filteredEdges
    };
  }, [graph, filterConfig]);

  // Search functionality
  const searchFilteredGraph = useMemo(() => {
    if (!searchQuery || searchQuery.trim() === '') return filteredGraph;

    const query = searchQuery.toLowerCase();
    const matchingNodes = filteredGraph.nodes.filter(node =>
      node.label.toLowerCase().includes(query) ||
      node.processType.toLowerCase().includes(query) ||
      node.metadata.description.toLowerCase().includes(query)
    );

    const matchingNodeIds = new Set(matchingNodes.map(node => node.id));
    const matchingEdges = filteredGraph.edges.filter(edge =>
      matchingNodeIds.has(edge.source) || matchingNodeIds.has(edge.target)
    );

    return {
      ...filteredGraph,
      nodes: matchingNodes,
      edges: matchingEdges
    };
  }, [filteredGraph, searchQuery]);

  // Initialize layout engine
  useEffect(() => {
    if (!layoutEngineRef.current) {
      layoutEngineRef.current = new CausalLayoutEngine(width, height, layoutConfig);
    } else {
      layoutEngineRef.current.updateDimensions(width, height);
      layoutEngineRef.current.updateConfig(layoutConfig);
    }
  }, [width, height, layoutConfig]);

  // Initialize zoom behavior
  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    
    zoomBehaviorRef.current = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        const { x, y, k } = event.transform;
        setState(prev => ({
          ...prev,
          viewport: { x, y, scale: k }
        }));
      });

    svg.call(zoomBehaviorRef.current);

    return () => {
      svg.on('.zoom', null);
    };
  }, []);

  // Layout computation
  useEffect(() => {
    if (!layoutEngineRef.current || !searchFilteredGraph.nodes.length) return;

    const simulation = layoutEngineRef.current.layoutGraph(
      searchFilteredGraph,
      () => {
        // Force re-render on each tick
        setState(prev => ({ ...prev }));
      },
      () => {
        onLayoutComplete?.();
      }
    );

    return () => {
      simulation.stop();
    };
  }, [searchFilteredGraph, onLayoutComplete]);

  // Node interaction handlers
  const handleNodeClick = useCallback((node: CausalNode) => {
    if (interactionMode === 'select') {
      setState(prev => {
        const isSelected = prev.selectedNodes.includes(node.id);
        const newSelectedNodes = isSelected
          ? prev.selectedNodes.filter(id => id !== node.id)
          : [...prev.selectedNodes, node.id];
        
        onSelectionChange?.(newSelectedNodes, prev.selectedEdges);
        return { ...prev, selectedNodes: newSelectedNodes };
      });
    }
    onNodeClick?.(node);
  }, [interactionMode, onNodeClick, onSelectionChange]);

  const handleNodeHover = useCallback((node: CausalNode | null) => {
    setState(prev => ({ ...prev, hoveredNode: node?.id || null }));
    onNodeHover?.(node);
  }, [onNodeHover]);

  // Edge interaction handlers
  const handleEdgeClick = useCallback((edge: CausalEdge) => {
    if (interactionMode === 'select') {
      setState(prev => {
        const isSelected = prev.selectedEdges.includes(edge.id);
        const newSelectedEdges = isSelected
          ? prev.selectedEdges.filter(id => id !== edge.id)
          : [...prev.selectedEdges, edge.id];
        
        onSelectionChange?.(prev.selectedNodes, newSelectedEdges);
        return { ...prev, selectedEdges: newSelectedEdges };
      });
    }
    onEdgeClick?.(edge);
  }, [interactionMode, onEdgeClick, onSelectionChange]);

  const handleEdgeHover = useCallback((edge: CausalEdge | null) => {
    setState(prev => ({ ...prev, hoveredEdge: edge?.id || null }));
    onEdgeHover?.(edge);
  }, [onEdgeHover]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    if (zoomBehaviorRef.current && svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomBehaviorRef.current.scaleBy, 1.5);
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (zoomBehaviorRef.current && svgRef.current) {
      d3.select(svgRef.current)
        .transition()
        .duration(300)
        .call(zoomBehaviorRef.current.scaleBy, 0.67);
    }
  }, []);

  const handleZoomFit = useCallback(() => {
    if (!zoomBehaviorRef.current || !svgRef.current || !searchFilteredGraph.nodes.length) return;

    const nodes = searchFilteredGraph.nodes;
    const xExtent = d3.extent(nodes, d => d.position?.x || 0) as [number, number];
    const yExtent = d3.extent(nodes, d => d.position?.y || 0) as [number, number];

    const padding = 50;
    const graphWidth = xExtent[1] - xExtent[0] + padding * 2;
    const graphHeight = yExtent[1] - yExtent[0] + padding * 2;

    const scale = Math.min(width / graphWidth, height / graphHeight, 2);
    const centerX = (xExtent[0] + xExtent[1]) / 2;
    const centerY = (yExtent[0] + yExtent[1]) / 2;

    const transform = d3.zoomIdentity
      .translate(width / 2 - centerX * scale, height / 2 - centerY * scale)
      .scale(scale);

    d3.select(svgRef.current)
      .transition()
      .duration(750)
      .call(zoomBehaviorRef.current.transform, transform);
  }, [searchFilteredGraph.nodes, width, height]);

  // Render nodes and edges
  const renderElements = useMemo(() => {
    const nodeMap = new Map(searchFilteredGraph.nodes.map(node => [node.id, node]));
    
    const edges = searchFilteredGraph.edges.map(edge => {
      const sourceNode = nodeMap.get(edge.source);
      const targetNode = nodeMap.get(edge.target);
      
      if (!sourceNode || !targetNode) return null;

      const isHighlighted = highlightedPaths.some(path => path.edges.includes(edge.id));
      
      return (
        <CausalEdgeRenderer
          key={edge.id}
          edge={edge}
          sourceNode={sourceNode}
          targetNode={targetNode}
          isSelected={state.selectedEdges.includes(edge.id)}
          isHovered={state.hoveredEdge === edge.id}
          isHighlighted={isHighlighted}
          showConfidenceIntervals={showConfidenceIntervals}
          showTemporalLags={showTemporalLags}
          scale={state.viewport.scale}
          onClick={handleEdgeClick}
          onMouseEnter={handleEdgeHover}
          onMouseLeave={() => handleEdgeHover(null)}
        />
      );
    }).filter(Boolean);

    const nodes = searchFilteredGraph.nodes.map(node => {
      const isHighlighted = highlightedPaths.some(path => path.nodes.includes(node.id));
      
      return (
        <CausalNodeRenderer
          key={node.id}
          node={node}
          isSelected={state.selectedNodes.includes(node.id)}
          isHovered={state.hoveredNode === node.id}
          isHighlighted={isHighlighted}
          showLabels={showLabels}
          showConfidence={showConfidenceIntervals}
          scale={state.viewport.scale}
          onClick={handleNodeClick}
          onMouseEnter={handleNodeHover}
          onMouseLeave={() => handleNodeHover(null)}
        />
      );
    });

    return { edges, nodes };
  }, [
    searchFilteredGraph,
    state,
    highlightedPaths,
    showLabels,
    showConfidenceIntervals,
    showTemporalLags,
    handleNodeClick,
    handleNodeHover,
    handleEdgeClick,
    handleEdgeHover
  ]);

  return (
    <Paper elevation={2} sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Toolbar */}
      <Toolbar variant="dense" sx={{ minHeight: 48, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexGrow: 1 }}>
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={handleZoomIn}>
              <ZoomIn />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={handleZoomOut}>
              <ZoomOut />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="Fit to View">
            <IconButton size="small" onClick={handleZoomFit}>
              <CenterFocusStrong />
            </IconButton>
          </Tooltip>

          <Box sx={{ mx: 2, minWidth: 100 }}>
            <Typography variant="caption" color="textSecondary">
              Zoom: {(state.viewport.scale * 100).toFixed(0)}%
            </Typography>
          </Box>

          <Tooltip title="Search">
            <IconButton 
              size="small" 
              onClick={() => setState(prev => ({ ...prev, showSearchPanel: !prev.showSearchPanel }))}
              color={state.showSearchPanel ? 'primary' : 'default'}
            >
              <Search />
            </IconButton>
          </Tooltip>

          <Tooltip title="Filter">
            <IconButton 
              size="small" 
              onClick={() => setState(prev => ({ ...prev, showFilterPanel: !prev.showFilterPanel }))}
              color={state.showFilterPanel ? 'primary' : 'default'}
            >
              <FilterList />
            </IconButton>
          </Tooltip>

          <Tooltip title="Timeline">
            <IconButton size="small">
              <Timeline />
            </IconButton>
          </Tooltip>

          <Tooltip title="Settings">
            <IconButton 
              size="small"
              onClick={() => setState(prev => ({ ...prev, showSettings: !prev.showSettings }))}
            >
              <Settings />
            </IconButton>
          </Tooltip>

          <Tooltip title="Share">
            <IconButton size="small">
              <Share />
            </IconButton>
          </Tooltip>
        </Box>

        <Typography variant="body2" color="textSecondary">
          {searchFilteredGraph.nodes.length} nodes, {searchFilteredGraph.edges.length} edges
        </Typography>
      </Toolbar>

      {/* Main visualization area */}
      <Box 
        ref={containerRef}
        sx={{ 
          position: 'relative', 
          width: '100%', 
          height: 'calc(100% - 48px)',
          overflow: 'hidden'
        }}
      >
        <svg
          ref={svgRef}
          width={width}
          height={height}
          style={{ display: 'block', cursor: interactionMode === 'explore' ? 'grab' : 'crosshair' }}
        >
          {/* Definitions for markers and patterns */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill="#666"
              />
            </marker>
          </defs>

          {/* Main visualization group with transform */}
          <g transform={`translate(${state.viewport.x}, ${state.viewport.y}) scale(${state.viewport.scale})`}>
            {/* Render edges first (behind nodes) */}
            {renderElements.edges}
            
            {/* Render nodes on top */}
            {renderElements.nodes}
          </g>
        </svg>

        {/* Filter panel */}
        {state.showFilterPanel && (
          <CausalFilterPanel
            filterConfig={filterConfig}
            onFilterChange={() => {}} // TODO: Implement filter change handler
            onClose={() => setState(prev => ({ ...prev, showFilterPanel: false }))}
          />
        )}

        {/* Search panel */}
        {state.showSearchPanel && (
          <CausalSearchInterface
            searchQuery={searchQuery}
            onSearchChange={() => {}} // TODO: Implement search change handler
            onClose={() => setState(prev => ({ ...prev, showSearchPanel: false }))}
          />
        )}
      </Box>
    </Paper>
  );
};