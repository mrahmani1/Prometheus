/**
 * Causal Network Visualization Component for Collaborative Analysis
 * 
 * Interactive visualization of causal relationships with real-time collaboration
 * features including multi-user cursors, live editing, and conflict detection.
 * 
 * Author: Prometheus AI Platform
 * Date: 2025-01-14
 */

import { useRef, useEffect, useState, forwardRef, useImperativeHandle } from 'react';
import { Box, IconButton, Tooltip, Menu, MenuItem, Fab } from '@mui/material';
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  FilterList as FilterIcon,
  Add as AddIcon,
  Comment as CommentIcon
} from '@mui/icons-material';
import * as d3 from 'd3';

interface CausalNode {
  id: string;
  label: string;
  type: string;
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface CausalEdge {
  id: string;
  source: string | CausalNode;
  target: string | CausalNode;
  strength: number;
  confidence: number;
  type: string;
}

interface CausalNetworkVisualizationProps {
  sessionId: string;
  onNodeSelect?: (nodeId: string) => void;
  onEdgeSelect?: (edgeId: string) => void;
  onCreateAnnotation?: (relationshipId: string, content: string, type: string) => void;
  onSendMessage?: (message: any) => void;
  readOnly?: boolean;
}

export interface CausalNetworkVisualizationRef {
  applyOperation: (operation: any) => void;
  centerView: () => void;
  zoomToFit: () => void;
  updateCursor: (userId: string, position: { x: number; y: number; user: string }) => void;
}

export const CausalNetworkVisualization = forwardRef<
  CausalNetworkVisualizationRef,
  CausalNetworkVisualizationProps
>(({ onNodeSelect, onEdgeSelect, onCreateAnnotation, onSendMessage, readOnly = false }, ref) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState<CausalNode[]>([]);
  const [edges, setEdges] = useState<CausalEdge[]>([]);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; target: any } | null>(null);
  const [userCursors, setUserCursors] = useState<Record<string, { x: number; y: number; user: string }>>({});

  // D3 simulation and zoom
  const simulationRef = useRef<d3.Simulation<CausalNode, CausalEdge> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const transformRef = useRef<d3.ZoomTransform>(d3.zoomIdentity);

  // Initialize visualization
  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Clear previous content
    svg.selectAll('*').remove();

    // Create main group for zooming/panning
    const g = svg.append('g').attr('class', 'main-group');

    // Create groups for different elements
    g.append('g').attr('class', 'edges');
    g.append('g').attr('class', 'nodes');
    g.append('g').attr('class', 'labels');
    g.append('g').attr('class', 'cursors');

    // Setup zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        transformRef.current = event.transform;
        g.attr('transform', event.transform);

        // Send cursor position to other users
        if (onSendMessage && !readOnly) {
          const [mouseX, mouseY] = d3.pointer(event, svg.node());
          onSendMessage({
            type: 'cursor_position',
            data: {
              x: mouseX,
              y: mouseY,
              transform: event.transform
            }
          });
        }
      });

    svg.call(zoom);
    zoomRef.current = zoom;

    // Create force simulation
    const simulation = d3.forceSimulation<CausalNode>(nodes)
      .force('link', d3.forceLink<CausalNode, CausalEdge>(edges).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    simulationRef.current = simulation;

    // Update visualization when data changes
    updateVisualization();

    // Handle resize
    const handleResize = () => {
      const newWidth = container.clientWidth;
      const newHeight = container.clientHeight;
      svg.attr('width', newWidth).attr('height', newHeight);
      simulation.force('center', d3.forceCenter(newWidth / 2, newHeight / 2));
      simulation.alpha(0.3).restart();
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      simulation.stop();
    };
  }, []);

  // Update visualization when nodes or edges change
  useEffect(() => {
    updateVisualization();
  }, [nodes, edges]);

  // Update user cursors
  useEffect(() => {
    updateUserCursors();
  }, [userCursors]);

  // Handle cursor updates from other users
  const handleCursorUpdate = (userId: string, position: { x: number; y: number; user: string }) => {
    setUserCursors(prev => ({
      ...prev,
      [userId]: position
    }));
  };

  const updateVisualization = () => {
    if (!svgRef.current || !simulationRef.current) return;

    const svg = d3.select(svgRef.current);
    const simulation = simulationRef.current;

    // Update edges
    const edgeSelection = svg.select('.edges')
      .selectAll<SVGLineElement, CausalEdge>('.edge')
      .data(edges, (d: CausalEdge) => d.id);

    const edgeEnter = edgeSelection.enter()
      .append('line')
      .attr('class', 'edge')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: CausalEdge) => Math.sqrt(d.strength * 10))
      .style('cursor', readOnly ? 'default' : 'pointer')
      .on('click', (event, d) => {
        if (readOnly) return;
        event.stopPropagation();
        setSelectedEdge(d.id);
        onEdgeSelect?.(d.id);
      })
      .on('contextmenu', (event, d) => {
        if (readOnly) return;
        event.preventDefault();
        setContextMenu({
          x: event.clientX,
          y: event.clientY,
          target: { type: 'edge', data: d }
        });
      });

    const edgeMerged = edgeSelection.merge(edgeEnter);
    edgeMerged
      .attr('stroke', (d: CausalEdge) => selectedEdge === d.id ? '#ff6b6b' : '#999')
      .attr('stroke-width', (d: CausalEdge) => {
        const baseWidth = Math.sqrt(d.strength * 10);
        return selectedEdge === d.id ? baseWidth + 2 : baseWidth;
      });

    edgeSelection.exit().remove();

    // Update nodes
    const nodeSelection = svg.select('.nodes')
      .selectAll<SVGCircleElement, CausalNode>('.node')
      .data(nodes, (d: CausalNode) => d.id);

    const nodeEnter = nodeSelection.enter()
      .append('circle')
      .attr('class', 'node')
      .attr('r', 20)
      .attr('fill', (d: CausalNode) => getNodeColor(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', readOnly ? 'default' : 'pointer')
      .on('click', (event, d) => {
        if (readOnly) return;
        event.stopPropagation();
        setSelectedNode(d.id);
        onNodeSelect?.(d.id);
      })
      .on('contextmenu', (event, d) => {
        if (readOnly) return;
        event.preventDefault();
        setContextMenu({
          x: event.clientX,
          y: event.clientY,
          target: { type: 'node', data: d }
        });
      });

    if (!readOnly) {
      nodeEnter.call(d3.drag<SVGCircleElement, CausalNode>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;

          // Send drag operation to other users
          onSendMessage?.({
            type: 'edit_operation',
            data: {
              operation_type: 'move',
              target_id: d.id,
              target_type: 'node',
              position: { x: event.x, y: event.y }
            }
          });
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      );
    }

    const nodeMerged = nodeSelection.merge(nodeEnter);
    nodeMerged
      .attr('fill', (d: CausalNode) => selectedNode === d.id ? '#ff6b6b' : getNodeColor(d.type))
      .attr('stroke-width', (d: CausalNode) => selectedNode === d.id ? 4 : 2);

    nodeSelection.exit().remove();

    // Update labels
    const labelSelection = svg.select('.labels')
      .selectAll<SVGTextElement, CausalNode>('.label')
      .data(nodes, (d: CausalNode) => d.id);

    const labelEnter = labelSelection.enter()
      .append('text')
      .attr('class', 'label')
      .attr('text-anchor', 'middle')
      .attr('dy', '.35em')
      .attr('font-size', '12px')
      .attr('font-family', 'Arial, sans-serif')
      .attr('fill', '#333')
      .style('pointer-events', 'none')
      .text((d: CausalNode) => d.label);

    const labelMerged = labelSelection.merge(labelEnter);
    labelMerged.text((d: CausalNode) => d.label);

    labelSelection.exit().remove();

    // Update simulation
    simulation.nodes(nodes);
    simulation.force('link', d3.forceLink<CausalNode, CausalEdge>(edges).id(d => d.id).distance(100));

    simulation.on('tick', () => {
      svg.selectAll<SVGLineElement, CausalEdge>('.edge')
        .attr('x1', (d: CausalEdge) => (d.source as CausalNode).x || 0)
        .attr('y1', (d: CausalEdge) => (d.source as CausalNode).y || 0)
        .attr('x2', (d: CausalEdge) => (d.target as CausalNode).x || 0)
        .attr('y2', (d: CausalEdge) => (d.target as CausalNode).y || 0);

      svg.selectAll<SVGCircleElement, CausalNode>('.node')
        .attr('cx', (d: CausalNode) => d.x || 0)
        .attr('cy', (d: CausalNode) => d.y || 0);

      svg.selectAll<SVGTextElement, CausalNode>('.label')
        .attr('x', (d: CausalNode) => d.x || 0)
        .attr('y', (d: CausalNode) => d.y || 0);
    });

    simulation.alpha(0.3).restart();
  };

  const updateUserCursors = () => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    type CursorData = [string, { x: number; y: number; user: string }];

    const cursorSelection = svg.select('.cursors')
      .selectAll<SVGGElement, CursorData>('.user-cursor')
      .data(Object.entries(userCursors), (d: CursorData) => d[0]);

    const cursorEnter = cursorSelection.enter()
      .append('g')
      .attr('class', 'user-cursor');

    cursorEnter.append('circle')
      .attr('r', 8)
      .attr('fill', (d: CursorData) => getUserColor(d[0]))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    cursorEnter.append('text')
      .attr('x', 12)
      .attr('y', 4)
      .attr('font-size', '10px')
      .attr('fill', '#333')
      .text((d: CursorData) => d[1].user);

    const cursorMerged = cursorSelection.merge(cursorEnter);
    cursorMerged.attr('transform', (d: CursorData) => `translate(${d[1].x}, ${d[1].y})`);

    cursorSelection.exit().remove();
  };

  const getNodeColor = (type: string): string => {
    const colors: Record<string, string> = {
      'seismic': '#ff6b6b',
      'volcanic': '#ffa726',
      'geochemical': '#66bb6a',
      'deformation': '#42a5f5',
      'thermal': '#ef5350',
      'default': '#9e9e9e'
    };
    return colors[type] || colors.default;
  };

  const getUserColor = (userId: string): string => {
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd'];
    const hash = userId.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    return colors[Math.abs(hash) % colors.length];
  };

  const centerView = () => {
    if (!svgRef.current || !zoomRef.current) return;

    const svg = d3.select(svgRef.current);
    const mainGroup = svg.select('.main-group').node() as SVGGElement;
    const bounds = mainGroup?.getBBox();

    if (bounds) {
      const width = containerRef.current?.clientWidth || 800;
      const height = containerRef.current?.clientHeight || 600;
      const scale = Math.min(width / bounds.width, height / bounds.height) * 0.8;
      const translate = [
        width / 2 - scale * (bounds.x + bounds.width / 2),
        height / 2 - scale * (bounds.y + bounds.height / 2)
      ];

      svg.transition()
        .duration(750)
        .call(zoomRef.current.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
    }
  };

  const zoomIn = () => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().call(zoomRef.current.scaleBy, 1.5);
  };

  const zoomOut = () => {
    if (!svgRef.current || !zoomRef.current) return;
    d3.select(svgRef.current).transition().call(zoomRef.current.scaleBy, 1 / 1.5);
  };

  const applyOperation = (operation: any) => {
    switch (operation.operation_type) {
      case 'move':
        if (operation.target_type === 'node') {
          setNodes(prev => prev.map(node =>
            node.id === operation.target_id
              ? { ...node, x: operation.position.x, y: operation.position.y }
              : node
          ));
        }
        break;
      case 'add':
        if (operation.target_type === 'node') {
          setNodes(prev => [...prev, operation.data]);
        } else if (operation.target_type === 'edge') {
          setEdges(prev => [...prev, operation.data]);
        }
        break;
      case 'delete':
        if (operation.target_type === 'node') {
          setNodes(prev => prev.filter(node => node.id !== operation.target_id));
        } else if (operation.target_type === 'edge') {
          setEdges(prev => prev.filter(edge => edge.id !== operation.target_id));
        }
        break;
    }
  };

  // Expose methods through ref
  useImperativeHandle(ref, () => ({
    applyOperation,
    centerView,
    zoomToFit: centerView,
    updateCursor: handleCursorUpdate
  }));

  const handleContextMenuAction = (action: string) => {
    if (!contextMenu) return;

    const { target } = contextMenu;

    switch (action) {
      case 'annotate':
        if (target.type === 'edge') {
          onCreateAnnotation?.(target.data.id, '', 'comment');
        }
        break;
      case 'delete':
        onSendMessage?.({
          type: 'edit_operation',
          data: {
            operation_type: 'delete',
            target_id: target.data.id,
            target_type: target.type
          }
        });
        break;
    }

    setContextMenu(null);
  };

  return (
    <Box ref={containerRef} sx={{ position: 'relative', width: '100%', height: '100%' }}>
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        style={{ background: '#f5f5f5' }}
        onClick={() => {
          setSelectedNode(null);
          setSelectedEdge(null);
          setContextMenu(null);
        }}
      />

      {/* Controls */}
      <Box sx={{ position: 'absolute', top: 16, right: 16, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Tooltip title="Zoom In">
          <IconButton onClick={zoomIn} sx={{ bgcolor: 'white' }}>
            <ZoomInIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Zoom Out">
          <IconButton onClick={zoomOut} sx={{ bgcolor: 'white' }}>
            <ZoomOutIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Center View">
          <IconButton onClick={centerView} sx={{ bgcolor: 'white' }}>
            <CenterIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Filter">
          <IconButton sx={{ bgcolor: 'white' }}>
            <FilterIcon />
          </IconButton>
        </Tooltip>
      </Box>

      {/* Add Node FAB */}
      {!readOnly && (
        <Fab
          color="primary"
          sx={{ position: 'absolute', bottom: 16, right: 16 }}
          onClick={() => {
            // Add new node at center
            const newNode: CausalNode = {
              id: `node_${Date.now()}`,
              label: 'New Node',
              type: 'default',
              x: (containerRef.current?.clientWidth || 800) / 2,
              y: (containerRef.current?.clientHeight || 600) / 2
            };

            onSendMessage?.({
              type: 'edit_operation',
              data: {
                operation_type: 'add',
                target_type: 'node',
                data: newNode
              }
            });
          }}
        >
          <AddIcon />
        </Fab>
      )}

      {/* Context Menu */}
      <Menu
        open={Boolean(contextMenu)}
        onClose={() => setContextMenu(null)}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu ? { top: contextMenu.y, left: contextMenu.x } : undefined
        }
      >
        <MenuItem onClick={() => handleContextMenuAction('annotate')}>
          <CommentIcon sx={{ mr: 1 }} />
          Add Annotation
        </MenuItem>
        {!readOnly && (
          <MenuItem onClick={() => handleContextMenuAction('delete')}>
            Delete
          </MenuItem>
        )}
      </Menu>
    </Box>
  );
});

CausalNetworkVisualization.displayName = 'CausalNetworkVisualization';

export default CausalNetworkVisualization;