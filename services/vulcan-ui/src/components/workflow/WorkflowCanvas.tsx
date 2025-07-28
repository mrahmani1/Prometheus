import React, { useRef, useCallback, useState, useEffect } from 'react';
import { Box, Paper } from '@mui/material';
import { WorkflowNode, WorkflowConnection } from '@/types/workflow';
import { WorkflowNodeComponent } from './WorkflowNodeComponent';
import { WorkflowConnectionComponent } from './WorkflowConnectionComponent';
// import { generateId } from '@/utils/idGenerator';

interface WorkflowCanvasProps {
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  selectedNode: WorkflowNode | null;
  onNodeSelect: (nodeId: string | null) => void;
  onNodeUpdate: (nodeId: string, updates: Partial<WorkflowNode>) => void;
  onNodeDelete: (nodeId: string) => void;
  onConnectionAdd: (connection: Omit<WorkflowConnection, 'id'>) => void;
  onConnectionDelete: (connectionId: string) => void;
}

interface DragState {
  isDragging: boolean;
  dragType: 'node' | 'canvas' | 'connection' | null;
  dragData: any;
  startPosition: { x: number; y: number };
  currentPosition: { x: number; y: number };
}

interface ConnectionDragState {
  isConnecting: boolean;
  sourceNodeId: string | null;
  sourcePortId: string | null;
  targetPosition: { x: number; y: number };
}

export const WorkflowCanvas: React.FC<WorkflowCanvasProps> = ({
  nodes,
  connections,
  selectedNode,
  onNodeSelect,
  onNodeUpdate,
  onNodeDelete,
  onConnectionAdd,
  onConnectionDelete
}) => {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [dragState, setDragState] = useState<DragState>({
    isDragging: false,
    dragType: null,
    dragData: null,
    startPosition: { x: 0, y: 0 },
    currentPosition: { x: 0, y: 0 }
  });
  const [connectionDrag, setConnectionDrag] = useState<ConnectionDragState>({
    isConnecting: false,
    sourceNodeId: null,
    sourcePortId: null,
    targetPosition: { x: 0, y: 0 }
  });

  // Handle canvas drag and drop
  const handleCanvasDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  }, []);

  const handleCanvasDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    
    try {
      const data = JSON.parse(event.dataTransfer.getData('application/json'));
      
      if (data.type === 'workflow-node') {
        const canvasRect = canvasRef.current?.getBoundingClientRect();
        if (!canvasRect) return;
        
        const position = {
          x: (event.clientX - canvasRect.left - canvasOffset.x) / zoom,
          y: (event.clientY - canvasRect.top - canvasOffset.y) / zoom
        };
        
        // This will be handled by the parent component
        // We need to emit an event or use a callback
        const addNodeEvent = new CustomEvent('addWorkflowNode', {
          detail: { nodeType: data.nodeType, position }
        });
        window.dispatchEvent(addNodeEvent);
      }
    } catch (error) {
      console.error('Error handling canvas drop:', error);
    }
  }, [canvasOffset, zoom]);

  // Handle canvas click (deselect nodes)
  const handleCanvasClick = useCallback((event: React.MouseEvent) => {
    if (event.target === event.currentTarget) {
      onNodeSelect(null);
    }
  }, [onNodeSelect]);

  // Handle node drag
  const handleNodeDragStart = useCallback((nodeId: string, startPosition: { x: number; y: number }) => {
    setDragState({
      isDragging: true,
      dragType: 'node',
      dragData: { nodeId },
      startPosition,
      currentPosition: startPosition
    });
  }, []);

  const handleNodeDrag = useCallback((position: { x: number; y: number }) => {
    if (dragState.isDragging && dragState.dragType === 'node') {
      setDragState(prev => ({
        ...prev,
        currentPosition: position
      }));
    }
  }, [dragState.isDragging, dragState.dragType]);

  const handleNodeDragEnd = useCallback((position: { x: number; y: number }) => {
    if (dragState.isDragging && dragState.dragType === 'node' && dragState.dragData) {
      const { nodeId } = dragState.dragData;
      onNodeUpdate(nodeId, { position });
    }
    
    setDragState({
      isDragging: false,
      dragType: null,
      dragData: null,
      startPosition: { x: 0, y: 0 },
      currentPosition: { x: 0, y: 0 }
    });
  }, [dragState, onNodeUpdate]);

  // Handle connection creation
  const handleConnectionStart = useCallback((nodeId: string, portId: string, position: { x: number; y: number }) => {
    setConnectionDrag({
      isConnecting: true,
      sourceNodeId: nodeId,
      sourcePortId: portId,
      targetPosition: position
    });
  }, []);

  const handleConnectionDrag = useCallback((position: { x: number; y: number }) => {
    if (connectionDrag.isConnecting) {
      setConnectionDrag(prev => ({
        ...prev,
        targetPosition: position
      }));
    }
  }, [connectionDrag.isConnecting]);

  const handleConnectionEnd = useCallback((targetNodeId?: string, targetPortId?: string) => {
    if (connectionDrag.isConnecting && connectionDrag.sourceNodeId && connectionDrag.sourcePortId) {
      if (targetNodeId && targetPortId) {
        // Create connection
        const sourceNode = nodes.find(n => n.id === connectionDrag.sourceNodeId);
        const targetNode = nodes.find(n => n.id === targetNodeId);
        
        if (sourceNode && targetNode) {
          const sourcePort = sourceNode.outputs.find(p => p.id === connectionDrag.sourcePortId);
          const targetPort = targetNode.inputs.find(p => p.id === targetPortId);
          
          if (sourcePort && targetPort) {
            onConnectionAdd({
              sourceNodeId: connectionDrag.sourceNodeId,
              sourcePortId: connectionDrag.sourcePortId,
              targetNodeId,
              targetPortId,
              dataType: sourcePort.type
            });
          }
        }
      }
    }
    
    setConnectionDrag({
      isConnecting: false,
      sourceNodeId: null,
      sourcePortId: null,
      targetPosition: { x: 0, y: 0 }
    });
  }, [connectionDrag, nodes, onConnectionAdd]);

  // Handle mouse events for canvas panning
  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (event.button === 1 || (event.button === 0 && event.ctrlKey)) {
      // Middle mouse button or Ctrl+Left click for panning
      event.preventDefault();
      setDragState({
        isDragging: true,
        dragType: 'canvas',
        dragData: null,
        startPosition: { x: event.clientX, y: event.clientY },
        currentPosition: { x: event.clientX, y: event.clientY }
      });
    }
  }, []);

  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (dragState.isDragging && dragState.dragType === 'canvas') {
      const deltaX = event.clientX - dragState.startPosition.x;
      const deltaY = event.clientY - dragState.startPosition.y;
      
      setCanvasOffset(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY
      }));
      
      setDragState(prev => ({
        ...prev,
        startPosition: { x: event.clientX, y: event.clientY }
      }));
    } else if (connectionDrag.isConnecting) {
      const canvasRect = canvasRef.current?.getBoundingClientRect();
      if (canvasRect) {
        const position = {
          x: (event.clientX - canvasRect.left - canvasOffset.x) / zoom,
          y: (event.clientY - canvasRect.top - canvasOffset.y) / zoom
        };
        handleConnectionDrag(position);
      }
    }
  }, [dragState, connectionDrag, canvasOffset, zoom, handleConnectionDrag]);

  const handleMouseUp = useCallback(() => {
    if (dragState.isDragging && dragState.dragType === 'canvas') {
      setDragState({
        isDragging: false,
        dragType: null,
        dragData: null,
        startPosition: { x: 0, y: 0 },
        currentPosition: { x: 0, y: 0 }
      });
    } else if (connectionDrag.isConnecting) {
      handleConnectionEnd();
    }
  }, [dragState, connectionDrag, handleConnectionEnd]);

  // Handle zoom
  const handleWheel = useCallback((event: React.WheelEvent) => {
    if (event.ctrlKey) {
      event.preventDefault();
      const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
      setZoom(prev => Math.max(0.1, Math.min(3, prev * zoomFactor)));
    }
  }, []);

  // Listen for add node events from drag and drop
  useEffect(() => {
    const handleAddNode = (event: CustomEvent) => {
      const { nodeType, position } = event.detail;
      // This would need to be passed up to the parent component
      console.log('Add node:', nodeType, position);
    };

    window.addEventListener('addWorkflowNode', handleAddNode as EventListener);
    return () => {
      window.removeEventListener('addWorkflowNode', handleAddNode as EventListener);
    };
  }, []);

  return (
    <Box
      ref={canvasRef}
      sx={{
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        cursor: dragState.isDragging && dragState.dragType === 'canvas' ? 'grabbing' : 'default',
        background: 'linear-gradient(90deg, #f5f5f5 1px, transparent 1px), linear-gradient(#f5f5f5 1px, transparent 1px)',
        backgroundSize: `${20 * zoom}px ${20 * zoom}px`,
        backgroundPosition: `${canvasOffset.x}px ${canvasOffset.y}px`
      }}
      onDragOver={handleCanvasDragOver}
      onDrop={handleCanvasDrop}
      onClick={handleCanvasClick}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onWheel={handleWheel}
    >
      {/* Canvas content container */}
      <Box
        sx={{
          position: 'absolute',
          transform: `translate(${canvasOffset.x}px, ${canvasOffset.y}px) scale(${zoom})`,
          transformOrigin: '0 0',
          width: '100%',
          height: '100%'
        }}
      >
        {/* Render connections */}
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            zIndex: 1
          }}
        >
          {connections.map(connection => (
            <WorkflowConnectionComponent
              key={connection.id}
              connection={connection}
              nodes={nodes}
              onDelete={() => onConnectionDelete(connection.id)}
            />
          ))}
          
          {/* Temporary connection during drag */}
          {connectionDrag.isConnecting && connectionDrag.sourceNodeId && connectionDrag.sourcePortId && (
            <WorkflowConnectionComponent
              connection={{
                id: 'temp',
                sourceNodeId: connectionDrag.sourceNodeId,
                sourcePortId: connectionDrag.sourcePortId,
                targetNodeId: 'temp',
                targetPortId: 'temp',
                dataType: 'any'
              }}
              nodes={nodes}
              targetPosition={connectionDrag.targetPosition}
              isTemporary
            />
          )}
        </svg>

        {/* Render nodes */}
        {nodes.map(node => (
          <WorkflowNodeComponent
            key={node.id}
            node={node}
            isSelected={selectedNode?.id === node.id}
            onSelect={() => onNodeSelect(node.id)}
            onUpdate={(updates) => onNodeUpdate(node.id, updates)}
            onDelete={() => onNodeDelete(node.id)}
            onDragStart={(startPos) => handleNodeDragStart(node.id, startPos)}
            onDrag={handleNodeDrag}
            onDragEnd={handleNodeDragEnd}
            onConnectionStart={handleConnectionStart}
            onConnectionEnd={handleConnectionEnd}
            isDragging={dragState.isDragging && dragState.dragData?.nodeId === node.id}
            dragPosition={dragState.isDragging && dragState.dragData?.nodeId === node.id ? dragState.currentPosition : undefined}
          />
        ))}
      </Box>

      {/* Canvas controls */}
      <Paper
        elevation={2}
        sx={{
          position: 'absolute',
          bottom: 16,
          right: 16,
          p: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: 1,
          minWidth: 120
        }}
      >
        <Box sx={{ textAlign: 'center', fontSize: '0.75rem', color: 'text.secondary' }}>
          Zoom: {Math.round(zoom * 100)}%
        </Box>
        <Box sx={{ textAlign: 'center', fontSize: '0.75rem', color: 'text.secondary' }}>
          Nodes: {nodes.length}
        </Box>
        <Box sx={{ textAlign: 'center', fontSize: '0.75rem', color: 'text.secondary' }}>
          Connections: {connections.length}
        </Box>
      </Paper>
    </Box>
  );
};