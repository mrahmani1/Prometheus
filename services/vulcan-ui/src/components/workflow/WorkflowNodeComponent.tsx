import React, { useCallback, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Chip,
  Tooltip
} from '@mui/material';
import {
  Delete,
  // Settings,
  PlayArrow,
  CheckCircle,
  Error,
  Schedule
} from '@mui/icons-material';
import { WorkflowNode } from '@/types/workflow';
import { WorkflowPortComponent } from './WorkflowPortComponent';

interface WorkflowNodeComponentProps {
  node: WorkflowNode;
  isSelected: boolean;
  onSelect: () => void;
  onUpdate: (updates: Partial<WorkflowNode>) => void;
  onDelete: () => void;
  onDragStart: (startPosition: { x: number; y: number }) => void;
  onDrag: (position: { x: number; y: number }) => void;
  onDragEnd: (position: { x: number; y: number }) => void;
  onConnectionStart: (nodeId: string, portId: string, position: { x: number; y: number }) => void;
  onConnectionEnd: (targetNodeId?: string, targetPortId?: string) => void;
  isDragging?: boolean;
  dragPosition?: { x: number; y: number };
}

const statusColors = {
  idle: '#9E9E9E',
  running: '#FF9800',
  completed: '#4CAF50',
  error: '#F44336'
};

const statusIcons = {
  idle: <Schedule />,
  running: <PlayArrow />,
  completed: <CheckCircle />,
  error: <Error />
};

const categoryColors = {
  'data-input': '#2196F3',
  'data-processing': '#FF9800',
  'geological-analysis': '#8BC34A',
  'seismic-analysis': '#E91E63',
  'deformation-analysis': '#9C27B0',
  'geochemical-analysis': '#00BCD4',
  'thermal-analysis': '#FF5722',
  'ai-modeling': '#673AB7',
  'visualization': '#4CAF50',
  'data-output': '#795548',
  'control-flow': '#607D8B',
  'utilities': '#9E9E9E'
};

export const WorkflowNodeComponent: React.FC<WorkflowNodeComponentProps> = ({
  node,
  isSelected,
  onSelect,
  // onUpdate,
  onDelete,
  onDragStart,
  onDrag,
  onDragEnd,
  onConnectionStart,
  onConnectionEnd,
  isDragging,
  dragPosition
}) => {
  const nodeRef = useRef<HTMLDivElement>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  const position = isDragging && dragPosition ? dragPosition : node.position;

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (event.button === 0) { // Left mouse button
      event.preventDefault();
      event.stopPropagation();
      
      onSelect();
      
      const rect = nodeRef.current?.getBoundingClientRect();
      if (rect) {
        dragStartRef.current = {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top
        };
        
        onDragStart(node.position);
      }
    }
  }, [node.position, onSelect, onDragStart]);

  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (isDragging && dragStartRef.current) {
      const rect = nodeRef.current?.getBoundingClientRect();
      if (rect) {
        const newPosition = {
          x: event.clientX - dragStartRef.current.x,
          y: event.clientY - dragStartRef.current.y
        };
        onDrag(newPosition);
      }
    }
  }, [isDragging, onDrag]);

  const handleMouseUp = useCallback((event: React.MouseEvent) => {
    if (isDragging && dragStartRef.current) {
      const rect = nodeRef.current?.getBoundingClientRect();
      if (rect) {
        const finalPosition = {
          x: event.clientX - dragStartRef.current.x,
          y: event.clientY - dragStartRef.current.y
        };
        onDragEnd(finalPosition);
      }
      dragStartRef.current = null;
    }
  }, [isDragging, onDragEnd]);

  const handlePortConnectionStart = useCallback((portId: string, portPosition: { x: number; y: number }) => {
    onConnectionStart(node.id, portId, portPosition);
  }, [node.id, onConnectionStart]);

  const handlePortConnectionEnd = useCallback((portId: string) => {
    onConnectionEnd(node.id, portId);
  }, [node.id, onConnectionEnd]);

  const categoryColor = categoryColors[node.category] || '#9E9E9E';
  const statusColor = statusColors[node.status || 'idle'];

  return (
    <Paper
      ref={nodeRef}
      elevation={isSelected ? 8 : 2}
      sx={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        minWidth: 200,
        maxWidth: 300,
        cursor: isDragging ? 'grabbing' : 'grab',
        border: isSelected ? `2px solid ${categoryColor}` : '1px solid transparent',
        borderRadius: 2,
        overflow: 'visible',
        zIndex: isSelected ? 10 : 5,
        transition: isDragging ? 'none' : 'all 0.2s ease',
        '&:hover': {
          elevation: 4,
          transform: isDragging ? 'none' : 'translateY(-1px)'
        }
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Node Header */}
      <Box
        sx={{
          p: 1.5,
          backgroundColor: categoryColor,
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          gap: 1
        }}
      >
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography
            variant="subtitle2"
            fontWeight="bold"
            sx={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}
          >
            {node.name}
          </Typography>
          <Typography
            variant="caption"
            sx={{
              opacity: 0.9,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden'
            }}
          >
            {node.description}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Tooltip title={`Status: ${node.status || 'idle'}`}>
            <Box sx={{ color: statusColor, display: 'flex' }}>
              {statusIcons[node.status || 'idle']}
            </Box>
          </Tooltip>
          
          <IconButton
            size="small"
            onClick={(e: any) => {
              e.stopPropagation();
              onDelete();
            }}
            sx={{ color: 'white', '&:hover': { backgroundColor: 'rgba(255,255,255,0.1)' } }}
          >
            <Delete fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* Node Body */}
      <Box sx={{ p: 1.5 }}>
        {/* Input Ports */}
        {node.inputs.length > 0 && (
          <Box sx={{ mb: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
              Inputs
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {node.inputs.map(port => (
                <WorkflowPortComponent
                  key={port.id}
                  port={port}
                  nodeId={node.id}
                  isInput
                  onConnectionStart={handlePortConnectionStart}
                  onConnectionEnd={handlePortConnectionEnd}
                />
              ))}
            </Box>
          </Box>
        )}

        {/* Output Ports */}
        {node.outputs.length > 0 && (
          <Box>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
              Outputs
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {node.outputs.map(port => (
                <WorkflowPortComponent
                  key={port.id}
                  port={port}
                  nodeId={node.id}
                  isInput={false}
                  onConnectionStart={handlePortConnectionStart}
                  onConnectionEnd={handlePortConnectionEnd}
                />
              ))}
            </Box>
          </Box>
        )}

        {/* Parameter Summary */}
        {Object.keys(node.parameters).length > 0 && (
          <Box sx={{ mt: 1, pt: 1, borderTop: 1, borderColor: 'divider' }}>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 0.5, display: 'block' }}>
              Parameters
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {Object.entries(node.parameters).slice(0, 3).map(([key, value]) => (
                <Chip
                  key={key}
                  label={`${key}: ${String(value).slice(0, 10)}${String(value).length > 10 ? '...' : ''}`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.6rem', height: 20 }}
                />
              ))}
              {Object.keys(node.parameters).length > 3 && (
                <Chip
                  label={`+${Object.keys(node.parameters).length - 3} more`}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.6rem', height: 20 }}
                />
              )}
            </Box>
          </Box>
        )}

        {/* Error Display */}
        {node.error && (
          <Box sx={{ mt: 1, p: 1, backgroundColor: 'error.light', borderRadius: 1 }}>
            <Typography variant="caption" color="error.contrastText">
              {node.error}
            </Typography>
          </Box>
        )}
      </Box>
    </Paper>
  );
};