import React, { useCallback, useRef } from 'react';
import { Box, Typography, Tooltip } from '@mui/material';
import { WorkflowPort } from '@/types/workflow';

interface WorkflowPortComponentProps {
  port: WorkflowPort;
  nodeId: string;
  isInput: boolean;
  onConnectionStart: (portId: string, position: { x: number; y: number }) => void;
  onConnectionEnd: (portId: string) => void;
}

const dataTypeColors = {
  'seismic-data': '#E91E63',
  'deformation-data': '#9C27B0',
  'geochemical-data': '#00BCD4',
  'thermal-data': '#FF5722',
  'geological-model': '#8BC34A',
  'time-series': '#FF9800',
  'spatial-data': '#4CAF50',
  'image': '#673AB7',
  'mesh': '#795548',
  'point-cloud': '#607D8B',
  'table': '#2196F3',
  'text': '#9E9E9E',
  'number': '#FFC107',
  'boolean': '#3F51B5',
  'array': '#009688',
  'object': '#CDDC39',
  'any': '#757575'
};

export const WorkflowPortComponent: React.FC<WorkflowPortComponentProps> = ({
  port,
  // nodeId,
  isInput,
  onConnectionStart,
  onConnectionEnd
}) => {
  const portRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (!isInput && event.button === 0) {
      event.preventDefault();
      event.stopPropagation();
      
      const rect = portRef.current?.getBoundingClientRect();
      if (rect) {
        const position = {
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2
        };
        onConnectionStart(port.id, position);
      }
    }
  }, [isInput, port.id, onConnectionStart]);

  const handleMouseUp = useCallback((event: React.MouseEvent) => {
    if (isInput && event.button === 0) {
      event.preventDefault();
      event.stopPropagation();
      onConnectionEnd(port.id);
    }
  }, [isInput, port.id, onConnectionEnd]);

  const portColor = dataTypeColors[port.type] || '#757575';

  return (
    <Tooltip
      title={
        <Box>
          <Typography variant="body2" fontWeight="bold">
            {port.name}
          </Typography>
          <Typography variant="caption" display="block">
            Type: {port.type}
          </Typography>
          <Typography variant="caption" display="block">
            {port.description}
          </Typography>
          {port.required && (
            <Typography variant="caption" display="block" color="error.light">
              Required
            </Typography>
          )}
        </Box>
      }
      placement={isInput ? 'left' : 'right'}
      arrow
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          cursor: isInput ? 'crosshair' : 'grab',
          '&:hover .port-connector': {
            transform: 'scale(1.2)',
            boxShadow: `0 0 8px ${portColor}`
          }
        }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
      >
        {/* Input port connector (left side) */}
        {isInput && (
          <Box
            ref={portRef}
            className="port-connector"
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: portColor,
              border: '2px solid white',
              boxShadow: '0 0 4px rgba(0,0,0,0.3)',
              transition: 'all 0.2s ease',
              position: 'relative',
              zIndex: 10,
              '&::before': {
                content: '""',
                position: 'absolute',
                left: -6,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 0,
                height: 0,
                borderTop: '4px solid transparent',
                borderBottom: '4px solid transparent',
                borderRight: `6px solid ${portColor}`,
                opacity: 0.7
              }
            }}
          />
        )}

        {/* Port label */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography
            variant="caption"
            sx={{
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              fontWeight: port.required ? 'bold' : 'normal',
              color: port.required ? 'text.primary' : 'text.secondary'
            }}
          >
            {port.name}
            {port.required && ' *'}
          </Typography>
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              color: 'text.secondary',
              fontSize: '0.65rem',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}
          >
            {port.type}
          </Typography>
        </Box>

        {/* Output port connector (right side) */}
        {!isInput && (
          <Box
            ref={portRef}
            className="port-connector"
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: portColor,
              border: '2px solid white',
              boxShadow: '0 0 4px rgba(0,0,0,0.3)',
              transition: 'all 0.2s ease',
              position: 'relative',
              zIndex: 10,
              '&::after': {
                content: '""',
                position: 'absolute',
                right: -6,
                top: '50%',
                transform: 'translateY(-50%)',
                width: 0,
                height: 0,
                borderTop: '4px solid transparent',
                borderBottom: '4px solid transparent',
                borderLeft: `6px solid ${portColor}`,
                opacity: 0.7
              }
            }}
          />
        )}
      </Box>
    </Tooltip>
  );
};