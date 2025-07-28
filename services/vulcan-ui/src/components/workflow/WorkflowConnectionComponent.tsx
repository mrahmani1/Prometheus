import React, { useMemo } from 'react';
import { WorkflowConnection, WorkflowNode } from '@/types/workflow';

interface WorkflowConnectionComponentProps {
  connection: WorkflowConnection;
  nodes: WorkflowNode[];
  targetPosition?: { x: number; y: number };
  isTemporary?: boolean;
  onDelete?: () => void;
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

export const WorkflowConnectionComponent: React.FC<WorkflowConnectionComponentProps> = ({
  connection,
  nodes,
  targetPosition,
  isTemporary = false,
  onDelete
}) => {
  const { startPoint, endPoint, pathData } = useMemo(() => {
    const sourceNode = nodes.find(n => n.id === connection.sourceNodeId);
    if (!sourceNode) {
      return { startPoint: { x: 0, y: 0 }, endPoint: { x: 0, y: 0 }, pathData: '' };
    }

    // Calculate source port position
    // const sourcePort = sourceNode.outputs.find(p => p.id === connection.sourcePortId);
    const sourcePortIndex = sourceNode.outputs.findIndex(p => p.id === connection.sourcePortId);
    
    const sourceX = sourceNode.position.x + 200; // Node width
    const sourceY = sourceNode.position.y + 60 + (sourcePortIndex * 30) + 15; // Header height + port spacing

    let targetX: number;
    let targetY: number;

    if (isTemporary && targetPosition) {
      targetX = targetPosition.x;
      targetY = targetPosition.y;
    } else {
      const targetNode = nodes.find(n => n.id === connection.targetNodeId);
      if (!targetNode) {
        return { startPoint: { x: sourceX, y: sourceY }, endPoint: { x: sourceX + 100, y: sourceY }, pathData: '' };
      }

      const targetPortIndex = targetNode.inputs.findIndex(p => p.id === connection.targetPortId);
      targetX = targetNode.position.x;
      targetY = targetNode.position.y + 60 + (targetPortIndex * 30) + 15;
    }

    // Create curved path
    const deltaX = targetX - sourceX;
    // const deltaY = targetY - sourceY;
    const controlPointOffset = Math.max(50, Math.abs(deltaX) * 0.5);

    const controlPoint1X = sourceX + controlPointOffset;
    const controlPoint1Y = sourceY;
    const controlPoint2X = targetX - controlPointOffset;
    const controlPoint2Y = targetY;

    const pathData = `M ${sourceX} ${sourceY} C ${controlPoint1X} ${controlPoint1Y}, ${controlPoint2X} ${controlPoint2Y}, ${targetX} ${targetY}`;

    return {
      startPoint: { x: sourceX, y: sourceY },
      endPoint: { x: targetX, y: targetY },
      pathData
    };
  }, [connection, nodes, targetPosition, isTemporary]);

  const connectionColor = dataTypeColors[connection.dataType] || '#757575';

  const handleConnectionClick = (event: React.MouseEvent) => {
    if (!isTemporary && onDelete) {
      event.stopPropagation();
      if (window.confirm('Delete this connection?')) {
        onDelete();
      }
    }
  };

  return (
    <g>
      {/* Connection path */}
      <path
        d={pathData}
        fill="none"
        stroke={connectionColor}
        strokeWidth={isTemporary ? 2 : 3}
        strokeDasharray={isTemporary ? '5,5' : 'none'}
        opacity={isTemporary ? 0.6 : 0.8}
        style={{
          cursor: isTemporary ? 'default' : 'pointer',
          pointerEvents: isTemporary ? 'none' : 'stroke'
        }}
        onClick={handleConnectionClick}
      />
      
      {/* Connection highlight on hover */}
      {!isTemporary && (
        <path
          d={pathData}
          fill="none"
          stroke="transparent"
          strokeWidth={8}
          style={{ cursor: 'pointer', pointerEvents: 'stroke' }}
          onClick={handleConnectionClick}
        />
      )}

      {/* Arrow marker at the end */}
      <defs>
        <marker
          id={`arrowhead-${connection.id}`}
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto"
        >
          <polygon
            points="0 0, 10 3.5, 0 7"
            fill={connectionColor}
            opacity={isTemporary ? 0.6 : 0.8}
          />
        </marker>
      </defs>
      
      <path
        d={pathData}
        fill="none"
        stroke="transparent"
        strokeWidth={1}
        markerEnd={`url(#arrowhead-${connection.id})`}
        style={{ pointerEvents: 'none' }}
      />

      {/* Data type label */}
      {!isTemporary && (
        <g>
          <rect
            x={(startPoint.x + endPoint.x) / 2 - 25}
            y={(startPoint.y + endPoint.y) / 2 - 8}
            width={50}
            height={16}
            fill="white"
            stroke={connectionColor}
            strokeWidth={1}
            rx={8}
            opacity={0.9}
            style={{ pointerEvents: 'none' }}
          />
          <text
            x={(startPoint.x + endPoint.x) / 2}
            y={(startPoint.y + endPoint.y) / 2 + 3}
            textAnchor="middle"
            fontSize="10"
            fill={connectionColor}
            fontWeight="bold"
            style={{ pointerEvents: 'none' }}
          >
            {connection.dataType.split('-')[0]}
          </text>
        </g>
      )}
    </g>
  );
};