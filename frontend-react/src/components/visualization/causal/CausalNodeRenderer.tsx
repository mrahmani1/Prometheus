import React, { useMemo } from 'react';
// import * as d3 from 'd3'; // Unused
import { CausalNode, GeologicalProcessType, SpatialScale, TemporalScale } from './types';

interface CausalNodeRendererProps {
  node: CausalNode;
  isSelected: boolean;
  isHovered: boolean;
  isHighlighted: boolean;
  showLabels: boolean;
  showConfidence: boolean;
  scale: number;
  onClick?: (node: CausalNode) => void;
  onMouseEnter?: (node: CausalNode) => void;
  onMouseLeave?: () => void;
}

export const CausalNodeRenderer: React.FC<CausalNodeRendererProps> = ({
  node,
  isSelected,
  isHovered,
  isHighlighted,
  showLabels,
  showConfidence,
  scale,
  onClick,
  onMouseEnter,
  onMouseLeave
}) => {
  const nodeStyle = useMemo(() => {
    const baseSize = node.size * scale;
    const strokeWidth = isSelected ? 3 : isHovered ? 2 : 1;
    const opacity = isHighlighted ? 1.0 : 0.8;

    return {
      radius: Math.max(baseSize, 8),
      strokeWidth,
      opacity,
      fill: getProcessTypeColor(node.processType),
      stroke: isSelected ? '#ff6b35' : isHovered ? '#4a90e2' : '#333',
      cursor: 'pointer'
    };
  }, [node, isSelected, isHovered, isHighlighted, scale]);

  const confidenceIndicator = useMemo(() => {
    if (!showConfidence) return null;

    const confidence = node.metadata.confidence;
    const ringRadius = nodeStyle.radius + 4;
    const circumference = 2 * Math.PI * ringRadius;
    const strokeDasharray = `${confidence * circumference} ${circumference}`;

    return (
      <circle
        cx={node.position?.x || 0}
        cy={node.position?.y || 0}
        r={ringRadius}
        fill="none"
        stroke="#4caf50"
        strokeWidth="2"
        strokeDasharray={strokeDasharray}
        strokeLinecap="round"
        opacity={0.7}
        transform={`rotate(-90 ${node.position?.x || 0} ${node.position?.y || 0})`}
      />
    );
  }, [node, showConfidence, nodeStyle.radius]);

  const scaleIndicator = useMemo(() => {
    const spatialScale = node.metadata.spatialScale;
    const temporalScale = node.metadata.temporalScale;
    
    return (
      <g>
        {/* Spatial scale indicator */}
        <rect
          x={(node.position?.x || 0) - nodeStyle.radius - 2}
          y={(node.position?.y || 0) - nodeStyle.radius - 8}
          width={getSpatialScaleWidth(spatialScale)}
          height={2}
          fill={getSpatialScaleColor(spatialScale)}
          opacity={0.8}
        />
        {/* Temporal scale indicator */}
        <rect
          x={(node.position?.x || 0) - nodeStyle.radius - 2}
          y={(node.position?.y || 0) - nodeStyle.radius - 5}
          width={getTemporalScaleWidth(temporalScale)}
          height={2}
          fill={getTemporalScaleColor(temporalScale)}
          opacity={0.8}
        />
      </g>
    );
  }, [node, nodeStyle.radius]);

  const nodeLabel = useMemo(() => {
    if (!showLabels || scale < 0.5) return null;

    const labelY = (node.position?.y || 0) + nodeStyle.radius + 15;
    const fontSize = Math.max(10 * scale, 8);

    return (
      <text
        x={node.position?.x || 0}
        y={labelY}
        textAnchor="middle"
        fontSize={fontSize}
        fill="#333"
        fontWeight={isSelected ? 'bold' : 'normal'}
        opacity={0.9}
      >
        {node.label}
      </text>
    );
  }, [node, showLabels, scale, isSelected, nodeStyle.radius]);

  const processTypeIcon = useMemo(() => {
    const iconSize = nodeStyle.radius * 0.6;
    const iconPath = getProcessTypeIconPath(node.processType);
    
    if (!iconPath) return null;

    return (
      <path
        d={iconPath}
        transform={`translate(${(node.position?.x || 0) - iconSize/2}, ${(node.position?.y || 0) - iconSize/2}) scale(${iconSize/24})`}
        fill="white"
        opacity={0.9}
      />
    );
  }, [node, nodeStyle.radius]);

  const handleClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    onClick?.(node);
  };

  const handleMouseEnter = () => {
    onMouseEnter?.(node);
  };

  return (
    <g className="causal-node">
      {/* Main node circle */}
      <circle
        cx={node.position?.x || 0}
        cy={node.position?.y || 0}
        r={nodeStyle.radius}
        fill={nodeStyle.fill}
        stroke={nodeStyle.stroke}
        strokeWidth={nodeStyle.strokeWidth}
        opacity={nodeStyle.opacity}
        style={{ cursor: nodeStyle.cursor }}
        onClick={handleClick}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={onMouseLeave}
      />

      {/* Process type icon */}
      {processTypeIcon}

      {/* Confidence indicator */}
      {confidenceIndicator}

      {/* Scale indicators */}
      {scaleIndicator}

      {/* Node label */}
      {nodeLabel}

      {/* Selection highlight */}
      {isSelected && (
        <circle
          cx={node.position?.x || 0}
          cy={node.position?.y || 0}
          r={nodeStyle.radius + 6}
          fill="none"
          stroke="#ff6b35"
          strokeWidth="2"
          strokeDasharray="5,5"
          opacity={0.8}
        />
      )}

      {/* Hover effect */}
      {isHovered && (
        <circle
          cx={node.position?.x || 0}
          cy={node.position?.y || 0}
          r={nodeStyle.radius + 3}
          fill="none"
          stroke="#4a90e2"
          strokeWidth="1"
          opacity={0.6}
        />
      )}
    </g>
  );
};

// Helper functions for styling and icons

function getProcessTypeColor(processType: GeologicalProcessType): string {
  const colorMap: Record<GeologicalProcessType, string> = {
    seismic: '#e74c3c',
    volcanic: '#e67e22',
    deformation: '#f39c12',
    geochemical: '#27ae60',
    thermal: '#e74c3c',
    electromagnetic: '#9b59b6',
    hydrological: '#3498db',
    atmospheric: '#95a5a6',
    tectonic: '#34495e',
    magmatic: '#e67e22',
    hydrothermal: '#16a085',
    metamorphic: '#8e44ad',
    sedimentary: '#d35400'
  };

  return colorMap[processType] || '#7f8c8d';
}

function getSpatialScaleColor(scale: SpatialScale): string {
  const colorMap: Record<SpatialScale, string> = {
    mineral: '#8e44ad',
    crystal: '#9b59b6',
    rock: '#3498db',
    outcrop: '#2980b9',
    local: '#27ae60',
    regional: '#f39c12',
    continental: '#e67e22',
    global: '#e74c3c'
  };

  return colorMap[scale] || '#7f8c8d';
}

function getTemporalScaleColor(scale: TemporalScale): string {
  const colorMap: Record<TemporalScale, string> = {
    millisecond: '#e74c3c',
    second: '#e67e22',
    minute: '#f39c12',
    hour: '#f1c40f',
    day: '#27ae60',
    month: '#2980b9',
    year: '#8e44ad',
    decade: '#34495e',
    century: '#7f8c8d',
    millennium: '#95a5a6',
    geological: '#2c3e50'
  };

  return colorMap[scale] || '#7f8c8d';
}

function getSpatialScaleWidth(scale: SpatialScale): number {
  const widthMap: Record<SpatialScale, number> = {
    mineral: 4,
    crystal: 6,
    rock: 8,
    outcrop: 10,
    local: 12,
    regional: 14,
    continental: 16,
    global: 18
  };

  return widthMap[scale] || 8;
}

function getTemporalScaleWidth(scale: TemporalScale): number {
  const widthMap: Record<TemporalScale, number> = {
    millisecond: 4,
    second: 5,
    minute: 6,
    hour: 7,
    day: 8,
    month: 10,
    year: 12,
    decade: 14,
    century: 16,
    millennium: 18,
    geological: 20
  };

  return widthMap[scale] || 8;
}

function getProcessTypeIconPath(processType: GeologicalProcessType): string | null {
  const iconPaths: Record<GeologicalProcessType, string> = {
    seismic: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    volcanic: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    deformation: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    geochemical: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    thermal: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    electromagnetic: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    hydrological: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    atmospheric: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    tectonic: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    magmatic: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    hydrothermal: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    metamorphic: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z',
    sedimentary: 'M12 2L2 7v10c0 5.55 3.84 9.74 9 9.95 5.16-.21 9-4.4 9-9.95V7l-10-5z'
  };

  return iconPaths[processType] || null;
}