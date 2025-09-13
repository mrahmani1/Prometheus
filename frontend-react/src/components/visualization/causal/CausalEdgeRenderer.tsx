import React, { useMemo } from 'react';
import { CausalEdge, CausalNode, CausalEdgeType } from './types';

interface CausalEdgeRendererProps {
  edge: CausalEdge;
  sourceNode: CausalNode;
  targetNode: CausalNode;
  isSelected: boolean;
  isHovered: boolean;
  isHighlighted: boolean;
  showConfidenceIntervals: boolean;
  showTemporalLags: boolean;
  scale: number;
  onClick?: (edge: CausalEdge) => void;
  onMouseEnter?: (edge: CausalEdge) => void;
  onMouseLeave?: () => void;
}

export const CausalEdgeRenderer: React.FC<CausalEdgeRendererProps> = ({
  edge,
  sourceNode,
  targetNode,
  isSelected,
  isHovered,
  isHighlighted,
  showConfidenceIntervals,
  showTemporalLags,
  scale,
  onClick,
  onMouseEnter,
  onMouseLeave
}) => {
  const edgeGeometry = useMemo(() => {
    const sourceX = sourceNode.position?.x || 0;
    const sourceY = sourceNode.position?.y || 0;
    const targetX = targetNode.position?.x || 0;
    const targetY = targetNode.position?.y || 0;

    // Calculate edge path based on type
    const dx = targetX - sourceX;
    const dy = targetY - sourceY;
    const distance = Math.sqrt(dx * dx + dy * dy);

    // Adjust for node radii
    const sourceRadius = sourceNode.size;
    const targetRadius = targetNode.size;
    
    const unitX = dx / distance;
    const unitY = dy / distance;

    const adjustedSourceX = sourceX + unitX * sourceRadius;
    const adjustedSourceY = sourceY + unitY * sourceRadius;
    const adjustedTargetX = targetX - unitX * targetRadius;
    const adjustedTargetY = targetY - unitY * targetRadius;

    return {
      sourceX: adjustedSourceX,
      sourceY: adjustedSourceY,
      targetX: adjustedTargetX,
      targetY: adjustedTargetY,
      midX: (adjustedSourceX + adjustedTargetX) / 2,
      midY: (adjustedSourceY + adjustedTargetY) / 2,
      distance: Math.sqrt(
        (adjustedTargetX - adjustedSourceX) ** 2 + 
        (adjustedTargetY - adjustedSourceY) ** 2
      ),
      angle: Math.atan2(adjustedTargetY - adjustedSourceY, adjustedTargetX - adjustedSourceX)
    };
  }, [sourceNode, targetNode]);

  const edgeStyle = useMemo(() => {
    const strengthOpacity = Math.max(0.3, Math.min(1.0, edge.strength));
    const strokeWidth = Math.max(1, edge.strength * 4 * scale);
    
    return {
      stroke: getEdgeTypeColor(edge.edgeType),
      strokeWidth: isSelected ? strokeWidth + 2 : strokeWidth,
      opacity: isHighlighted ? strengthOpacity : strengthOpacity * 0.7,
      strokeDasharray: getEdgeTypeDashArray(edge.edgeType),
      markerEnd: 'url(#arrowhead)',
      cursor: 'pointer'
    };
  }, [edge, isSelected, isHighlighted, scale]);

  const edgePath = useMemo(() => {
    const { sourceX, sourceY, targetX, targetY } = edgeGeometry;

    switch (edge.edgeType) {
      case 'bidirectional':
        return `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`;
      
      case 'conditional':
      case 'mediating':
        // Curved path for complex relationships
        const controlX = (sourceX + targetX) / 2 + (targetY - sourceY) * 0.2;
        const controlY = (sourceY + targetY) / 2 - (targetX - sourceX) * 0.2;
        return `M ${sourceX} ${sourceY} Q ${controlX} ${controlY} ${targetX} ${targetY}`;
      
      case 'confounding':
        // Dashed curved path
        const confoundControlX = (sourceX + targetX) / 2 - (targetY - sourceY) * 0.3;
        const confoundControlY = (sourceY + targetY) / 2 + (targetX - sourceX) * 0.3;
        return `M ${sourceX} ${sourceY} Q ${confoundControlX} ${confoundControlY} ${targetX} ${targetY}`;
      
      default:
        return `M ${sourceX} ${sourceY} L ${targetX} ${targetY}`;
    }
  }, [edge.edgeType, edgeGeometry]);

  const confidenceInterval = useMemo(() => {
    if (!showConfidenceIntervals) return null;

    const [lowerBound, upperBound] = edge.confidence;
    const confidenceWidth = (upperBound - lowerBound) * 20; // Scale confidence to visual width
    
    return (
      <line
        x1={edgeGeometry.sourceX}
        y1={edgeGeometry.sourceY}
        x2={edgeGeometry.targetX}
        y2={edgeGeometry.targetY}
        stroke="#4caf50"
        strokeWidth={confidenceWidth}
        opacity={0.3}
        strokeLinecap="round"
      />
    );
  }, [edge.confidence, showConfidenceIntervals, edgeGeometry]);

  const temporalLagIndicator = useMemo(() => {
    if (!showTemporalLags || !edge.temporalLag) return null;

    const lagText = formatTemporalLag(edge.temporalLag);
    const { midX, midY, angle } = edgeGeometry;

    return (
      <g>
        <rect
          x={midX - 20}
          y={midY - 8}
          width={40}
          height={16}
          fill="white"
          stroke="#666"
          strokeWidth={1}
          rx={3}
          opacity={0.9}
        />
        <text
          x={midX}
          y={midY + 4}
          textAnchor="middle"
          fontSize={10}
          fill="#333"
          fontWeight="bold"
        >
          {lagText}
        </text>
      </g>
    );
  }, [edge.temporalLag, showTemporalLags, edgeGeometry]);

  const strengthIndicator = useMemo(() => {
    const { midX, midY } = edgeGeometry;
    const strengthText = edge.strength.toFixed(2);
    
    return (
      <g>
        <circle
          cx={midX}
          cy={midY}
          r={12}
          fill="white"
          stroke={edgeStyle.stroke}
          strokeWidth={2}
          opacity={0.9}
        />
        <text
          x={midX}
          y={midY + 3}
          textAnchor="middle"
          fontSize={9}
          fill="#333"
          fontWeight="bold"
        >
          {strengthText}
        </text>
      </g>
    );
  }, [edge.strength, edgeGeometry, edgeStyle.stroke]);

  const validationIndicator = useMemo(() => {
    const validationColor = getValidationStatusColor(edge.metadata.validationStatus);
    const { targetX, targetY, angle } = edgeGeometry;
    
    // Position validation indicator near the target
    const indicatorX = targetX - Math.cos(angle) * 15;
    const indicatorY = targetY - Math.sin(angle) * 15;

    return (
      <circle
        cx={indicatorX}
        cy={indicatorY}
        r={4}
        fill={validationColor}
        stroke="white"
        strokeWidth={1}
        opacity={0.8}
      />
    );
  }, [edge.metadata.validationStatus, edgeGeometry]);

  const handleClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    onClick?.(edge);
  };

  const handleMouseEnter = () => {
    onMouseEnter?.(edge);
  };

  return (
    <g className="causal-edge">
      {/* Confidence interval background */}
      {confidenceInterval}

      {/* Main edge path */}
      <path
        d={edgePath}
        fill="none"
        stroke={edgeStyle.stroke}
        strokeWidth={edgeStyle.strokeWidth}
        strokeDasharray={edgeStyle.strokeDasharray}
        opacity={edgeStyle.opacity}
        markerEnd={edgeStyle.markerEnd}
        style={{ cursor: edgeStyle.cursor }}
        onClick={handleClick}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={onMouseLeave}
      />

      {/* Bidirectional arrow for bidirectional edges */}
      {edge.edgeType === 'bidirectional' && (
        <path
          d={edgePath}
          fill="none"
          stroke={edgeStyle.stroke}
          strokeWidth={edgeStyle.strokeWidth}
          opacity={edgeStyle.opacity}
          markerStart="url(#arrowhead)"
        />
      )}

      {/* Strength indicator */}
      {scale > 0.5 && strengthIndicator}

      {/* Temporal lag indicator */}
      {temporalLagIndicator}

      {/* Validation status indicator */}
      {validationIndicator}

      {/* Selection highlight */}
      {isSelected && (
        <path
          d={edgePath}
          fill="none"
          stroke="#ff6b35"
          strokeWidth={edgeStyle.strokeWidth + 4}
          opacity={0.5}
          strokeDasharray="8,4"
        />
      )}

      {/* Hover effect */}
      {isHovered && (
        <path
          d={edgePath}
          fill="none"
          stroke="#4a90e2"
          strokeWidth={edgeStyle.strokeWidth + 2}
          opacity={0.7}
        />
      )}
    </g>
  );
};

// Helper functions

function getEdgeTypeColor(edgeType: CausalEdgeType): string {
  const colorMap: Record<CausalEdgeType, string> = {
    direct: '#2ecc71',
    indirect: '#3498db',
    bidirectional: '#9b59b6',
    conditional: '#f39c12',
    mediating: '#e67e22',
    moderating: '#e74c3c',
    confounding: '#95a5a6'
  };

  return colorMap[edgeType] || '#7f8c8d';
}

function getEdgeTypeDashArray(edgeType: CausalEdgeType): string {
  const dashMap: Record<CausalEdgeType, string> = {
    direct: 'none',
    indirect: '5,5',
    bidirectional: 'none',
    conditional: '10,5,2,5',
    mediating: '8,4',
    moderating: '12,3,3,3',
    confounding: '15,5'
  };

  return dashMap[edgeType] || 'none';
}

function getValidationStatusColor(status: string): string {
  const colorMap: Record<string, string> = {
    validated: '#27ae60',
    pending: '#f39c12',
    rejected: '#e74c3c',
    uncertain: '#95a5a6',
    conflicting: '#8e44ad'
  };

  return colorMap[status] || '#7f8c8d';
}

function formatTemporalLag(lagMs: number): string {
  const seconds = lagMs / 1000;
  const minutes = seconds / 60;
  const hours = minutes / 60;
  const days = hours / 24;
  const years = days / 365.25;

  if (years >= 1) {
    return `${years.toFixed(1)}y`;
  } else if (days >= 1) {
    return `${days.toFixed(1)}d`;
  } else if (hours >= 1) {
    return `${hours.toFixed(1)}h`;
  } else if (minutes >= 1) {
    return `${minutes.toFixed(1)}m`;
  } else {
    return `${seconds.toFixed(1)}s`;
  }
}