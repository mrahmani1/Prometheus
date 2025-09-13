import React, { useRef, useEffect, useMemo, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  PerspectiveCamera, 
  Environment, 
  Text, 
  Line, 
  Sphere,
  Box,
  Trail
} from '@react-three/drei';
import * as THREE from 'three';
import { 
  TemporalCausalSnapshot, 
  CausalNode, 
  CausalEdge, 
  TemporalVisualizationConfig,
  TemporalScale 
} from './types';

export interface TemporalCausal4DProps {
  snapshots: TemporalCausalSnapshot[];
  currentTime: Date;
  config: TemporalVisualizationConfig;
  selectedNodes: string[];
  onNodeSelect: (nodeId: string) => void;
  onEdgeSelect: (edgeId: string) => void;
  width: number;
  height: number;
}

interface Node4D extends CausalNode {
  position3D: THREE.Vector3;
  temporalPosition: number;
  trajectory: THREE.Vector3[];
  velocity: THREE.Vector3;
  acceleration: THREE.Vector3;
  lifespan: [number, number]; // Start and end time indices
}

interface Edge4D extends CausalEdge {
  temporalStrength: number[];
  strengthTrajectory: { time: number; strength: number }[];
  lagVisualization: THREE.Vector3[];
}

const TEMPORAL_AXIS_LENGTH = 20;
const SPATIAL_SCALE = 10;
const TIME_SCALE_FACTORS = {
  millisecond: 0.001,
  second: 1,
  minute: 60,
  hour: 3600,
  day: 86400,
  month: 2592000,
  year: 31536000,
  decade: 315360000,
  century: 3153600000,
  millennium: 31536000000,
  geological: 31536000000000,
};

/**
 * Advanced 4D visualization of temporal causal relationships
 * Implements cutting-edge techniques for visualizing causal evolution over time
 */
export const TemporalCausal4D: React.FC<TemporalCausal4DProps> = ({
  snapshots,
  currentTime,
  config,
  selectedNodes,
  onNodeSelect,
  onEdgeSelect,
  width,
  height,
}) => {
  const [nodes4D, setNodes4D] = useState<Node4D[]>([]);
  const [edges4D, setEdges4D] = useState<Edge4D[]>([]);
  const [temporalIndex, setTemporalIndex] = useState(0);

  // Process snapshots into 4D data structures
  const processedData = useMemo(() => {
    if (snapshots.length === 0) return { nodes: [], edges: [] };

    const nodeTrajectories = new Map<string, Node4D>();
    const edgeEvolution = new Map<string, Edge4D>();

    // Process each snapshot
    snapshots.forEach((snapshot, timeIndex) => {
      const temporalPosition = (timeIndex / (snapshots.length - 1)) * TEMPORAL_AXIS_LENGTH - TEMPORAL_AXIS_LENGTH / 2;

      // Process nodes
      snapshot.graph.nodes.forEach(node => {
        if (!nodeTrajectories.has(node.id)) {
          // Initialize node trajectory
          const node4D: Node4D = {
            ...node,
            position3D: new THREE.Vector3(
              (Math.random() - 0.5) * SPATIAL_SCALE,
              (Math.random() - 0.5) * SPATIAL_SCALE,
              temporalPosition
            ),
            temporalPosition,
            trajectory: [],
            velocity: new THREE.Vector3(),
            acceleration: new THREE.Vector3(),
            lifespan: [timeIndex, timeIndex],
          };
          nodeTrajectories.set(node.id, node4D);
        } else {
          // Update existing node
          const existingNode = nodeTrajectories.get(node.id)!;
          existingNode.lifespan[1] = timeIndex;
          
          // Calculate new position based on causal forces
          const newPosition = calculateCausalPosition(
            node,
            snapshot.graph,
            existingNode.position3D,
            temporalPosition
          );
          
          // Update trajectory
          existingNode.trajectory.push(newPosition.clone());
          
          // Calculate velocity and acceleration
          if (existingNode.trajectory.length > 1) {
            const prevPos = existingNode.trajectory[existingNode.trajectory.length - 2];
            existingNode.velocity = newPosition.clone().sub(prevPos);
            
            if (existingNode.trajectory.length > 2) {
              const prevVel = existingNode.trajectory[existingNode.trajectory.length - 2]
                .clone().sub(existingNode.trajectory[existingNode.trajectory.length - 3]);
              existingNode.acceleration = existingNode.velocity.clone().sub(prevVel);
            }
          }
          
          existingNode.position3D = newPosition;
          existingNode.temporalPosition = temporalPosition;
        }
      });

      // Process edges
      snapshot.graph.edges.forEach(edge => {
        const edgeId = `${edge.source}-${edge.target}`;
        
        if (!edgeEvolution.has(edgeId)) {
          // Initialize edge evolution
          const edge4D: Edge4D = {
            ...edge,
            temporalStrength: [edge.strength],
            strengthTrajectory: [{ time: timeIndex, strength: edge.strength }],
            lagVisualization: [],
          };
          edgeEvolution.set(edgeId, edge4D);
        } else {
          // Update existing edge
          const existingEdge = edgeEvolution.get(edgeId)!;
          existingEdge.temporalStrength.push(edge.strength);
          existingEdge.strengthTrajectory.push({ time: timeIndex, strength: edge.strength });
          
          // Update lag visualization if temporal lag exists
          if (edge.temporalLag && config.showTemporalLags) {
            const sourceNode = nodeTrajectories.get(edge.source);
            const targetNode = nodeTrajectories.get(edge.target);
            
            if (sourceNode && targetNode) {
              const lagVector = calculateLagVisualization(
                sourceNode.position3D,
                targetNode.position3D,
                edge.temporalLag,
                temporalPosition
              );
              existingEdge.lagVisualization.push(lagVector);
            }
          }
        }
      });
    });

    return {
      nodes: Array.from(nodeTrajectories.values()),
      edges: Array.from(edgeEvolution.values()),
    };
  }, [snapshots, config.showTemporalLags]);

  // Update current temporal index based on current time
  useEffect(() => {
    if (snapshots.length === 0) return;
    
    const timeRange = snapshots[snapshots.length - 1].timestamp.getTime() - snapshots[0].timestamp.getTime();
    const currentOffset = currentTime.getTime() - snapshots[0].timestamp.getTime();
    const normalizedTime = Math.max(0, Math.min(1, currentOffset / timeRange));
    
    setTemporalIndex(normalizedTime * (snapshots.length - 1));
  }, [currentTime, snapshots]);

  // Update processed data
  useEffect(() => {
    setNodes4D(processedData.nodes);
    setEdges4D(processedData.edges);
  }, [processedData]);

  return (
    <Canvas camera={{ position: [15, 15, 15], fov: 60 }}>
      <PerspectiveCamera makeDefault />
      <OrbitControls enablePan enableZoom enableRotate />
      <Environment preset="studio" />
      
      {/* Lighting setup for 4D visualization */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 10, 5]} intensity={0.8} castShadow />
      <pointLight position={[-10, -10, -5]} intensity={0.3} />
      
      {/* Temporal axis */}
      <TemporalAxis 
        length={TEMPORAL_AXIS_LENGTH}
        snapshots={snapshots}
        currentIndex={temporalIndex}
      />
      
      {/* 4D Nodes */}
      {nodes4D.map(node => (
        <Node4DVisualization
          key={node.id}
          node={node}
          temporalIndex={temporalIndex}
          config={config}
          isSelected={selectedNodes.includes(node.id)}
          onSelect={() => onNodeSelect(node.id)}
        />
      ))}
      
      {/* 4D Edges */}
      {edges4D.map(edge => (
        <Edge4DVisualization
          key={edge.id}
          edge={edge}
          nodes={nodes4D}
          temporalIndex={temporalIndex}
          config={config}
          onSelect={() => onEdgeSelect(edge.id)}
        />
      ))}
      
      {/* Confidence uncertainty visualization */}
      {config.showConfidenceBands && (
        <ConfidenceVisualization
          snapshots={snapshots}
          temporalIndex={temporalIndex}
        />
      )}
      
      {/* Pattern highlights */}
      {config.showPatternHighlights && (
        <PatternHighlights
          snapshots={snapshots}
          temporalIndex={temporalIndex}
        />
      )}
    </Canvas>
  );
};

/**
 * Temporal axis component showing time progression
 */
const TemporalAxis: React.FC<{
  length: number;
  snapshots: TemporalCausalSnapshot[];
  currentIndex: number;
}> = ({ length, snapshots, currentIndex }) => {
  const axisRef = useRef<THREE.Group>(null);

  return (
    <group ref={axisRef}>
      {/* Main temporal axis */}
      <Line
        points={[
          [-length / 2, 0, 0],
          [length / 2, 0, 0]
        ]}
        color="#666666"
        lineWidth={2}
      />
      
      {/* Time markers */}
      {snapshots.map((snapshot, index) => {
        const position = (index / (snapshots.length - 1)) * length - length / 2;
        const isCurrentTime = Math.abs(index - currentIndex) < 0.5;
        
        return (
          <group key={index} position={[position, 0, 0]}>
            {/* Time marker */}
            <Box
              args={[0.1, 0.5, 0.1]}
              material-color={isCurrentTime ? "#ff4444" : "#888888"}
            />
            
            {/* Time label */}
            <Text
              position={[0, -1, 0]}
              fontSize={0.3}
              color="#333333"
              anchorX="center"
              anchorY="middle"
            >
              {snapshot.timestamp.toLocaleDateString()}
            </Text>
          </group>
        );
      })}
      
      {/* Current time indicator */}
      <group position={[(currentIndex / (snapshots.length - 1)) * length - length / 2, 0, 0]}>
        <Sphere args={[0.2]} material-color="#ff0000" />
        <Text
          position={[0, 1, 0]}
          fontSize={0.4}
          color="#ff0000"
          anchorX="center"
          anchorY="middle"
        >
          NOW
        </Text>
      </group>
    </group>
  );
};

/**
 * 4D Node visualization with trajectory and temporal evolution
 */
const Node4DVisualization: React.FC<{
  node: Node4D;
  temporalIndex: number;
  config: TemporalVisualizationConfig;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ node, temporalIndex, config, isSelected, onSelect }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const trailRef = useRef<THREE.Group>(null);

  // Animate node based on temporal position
  useFrame((state, delta) => {
    if (!meshRef.current) return;

    // Interpolate position based on temporal index
    if (node.trajectory.length > 1) {
      const segmentIndex = Math.floor(temporalIndex);
      const segmentProgress = temporalIndex - segmentIndex;
      
      if (segmentIndex < node.trajectory.length - 1) {
        const currentPos = node.trajectory[segmentIndex];
        const nextPos = node.trajectory[segmentIndex + 1];
        
        const interpolatedPos = currentPos.clone().lerp(nextPos, segmentProgress);
        meshRef.current.position.copy(interpolatedPos);
      }
    }

    // Animate based on causal activity
    const activityLevel = Math.sin(state.clock.elapsedTime * 2) * 0.1 + 1;
    meshRef.current.scale.setScalar(activityLevel * (isSelected ? 1.5 : 1));
  });

  // Calculate node color based on process type and confidence
  const nodeColor = useMemo(() => {
    const baseColor = new THREE.Color(node.color);
    const confidence = node.metadata.confidence;
    return baseColor.multiplyScalar(confidence);
  }, [node.color, node.metadata.confidence]);

  return (
    <group>
      {/* Main node sphere */}
      <mesh
        ref={meshRef}
        position={node.position3D}
        onClick={onSelect}
        onPointerOver={(e) => {
          e.stopPropagation();
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          document.body.style.cursor = 'auto';
        }}
      >
        <sphereGeometry args={[node.size / 10, 16, 16]} />
        <meshStandardMaterial
          color={nodeColor}
          transparent
          opacity={isSelected ? 1.0 : 0.8}
          emissive={isSelected ? nodeColor.clone().multiplyScalar(0.2) : new THREE.Color(0x000000)}
        />
      </mesh>

      {/* Trajectory trail */}
      {node.trajectory.length > 1 && (
        <Trail
          width={0.05}
          color={nodeColor}
          length={node.trajectory.length}
          decay={1}
          local={false}
          stride={0}
          interval={1}
          target={meshRef}
          attenuation={(width) => width}
        />
      )}

      {/* Confidence uncertainty visualization */}
      {config.showConfidenceBands && (
        <mesh position={node.position3D}>
          <sphereGeometry args={[node.size / 8, 8, 8]} />
          <meshBasicMaterial
            color="#ffcc00"
            transparent
            opacity={0.2 * (1 - node.metadata.confidence)}
            wireframe
          />
        </mesh>
      )}

      {/* Node label */}
      <Text
        position={[
          node.position3D.x,
          node.position3D.y + node.size / 8 + 0.5,
          node.position3D.z
        ]}
        fontSize={0.3}
        color="#333333"
        anchorX="center"
        anchorY="middle"
      >
        {node.label}
      </Text>

      {/* Velocity vector visualization */}
      {node.velocity.length() > 0.01 && (
        <Line
          points={[
            node.position3D,
            node.position3D.clone().add(node.velocity.clone().multiplyScalar(2))
          ]}
          color="#00ff00"
          lineWidth={1}
        />
      )}
    </group>
  );
};

/**
 * 4D Edge visualization with temporal lag and strength evolution
 */
const Edge4DVisualization: React.FC<{
  edge: Edge4D;
  nodes: Node4D[];
  temporalIndex: number;
  config: TemporalVisualizationConfig;
  onSelect: () => void;
}> = ({ edge, nodes, temporalIndex, config, onSelect }) => {
  const sourceNode = nodes.find(n => n.id === edge.source);
  const targetNode = nodes.find(n => n.id === edge.target);

  if (!sourceNode || !targetNode) return null;

  // Interpolate edge strength based on temporal index
  const currentStrength = useMemo(() => {
    if (edge.temporalStrength.length === 0) return edge.strength;
    
    const segmentIndex = Math.floor(temporalIndex);
    const segmentProgress = temporalIndex - segmentIndex;
    
    if (segmentIndex >= edge.temporalStrength.length - 1) {
      return edge.temporalStrength[edge.temporalStrength.length - 1];
    }
    
    const currentStr = edge.temporalStrength[segmentIndex];
    const nextStr = edge.temporalStrength[segmentIndex + 1];
    
    return currentStr + (nextStr - currentStr) * segmentProgress;
  }, [edge.temporalStrength, temporalIndex]);

  // Calculate edge color and opacity based on strength
  const edgeColor = useMemo(() => {
    const hue = (1 - currentStrength) * 240; // Blue to red gradient
    return new THREE.Color().setHSL(hue / 360, 0.8, 0.5);
  }, [currentStrength]);

  return (
    <group>
      {/* Main edge line */}
      <Line
        points={[sourceNode.position3D, targetNode.position3D]}
        color={edgeColor}
        lineWidth={Math.max(1, currentStrength * 5)}
        transparent
        opacity={currentStrength}
        onClick={onSelect}
      />

      {/* Temporal lag visualization */}
      {config.showTemporalLags && edge.temporalLag && (
        <TemporalLagVisualization
          sourcePos={sourceNode.position3D}
          targetPos={targetNode.position3D}
          lag={edge.temporalLag}
          strength={currentStrength}
        />
      )}

      {/* Causal strength indicator */}
      <mesh
        position={sourceNode.position3D.clone().lerp(targetNode.position3D, 0.5)}
        onClick={onSelect}
      >
        <boxGeometry args={[0.2, 0.1, 0.1]} />
        <meshStandardMaterial
          color={edgeColor}
          transparent
          opacity={currentStrength}
        />
      </mesh>

      {/* Confidence interval visualization */}
      {config.showConfidenceBands && edge.confidence && (
        <ConfidenceInterval
          sourcePos={sourceNode.position3D}
          targetPos={targetNode.position3D}
          confidence={edge.confidence}
          strength={currentStrength}
        />
      )}
    </group>
  );
};

/**
 * Temporal lag visualization component
 */
const TemporalLagVisualization: React.FC<{
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
  lag: number;
  strength: number;
}> = ({ sourcePos, targetPos, lag, strength }) => {
  const lagVisualizationRef = useRef<THREE.Group>(null);

  // Animate lag visualization
  useFrame((state) => {
    if (!lagVisualizationRef.current) return;
    
    const time = state.clock.elapsedTime;
    const lagProgress = (Math.sin(time * 2) + 1) / 2; // 0 to 1
    
    // Move indicator along the edge to show temporal progression
    const position = sourcePos.clone().lerp(targetPos, lagProgress);
    lagVisualizationRef.current.position.copy(position);
  });

  return (
    <group ref={lagVisualizationRef}>
      <Sphere args={[0.1]} material-color="#ff9800" material-transparent material-opacity={strength} />
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.2}
        color="#ff9800"
        anchorX="center"
        anchorY="middle"
      >
        {`${lag}ms`}
      </Text>
    </group>
  );
};

/**
 * Confidence interval visualization
 */
const ConfidenceInterval: React.FC<{
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
  confidence: [number, number];
  strength: number;
}> = ({ sourcePos, targetPos, confidence, strength }) => {
  const midPoint = sourcePos.clone().lerp(targetPos, 0.5);
  const confidenceWidth = (confidence[1] - confidence[0]) * 2;

  return (
    <group position={midPoint}>
      <Box
        args={[confidenceWidth, 0.05, 0.05]}
        material-color="#ffeb3b"
        material-transparent
        material-opacity={0.3 * strength}
      />
    </group>
  );
};

/**
 * Confidence visualization for entire snapshots
 */
const ConfidenceVisualization: React.FC<{
  snapshots: TemporalCausalSnapshot[];
  temporalIndex: number;
}> = ({ snapshots, temporalIndex }) => {
  if (snapshots.length === 0) return null;

  const currentSnapshot = snapshots[Math.floor(temporalIndex)] || snapshots[0];
  const confidence = currentSnapshot.metadata.confidence;

  return (
    <group>
      {/* Global confidence indicator */}
      <mesh position={[0, 8, 0]}>
        <cylinderGeometry args={[2, 2, 0.2, 16]} />
        <meshStandardMaterial
          color={confidence > 0.8 ? "#4caf50" : confidence > 0.5 ? "#ff9800" : "#f44336"}
          transparent
          opacity={0.3}
        />
      </mesh>
      
      <Text
        position={[0, 9, 0]}
        fontSize={0.5}
        color="#333333"
        anchorX="center"
        anchorY="middle"
      >
        {`Confidence: ${(confidence * 100).toFixed(1)}%`}
      </Text>
    </group>
  );
};

/**
 * Pattern highlights visualization
 */
const PatternHighlights: React.FC<{
  snapshots: TemporalCausalSnapshot[];
  temporalIndex: number;
}> = ({ snapshots, temporalIndex }) => {
  // This would integrate with the pattern detection system
  // For now, showing a placeholder visualization
  return (
    <group>
      {/* Pattern highlight indicators would go here */}
    </group>
  );
};

// Helper functions

/**
 * Calculate causal position based on network forces
 */
function calculateCausalPosition(
  node: CausalNode,
  graph: { nodes: CausalNode[]; edges: CausalEdge[] },
  currentPosition: THREE.Vector3,
  temporalPosition: number
): THREE.Vector3 {
  const newPosition = currentPosition.clone();
  
  // Apply forces based on causal relationships
  const connectedEdges = graph.edges.filter(
    edge => edge.source === node.id || edge.target === node.id
  );
  
  connectedEdges.forEach(edge => {
    const isSource = edge.source === node.id;
    const otherNodeId = isSource ? edge.target : edge.source;
    const otherNode = graph.nodes.find(n => n.id === otherNodeId);
    
    if (otherNode && otherNode.position) {
      const direction = isSource ? 1 : -1;
      const force = new THREE.Vector3(
        (Math.random() - 0.5) * edge.strength * direction,
        (Math.random() - 0.5) * edge.strength * direction,
        0
      );
      
      newPosition.add(force.multiplyScalar(0.1));
    }
  });
  
  // Set temporal position
  newPosition.z = temporalPosition;
  
  return newPosition;
}

/**
 * Calculate lag visualization vector
 */
function calculateLagVisualization(
  sourcePos: THREE.Vector3,
  targetPos: THREE.Vector3,
  lag: number,
  temporalPosition: number
): THREE.Vector3 {
  const midPoint = sourcePos.clone().lerp(targetPos, 0.5);
  const lagOffset = new THREE.Vector3(0, 0, lag * 0.001); // Scale lag to visualization
  
  return midPoint.add(lagOffset);
}

export default TemporalCausal4D;