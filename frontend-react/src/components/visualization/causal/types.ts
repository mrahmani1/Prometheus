// import { ReactNode } from 'react'; // Unused

export interface CausalNode {
  id: string;
  label: string;
  processType: GeologicalProcessType;
  position?: { x: number; y: number };
  size: number;
  color: string;
  metadata: {
    description: string;
    dataSource: string;
    lastUpdated: Date;
    confidence: number;
    spatialScale: SpatialScale;
    temporalScale: TemporalScale;
  };
  properties: Record<string, any>;
  // D3 simulation properties
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface CausalEdge {
  id: string;
  source: string;
  target: string;
  strength: number;
  confidence: [number, number]; // confidence interval
  temporalLag?: number; // in milliseconds
  edgeType: CausalEdgeType;
  metadata: {
    discoveryMethod: string;
    statisticalSignificance: number;
    validationStatus: ValidationStatus;
    geologicalContext: string;
  };
  properties: Record<string, any>;
}

export interface CausalGraph {
  nodes: CausalNode[];
  edges: CausalEdge[];
  metadata: {
    createdAt: Date;
    lastUpdated: Date;
    totalNodes: number;
    totalEdges: number;
    averageCausalStrength: number;
    spatialExtent: GeographicBounds;
    temporalExtent: TemporalBounds;
  };
}

export interface CausalLayoutConfig {
  algorithm: LayoutAlgorithm;
  forceStrength: number;
  linkDistance: number;
  nodeRepulsion: number;
  centerForce: number;
  collisionRadius: number;
  alphaDecay: number;
  velocityDecay: number;
  clustering: {
    enabled: boolean;
    clusterBy: ClusteringCriteria;
    clusterStrength: number;
  };
  hierarchical: {
    enabled: boolean;
    direction: 'TB' | 'BT' | 'LR' | 'RL';
    levelSeparation: number;
    nodeSeparation: number;
  };
}

export interface CausalVisualizationProps {
  graph: CausalGraph;
  width: number;
  height: number;
  layoutConfig: CausalLayoutConfig;
  onNodeClick?: (node: CausalNode) => void;
  onEdgeClick?: (edge: CausalEdge) => void;
  onNodeHover?: (node: CausalNode | null) => void;
  onEdgeHover?: (edge: CausalEdge | null) => void;
  onSelectionChange?: (selectedNodes: string[], selectedEdges: string[]) => void;
  onLayoutComplete?: () => void;
  filterConfig?: CausalFilterConfig;
  searchQuery?: string;
  highlightedPaths?: CausalPath[];
  interactionMode: InteractionMode;
  renderMode: RenderMode;
  showLabels: boolean;
  showConfidenceIntervals: boolean;
  showTemporalLags: boolean;
  animationEnabled: boolean;
}

export interface CausalFilterConfig {
  strengthRange: [number, number];
  confidenceThreshold: number;
  processTypes: GeologicalProcessType[];
  spatialScales: SpatialScale[];
  temporalScales: TemporalScale[];
  edgeTypes: CausalEdgeType[];
  validationStatus: ValidationStatus[];
  discoveryMethods: string[];
}

export interface CausalPath {
  id: string;
  nodes: string[];
  edges: string[];
  pathStrength: number;
  pathLength: number;
  totalLag: number;
  pathType: PathType;
}

export interface CausalRelationship {
  sourceVariable: string;
  targetVariable: string;
  causalStrength: number;
  confidenceInterval: [number, number];
  temporalLag?: number;
  spatialScale: SpatialScale;
  statisticalSignificance: number;
  discoveryMethod: string;
  validationStatus: ValidationStatus;
  geologicalContext: GeologicalContext;
}

export interface CausalStrength {
  value: number;
  confidence: [number, number];
  pValue: number;
  effectSize: number;
  method: string;
}

export interface GeographicBounds {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface TemporalBounds {
  start: Date;
  end: Date;
}

export interface GeologicalContext {
  tectonicSetting: string;
  volcanicSystem?: string;
  geologicalUnit: string;
  spatialContext: string;
  temporalContext: string;
}

export type GeologicalProcessType = 
  | 'seismic'
  | 'volcanic'
  | 'deformation'
  | 'geochemical'
  | 'thermal'
  | 'electromagnetic'
  | 'hydrological'
  | 'atmospheric'
  | 'tectonic'
  | 'magmatic'
  | 'hydrothermal'
  | 'metamorphic'
  | 'sedimentary';

export type SpatialScale = 
  | 'mineral'
  | 'crystal'
  | 'rock'
  | 'outcrop'
  | 'local'
  | 'regional'
  | 'continental'
  | 'global';

export type TemporalScale = 
  | 'millisecond'
  | 'second'
  | 'minute'
  | 'hour'
  | 'day'
  | 'month'
  | 'year'
  | 'decade'
  | 'century'
  | 'millennium'
  | 'geological';

export type CausalEdgeType = 
  | 'direct'
  | 'indirect'
  | 'bidirectional'
  | 'conditional'
  | 'mediating'
  | 'moderating'
  | 'confounding';

export type ValidationStatus = 
  | 'validated'
  | 'pending'
  | 'rejected'
  | 'uncertain'
  | 'conflicting';

export type LayoutAlgorithm = 
  | 'force-directed'
  | 'hierarchical'
  | 'circular'
  | 'grid'
  | 'dagre'
  | 'cola'
  | 'cose-bilkent'
  | 'breadthfirst'
  | 'concentric';

export type ClusteringCriteria = 
  | 'processType'
  | 'spatialScale'
  | 'temporalScale'
  | 'causalStrength'
  | 'geologicalContext'
  | 'dataSource';

export type InteractionMode = 
  | 'explore'
  | 'select'
  | 'edit'
  | 'annotate'
  | 'measure'
  | 'path-trace';

export type RenderMode = 
  | 'canvas'
  | 'webgl'
  | 'svg'
  | 'hybrid';

export type PathType = 
  | 'causal-chain'
  | 'feedback-loop'
  | 'convergent'
  | 'divergent'
  | 'mediating'
  | 'confounding';

export interface CausalNetworkState {
  graph: CausalGraph;
  layout: CausalLayoutConfig;
  filter: CausalFilterConfig;
  selection: {
    nodes: string[];
    edges: string[];
  };
  viewport: {
    x: number;
    y: number;
    scale: number;
  };
  interaction: {
    mode: InteractionMode;
    hoveredNode?: string;
    hoveredEdge?: string;
    draggedNode?: string;
  };
  animation: {
    enabled: boolean;
    duration: number;
    easing: string;
  };
}

export interface CausalAnalysisResult {
  relationships: CausalRelationship[];
  confidence: number;
  method: string;
  parameters: Record<string, any>;
  validationMetrics: ValidationMetrics;
  temporalAnalysis?: TemporalCausalAnalysis;
  spatialAnalysis?: SpatialCausalAnalysis;
}

export interface ValidationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  crossValidationScore: number;
  bootstrapConfidence: [number, number];
}

export interface TemporalCausalAnalysis {
  lagAnalysis: LagAnalysisResult[];
  timeVaryingStrength: TimeSeriesPoint[];
  seasonalPatterns: SeasonalPattern[];
  trendAnalysis: TrendAnalysis;
}

export interface SpatialCausalAnalysis {
  spatialAutocorrelation: number;
  spatialClusters: SpatialCluster[];
  distanceDecay: DistanceDecayModel;
  spatialHeterogeneity: SpatialHeterogeneityMetrics;
}

export interface LagAnalysisResult {
  lag: number;
  strength: number;
  confidence: [number, number];
  significance: number;
}

export interface TimeSeriesPoint {
  timestamp: Date;
  value: number;
  confidence: [number, number];
}

export interface SeasonalPattern {
  period: number;
  amplitude: number;
  phase: number;
  significance: number;
}

export interface TrendAnalysis {
  slope: number;
  intercept: number;
  rSquared: number;
  pValue: number;
  changePoints: Date[];
}

export interface SpatialCluster {
  id: string;
  centroid: { x: number; y: number };
  radius: number;
  strength: number;
  nodes: string[];
}

export interface DistanceDecayModel {
  function: 'exponential' | 'power' | 'linear';
  parameters: number[];
  rSquared: number;
}

export interface SpatialHeterogeneityMetrics {
  moranI: number;
  gearyC: number;
  localIndicators: LocalSpatialIndicator[];
}

export interface LocalSpatialIndicator {
  nodeId: string;
  value: number;
  pValue: number;
  clusterType: 'HH' | 'LL' | 'HL' | 'LH' | 'NS';
}

// Additional missing types for temporal analysis
export interface TemporalCausalSnapshot {
  timestamp: Date;
  graph: CausalGraph;
  metadata: Record<string, any>;
}

export interface DetectedPattern {
  id: string;
  type: 'periodic' | 'trend' | 'anomaly' | 'phase-transition' | 'cascade';
  startTime: Date;
  endTime: Date;
  confidence: number;
  description: string;
  parameters: Record<string, any>;
  timeRange: [Date, Date];
  statisticalSignificance: number;
  affectedNodes: string[];
}

export interface TemporalVisualizationConfig {
  timeRange: [Date, Date];
  resolution: number;
  showPatterns: boolean;
  showTrends: boolean;
  animationSpeed: number;
  colorScheme: string;
  showTemporalLags: boolean;
  showConfidenceBands: boolean;
  showPatternHighlights: boolean;
  enableSynchronization: boolean;
  playbackSpeed: number;
  viewMode?: string;
  temporalFilters?: any;
}

// Additional missing types
export interface TemporalEvolutionData {
  timestamp: Date;
  nodes: CausalNode[];
  edges: CausalEdge[];
  metrics: Record<string, number>;
}

export interface CausalEvolutionAnalysis {
  timeRange: [Date, Date];
  evolutionData: TemporalEvolutionData[];
  patterns: DetectedPattern[];
  trends: any[];
}