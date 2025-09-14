/**
 * Comprehensive Causal Analysis Types
 * 
 * TypeScript definitions for causal analysis, discovery, and visualization.
 * Supports multi-scale temporal analysis, hypothesis testing, and collaborative features.
 */

// Missing type definitions
export interface DomainExpertReview {
  expertId: string;
  reviewDate: Date;
  confidence: number;
  comments: string;
  recommendations: string[];
}

export interface PowerAnalysis {
  sampleSize: number;
  effectSize: number;
  power: number;
  alpha: number;
}

export interface IndividualTestResult {
  testName: string;
  pValue: number;
  statistic: number;
  confidence: number;
}

export interface RobustnessCheck {
  method: string;
  passed: boolean;
  details: Record<string, any>;
}

export interface AlternativeHypothesis {
  id: string;
  description: string;
  probability: number;
}

export interface TemporalStability {
  isStable: boolean;
  changePoints: Date[];
  stabilityScore: number;
}

export interface CrossScaleInteraction {
  scaleFrom: string;
  scaleTo: string;
  interactionType: string;
  strength: number;
}

export interface GraphChange {
  timestamp: Date;
  changeType: 'node_added' | 'node_removed' | 'edge_added' | 'edge_removed';
  elementId: string;
  details: Record<string, any>;
}

export interface LagPattern {
  sourceNode: string;
  targetNode: string;
  lag: number;
  confidence: number;
}

export interface EmergentProperty {
  name: string;
  description: string;
  value: number;
  significance: number;
}

export interface ScaleValidationResults {
  crossScaleConsistency: number;
  emergentProperties: EmergentProperty[];
  validationErrors: string[];
}

export interface MultiLevelLayoutConfig {
  levels: number;
  spacing: number;
  alignment: 'center' | 'left' | 'right';
}

export interface TemporalLayoutConfig {
  timeAxis: 'horizontal' | 'vertical';
  timeScale: 'linear' | 'log';
  showTimeline: boolean;
}

export interface SpatialLayoutConfig {
  projection: string;
  bounds: [number, number, number, number];
  zoomLevel: number;
}

export interface TemporalFilter {
  startTime: Date;
  endTime: Date;
  resolution: number;
}

export interface SpatialFilter {
  bounds: [number, number, number, number];
  spatialScale: string;
}

export interface StatisticalFilter {
  method: string;
  threshold: number;
  parameters: Record<string, any>;
}

export interface RealTimeMetrics {
  latency: number;
  throughput: number;
  accuracy: number;
  lastUpdate: Date;
}

export interface TemporalContext {
  timeRange: [Date, Date];
  resolution: number;
  referenceFrame: string;
}

export interface Participant {
  id: string;
  name: string;
  role: string;
  permissions: string[];
}

export interface SessionPermissions {
  canEdit: boolean;
  canView: boolean;
  canComment: boolean;
  canShare: boolean;
}

export interface ConflictingElement {
  elementId: string;
  elementType: 'node' | 'edge';
  conflictType: string;
  participants: string[];
}

export interface ResolutionOption {
  id: string;
  description: string;
  action: string;
  votes: number;
}

export interface ConflictResolution {
  resolutionId: string;
  resolvedBy: string;
  resolvedAt: Date;
  method: string;
}

export interface StatisticalValidation {
  method: string;
  pValue: number;
  confidence: number;
  passed: boolean;
}

export interface DomainValidation {
  expertReviews: DomainExpertReview[];
  consensusScore: number;
  validationNotes: string[];
}

export interface CrossValidation {
  folds: number;
  accuracy: number;
  precision: number;
  recall: number;
}

export interface UncertaintySource {
  source: string;
  type: 'epistemic' | 'aleatory';
  magnitude: number;
  description: string;
}

export interface TemporalConstraint {
  type: string;
  startTime: Date;
  endTime: Date;
  description: string;
}

export interface SpatialConstraint {
  type: string;
  bounds: [number, number, number, number];
  description: string;
}

// Core causal graph types
export interface CausalNode {
  id: string;
  label: string;
  processType: GeologicalProcessType;
  spatialScale: SpatialScale;
  temporalScale: TemporalScale;
  dataSource: string;
  metadata: Record<string, any>;
  
  // D3 simulation properties (required for force simulation)
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
  fx?: number | null;
  fy?: number | null;
  fz?: number | null;
  
  // Visualization properties
  position?: { x: number; y: number; z?: number };
  color?: string;
  size?: number;
  shape?: NodeShape;
  visible?: boolean;
  selected?: boolean;
  highlighted?: boolean;
  
  // Statistical properties
  uncertainty?: number;
  confidence?: number;
  dataQuality?: DataQuality;
  
  // Temporal properties
  timeRange?: [Date, Date];
  samplingRate?: number;
  
  // Spatial properties
  location?: GeoLocation;
  spatialExtent?: SpatialExtent;
}

export interface CausalEdge {
  id: string;
  source: string | CausalNode;
  target: string | CausalNode;
  causalStrength: number;
  confidenceInterval: [number, number];
  temporalLag?: number;
  statisticalSignificance: number;
  discoveryMethod: CausalDiscoveryMethod;
  validationStatus: ValidationStatus;
  geologicalContext: Record<string, any>;
  
  // Visualization properties
  color?: string;
  width?: number;
  style?: EdgeStyle;
  visible?: boolean;
  selected?: boolean;
  highlighted?: boolean;
  animated?: boolean;
  
  // Temporal properties
  lagUncertainty?: number;
  temporalStability?: number;
  
  // Statistical properties
  pValue?: number;
  effectSize?: number;
  bootstrapConfidence?: [number, number];
}

export interface CausalGraph {
  id: string;
  nodes: CausalNode[];
  edges: CausalEdge[];
  metadata: CausalGraphMetadata;
  createdAt: Date;
  updatedAt: Date;
  
  // Analysis properties
  complexity?: GraphComplexity;
  stability?: number;
  reliability?: number;
  
  // Temporal properties
  timeRange?: [Date, Date];
  temporalResolution?: number;
  
  // Spatial properties
  spatialBounds?: SpatialExtent;
  spatialResolution?: number;
}

export interface CausalGraphMetadata {
  algorithmsUsed: CausalDiscoveryMethod[];
  ensembleMethod?: string;
  confidenceThreshold: number;
  totalNodes: number;
  totalEdges: number;
  graphDensity: number;
  dataSourcesUsed: string[];
  processingTime?: number;
  validationResults?: ValidationResults;
  domainExpertReview?: DomainExpertReview;
}

// Causal discovery types
export interface CausalDiscoveryRequest {
  dataSources: string[];
  timeRange: [Date, Date];
  spatialBounds?: SpatialExtent;
  algorithms: CausalDiscoveryMethod[];
  confidenceThreshold: number;
  maxLag?: number;
  backgroundKnowledge?: BackgroundKnowledge;
  
  // Advanced options
  preprocessingOptions?: PreprocessingOptions;
  validationOptions?: ValidationOptions;
  performanceOptions?: PerformanceOptions;
}

export interface CausalAnalysisResponse {
  graph: CausalGraph;
  performanceMetrics: PerformanceMetrics;
  validationResults: ValidationResults;
  recommendations: string[];
  
  // Additional analysis results
  alternativeGraphs?: CausalGraph[];
  uncertaintyAnalysis?: UncertaintyAnalysis;
  sensitivityAnalysis?: SensitivityAnalysis;
}

// Hypothesis testing types
export interface HypothesisTestRequest {
  hypothesisGraph: CausalGraph;
  testData: Record<string, any>;
  testMethods: HypothesisTestMethod[];
  significanceLevel: number;
  
  // Advanced options
  bootstrapSamples?: number;
  permutationSamples?: number;
  crossValidationFolds?: number;
}

export interface HypothesisTestResult {
  hypothesisId: string;
  testResults: {
    bootstrapPValue: number;
    permutationPValue: number;
    confidenceInterval: [number, number];
    effectSize: number;
    powerAnalysis?: PowerAnalysis;
  };
  conclusion: string;
  recommendations: string[];
  
  // Detailed results
  individualTests?: IndividualTestResult[];
  robustnessCheck?: RobustnessCheck;
  alternativeHypotheses?: AlternativeHypothesis[];
}

// Temporal analysis types
export interface TemporalCausalModel {
  baseGraph: CausalGraph;
  temporalEvolution: TemporalSnapshot[];
  lagStructure: LagStructure;
  temporalStability: TemporalStability;
  
  // Multi-scale properties
  scaleHierarchy?: ScaleLevel[];
  crossScaleInteractions?: CrossScaleInteraction[];
}

export interface TemporalSnapshot {
  timestamp: Date;
  graph: CausalGraph;
  changes: GraphChange[];
  confidence: number;
}

export interface DetectedPattern {
  id: string;
  type: 'periodic' | 'trend' | 'anomaly' | 'phase-transition' | 'cascade';
  startTime: Date;
  endTime: Date;
  timeRange: [Date, Date];
  confidence: number;
  description: string;
  affectedNodes: string[];
  statisticalSignificance: number;
  parameters: Record<string, any>;
}

export interface TemporalEvolutionData {
  snapshots: TemporalCausalSnapshot[];
  temporalRange: [Date, Date];
  scaleHierarchy: ScaleLevel[];
  detectedPatterns: DetectedPattern[];
  evolutionMetrics: EvolutionMetrics;
}

export interface TemporalCausalSnapshot {
  timestamp: Date;
  graph: CausalGraph;
  metrics: SnapshotMetrics;
  changes: GraphChange[];
}

export interface EvolutionMetrics {
  stabilityScore: number;
  changeRate: number;
  complexityTrend: number;
  emergenceEvents: EmergenceEvent[];
}

export interface SnapshotMetrics {
  nodeCount: number;
  edgeCount: number;
  averageStrength: number;
  networkDensity: number;
  clusteringCoefficient: number;
}

export interface EmergenceEvent {
  timestamp: Date;
  type: 'emergence' | 'dissolution' | 'transformation';
  description: string;
  significance: number;
}

export interface LagStructure {
  edges: Array<{
    edgeId: string;
    optimalLag: number;
    lagRange: [number, number];
    lagConfidence: number;
  }>;
  globalLagPattern?: LagPattern;
}

// Multi-scale analysis types
export interface MultiScaleCausalModel {
  scaleLevels: ScaleLevel[];
  crossScaleRelationships: CrossScaleRelationship[];
  scaleCouplingStrength: Record<string, number>;
  emergentProperties: EmergentProperty[];
  validationAcrossScales: ScaleValidationResults;
}

export interface ScaleLevel {
  id: string;
  name: string;
  spatialExtent: SpatialExtent;
  temporalExtent: TemporalExtent;
  resolution: Resolution;
  processes: GeologicalProcessType[];
  causalGraph: CausalGraph;
}

export interface CrossScaleRelationship {
  id: string;
  sourceScale: string;
  targetScale: string;
  sourceProcess: string;
  targetProcess: string;
  couplingStrength: number;
  couplingType: CrossScaleCouplingType;
  emergenceType?: EmergenceType;
}

// Visualization types
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
    clusterBy: ClusteringMethod;
    clusterStrength: number;
  };
  
  hierarchical: {
    enabled: boolean;
    direction: 'TB' | 'BT' | 'LR' | 'RL';
    levelSeparation: number;
    nodeSeparation: number;
  };
  
  // Advanced layout options
  multiLevel?: MultiLevelLayoutConfig;
  temporal?: TemporalLayoutConfig;
  spatial?: SpatialLayoutConfig;
}

export interface CausalFilterConfig {
  strengthRange: [number, number];
  confidenceThreshold: number;
  processTypes: GeologicalProcessType[];
  spatialScales: SpatialScale[];
  temporalScales: TemporalScale[];
  edgeTypes: EdgeType[];
  validationStatus: ValidationStatus[];
  discoveryMethods: CausalDiscoveryMethod[];
  
  // Advanced filters
  temporalFilters?: TemporalFilter[];
  spatialFilters?: SpatialFilter[];
  statisticalFilters?: StatisticalFilter[];
}

export interface CausalPath {
  id: string;
  nodes: string[];
  edges: string[];
  totalStrength: number;
  confidence: number;
  pathType: PathType;
  
  // Path analysis
  bottlenecks?: string[];
  criticalNodes?: string[];
  alternativePaths?: CausalPath[];
}

// Real-time monitoring types
export interface RealTimeUpdate {
  type: RealTimeUpdateType;
  timestamp: Date;
  graphId?: string;
  changes?: GraphChange[];
  alerts?: CausalAlert[];
  metrics?: RealTimeMetrics;
}

export interface CausalAlert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  message: string;
  timestamp: Date;
  
  // Causal context
  affectedNodes?: string[];
  affectedEdges?: string[];
  causalPathway?: CausalPath;
  
  // Geological context
  geologicalProcess?: GeologicalProcessType;
  spatialLocation?: GeoLocation;
  temporalContext?: TemporalContext;
}

// Collaborative analysis types
export interface CollaborativeSession {
  id: string;
  name: string;
  participants: Participant[];
  sharedGraph: CausalGraph;
  annotations: Annotation[];
  conflicts: Conflict[];
  
  // Session management
  createdAt: Date;
  lastActivity: Date;
  permissions: SessionPermissions;
}

export interface Annotation {
  id: string;
  type: AnnotationType;
  author: string;
  timestamp: Date;
  content: string;
  
  // Targets
  targetNodes?: string[];
  targetEdges?: string[];
  targetRegion?: SpatialExtent;
  
  // Metadata
  tags?: string[];
  priority?: Priority;
  resolved?: boolean;
}

export interface Conflict {
  id: string;
  type: ConflictType;
  participants: string[];
  description: string;
  
  // Conflict details
  conflictingElements: ConflictingElement[];
  resolutionOptions: ResolutionOption[];
  status: ConflictStatus;
  
  // Resolution
  resolution?: ConflictResolution;
  resolvedBy?: string;
  resolvedAt?: Date;
}

// Performance and metrics types
export interface CausalMetrics {
  activeGraphs: number;
  discoveryRequestsToday: number;
  averageDiscoveryTime: number;
  
  systemPerformance: {
    cpuUsage: number;
    memoryUsage: number;
    databaseConnections: number;
    activeWebSocketConnections: number;
  };
  
  // Analysis metrics
  analysisMetrics: {
    totalNodesProcessed: number;
    totalEdgesDiscovered: number;
    averageGraphComplexity: number;
    validationSuccessRate: number;
  };
  
  // Quality metrics
  qualityMetrics: {
    averageConfidence: number;
    dataQualityScore: number;
    algorithmAgreement: number;
    domainExpertApproval: number;
  };
}

export interface PerformanceMetrics {
  discoveryTime: number;
  algorithmsUsed: CausalDiscoveryMethod[];
  dataSources: string[];
  graphComplexity: number;
  
  // Detailed timing
  dataFetchTime?: number;
  preprocessingTime?: number;
  algorithmTimes?: Record<string, number>;
  ensembleTime?: number;
  validationTime?: number;
  
  // Resource usage
  memoryUsage?: number;
  cpuUsage?: number;
  networkTraffic?: number;
}

// Enum types
export enum GeologicalProcessType {
  SEISMIC = 'seismic',
  VOLCANIC = 'volcanic',
  DEFORMATION = 'deformation',
  THERMAL = 'thermal',
  GEOCHEMICAL = 'geochemical',
  HYDROLOGICAL = 'hydrological',
  ATMOSPHERIC = 'atmospheric',
  TECTONIC = 'tectonic',
  MAGMATIC = 'magmatic',
  HYDROTHERMAL = 'hydrothermal',
  GRAVITATIONAL = 'gravitational',
  MAGNETIC = 'magnetic',
  ELECTROMAGNETIC = 'electromagnetic',
  UNKNOWN = 'unknown'
}

export enum SpatialScale {
  MICROSCOPIC = 'microscopic',
  MINERAL = 'mineral',
  ROCK = 'rock',
  OUTCROP = 'outcrop',
  LOCAL = 'local',
  REGIONAL = 'regional',
  CONTINENTAL = 'continental',
  GLOBAL = 'global'
}

export enum TemporalScale {
  MILLISECONDS = 'milliseconds',
  SECONDS = 'seconds',
  MINUTES = 'minutes',
  HOURS = 'hours',
  DAYS = 'days',
  WEEKS = 'weeks',
  MONTHS = 'months',
  YEARS = 'years',
  DECADES = 'decades',
  CENTURIES = 'centuries',
  MILLENNIA = 'millennia',
  GEOLOGICAL = 'geological'
}

export enum CausalDiscoveryMethod {
  PC = 'pc',
  GES = 'ges',
  LINGAM = 'lingam',
  CAUSAL_FOREST = 'causal_forest',
  NOTEARS = 'notears',
  GRANGERCAUSALITY = 'granger_causality',
  TRANSFER_ENTROPY = 'transfer_entropy',
  CCM = 'convergent_cross_mapping',
  PCMCI = 'pcmci'
}

export enum ValidationStatus {
  VALIDATED = 'validated',
  PENDING = 'pending',
  REJECTED = 'rejected',
  UNCERTAIN = 'uncertain',
  ENSEMBLE_VALIDATED = 'ensemble_validated',
  DOMAIN_VALIDATED = 'domain_validated'
}

export enum LayoutAlgorithm {
  FORCE_DIRECTED = 'force-directed',
  HIERARCHICAL = 'hierarchical',
  CIRCULAR = 'circular',
  GRID = 'grid',
  SPRING = 'spring',
  FRUCHTERMAN_REINGOLD = 'fruchterman-reingold',
  KAMADA_KAWAI = 'kamada-kawai',
  MULTILEVEL = 'multilevel'
}

export enum InteractionMode {
  EXPLORE = 'explore',
  SELECT = 'select',
  EDIT = 'edit',
  ANNOTATE = 'annotate',
  MEASURE = 'measure',
  FILTER = 'filter'
}

export enum RenderMode {
  CANVAS = 'canvas',
  SVG = 'svg',
  WEBGL = 'webgl',
  THREE_JS = 'three_js'
}

export enum AlertType {
  CAUSAL_CHANGE = 'causal_change',
  STRENGTH_THRESHOLD = 'strength_threshold',
  NEW_RELATIONSHIP = 'new_relationship',
  LOST_RELATIONSHIP = 'lost_relationship',
  ANOMALY_DETECTED = 'anomaly_detected',
  DATA_QUALITY = 'data_quality',
  SYSTEM_PERFORMANCE = 'system_performance'
}

export enum AlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical'
}

export enum AnnotationType {
  NOTE = 'note',
  QUESTION = 'question',
  HYPOTHESIS = 'hypothesis',
  VALIDATION = 'validation',
  CORRECTION = 'correction',
  INTERPRETATION = 'interpretation'
}

export enum ConflictType {
  EDGE_DISAGREEMENT = 'edge_disagreement',
  NODE_CLASSIFICATION = 'node_classification',
  STRENGTH_DISAGREEMENT = 'strength_disagreement',
  INTERPRETATION_CONFLICT = 'interpretation_conflict',
  VALIDATION_CONFLICT = 'validation_conflict'
}

// Additional supporting types
export interface GeoLocation {
  latitude: number;
  longitude: number;
  elevation?: number;
  uncertainty?: number;
}

export interface SpatialExtent {
  minLat: number;
  maxLat: number;
  minLon: number;
  maxLon: number;
  minElevation?: number;
  maxElevation?: number;
}

export interface TemporalExtent {
  start: Date;
  end: Date;
  resolution?: number;
  timezone?: string;
}

export interface Resolution {
  spatial: number;
  temporal: number;
  units: {
    spatial: string;
    temporal: string;
  };
}

export interface DataQuality {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  overall: number;
}

export interface GraphComplexity {
  nodeCount: number;
  edgeCount: number;
  density: number;
  clustering: number;
  pathLength: number;
  centrality: Record<string, number>;
}

export interface ValidationResults {
  overallScore: number;
  statisticalValidation: StatisticalValidation;
  domainValidation: DomainValidation;
  crossValidation: CrossValidation;
  robustnessCheck: RobustnessCheck;
}

export interface UncertaintyAnalysis {
  nodeUncertainties: Record<string, number>;
  edgeUncertainties: Record<string, number>;
  globalUncertainty: number;
  uncertaintySources: UncertaintySource[];
}

export interface SensitivityAnalysis {
  parameterSensitivity: Record<string, number>;
  dataPointSensitivity: Record<string, number>;
  algorithmSensitivity: Record<string, number>;
  overallRobustness: number;
}

// Type aliases for convenience
export type NodeShape = 'circle' | 'square' | 'triangle' | 'diamond' | 'star' | 'hexagon';
export type EdgeStyle = 'solid' | 'dashed' | 'dotted' | 'curved' | 'straight';
export type EdgeType = 'causal' | 'correlational' | 'temporal' | 'spatial';
export type PathType = 'direct' | 'indirect' | 'feedback' | 'mediated';
export type ClusteringMethod = 'processType' | 'spatialScale' | 'temporalScale' | 'strength' | 'community';
export type RealTimeUpdateType = 'graph_update' | 'node_change' | 'edge_change' | 'alert' | 'metrics';
export type HypothesisTestMethod = 'bootstrap' | 'permutation' | 'cross_validation' | 'bayesian';
export type CrossScaleCouplingType = 'upscaling' | 'downscaling' | 'bidirectional';
export type EmergenceType = 'weak' | 'strong' | 'synchronic' | 'diachronic';
export type Priority = 'low' | 'medium' | 'high' | 'critical';
export type ConflictStatus = 'open' | 'in_progress' | 'resolved' | 'escalated';
export type ProcessingStatus = 'RECEIVED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'QUEUED';

// Additional interfaces for completeness
export interface BackgroundKnowledge {
  forbiddenEdges?: Array<[string, string]>;
  requiredEdges?: Array<[string, string]>;
  processConstraints?: Record<string, string[]>;
  temporalConstraints?: TemporalConstraint[];
  spatialConstraints?: SpatialConstraint[];
}

export interface PreprocessingOptions {
  normalization?: 'standard' | 'minmax' | 'robust' | 'none';
  missingValueHandling?: 'interpolation' | 'forward_fill' | 'backward_fill' | 'drop';
  outlierDetection?: 'iqr' | 'zscore' | 'isolation_forest' | 'none';
  temporalAlignment?: 'interpolation' | 'resampling' | 'none';
}

export interface ValidationOptions {
  crossValidationFolds?: number;
  bootstrapSamples?: number;
  holdoutPercentage?: number;
  domainExpertValidation?: boolean;
  statisticalTests?: string[];
}

export interface PerformanceOptions {
  maxNodes?: number;
  maxEdges?: number;
  timeoutSeconds?: number;
  parallelProcessing?: boolean;
  memoryLimit?: number;
  cacheResults?: boolean;
}

// All types are exported individually above