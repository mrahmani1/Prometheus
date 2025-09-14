// Core Causal Analysis Types

export interface CausalVariable {
  id: string;
  name: string;
  type: VariableType;
  unit: string;
  description: string;
  geologicalContext: GeologicalContext;
  constraints: VariableConstraint[];
  metadata: VariableMetadata;
  x?: number; // For graph positioning
  y?: number; // For graph positioning
}

export interface CausalEdge {
  id: string;
  sourceId: string;
  targetId: string;
  relationshipType: CausalRelationshipType;
  strength: number; // 0-1
  confidence: number; // 0-1
  temporalLag: number; // in days
  spatialScale: SpatialScale;
  constraints: EdgeConstraint[];
  metadata: EdgeMetadata;
}

export interface CausalHypothesis {
  id: string;
  name: string;
  description: string;
  variables: CausalVariable[];
  edges: CausalEdge[];
  constraints: HypothesisConstraints;
  metadata: HypothesisMetadata;
}

export interface GeologicalProcess {
  id: string;
  name: string;
  category: ProcessCategory;
  description: string;
  typicalVariables: string[];
  spatialScale: SpatialScale;
  temporalScale: TemporalScale;
  physicsType: PhysicsType;
}

// Statistical Testing Types

export interface TestParameters {
  significanceLevel: number;
  bootstrapSamples: number;
  crossValidationFolds: number;
  temporalLagRange: [number, number];
  spatialScaleRange: [number, number];
  confidenceInterval: number;
  robustnessTests: boolean;
  multipleTestingCorrection: MultipleTestingCorrection;
  bootstrapMethod?: BootstrapMethod;
  crossValidationMethod?: CrossValidationMethod;
  bayesianAnalysis?: boolean;
  nonParametricTests?: boolean;
  permutationTests?: boolean;
  permutationIterations?: number;
  randomSeed?: number;
  minimumEffectSize?: number;
}

export interface HypothesisTestResult {
  id: string;
  hypothesisId: string;
  timestamp: Date;
  pValue: number;
  effectSize: number;
  confidenceInterval: {
    lower: number;
    upper: number;
  };
  testStatistic: number;
  degreesOfFreedom: number;
  robustnessTests?: RobustnessTestResults;
  bayesianResults?: BayesianTestResults;
  metadata: TestResultMetadata;
}

export interface RobustnessTestResults {
  bootstrapStability: number;
  crossValidationScore: number;
  sensitivityAnalysis: SensitivityResult[];
  permutationPValue?: number;
}

export interface BayesianTestResults {
  posteriorProbability: number;
  bayesFactor: number;
  credibleInterval: {
    lower: number;
    upper: number;
  };
  priorDistribution: string;
}

export interface SensitivityResult {
  parameter: string;
  originalValue: number;
  perturbedValue: number;
  resultChange: number;
}

// Hypothesis Comparison Types

export interface HypothesisComparison {
  id: string;
  hypotheses: CausalHypothesis[];
  comparisonMetrics: ComparisonMetric[];
  ranking: HypothesisRanking[];
  timestamp: Date;
}

export interface ComparisonMetric {
  name: string;
  values: Map<string, number>; // hypothesis ID -> metric value
  higherIsBetter: boolean;
  description: string;
}

export interface HypothesisRanking {
  hypothesisId: string;
  rank: number;
  overallScore: number;
  individualScores: Map<string, number>;
}

// Collaborative Features Types

export interface CollaborativeSession {
  id: string;
  name: string;
  participants: CollaborativeUser[];
  currentUser: string;
  permissions: CollaborativePermissions;
  createdAt: Date;
  lastActivity: Date;
}

export interface CollaborativeUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  isOnline: boolean;
  lastSeen: Date;
  avatar?: string;
}

export interface CollaborativePermissions {
  canEdit: boolean;
  canComment: boolean;
  canReview: boolean;
  canShare: boolean;
  canDelete: boolean;
}

export interface Comment {
  id: string;
  author: string;
  content: string;
  timestamp: Date;
  type: CommentType;
  targetElement: string; // ID of hypothesis, variable, or edge
  resolved: boolean;
  resolvedBy?: string;
  resolvedAt?: Date;
  likes: number;
  replies: Comment[];
}

export interface Review {
  id: string;
  reviewer: string;
  hypothesis: string;
  status: ReviewStatus;
  feedback: string;
  timestamp: Date;
  suggestions: ReviewSuggestion[];
}

export interface ReviewSuggestion {
  id: string;
  type: SuggestionType;
  description: string;
  targetElement: string;
  proposedChange: any;
  priority: Priority;
}

export interface ChangeRequest {
  id: string;
  requester: string;
  description: string;
  targetElement: string;
  proposedChange: any;
  status: ChangeRequestStatus;
  timestamp: Date;
  votes: {
    approve: number;
    reject: number;
  };
  reviewers: string[];
}

export interface ConflictResolution {
  id: string;
  description: string;
  conflictingUsers: string[];
  conflictType: ConflictType;
  status: ConflictStatus;
  resolution?: ConflictResolutionType;
  resolvedBy?: string;
  resolvedAt?: Date;
}

// Auto-Generation Types

export interface HypothesisGenerationConfig {
  algorithms: GenerationAlgorithm[];
  maxHypotheses: number;
  minVariables: number;
  maxVariables: number;
  minEdges: number;
  maxEdges: number;
  significanceThreshold: number;
  effectSizeThreshold: number;
  temporalConstraints: boolean;
  spatialConstraints: boolean;
  geologicalConstraints: boolean;
  domainKnowledge: boolean;
  noveltyWeight: number;
  plausibilityWeight: number;
  simplicityWeight: number;
  bootstrapMethod?: BootstrapMethod;
  crossValidationMethod?: CrossValidationMethod;
}

export interface GeneratedHypothesisMetrics {
  algorithm: GenerationAlgorithm;
  overallScore: number;
  noveltyScore: number;
  plausibilityScore: number;
  complexity: number;
  patterns: PatternType[];
  confidence: number;
  domainKnowledgeAlignment?: number;
  statisticalSupport?: number;
}

// Version Control Types

export interface HypothesisVersion {
  id: string;
  hypothesis: CausalHypothesis;
  timestamp: Date;
  author: string;
  message: string;
  branch: string;
  parentVersion: string | null;
  tags: string[];
  metadata: VersionMetadata;
}

export interface VersionBranch {
  id: string;
  name: string;
  description: string;
  created: Date;
  lastCommit: string | null;
  author: string;
  isProtected: boolean;
}

export interface VersionDiff {
  version1: HypothesisVersion;
  version2: HypothesisVersion;
  changes: VersionChange[];
  summary: string;
}

export interface VersionChange {
  type: ChangeType;
  field: string;
  description: string;
  oldValue: any;
  newValue: any;
}

export interface MergeConflict {
  id: string;
  field: string;
  description: string;
  currentValue: any;
  incomingValue: any;
  type: ConflictType;
  resolved: boolean;
  resolution?: any;
}

export interface VersionMetadata {
  changeType: ChangeType;
  changesCount: number;
  testResults: HypothesisTestResult | null;
  reviewStatus: ReviewStatus;
  branchInfo?: BranchInfo;
}

export interface BranchInfo {
  sourceBranch?: string;
  mergeCommit?: string;
  conflictsResolved?: number;
}

// Metadata Types

export interface VariableMetadata {
  created: Date;
  source: DataSource;
  quality: DataQuality;
  lastUpdated?: Date;
  validationStatus?: ValidationStatus;
}

export interface EdgeMetadata {
  created: Date;
  discoveryMethod: DiscoveryMethod;
  validationStatus: ValidationStatus;
  lastValidated?: Date;
  validationScore?: number;
}

export interface HypothesisMetadata {
  created: Date;
  lastModified: Date;
  author: string;
  version: number;
  reviewStatus?: ReviewStatus;
  lastReviewed?: Date;
  reviewers?: string[];
  generationMetrics?: GeneratedHypothesisMetrics;
  testHistory?: string[]; // Test result IDs
  collaborators?: string[];
}

export interface TestResultMetadata {
  testType: StatisticalTest;
  datasetSize: number;
  dataQuality: DataQuality;
  assumptions: TestAssumption[];
  warnings: string[];
  computationTime: number;
}

// Constraint Types

export interface VariableConstraint {
  type: ConstraintType;
  description: string;
  parameters: Record<string, any>;
  isHard: boolean; // Hard vs soft constraint
}

export interface EdgeConstraint {
  type: ConstraintType;
  description: string;
  parameters: Record<string, any>;
  isHard: boolean;
}

export interface HypothesisConstraints {
  temporal: TemporalConstraint[];
  spatial: SpatialConstraint[];
  geological: GeologicalConstraint[];
}

export interface TemporalConstraint {
  type: TemporalConstraintType;
  minLag?: number;
  maxLag?: number;
  periodicity?: number;
  seasonality?: boolean;
}

export interface SpatialConstraint {
  type: SpatialConstraintType;
  minDistance?: number;
  maxDistance?: number;
  spatialScale: SpatialScale;
  coordinateSystem?: string;
}

export interface GeologicalConstraint {
  type: GeologicalConstraintType;
  processType: ProcessCategory;
  physicsType: PhysicsType;
  validContexts: GeologicalContext[];
}

export interface TestAssumption {
  name: string;
  satisfied: boolean;
  pValue?: number;
  description: string;
}

// Enum Types

export type VariableType = 'continuous' | 'discrete' | 'categorical' | 'binary';

export type CausalRelationshipType = 'causal' | 'inhibitory' | 'correlational' | 'confounding';

export type GeologicalContext = 'volcanic' | 'seismic' | 'tectonic' | 'hydrothermal' | 'magmatic' | 'structural';

export type ProcessCategory = 'thermal' | 'mechanical' | 'chemical' | 'electromagnetic' | 'fluid' | 'deformation';

export type SpatialScale = 'microscopic' | 'local' | 'regional' | 'global';

export type TemporalScale = 'seconds' | 'minutes' | 'hours' | 'days' | 'months' | 'years' | 'geological';

export type PhysicsType = 'thermodynamics' | 'mechanics' | 'electromagnetics' | 'fluid-dynamics' | 'quantum';

export type MultipleTestingCorrection = 'none' | 'bonferroni' | 'holm' | 'benjamini-hochberg' | 'benjamini-yekutieli';

export type BootstrapMethod = 'percentile' | 'bias-corrected' | 'accelerated' | 'studentized';

export type CrossValidationMethod = 'k-fold' | 'stratified' | 'time-series' | 'leave-one-out';

export type StatisticalTest = 'pearson-correlation' | 'granger-causality' | 'pc-algorithm' | 'ges' | 'lingam' | 'causal-forests';

export type CommentType = 'general' | 'suggestion' | 'question' | 'concern' | 'praise';

export type ReviewStatus = 'draft' | 'pending' | 'approved' | 'needs-revision' | 'rejected';

export type UserRole = 'owner' | 'collaborator' | 'reviewer' | 'viewer';

export type SuggestionType = 'variable-addition' | 'edge-modification' | 'constraint-update' | 'methodology-change';

export type Priority = 'low' | 'medium' | 'high' | 'critical';

export type ChangeRequestStatus = 'pending' | 'approved' | 'rejected' | 'implemented';

export type ConflictType = 'variable_conflict' | 'edge_conflict' | 'constraint_conflict' | 'metadata_conflict';

export type ConflictStatus = 'pending' | 'resolved' | 'escalated';

export type ConflictResolutionType = 'accept' | 'reject' | 'merge';

export type GenerationAlgorithm = 
  | 'correlation-based' 
  | 'mutual-information' 
  | 'granger-causality' 
  | 'pc-algorithm' 
  | 'ges-algorithm' 
  | 'lingam' 
  | 'causal-forests';

export type PatternType = 
  | 'volcanic-unrest' 
  | 'seismic-precursors' 
  | 'deformation-patterns' 
  | 'gas-emissions' 
  | 'thermal-anomalies' 
  | 'magma-intrusion' 
  | 'hydrothermal-activity' 
  | 'tectonic-stress';

export type ChangeType = 'creation' | 'modification' | 'addition' | 'removal' | 'merge';

export type DataSource = 'manual' | 'imported' | 'generated' | 'api' | 'database';

export type DataQuality = 'high' | 'medium' | 'low' | 'unknown';

export type ValidationStatus = 'proposed' | 'validated' | 'rejected' | 'under-review';

export type DiscoveryMethod = 'manual' | 'statistical' | 'machine-learning' | 'domain-knowledge' | 'auto-generated';

export type ConstraintType = 'temporal' | 'spatial' | 'geological' | 'statistical' | 'physical';

export type TemporalConstraintType = 'lag' | 'periodicity' | 'causality-direction' | 'simultaneity';

export type SpatialConstraintType = 'distance' | 'scale' | 'locality' | 'coordinate-system';

export type GeologicalConstraintType = 'process-compatibility' | 'physics-consistency' | 'context-validity';

export type ModelSelectionCriterion = 'aic' | 'bic' | 'cross-validation' | 'bayes-factor';

// Utility Types

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface GeographicRegion {
  minLat: number;
  maxLat: number;
  minLon: number;
  maxLon: number;
}

export interface TimeRange {
  start: Date;
  end: Date;
}

// Export all types as a namespace for easier importing
export namespace CausalAnalysis {
  export type Variable = CausalVariable;
  export type Edge = CausalEdge;
  export type Hypothesis = CausalHypothesis;
  export type TestResult = HypothesisTestResult;
  export type Parameters = TestParameters;
}