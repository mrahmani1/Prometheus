/**
 * API types for the Prometheus AI Platform
 * Defines interfaces for API communication and data exchange
 */

import {
  SeismicEvent,
  VolcanicState,
  DeformationData,
  ResourceProspect,
  // MagmaticSystem, // Unused
  // GeologicalAge, // Unused
  QualityMetrics,
  // ValidationResult, // Unused
  StreamEvent
} from './geological';

// Base API response types
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: ApiError;
  metadata?: ResponseMetadata;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
  requestId: string;
}

export interface ResponseMetadata {
  requestId: string;
  timestamp: string;
  processingTime: number;
  version: string;
  pagination?: PaginationInfo;
}

export interface PaginationInfo {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

// Authentication and authorization types
export interface AuthRequest {
  username: string;
  password: string;
  mfaToken?: string;
}

export interface AuthResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  user: UserProfile;
}

export interface UserProfile {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  roles: UserRole[];
  permissions: Permission[];
  preferences: UserPreferences;
  lastLogin: string;
}

export interface UserRole {
  id: string;
  name: string;
  description: string;
  permissions: Permission[];
}

export interface Permission {
  id: string;
  name: string;
  resource: string;
  action: string;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  units: UnitSystem;
  notifications: NotificationSettings;
  dashboard: DashboardPreferences;
}

export interface UnitSystem {
  length: 'metric' | 'imperial';
  temperature: 'celsius' | 'fahrenheit' | 'kelvin';
  pressure: 'pascal' | 'bar' | 'psi';
  mass: 'kilogram' | 'pound';
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  sms: boolean;
  alertLevels: string[];
}

export interface DashboardPreferences {
  layout: string;
  widgets: WidgetConfig[];
  refreshInterval: number;
}

export interface WidgetConfig {
  id: string;
  type: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  config: Record<string, unknown>;
}

// Data query and filtering types
export interface DataQuery {
  filters: QueryFilter[];
  sorting: SortConfig[];
  pagination: PaginationConfig;
  fields?: string[];
  aggregations?: AggregationConfig[];
}

export interface QueryFilter {
  field: string;
  operator: FilterOperator;
  value: unknown;
  logicalOperator?: LogicalOperator;
}

export enum FilterOperator {
  EQUALS = 'eq',
  NOT_EQUALS = 'ne',
  GREATER_THAN = 'gt',
  GREATER_THAN_OR_EQUAL = 'gte',
  LESS_THAN = 'lt',
  LESS_THAN_OR_EQUAL = 'lte',
  IN = 'in',
  NOT_IN = 'nin',
  CONTAINS = 'contains',
  STARTS_WITH = 'starts_with',
  ENDS_WITH = 'ends_with',
  REGEX = 'regex',
  IS_NULL = 'is_null',
  IS_NOT_NULL = 'is_not_null'
}

export enum LogicalOperator {
  AND = 'and',
  OR = 'or',
  NOT = 'not'
}

export interface SortConfig {
  field: string;
  direction: 'asc' | 'desc';
}

export interface PaginationConfig {
  page: number;
  pageSize: number;
}

export interface AggregationConfig {
  field: string;
  operation: AggregationOperation;
  alias?: string;
}

export enum AggregationOperation {
  COUNT = 'count',
  SUM = 'sum',
  AVG = 'avg',
  MIN = 'min',
  MAX = 'max',
  STDDEV = 'stddev',
  VARIANCE = 'variance'
}

// Seismic data API types
export interface SeismicDataRequest {
  startTime: string;
  endTime: string;
  boundingBox?: BoundingBox;
  magnitudeRange?: [number, number];
  depthRange?: [number, number];
  eventTypes?: string[];
  stations?: string[];
  includeWaveforms?: boolean;
}

export interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

export interface SeismicDataResponse {
  events: SeismicEvent[];
  summary: SeismicSummary;
  qualityMetrics: QualityMetrics;
}

export interface SeismicSummary {
  totalEvents: number;
  magnitudeDistribution: MagnitudeDistribution;
  depthDistribution: DepthDistribution;
  temporalDistribution: TemporalDistribution;
  spatialClustering: SpatialCluster[];
}

export interface MagnitudeDistribution {
  bins: number[];
  counts: number[];
  bValue: number;
  completeness: number;
}

export interface DepthDistribution {
  bins: number[];
  counts: number[];
  averageDepth: number;
  standardDeviation: number;
}

export interface TemporalDistribution {
  hourly: number[];
  daily: number[];
  monthly: number[];
  seasonal: number[];
}

export interface SpatialCluster {
  centroid: [number, number];
  radius: number;
  eventCount: number;
  significance: number;
}

// Volcanic monitoring API types
export interface VolcanicMonitoringRequest {
  volcanoIds?: string[];
  alertLevels?: string[];
  startTime?: string;
  endTime?: string;
  includeForecasts?: boolean;
  includeHistorical?: boolean;
}

export interface VolcanicMonitoringResponse {
  volcanoes: VolcanicState[];
  alerts: VolcanicAlert[];
  forecasts: EruptionForecast[];
  trends: VolcanicTrend[];
}

export interface VolcanicAlert {
  id: string;
  volcanoId: string;
  alertLevel: string;
  message: string;
  timestamp: string;
  severity: string;
  acknowledged: boolean;
  actions: AlertAction[];
}

export interface AlertAction {
  id: string;
  description: string;
  priority: number;
  deadline?: string;
  assignedTo?: string;
  status: 'pending' | 'in_progress' | 'completed';
}

export interface EruptionForecast {
  volcanoId: string;
  probability: number;
  timeWindow: TimeWindow;
  veiEstimate: number;
  confidence: number;
  methodology: string;
  lastUpdated: string;
  scenarios: EruptionScenario[];
}

export interface TimeWindow {
  start: string;
  end: string;
  duration: number;
  units: string;
}

export interface EruptionScenario {
  id: string;
  probability: number;
  vei: number;
  impactRadius: number;
  ashfallMap?: string;
  pyroclasticFlowMap?: string;
  laharMap?: string;
}

export interface VolcanicTrend {
  volcanoId: string;
  parameter: string;
  trend: 'increasing' | 'decreasing' | 'stable' | 'oscillating';
  significance: number;
  timespan: string;
  dataPoints: TrendDataPoint[];
}

export interface TrendDataPoint {
  timestamp: string;
  value: number;
  uncertainty?: number;
}

// Deformation analysis API types
export interface DeformationAnalysisRequest {
  region: BoundingBox;
  startTime: string;
  endTime: string;
  sources?: string[];
  processingLevel?: string;
  referenceFrame?: string;
}

export interface DeformationAnalysisResponse {
  data: DeformationData[];
  analysis: DeformationAnalysis;
  visualizations: DeformationVisualization[];
}

export interface DeformationAnalysis {
  totalDisplacement: number;
  maximumVelocity: number;
  principalStrains: PrincipalStrain[];
  sourceModels: SourceModel[];
  uncertaintyAnalysis: UncertaintyAnalysis;
}

export interface PrincipalStrain {
  magnitude: number;
  direction: number;
  type: 'extension' | 'compression';
  location: [number, number];
}

export interface SourceModel {
  type: 'mogi' | 'okada' | 'yang' | 'mctighe';
  parameters: Record<string, number>;
  location: [number, number, number];
  confidence: number;
  residuals: number[];
}

export interface UncertaintyAnalysis {
  method: string;
  confidenceIntervals: ConfidenceInterval[];
  sensitivityAnalysis: SensitivityResult[];
}

export interface ConfidenceInterval {
  parameter: string;
  level: number;
  lowerBound: number;
  upperBound: number;
}

export interface SensitivityResult {
  parameter: string;
  sensitivity: number;
  correlation: number;
}

export interface DeformationVisualization {
  type: 'vector_field' | 'contour_map' | 'time_series' | '3d_surface';
  data: string; // Base64 encoded or URL
  metadata: VisualizationMetadata;
}

export interface VisualizationMetadata {
  title: string;
  description: string;
  units: string;
  colorScale: string;
  projection: string;
  bounds: BoundingBox;
}

// Resource assessment API types
export interface ResourceAssessmentRequest {
  region: BoundingBox;
  resourceTypes?: string[];
  confidenceThreshold?: number;
  includeEconomics?: boolean;
  includeUncertainty?: boolean;
}

export interface ResourceAssessmentResponse {
  prospects: ResourceProspect[];
  summary: AssessmentSummary;
  recommendations: AssessmentRecommendation[];
}

export interface AssessmentSummary {
  totalProspects: number;
  resourceDistribution: ResourceDistribution[];
  confidenceDistribution: ConfidenceDistribution;
  economicViability: EconomicViability;
}

export interface ResourceDistribution {
  resourceType: string;
  count: number;
  averageGrade: number;
  totalTonnage: number;
  averageConfidence: number;
}

export interface ConfidenceDistribution {
  high: number;
  medium: number;
  low: number;
}

export interface EconomicViability {
  viable: number;
  marginal: number;
  uneconomic: number;
  averageNPV: number;
}

export interface AssessmentRecommendation {
  prospectId: string;
  priority: 'high' | 'medium' | 'low';
  recommendation: string;
  nextSteps: string[];
  estimatedCost: number;
  timeframe: string;
}

// Model management API types
export interface ModelRequest {
  modelType: string;
  parameters: Record<string, unknown>;
  trainingData?: string;
  validationData?: string;
  hyperparameters?: Record<string, unknown>;
}

export interface ModelResponse {
  modelId: string;
  status: ModelStatus;
  metrics: ModelMetrics;
  predictions?: ModelPrediction[];
  artifacts?: ModelArtifact[];
}

export enum ModelStatus {
  TRAINING = 'training',
  TRAINED = 'trained',
  DEPLOYED = 'deployed',
  FAILED = 'failed',
  DEPRECATED = 'deprecated'
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  rmse?: number;
  mae?: number;
  r2?: number;
  customMetrics?: Record<string, number>;
}

export interface ModelPrediction {
  input: Record<string, unknown>;
  output: Record<string, unknown>;
  confidence: number;
  uncertainty?: Record<string, number>;
  explanation?: string;
}

export interface ModelArtifact {
  type: 'model' | 'weights' | 'config' | 'metadata';
  url: string;
  size: number;
  checksum: string;
  createdAt: string;
}

// Real-time streaming API types
export interface StreamSubscription {
  topics: string[];
  filters?: StreamFilter[];
  batchSize?: number;
  maxLatency?: number;
}

export interface StreamFilter {
  field: string;
  operator: FilterOperator;
  value: unknown;
}

export interface StreamMessage {
  topic: string;
  partition: number;
  offset: number;
  timestamp: string;
  key?: string;
  value: StreamEvent;
  headers?: Record<string, string>;
}

export interface StreamStatus {
  connected: boolean;
  subscriptions: string[];
  messageCount: number;
  lastMessage?: string;
  errors: StreamError[];
}

export interface StreamError {
  code: string;
  message: string;
  timestamp: string;
  recoverable: boolean;
}

// Data export and import types
export interface ExportRequest {
  format: ExportFormat;
  data: string[];
  filters?: QueryFilter[];
  compression?: CompressionType;
  includeMetadata?: boolean;
}

export enum ExportFormat {
  CSV = 'csv',
  JSON = 'json',
  GEOJSON = 'geojson',
  SHAPEFILE = 'shapefile',
  NETCDF = 'netcdf',
  HDF5 = 'hdf5',
  PARQUET = 'parquet'
}

export enum CompressionType {
  NONE = 'none',
  GZIP = 'gzip',
  ZIP = 'zip',
  BZIP2 = 'bzip2'
}

export interface ExportResponse {
  exportId: string;
  status: ExportStatus;
  downloadUrl?: string;
  fileSize?: number;
  expiresAt?: string;
}

export enum ExportStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  EXPIRED = 'expired'
}

export interface ImportRequest {
  format: ExportFormat;
  fileUrl: string;
  validation?: ValidationConfig;
  transformation?: TransformationConfig;
}

export interface ValidationConfig {
  strict: boolean;
  skipErrors: boolean;
  maxErrors: number;
  customRules?: ValidationRule[];
}

export interface ValidationRule {
  field: string;
  rule: string;
  parameters?: Record<string, unknown>;
}

export interface TransformationConfig {
  coordinateSystem?: string;
  unitConversions?: UnitConversion[];
  fieldMappings?: FieldMapping[];
}

export interface UnitConversion {
  field: string;
  fromUnit: string;
  toUnit: string;
}

export interface FieldMapping {
  sourceField: string;
  targetField: string;
  transformation?: string;
}

export interface ImportResponse {
  importId: string;
  status: ImportStatus;
  recordsProcessed: number;
  recordsImported: number;
  errors: ImportError[];
  warnings: ImportWarning[];
}

export enum ImportStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

export interface ImportError {
  line: number;
  field: string;
  message: string;
  value: unknown;
}

export interface ImportWarning {
  line: number;
  field: string;
  message: string;
  value: unknown;
}

// System health and monitoring types
export interface HealthCheckResponse {
  status: HealthStatus;
  services: ServiceHealth[];
  timestamp: string;
  version: string;
}

export enum HealthStatus {
  HEALTHY = 'healthy',
  DEGRADED = 'degraded',
  UNHEALTHY = 'unhealthy'
}

export interface ServiceHealth {
  name: string;
  status: HealthStatus;
  responseTime: number;
  lastCheck: string;
  details?: Record<string, unknown>;
}

export interface SystemMetrics {
  cpu: ResourceMetric;
  memory: ResourceMetric;
  disk: ResourceMetric;
  network: NetworkMetric;
  database: DatabaseMetric[];
  cache: CacheMetric;
}

export interface ResourceMetric {
  usage: number;
  available: number;
  total: number;
  percentage: number;
}

export interface NetworkMetric {
  bytesIn: number;
  bytesOut: number;
  packetsIn: number;
  packetsOut: number;
  errors: number;
}

export interface DatabaseMetric {
  name: string;
  connections: number;
  maxConnections: number;
  queryTime: number;
  slowQueries: number;
  size: number;
}

export interface CacheMetric {
  hits: number;
  misses: number;
  hitRate: number;
  size: number;
  maxSize: number;
  evictions: number;
}

// Configuration and settings types
export interface SystemConfig {
  api: ApiConfig;
  database: DatabaseConfig;
  cache: CacheConfig;
  monitoring: MonitoringConfig;
  security: SecurityConfig;
}

export interface ApiConfig {
  baseUrl: string;
  timeout: number;
  retryAttempts: number;
  rateLimiting: RateLimitConfig;
}

export interface RateLimitConfig {
  enabled: boolean;
  requestsPerMinute: number;
  burstSize: number;
}

export interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  poolSize: number;
  timeout: number;
}

export interface CacheConfig {
  enabled: boolean;
  ttl: number;
  maxSize: number;
  strategy: CacheStrategy;
}

export enum CacheStrategy {
  LRU = 'lru',
  LFU = 'lfu',
  FIFO = 'fifo',
  TTL = 'ttl'
}

export interface MonitoringConfig {
  enabled: boolean;
  interval: number;
  metrics: string[];
  alerting: AlertingConfig;
}

export interface AlertingConfig {
  enabled: boolean;
  channels: AlertChannel[];
  thresholds: AlertThreshold[];
}

export interface AlertChannel {
  type: 'email' | 'sms' | 'webhook' | 'slack';
  config: Record<string, unknown>;
}

export interface AlertThreshold {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'ne';
  value: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface SecurityConfig {
  authentication: AuthConfig;
  authorization: AuthzConfig;
  encryption: EncryptionConfig;
}

export interface AuthConfig {
  provider: string;
  tokenExpiry: number;
  refreshTokenExpiry: number;
  mfaRequired: boolean;
}

export interface AuthzConfig {
  rbacEnabled: boolean;
  defaultRole: string;
  permissionModel: string;
}

export interface EncryptionConfig {
  algorithm: string;
  keySize: number;
  saltRounds: number;
}