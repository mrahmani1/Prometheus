/**
 * Comprehensive geological data types for the Prometheus AI Platform
 * These types ensure type safety and geological domain accuracy
 */

// Core geological coordinate and spatial types
export interface GeographicCoordinate {
  latitude: number;
  longitude: number;
  elevation?: number;
  datum?: string; // e.g., 'WGS84', 'NAD83'
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
  coordinateSystem?: string;
}

export interface GeographicBounds {
  north: number;
  south: number;
  east: number;
  west: number;
  minElevation?: number;
  maxElevation?: number;
}

export interface TemporalRange {
  startTime: Date;
  endTime: Date;
  resolution?: number; // in seconds
}

// Seismic data types
export interface SeismicEvent {
  id: string;
  eventId: string;
  timestamp: Date;
  location: Point3D;
  magnitude: number;
  depth: number;
  eventType: SeismicEventType;
  confidence: number;
  waveformData?: Float32Array;
  momentTensor?: MomentTensorSolution;
  sourceMechanism?: SourceMechanism;
  stationData?: SeismicStationData[];
}

export enum SeismicEventType {
  EARTHQUAKE = 'earthquake',
  VOLCANIC_TREMOR = 'volcanic_tremor',
  EXPLOSION = 'explosion',
  ROCKFALL = 'rockfall',
  LONG_PERIOD = 'long_period',
  VERY_LONG_PERIOD = 'very_long_period',
  HYBRID = 'hybrid'
}

export interface MomentTensorSolution {
  mxx: number;
  myy: number;
  mzz: number;
  mxy: number;
  mxz: number;
  myz: number;
  magnitude: number;
  strike: number;
  dip: number;
  rake: number;
  pAxis: Point3D;
  tAxis: Point3D;
  nAxis: Point3D;
}

export interface SourceMechanism {
  strike1: number;
  dip1: number;
  rake1: number;
  strike2: number;
  dip2: number;
  rake2: number;
  pAxis: Point3D;
  tAxis: Point3D;
  nAxis: Point3D;
}

export interface SeismicStationData {
  stationId: string;
  networkCode: string;
  location: GeographicCoordinate;
  channels: SeismicChannel[];
  instrumentResponse?: InstrumentResponse;
}

export interface SeismicChannel {
  channelCode: string;
  sampleRate: number;
  sensitivity: number;
  units: string;
  orientation: Point3D;
}

export interface InstrumentResponse {
  poles: Complex[];
  zeros: Complex[];
  gain: number;
  sensitivity: number;
}

export interface Complex {
  real: number;
  imaginary: number;
}

// Volcanic monitoring types
export interface VolcanicState {
  id: string;
  volcanoId: string;
  timestamp: Date;
  alertLevel: AlertLevel;
  magmaChamberPressure: number;
  deformationRate: number;
  gasEmissions: GasComposition;
  thermalAnomalies: ThermalAnomaly[];
  eruptionProbability: number;
  calderaClassification?: CalderaClassification;
  supereruptionRisk?: SupereruptionRiskAssessment;
}

export enum AlertLevel {
  GREEN = 'green',
  YELLOW = 'yellow',
  ORANGE = 'orange',
  RED = 'red'
}

export interface GasComposition {
  co2: number; // ppm
  so2: number; // ppm
  h2s: number; // ppm
  h2o: number; // ppm
  temperature: number; // Celsius
  flux: number; // tons/day
  isotopicRatios?: IsotopicRatios;
}

export interface IsotopicRatios {
  delta13C: number;
  delta34S: number;
  he3He4: number;
}

export interface ThermalAnomaly {
  location: GeographicCoordinate;
  temperature: number;
  area: number; // square meters
  intensity: number;
  timestamp: Date;
  source: 'satellite' | 'ground' | 'airborne';
}

export interface CalderaClassification {
  type: CalderaType;
  diameter: number;
  volume: number;
  age: number; // years
  formationMechanism: FormationMechanism;
  resurgentDome?: ResurgentDome;
}

export enum CalderaType {
  SHIELD = 'shield',
  STRATOVOLCANO = 'stratovolcano',
  MAAR = 'maar',
  SUPERVOLCANO = 'supervolcano',
  NESTED = 'nested'
}

export enum FormationMechanism {
  COLLAPSE = 'collapse',
  EXPLOSION = 'explosion',
  SUBSIDENCE = 'subsidence',
  EROSION = 'erosion'
}

export interface ResurgentDome {
  location: GeographicCoordinate;
  upliftRate: number; // mm/year
  volume: number;
  composition: MagmaComposition;
}

export interface SupereruptionRiskAssessment {
  probability: number;
  timeframe: number; // years
  veiEstimate: number;
  impactRadius: number; // kilometers
  confidenceInterval: [number, number];
}

// Deformation and InSAR types
export interface DeformationData {
  location: GeographicCoordinate;
  displacement: Point3D;
  velocity: Point3D;
  timestamp: Date;
  uncertainty: Point3D;
  source: DeformationSource;
  processingLevel: ProcessingLevel;
}

export enum DeformationSource {
  INSAR = 'insar',
  GPS = 'gps',
  LEVELING = 'leveling',
  TILTMETER = 'tiltmeter',
  STRAINMETER = 'strainmeter'
}

export enum ProcessingLevel {
  RAW = 'raw',
  PROCESSED = 'processed',
  ANALYZED = 'analyzed',
  INTERPRETED = 'interpreted'
}

export interface InSARData {
  acquisitionDate: Date;
  satellite: string;
  track: number;
  frame: number;
  interferogram: InterferogramData;
  coherence: CoherenceData;
  unwrappedPhase: PhaseData;
  atmosphericCorrection?: AtmosphericCorrection;
}

export interface InterferogramData {
  wrappedPhase: Float32Array;
  amplitude: Float32Array;
  bounds: GeographicBounds;
  pixelSpacing: [number, number];
  wavelength: number;
}

export interface CoherenceData {
  values: Float32Array;
  threshold: number;
  bounds: GeographicBounds;
}

export interface PhaseData {
  values: Float32Array;
  referencePoint: GeographicCoordinate;
  bounds: GeographicBounds;
}

export interface AtmosphericCorrection {
  method: string;
  weatherModel: string;
  correction: Float32Array;
  uncertainty: Float32Array;
}

// Magmatic system types
export interface MagmaticSystem {
  systemId: string;
  parentVolcanicSystem: string;
  chambers: MagmaReservoir[];
  composition: MagmaComposition;
  thermalState: ThermalModel;
  seismicProperties: SeismicProperties;
  meltFractionModel: MeltFractionModel;
  validationMetrics: Record<string, number>;
}

export interface MagmaReservoir {
  id: string;
  location: Point3D;
  volume: number;
  pressure: number;
  temperature: number;
  composition: MagmaComposition;
  crystallinity: number;
  volatileContent: VolatileContent;
  connectivity: ReservoirConnectivity[];
}

export interface MagmaComposition {
  sio2: number;
  al2o3: number;
  fe2o3: number;
  mgo: number;
  cao: number;
  na2o: number;
  k2o: number;
  tio2: number;
  p2o5: number;
  mno: number;
  totalAlkalis: number;
  silicaIndex: number;
}

export interface VolatileContent {
  h2o: number;
  co2: number;
  so2: number;
  cl: number;
  f: number;
}

export interface ReservoirConnectivity {
  connectedReservoirId: string;
  connectionType: ConnectionType;
  flowRate: number;
  resistance: number;
}

export enum ConnectionType {
  DIKE = 'dike',
  CONDUIT = 'conduit',
  FRACTURE = 'fracture',
  PERMEABLE_ZONE = 'permeable_zone'
}

export interface ThermalModel {
  temperatureDistribution: Float32Array;
  heatFlow: Float32Array;
  thermalConductivity: Float32Array;
  heatCapacity: Float32Array;
  bounds: GeographicBounds;
  resolution: [number, number, number];
}

export interface SeismicProperties {
  pWaveVelocity: Float32Array;
  sWaveVelocity: Float32Array;
  density: Float32Array;
  attenuation: Float32Array;
  anisotropy?: AnisotropyTensor;
}

export interface AnisotropyTensor {
  c11: number;
  c12: number;
  c13: number;
  c22: number;
  c23: number;
  c33: number;
  c44: number;
  c55: number;
  c66: number;
}

export interface MeltFractionModel {
  meltFraction: Float32Array;
  solidusTemperature: Float32Array;
  liquidusTemperature: Float32Array;
  bounds: GeographicBounds;
  resolution: [number, number, number];
}

// Resource assessment types
export interface ResourceProspect {
  prospectId: string;
  location: GeographicBounds;
  resourceType: ResourceType;
  geologicalSetting: GeologicalSetting;
  geophysicalSignature: GeophysicalData;
  geochemicalSignature: GeochemicalData;
  prospectivityScore: number;
  confidenceInterval: [number, number];
  economicAssessment?: EconomicAnalysis;
}

export enum ResourceType {
  COPPER = 'copper',
  GOLD = 'gold',
  SILVER = 'silver',
  LITHIUM = 'lithium',
  RARE_EARTH = 'rare_earth',
  GEOTHERMAL = 'geothermal',
  HYDROCARBON = 'hydrocarbon',
  GROUNDWATER = 'groundwater'
}

export interface GeologicalSetting {
  tectonicSetting: TectonicSetting;
  lithology: LithologyUnit[];
  structuralFeatures: StructuralFeature[];
  alterationZones: AlterationZone[];
  age: GeologicalAge;
}

export enum TectonicSetting {
  CONVERGENT_MARGIN = 'convergent_margin',
  DIVERGENT_MARGIN = 'divergent_margin',
  TRANSFORM_FAULT = 'transform_fault',
  INTRAPLATE = 'intraplate',
  BACK_ARC = 'back_arc',
  FORE_ARC = 'fore_arc'
}

export interface LithologyUnit {
  name: string;
  rockType: RockType;
  composition: MineralComposition;
  texture: RockTexture;
  porosity: number;
  permeability: number;
  density: number;
  magneticSusceptibility: number;
}

export enum RockType {
  IGNEOUS = 'igneous',
  SEDIMENTARY = 'sedimentary',
  METAMORPHIC = 'metamorphic',
  VOLCANIC = 'volcanic',
  PLUTONIC = 'plutonic'
}

export interface MineralComposition {
  primaryMinerals: Mineral[];
  secondaryMinerals: Mineral[];
  accessoryMinerals: Mineral[];
}

export interface Mineral {
  name: string;
  formula: string;
  percentage: number;
  crystalSystem: CrystalSystem;
  hardness: number;
  density: number;
}

export enum CrystalSystem {
  CUBIC = 'cubic',
  TETRAGONAL = 'tetragonal',
  ORTHORHOMBIC = 'orthorhombic',
  HEXAGONAL = 'hexagonal',
  TRIGONAL = 'trigonal',
  MONOCLINIC = 'monoclinic',
  TRICLINIC = 'triclinic'
}

export interface RockTexture {
  grainSize: GrainSize;
  sorting: Sorting;
  roundness: Roundness;
  fabric: Fabric;
}

export enum GrainSize {
  VERY_FINE = 'very_fine',
  FINE = 'fine',
  MEDIUM = 'medium',
  COARSE = 'coarse',
  VERY_COARSE = 'very_coarse'
}

export enum Sorting {
  VERY_WELL_SORTED = 'very_well_sorted',
  WELL_SORTED = 'well_sorted',
  MODERATELY_SORTED = 'moderately_sorted',
  POORLY_SORTED = 'poorly_sorted',
  VERY_POORLY_SORTED = 'very_poorly_sorted'
}

export enum Roundness {
  ANGULAR = 'angular',
  SUBANGULAR = 'subangular',
  SUBROUNDED = 'subrounded',
  ROUNDED = 'rounded',
  WELL_ROUNDED = 'well_rounded'
}

export enum Fabric {
  MASSIVE = 'massive',
  FOLIATED = 'foliated',
  LINEATED = 'lineated',
  BANDED = 'banded',
  SCHISTOSE = 'schistose'
}

export interface StructuralFeature {
  type: StructureType;
  orientation: Orientation;
  length: number;
  displacement: number;
  age: GeologicalAge;
  kinematics: Kinematics;
}

export enum StructureType {
  FAULT = 'fault',
  FRACTURE = 'fracture',
  FOLD = 'fold',
  JOINT = 'joint',
  SHEAR_ZONE = 'shear_zone',
  DIKE = 'dike',
  SILL = 'sill'
}

export interface Orientation {
  strike: number;
  dip: number;
  dipDirection: number;
}

export interface Kinematics {
  movementType: MovementType;
  slipVector: Point3D;
  rotationAxis: Point3D;
  rotationAngle: number;
}

export enum MovementType {
  NORMAL = 'normal',
  REVERSE = 'reverse',
  STRIKE_SLIP = 'strike_slip',
  OBLIQUE = 'oblique',
  THRUST = 'thrust'
}

export interface AlterationZone {
  type: AlterationType;
  intensity: AlterationIntensity;
  mineralogy: MineralComposition;
  extent: GeographicBounds;
  temperature: number;
  pressure: number;
  fluidComposition: FluidComposition;
}

export enum AlterationType {
  HYDROTHERMAL = 'hydrothermal',
  WEATHERING = 'weathering',
  METAMORPHIC = 'metamorphic',
  METASOMATIC = 'metasomatic',
  DIAGENETIC = 'diagenetic'
}

export enum AlterationIntensity {
  WEAK = 'weak',
  MODERATE = 'moderate',
  STRONG = 'strong',
  INTENSE = 'intense',
  COMPLETE = 'complete'
}

export interface FluidComposition {
  salinity: number;
  ph: number;
  temperature: number;
  pressure: number;
  majorIons: IonConcentration[];
  traceElements: ElementConcentration[];
  isotopes: IsotopicComposition[];
}

export interface IonConcentration {
  ion: string;
  concentration: number;
  units: string;
}

export interface ElementConcentration {
  element: string;
  concentration: number;
  units: string;
}

export interface IsotopicComposition {
  isotope: string;
  ratio: number;
  uncertainty: number;
}

export interface GeologicalAge {
  minimumAge: number;
  maximumAge: number;
  bestEstimate: number;
  uncertainty: number;
  method: DatingMethod;
  units: AgeUnits;
}

export enum DatingMethod {
  RADIOMETRIC = 'radiometric',
  BIOSTRATIGRAPHIC = 'biostratigraphic',
  MAGNETOSTRATIGRAPHIC = 'magnetostratigraphic',
  RELATIVE = 'relative',
  ABSOLUTE = 'absolute'
}

export enum AgeUnits {
  YEARS = 'years',
  THOUSAND_YEARS = 'thousand_years',
  MILLION_YEARS = 'million_years',
  BILLION_YEARS = 'billion_years'
}

export interface GeophysicalData {
  gravity: GravityData;
  magnetic: MagneticData;
  electromagnetic: ElectromagneticData;
  seismic: SeismicSurveyData;
  radiometric: RadiometricData;
}

export interface GravityData {
  bouguerAnomaly: Float32Array;
  freeAirAnomaly: Float32Array;
  terrain: Float32Array;
  bounds: GeographicBounds;
  resolution: [number, number];
  units: string;
}

export interface MagneticData {
  totalMagneticIntensity: Float32Array;
  magneticAnomaly: Float32Array;
  reducedToPoleMagnetic: Float32Array;
  bounds: GeographicBounds;
  resolution: [number, number];
  units: string;
}

export interface ElectromagneticData {
  conductivity: Float32Array;
  resistivity: Float32Array;
  chargeability: Float32Array;
  frequency: number;
  bounds: GeographicBounds;
  resolution: [number, number];
}

export interface SeismicSurveyData {
  reflectionData: ReflectionData;
  refractionData: RefractionData;
  velocityModel: VelocityModel;
  bounds: GeographicBounds;
}

export interface ReflectionData {
  travelTimes: Float32Array;
  amplitudes: Float32Array;
  frequencies: Float32Array;
  offsets: Float32Array;
}

export interface RefractionData {
  firstArrivals: Float32Array;
  velocities: Float32Array;
  depths: Float32Array;
}

export interface VelocityModel {
  pWaveVelocity: Float32Array;
  sWaveVelocity: Float32Array;
  density: Float32Array;
  bounds: GeographicBounds;
  resolution: [number, number, number];
}

export interface RadiometricData {
  potassium: Float32Array;
  uranium: Float32Array;
  thorium: Float32Array;
  totalCount: Float32Array;
  bounds: GeographicBounds;
  resolution: [number, number];
}

export interface GeochemicalData {
  rockSamples: RockSample[];
  soilSamples: SoilSample[];
  waterSamples: WaterSample[];
  gasSamples: GasSample[];
  streamSediments: StreamSedimentSample[];
}

export interface RockSample {
  sampleId: string;
  location: GeographicCoordinate;
  rockType: RockType;
  majorElements: ElementConcentration[];
  traceElements: ElementConcentration[];
  rareEarthElements: ElementConcentration[];
  isotopes: IsotopicComposition[];
  mineralogy: MineralComposition;
  texture: RockTexture;
  alteration: AlterationZone;
}

export interface SoilSample {
  sampleId: string;
  location: GeographicCoordinate;
  depth: number;
  horizon: SoilHorizon;
  elements: ElementConcentration[];
  organicMatter: number;
  ph: number;
  moisture: number;
}

export enum SoilHorizon {
  O = 'O',
  A = 'A',
  E = 'E',
  B = 'B',
  C = 'C',
  R = 'R'
}

export interface WaterSample {
  sampleId: string;
  location: GeographicCoordinate;
  waterType: WaterType;
  temperature: number;
  ph: number;
  conductivity: number;
  dissolvedOxygen: number;
  majorIons: IonConcentration[];
  traceElements: ElementConcentration[];
  isotopes: IsotopicComposition[];
}

export enum WaterType {
  SURFACE = 'surface',
  GROUNDWATER = 'groundwater',
  SPRING = 'spring',
  GEOTHERMAL = 'geothermal',
  HYDROTHERMAL = 'hydrothermal'
}

export interface GasSample {
  sampleId: string;
  location: GeographicCoordinate;
  gasType: GasType;
  composition: GasComposition;
  temperature: number;
  pressure: number;
  flowRate: number;
  isotopes: IsotopicComposition[];
}

export enum GasType {
  VOLCANIC = 'volcanic',
  FUMAROLIC = 'fumarolic',
  SOIL = 'soil',
  GROUNDWATER = 'groundwater',
  ATMOSPHERIC = 'atmospheric'
}

export interface StreamSedimentSample {
  sampleId: string;
  location: GeographicCoordinate;
  grainSize: GrainSize;
  elements: ElementConcentration[];
  heavyMinerals: Mineral[];
  organicContent: number;
}

export interface EconomicAnalysis {
  netPresentValue: number;
  internalRateOfReturn: number;
  paybackPeriod: number;
  capitalCosts: CapitalCosts;
  operatingCosts: OperatingCosts;
  revenue: Revenue;
  riskFactors: RiskFactor[];
  sensitivity: SensitivityAnalysis;
}

export interface CapitalCosts {
  exploration: number;
  development: number;
  infrastructure: number;
  equipment: number;
  contingency: number;
  total: number;
}

export interface OperatingCosts {
  mining: number;
  processing: number;
  transportation: number;
  administration: number;
  environmental: number;
  total: number;
}

export interface Revenue {
  commodityPrices: CommodityPrice[];
  production: ProductionSchedule[];
  totalRevenue: number;
}

export interface CommodityPrice {
  commodity: string;
  price: number;
  units: string;
  forecast: PriceForecast[];
}

export interface PriceForecast {
  year: number;
  price: number;
  confidence: number;
}

export interface ProductionSchedule {
  year: number;
  production: number;
  grade: number;
  recovery: number;
}

export interface RiskFactor {
  category: RiskCategory;
  description: string;
  probability: number;
  impact: number;
  mitigation: string;
}

export enum RiskCategory {
  GEOLOGICAL = 'geological',
  TECHNICAL = 'technical',
  ECONOMIC = 'economic',
  ENVIRONMENTAL = 'environmental',
  POLITICAL = 'political',
  SOCIAL = 'social'
}

export interface SensitivityAnalysis {
  variables: SensitivityVariable[];
  scenarios: Scenario[];
  tornadoDiagram: TornadoData[];
}

export interface SensitivityVariable {
  name: string;
  baseCase: number;
  lowCase: number;
  highCase: number;
  distribution: ProbabilityDistribution;
}

export interface ProbabilityDistribution {
  type: DistributionType;
  parameters: Record<string, number>;
}

export enum DistributionType {
  NORMAL = 'normal',
  LOGNORMAL = 'lognormal',
  UNIFORM = 'uniform',
  TRIANGULAR = 'triangular',
  BETA = 'beta'
}

export interface Scenario {
  name: string;
  variables: Record<string, number>;
  npv: number;
  probability: number;
}

export interface TornadoData {
  variable: string;
  lowImpact: number;
  highImpact: number;
  range: number;
}

// Data quality and validation types
export interface QualityMetrics {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  validity: number;
  uniqueness: number;
  overall: number;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  qualityScore: number;
}

export interface ValidationError {
  field: string;
  message: string;
  severity: ErrorSeverity;
  code: string;
}

export interface ValidationWarning {
  field: string;
  message: string;
  suggestion: string;
}

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Real-time streaming types
export interface StreamEvent {
  eventId: string;
  eventType: StreamEventType;
  timestamp: Date;
  sourceId: string;
  data: Record<string, unknown>;
  metadata: Record<string, unknown>;
  location?: Point3D;
  processingStatus: ProcessingStatus;
}

export enum StreamEventType {
  SEISMIC_EVENT = 'seismic_event',
  DEFORMATION_UPDATE = 'deformation_update',
  GAS_MEASUREMENT = 'gas_measurement',
  THERMAL_ANOMALY = 'thermal_anomaly',
  ALERT_TRIGGERED = 'alert_triggered',
  MODEL_UPDATE = 'model_update'
}

export enum ProcessingStatus {
  RECEIVED = 'received',
  PROCESSING = 'processing',
  PROCESSED = 'processed',
  ERROR = 'error',
  ARCHIVED = 'archived'
}

// User interface and visualization types
export interface VisualizationConfig {
  type: VisualizationType;
  dataSource: string;
  parameters: Record<string, unknown>;
  styling: VisualizationStyle;
  interactivity: InteractivityConfig;
}

export enum VisualizationType {
  MAP_2D = 'map_2d',
  MAP_3D = 'map_3d',
  CROSS_SECTION = 'cross_section',
  TIME_SERIES = 'time_series',
  SCATTER_PLOT = 'scatter_plot',
  HISTOGRAM = 'histogram',
  CONTOUR_MAP = 'contour_map',
  VOLUME_RENDERING = 'volume_rendering'
}

export interface VisualizationStyle {
  colorScheme: ColorScheme;
  opacity: number;
  lineWidth: number;
  pointSize: number;
  font: FontConfig;
}

export interface ColorScheme {
  name: string;
  colors: string[];
  continuous: boolean;
  reversed: boolean;
}

export interface FontConfig {
  family: string;
  size: number;
  weight: string;
  color: string;
}

export interface InteractivityConfig {
  zoomable: boolean;
  pannable: boolean;
  selectable: boolean;
  hoverable: boolean;
  clickable: boolean;
  brushable: boolean;
}

// Comprehensive geological data container
export interface GeologicalData {
  id: string;
  timestamp: Date;
  location: GeographicCoordinate;
  dataType: GeologicalDataType;
  seismicEvents?: SeismicEvent[];
  volcanicStates?: VolcanicState[];
  deformationData?: DeformationData[];
  resourceProspects?: ResourceProspect[];
  magmaticSystems?: MagmaticSystem[];
  qualityMetrics: QualityMetrics;
  metadata: Record<string, unknown>;
}

export enum GeologicalDataType {
  SEISMIC = 'seismic',
  VOLCANIC = 'volcanic',
  DEFORMATION = 'deformation',
  RESOURCE = 'resource',
  MAGMATIC = 'magmatic',
  GEOCHEMICAL = 'geochemical',
  GEOPHYSICAL = 'geophysical'
}

// 3D Geological Model
export interface GeologicalModel3D {
  id: string;
  name: string;
  bounds: GeographicBounds;
  resolution: [number, number, number];
  layers: GeologicalLayer3D[];
  structures: StructuralFeature[];
  properties: ModelProperties;
  metadata: Record<string, unknown>;
}

export interface GeologicalLayer3D {
  id: string;
  name: string;
  lithology: LithologyUnit;
  geometry: LayerGeometry;
  properties: LayerProperties;
  age: GeologicalAge;
}

export interface LayerGeometry {
  vertices: Float32Array;
  faces: Uint32Array;
  normals: Float32Array;
  textureCoordinates?: Float32Array;
}

export interface LayerProperties {
  density: number;
  porosity: number;
  permeability: number;
  elasticModulus: number;
  poissonRatio: number;
  thermalConductivity: number;
}

export interface ModelProperties {
  coordinateSystem: string;
  verticalDatum: string;
  units: string;
  accuracy: number;
  confidence: number;
  validationResults: ValidationResult;
}

// Error handling and user feedback types
export interface GeologicalError extends Error {
  code: string;
  severity: ErrorSeverity;
  context: Record<string, unknown>;
  suggestions: string[];
  recoverable: boolean;
}

export interface UserFeedback {
  type: FeedbackType;
  message: string;
  duration?: number;
  actions?: FeedbackAction[];
}

export enum FeedbackType {
  SUCCESS = 'success',
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error'
}

export interface FeedbackAction {
  label: string;
  action: () => void;
  primary?: boolean;
}