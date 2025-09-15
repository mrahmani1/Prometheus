// Immersive VR/AR Types for Geological Exploration

export interface GeologicalLayer {
  id: string;
  name: string;
  type: 'sedimentary' | 'igneous' | 'metamorphic' | 'structural';
  position: {
    x: number;
    y: number;
    z: number;
  };
  dimensions: {
    width?: number;
    height?: number;
    depth?: number;
    radius?: number;
    topRadius?: number;
    bottomRadius?: number;
  };
  properties: {
    color?: number;
    age?: string;
    hardness?: number;
    density?: number;
    composition?: string[];
    [key: string]: any;
  };
  features?: GeologicalFeature[];
}

export interface GeologicalFeature {
  type: 'fault' | 'fold' | 'mineral_vein' | 'joint' | 'bedding';
  position: {
    x: number;
    y: number;
    z: number;
  };
  points?: Array<{ x: number; y: number; z: number }>;
  length?: number;
  height?: number;
  mineralColor?: number;
  properties?: Record<string, any>;
}

export interface VRSession {
  id: string;
  userId: string;
  startTime: Date;
  endTime?: Date;
  interactions: VRInteraction[];
  settings: VRSettings;
}

export interface VRSettings {
  hapticFeedback: boolean;
  audioFeedback: boolean;
  gestureRecognition: boolean;
  voiceControl: boolean;
  collaborativeMode: boolean;
  accessibilityMode: boolean;
}

export interface VRInteraction {
  id: string;
  type: 'select' | 'grab' | 'point' | 'gesture' | 'voice';
  timestamp: Date;
  position: {
    x: number;
    y: number;
    z: number;
  };
  target?: string;
  data?: any;
}

export interface HapticFeedback {
  intensity: number;
  duration: number;
  pattern?: string;
  timestamp?: number;
}

export interface HapticPattern {
  name: string;
  type: 'contact' | 'texture' | 'resistance' | 'vibration';
  intensity: number;
  frequency: number;
  duration: number;
  waveform: 'sine' | 'square' | 'triangle' | 'sawtooth' | 'random';
}

export interface HapticDevice {
  id: string;
  type: 'gamepad' | 'haptic_glove' | 'force_feedback' | 'ultrasound';
  name: string;
  capabilities: {
    supportsVibration: boolean;
    supportsForce: boolean;
    supportsTexture: boolean;
    maxIntensity: number;
    minDuration: number;
    maxDuration: number;
  };
  connected: boolean;
}

export interface GeologicalMaterial {
  type: string;
  hardness: number;
  density: number;
  texture?: string;
  composition?: string[];
  properties?: Record<string, any>;
}

// AR Field Data Collection Types
export interface FieldData {
  id: string;
  type: 'sample' | 'measurement' | 'observation' | 'photo' | 'note';
  position: {
    x: number;
    y: number;
    z: number;
  };
  gpsCoordinates?: {
    latitude: number;
    longitude: number;
    altitude?: number;
    accuracy: number;
  };
  timestamp: Date;
  photo?: string; // Base64 encoded image
  properties: Record<string, any>;
}

export interface ARSession {
  id: string;
  userId: string;
  startTime: Date;
  endTime?: Date;
  fieldData: FieldData[];
  markers: ARMarker[];
  settings: ARSettings;
}

export interface ARSettings {
  showGeologicalLayers: boolean;
  enableGPS: boolean;
  enableCamera: boolean;
  markerPersistence: boolean;
  dataSync: boolean;
}

export interface ARMarker {
  id: string;
  type: 'sample' | 'measurement' | 'observation' | 'hazard' | 'poi';
  position: {
    x: number;
    y: number;
    z: number;
  };
  data: any;
  timestamp: Date;
  persistent: boolean;
}

export interface GeolocationData {
  latitude: number;
  longitude: number;
  altitude?: number;
  accuracy: number;
  heading?: number;
  speed?: number;
  timestamp: Date;
}

// Collaborative VR Types
export interface CollaborativeSession {
  id: string;
  name: string;
  hostUserId: string;
  participants: VRUser[];
  sharedData: SharedData;
  settings: CollaborativeSettings;
  createdAt: Date;
  active: boolean;
}

export interface VRUser {
  id: string;
  name: string;
  avatar: UserAvatar;
  position: {
    x: number;
    y: number;
    z: number;
  };
  rotation: {
    x: number;
    y: number;
    z: number;
  };
  headPosition: {
    x: number;
    y: number;
    z: number;
  };
  leftHandPosition: {
    x: number;
    y: number;
    z: number;
  };
  rightHandPosition: {
    x: number;
    y: number;
    z: number;
  };
  isActive: boolean;
  permissions: UserPermissions;
}

export interface UserAvatar {
  color: number;
  model?: string;
  accessories?: string[];
}

export interface UserPermissions {
  canEdit: boolean;
  canAnnotate: boolean;
  canShare: boolean;
  canInvite: boolean;
}

export interface SharedData {
  annotations: SharedAnnotation[];
  measurements: SharedMeasurement[];
  models: SharedModel[];
  discussions: Discussion[];
}

export interface SharedAnnotation {
  id: string;
  userId: string;
  userName: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  text: string;
  type: 'note' | 'measurement' | 'highlight' | 'question';
  timestamp: Date;
  color: number;
  replies?: AnnotationReply[];
}

export interface AnnotationReply {
  id: string;
  userId: string;
  userName: string;
  text: string;
  timestamp: Date;
}

export interface SharedMeasurement {
  id: string;
  userId: string;
  type: 'distance' | 'angle' | 'area' | 'volume';
  points: Array<{ x: number; y: number; z: number }>;
  value: number;
  unit: string;
  timestamp: Date;
}

export interface SharedModel {
  id: string;
  userId: string;
  name: string;
  type: 'geological' | 'structural' | 'analytical';
  data: any;
  position: {
    x: number;
    y: number;
    z: number;
  };
  scale: {
    x: number;
    y: number;
    z: number;
  };
  timestamp: Date;
}

export interface Discussion {
  id: string;
  title: string;
  participants: string[];
  messages: DiscussionMessage[];
  createdAt: Date;
  active: boolean;
}

export interface DiscussionMessage {
  id: string;
  userId: string;
  userName: string;
  message: string;
  timestamp: Date;
  type: 'text' | 'voice' | 'annotation';
}

export interface CollaborativeSettings {
  maxParticipants: number;
  allowVoiceChat: boolean;
  allowScreenShare: boolean;
  requirePermissions: boolean;
  recordSession: boolean;
}

export interface VoiceChat {
  enabled: boolean;
  participants: string[];
  spatialAudio: boolean;
  noiseReduction: boolean;
  pushToTalk: boolean;
}

// Gesture and Voice Control Types
export interface GestureCommand {
  type: 'gesture';
  gesture: string;
  action: string;
  confidence: number;
  timestamp: number;
  description: string;
}

export interface VoiceCommand {
  type: 'voice';
  phrase: string;
  action: string;
  intent?: string;
  confidence: number;
  timestamp: number;
  transcript: string;
  entities?: Array<{ type: string; value: string }>;
}

export interface HandTracking {
  enabled: boolean;
  landmarks: HandLandmark[];
  gestures: RecognizedGesture[];
  confidence: number;
}

export interface HandLandmark {
  id: number;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface RecognizedGesture {
  name: string;
  confidence: number;
  timestamp: number;
  landmarks: number[];
}

export interface SpeechRecognition {
  enabled: boolean;
  language: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
}

export interface GestureRecognition {
  enabled: boolean;
  sensitivity: number;
  patterns: GesturePattern[];
  currentGesture?: string;
}

export interface GesturePattern {
  name: string;
  landmarks: number[];
  threshold: number;
  description: string;
}

// Immersive Data Analysis Types
export interface DataVisualization {
  id: string;
  type: '3d_scatter' | 'heatmap' | 'volume' | 'network' | 'temporal';
  data: any[];
  position: {
    x: number;
    y: number;
    z: number;
  };
  scale: {
    x: number;
    y: number;
    z: number;
  };
  colorScheme: string;
  interactive: boolean;
}

export interface ImmersiveAnalysis {
  id: string;
  type: 'statistical' | 'spatial' | 'temporal' | 'comparative';
  inputData: any[];
  results: AnalysisResult;
  visualization: DataVisualization;
  timestamp: Date;
}

export interface AnalysisResult {
  id: string;
  type: string;
  title: string;
  data: Record<string, any>;
  visualization: string;
  timestamp: Date;
}

export interface SpatialQuery {
  id: string;
  region: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  };
  pointCount: number;
  timestamp: Date;
}

export interface DataManipulation {
  id: string;
  type: 'filter' | 'transform' | 'aggregate' | 'interpolate';
  parameters: Record<string, any>;
  inputData: any[];
  outputData: any[];
  timestamp: Date;
}

// Holographic Display Types
export interface HolographicProjection {
  id: string;
  type: 'volumetric' | 'planar' | 'cylindrical' | 'spherical';
  bounds: {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
  };
  resolution: {
    width: number;
    height: number;
    depth: number;
  };
  active: boolean;
}

export interface SpatialComputing {
  enabled: boolean;
  trackingActive: boolean;
  anchors: SpatialAnchor[];
  meshDetection: boolean;
  planeDetection: boolean;
}

export interface SpatialAnchor {
  id: string;
  position: {
    x: number;
    y: number;
    z: number;
  };
  rotation: {
    x: number;
    y: number;
    z: number;
  };
  persistent: boolean;
  holograms: string[];
}

export interface MixedRealityInterface {
  spatialMapping: boolean;
  handTracking: boolean;
  eyeTracking: boolean;
  voiceCommands: boolean;
  gestureRecognition: boolean;
}

export interface HologramData {
  id: string;
  type: 'geological_model' | 'data_visualization' | 'interface_element';
  position: {
    x: number;
    y: number;
    z: number;
  };
  rotation: {
    x: number;
    y: number;
    z: number;
  };
  scale: {
    x: number;
    y: number;
    z: number;
  };
  data: any;
  opacity: number;
  interactive: boolean;
}

// Training and Education Types
export interface TrainingModule {
  id: string;
  title: string;
  description: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  duration: number; // minutes
  objectives: string[];
  prerequisites: string[];
  lessons: InteractiveLesson[];
  assessments: Assessment[];
}

export interface InteractiveLesson {
  id: string;
  title: string;
  content: string;
  objectives: string[];
  interactiveElements: InteractiveElement[];
  duration: number;
  completed: boolean;
}

export interface InteractiveElement {
  id: string;
  type: 'rock_sample' | 'geological_structure' | 'tool' | 'measurement' | 'quiz';
  position: {
    x: number;
    y: number;
    z: number;
  };
  data: any;
  interactions: string[];
  feedback: string[];
}

export interface Assessment {
  id: string;
  type: 'practical' | 'theoretical' | 'identification' | 'measurement';
  questions: AssessmentQuestion[];
  passingScore: number;
  timeLimit?: number;
}

export interface AssessmentQuestion {
  id: string;
  type: 'multiple_choice' | 'identification' | 'measurement' | 'sequence';
  question: string;
  options?: string[];
  correctAnswer: any;
  explanation: string;
  points: number;
}

export interface LearningProgress {
  userId: string;
  completedLessons: string[];
  currentModule: string;
  skillLevel: 'beginner' | 'intermediate' | 'advanced' | 'expert';
  totalScore: number;
  assessmentResults: AssessmentResult[];
  timeSpent: number; // minutes
  achievements: Achievement[];
}

export interface AssessmentResult {
  id: string;
  assessmentId: string;
  userId: string;
  score: number;
  passed: boolean;
  answers: Array<[string, any]>;
  completedAt: Date;
  timeSpent: number;
}

export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  unlockedAt: Date;
  category: 'learning' | 'exploration' | 'collaboration' | 'mastery';
}

export interface EducationalContent {
  id: string;
  type: 'lesson' | 'tutorial' | 'simulation' | 'assessment';
  title: string;
  content: string;
  mediaAssets: MediaAsset[];
  interactiveElements: InteractiveElement[];
  difficulty: number;
  tags: string[];
}

export interface MediaAsset {
  id: string;
  type: 'image' | 'video' | '3d_model' | 'audio' | 'animation';
  url: string;
  metadata: Record<string, any>;
}

// Accessibility Types
export interface AccessibilitySettings {
  voiceNavigation: boolean;
  highContrast: boolean;
  motionReduction: boolean;
  cognitiveSupport: boolean;
  audioDescriptions: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'extra-large';
  colorBlindnessSupport: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';
  customizations: Record<string, any>;
}

export interface VoiceNavigation {
  enabled: boolean;
  language: string;
  commands: VoiceNavigationCommand[];
  sensitivity: number;
  continuous: boolean;
}

export interface VoiceNavigationCommand {
  phrase: string;
  action: string;
  description: string;
  category: 'navigation' | 'interaction' | 'settings' | 'help';
}

export interface HighContrastMode {
  enabled: boolean;
  backgroundColor: string;
  foregroundColor: string;
  accentColor: string;
  contrastRatio: number;
}

export interface MotionSettings {
  reduceMotion: boolean;
  animationSpeed: number;
  disableAutoRotation: boolean;
  reduceParallax: boolean;
  vestibularSafety: boolean;
}

export interface CognitiveSupport {
  enabled: boolean;
  showInstructions: boolean;
  simplifyInterface: boolean;
  provideHints: boolean;
  extendedTimeouts: boolean;
  confirmActions: boolean;
}

// Event Types for Redux Actions
export interface ImmersiveEvent {
  type: string;
  payload: any;
  timestamp: number;
  userId?: string;
  sessionId?: string;
}

export interface VRSessionEvent extends ImmersiveEvent {
  type: 'VR_SESSION_START' | 'VR_SESSION_END' | 'VR_INTERACTION' | 'VR_SETTINGS_CHANGE';
}

export interface ARSessionEvent extends ImmersiveEvent {
  type: 'AR_SESSION_START' | 'AR_SESSION_END' | 'AR_MARKER_ADD' | 'AR_DATA_CAPTURE';
}

export interface CollaborativeEvent extends ImmersiveEvent {
  type: 'USER_JOIN' | 'USER_LEAVE' | 'ANNOTATION_ADD' | 'VOICE_CHAT_START' | 'VOICE_CHAT_END';
}

export interface AccessibilityEvent extends ImmersiveEvent {
  type: 'ACCESSIBILITY_ENABLE' | 'ACCESSIBILITY_DISABLE' | 'VOICE_COMMAND' | 'FOCUS_CHANGE';
}