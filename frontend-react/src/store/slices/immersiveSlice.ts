import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import {
  VRSession,
  ARSession,
  CollaborativeSession,
  HapticFeedback,
  GestureRecognition,
  VoiceCommand,
  GestureCommand,
  ImmersiveAnalysis,
  // HolographicProjection, // Unused
  SpatialComputing,
  TrainingModule,
  LearningProgress,
  AccessibilitySettings,
  FieldData,
  SharedAnnotation,
  VRUser,
  SpatialQuery,
  DataVisualization,
  AssessmentResult,
  EducationalContent,
  VoiceChat,
  HapticDevice
} from '../../types/immersive';

interface ImmersiveState {
  // VR Session Management
  vrSession: VRSession | null;
  vrActive: boolean;
  vrSupported: boolean;
  
  // AR Session Management
  arSession: ARSession | null;
  arActive: boolean;
  arSupported: boolean;
  
  // Collaborative Features
  collaborativeSession: CollaborativeSession | null;
  connectedUsers: VRUser[];
  sharedAnnotations: SharedAnnotation[];
  voiceChat: VoiceChat | null;
  
  // Haptic Feedback
  hapticFeedback: HapticFeedback | null;
  hapticDevices: HapticDevice[];
  hapticEnabled: boolean;
  
  // Gesture Recognition
  gestureRecognition: GestureRecognition | null;
  currentGesture: string | null;
  gestureHistory: string[];
  
  // Voice Recognition
  voiceRecognition: {
    listening: boolean;
    transcript: string;
    confidence: number;
    error?: string;
  };
  voiceCommands: VoiceCommand[];
  
  // Immersive Data Analysis
  immersiveAnalysis: ImmersiveAnalysis | null;
  dataVisualizations: DataVisualization[];
  spatialQueries: SpatialQuery[];
  
  // Holographic Display
  holographicDisplay: {
    active: boolean;
    projectionType: string;
    hologramCount: number;
  };
  spatialComputing: SpatialComputing | null;
  
  // Training and Education
  trainingProgress: LearningProgress | null;
  currentTrainingModule: TrainingModule | null;
  educationalContent: EducationalContent[];
  assessmentResults: AssessmentResult[];
  
  // Accessibility
  accessibilitySettings: AccessibilitySettings;
  accessibilityActive: boolean;
  
  // Field Data Collection
  fieldData: FieldData[];
  currentLocation: {
    latitude: number;
    longitude: number;
    altitude?: number;
    accuracy: number;
  } | null;
  
  // System Status
  loading: boolean;
  error: string | null;
  lastUpdate: number;
}

const initialState: ImmersiveState = {
  // VR Session
  vrSession: null,
  vrActive: false,
  vrSupported: false,
  
  // AR Session
  arSession: null,
  arActive: false,
  arSupported: false,
  
  // Collaborative
  collaborativeSession: null,
  connectedUsers: [],
  sharedAnnotations: [],
  voiceChat: null,
  
  // Haptic
  hapticFeedback: null,
  hapticDevices: [],
  hapticEnabled: false,
  
  // Gesture
  gestureRecognition: null,
  currentGesture: null,
  gestureHistory: [],
  
  // Voice
  voiceRecognition: {
    listening: false,
    transcript: '',
    confidence: 0,
  },
  voiceCommands: [],
  
  // Analysis
  immersiveAnalysis: null,
  dataVisualizations: [],
  spatialQueries: [],
  
  // Holographic
  holographicDisplay: {
    active: false,
    projectionType: 'volumetric',
    hologramCount: 0,
  },
  spatialComputing: null,
  
  // Training
  trainingProgress: null,
  currentTrainingModule: null,
  educationalContent: [],
  assessmentResults: [],
  
  // Accessibility
  accessibilitySettings: {
    voiceNavigation: false,
    highContrast: false,
    motionReduction: false,
    cognitiveSupport: false,
    audioDescriptions: false,
    fontSize: 'medium',
    colorBlindnessSupport: 'none',
    customizations: {},
  },
  accessibilityActive: false,
  
  // Field Data
  fieldData: [],
  currentLocation: null,
  
  // System
  loading: false,
  error: null,
  lastUpdate: Date.now(),
};

const immersiveSlice = createSlice({
  name: 'immersive',
  initialState,
  reducers: {
    // VR Session Actions
    startVRSession: (state, action: PayloadAction<VRSession>) => {
      state.vrSession = action.payload;
      state.vrActive = true;
      state.lastUpdate = Date.now();
    },
    
    endVRSession: (state) => {
      if (state.vrSession) {
        state.vrSession.endTime = new Date();
      }
      state.vrActive = false;
      state.lastUpdate = Date.now();
    },
    
    updateVRSession: (state, action: PayloadAction<Partial<VRSession>>) => {
      if (state.vrSession) {
        state.vrSession = { ...state.vrSession, ...action.payload };
      }
      state.lastUpdate = Date.now();
    },
    
    setVRSupported: (state, action: PayloadAction<boolean>) => {
      state.vrSupported = action.payload;
    },
    
    // AR Session Actions
    startARSession: (state, action: PayloadAction<ARSession>) => {
      state.arSession = action.payload;
      state.arActive = true;
      state.lastUpdate = Date.now();
    },
    
    endARSession: (state) => {
      if (state.arSession) {
        state.arSession.endTime = new Date();
      }
      state.arActive = false;
      state.lastUpdate = Date.now();
    },
    
    updateARSession: (state, action: PayloadAction<Partial<ARSession>>) => {
      if (state.arSession) {
        state.arSession = { ...state.arSession, ...action.payload };
      }
      state.lastUpdate = Date.now();
    },
    
    setARSupported: (state, action: PayloadAction<boolean>) => {
      state.arSupported = action.payload;
    },
    
    // Collaborative Actions
    updateCollaborativeSession: (state, action: PayloadAction<Partial<CollaborativeSession>>) => {
      if (state.collaborativeSession) {
        state.collaborativeSession = { ...state.collaborativeSession, ...action.payload };
      } else {
        state.collaborativeSession = action.payload as CollaborativeSession;
      }
      state.lastUpdate = Date.now();
    },
    
    addVRUser: (state, action: PayloadAction<VRUser>) => {
      const existingIndex = state.connectedUsers.findIndex(user => user.id === action.payload.id);
      if (existingIndex >= 0) {
        state.connectedUsers[existingIndex] = action.payload;
      } else {
        state.connectedUsers.push(action.payload);
      }
      state.lastUpdate = Date.now();
    },
    
    removeVRUser: (state, action: PayloadAction<string>) => {
      state.connectedUsers = state.connectedUsers.filter(user => user.id !== action.payload);
      state.lastUpdate = Date.now();
    },
    
    addSharedAnnotation: (state, action: PayloadAction<SharedAnnotation>) => {
      state.sharedAnnotations.push(action.payload);
      state.lastUpdate = Date.now();
    },
    
    updateVoiceChat: (state, action: PayloadAction<Partial<VoiceChat>>) => {
      if (state.voiceChat) {
        state.voiceChat = { ...state.voiceChat, ...action.payload };
      } else {
        state.voiceChat = action.payload as VoiceChat;
      }
      state.lastUpdate = Date.now();
    },
    
    // Haptic Feedback Actions
    setHapticFeedback: (state, action: PayloadAction<HapticFeedback>) => {
      state.hapticFeedback = action.payload;
      state.lastUpdate = Date.now();
    },
    
    updateHapticDevice: (state, action: PayloadAction<HapticDevice>) => {
      const existingIndex = state.hapticDevices.findIndex(device => device.id === action.payload.id);
      if (existingIndex >= 0) {
        state.hapticDevices[existingIndex] = action.payload;
      } else {
        state.hapticDevices.push(action.payload);
      }
      state.lastUpdate = Date.now();
    },
    
    setHapticEnabled: (state, action: PayloadAction<boolean>) => {
      state.hapticEnabled = action.payload;
    },
    
    // Gesture Recognition Actions
    updateGestureRecognition: (state, action: PayloadAction<Partial<GestureRecognition>>) => {
      if (state.gestureRecognition) {
        state.gestureRecognition = { ...state.gestureRecognition, ...action.payload };
      } else {
        state.gestureRecognition = action.payload as GestureRecognition;
      }
      
      if (action.payload.currentGesture) {
        state.currentGesture = action.payload.currentGesture;
        state.gestureHistory.push(action.payload.currentGesture);
        
        // Keep only last 10 gestures
        if (state.gestureHistory.length > 10) {
          state.gestureHistory = state.gestureHistory.slice(-10);
        }
      }
      
      state.lastUpdate = Date.now();
    },
    
    // Voice Recognition Actions
    updateVoiceRecognition: (state, action: PayloadAction<Partial<typeof initialState.voiceRecognition>>) => {
      state.voiceRecognition = { ...state.voiceRecognition, ...action.payload };
      state.lastUpdate = Date.now();
    },
    
    addVoiceCommand: (state, action: PayloadAction<VoiceCommand>) => {
      state.voiceCommands.push(action.payload);
      state.lastUpdate = Date.now();
    },
    
    executeCommand: (state, action: PayloadAction<VoiceCommand | GestureCommand>) => {
      // Log command execution
      console.log('Executing command:', action.payload);
      state.lastUpdate = Date.now();
    },
    
    // Immersive Analysis Actions
    updateImmersiveAnalysis: (state, action: PayloadAction<Partial<ImmersiveAnalysis>>) => {
      if (state.immersiveAnalysis) {
        state.immersiveAnalysis = { ...state.immersiveAnalysis, ...action.payload };
      } else {
        state.immersiveAnalysis = action.payload as ImmersiveAnalysis;
      }
      state.lastUpdate = Date.now();
    },
    
    addDataVisualization: (state, action: PayloadAction<DataVisualization>) => {
      state.dataVisualizations.push(action.payload);
      state.lastUpdate = Date.now();
    },
    
    updateDataVisualization: (state, action: PayloadAction<{ id: string; updates: Partial<DataVisualization> }>) => {
      const index = state.dataVisualizations.findIndex(viz => viz.id === action.payload.id);
      if (index >= 0) {
        state.dataVisualizations[index] = { ...state.dataVisualizations[index], ...action.payload.updates };
      }
      state.lastUpdate = Date.now();
    },
    
    addSpatialQuery: (state, action: PayloadAction<SpatialQuery>) => {
      state.spatialQueries.push(action.payload);
      state.lastUpdate = Date.now();
    },
    
    // Holographic Display Actions
    updateHolographicDisplay: (state, action: PayloadAction<Partial<typeof initialState.holographicDisplay>>) => {
      state.holographicDisplay = { ...state.holographicDisplay, ...action.payload };
      state.lastUpdate = Date.now();
    },
    
    addHologram: (state, _action: PayloadAction<any>) => {
      // Add hologram to display
      state.holographicDisplay.hologramCount += 1;
      state.lastUpdate = Date.now();
    },
    
    updateSpatialComputing: (state, action: PayloadAction<Partial<SpatialComputing>>) => {
      if (state.spatialComputing) {
        state.spatialComputing = { ...state.spatialComputing, ...action.payload };
      } else {
        state.spatialComputing = action.payload as SpatialComputing;
      }
      state.lastUpdate = Date.now();
    },
    
    // Training and Education Actions
    updateTrainingProgress: (state, action: PayloadAction<Partial<LearningProgress>>) => {
      if (state.trainingProgress) {
        state.trainingProgress = { ...state.trainingProgress, ...action.payload };
      } else {
        state.trainingProgress = action.payload as LearningProgress;
      }
      state.lastUpdate = Date.now();
    },
    
    setCurrentTrainingModule: (state, action: PayloadAction<TrainingModule>) => {
      state.currentTrainingModule = action.payload;
      state.lastUpdate = Date.now();
    },
    
    completeLesson: (state, action: PayloadAction<{ lessonId: string; completionTime: number; score: number }>) => {
      if (state.trainingProgress) {
        if (!state.trainingProgress.completedLessons.includes(action.payload.lessonId)) {
          state.trainingProgress.completedLessons.push(action.payload.lessonId);
        }
        state.trainingProgress.totalScore += action.payload.score;
      }
      state.lastUpdate = Date.now();
    },
    
    addAssessmentResult: (state, action: PayloadAction<AssessmentResult>) => {
      state.assessmentResults.push(action.payload);
      if (state.trainingProgress) {
        state.trainingProgress.assessmentResults.push(action.payload);
      }
      state.lastUpdate = Date.now();
    },
    
    updateEducationalContent: (state, _action: PayloadAction<Partial<EducationalContent>>) => {
      // Update current educational content
      state.lastUpdate = Date.now();
    },
    
    // Accessibility Actions
    updateAccessibilitySettings: (state, action: PayloadAction<Partial<AccessibilitySettings>>) => {
      state.accessibilitySettings = { ...state.accessibilitySettings, ...action.payload };
      state.accessibilityActive = Object.values(state.accessibilitySettings).some(setting => 
        typeof setting === 'boolean' ? setting : setting !== 'medium' && setting !== 'none'
      );
      state.lastUpdate = Date.now();
    },
    
    enableVoiceNavigation: (state, action: PayloadAction<boolean>) => {
      state.accessibilitySettings.voiceNavigation = action.payload;
      state.lastUpdate = Date.now();
    },
    
    updateMotionSettings: (state, action: PayloadAction<boolean>) => {
      state.accessibilitySettings.motionReduction = action.payload;
      state.lastUpdate = Date.now();
    },
    
    setCognitiveSupport: (state, action: PayloadAction<boolean>) => {
      state.accessibilitySettings.cognitiveSupport = action.payload;
      state.lastUpdate = Date.now();
    },
    
    // Field Data Actions
    addFieldObservation: (state, action: PayloadAction<FieldData>) => {
      state.fieldData.push(action.payload);
      if (state.arSession) {
        state.arSession.fieldData.push(action.payload);
      }
      state.lastUpdate = Date.now();
    },
    
    updateCurrentLocation: (state, action: PayloadAction<typeof initialState.currentLocation>) => {
      state.currentLocation = action.payload;
      state.lastUpdate = Date.now();
    },
    
    // System Actions
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    
    clearError: (state) => {
      state.error = null;
    },
    
    resetImmersiveState: () => {
      return initialState;
    },
  },
});

export const {
  // VR Session
  startVRSession,
  endVRSession,
  updateVRSession,
  setVRSupported,
  
  // AR Session
  startARSession,
  endARSession,
  updateARSession,
  setARSupported,
  
  // Collaborative
  updateCollaborativeSession,
  addVRUser,
  removeVRUser,
  addSharedAnnotation,
  updateVoiceChat,
  
  // Haptic
  setHapticFeedback,
  updateHapticDevice,
  setHapticEnabled,
  
  // Gesture
  updateGestureRecognition,
  
  // Voice
  updateVoiceRecognition,
  addVoiceCommand,
  executeCommand,
  
  // Analysis
  updateImmersiveAnalysis,
  addDataVisualization,
  updateDataVisualization,
  addSpatialQuery,
  
  // Holographic
  updateHolographicDisplay,
  addHologram,
  updateSpatialComputing,
  
  // Training
  updateTrainingProgress,
  setCurrentTrainingModule,
  completeLesson,
  addAssessmentResult,
  updateEducationalContent,
  
  // Accessibility
  updateAccessibilitySettings,
  enableVoiceNavigation,
  updateMotionSettings,
  setCognitiveSupport,
  
  // Field Data
  addFieldObservation,
  updateCurrentLocation,
  
  // System
  setLoading,
  setError,
  clearError,
  resetImmersiveState,
} = immersiveSlice.actions;

export default immersiveSlice.reducer;

// Selectors
export const selectVRSession = (state: { immersive: ImmersiveState }) => state.immersive.vrSession;
export const selectVRActive = (state: { immersive: ImmersiveState }) => state.immersive.vrActive;
export const selectARSession = (state: { immersive: ImmersiveState }) => state.immersive.arSession;
export const selectARActive = (state: { immersive: ImmersiveState }) => state.immersive.arActive;
export const selectCollaborativeSession = (state: { immersive: ImmersiveState }) => state.immersive.collaborativeSession;
export const selectConnectedUsers = (state: { immersive: ImmersiveState }) => state.immersive.connectedUsers;
export const selectHapticFeedback = (state: { immersive: ImmersiveState }) => state.immersive.hapticFeedback;
export const selectGestureRecognition = (state: { immersive: ImmersiveState }) => state.immersive.gestureRecognition;
export const selectVoiceRecognition = (state: { immersive: ImmersiveState }) => state.immersive.voiceRecognition;
export const selectImmersiveAnalysis = (state: { immersive: ImmersiveState }) => state.immersive.immersiveAnalysis;
export const selectHolographicDisplay = (state: { immersive: ImmersiveState }) => state.immersive.holographicDisplay;
export const selectTrainingProgress = (state: { immersive: ImmersiveState }) => state.immersive.trainingProgress;
export const selectAccessibilitySettings = (state: { immersive: ImmersiveState }) => state.immersive.accessibilitySettings;
export const selectFieldData = (state: { immersive: ImmersiveState }) => state.immersive.fieldData;
export const selectCurrentLocation = (state: { immersive: ImmersiveState }) => state.immersive.currentLocation;