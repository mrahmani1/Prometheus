/**
 * Visualization state management slice
 * Manages 3D/4D visualization settings, rendering options, and display preferences
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface VisualizationState {
  // 3D rendering settings
  rendering: {
    quality: 'low' | 'medium' | 'high' | 'ultra';
    antialiasing: boolean;
    shadows: boolean;
    lighting: 'basic' | 'advanced' | 'realistic';
    postProcessing: boolean;
    frameRate: number;
    maxFrameRate: number;
  };
  
  // Camera and view settings
  camera: {
    position: [number, number, number];
    target: [number, number, number];
    fov: number;
    near: number;
    far: number;
    projection: 'perspective' | 'orthographic';
  };
  
  // Layer management
  layers: {
    [layerId: string]: {
      id: string;
      name: string;
      type: 'seismic' | 'volcanic' | 'deformation' | 'geological' | 'topographic' | 'satellite';
      visible: boolean;
      opacity: number;
      colorScheme: string;
      renderStyle: 'points' | 'lines' | 'surfaces' | 'volumes';
      zIndex: number;
      data: any;
      metadata: any;
    };
  };
  
  // Color schemes and styling
  colorSchemes: {
    seismic: string;
    volcanic: string;
    deformation: string;
    topographic: string;
    geological: string;
    temperature: string;
    magnitude: string;
    depth: string;
  };
  
  // Animation and temporal settings
  animation: {
    enabled: boolean;
    speed: number;
    loop: boolean;
    currentTime: Date | null;
    timeRange: {
      start: Date | null;
      end: Date | null;
    };
    playbackRate: number;
    interpolation: 'linear' | 'cubic' | 'spline';
  };
  
  // Interaction settings
  interaction: {
    enablePan: boolean;
    enableZoom: boolean;
    enableRotate: boolean;
    enableSelection: boolean;
    enableMeasurement: boolean;
    enableAnnotation: boolean;
    mouseControls: 'orbit' | 'fly' | 'first-person';
    touchControls: boolean;
  };
  
  // Cross-section and slicing
  crossSection: {
    enabled: boolean;
    plane: {
      normal: [number, number, number];
      point: [number, number, number];
    };
    thickness: number;
    showBoth: boolean;
  };
  
  // Volume rendering
  volumeRendering: {
    enabled: boolean;
    transferFunction: {
      opacity: Array<[number, number]>;
      color: Array<[number, string]>;
    };
    raycastingSteps: number;
    gradientThreshold: number;
    ambientOcclusion: boolean;
  };
  
  // Geological visualization specific
  geological: {
    showFaults: boolean;
    showFolds: boolean;
    showBedding: boolean;
    showJoints: boolean;
    rockUnitColors: Record<string, string>;
    structuralSymbols: boolean;
    ageColors: boolean;
    lithologyPatterns: boolean;
  };
  
  // Seismic visualization
  seismic: {
    showWaveforms: boolean;
    amplitudeScale: number;
    frequencyFilter: [number, number];
    phaseColors: Record<string, string>;
    magnitudeScaling: 'linear' | 'logarithmic';
    depthColorCoding: boolean;
    eventClustering: boolean;
    clusterRadius: number;
  };
  
  // Volcanic visualization
  volcanic: {
    showGasPlumes: boolean;
    showLavaFlows: boolean;
    showAshfall: boolean;
    showDeformationField: boolean;
    alertLevelColors: Record<string, string>;
    temperatureColormap: string;
    eruptionAnimation: boolean;
    hazardZones: boolean;
  };
  
  // Deformation visualization
  deformation: {
    vectorScale: number;
    showVectors: boolean;
    showContours: boolean;
    contourInterval: number;
    referenceFrame: 'absolute' | 'relative';
    uncertaintyVisualization: boolean;
    strainVisualization: boolean;
    sourceModelOverlay: boolean;
  };
  
  // Export and sharing
  export: {
    format: 'png' | 'jpg' | 'svg' | 'pdf' | 'webm' | 'gif';
    resolution: [number, number];
    quality: number;
    includeMetadata: boolean;
    includeColorbar: boolean;
    includeScale: boolean;
    includeNorthArrow: boolean;
  };
  
  // Performance monitoring
  performance: {
    fps: number;
    renderTime: number;
    memoryUsage: number;
    triangleCount: number;
    drawCalls: number;
    adaptiveQuality: boolean;
    lodEnabled: boolean;
    frustumCulling: boolean;
  };
  
  // UI overlay settings
  overlay: {
    showCoordinates: boolean;
    showScale: boolean;
    showNorthArrow: boolean;
    showColorbar: boolean;
    showLegend: boolean;
    showTimestamp: boolean;
    showMetadata: boolean;
    overlayOpacity: number;
  };
}

const initialState: VisualizationState = {
  rendering: {
    quality: 'high',
    antialiasing: true,
    shadows: true,
    lighting: 'advanced',
    postProcessing: true,
    frameRate: 60,
    maxFrameRate: 120,
  },
  
  camera: {
    position: [0, 0, 100],
    target: [0, 0, 0],
    fov: 75,
    near: 0.1,
    far: 10000,
    projection: 'perspective',
  },
  
  layers: {},
  
  colorSchemes: {
    seismic: 'viridis',
    volcanic: 'hot',
    deformation: 'coolwarm',
    topographic: 'terrain',
    geological: 'geological',
    temperature: 'plasma',
    magnitude: 'magma',
    depth: 'deep',
  },
  
  animation: {
    enabled: false,
    speed: 1.0,
    loop: true,
    currentTime: null,
    timeRange: {
      start: null,
      end: null,
    },
    playbackRate: 1.0,
    interpolation: 'linear',
  },
  
  interaction: {
    enablePan: true,
    enableZoom: true,
    enableRotate: true,
    enableSelection: true,
    enableMeasurement: true,
    enableAnnotation: true,
    mouseControls: 'orbit',
    touchControls: true,
  },
  
  crossSection: {
    enabled: false,
    plane: {
      normal: [0, 0, 1],
      point: [0, 0, 0],
    },
    thickness: 1.0,
    showBoth: false,
  },
  
  volumeRendering: {
    enabled: false,
    transferFunction: {
      opacity: [[0, 0], [0.5, 0.5], [1, 1]],
      color: [[0, '#000080'], [0.5, '#ff0000'], [1, '#ffff00']],
    },
    raycastingSteps: 256,
    gradientThreshold: 0.1,
    ambientOcclusion: false,
  },
  
  geological: {
    showFaults: true,
    showFolds: true,
    showBedding: true,
    showJoints: false,
    rockUnitColors: {},
    structuralSymbols: true,
    ageColors: true,
    lithologyPatterns: false,
  },
  
  seismic: {
    showWaveforms: false,
    amplitudeScale: 1.0,
    frequencyFilter: [1, 50],
    phaseColors: {
      P: '#ff0000',
      S: '#0000ff',
      Lg: '#00ff00',
      Rg: '#ffff00',
    },
    magnitudeScaling: 'logarithmic',
    depthColorCoding: true,
    eventClustering: true,
    clusterRadius: 10,
  },
  
  volcanic: {
    showGasPlumes: true,
    showLavaFlows: true,
    showAshfall: true,
    showDeformationField: true,
    alertLevelColors: {
      green: '#00ff00',
      yellow: '#ffff00',
      orange: '#ff8000',
      red: '#ff0000',
    },
    temperatureColormap: 'hot',
    eruptionAnimation: true,
    hazardZones: true,
  },
  
  deformation: {
    vectorScale: 1000,
    showVectors: true,
    showContours: true,
    contourInterval: 1.0,
    referenceFrame: 'absolute',
    uncertaintyVisualization: true,
    strainVisualization: false,
    sourceModelOverlay: true,
  },
  
  export: {
    format: 'png',
    resolution: [1920, 1080],
    quality: 0.9,
    includeMetadata: true,
    includeColorbar: true,
    includeScale: true,
    includeNorthArrow: true,
  },
  
  performance: {
    fps: 60,
    renderTime: 16.67,
    memoryUsage: 0,
    triangleCount: 0,
    drawCalls: 0,
    adaptiveQuality: true,
    lodEnabled: true,
    frustumCulling: true,
  },
  
  overlay: {
    showCoordinates: true,
    showScale: true,
    showNorthArrow: true,
    showColorbar: true,
    showLegend: true,
    showTimestamp: true,
    showMetadata: false,
    overlayOpacity: 0.8,
  },
};

const visualizationSlice = createSlice({
  name: 'visualization',
  initialState,
  reducers: {
    // Rendering settings
    setRenderingQuality: (state, action: PayloadAction<'low' | 'medium' | 'high' | 'ultra'>) => {
      state.rendering.quality = action.payload;
    },
    setRenderingSettings: (state, action: PayloadAction<Partial<VisualizationState['rendering']>>) => {
      state.rendering = { ...state.rendering, ...action.payload };
    },
    
    // Camera controls
    setCameraPosition: (state, action: PayloadAction<[number, number, number]>) => {
      state.camera.position = action.payload;
    },
    setCameraTarget: (state, action: PayloadAction<[number, number, number]>) => {
      state.camera.target = action.payload;
    },
    setCameraSettings: (state, action: PayloadAction<Partial<VisualizationState['camera']>>) => {
      state.camera = { ...state.camera, ...action.payload };
    },
    resetCamera: state => {
      state.camera = initialState.camera;
    },
    
    // Layer management
    addLayer: (state, action: PayloadAction<VisualizationState['layers'][string]>) => {
      state.layers[action.payload.id] = action.payload;
    },
    removeLayer: (state, action: PayloadAction<string>) => {
      delete state.layers[action.payload];
    },
    updateLayer: (state, action: PayloadAction<{ id: string; updates: Partial<VisualizationState['layers'][string]> }>) => {
      if (state.layers[action.payload.id]) {
        state.layers[action.payload.id] = { ...state.layers[action.payload.id], ...action.payload.updates };
      }
    },
    setLayerVisibility: (state, action: PayloadAction<{ id: string; visible: boolean }>) => {
      if (state.layers[action.payload.id]) {
        state.layers[action.payload.id].visible = action.payload.visible;
      }
    },
    setLayerOpacity: (state, action: PayloadAction<{ id: string; opacity: number }>) => {
      if (state.layers[action.payload.id]) {
        state.layers[action.payload.id].opacity = action.payload.opacity;
      }
    },
    reorderLayers: (state, action: PayloadAction<string[]>) => {
      action.payload.forEach((layerId, index) => {
        if (state.layers[layerId]) {
          state.layers[layerId].zIndex = index;
        }
      });
    },
    
    // Color schemes
    setColorScheme: (state, action: PayloadAction<{ type: keyof VisualizationState['colorSchemes']; scheme: string }>) => {
      state.colorSchemes[action.payload.type] = action.payload.scheme;
    },
    setColorSchemes: (state, action: PayloadAction<Partial<VisualizationState['colorSchemes']>>) => {
      state.colorSchemes = { ...state.colorSchemes, ...action.payload };
    },
    
    // Animation controls
    setAnimationEnabled: (state, action: PayloadAction<boolean>) => {
      state.animation.enabled = action.payload;
    },
    setAnimationSpeed: (state, action: PayloadAction<number>) => {
      state.animation.speed = action.payload;
    },
    setAnimationTime: (state, action: PayloadAction<Date>) => {
      state.animation.currentTime = action.payload;
    },
    setAnimationTimeRange: (state, action: PayloadAction<{ start: Date; end: Date }>) => {
      state.animation.timeRange = action.payload;
    },
    setAnimationSettings: (state, action: PayloadAction<Partial<VisualizationState['animation']>>) => {
      state.animation = { ...state.animation, ...action.payload };
    },
    
    // Interaction settings
    setInteractionSettings: (state, action: PayloadAction<Partial<VisualizationState['interaction']>>) => {
      state.interaction = { ...state.interaction, ...action.payload };
    },
    
    // Cross-section controls
    setCrossSectionEnabled: (state, action: PayloadAction<boolean>) => {
      state.crossSection.enabled = action.payload;
    },
    setCrossSectionPlane: (state, action: PayloadAction<{ normal: [number, number, number]; point: [number, number, number] }>) => {
      state.crossSection.plane = action.payload;
    },
    setCrossSectionSettings: (state, action: PayloadAction<Partial<VisualizationState['crossSection']>>) => {
      state.crossSection = { ...state.crossSection, ...action.payload };
    },
    
    // Volume rendering
    setVolumeRenderingEnabled: (state, action: PayloadAction<boolean>) => {
      state.volumeRendering.enabled = action.payload;
    },
    setVolumeTransferFunction: (state, action: PayloadAction<VisualizationState['volumeRendering']['transferFunction']>) => {
      state.volumeRendering.transferFunction = action.payload;
    },
    setVolumeRenderingSettings: (state, action: PayloadAction<Partial<VisualizationState['volumeRendering']>>) => {
      state.volumeRendering = { ...state.volumeRendering, ...action.payload };
    },
    
    // Domain-specific settings
    setGeologicalSettings: (state, action: PayloadAction<Partial<VisualizationState['geological']>>) => {
      state.geological = { ...state.geological, ...action.payload };
    },
    setSeismicSettings: (state, action: PayloadAction<Partial<VisualizationState['seismic']>>) => {
      state.seismic = { ...state.seismic, ...action.payload };
    },
    setVolcanicSettings: (state, action: PayloadAction<Partial<VisualizationState['volcanic']>>) => {
      state.volcanic = { ...state.volcanic, ...action.payload };
    },
    setDeformationSettings: (state, action: PayloadAction<Partial<VisualizationState['deformation']>>) => {
      state.deformation = { ...state.deformation, ...action.payload };
    },
    
    // Export settings
    setExportSettings: (state, action: PayloadAction<Partial<VisualizationState['export']>>) => {
      state.export = { ...state.export, ...action.payload };
    },
    
    // Performance monitoring
    updatePerformanceMetrics: (state, action: PayloadAction<Partial<VisualizationState['performance']>>) => {
      state.performance = { ...state.performance, ...action.payload };
    },
    
    // Overlay settings
    setOverlaySettings: (state, action: PayloadAction<Partial<VisualizationState['overlay']>>) => {
      state.overlay = { ...state.overlay, ...action.payload };
    },
    
    // Presets and reset
    applyVisualizationPreset: (state, action: PayloadAction<'seismic' | 'volcanic' | 'geological' | 'deformation'>) => {
      const preset = action.payload;
      
      // Apply preset-specific settings
      switch (preset) {
        case 'seismic':
          state.colorSchemes.seismic = 'viridis';
          state.seismic.depthColorCoding = true;
          state.seismic.eventClustering = true;
          break;
        case 'volcanic':
          state.colorSchemes.volcanic = 'hot';
          state.volcanic.showGasPlumes = true;
          state.volcanic.eruptionAnimation = true;
          break;
        case 'geological':
          state.geological.ageColors = true;
          state.geological.structuralSymbols = true;
          break;
        case 'deformation':
          state.deformation.showVectors = true;
          state.deformation.uncertaintyVisualization = true;
          break;
      }
    },
    resetVisualizationSettings: () => initialState,
  },
});

export const {
  // Rendering settings
  setRenderingQuality,
  setRenderingSettings,
  
  // Camera controls
  setCameraPosition,
  setCameraTarget,
  setCameraSettings,
  resetCamera,
  
  // Layer management
  addLayer,
  removeLayer,
  updateLayer,
  setLayerVisibility,
  setLayerOpacity,
  reorderLayers,
  
  // Color schemes
  setColorScheme,
  setColorSchemes,
  
  // Animation controls
  setAnimationEnabled,
  setAnimationSpeed,
  setAnimationTime,
  setAnimationTimeRange,
  setAnimationSettings,
  
  // Interaction settings
  setInteractionSettings,
  
  // Cross-section controls
  setCrossSectionEnabled,
  setCrossSectionPlane,
  setCrossSectionSettings,
  
  // Volume rendering
  setVolumeRenderingEnabled,
  setVolumeTransferFunction,
  setVolumeRenderingSettings,
  
  // Domain-specific settings
  setGeologicalSettings,
  setSeismicSettings,
  setVolcanicSettings,
  setDeformationSettings,
  
  // Export settings
  setExportSettings,
  
  // Performance monitoring
  updatePerformanceMetrics,
  
  // Overlay settings
  setOverlaySettings,
  
  // Presets and reset
  applyVisualizationPreset,
  resetVisualizationSettings,
} = visualizationSlice.actions;

export default visualizationSlice.reducer;