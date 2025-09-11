/**
 * Geological data state management slice
 * Manages geological datasets, analysis results, and domain-specific state
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type {
  SeismicEvent,
  VolcanicState,
  DeformationData,
  ResourceProspect,
  MagmaticSystem,
  GeographicBounds,
  QualityMetrics,
} from '../../types/geological';

interface GeologicalState {
  // Active datasets
  activeRegion: GeographicBounds | null;
  activeTimeRange: {
    start: Date | null;
    end: Date | null;
  };
  
  // Seismic data
  seismicEvents: {
    data: SeismicEvent[];
    loading: boolean;
    error: string | null;
    lastUpdated: Date | null;
    filters: {
      magnitudeRange: [number, number];
      depthRange: [number, number];
      eventTypes: string[];
    };
  };
  
  // Volcanic monitoring
  volcanicStates: {
    data: VolcanicState[];
    loading: boolean;
    error: string | null;
    lastUpdated: Date | null;
    alerts: {
      active: number;
      total: number;
      byLevel: Record<string, number>;
    };
  };
  
  // Deformation analysis
  deformationData: {
    data: DeformationData[];
    loading: boolean;
    error: string | null;
    lastUpdated: Date | null;
    processingStatus: {
      insar: 'idle' | 'processing' | 'completed' | 'error';
      gps: 'idle' | 'processing' | 'completed' | 'error';
    };
  };
  
  // Resource assessment
  resourceProspects: {
    data: ResourceProspect[];
    loading: boolean;
    error: string | null;
    lastUpdated: Date | null;
    filters: {
      resourceTypes: string[];
      confidenceThreshold: number;
      economicViability: string[];
    };
  };
  
  // Magmatic systems
  magmaticSystems: {
    data: MagmaticSystem[];
    loading: boolean;
    error: string | null;
    lastUpdated: Date | null;
    activeSystemId: string | null;
  };
  
  // Data quality and validation
  dataQuality: {
    overall: QualityMetrics | null;
    byDataset: Record<string, QualityMetrics>;
    validationResults: Record<string, any>;
  };
  
  // Analysis settings
  analysisSettings: {
    seismic: {
      detectionThreshold: number;
      filterFrequency: [number, number];
      windowLength: number;
      overlapPercent: number;
    };
    volcanic: {
      alertThresholds: Record<string, number>;
      forecastHorizon: number;
      uncertaintyLevel: number;
    };
    deformation: {
      coherenceThreshold: number;
      unwrappingMethod: string;
      atmosphericCorrection: boolean;
      referencePoint: [number, number] | null;
    };
    resource: {
      prospectivityThreshold: number;
      economicParameters: Record<string, number>;
      riskFactors: Record<string, number>;
    };
  };
  
  // Processing status
  processing: {
    activeJobs: string[];
    completedJobs: string[];
    failedJobs: string[];
    jobProgress: Record<string, number>;
  };
  
  // Cache management
  cache: {
    enabled: boolean;
    maxAge: number;
    maxSize: number;
    currentSize: number;
    hitRate: number;
  };
}

const initialState: GeologicalState = {
  activeRegion: null,
  activeTimeRange: {
    start: null,
    end: null,
  },
  
  seismicEvents: {
    data: [],
    loading: false,
    error: null,
    lastUpdated: null,
    filters: {
      magnitudeRange: [0, 10],
      depthRange: [0, 700],
      eventTypes: [],
    },
  },
  
  volcanicStates: {
    data: [],
    loading: false,
    error: null,
    lastUpdated: null,
    alerts: {
      active: 0,
      total: 0,
      byLevel: {},
    },
  },
  
  deformationData: {
    data: [],
    loading: false,
    error: null,
    lastUpdated: null,
    processingStatus: {
      insar: 'idle',
      gps: 'idle',
    },
  },
  
  resourceProspects: {
    data: [],
    loading: false,
    error: null,
    lastUpdated: null,
    filters: {
      resourceTypes: [],
      confidenceThreshold: 0.7,
      economicViability: ['viable', 'marginal'],
    },
  },
  
  magmaticSystems: {
    data: [],
    loading: false,
    error: null,
    lastUpdated: null,
    activeSystemId: null,
  },
  
  dataQuality: {
    overall: null,
    byDataset: {},
    validationResults: {},
  },
  
  analysisSettings: {
    seismic: {
      detectionThreshold: 0.5,
      filterFrequency: [1, 50],
      windowLength: 30,
      overlapPercent: 50,
    },
    volcanic: {
      alertThresholds: {
        deformation: 10,
        seismicity: 5,
        gasEmissions: 100,
        thermalAnomalies: 50,
      },
      forecastHorizon: 30,
      uncertaintyLevel: 0.95,
    },
    deformation: {
      coherenceThreshold: 0.3,
      unwrappingMethod: 'snaphu',
      atmosphericCorrection: true,
      referencePoint: null,
    },
    resource: {
      prospectivityThreshold: 0.6,
      economicParameters: {
        discountRate: 0.08,
        commodityPrice: 0,
        operatingCost: 0,
      },
      riskFactors: {
        geological: 0.2,
        technical: 0.15,
        economic: 0.25,
        environmental: 0.1,
      },
    },
  },
  
  processing: {
    activeJobs: [],
    completedJobs: [],
    failedJobs: [],
    jobProgress: {},
  },
  
  cache: {
    enabled: true,
    maxAge: 3600000, // 1 hour
    maxSize: 100 * 1024 * 1024, // 100MB
    currentSize: 0,
    hitRate: 0,
  },
};

const geologicalSlice = createSlice({
  name: 'geological',
  initialState,
  reducers: {
    // Region and time management
    setActiveRegion: (state, action: PayloadAction<GeographicBounds>) => {
      state.activeRegion = action.payload;
    },
    setActiveTimeRange: (state, action: PayloadAction<{ start: Date; end: Date }>) => {
      state.activeTimeRange = action.payload;
    },
    clearActiveRegion: state => {
      state.activeRegion = null;
    },
    clearActiveTimeRange: state => {
      state.activeTimeRange = { start: null, end: null };
    },
    
    // Seismic data management
    setSeismicEventsLoading: (state, action: PayloadAction<boolean>) => {
      state.seismicEvents.loading = action.payload;
    },
    setSeismicEventsData: (state, action: PayloadAction<SeismicEvent[]>) => {
      state.seismicEvents.data = action.payload;
      state.seismicEvents.lastUpdated = new Date();
      state.seismicEvents.error = null;
    },
    addSeismicEvent: (state, action: PayloadAction<SeismicEvent>) => {
      state.seismicEvents.data.unshift(action.payload);
    },
    updateSeismicEvent: (state, action: PayloadAction<SeismicEvent>) => {
      const index = state.seismicEvents.data.findIndex(event => event.eventId === action.payload.eventId);
      if (index !== -1) {
        state.seismicEvents.data[index] = action.payload;
      }
    },
    setSeismicEventsError: (state, action: PayloadAction<string>) => {
      state.seismicEvents.error = action.payload;
      state.seismicEvents.loading = false;
    },
    setSeismicFilters: (state, action: PayloadAction<Partial<typeof initialState.seismicEvents.filters>>) => {
      state.seismicEvents.filters = { ...state.seismicEvents.filters, ...action.payload };
    },
    
    // Volcanic monitoring management
    setVolcanicStatesLoading: (state, action: PayloadAction<boolean>) => {
      state.volcanicStates.loading = action.payload;
    },
    setVolcanicStatesData: (state, action: PayloadAction<VolcanicState[]>) => {
      state.volcanicStates.data = action.payload;
      state.volcanicStates.lastUpdated = new Date();
      state.volcanicStates.error = null;
      
      // Update alert statistics
      const alerts = action.payload.reduce((acc, volcano) => {
        acc.total++;
        if (volcano.alertLevel !== 'green') {
          acc.active++;
        }
        acc.byLevel[volcano.alertLevel] = (acc.byLevel[volcano.alertLevel] || 0) + 1;
        return acc;
      }, { active: 0, total: 0, byLevel: {} as Record<string, number> });
      
      state.volcanicStates.alerts = alerts;
    },
    updateVolcanicState: (state, action: PayloadAction<VolcanicState>) => {
      const index = state.volcanicStates.data.findIndex(volcano => volcano.volcanoId === action.payload.volcanoId);
      if (index !== -1) {
        state.volcanicStates.data[index] = action.payload;
      } else {
        state.volcanicStates.data.push(action.payload);
      }
    },
    setVolcanicStatesError: (state, action: PayloadAction<string>) => {
      state.volcanicStates.error = action.payload;
      state.volcanicStates.loading = false;
    },
    
    // Deformation data management
    setDeformationDataLoading: (state, action: PayloadAction<boolean>) => {
      state.deformationData.loading = action.payload;
    },
    setDeformationData: (state, action: PayloadAction<DeformationData[]>) => {
      state.deformationData.data = action.payload;
      state.deformationData.lastUpdated = new Date();
      state.deformationData.error = null;
    },
    addDeformationData: (state, action: PayloadAction<DeformationData>) => {
      state.deformationData.data.push(action.payload);
    },
    setDeformationError: (state, action: PayloadAction<string>) => {
      state.deformationData.error = action.payload;
      state.deformationData.loading = false;
    },
    setDeformationProcessingStatus: (state, action: PayloadAction<{
      type: 'insar' | 'gps';
      status: 'idle' | 'processing' | 'completed' | 'error';
    }>) => {
      state.deformationData.processingStatus[action.payload.type] = action.payload.status;
    },
    
    // Resource assessment management
    setResourceProspectsLoading: (state, action: PayloadAction<boolean>) => {
      state.resourceProspects.loading = action.payload;
    },
    setResourceProspectsData: (state, action: PayloadAction<ResourceProspect[]>) => {
      state.resourceProspects.data = action.payload;
      state.resourceProspects.lastUpdated = new Date();
      state.resourceProspects.error = null;
    },
    updateResourceProspect: (state, action: PayloadAction<ResourceProspect>) => {
      const index = state.resourceProspects.data.findIndex(prospect => prospect.prospectId === action.payload.prospectId);
      if (index !== -1) {
        state.resourceProspects.data[index] = action.payload;
      } else {
        state.resourceProspects.data.push(action.payload);
      }
    },
    setResourceProspectsError: (state, action: PayloadAction<string>) => {
      state.resourceProspects.error = action.payload;
      state.resourceProspects.loading = false;
    },
    setResourceFilters: (state, action: PayloadAction<Partial<typeof initialState.resourceProspects.filters>>) => {
      state.resourceProspects.filters = { ...state.resourceProspects.filters, ...action.payload };
    },
    
    // Magmatic systems management
    setMagmaticSystemsLoading: (state, action: PayloadAction<boolean>) => {
      state.magmaticSystems.loading = action.payload;
    },
    setMagmaticSystemsData: (state, action: PayloadAction<MagmaticSystem[]>) => {
      state.magmaticSystems.data = action.payload;
      state.magmaticSystems.lastUpdated = new Date();
      state.magmaticSystems.error = null;
    },
    setActiveMagmaticSystem: (state, action: PayloadAction<string>) => {
      state.magmaticSystems.activeSystemId = action.payload;
    },
    updateMagmaticSystem: (state, action: PayloadAction<MagmaticSystem>) => {
      const index = state.magmaticSystems.data.findIndex(system => system.systemId === action.payload.systemId);
      if (index !== -1) {
        state.magmaticSystems.data[index] = action.payload;
      } else {
        state.magmaticSystems.data.push(action.payload);
      }
    },
    setMagmaticSystemsError: (state, action: PayloadAction<string>) => {
      state.magmaticSystems.error = action.payload;
      state.magmaticSystems.loading = false;
    },
    
    // Data quality management
    setOverallDataQuality: (state, action: PayloadAction<QualityMetrics>) => {
      state.dataQuality.overall = action.payload;
    },
    setDatasetQuality: (state, action: PayloadAction<{ dataset: string; quality: QualityMetrics }>) => {
      state.dataQuality.byDataset[action.payload.dataset] = action.payload.quality;
    },
    setValidationResults: (state, action: PayloadAction<{ dataset: string; results: any }>) => {
      state.dataQuality.validationResults[action.payload.dataset] = action.payload.results;
    },
    
    // Analysis settings management
    setSeismicAnalysisSettings: (state, action: PayloadAction<Partial<typeof initialState.analysisSettings.seismic>>) => {
      state.analysisSettings.seismic = { ...state.analysisSettings.seismic, ...action.payload };
    },
    setVolcanicAnalysisSettings: (state, action: PayloadAction<Partial<typeof initialState.analysisSettings.volcanic>>) => {
      state.analysisSettings.volcanic = { ...state.analysisSettings.volcanic, ...action.payload };
    },
    setDeformationAnalysisSettings: (state, action: PayloadAction<Partial<typeof initialState.analysisSettings.deformation>>) => {
      state.analysisSettings.deformation = { ...state.analysisSettings.deformation, ...action.payload };
    },
    setResourceAnalysisSettings: (state, action: PayloadAction<Partial<typeof initialState.analysisSettings.resource>>) => {
      state.analysisSettings.resource = { ...state.analysisSettings.resource, ...action.payload };
    },
    
    // Processing management
    addProcessingJob: (state, action: PayloadAction<string>) => {
      state.processing.activeJobs.push(action.payload);
      state.processing.jobProgress[action.payload] = 0;
    },
    updateJobProgress: (state, action: PayloadAction<{ jobId: string; progress: number }>) => {
      state.processing.jobProgress[action.payload.jobId] = action.payload.progress;
    },
    completeProcessingJob: (state, action: PayloadAction<string>) => {
      const index = state.processing.activeJobs.indexOf(action.payload);
      if (index !== -1) {
        state.processing.activeJobs.splice(index, 1);
        state.processing.completedJobs.push(action.payload);
        delete state.processing.jobProgress[action.payload];
      }
    },
    failProcessingJob: (state, action: PayloadAction<string>) => {
      const index = state.processing.activeJobs.indexOf(action.payload);
      if (index !== -1) {
        state.processing.activeJobs.splice(index, 1);
        state.processing.failedJobs.push(action.payload);
        delete state.processing.jobProgress[action.payload];
      }
    },
    
    // Cache management
    setCacheSettings: (state, action: PayloadAction<Partial<typeof initialState.cache>>) => {
      state.cache = { ...state.cache, ...action.payload };
    },
    updateCacheStats: (state, action: PayloadAction<{ size: number; hitRate: number }>) => {
      state.cache.currentSize = action.payload.size;
      state.cache.hitRate = action.payload.hitRate;
    },
    
    // Reset functions
    resetSeismicData: state => {
      state.seismicEvents = initialState.seismicEvents;
    },
    resetVolcanicData: state => {
      state.volcanicStates = initialState.volcanicStates;
    },
    resetDeformationData: state => {
      state.deformationData = initialState.deformationData;
    },
    resetResourceData: state => {
      state.resourceProspects = initialState.resourceProspects;
    },
    resetAllGeologicalData: () => initialState,
  },
});

export const {
  // Region and time management
  setActiveRegion,
  setActiveTimeRange,
  clearActiveRegion,
  clearActiveTimeRange,
  
  // Seismic data management
  setSeismicEventsLoading,
  setSeismicEventsData,
  addSeismicEvent,
  updateSeismicEvent,
  setSeismicEventsError,
  setSeismicFilters,
  
  // Volcanic monitoring management
  setVolcanicStatesLoading,
  setVolcanicStatesData,
  updateVolcanicState,
  setVolcanicStatesError,
  
  // Deformation data management
  setDeformationDataLoading,
  setDeformationData,
  addDeformationData,
  setDeformationError,
  setDeformationProcessingStatus,
  
  // Resource assessment management
  setResourceProspectsLoading,
  setResourceProspectsData,
  updateResourceProspect,
  setResourceProspectsError,
  setResourceFilters,
  
  // Magmatic systems management
  setMagmaticSystemsLoading,
  setMagmaticSystemsData,
  setActiveMagmaticSystem,
  updateMagmaticSystem,
  setMagmaticSystemsError,
  
  // Data quality management
  setOverallDataQuality,
  setDatasetQuality,
  setValidationResults,
  
  // Analysis settings management
  setSeismicAnalysisSettings,
  setVolcanicAnalysisSettings,
  setDeformationAnalysisSettings,
  setResourceAnalysisSettings,
  
  // Processing management
  addProcessingJob,
  updateJobProgress,
  completeProcessingJob,
  failProcessingJob,
  
  // Cache management
  setCacheSettings,
  updateCacheStats,
  
  // Reset functions
  resetSeismicData,
  resetVolcanicData,
  resetDeformationData,
  resetResourceData,
  resetAllGeologicalData,
} = geologicalSlice.actions;

export default geologicalSlice.reducer;