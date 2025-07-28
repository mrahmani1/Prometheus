import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import {
  SeismicEvent,
  DeformationMeasurement,
  GeochemicalSample,
  ThermalMeasurement,
  VolcanicRiskAssessment,
  ModelPrediction,
  SimulationParameters,
} from '@/types/geological';
import { geologicalDataService } from '@/services/geologicalDataService';

interface GeologicalDataState {
  seismicEvents: SeismicEvent[];
  deformationMeasurements: DeformationMeasurement[];
  geochemicalSamples: GeochemicalSample[];
  thermalMeasurements: ThermalMeasurement[];
  riskAssessments: VolcanicRiskAssessment[];
  modelPredictions: ModelPrediction[];
  activeSimulation: {
    parameters: SimulationParameters | null;
    results: ModelPrediction[];
    isRunning: boolean;
    progress: number;
  };
  selectedTimeRange: [string, string] | null;
  selectedLocation: {
    latitude: number;
    longitude: number;
    radius: number;
  } | null;
  filters: {
    eventTypes: string[];
    magnitudeRange: [number, number] | null;
    depthRange: [number, number] | null;
    confidenceThreshold: number;
  };
  isLoading: boolean;
  error: string | null;
}

const initialState: GeologicalDataState = {
  seismicEvents: [],
  deformationMeasurements: [],
  geochemicalSamples: [],
  thermalMeasurements: [],
  riskAssessments: [],
  modelPredictions: [],
  activeSimulation: {
    parameters: null,
    results: [],
    isRunning: false,
    progress: 0,
  },
  selectedTimeRange: null,
  selectedLocation: null,
  filters: {
    eventTypes: [],
    magnitudeRange: null,
    depthRange: null,
    confidenceThreshold: 0.5,
  },
  isLoading: false,
  error: null,
};

// Async thunks for data fetching
export const fetchSeismicEvents = createAsyncThunk(
  'geologicalData/fetchSeismicEvents',
  async (params: { timeRange?: [string, string]; location?: any }, { rejectWithValue }) => {
    try {
      return await geologicalDataService.getSeismicEvents(JSON.stringify(params));
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch seismic events');
    }
  }
);

export const fetchDeformationData = createAsyncThunk(
  'geologicalData/fetchDeformationData',
  async (params: { timeRange?: [string, string]; location?: any }, { rejectWithValue }) => {
    try {
      return await geologicalDataService.getDeformationMeasurements(params);
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch deformation data');
    }
  }
);

export const fetchGeochemicalData = createAsyncThunk(
  'geologicalData/fetchGeochemicalData',
  async (params: { timeRange?: [string, string]; location?: any }, { rejectWithValue }) => {
    try {
      return await geologicalDataService.getGeochemicalSamples(params);
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch geochemical data');
    }
  }
);

export const fetchThermalData = createAsyncThunk(
  'geologicalData/fetchThermalData',
  async (params: { timeRange?: [string, string]; location?: any }, { rejectWithValue }) => {
    try {
      return await geologicalDataService.getThermalMeasurements(params);
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch thermal data');
    }
  }
);

export const fetchRiskAssessments = createAsyncThunk(
  'geologicalData/fetchRiskAssessments',
  async (volcanoId: string, { rejectWithValue }) => {
    try {
      return await geologicalDataService.getRiskAssessments(volcanoId);
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch risk assessments');
    }
  }
);

export const runSimulation = createAsyncThunk(
  'geologicalData/runSimulation',
  async (parameters: SimulationParameters, { rejectWithValue, dispatch }) => {
    try {
      // Start simulation
      const simulationId = await geologicalDataService.startSimulation(parameters);
      
      // Poll for progress
      const pollProgress = async () => {
        const status = await geologicalDataService.getSimulationStatus(simulationId);
        dispatch(updateSimulationProgress(status.progress));
        
        if (status.isComplete) {
          const results = await geologicalDataService.getSimulationResults(simulationId);
          return results;
        } else if (status.hasError) {
          throw new Error(status.error);
        } else {
          // Continue polling
          setTimeout(pollProgress, 1000);
        }
      };
      
      return await pollProgress();
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Simulation failed');
    }
  }
);

const geologicalDataSlice = createSlice({
  name: 'geologicalData',
  initialState,
  reducers: {
    // Filter actions
    setTimeRange: (state, action: PayloadAction<[string, string] | null>) => {
      state.selectedTimeRange = action.payload;
    },
    setLocationFilter: (state, action: PayloadAction<{ latitude: number; longitude: number; radius: number } | null>) => {
      state.selectedLocation = action.payload;
    },
    updateFilters: (state, action: PayloadAction<Partial<typeof initialState.filters>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    clearFilters: (state) => {
      state.filters = initialState.filters;
      state.selectedTimeRange = null;
      state.selectedLocation = null;
    },
    
    // Simulation actions
    setSimulationParameters: (state, action: PayloadAction<SimulationParameters>) => {
      state.activeSimulation.parameters = action.payload;
    },
    updateSimulationProgress: (state, action: PayloadAction<number>) => {
      state.activeSimulation.progress = action.payload;
    },
    clearSimulation: (state) => {
      state.activeSimulation = initialState.activeSimulation;
    },
    
    // Data management actions
    addSeismicEvent: (state, action: PayloadAction<SeismicEvent>) => {
      state.seismicEvents.push(action.payload);
    },
    updateSeismicEvent: (state, action: PayloadAction<SeismicEvent>) => {
      const index = state.seismicEvents.findIndex(event => event.id === action.payload.id);
      if (index !== -1) {
        state.seismicEvents[index] = action.payload;
      }
    },
    removeSeismicEvent: (state, action: PayloadAction<string>) => {
      state.seismicEvents = state.seismicEvents.filter(event => event.id !== action.payload);
    },
    
    // Clear data actions
    clearAllData: (state) => {
      state.seismicEvents = [];
      state.deformationMeasurements = [];
      state.geochemicalSamples = [];
      state.thermalMeasurements = [];
      state.riskAssessments = [];
      state.modelPredictions = [];
    },
    
    // Error handling
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Seismic events
      .addCase(fetchSeismicEvents.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchSeismicEvents.fulfilled, (state, action) => {
        state.isLoading = false;
        state.seismicEvents = action.payload;
      })
      .addCase(fetchSeismicEvents.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      
      // Deformation data
      .addCase(fetchDeformationData.fulfilled, (state, action) => {
        state.deformationMeasurements = action.payload;
      })
      
      // Geochemical data
      .addCase(fetchGeochemicalData.fulfilled, (state, action) => {
        state.geochemicalSamples = action.payload;
      })
      
      // Thermal data
      .addCase(fetchThermalData.fulfilled, (state, action) => {
        state.thermalMeasurements = action.payload;
      })
      
      // Risk assessments
      .addCase(fetchRiskAssessments.fulfilled, (state, action) => {
        state.riskAssessments = action.payload;
      })
      
      // Simulation
      .addCase(runSimulation.pending, (state) => {
        state.activeSimulation.isRunning = true;
        state.activeSimulation.progress = 0;
        state.error = null;
      })
      .addCase(runSimulation.fulfilled, (state, action) => {
        state.activeSimulation.isRunning = false;
        state.activeSimulation.results = action.payload || [];
        state.activeSimulation.progress = 100;
      })
      .addCase(runSimulation.rejected, (state, action) => {
        state.activeSimulation.isRunning = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  setTimeRange,
  setLocationFilter,
  updateFilters,
  clearFilters,
  setSimulationParameters,
  updateSimulationProgress,
  clearSimulation,
  addSeismicEvent,
  updateSeismicEvent,
  removeSeismicEvent,
  clearAllData,
  clearError,
} = geologicalDataSlice.actions;

export default geologicalDataSlice.reducer;