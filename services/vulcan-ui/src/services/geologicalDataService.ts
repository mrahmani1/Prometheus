import apiClient from './authService';
import {
  SeismicEvent,
  DeformationMeasurement,
  GeochemicalSample,
  ThermalMeasurement,
  VolcanicRiskAssessment,
  ModelPrediction,
  SimulationParameters,
} from '@/types/geological';

interface DataQueryParams {
  timeRange?: [string, string];
  location?: {
    latitude: number;
    longitude: number;
    radius: number;
  };
  limit?: number;
  offset?: number;
}

interface SimulationStatus {
  id: string;
  isComplete: boolean;
  hasError: boolean;
  error?: string;
  progress: number;
  currentStep?: string;
}

export const geologicalDataService = {
  // Seismic data
  async getSeismicEvents(startTime?: string, endTime?: string): Promise<SeismicEvent[]> {
    const params = startTime && endTime ? { timeRange: [startTime, endTime] } : {};
    const response = await apiClient.get('/data/seismic', { params });
    return response.data;
  },

  async getSeismicEvent(id: string): Promise<SeismicEvent> {
    const response = await apiClient.get(`/data/seismic/${id}`);
    return response.data;
  },

  async createSeismicEvent(event: Omit<SeismicEvent, 'id'>): Promise<SeismicEvent> {
    const response = await apiClient.post('/data/seismic', event);
    return response.data;
  },

  async updateSeismicEvent(id: string, event: Partial<SeismicEvent>): Promise<SeismicEvent> {
    const response = await apiClient.put(`/data/seismic/${id}`, event);
    return response.data;
  },

  async deleteSeismicEvent(id: string): Promise<void> {
    await apiClient.delete(`/data/seismic/${id}`);
  },

  // Deformation data
  async getDeformationData(startTime?: string, endTime?: string): Promise<DeformationMeasurement[]> {
    const params = startTime && endTime ? { timeRange: [startTime, endTime] } : {};
    const response = await apiClient.get('/data/deformation', { params });
    return response.data;
  },
  
  async getDeformationMeasurements(params: DataQueryParams): Promise<DeformationMeasurement[]> {
    const response = await apiClient.get('/data/deformation', { params });
    return response.data;
  },

  async getDeformationMeasurement(id: string): Promise<DeformationMeasurement> {
    const response = await apiClient.get(`/data/deformation/${id}`);
    return response.data;
  },

  async createDeformationMeasurement(measurement: Omit<DeformationMeasurement, 'id'>): Promise<DeformationMeasurement> {
    const response = await apiClient.post('/data/deformation', measurement);
    return response.data;
  },

  // Geochemical data
  async getGeochemicalSamples(params: DataQueryParams): Promise<GeochemicalSample[]> {
    const response = await apiClient.get('/data/geochemical', { params });
    return response.data;
  },

  async getGeochemicalSample(id: string): Promise<GeochemicalSample> {
    const response = await apiClient.get(`/data/geochemical/${id}`);
    return response.data;
  },

  async createGeochemicalSample(sample: Omit<GeochemicalSample, 'id'>): Promise<GeochemicalSample> {
    const response = await apiClient.post('/data/geochemical', sample);
    return response.data;
  },

  // Thermal data
  async getThermalData(startTime?: string, endTime?: string): Promise<ThermalMeasurement[]> {
    const params = startTime && endTime ? { timeRange: [startTime, endTime] } : {};
    const response = await apiClient.get('/data/thermal', { params });
    return response.data;
  },
  
  async getThermalMeasurements(params: DataQueryParams): Promise<ThermalMeasurement[]> {
    const response = await apiClient.get('/data/thermal', { params });
    return response.data;
  },

  async getThermalMeasurement(id: string): Promise<ThermalMeasurement> {
    const response = await apiClient.get(`/data/thermal/${id}`);
    return response.data;
  },

  async createThermalMeasurement(measurement: Omit<ThermalMeasurement, 'id'>): Promise<ThermalMeasurement> {
    const response = await apiClient.post('/data/thermal', measurement);
    return response.data;
  },

  // Risk assessments
  async getRiskAssessments(volcanoId: string): Promise<VolcanicRiskAssessment[]> {
    const response = await apiClient.get(`/data/risk-assessments/${volcanoId}`);
    return response.data;
  },

  async getRiskAssessment(id: string): Promise<VolcanicRiskAssessment> {
    const response = await apiClient.get(`/data/risk-assessments/single/${id}`);
    return response.data;
  },

  async createRiskAssessment(assessment: Omit<VolcanicRiskAssessment, 'timestamp'>): Promise<VolcanicRiskAssessment> {
    const response = await apiClient.post('/data/risk-assessments', assessment);
    return response.data;
  },

  // Model predictions
  async getModelPredictions(startTime?: string, endTime?: string): Promise<ModelPrediction[]> {
    const params = startTime && endTime ? { timeRange: [startTime, endTime] } : {};
    const response = await apiClient.get('/data/predictions', { params });
    return response.data;
  },

  async getModelPrediction(id: string): Promise<ModelPrediction> {
    const response = await apiClient.get(`/data/predictions/${id}`);
    return response.data;
  },

  // Simulations
  async startSimulation(parameters: SimulationParameters): Promise<string> {
    const response = await apiClient.post('/simulations/start', parameters);
    return response.data.simulationId;
  },

  async getSimulationStatus(simulationId: string): Promise<SimulationStatus> {
    const response = await apiClient.get(`/simulations/${simulationId}/status`);
    return response.data;
  },

  async getSimulationResults(simulationId: string): Promise<ModelPrediction[]> {
    const response = await apiClient.get(`/simulations/${simulationId}/results`);
    return response.data;
  },

  async cancelSimulation(simulationId: string): Promise<void> {
    await apiClient.post(`/simulations/${simulationId}/cancel`);
  },

  async getSimulationHistory(): Promise<any[]> {
    const response = await apiClient.get('/simulations/history');
    return response.data;
  },

  // Data export
  async exportData(params: {
    dataTypes: string[];
    format: 'csv' | 'json' | 'geojson';
    timeRange?: [string, string];
    location?: any;
  }): Promise<Blob> {
    const response = await apiClient.post('/data/export', params, {
      responseType: 'blob',
    });
    return response.data;
  },

  // Data import
  async importData(file: File, dataType: string): Promise<{ success: boolean; imported: number; errors: string[] }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('dataType', dataType);

    const response = await apiClient.post('/data/import', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Real-time data streams
  async subscribeToRealTimeData(dataTypes: string[], callback: (data: any) => void): Promise<() => void> {
    // This would typically use WebSocket or Server-Sent Events
    const eventSource = new EventSource(`/api/data/stream?types=${dataTypes.join(',')}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      callback(data);
    };

    eventSource.onerror = (error) => {
      console.error('Real-time data stream error:', error);
    };

    // Return unsubscribe function
    return () => {
      eventSource.close();
    };
  },

  // Data quality and validation
  async validateData(data: any, dataType: string): Promise<{ isValid: boolean; errors: string[] }> {
    const response = await apiClient.post('/data/validate', { data, dataType });
    return response.data;
  },

  async getDataQualityReport(params: DataQueryParams): Promise<any> {
    const response = await apiClient.get('/data/quality-report', { params });
    return response.data;
  },
};