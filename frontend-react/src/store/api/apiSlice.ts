/**
 * RTK Query API slice for Prometheus AI Platform
 * Provides comprehensive API integration with caching and real-time updates
 */

import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import type { BaseQueryFn, FetchArgs, FetchBaseQueryError } from '@reduxjs/toolkit/query';

import type { RootState } from '../index';
import type {
  ApiResponse,
  AuthRequest,
  AuthResponse,
  SeismicDataRequest,
  SeismicDataResponse,
  VolcanicMonitoringRequest,
  VolcanicMonitoringResponse,
  DeformationAnalysisRequest,
  DeformationAnalysisResponse,
  ResourceAssessmentRequest,
  ResourceAssessmentResponse,
  ModelRequest,
  ModelResponse,
  ExportRequest,
  ExportResponse,
  ImportRequest,
  ImportResponse,
  HealthCheckResponse,
  SystemMetrics,
  SystemConfig,
} from '../../types/api';

// Base query with authentication
const baseQuery = fetchBaseQuery({
  baseUrl: '/api/v1',
  prepareHeaders: (headers, { getState }) => {
    const token = (getState() as RootState).auth.token;
    if (token) {
      headers.set('authorization', `Bearer ${token}`);
    }
    headers.set('content-type', 'application/json');
    return headers;
  },
});

// Enhanced base query with error handling and retry logic
const baseQueryWithReauth: BaseQueryFn<
  string | FetchArgs,
  unknown,
  FetchBaseQueryError
> = async (args, api, extraOptions) => {
  let result = await baseQuery(args, api, extraOptions);

  if (result.error && result.error.status === 401) {
    // Try to refresh token
    const refreshResult = await baseQuery(
      {
        url: '/auth/refresh',
        method: 'POST',
        body: {
          refreshToken: (api.getState() as RootState).auth.refreshToken,
        },
      },
      api,
      extraOptions
    );

    if (refreshResult.data) {
      // Store new token and retry original request
      api.dispatch(
        // Update auth state with new token data
        { type: 'auth/setCredentials', payload: refreshResult.data }
      );
      result = await baseQuery(args, api, extraOptions);
    } else {
      // Refresh failed, logout user
      api.dispatch({ type: 'auth/logout' });
    }
  }

  return result;
};

// Main API slice
export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: baseQueryWithReauth,
  tagTypes: [
    'Auth',
    'SeismicData',
    'VolcanicMonitoring',
    'DeformationAnalysis',
    'ResourceAssessment',
    'Model',
    'Export',
    'Import',
    'Health',
    'Metrics',
    'Config',
  ],
  endpoints: builder => ({
    // Authentication endpoints
    login: builder.mutation<AuthResponse, AuthRequest>({
      query: credentials => ({
        url: '/auth/login',
        method: 'POST',
        body: credentials,
      }),
      invalidatesTags: ['Auth'],
    }),

    logout: builder.mutation<void, void>({
      query: () => ({
        url: '/auth/logout',
        method: 'POST',
      }),
      invalidatesTags: ['Auth'],
    }),

    refreshToken: builder.mutation<AuthResponse, { refreshToken: string }>({
      query: ({ refreshToken }) => ({
        url: '/auth/refresh',
        method: 'POST',
        body: { refreshToken },
      }),
      invalidatesTags: ['Auth'],
    }),

    // Seismic data endpoints
    getSeismicData: builder.query<SeismicDataResponse, SeismicDataRequest>({
      query: params => ({
        url: '/seismic/events',
        params,
      }),
      providesTags: ['SeismicData'],
      keepUnusedDataFor: 300, // 5 minutes
    }),

    getSeismicEvent: builder.query<ApiResponse<any>, string>({
      query: eventId => `/seismic/events/${eventId}`,
      providesTags: (_result, _error, eventId) => [{ type: 'SeismicData', id: eventId }],
    }),

    analyzeSeismicWaveform: builder.mutation<ApiResponse<any>, { eventId: string; analysisType: string }>({
      query: ({ eventId, analysisType }) => ({
        url: `/seismic/events/${eventId}/analyze`,
        method: 'POST',
        body: { analysisType },
      }),
      invalidatesTags: (_result, _error, { eventId }) => [{ type: 'SeismicData', id: eventId }],
    }),

    // Volcanic monitoring endpoints
    getVolcanicMonitoring: builder.query<VolcanicMonitoringResponse, VolcanicMonitoringRequest>({
      query: params => ({
        url: '/volcanic/monitoring',
        params,
      }),
      providesTags: ['VolcanicMonitoring'],
      keepUnusedDataFor: 60, // 1 minute for real-time data
    }),

    getVolcanoDetails: builder.query<ApiResponse<any>, string>({
      query: volcanoId => `/volcanic/volcanoes/${volcanoId}`,
      providesTags: (_result, _error, volcanoId) => [{ type: 'VolcanicMonitoring', id: volcanoId }],
    }),

    updateAlertLevel: builder.mutation<ApiResponse<any>, { volcanoId: string; alertLevel: string; reason: string }>({
      query: ({ volcanoId, alertLevel, reason }) => ({
        url: `/volcanic/volcanoes/${volcanoId}/alert`,
        method: 'PUT',
        body: { alertLevel, reason },
      }),
      invalidatesTags: (_result, _error, { volcanoId }) => [{ type: 'VolcanicMonitoring', id: volcanoId }],
    }),

    generateEruptionForecast: builder.mutation<ApiResponse<any>, { volcanoId: string; timeWindow: string }>({
      query: ({ volcanoId, timeWindow }) => ({
        url: `/volcanic/volcanoes/${volcanoId}/forecast`,
        method: 'POST',
        body: { timeWindow },
      }),
      invalidatesTags: (_result, _error, { volcanoId }) => [{ type: 'VolcanicMonitoring', id: volcanoId }],
    }),

    // Deformation analysis endpoints
    getDeformationAnalysis: builder.query<DeformationAnalysisResponse, DeformationAnalysisRequest>({
      query: params => ({
        url: '/deformation/analysis',
        params,
      }),
      providesTags: ['DeformationAnalysis'],
      keepUnusedDataFor: 300,
    }),

    processInSARData: builder.mutation<ApiResponse<any>, { datasetId: string; processingParams: any }>({
      query: ({ datasetId, processingParams }) => ({
        url: `/deformation/insar/${datasetId}/process`,
        method: 'POST',
        body: processingParams,
      }),
      invalidatesTags: ['DeformationAnalysis'],
    }),

    generateDeformationModel: builder.mutation<ApiResponse<any>, { region: any; sourceType: string; parameters: any }>({
      query: ({ region, sourceType, parameters }) => ({
        url: '/deformation/model',
        method: 'POST',
        body: { region, sourceType, parameters },
      }),
      invalidatesTags: ['DeformationAnalysis'],
    }),

    // Resource assessment endpoints
    getResourceAssessment: builder.query<ResourceAssessmentResponse, ResourceAssessmentRequest>({
      query: params => ({
        url: '/resources/assessment',
        params,
      }),
      providesTags: ['ResourceAssessment'],
      keepUnusedDataFor: 600, // 10 minutes
    }),

    analyzeProspectivity: builder.mutation<ApiResponse<any>, { region: any; resourceType: string; parameters: any }>({
      query: ({ region, resourceType, parameters }) => ({
        url: '/resources/prospectivity',
        method: 'POST',
        body: { region, resourceType, parameters },
      }),
      invalidatesTags: ['ResourceAssessment'],
    }),

    generateEconomicAnalysis: builder.mutation<ApiResponse<any>, { prospectId: string; parameters: any }>({
      query: ({ prospectId, parameters }) => ({
        url: `/resources/prospects/${prospectId}/economics`,
        method: 'POST',
        body: parameters,
      }),
      invalidatesTags: (_result, _error, { prospectId }) => [{ type: 'ResourceAssessment', id: prospectId }],
    }),

    // Model management endpoints
    getModels: builder.query<ApiResponse<ModelResponse[]>, { modelType?: string; status?: string }>({
      query: params => ({
        url: '/models',
        params,
      }),
      providesTags: ['Model'],
    }),

    getModel: builder.query<ApiResponse<ModelResponse>, string>({
      query: modelId => `/models/${modelId}`,
      providesTags: (_result, _error, modelId) => [{ type: 'Model', id: modelId }],
    }),

    trainModel: builder.mutation<ApiResponse<ModelResponse>, ModelRequest>({
      query: modelRequest => ({
        url: '/models/train',
        method: 'POST',
        body: modelRequest,
      }),
      invalidatesTags: ['Model'],
    }),

    deployModel: builder.mutation<ApiResponse<any>, { modelId: string; deploymentConfig: any }>({
      query: ({ modelId, deploymentConfig }) => ({
        url: `/models/${modelId}/deploy`,
        method: 'POST',
        body: deploymentConfig,
      }),
      invalidatesTags: (_result, _error, { modelId }) => [{ type: 'Model', id: modelId }],
    }),

    makePrediction: builder.mutation<ApiResponse<any>, { modelId: string; input: any }>({
      query: ({ modelId, input }) => ({
        url: `/models/${modelId}/predict`,
        method: 'POST',
        body: { input },
      }),
    }),

    // Data export/import endpoints
    exportData: builder.mutation<ExportResponse, ExportRequest>({
      query: exportRequest => ({
        url: '/data/export',
        method: 'POST',
        body: exportRequest,
      }),
      invalidatesTags: ['Export'],
    }),

    getExportStatus: builder.query<ExportResponse, string>({
      query: exportId => `/data/export/${exportId}`,
      providesTags: (_result, _error, exportId) => [{ type: 'Export', id: exportId }],
    }),

    importData: builder.mutation<ImportResponse, ImportRequest>({
      query: importRequest => ({
        url: '/data/import',
        method: 'POST',
        body: importRequest,
      }),
      invalidatesTags: ['Import'],
    }),

    getImportStatus: builder.query<ImportResponse, string>({
      query: importId => `/data/import/${importId}`,
      providesTags: (_result, _error, importId) => [{ type: 'Import', id: importId }],
    }),

    // System health and monitoring endpoints
    getHealthCheck: builder.query<HealthCheckResponse, void>({
      query: () => '/health',
      providesTags: ['Health'],
      keepUnusedDataFor: 30, // 30 seconds
    }),

    getSystemMetrics: builder.query<SystemMetrics, void>({
      query: () => '/metrics',
      providesTags: ['Metrics'],
      keepUnusedDataFor: 60, // 1 minute
    }),

    getSystemConfig: builder.query<SystemConfig, void>({
      query: () => '/config',
      providesTags: ['Config'],
    }),

    updateSystemConfig: builder.mutation<ApiResponse<SystemConfig>, Partial<SystemConfig>>({
      query: config => ({
        url: '/config',
        method: 'PUT',
        body: config,
      }),
      invalidatesTags: ['Config'],
    }),

    // Real-time data streaming endpoints
    subscribeToStream: builder.mutation<ApiResponse<any>, { topics: string[]; filters?: any }>({
      query: ({ topics, filters }) => ({
        url: '/stream/subscribe',
        method: 'POST',
        body: { topics, filters },
      }),
    }),

    unsubscribeFromStream: builder.mutation<ApiResponse<any>, { subscriptionId: string }>({
      query: ({ subscriptionId }) => ({
        url: `/stream/unsubscribe/${subscriptionId}`,
        method: 'DELETE',
      }),
    }),

    // Advanced geological analysis endpoints
    analyze3DGeology: builder.mutation<ApiResponse<any>, { region: any; dataTypes: string[]; parameters: any }>({
      query: ({ region, dataTypes, parameters }) => ({
        url: '/geology/3d-analysis',
        method: 'POST',
        body: { region, dataTypes, parameters },
      }),
    }),

    generateMagmaticSystemModel: builder.mutation<ApiResponse<any>, { volcanoId: string; seismicData: any; parameters: any }>({
      query: ({ volcanoId, seismicData, parameters }) => ({
        url: `/geology/magmatic-system/${volcanoId}`,
        method: 'POST',
        body: { seismicData, parameters },
      }),
    }),

    analyzeStressField: builder.mutation<ApiResponse<any>, { region: any; structuralData: any; parameters: any }>({
      query: ({ region, structuralData, parameters }) => ({
        url: '/geology/stress-field',
        method: 'POST',
        body: { region, structuralData, parameters },
      }),
    }),

    simulateMantelConvection: builder.mutation<ApiResponse<any>, { parameters: any }>({
      query: ({ parameters }) => ({
        url: '/geology/mantle-convection',
        method: 'POST',
        body: parameters,
      }),
    }),

    // Crystallographic analysis endpoints
    analyzeCrystalStructure: builder.mutation<ApiResponse<any>, { sampleId: string; analysisType: string; parameters: any }>({
      query: ({ sampleId, analysisType, parameters }) => ({
        url: `/crystallography/analyze/${sampleId}`,
        method: 'POST',
        body: { analysisType, parameters },
      }),
    }),

    searchCrystalDatabase: builder.query<ApiResponse<any>, { query: string; filters?: any }>({
      query: ({ query, filters }) => ({
        url: '/crystallography/search',
        params: { query, ...filters },
      }),
    }),

    // Geochemical analysis endpoints
    analyzeGeochemistry: builder.mutation<ApiResponse<any>, { samples: any[]; analysisType: string; parameters: any }>({
      query: ({ samples, analysisType, parameters }) => ({
        url: '/geochemistry/analyze',
        method: 'POST',
        body: { samples, analysisType, parameters },
      }),
    }),

    generateGeochemicalPlots: builder.mutation<ApiResponse<any>, { data: any; plotTypes: string[]; parameters: any }>({
      query: ({ data, plotTypes, parameters }) => ({
        url: '/geochemistry/plots',
        method: 'POST',
        body: { data, plotTypes, parameters },
      }),
    }),
  }),
});

// Export hooks for use in components
export const {
  // Authentication
  useLoginMutation,
  useLogoutMutation,
  useRefreshTokenMutation,

  // Seismic data
  useGetSeismicDataQuery,
  useGetSeismicEventQuery,
  useAnalyzeSeismicWaveformMutation,

  // Volcanic monitoring
  useGetVolcanicMonitoringQuery,
  useGetVolcanoDetailsQuery,
  useUpdateAlertLevelMutation,
  useGenerateEruptionForecastMutation,

  // Deformation analysis
  useGetDeformationAnalysisQuery,
  useProcessInSARDataMutation,
  useGenerateDeformationModelMutation,

  // Resource assessment
  useGetResourceAssessmentQuery,
  useAnalyzeProspectivityMutation,
  useGenerateEconomicAnalysisMutation,

  // Model management
  useGetModelsQuery,
  useGetModelQuery,
  useTrainModelMutation,
  useDeployModelMutation,
  useMakePredictionMutation,

  // Data export/import
  useExportDataMutation,
  useGetExportStatusQuery,
  useImportDataMutation,
  useGetImportStatusQuery,

  // System health
  useGetHealthCheckQuery,
  useGetSystemMetricsQuery,
  useGetSystemConfigQuery,
  useUpdateSystemConfigMutation,

  // Real-time streaming
  useSubscribeToStreamMutation,
  useUnsubscribeFromStreamMutation,

  // Advanced geological analysis
  useAnalyze3DGeologyMutation,
  useGenerateMagmaticSystemModelMutation,
  useAnalyzeStressFieldMutation,
  useSimulateMantelConvectionMutation,

  // Crystallographic analysis
  useAnalyzeCrystalStructureMutation,
  useSearchCrystalDatabaseQuery,

  // Geochemical analysis
  useAnalyzeGeochemistryMutation,
  useGenerateGeochemicalPlotsMutation,
} = apiSlice;