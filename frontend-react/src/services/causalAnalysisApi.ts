/**
 * Causal Analysis API Client
 * 
 * Production-grade API client for interacting with the causal analysis service.
 * Provides comprehensive causal discovery, hypothesis testing, and real-time updates.
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { io, Socket } from 'socket.io-client';
import { 
  CausalGraph, 
  CausalDiscoveryRequest, 
  CausalAnalysisResponse,
  HypothesisTestRequest,
  CausalMetrics,
  RealTimeUpdate
} from '../types/causal';

export interface ApiConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  retryDelay: number;
  enableWebSocket: boolean;
  wsURL?: string;
}

export interface AuthToken {
  token: string;
  expiresAt: Date;
}

export class CausalAnalysisApiClient {
  private client: AxiosInstance;
  private wsClient: Socket | null = null;
  private config: ApiConfig;
  private authToken: AuthToken | null = null;
  private retryCount = new Map<string, number>();

  constructor(config: Partial<ApiConfig> = {}) {
    this.config = {
      baseURL: process.env.REACT_APP_CAUSAL_API_URL || 'https://api.prometheus-ai.platform/causal',
      timeout: 30000,
      retries: 3,
      retryDelay: 1000,
      enableWebSocket: true,
      wsURL: process.env.REACT_APP_CAUSAL_WS_URL || 'wss://api.prometheus-ai.platform/ws/causal',
      ...config,
    };

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
    
    if (this.config.enableWebSocket) {
      this.initializeWebSocket();
    }
  }

  private setupInterceptors(): void {
    // Request interceptor for authentication
    this.client.interceptors.request.use(
      (config) => {
        if (this.authToken && this.authToken.expiresAt > new Date()) {
          config.headers.Authorization = `Bearer ${this.authToken.token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling and retries
    this.client.interceptors.response.use(
      (response) => {
        // Reset retry count on successful response
        const requestId = this.getRequestId(response.config);
        this.retryCount.delete(requestId);
        return response;
      },
      async (error) => {
        const config = error.config;
        const requestId = this.getRequestId(config);
        const currentRetries = this.retryCount.get(requestId) || 0;

        // Retry logic for network errors and 5xx responses
        if (
          currentRetries < this.config.retries &&
          (error.code === 'NETWORK_ERROR' || 
           error.code === 'TIMEOUT' ||
           (error.response && error.response.status >= 500))
        ) {
          this.retryCount.set(requestId, currentRetries + 1);
          
          // Exponential backoff
          const delay = this.config.retryDelay * Math.pow(2, currentRetries);
          await new Promise(resolve => setTimeout(resolve, delay));
          
          return this.client.request(config);
        }

        this.retryCount.delete(requestId);
        return Promise.reject(error);
      }
    );
  }

  private getRequestId(config: AxiosRequestConfig): string {
    return `${config.method}-${config.url}-${JSON.stringify(config.params)}`;
  }

  private initializeWebSocket(): void {
    try {
      this.wsClient = io(this.config.wsURL!, {
        transports: ['websocket'],
        upgrade: true,
        rememberUpgrade: true,
        timeout: 20000,
        forceNew: true,
      });

      this.wsClient.on('connect', () => {
        console.log('WebSocket connected to causal analysis service');
      });

      this.wsClient.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        if (reason === 'io server disconnect') {
          // Reconnect manually if server disconnected
          setTimeout(() => this.wsClient?.connect(), 5000);
        }
      });

      this.wsClient.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
      });

    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  }

  /**
   * Set authentication token
   */
  setAuthToken(token: string, expiresAt: Date): void {
    this.authToken = { token, expiresAt };
  }

  /**
   * Clear authentication token
   */
  clearAuthToken(): void {
    this.authToken = null;
  }

  /**
   * Check service health
   */
  async checkHealth(): Promise<{ status: string; version: string; timestamp: string }> {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw new Error(`Health check failed: ${error}`);
    }
  }

  /**
   * Check service readiness
   */
  async checkReadiness(): Promise<{ status: string; databases: Record<string, string> }> {
    try {
      const response = await this.client.get('/ready');
      return response.data;
    } catch (error) {
      throw new Error(`Readiness check failed: ${error}`);
    }
  }

  /**
   * Discover causal relationships in geological data
   */
  async discoverCausalRelationships(
    request: CausalDiscoveryRequest,
    onProgress?: (progress: number) => void
  ): Promise<CausalAnalysisResponse> {
    try {
      // Set up progress tracking if callback provided
      if (onProgress && this.wsClient) {
        const progressHandler = (data: any) => {
          if (data.type === 'discovery_progress') {
            onProgress(data.progress);
          }
        };
        this.wsClient.on('causal_discovery_progress', progressHandler);
      }

      const response = await this.client.post<CausalAnalysisResponse>(
        '/api/v1/causal/discover',
        request
      );

      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Causal discovery failed: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  /**
   * Retrieve a specific causal graph
   */
  async getCausalGraph(graphId: string): Promise<CausalGraph> {
    try {
      const response = await this.client.get<CausalGraph>(`/api/v1/causal/graphs/${graphId}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 404) {
          throw new Error(`Causal graph not found: ${graphId}`);
        }
        throw new Error(`Failed to retrieve causal graph: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  /**
   * List available causal graphs
   */
  async listCausalGraphs(
    filters?: {
      dateRange?: [Date, Date];
      processTypes?: string[];
      minNodes?: number;
      maxNodes?: number;
    }
  ): Promise<CausalGraph[]> {
    try {
      const params: Record<string, any> = {};
      
      if (filters?.dateRange) {
        params.start_date = filters.dateRange[0].toISOString();
        params.end_date = filters.dateRange[1].toISOString();
      }
      
      if (filters?.processTypes) {
        params.process_types = filters.processTypes.join(',');
      }
      
      if (filters?.minNodes !== undefined) {
        params.min_nodes = filters.minNodes;
      }
      
      if (filters?.maxNodes !== undefined) {
        params.max_nodes = filters.maxNodes;
      }

      const response = await this.client.get<CausalGraph[]>('/api/v1/causal/graphs', { params });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Failed to list causal graphs: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  /**
   * Test a causal hypothesis
   */
  async testHypothesis(request: HypothesisTestRequest): Promise<{
    hypothesis_id: string;
    test_results: {
      bootstrap_p_value: number;
      permutation_p_value: number;
      confidence_interval: [number, number];
      effect_size: number;
    };
    conclusion: string;
    recommendations: string[];
  }> {
    try {
      const response = await this.client.post('/api/v1/causal/hypothesis/test', request);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Hypothesis testing failed: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  /**
   * Get causal analysis performance metrics
   */
  async getMetrics(): Promise<CausalMetrics> {
    try {
      const response = await this.client.get<CausalMetrics>('/api/v1/causal/metrics');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Failed to retrieve metrics: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  /**
   * Subscribe to real-time causal analysis updates
   */
  subscribeToRealTimeUpdates(
    callbacks: {
      onGraphUpdate?: (update: RealTimeUpdate) => void;
      onDiscoveryComplete?: (graphId: string, nodeCount: number, edgeCount: number) => void;
      onError?: (error: string) => void;
      onConnect?: () => void;
      onDisconnect?: () => void;
    }
  ): () => void {
    if (!this.wsClient) {
      throw new Error('WebSocket not initialized');
    }

    // Set up event handlers
    if (callbacks.onConnect) {
      this.wsClient.on('connect', callbacks.onConnect);
    }

    if (callbacks.onDisconnect) {
      this.wsClient.on('disconnect', callbacks.onDisconnect);
    }

    if (callbacks.onGraphUpdate) {
      this.wsClient.on('causal_graph_update', callbacks.onGraphUpdate);
    }

    if (callbacks.onDiscoveryComplete) {
      this.wsClient.on('causal_discovery_complete', (data) => {
        callbacks.onDiscoveryComplete!(data.graph_id, data.nodes, data.edges);
      });
    }

    if (callbacks.onError) {
      this.wsClient.on('error', callbacks.onError);
    }

    // Connect to real-time endpoint
    this.wsClient.emit('subscribe', { type: 'causal_updates' });

    // Return unsubscribe function
    return () => {
      if (this.wsClient) {
        this.wsClient.emit('unsubscribe', { type: 'causal_updates' });
        this.wsClient.off('connect');
        this.wsClient.off('disconnect');
        this.wsClient.off('causal_graph_update');
        this.wsClient.off('causal_discovery_complete');
        this.wsClient.off('error');
      }
    };
  }

  /**
   * Subscribe to a specific causal graph updates
   */
  subscribeToGraph(
    graphId: string,
    onUpdate: (graph: CausalGraph) => void
  ): () => void {
    if (!this.wsClient) {
      throw new Error('WebSocket not initialized');
    }

    const handler = (data: any) => {
      if (data.graph_id === graphId) {
        onUpdate(data.graph);
      }
    };

    this.wsClient.on('graph_update', handler);
    this.wsClient.emit('subscribe_graph', { graph_id: graphId });

    return () => {
      if (this.wsClient) {
        this.wsClient.off('graph_update', handler);
        this.wsClient.emit('unsubscribe_graph', { graph_id: graphId });
      }
    };
  }

  /**
   * Send ping to keep WebSocket connection alive
   */
  ping(): void {
    if (this.wsClient && this.wsClient.connected) {
      this.wsClient.emit('ping');
    }
  }

  /**
   * Export causal graph data
   */
  async exportGraph(
    graphId: string,
    format: 'json' | 'csv' | 'graphml' | 'gexf' | 'png' | 'svg'
  ): Promise<Blob> {
    try {
      const response = await this.client.get(
        `/api/v1/causal/graphs/${graphId}/export`,
        {
          params: { format },
          responseType: 'blob',
        }
      );

      return new Blob([response.data], {
        type: this.getContentType(format),
      });
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Export failed: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  private getContentType(format: string): string {
    const contentTypes: Record<string, string> = {
      json: 'application/json',
      csv: 'text/csv',
      graphml: 'application/xml',
      gexf: 'application/xml',
      png: 'image/png',
      svg: 'image/svg+xml',
    };
    return contentTypes[format] || 'application/octet-stream';
  }

  /**
   * Batch process multiple causal discovery requests
   */
  async batchDiscovery(
    requests: CausalDiscoveryRequest[],
    onProgress?: (completed: number, total: number) => void
  ): Promise<CausalAnalysisResponse[]> {
    const results: CausalAnalysisResponse[] = [];
    
    for (let i = 0; i < requests.length; i++) {
      try {
        const result = await this.discoverCausalRelationships(requests[i]);
        results.push(result);
        
        if (onProgress) {
          onProgress(i + 1, requests.length);
        }
      } catch (error) {
        console.error(`Batch discovery failed for request ${i}:`, error);
        // Continue with other requests
      }
    }
    
    return results;
  }

  /**
   * Get causal pathway between two nodes
   */
  async getCausalPathway(
    graphId: string,
    sourceNodeId: string,
    targetNodeId: string,
    maxDepth: number = 5
  ): Promise<{
    paths: Array<{
      nodes: string[];
      edges: string[];
      totalStrength: number;
      confidence: number;
    }>;
    directConnection: boolean;
  }> {
    try {
      const response = await this.client.get(
        `/api/v1/causal/graphs/${graphId}/pathway`,
        {
          params: {
            source: sourceNodeId,
            target: targetNodeId,
            max_depth: maxDepth,
          },
        }
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Pathway analysis failed: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  /**
   * Validate causal graph against domain knowledge
   */
  async validateGraph(
    graphId: string,
    domainKnowledge?: {
      forbiddenEdges?: Array<[string, string]>;
      requiredEdges?: Array<[string, string]>;
      processConstraints?: Record<string, string[]>;
    }
  ): Promise<{
    isValid: boolean;
    violations: Array<{
      type: 'forbidden_edge' | 'missing_required_edge' | 'process_constraint';
      description: string;
      severity: 'error' | 'warning';
    }>;
    confidence: number;
  }> {
    try {
      const response = await this.client.post(
        `/api/v1/causal/graphs/${graphId}/validate`,
        { domain_knowledge: domainKnowledge }
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(`Graph validation failed: ${error.response?.data?.detail || error.message}`);
      }
      throw error;
    }
  }

  /**
   * Close connections and cleanup
   */
  disconnect(): void {
    if (this.wsClient) {
      this.wsClient.disconnect();
      this.wsClient = null;
    }
  }
}

// Create singleton instance
export const causalAnalysisApi = new CausalAnalysisApiClient();

// Export types for convenience
export type {
  CausalGraph,
  CausalDiscoveryRequest,
  CausalAnalysisResponse,
  HypothesisTestRequest,
  CausalMetrics,
  RealTimeUpdate,
} from '../types/causal';