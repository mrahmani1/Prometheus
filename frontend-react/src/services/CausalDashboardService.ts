/**
 * Advanced Causal Dashboard Service
 * Integrates with real geological data sources and backend causal analysis modules
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  CausalGraph, 
  CausalNode,
  CausalEdge,
  // CausalRelationship, // Unused
  CausalAnalysisResult,
  GeologicalProcessType,
  SpatialScale,
  TemporalScale,
} from '../components/visualization/causal/types';
import DatabaseService from './DatabaseService';
import AcademicAPIService from './AcademicAPIService';

interface RealDataSource {
  id: string;
  name: string;
  type: 'seismic' | 'volcanic' | 'deformation' | 'geochemical' | 'thermal';
  endpoint: string;
  status: 'active' | 'inactive' | 'error';
  lastUpdate: Date;
  dataQuality: number;
  coverage: {
    spatial: { lat: [number, number]; lon: [number, number] };
    temporal: { start: Date; end: Date };
  };
}

interface CausalDiscoveryRequest {
  dataSourceIds: string[];
  algorithms: string[];
  parameters: {
    confidenceThreshold: number;
    temporalLagMax: number;
    spatialScaleFilter: SpatialScale[];
    temporalScaleFilter: TemporalScale[];
  };
  validationMethod: 'cross-validation' | 'bootstrap' | 'holdout';
}

interface RealTimeStreamConfig {
  sources: string[];
  updateFrequency: number;
  bufferSize: number;
  qualityThreshold: number;
  alertThresholds: {
    causalStrengthChange: number;
    newRelationshipStrength: number;
    dataQualityDrop: number;
  };
}

// Database connection configuration interface removed as it's not used

class CausalDashboardService {
  private static instance: CausalDashboardService;
  private apiClient: AxiosInstance;
  private websocket: WebSocket | null = null;
  private databaseService: DatabaseService;
  private academicService: AcademicAPIService;
  // private realTimeCallbacks: Map<string, (data: any) => void> = new Map(); // Unused
  // private connectionConfig: DatabaseConnectionConfig | null = null; // Unused

  private constructor() {
    this.apiClient = axios.create({
      baseURL: process.env.REACT_APP_API_BASE_URL || 'https://api.prometheus-ai.platform/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'PrometheusAI-Frontend/1.0',
      },
    });

    // Add request interceptor for authentication
    this.apiClient.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.apiClient.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );

    // Initialize database and academic services
    this.databaseService = DatabaseService.getInstance();
    this.academicService = AcademicAPIService.getInstance();
  }

  static getInstance(): CausalDashboardService {
    if (!CausalDashboardService.instance) {
      CausalDashboardService.instance = new CausalDashboardService();
    }
    return CausalDashboardService.instance;
  }

  /**
   * Initialize database connections and validate backend modules
   */
  async initializeConnections(): Promise<void> {
    try {
      const response = await this.apiClient.post('/api/v1/causal/initialize', {
        validateConnections: true,
        checkDataSources: true,
        auditPlaceholders: true,
      });

      // Connection config handling removed as interface was unused
      
      if (response.data.placeholdersFound > 0) {
        throw new Error(`Found ${response.data.placeholdersFound} placeholder implementations that need replacement`);
      }

      console.log('Causal dashboard service initialized successfully');
    } catch (error) {
      console.error('Failed to initialize causal dashboard service:', error);
      throw error;
    }
  }

  /**
   * Fetch comprehensive causal graph from real data sources
   */
  async fetchCausalGraph(filters?: {
    processTypes?: GeologicalProcessType[];
    spatialBounds?: { north: number; south: number; east: number; west: number };
    temporalRange?: { start: Date; end: Date };
    confidenceThreshold?: number;
  }): Promise<CausalGraph> {
    try {
      const response: AxiosResponse<CausalGraph> = await this.apiClient.post('/api/v1/causal/graph', {
        filters,
        includeMetadata: true,
        validateRelationships: true,
      });

      // Validate that all data is from real sources
      const graph = response.data;
      this.validateRealDataSources(graph);

      return graph;
    } catch (error) {
      console.error('Failed to fetch causal graph:', error);
      throw error;
    }
  }

  /**
   * Discover new causal relationships using advanced algorithms
   */
  async discoverCausalRelationships(request: CausalDiscoveryRequest): Promise<CausalAnalysisResult> {
    try {
      const response: AxiosResponse<CausalAnalysisResult> = await this.apiClient.post('/api/v1/causal/discover', request);
      return response.data;
    } catch (error) {
      console.error('Failed to discover causal relationships:', error);
      throw error;
    }
  }

  /**
   * Get available real data sources
   */
  async getRealDataSources(): Promise<RealDataSource[]> {
    try {
      const response: AxiosResponse<RealDataSource[]> = await this.apiClient.get('/api/v1/data-sources/real');
      return response.data;
    } catch (error) {
      console.error('Failed to fetch real data sources:', error);
      throw error;
    }
  }

  /**
   * Validate data source authenticity
   */
  async validateDataSourceAuthenticity(sourceId: string): Promise<{
    isAuthentic: boolean;
    dataSource: string;
    lastValidation: Date;
    qualityScore: number;
    issues: string[];
  }> {
    try {
      const response = await this.apiClient.post(`/api/v1/data-sources/${sourceId}/validate`);
      return response.data;
    } catch (error) {
      console.error('Failed to validate data source authenticity:', error);
      throw error;
    }
  }

  /**
   * Connect to real-time causal analysis stream
   */
  connectRealTimeStream(config: RealTimeStreamConfig, callbacks: {
    onCausalUpdate: (update: any) => void;
    onAlert: (alert: any) => void;
    onStatusChange: (status: 'connected' | 'disconnected' | 'error') => void;
    onDataQuality: (quality: any) => void;
  }): void {
    try {
      const wsUrl = `${process.env.REACT_APP_WS_URL?.replace('http', 'ws') || 'wss://api.prometheus-ai.platform/ws'}/causal-stream`;
      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        console.log('Real-time causal stream connected');
        callbacks.onStatusChange('connected');
        
        // Send configuration
        this.websocket?.send(JSON.stringify({
          type: 'configure',
          config,
        }));
      };

      this.websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'causal_update':
              callbacks.onCausalUpdate(message.data);
              break;
            case 'alert':
              callbacks.onAlert(message.data);
              break;
            case 'data_quality':
              callbacks.onDataQuality(message.data);
              break;
            default:
              console.warn('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.websocket.onclose = () => {
        console.log('Real-time causal stream disconnected');
        callbacks.onStatusChange('disconnected');
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        callbacks.onStatusChange('error');
      };
    } catch (error) {
      console.error('Failed to connect to real-time stream:', error);
      callbacks.onStatusChange('error');
    }
  }

  /**
   * Disconnect from real-time stream
   */
  disconnectRealTimeStream(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  /**
   * Execute hypothesis test on causal relationship
   */
  async testCausalHypothesis(hypothesis: {
    sourceVariable: string;
    targetVariable: string;
    expectedStrength: number;
    temporalLag?: number;
    conditions?: Record<string, any>;
  }): Promise<{
    result: 'accepted' | 'rejected' | 'inconclusive';
    actualStrength: number;
    confidence: [number, number];
    pValue: number;
    statisticalTest: string;
    evidence: any[];
  }> {
    try {
      const response = await this.apiClient.post('/api/v1/causal/hypothesis/test', hypothesis);
      return response.data;
    } catch (error) {
      console.error('Failed to test causal hypothesis:', error);
      throw error;
    }
  }

  /**
   * Get multi-scale temporal analysis
   */
  async getMultiScaleTemporalAnalysis(nodeId: string, timeRange: {
    start: Date;
    end: Date;
    scales: TemporalScale[];
  }): Promise<{
    scales: Array<{
      scale: TemporalScale;
      data: Array<{ timestamp: Date; value: number; confidence: [number, number] }>;
      trends: any[];
      patterns: any[];
    }>;
    crossScaleRelationships: Array<{
      sourceScale: TemporalScale;
      targetScale: TemporalScale;
      coupling: number;
      lagTime: number;
    }>;
  }> {
    try {
      const response = await this.apiClient.post('/api/v1/causal/temporal/multi-scale', {
        nodeId,
        timeRange,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to get multi-scale temporal analysis:', error);
      throw error;
    }
  }

  /**
   * Export causal graph data
   */
  async exportCausalData(format: 'json' | 'csv' | 'graphml' | 'cytoscape', options?: {
    includeMetadata?: boolean;
    includeValidation?: boolean;
    filterByConfidence?: number;
  }): Promise<Blob> {
    try {
      const response = await this.apiClient.post('/api/v1/causal/export', {
        format,
        options,
      }, {
        responseType: 'blob',
      });
      
      return new Blob([response.data], { 
        type: this.getContentType(format) 
      });
    } catch (error) {
      console.error('Failed to export causal data:', error);
      throw error;
    }
  }

  /**
   * Share causal dashboard configuration
   */
  async shareDashboard(shareConfig: {
    title: string;
    description?: string;
    graphConfig: any;
    filterConfig: any;
    permissions: {
      public: boolean;
      allowEdit: boolean;
      allowDownload: boolean;
    };
    expiresAt?: Date;
  }): Promise<{
    shareId: string;
    shareUrl: string;
    qrCode: string;
  }> {
    try {
      const response = await this.apiClient.post('/api/v1/causal/share', shareConfig);
      return response.data;
    } catch (error) {
      console.error('Failed to share dashboard:', error);
      throw error;
    }
  }

  /**
   * Get collaborative session information
   */
  async getCollaborativeSession(sessionId: string): Promise<{
    id: string;
    participants: Array<{
      id: string;
      name: string;
      avatar?: string;
      status: 'online' | 'away' | 'offline';
      permissions: string[];
    }>;
    sharedGraph: CausalGraph;
    annotations: Array<{
      id: string;
      authorId: string;
      nodeId?: string;
      edgeId?: string;
      content: string;
      timestamp: Date;
    }>;
  }> {
    try {
      const response = await this.apiClient.get(`/api/v1/causal/collaborative/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get collaborative session:', error);
      throw error;
    }
  }

  /**
   * Audit placeholder implementations across backend modules
   */
  async auditPlaceholderImplementations(): Promise<{
    totalModules: number;
    modulesWithPlaceholders: number;
    placeholdersByType: Record<string, number>;
    criticalPlaceholders: Array<{
      module: string;
      file: string;
      line: number;
      type: string;
      description: string;
      priority: 'high' | 'medium' | 'low';
    }>;
    replacementProgress: number;
  }> {
    try {
      const response = await this.apiClient.get('/api/v1/audit/placeholders');
      return response.data;
    } catch (error) {
      console.error('Failed to audit placeholder implementations:', error);
      throw error;
    }
  }

  /**
   * Get database performance metrics
   */
  async getDatabaseMetrics(): Promise<{
    postgresql: {
      connectionCount: number;
      queryPerformance: number;
      storageUsage: number;
      indexEfficiency: number;
    };
    influxdb: {
      writeRate: number;
      queryRate: number;
      seriesCardinality: number;
      retentionPolicy: string;
    };
    neo4j: {
      nodeCount: number;
      relationshipCount: number;
      queryPerformance: number;
      memoryUsage: number;
    };
    redis: {
      memoryUsage: number;
      hitRate: number;
      connectionCount: number;
      keyspaceSize: number;
    };
    timescaledb: {
      compressionRatio: number;
      chunkCount: number;
      queryPerformance: number;
      continuousAggregates: number;
    };
  }> {
    try {
      const response = await this.apiClient.get('/api/v1/database/metrics');
      return response.data;
    } catch (error) {
      console.error('Failed to get database metrics:', error);
      throw error;
    }
  }

  /**
   * Validate that causal graph uses only real data sources
   */
  private validateRealDataSources(graph: CausalGraph): void {
    // Validate against authenticated real data sources
    const validSources = [
      'iris', 'usgs', 'nasa', 'copernicus', 'asf-hyp3', 'earthchem', 'georoc', 'navdat', 'petdb',
      'global volcanism program', 'volcano discovery', 'smithsonian', 'real-time monitoring',
      'semantic scholar', 'connected papers', 'wiley online', 'geoscience world',
      'postgresql', 'influxdb', 'redis', 'neo4j', 'timescaledb'
    ];

    graph.nodes.forEach(node => {
      const dataSource = node.metadata.dataSource?.toLowerCase() || '';
      const hasValidSource = validSources.some(source => dataSource.includes(source));
      
      if (!hasValidSource) {
        throw new Error(`Node ${node.id} does not have a verified real data source: ${node.metadata.dataSource}. All nodes must connect to authenticated geological databases or academic sources.`);
      }

      // Validate data quality and authenticity
      if ((node as any).confidence < 0.7) {
        console.warn(`Node ${node.id} has low confidence (${(node as any).confidence}). Consider additional validation.`);
      }

      // Check for required metadata
      if (!node.metadata.lastUpdated || (node.metadata as any).dataQuality === undefined) {
        throw new Error(`Node ${node.id} missing required metadata (lastUpdated, dataQuality) for real data validation.`);
      }
    });

    graph.edges.forEach(edge => {
      // Validate causal discovery methods
      const validMethods = [
        'pc algorithm', 'ges', 'lingam', 'causal forests', 'granger causality',
        'transfer entropy', 'cross-correlation', 'literature analysis', 'expert knowledge'
      ];

      const discoveryMethod = edge.metadata.discoveryMethod?.toLowerCase() || '';
      const hasValidMethod = validMethods.some(method => discoveryMethod.includes(method));
      
      if (!hasValidMethod) {
        throw new Error(`Edge ${edge.id} uses unvalidated discovery method: ${edge.metadata.discoveryMethod}. All causal relationships must use established causal inference algorithms or literature validation.`);
      }

      // Validate statistical significance
      if ((edge as any).statisticalSignificance > 0.05) {
        console.warn(`Edge ${edge.id} has low statistical significance (p=${(edge as any).statisticalSignificance}). Consider additional validation.`);
      }
    });
  }

  /**
   * Build causal graph from real database sources
   */
  async buildCausalGraphFromDatabase(params: {
    spatial?: {
      bounds?: {
        minLatitude: number;
        maxLatitude: number;
        minLongitude: number;
        maxLongitude: number;
      };
    };
    temporal?: {
      start: Date;
      end: Date;
    };
    processTypes?: GeologicalProcessType[];
    confidenceThreshold?: number;
  }): Promise<CausalGraph> {
    try {
      // Query causal nodes and relationships from Neo4j
      const [nodes, edges] = await Promise.all([
        this.databaseService.queryCausalNodes({
          processTypes: params.processTypes?.map(p => p.toString()),
          confidenceThreshold: params.confidenceThreshold || 0.7,
        }),
        this.databaseService.queryCausalRelationships({
          confidenceThreshold: params.confidenceThreshold || 0.7,
          validationStatus: ['validated', 'peer_reviewed'],
        }),
      ]);

      // Convert database records to causal graph format
      const causalNodes = nodes.map(node => ({
        id: node.id,
        label: node.label,
        processType: node.process_type as GeologicalProcessType,
        spatialScale: node.spatial_scale as SpatialScale,
        temporalScale: node.temporal_scale as TemporalScale,
        dataSource: node.data_source,
        metadata: {
          ...node.metadata,
          lastUpdated: node.updated_at,
          dataQuality: node.confidence,
        },
        size: node.size,
        color: node.color,
        confidence: node.confidence,
        location: node.latitude && node.longitude ? {
          latitude: node.latitude,
          longitude: node.longitude,
        } : undefined,
      }));

      const causalEdges = edges.map(edge => ({
        id: edge.id,
        source: edge.source_node_id,
        target: edge.target_node_id,
        strength: edge.strength,
        confidence: [edge.confidence_lower, edge.confidence_upper] as [number, number],
        temporalLag: edge.temporal_lag,
        statisticalSignificance: edge.statistical_significance,
        discoveryMethod: edge.discovery_method,
        validationStatus: edge.validation_status,
        metadata: {
          discoveryMethod: edge.discovery_method,
          evidence: edge.evidence,
          lastUpdated: edge.updated_at,
        },
      }));

      return {
        nodes: causalNodes as unknown as CausalNode[],
        edges: causalEdges as unknown as CausalEdge[],
        metadata: {
          createdAt: new Date(),
          lastUpdated: new Date(),
          totalNodes: causalNodes.length,
          totalEdges: causalEdges.length,
          averageCausalStrength: causalEdges.reduce((sum, edge) => sum + edge.strength, 0) / causalEdges.length || 0,
          spatialExtent: {
            north: 90,
            south: -90,
            east: 180,
            west: -180,
          },
          temporalExtent: {
            start: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000),
            end: new Date(),
          },
        },
      };
    } catch (error) {
      console.error('Failed to build causal graph from database:', error);
      throw error;
    }
  }

  /**
   * Enhance causal graph with academic literature
   */
  async enhanceCausalGraphWithLiterature(
    graph: CausalGraph,
    searchTerms: string[]
  ): Promise<CausalGraph> {
    try {
      // Search academic literature for each search term
      const literaturePromises = searchTerms.map(term =>
        this.academicService.searchLiterature({
          query: term,
          limit: 50,
          yearRange: { start: 2015, end: new Date().getFullYear() },
        })
      );

      const literatureResults = await Promise.all(literaturePromises);
      const allPapers = literatureResults.flat();

      // Extract causal relationships from literature
      const causalExtraction = await this.academicService.extractCausalRelationships(
        allPapers,
        'geology'
      );

      // Enhance existing nodes with literature evidence
      const enhancedNodes = graph.nodes.map(node => {
        const relevantPapers = allPapers.filter(paper =>
          paper.title.toLowerCase().includes(node.label.toLowerCase()) ||
          paper.abstract.toLowerCase().includes(node.label.toLowerCase())
        );

        return {
          ...node,
          metadata: {
            ...node.metadata,
            literatureSupport: relevantPapers.length,
            supportingPapers: relevantPapers.slice(0, 5).map(p => ({
              title: p.title,
              authors: p.authors.map(a => a.name).join(', '),
              year: p.year,
              citationCount: p.citationCount,
            })),
          },
        };
      });

      // Add new relationships discovered from literature
      const literatureEdges = causalExtraction.relationships.map(rel => ({
        id: `lit_${rel.source}_${rel.target}`,
        source: rel.source,
        target: rel.target,
        strength: rel.confidence,
        confidence: [rel.confidence * 0.8, rel.confidence] as [number, number],
        statisticalSignificance: 0.05, // Literature-based relationships
        discoveryMethod: 'literature_analysis',
        validationStatus: 'literature_supported',
        metadata: {
          discoveryMethod: 'literature_analysis',
          evidence: rel.evidence,
          supportingPapers: rel.paperIds.length,
          lastUpdated: new Date(),
        },
      }));

      return {
        ...graph,
        nodes: enhancedNodes,
        edges: [...graph.edges, ...literatureEdges] as CausalEdge[],
        metadata: {
          ...graph.metadata,
          lastUpdated: new Date(),
        },
      };
    } catch (error) {
      console.error('Failed to enhance causal graph with literature:', error);
      throw error;
    }
  }

  /**
   * Validate causal hypothesis using academic literature
   */
  async validateCausalHypothesisWithLiterature(hypothesis: {
    sourceVariable: string;
    targetVariable: string;
    expectedStrength: number;
    domain?: string;
  }): Promise<{
    validation: 'supported' | 'contradicted' | 'inconclusive';
    confidence: number;
    supportingPapers: number;
    contradictingPapers: number;
    recommendations: string[];
  }> {
    try {
      const causalHypothesis = {
        id: `hyp_${hypothesis.sourceVariable}_${hypothesis.targetVariable}`,
        hypothesis: `${hypothesis.sourceVariable} has a causal influence on ${hypothesis.targetVariable}`,
        variables: [hypothesis.sourceVariable, hypothesis.targetVariable],
        expectedRelationships: [{
          source: hypothesis.sourceVariable,
          target: hypothesis.targetVariable,
          type: 'causal' as const,
          strength: hypothesis.expectedStrength,
        }],
        supportingEvidence: [],
        contradictingEvidence: [],
        confidence: hypothesis.expectedStrength,
        domain: hypothesis.domain || 'geology',
      };

      const validation = await this.academicService.validateHypothesis(causalHypothesis);

      return {
        validation: validation.validation,
        confidence: validation.confidence,
        supportingPapers: validation.supportingPapers.length,
        contradictingPapers: validation.contradictingPapers.length,
        recommendations: validation.recommendations,
      };
    } catch (error) {
      console.error('Failed to validate hypothesis with literature:', error);
      throw error;
    }
  }



  /**
   * Get content type for export format
   */
  private getContentType(format: string): string {
    const contentTypes: Record<string, string> = {
      json: 'application/json',
      csv: 'text/csv',
      graphml: 'application/xml',
      cytoscape: 'application/json',
    };
    return contentTypes[format] || 'application/octet-stream';
  }
}

export default CausalDashboardService;