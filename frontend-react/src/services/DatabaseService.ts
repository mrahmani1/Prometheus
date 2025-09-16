/**
 * Database Service
 * Real database connections to PostgreSQL+PostGIS, InfluxDB, Redis, Neo4j, TimescaleDB
 * Replaces all simulated database queries with production database operations
 */

import axios, { AxiosInstance } from 'axios';

export interface DatabaseConnection {
  id: string;
  type: 'postgresql' | 'influxdb' | 'redis' | 'neo4j' | 'timescaledb';
  host: string;
  port: number;
  database: string;
  username?: string;
  ssl: boolean;
  status: 'connected' | 'disconnected' | 'error';
  lastPing: Date;
  responseTime: number;
}

export interface SeismicStationRecord {
  id: string;
  network: string;
  station: string;
  location: string;
  channel: string;
  latitude: number;
  longitude: number;
  elevation: number;
  start_date: Date;
  end_date?: Date;
  sample_rate: number;
  instrument_type: string;
  data_quality: number;
  created_at: Date;
  updated_at: Date;
}

export interface CausalNodeRecord {
  id: string;
  label: string;
  process_type: string;
  spatial_scale: string;
  temporal_scale: string;
  data_source: string;
  metadata: Record<string, any>;
  size: number;
  color: string;
  confidence: number;
  latitude?: number;
  longitude?: number;
  created_at: Date;
  updated_at: Date;
}

export interface CausalEdgeRecord {
  id: string;
  source_node_id: string;
  target_node_id: string;
  relationship_type: string;
  strength: number;
  confidence_lower: number;
  confidence_upper: number;
  temporal_lag?: number;
  statistical_significance: number;
  discovery_method: string;
  validation_status: string;
  evidence: Record<string, any>;
  created_at: Date;
  updated_at: Date;
}

export interface VolcanicSystemRecord {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation: number;
  volcano_type: string;
  tectonic_setting: string;
  last_eruption?: Date;
  alert_level: string;
  monitoring_status: string;
  population_5km: number;
  population_10km: number;
  population_30km: number;
  population_100km: number;
  hazards: string[];
  observatories: string[];
  created_at: Date;
  updated_at: Date;
}

export interface TimeSeriesData {
  timestamp: Date;
  value: number;
  quality: number;
  source: string;
  metadata?: Record<string, any>;
}

export interface GeospatialQuery {
  bounds?: {
    minLatitude: number;
    maxLatitude: number;
    minLongitude: number;
    maxLongitude: number;
  };
  center?: {
    latitude: number;
    longitude: number;
    radius: number; // in kilometers
  };
  polygon?: Array<{ latitude: number; longitude: number }>;
}

class DatabaseService {
  private static instance: DatabaseService;
  private apiClient: AxiosInstance;
  private connections: Map<string, DatabaseConnection> = new Map();

  private constructor() {
    this.apiClient = axios.create({
      baseURL: process.env.REACT_APP_DATABASE_API_URL || 'https://api.prometheus-ai.platform/api/v1/database',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.REACT_APP_DATABASE_API_TOKEN || ''}`,
      },
    });

    this.initializeConnections();
  }

  public static getInstance(): DatabaseService {
    if (!DatabaseService.instance) {
      DatabaseService.instance = new DatabaseService();
    }
    return DatabaseService.instance;
  }

  private async initializeConnections(): Promise<void> {
    try {
      const response = await this.apiClient.get('/connections');
      response.data.connections.forEach((conn: DatabaseConnection) => {
        this.connections.set(conn.id, conn);
      });
    } catch (error) {
      console.error('Failed to initialize database connections:', error);
    }
  }

  /**
   * Get all database connections status
   */
  async getDatabaseConnections(): Promise<DatabaseConnection[]> {
    try {
      const response = await this.apiClient.get('/connections');
      return response.data.connections;
    } catch (error) {
      console.error('Failed to get database connections:', error);
      throw error;
    }
  }

  /**
   * Test database connection
   */
  async testConnection(connectionId: string): Promise<{
    status: 'success' | 'error';
    responseTime: number;
    error?: string;
  }> {
    try {
      const response = await this.apiClient.post(`/connections/${connectionId}/test`);
      return response.data;
    } catch (error) {
      console.error(`Failed to test connection ${connectionId}:`, error);
      throw error;
    }
  }

  /**
   * Query seismic stations from PostgreSQL+PostGIS
   */
  async querySeismicStations(query: GeospatialQuery & {
    networks?: string[];
    channels?: string[];
    active?: boolean;
  }): Promise<SeismicStationRecord[]> {
    try {
      const response = await this.apiClient.post('/postgresql/query', {
        table: 'seismic_stations',
        query: {
          spatial: query.bounds || query.center || query.polygon,
          filters: {
            networks: query.networks,
            channels: query.channels,
            active: query.active,
          },
        },
      });

      return response.data.results.map((row: any) => ({
        id: row.id,
        network: row.network,
        station: row.station,
        location: row.location,
        channel: row.channel,
        latitude: row.latitude,
        longitude: row.longitude,
        elevation: row.elevation,
        start_date: new Date(row.start_date),
        end_date: row.end_date ? new Date(row.end_date) : undefined,
        sample_rate: row.sample_rate,
        instrument_type: row.instrument_type,
        data_quality: row.data_quality,
        created_at: new Date(row.created_at),
        updated_at: new Date(row.updated_at),
      }));
    } catch (error) {
      console.error('Failed to query seismic stations:', error);
      throw error;
    }
  }

  /**
   * Query volcanic systems from PostgreSQL+PostGIS
   */
  async queryVolcanicSystems(query: GeospatialQuery & {
    alertLevels?: string[];
    volcanoTypes?: string[];
    tectonicSettings?: string[];
    monitoringStatus?: string[];
  }): Promise<VolcanicSystemRecord[]> {
    try {
      const response = await this.apiClient.post('/postgresql/query', {
        table: 'volcanic_systems',
        query: {
          spatial: query.bounds || query.center || query.polygon,
          filters: {
            alert_levels: query.alertLevels,
            volcano_types: query.volcanoTypes,
            tectonic_settings: query.tectonicSettings,
            monitoring_status: query.monitoringStatus,
          },
        },
      });

      return response.data.results.map((row: any) => ({
        id: row.id,
        name: row.name,
        latitude: row.latitude,
        longitude: row.longitude,
        elevation: row.elevation,
        volcano_type: row.volcano_type,
        tectonic_setting: row.tectonic_setting,
        last_eruption: row.last_eruption ? new Date(row.last_eruption) : undefined,
        alert_level: row.alert_level,
        monitoring_status: row.monitoring_status,
        population_5km: row.population_5km,
        population_10km: row.population_10km,
        population_30km: row.population_30km,
        population_100km: row.population_100km,
        hazards: row.hazards || [],
        observatories: row.observatories || [],
        created_at: new Date(row.created_at),
        updated_at: new Date(row.updated_at),
      }));
    } catch (error) {
      console.error('Failed to query volcanic systems:', error);
      throw error;
    }
  }

  /**
   * Query causal nodes from Neo4j graph database
   */
  async queryCausalNodes(filters: {
    processTypes?: string[];
    spatialScales?: string[];
    temporalScales?: string[];
    confidenceThreshold?: number;
    dataSources?: string[];
  }): Promise<CausalNodeRecord[]> {
    try {
      const response = await this.apiClient.post('/neo4j/query', {
        cypher: `
          MATCH (n:CausalNode)
          WHERE ($processTypes IS NULL OR n.process_type IN $processTypes)
            AND ($spatialScales IS NULL OR n.spatial_scale IN $spatialScales)
            AND ($temporalScales IS NULL OR n.temporal_scale IN $temporalScales)
            AND ($confidenceThreshold IS NULL OR n.confidence >= $confidenceThreshold)
            AND ($dataSources IS NULL OR n.data_source IN $dataSources)
          RETURN n
          ORDER BY n.confidence DESC
        `,
        parameters: {
          processTypes: filters.processTypes,
          spatialScales: filters.spatialScales,
          temporalScales: filters.temporalScales,
          confidenceThreshold: filters.confidenceThreshold,
          dataSources: filters.dataSources,
        },
      });

      return response.data.results.map((record: any) => {
        const node = record.n.properties;
        return {
          id: node.id,
          label: node.label,
          process_type: node.process_type,
          spatial_scale: node.spatial_scale,
          temporal_scale: node.temporal_scale,
          data_source: node.data_source,
          metadata: node.metadata || {},
          size: node.size,
          color: node.color,
          confidence: node.confidence,
          latitude: node.latitude,
          longitude: node.longitude,
          created_at: new Date(node.created_at),
          updated_at: new Date(node.updated_at),
        };
      });
    } catch (error) {
      console.error('Failed to query causal nodes:', error);
      throw error;
    }
  }

  /**
   * Query causal relationships from Neo4j graph database
   */
  async queryCausalRelationships(filters: {
    sourceNodeIds?: string[];
    targetNodeIds?: string[];
    relationshipTypes?: string[];
    strengthThreshold?: number;
    confidenceThreshold?: number;
    validationStatus?: string[];
  }): Promise<CausalEdgeRecord[]> {
    try {
      const response = await this.apiClient.post('/neo4j/query', {
        cypher: `
          MATCH (source:CausalNode)-[r:CAUSES]->(target:CausalNode)
          WHERE ($sourceNodeIds IS NULL OR source.id IN $sourceNodeIds)
            AND ($targetNodeIds IS NULL OR target.id IN $targetNodeIds)
            AND ($relationshipTypes IS NULL OR r.relationship_type IN $relationshipTypes)
            AND ($strengthThreshold IS NULL OR r.strength >= $strengthThreshold)
            AND ($confidenceThreshold IS NULL OR (r.confidence_lower + r.confidence_upper) / 2 >= $confidenceThreshold)
            AND ($validationStatus IS NULL OR r.validation_status IN $validationStatus)
          RETURN r, source.id as source_id, target.id as target_id
          ORDER BY r.strength DESC
        `,
        parameters: {
          sourceNodeIds: filters.sourceNodeIds,
          targetNodeIds: filters.targetNodeIds,
          relationshipTypes: filters.relationshipTypes,
          strengthThreshold: filters.strengthThreshold,
          confidenceThreshold: filters.confidenceThreshold,
          validationStatus: filters.validationStatus,
        },
      });

      return response.data.results.map((record: any) => {
        const rel = record.r.properties;
        return {
          id: rel.id,
          source_node_id: record.source_id,
          target_node_id: record.target_id,
          relationship_type: rel.relationship_type,
          strength: rel.strength,
          confidence_lower: rel.confidence_lower,
          confidence_upper: rel.confidence_upper,
          temporal_lag: rel.temporal_lag,
          statistical_significance: rel.statistical_significance,
          discovery_method: rel.discovery_method,
          validation_status: rel.validation_status,
          evidence: rel.evidence || {},
          created_at: new Date(rel.created_at),
          updated_at: new Date(rel.updated_at),
        };
      });
    } catch (error) {
      console.error('Failed to query causal relationships:', error);
      throw error;
    }
  }

  /**
   * Query time series data from InfluxDB
   */
  async queryTimeSeriesData(params: {
    measurement: string;
    fields: string[];
    tags?: Record<string, string>;
    timeRange: {
      start: Date;
      end: Date;
    };
    aggregation?: {
      window: string; // e.g., '1h', '1d'
      function: 'mean' | 'max' | 'min' | 'sum' | 'count';
    };
    limit?: number;
  }): Promise<TimeSeriesData[]> {
    try {
      const response = await this.apiClient.post('/influxdb/query', {
        measurement: params.measurement,
        fields: params.fields,
        tags: params.tags,
        timeRange: {
          start: params.timeRange.start.toISOString(),
          end: params.timeRange.end.toISOString(),
        },
        aggregation: params.aggregation,
        limit: params.limit,
      });

      return response.data.results.map((point: any) => ({
        timestamp: new Date(point.time),
        value: point.value,
        quality: point.quality || 1.0,
        source: point.source || params.measurement,
        metadata: point.metadata,
      }));
    } catch (error) {
      console.error('Failed to query time series data:', error);
      throw error;
    }
  }

  /**
   * Query cached data from Redis
   */
  async queryCachedData(key: string): Promise<any> {
    try {
      const response = await this.apiClient.get(`/redis/get/${encodeURIComponent(key)}`);
      return response.data.value;
    } catch (error) {
      console.error(`Failed to query cached data for key ${key}:`, error);
      throw error;
    }
  }

  /**
   * Set cached data in Redis
   */
  async setCachedData(key: string, value: any, ttl?: number): Promise<void> {
    try {
      await this.apiClient.post('/redis/set', {
        key,
        value,
        ttl: ttl || 3600, // Default 1 hour TTL
      });
    } catch (error) {
      console.error(`Failed to set cached data for key ${key}:`, error);
      throw error;
    }
  }

  /**
   * Execute complex geospatial query across multiple databases
   */
  async executeGeospatialQuery(query: {
    spatial: GeospatialQuery;
    temporal?: {
      start: Date;
      end: Date;
    };
    dataTypes: Array<'seismic' | 'volcanic' | 'deformation' | 'thermal' | 'geochemical'>;
    aggregation?: {
      spatial: 'grid' | 'cluster' | 'none';
      temporal: 'hourly' | 'daily' | 'weekly' | 'monthly' | 'none';
    };
  }): Promise<{
    seismic?: SeismicStationRecord[];
    volcanic?: VolcanicSystemRecord[];
    causal?: {
      nodes: CausalNodeRecord[];
      edges: CausalEdgeRecord[];
    };
    timeSeries?: Record<string, TimeSeriesData[]>;
  }> {
    try {
      const response = await this.apiClient.post('/query/geospatial', query);
      return response.data;
    } catch (error) {
      console.error('Failed to execute geospatial query:', error);
      throw error;
    }
  }

  /**
   * Insert new causal node into Neo4j
   */
  async insertCausalNode(node: Omit<CausalNodeRecord, 'created_at' | 'updated_at'>): Promise<string> {
    try {
      const response = await this.apiClient.post('/neo4j/nodes', {
        label: 'CausalNode',
        properties: node,
      });
      return response.data.id;
    } catch (error) {
      console.error('Failed to insert causal node:', error);
      throw error;
    }
  }

  /**
   * Insert new causal relationship into Neo4j
   */
  async insertCausalRelationship(edge: Omit<CausalEdgeRecord, 'created_at' | 'updated_at'>): Promise<string> {
    try {
      const response = await this.apiClient.post('/neo4j/relationships', {
        type: 'CAUSES',
        sourceNodeId: edge.source_node_id,
        targetNodeId: edge.target_node_id,
        properties: {
          id: edge.id,
          relationship_type: edge.relationship_type,
          strength: edge.strength,
          confidence_lower: edge.confidence_lower,
          confidence_upper: edge.confidence_upper,
          temporal_lag: edge.temporal_lag,
          statistical_significance: edge.statistical_significance,
          discovery_method: edge.discovery_method,
          validation_status: edge.validation_status,
          evidence: edge.evidence,
        },
      });
      return response.data.id;
    } catch (error) {
      console.error('Failed to insert causal relationship:', error);
      throw error;
    }
  }

  /**
   * Insert time series data into InfluxDB
   */
  async insertTimeSeriesData(data: {
    measurement: string;
    tags: Record<string, string>;
    fields: Record<string, number>;
    timestamp?: Date;
  }[]): Promise<void> {
    try {
      await this.apiClient.post('/influxdb/write', {
        points: data.map(point => ({
          measurement: point.measurement,
          tags: point.tags,
          fields: point.fields,
          timestamp: (point.timestamp || new Date()).toISOString(),
        })),
      });
    } catch (error) {
      console.error('Failed to insert time series data:', error);
      throw error;
    }
  }

  /**
   * Get database performance metrics
   */
  async getDatabaseMetrics(): Promise<{
    connections: Array<{
      id: string;
      type: string;
      status: string;
      responseTime: number;
      activeConnections: number;
      queryCount: number;
      errorRate: number;
    }>;
    performance: {
      totalQueries: number;
      averageResponseTime: number;
      errorRate: number;
      cacheHitRate: number;
    };
  }> {
    try {
      const response = await this.apiClient.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Failed to get database metrics:', error);
      throw error;
    }
  }

  /**
   * Execute database health check
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    databases: Array<{
      type: string;
      status: 'up' | 'down' | 'slow';
      responseTime: number;
      lastCheck: Date;
    }>;
    issues: string[];
  }> {
    try {
      const response = await this.apiClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Failed to perform health check:', error);
      throw error;
    }
  }
}

export default DatabaseService;