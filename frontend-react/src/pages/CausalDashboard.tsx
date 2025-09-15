import React, { useEffect, useState, useCallback } from 'react';
import { Box, Container, Typography, Alert, Skeleton } from '@mui/material';
// import { useTheme } from '@mui/material/styles'; // Unused
import { Helmet } from 'react-helmet-async';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '../store';
import { 
  setLoading, 
  setError, 
  addNotification,
  updatePerformanceMetrics,
  setRealTimeStatus,
} from '../store/slices/dashboardSlice';

import ComprehensiveCausalDashboard from '../components/dashboard/ComprehensiveCausalDashboard';
import { 
  CausalGraph, 
  CausalNode, 
  CausalEdge, 
  GeologicalProcessType, 
  SpatialScale, 
  TemporalScale, 
  CausalDiscoveryMethod 
} from '../types/causal';

// Mock data service - in production this would be replaced with real API calls
class CausalDataService {
  private static instance: CausalDataService;
  private websocket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  static getInstance(): CausalDataService {
    if (!CausalDataService.instance) {
      CausalDataService.instance = new CausalDataService();
    }
    return CausalDataService.instance;
  }

  async fetchCausalGraph(): Promise<CausalGraph> {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Generate realistic causal graph data
    const nodes: CausalNode[] = [
      {
        id: 'seismic-activity',
        label: 'Seismic Activity',
        processType: GeologicalProcessType.SEISMIC,
        spatialScale: SpatialScale.REGIONAL,
        temporalScale: TemporalScale.SECONDS,
        dataSource: 'IRIS/EarthScope',
        metadata: {
          description: 'Real-time seismic monitoring from IRIS network',
          lastUpdated: new Date(),
          magnitude: 4.2,
          depth: 15.5,
        },
        size: 20,
        color: '#FF6B6B',
        confidence: 0.95,
        location: { latitude: 40.7128, longitude: -74.0060 },
      },
      {
        id: 'ground-deformation',
        label: 'Ground Deformation',
        processType: GeologicalProcessType.DEFORMATION,
        size: 18,
        color: '#4ECDC4',
        metadata: {
          description: 'MT-SAR interferometry from Sentinel-1',
          dataSource: 'ASF-Hyp3/Copernicus',
          lastUpdated: new Date(),
          confidence: 0.88,
          spatialScale: 'local',
          temporalScale: 'day',
        },
        properties: {
          displacement: 2.3,
          coherence: 0.85,
          velocity: 0.5,
        },
      },
      {
        id: 'gas-emissions',
        label: 'Gas Emissions',
        processType: GeologicalProcessType.GEOCHEMICAL,
        size: 16,
        color: '#45B7D1',
        metadata: {
          description: 'Volcanic gas composition from observatory',
          dataSource: 'USGS Volcano Hazards Program',
          lastUpdated: new Date(),
          confidence: 0.82,
          spatialScale: 'local',
          temporalScale: 'hour',
          so2_flux: 150,
          co2_ratio: 0.25,
          temperature: 85.2,
        },
      },
      {
        id: 'thermal-anomalies',
        label: 'Thermal Anomalies',
        processType: GeologicalProcessType.THERMAL,
        size: 14,
        color: '#F7DC6F',
        metadata: {
          description: 'Thermal infrared from NASA FIRMS',
          dataSource: 'NASA EarthData/MODIS',
          lastUpdated: new Date(),
          confidence: 0.76,
          spatialScale: 'regional',
          temporalScale: 'day',
          temperature: 45.8,
          area: 1250,
          intensity: 0.65,
        },
      },
      {
        id: 'magma-intrusion',
        label: 'Magma Intrusion',
        processType: GeologicalProcessType.MAGMATIC,
        size: 22,
        color: '#E74C3C',
        metadata: {
          description: 'Inferred from multi-parameter analysis',
          dataSource: 'Integrated Analysis',
          lastUpdated: new Date(),
          confidence: 0.71,
          spatialScale: 'local',
          temporalScale: 'day',
        },
        properties: {
          volume: 0.05,
          depth: 8.2,
          pressure: 150,
        },
      },
    ];

    const edges: CausalEdge[] = [
      {
        id: 'magma-seismic',
        source: 'magma-intrusion',
        target: 'seismic-activity',
        causalStrength: 0.85,
        confidence: [0.78, 0.92],
        temporalLag: 3600000, // 1 hour in milliseconds
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'Granger Causality',
          statisticalSignificance: 0.001,
          validationStatus: 'validated',
          geologicalContext: 'Magma intrusion causes stress changes leading to seismic activity',
        },
        properties: {
          correlation: 0.82,
          pValue: 0.001,
        },
      },
      {
        id: 'magma-deformation',
        source: 'magma-intrusion',
        target: 'ground-deformation',
        causalStrength: 0.92,
        confidence: [0.87, 0.97],
        temporalLag: 7200000, // 2 hours
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'Cross-correlation Analysis',
          statisticalSignificance: 0.0001,
          validationStatus: 'validated',
          geologicalContext: 'Magma intrusion causes surface deformation',
        },
        properties: {
          correlation: 0.89,
          pValue: 0.0001,
        },
      },
      {
        id: 'seismic-gas',
        source: 'seismic-activity',
        target: 'gas-emissions',
        causalStrength: 0.67,
        confidence: [0.58, 0.76],
        temporalLag: 1800000, // 30 minutes
        edgeType: 'indirect',
        metadata: {
          discoveryMethod: 'Transfer Entropy',
          statisticalSignificance: 0.01,
          validationStatus: 'pending',
          geologicalContext: 'Seismic activity may enhance gas pathways',
        },
        properties: {
          correlation: 0.64,
          pValue: 0.01,
        },
      },
      {
        id: 'gas-thermal',
        source: 'gas-emissions',
        target: 'thermal-anomalies',
        causalStrength: 0.73,
        confidence: [0.65, 0.81],
        temporalLag: 10800000, // 3 hours
        edgeType: 'direct',
        metadata: {
          discoveryMethod: 'Causal Forest',
          statisticalSignificance: 0.005,
          validationStatus: 'validated',
          geologicalContext: 'Gas emissions indicate thermal processes',
        },
        properties: {
          correlation: 0.71,
          pValue: 0.005,
        },
      },
    ];

    return {
      id: 'mock-causal-graph',
      nodes,
      edges,
      metadata: {
        algorithmsUsed: [CausalDiscoveryMethod.PC, CausalDiscoveryMethod.GES, CausalDiscoveryMethod.LINGAM],
        confidenceThreshold: 0.8,
        totalNodes: nodes.length,
        totalEdges: edges.length,
        graphDensity: edges.length / (nodes.length * (nodes.length - 1)),
        dataSourcesUsed: ['IRIS/EarthScope', 'USGS', 'NASA'],
        processingTime: 1500
      },
      createdAt: new Date(),
      updatedAt: new Date()
    };
  }

  connectRealTimeStream(onMessage: (data: any) => void, onStatusChange: (status: 'connected' | 'disconnected' | 'error') => void): void {
    try {
      // In production, this would connect to a real WebSocket endpoint
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8080/causal-stream';
      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        console.log('Real-time causal stream connected');
        onStatusChange('connected');
        this.reconnectAttempts = 0;
      };

      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.websocket.onclose = () => {
        console.log('Real-time causal stream disconnected');
        onStatusChange('disconnected');
        this.attemptReconnect(onMessage, onStatusChange);
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        onStatusChange('error');
      };
    } catch (error) {
      console.error('Failed to connect to real-time stream:', error);
      onStatusChange('error');
      
      // Simulate real-time updates for demo purposes
      this.simulateRealTimeUpdates(onMessage, onStatusChange);
    }
  }

  private simulateRealTimeUpdates(onMessage: (data: any) => void, onStatusChange: (status: 'connected' | 'disconnected' | 'error') => void): void {
    // Simulate connection for demo
    setTimeout(() => onStatusChange('connected'), 1000);
    
    // Send periodic updates
    setInterval(() => {
      const updateTypes = ['node_update', 'edge_update', 'new_relationship', 'alert'];
      const updateType = updateTypes[Math.floor(Math.random() * updateTypes.length)];
      
      const simulatedData = {
        type: updateType,
        timestamp: new Date().toISOString(),
        data: this.generateSimulatedUpdate(updateType),
      };
      
      onMessage(simulatedData);
    }, 5000); // Update every 5 seconds
  }

  private generateSimulatedUpdate(type: string): any {
    switch (type) {
      case 'node_update':
        return {
          nodeId: 'seismic-activity',
          properties: {
            magnitude: 3.5 + Math.random() * 2,
            depth: 10 + Math.random() * 20,
          },
          confidence: 0.8 + Math.random() * 0.2,
        };
      case 'edge_update':
        return {
          edgeId: 'magma-seismic',
          strength: 0.7 + Math.random() * 0.3,
          confidence: [0.6 + Math.random() * 0.2, 0.8 + Math.random() * 0.2],
        };
      case 'new_relationship':
        return {
          source: 'thermal-anomalies',
          target: 'ground-deformation',
          strength: 0.5 + Math.random() * 0.4,
          discoveryMethod: 'Real-time Analysis',
        };
      case 'alert':
        return {
          severity: Math.random() > 0.7 ? 'warning' : 'info',
          title: 'Causal Relationship Change',
          message: 'Significant change detected in magma-seismic relationship strength',
          threshold: 0.1,
          currentValue: 0.85 + (Math.random() - 0.5) * 0.2,
        };
      default:
        return {};
    }
  }

  private attemptReconnect(onMessage: (data: any) => void, onStatusChange: (status: 'connected' | 'disconnected' | 'error') => void): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        this.connectRealTimeStream(onMessage, onStatusChange);
      }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)); // Exponential backoff
    } else {
      console.error('Max reconnection attempts reached');
      onStatusChange('error');
    }
  }

  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }
}

const CausalDashboardPage: React.FC = () => {
  // const theme = useTheme(); // Unused
  const dispatch = useDispatch();
  const { isLoading, error } = useSelector((state: RootState) => state.dashboard);
  
  const [causalGraph, setCausalGraph] = useState<CausalGraph | null>(null);
  const [dataService] = useState(() => CausalDataService.getInstance());

  // Load initial causal graph data
  const loadCausalGraph = useCallback(async () => {
    try {
      dispatch(setLoading(true));
      dispatch(setError(null));
      
      const graph = await dataService.fetchCausalGraph();
      setCausalGraph(graph);
      
      dispatch(updatePerformanceMetrics({
        nodeCount: graph.nodes.length,
        edgeCount: graph.edges.length,
      }));
      
      dispatch(addNotification({
        id: `graph-loaded-${Date.now()}`,
        type: 'success',
        title: 'Causal Graph Loaded',
        message: `Successfully loaded ${graph.nodes.length} nodes and ${graph.edges.length} relationships`,
        timestamp: new Date(),
        autoHide: true,
        duration: 3000,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to load causal graph';
      dispatch(setError(errorMessage));
      dispatch(addNotification({
        id: `graph-error-${Date.now()}`,
        type: 'error',
        title: 'Loading Failed',
        message: errorMessage,
        timestamp: new Date(),
        autoHide: false,
      }));
    } finally {
      dispatch(setLoading(false));
    }
  }, [dataService, dispatch]);

  // Handle real-time updates
  const handleRealTimeMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'node_update':
        setCausalGraph(prev => {
          if (!prev) return prev;
          const updatedNodes = prev.nodes.map(node => 
            node.id === data.data.nodeId 
              ? { ...node, properties: { ...node.properties, ...data.data.properties } }
              : node
          );
          return { ...prev, nodes: updatedNodes };
        });
        break;
      
      case 'edge_update':
        setCausalGraph(prev => {
          if (!prev) return prev;
          const updatedEdges = prev.edges.map(edge => 
            edge.id === data.data.edgeId 
              ? { ...edge, strength: data.data.strength, confidence: data.data.confidence }
              : edge
          );
          return { ...prev, edges: updatedEdges };
        });
        break;
      
      case 'alert':
        dispatch(addNotification({
          id: `realtime-alert-${Date.now()}`,
          type: data.data.severity,
          title: data.data.title,
          message: data.data.message,
          timestamp: new Date(),
          autoHide: data.data.severity !== 'error',
          duration: data.data.severity === 'error' ? undefined : 5000,
        }));
        break;
    }
  }, [dispatch]);

  const handleRealTimeStatusChange = useCallback((status: 'connected' | 'disconnected' | 'error') => {
    dispatch(setRealTimeStatus(status));
  }, [dispatch]);

  // Handle export functionality
  const handleExport = useCallback((format: string, data: any) => {
    try {
      let exportData: string;
      let filename: string;
      let mimeType: string;

      switch (format) {
        case 'json':
          exportData = JSON.stringify(data, null, 2);
          filename = `causal-dashboard-${new Date().toISOString().split('T')[0]}.json`;
          mimeType = 'application/json';
          break;
        case 'csv':
          // Convert graph data to CSV format
          const csvRows = [
            'Source,Target,Strength,Confidence_Lower,Confidence_Upper,Temporal_Lag,Edge_Type,Discovery_Method',
            ...data.graph.edges.map((edge: CausalEdge) => 
              `${edge.source},${edge.target},${edge.strength},${edge.confidence[0]},${edge.confidence[1]},${edge.temporalLag || ''},${edge.edgeType},${edge.metadata.discoveryMethod}`
            )
          ];
          exportData = csvRows.join('\n');
          filename = `causal-relationships-${new Date().toISOString().split('T')[0]}.csv`;
          mimeType = 'text/csv';
          break;
        default:
          throw new Error(`Unsupported export format: ${format}`);
      }

      // Create and trigger download
      const blob = new Blob([exportData], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      dispatch(addNotification({
        id: `export-success-${Date.now()}`,
        type: 'success',
        title: 'Export Complete',
        message: `Dashboard data exported as ${format.toUpperCase()}`,
        timestamp: new Date(),
        autoHide: true,
        duration: 3000,
      }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Export failed';
      dispatch(addNotification({
        id: `export-error-${Date.now()}`,
        type: 'error',
        title: 'Export Failed',
        message: errorMessage,
        timestamp: new Date(),
        autoHide: false,
      }));
    }
  }, [dispatch]);

  // Handle share functionality
  const handleShare = useCallback((_shareConfig: any) => {
    // In production, this would integrate with sharing services
    const shareUrl = `${window.location.origin}/causal-dashboard/shared/${Date.now()}`;
    
    if (navigator.share) {
      navigator.share({
        title: 'Comprehensive Causal Dashboard',
        text: 'Check out this causal analysis dashboard',
        url: shareUrl,
      }).catch(console.error);
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(shareUrl).then(() => {
        dispatch(addNotification({
          id: `share-success-${Date.now()}`,
          type: 'success',
          title: 'Link Copied',
          message: 'Dashboard share link copied to clipboard',
          timestamp: new Date(),
          autoHide: true,
          duration: 3000,
        }));
      }).catch(() => {
        dispatch(addNotification({
          id: `share-error-${Date.now()}`,
          type: 'error',
          title: 'Share Failed',
          message: 'Failed to copy share link to clipboard',
          timestamp: new Date(),
          autoHide: false,
        }));
      });
    }
  }, [dispatch]);

  // Initialize dashboard
  useEffect(() => {
    loadCausalGraph();
    
    // Connect to real-time stream
    dataService.connectRealTimeStream(handleRealTimeMessage, handleRealTimeStatusChange);
    
    // Performance monitoring
    const performanceInterval = setInterval(() => {
      dispatch(updatePerformanceMetrics({
        fps: Math.floor(55 + Math.random() * 10), // Simulate FPS
        renderTime: Math.random() * 16.67, // Simulate render time
        memoryUsage: Math.floor(50 + Math.random() * 30), // Simulate memory usage
      }));
    }, 1000);

    return () => {
      dataService.disconnect();
      clearInterval(performanceInterval);
    };
  }, [loadCausalGraph, dataService, handleRealTimeMessage, handleRealTimeStatusChange, dispatch]);

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="h6">Failed to Load Causal Dashboard</Typography>
          <Typography variant="body2">{error}</Typography>
        </Alert>
      </Container>
    );
  }

  if (isLoading || !causalGraph) {
    return (
      <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        <Helmet>
          <title>Loading Causal Dashboard - Prometheus AI Platform</title>
        </Helmet>
        
        <Box sx={{ p: 2 }}>
          <Skeleton variant="rectangular" height={64} sx={{ mb: 2 }} />
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Skeleton variant="rectangular" width={280} height="calc(100vh - 120px)" />
            <Box sx={{ flex: 1 }}>
              <Skeleton variant="rectangular" height="calc(100vh - 120px)" />
            </Box>
          </Box>
        </Box>
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100vh', overflow: 'hidden' }}>
      <Helmet>
        <title>Comprehensive Causal Dashboard - Prometheus AI Platform</title>
        <meta name="description" content="Advanced causal relationship analysis and visualization dashboard for geological processes" />
      </Helmet>
      
      <ComprehensiveCausalDashboard
        initialGraph={causalGraph}
        realTimeEnabled={true}
        collaborativeMode={false}
        onExport={handleExport}
        onShare={handleShare}
      />
    </Box>
  );
};

export default CausalDashboardPage;