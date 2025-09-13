/**
 * Real-Time Process Dashboard Component
 * 
 * Advanced real-time monitoring dashboard for geological processes with causal context,
 * streaming data visualization, alert management, and anomaly detection display.
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Alert,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Badge,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider
} from '@mui/material';
import {
  Timeline,
  Warning,
  Error,
  Info,
  PlayArrow,
  Pause,
  Refresh,
  Settings,
  TrendingUp,
  TrendingDown,
  NetworkCheck,
  Speed,
  Memory,
  DataUsage
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  Cell,
  AreaChart,
  Area,
  ComposedChart,
  Bar
} from 'recharts';
import { format, subMinutes, isAfter } from 'date-fns';

// Types for real-time data
interface GeologicalDataPoint {
  timestamp: string;
  process_type: 'seismic' | 'volcanic' | 'deformation' | 'geochemical' | 'thermal' | 'electromagnetic';
  station_id: string;
  location: [number, number, number];
  parameters: Record<string, number>;
  quality_score: number;
  uncertainty: Record<string, number>;
  source_id: string;
  processing_latency: number;
}

interface CausalRelationship {
  source_process: string;
  target_process: string;
  causal_strength: number;
  confidence_interval: [number, number];
  temporal_lag: number;
  spatial_correlation: number;
  last_updated: string;
  validation_score: number;
}

interface RealTimeAlert {
  alert_id: string;
  timestamp: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  process_type: string;
  station_id: string;
  location: [number, number, number];
  message: string;
  causal_context: CausalRelationship[];
  confidence: number;
  predicted_evolution: Record<string, any>;
  recommended_actions: string[];
  data_quality_flags: string[];
}

interface AnomalyDetection {
  anomaly_id: string;
  timestamp: string;
  anomaly_type: 'statistical' | 'causal' | 'temporal' | 'spatial' | 'multivariate';
  process_type: string;
  station_id: string;
  anomaly_score: number;
  statistical_significance: number;
  causal_pathway: string[];
  affected_processes: string[];
  temporal_pattern: Record<string, any>;
  spatial_extent: Record<string, any>;
}

interface PerformanceMetrics {
  avg_processing_time_ms: number;
  max_processing_time_ms: number;
  p95_processing_time_ms: number;
  throughput_points_per_sec: number;
  active_processes: number;
  active_buffers: number;
  total_data_points: number;
  causal_relationships: number;
  websocket_connections: number;
  alert_history_size: number;
}

interface MonitoringState {
  isConnected: boolean;
  isMonitoring: boolean;
  dataPoints: GeologicalDataPoint[];
  alerts: RealTimeAlert[];
  anomalies: AnomalyDetection[];
  causalRelationships: CausalRelationship[];
  performanceMetrics: PerformanceMetrics;
  activeProcesses: string[];
}

const SEVERITY_COLORS = {
  low: '#4caf50',
  medium: '#ff9800',
  high: '#f44336',
  critical: '#9c27b0'
};

const PROCESS_TYPE_COLORS = {
  seismic: '#2196f3',
  volcanic: '#f44336',
  deformation: '#4caf50',
  geochemical: '#ff9800',
  thermal: '#e91e63',
  electromagnetic: '#9c27b0'
};

export const RealTimeProcessDashboard: React.FC = () => {
  const [state, setState] = useState<MonitoringState>({
    isConnected: false,
    isMonitoring: false,
    dataPoints: [],
    alerts: [],
    anomalies: [],
    causalRelationships: [],
    performanceMetrics: {
      avg_processing_time_ms: 0,
      max_processing_time_ms: 0,
      p95_processing_time_ms: 0,
      throughput_points_per_sec: 0,
      active_processes: 0,
      active_buffers: 0,
      total_data_points: 0,
      causal_relationships: 0,
      websocket_connections: 0,
      alert_history_size: 0
    },
    activeProcesses: []
  });

  const [selectedProcess, setSelectedProcess] = useState<string>('all');
  const [timeWindow, setTimeWindow] = useState<number>(300); // 5 minutes in seconds
  const [showAnomalies, setShowAnomalies] = useState<boolean>(true);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    try {
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8765';
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setState(prev => ({ ...prev, isConnected: true }));
        
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setState(prev => ({ ...prev, isConnected: false }));
        
        // Attempt to reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket();
        }, 5000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

    } catch (error) {
      console.error('Error connecting WebSocket:', error);
    }
  }, []);

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'initial_state':
        setState(prev => ({
          ...prev,
          activeProcesses: message.data.active_processes || [],
          performanceMetrics: message.data.performance_metrics || prev.performanceMetrics
        }));
        break;

      case 'data_update':
        const dataPoint: GeologicalDataPoint = message.data;
        setState(prev => {
          const newDataPoints = [...prev.dataPoints, dataPoint];
          
          // Keep only recent data points within time window
          const cutoffTime = new Date(Date.now() - timeWindow * 1000);
          const filteredPoints = newDataPoints.filter(dp => 
            isAfter(new Date(dp.timestamp), cutoffTime)
          );

          return {
            ...prev,
            dataPoints: filteredPoints.slice(-1000) // Keep max 1000 points
          };
        });
        break;

      case 'alert':
        const alert: RealTimeAlert = message.data;
        setState(prev => ({
          ...prev,
          alerts: [alert, ...prev.alerts].slice(0, 100) // Keep max 100 alerts
        }));
        break;

      case 'anomaly':
        const anomaly: AnomalyDetection = message.data;
        setState(prev => ({
          ...prev,
          anomalies: [anomaly, ...prev.anomalies].slice(0, 100) // Keep max 100 anomalies
        }));
        break;

      case 'causal_update':
        setState(prev => ({
          ...prev,
          causalRelationships: message.data.relationships || []
        }));
        break;

      case 'performance_metrics':
        setState(prev => ({
          ...prev,
          performanceMetrics: message.data
        }));
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }, [timeWindow]);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connectWebSocket]);

  // Filter data based on selected process and time window
  const filteredData = useMemo(() => {
    const cutoffTime = new Date(Date.now() - timeWindow * 1000);
    
    return state.dataPoints.filter(dp => {
      const isInTimeWindow = isAfter(new Date(dp.timestamp), cutoffTime);
      const matchesProcess = selectedProcess === 'all' || 
        dp.process_type === selectedProcess ||
        `${dp.process_type}_${dp.station_id}` === selectedProcess;
      
      return isInTimeWindow && matchesProcess;
    });
  }, [state.dataPoints, selectedProcess, timeWindow]);

  // Prepare chart data
  const chartData = useMemo(() => {
    const data: Record<string, any>[] = [];
    
    filteredData.forEach(dp => {
      const timestamp = new Date(dp.timestamp).getTime();
      
      // Create data point for each parameter
      Object.entries(dp.parameters).forEach(([param, value]) => {
        data.push({
          timestamp,
          time: format(new Date(timestamp), 'HH:mm:ss'),
          process: `${dp.process_type}_${dp.station_id}`,
          parameter: param,
          value,
          quality: dp.quality_score,
          anomaly: state.anomalies.some(a => 
            a.station_id === dp.station_id && 
            Math.abs(new Date(a.timestamp).getTime() - timestamp) < 30000 // Within 30 seconds
          )
        });
      });
    });

    return data.sort((a, b) => a.timestamp - b.timestamp);
  }, [filteredData, state.anomalies]);

  // Get recent alerts
  const recentAlerts = useMemo(() => {
    const cutoffTime = new Date(Date.now() - timeWindow * 1000);
    return state.alerts.filter(alert => 
      isAfter(new Date(alert.timestamp), cutoffTime)
    );
  }, [state.alerts, timeWindow]);

  // Get recent anomalies
  const recentAnomalies = useMemo(() => {
    const cutoffTime = new Date(Date.now() - timeWindow * 1000);
    return state.anomalies.filter(anomaly => 
      isAfter(new Date(anomaly.timestamp), cutoffTime)
    );
  }, [state.anomalies, timeWindow]);

  // Toggle monitoring
  const toggleMonitoring = useCallback(() => {
    setState(prev => ({ ...prev, isMonitoring: !prev.isMonitoring }));
  }, []);

  // Clear data
  const clearData = useCallback(() => {
    setState(prev => ({
      ...prev,
      dataPoints: [],
      alerts: [],
      anomalies: []
    }));
  }, []);

  return (
    <Box sx={{ p: 3, bgcolor: '#f5f5f5', minHeight: '100vh' }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h4" component="h1" sx={{ fontWeight: 'bold' }}>
          Real-Time Geological Process Monitor
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            icon={<NetworkCheck />}
            label={state.isConnected ? 'Connected' : 'Disconnected'}
            color={state.isConnected ? 'success' : 'error'}
            variant="outlined"
          />
          
          <IconButton
            onClick={toggleMonitoring}
            color={state.isMonitoring ? 'secondary' : 'primary'}
            size="large"
          >
            {state.isMonitoring ? <Pause /> : <PlayArrow />}
          </IconButton>
          
          <IconButton onClick={clearData} color="warning">
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Process</InputLabel>
              <Select
                value={selectedProcess}
                onChange={(e) => setSelectedProcess(e.target.value)}
                label="Process"
              >
                <MenuItem value="all">All Processes</MenuItem>
                {state.activeProcesses.map(process => (
                  <MenuItem key={process} value={process}>
                    {process.replace('_', ' - ')}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Time Window</InputLabel>
              <Select
                value={timeWindow}
                onChange={(e) => setTimeWindow(Number(e.target.value))}
                label="Time Window"
              >
                <MenuItem value={60}>1 Minute</MenuItem>
                <MenuItem value={300}>5 Minutes</MenuItem>
                <MenuItem value={900}>15 Minutes</MenuItem>
                <MenuItem value={1800}>30 Minutes</MenuItem>
                <MenuItem value={3600}>1 Hour</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <FormControlLabel
              control={
                <Switch
                  checked={showAnomalies}
                  onChange={(e) => setShowAnomalies(e.target.checked)}
                />
              }
              label="Show Anomalies"
            />
          </Grid>
          
          <Grid item xs={12} sm={6} md={3}>
            <FormControlLabel
              control={
                <Switch
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                />
              }
              label="Auto Scroll"
            />
          </Grid>
        </Grid>
      </Paper>

      <Grid container spacing={3}>
        {/* Performance Metrics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Performance
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Speed color="primary" sx={{ fontSize: 40, mb: 1 }} />
                    <Typography variant="h6">
                      {state.performanceMetrics.throughput_points_per_sec.toFixed(1)}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Points/sec
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Memory color="secondary" sx={{ fontSize: 40, mb: 1 }} />
                    <Typography variant="h6">
                      {state.performanceMetrics.avg_processing_time_ms.toFixed(1)}ms
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Avg Latency
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <DataUsage color="success" sx={{ fontSize: 40, mb: 1 }} />
                    <Typography variant="h6">
                      {state.performanceMetrics.active_processes}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Active Processes
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={6} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Timeline color="info" sx={{ fontSize: 40, mb: 1 }} />
                    <Typography variant="h6">
                      {state.performanceMetrics.causal_relationships}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      Causal Links
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Real-time Data Visualization */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ height: 500 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Real-Time Data Streams
              </Typography>
              
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="time"
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis 
                    tick={{ fontSize: 12 }}
                  />
                  <RechartsTooltip 
                    labelFormatter={(value) => `Time: ${value}`}
                    formatter={(value: any, name: string) => [
                      typeof value === 'number' ? value.toFixed(3) : value,
                      name
                    ]}
                  />
                  
                  {/* Data lines for different parameters */}
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#2196f3"
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                  />
                  
                  {/* Anomaly markers */}
                  {showAnomalies && (
                    <Scatter
                      dataKey="anomaly"
                      fill="#f44336"
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Alerts Panel */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: 500 }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  Active Alerts
                </Typography>
                <Badge badgeContent={recentAlerts.length} color="error">
                  <Warning />
                </Badge>
              </Box>
              
              <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                <List dense>
                  {recentAlerts.map((alert, index) => (
                    <React.Fragment key={alert.alert_id}>
                      <ListItem>
                        <ListItemIcon>
                          {alert.severity === 'critical' && <Error color="error" />}
                          {alert.severity === 'high' && <Warning color="warning" />}
                          {alert.severity === 'medium' && <Info color="info" />}
                          {alert.severity === 'low' && <Info color="success" />}
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Chip
                                label={alert.severity}
                                size="small"
                                sx={{ 
                                  bgcolor: SEVERITY_COLORS[alert.severity],
                                  color: 'white',
                                  fontSize: '0.7rem'
                                }}
                              />
                              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                {alert.station_id}
                              </Typography>
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="caption" display="block">
                                {alert.message}
                              </Typography>
                              <Typography variant="caption" color="textSecondary">
                                {format(new Date(alert.timestamp), 'HH:mm:ss')} - 
                                Confidence: {(alert.confidence * 100).toFixed(1)}%
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < recentAlerts.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                  
                  {recentAlerts.length === 0 && (
                    <ListItem>
                      <ListItemText
                        primary="No active alerts"
                        secondary="System operating normally"
                      />
                    </ListItem>
                  )}
                </List>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Anomaly Detection */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Anomaly Detection
              </Typography>
              
              <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                <List dense>
                  {recentAnomalies.map((anomaly, index) => (
                    <React.Fragment key={anomaly.anomaly_id}>
                      <ListItem>
                        <ListItemIcon>
                          <Chip
                            label={anomaly.anomaly_type}
                            size="small"
                            color="warning"
                            variant="outlined"
                          />
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Typography variant="body2">
                              {anomaly.station_id} - Score: {anomaly.anomaly_score.toFixed(3)}
                            </Typography>
                          }
                          secondary={
                            <Typography variant="caption" color="textSecondary">
                              {format(new Date(anomaly.timestamp), 'HH:mm:ss')} - 
                              Significance: {anomaly.statistical_significance.toFixed(3)}
                            </Typography>
                          }
                        />
                      </ListItem>
                      {index < recentAnomalies.length - 1 && <Divider />}
                    </React.Fragment>
                  ))}
                  
                  {recentAnomalies.length === 0 && (
                    <ListItem>
                      <ListItemText
                        primary="No anomalies detected"
                        secondary="All processes within normal parameters"
                      />
                    </ListItem>
                  )}
                </List>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Causal Relationships */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Causal Relationships
              </Typography>
              
              <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
                <List dense>
                  {state.causalRelationships.slice(0, 10).map((relationship, index) => (
                    <React.Fragment key={`${relationship.source_process}-${relationship.target_process}`}>
                      <ListItem>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                {relationship.source_process}
                              </Typography>
                              <Typography variant="body2">→</Typography>
                              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                {relationship.target_process}
                              </Typography>
                            </Box>
                          }
                          secondary={
                            <Box>
                              <LinearProgress
                                variant="determinate"
                                value={relationship.causal_strength * 100}
                                sx={{ mb: 0.5 }}
                              />
                              <Typography variant="caption" color="textSecondary">
                                Strength: {relationship.causal_strength.toFixed(3)} - 
                                Lag: {relationship.temporal_lag}s
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                      {index < Math.min(state.causalRelationships.length - 1, 9) && <Divider />}
                    </React.Fragment>
                  ))}
                  
                  {state.causalRelationships.length === 0 && (
                    <ListItem>
                      <ListItemText
                        primary="No causal relationships detected"
                        secondary="Insufficient data for causal analysis"
                      />
                    </ListItem>
                  )}
                </List>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RealTimeProcessDashboard;