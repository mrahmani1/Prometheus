import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer
} from 'recharts';
import { AlertTriangle, Activity, TrendingUp, Zap } from 'lucide-react';

interface GeologicalProcess {
  id: string;
  name: string;
  type: 'seismic' | 'volcanic' | 'deformation' | 'geochemical' | 'thermal';
  location: { lat: number; lon: number; depth?: number };
  currentValue: number;
  unit: string;
  trend: 'increasing' | 'decreasing' | 'stable' | 'volatile';
  alertLevel: 'normal' | 'elevated' | 'high' | 'critical';
  causalStrength: number;
  lastUpdate: Date;
}

interface CausalRelationship {
  sourceProcess: string;
  targetProcess: string;
  strength: number;
  confidence: number;
  lag: number; // milliseconds
  type: 'direct' | 'indirect' | 'feedback';
}

interface ProcessAlert {
  id: string;
  processId: string;
  type: 'threshold' | 'anomaly' | 'causal_change' | 'predictive';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  causalContext?: string[];
  acknowledged: boolean;
}

interface StreamingDataPoint {
  timestamp: Date;
  processId: string;
  value: number;
  quality: number;
  causalInfluence: number;
}

interface AnomalyDetection {
  processId: string;
  anomalyScore: number;
  threshold: number;
  causalPathway: string[];
  confidence: number;
}

const RealTimeProcessMonitor: React.FC = () => {
  const [processes, setProcesses] = useState<GeologicalProcess[]>([]);
  const [causalRelationships, setCausalRelationships] = useState<CausalRelationship[]>([]);
  const [alerts, setAlerts] = useState<ProcessAlert[]>([]);
  const [streamingData, setStreamingData] = useState<Map<string, StreamingDataPoint[]>>(new Map());
  const [anomalies, setAnomalies] = useState<AnomalyDetection[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [timeWindow, setTimeWindow] = useState(3600000); // 1 hour in milliseconds
  const [_selectedProcess, setSelectedProcess] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize WebSocket connection for real-time data
  const initializeWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    wsRef.current = new WebSocket('ws://localhost:8080/ws/geological-monitoring');
    
    wsRef.current.onopen = () => {
      console.log('Real-time monitoring WebSocket connected');
      setIsMonitoring(true);
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleRealtimeData(data);
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket connection closed');
      setIsMonitoring(false);
      // Attempt to reconnect after 5 seconds
      setTimeout(initializeWebSocket, 5000);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, []);

  // Handle incoming real-time data
  const handleRealtimeData = useCallback((data: any) => {
    switch (data.type) {
      case 'process_update':
        updateProcessData(data.payload);
        break;
      case 'causal_relationship_update':
        updateCausalRelationships(data.payload);
        break;
      case 'alert':
        addAlert(data.payload);
        break;
      case 'streaming_data':
        addStreamingDataPoint(data.payload);
        break;
      case 'anomaly_detection':
        updateAnomalies(data.payload);
        break;
      default:
        console.warn('Unknown data type:', data.type);
    }
  }, []);

  // Update process data with real-time values
  const updateProcessData = useCallback((processData: Partial<GeologicalProcess>) => {
    setProcesses(prev => prev.map(process => 
      process.id === processData.id 
        ? { ...process, ...processData, lastUpdate: new Date() }
        : process
    ));
  }, []);

  // Update causal relationships with new strength values
  const updateCausalRelationships = useCallback((relationships: CausalRelationship[]) => {
    setCausalRelationships(relationships);
  }, []);

  // Add new alert to the system
  const addAlert = useCallback((alert: ProcessAlert) => {
    setAlerts(prev => [alert, ...prev].slice(0, 100)); // Keep last 100 alerts
  }, []);

  // Add streaming data point and maintain time window
  const addStreamingDataPoint = useCallback((dataPoint: StreamingDataPoint) => {
    const processId = dataPoint.processId;
    const currentTime = new Date().getTime();
    
    setStreamingData(prev => {
      const newMap = new Map(prev);
      const processData = newMap.get(processId) || [];
      
      // Add new data point
      const updatedData = [...processData, dataPoint];
      
      // Remove data points outside time window
      const filteredData = updatedData.filter(
        point => currentTime - point.timestamp.getTime() <= timeWindow
      );
      
      newMap.set(processId, filteredData);
      return newMap;
    });
  }, [timeWindow]);

  // Update anomaly detection results
  const updateAnomalies = useCallback((anomalyData: AnomalyDetection[]) => {
    setAnomalies(anomalyData);
  }, []);

  // Acknowledge alert
  const acknowledgeAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.map(alert => 
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ));
  }, []);

  // Get alert color based on severity
  const getAlertColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'high': return 'bg-orange-500';
      case 'medium': return 'bg-yellow-500';
      case 'low': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  // Get process status color
  const getProcessStatusColor = (alertLevel: string) => {
    switch (alertLevel) {
      case 'critical': return 'text-red-600';
      case 'high': return 'text-orange-600';
      case 'elevated': return 'text-yellow-600';
      case 'normal': return 'text-green-600';
      default: return 'text-gray-600';
    }
  };

  // Calculate causal influence for a process
  const calculateCausalInfluence = useCallback((processId: string): number => {
    const incomingInfluence = causalRelationships
      .filter(rel => rel.targetProcess === processId)
      .reduce((sum, rel) => sum + rel.strength * rel.confidence, 0);
    
    const outgoingInfluence = causalRelationships
      .filter(rel => rel.sourceProcess === processId)
      .reduce((sum, rel) => sum + rel.strength * rel.confidence, 0);
    
    return (incomingInfluence + outgoingInfluence) / 2;
  }, [causalRelationships]);

  // Generate predictive alerts based on causal models
  const generatePredictiveAlerts = useCallback(() => {
    processes.forEach(process => {
      const causalInfluence = calculateCausalInfluence(process.id);
      const processData = streamingData.get(process.id) || [];
      
      if (processData.length < 10) return; // Need minimum data for prediction
      
      // Simple trend analysis for prediction
      const recentData = processData.slice(-10);
      const trend = recentData[recentData.length - 1].value - recentData[0].value;
      const trendRate = trend / recentData.length;
      
      // Predict future value based on trend and causal influence
      const predictedValue = process.currentValue + (trendRate * 5 * (1 + causalInfluence));
      
      // Check if predicted value exceeds thresholds
      if (Math.abs(predictedValue - process.currentValue) > process.currentValue * 0.2) {
        const alert: ProcessAlert = {
          id: `pred_${Date.now()}_${process.id}`,
          processId: process.id,
          type: 'predictive',
          severity: predictedValue > process.currentValue * 1.5 ? 'high' : 'medium',
          message: `Predicted significant change in ${process.name}: ${predictedValue.toFixed(2)} ${process.unit}`,
          timestamp: new Date(),
          causalContext: causalRelationships
            .filter(rel => rel.targetProcess === process.id)
            .map(rel => rel.sourceProcess),
          acknowledged: false
        };
        addAlert(alert);
      }
    });
  }, [processes, streamingData, causalRelationships, calculateCausalInfluence, addAlert]);

  // Initialize monitoring system
  useEffect(() => {
    initializeWebSocket();
    
    // Set up predictive alert generation
    const predictiveInterval = setInterval(generatePredictiveAlerts, 30000); // Every 30 seconds
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      clearInterval(predictiveInterval);
    };
  }, [initializeWebSocket, generatePredictiveAlerts]);

  // Mock initial data (in production, this would come from the backend)
  useEffect(() => {
    const mockProcesses: GeologicalProcess[] = [
      {
        id: 'seismic_001',
        name: 'Seismic Activity - Mt. Vesuvius',
        type: 'seismic',
        location: { lat: 40.8214, lon: 14.4260 },
        currentValue: 2.3,
        unit: 'magnitude',
        trend: 'increasing',
        alertLevel: 'elevated',
        causalStrength: 0.75,
        lastUpdate: new Date()
      },
      {
        id: 'deform_001',
        name: 'Ground Deformation - Campi Flegrei',
        type: 'deformation',
        location: { lat: 40.8270, lon: 14.1390 },
        currentValue: 15.2,
        unit: 'mm/month',
        trend: 'increasing',
        alertLevel: 'high',
        causalStrength: 0.82,
        lastUpdate: new Date()
      },
      {
        id: 'thermal_001',
        name: 'Thermal Anomaly - Stromboli',
        type: 'thermal',
        location: { lat: 38.7890, lon: 15.2130 },
        currentValue: 850,
        unit: '°C',
        trend: 'volatile',
        alertLevel: 'critical',
        causalStrength: 0.68,
        lastUpdate: new Date()
      }
    ];
    
    setProcesses(mockProcesses);
  }, []);

  return (
    <div className="w-full h-full p-6 bg-gray-50">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <h1 className="text-3xl font-bold text-gray-900">Real-Time Process Monitor</h1>
          <div className="flex items-center space-x-4">
            <Badge variant={isMonitoring ? "default" : "destructive"}>
              {isMonitoring ? "Connected" : "Disconnected"}
            </Badge>
            <Button
              onClick={() => setTimeWindow(timeWindow === 3600000 ? 7200000 : 3600000)}
              variant="outline"
              size="sm"
            >
              {timeWindow === 3600000 ? "1H" : "2H"} Window
            </Button>
          </div>
        </div>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="processes">Processes</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="causal">Causal Analysis</TabsTrigger>
          <TabsTrigger value="anomalies">Anomalies</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Processes</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{processes.length}</div>
                <p className="text-xs text-muted-foreground">
                  {processes.filter(p => p.alertLevel !== 'normal').length} elevated
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{alerts.filter(a => !a.acknowledged).length}</div>
                <p className="text-xs text-muted-foreground">
                  {alerts.filter(a => a.severity === 'critical' && !a.acknowledged).length} critical
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Causal Strength</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(causalRelationships.reduce((sum, rel) => sum + rel.strength, 0) / Math.max(causalRelationships.length, 1)).toFixed(2)}
                </div>
                <p className="text-xs text-muted-foreground">
                  Average relationship strength
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Anomalies</CardTitle>
                <Zap className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{anomalies.length}</div>
                <p className="text-xs text-muted-foreground">
                  {anomalies.filter(a => a.anomalyScore > a.threshold * 2).length} severe
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Real-Time Process Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    type="number"
                    scale="time"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                  />
                  <Legend />
                  {processes.map((process, index) => {
                    const data = streamingData.get(process.id) || [];
                    return (
                      <Line
                        key={process.id}
                        data={data}
                        type="monotone"
                        dataKey="value"
                        stroke={`hsl(${index * 60}, 70%, 50%)`}
                        name={process.name}
                        strokeWidth={2}
                        dot={false}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="processes" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {processes.map((process) => (
              <Card key={process.id} className="cursor-pointer hover:shadow-lg transition-shadow"
                    onClick={() => setSelectedProcess(process.id)}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{process.name}</CardTitle>
                    <Badge variant="outline" className={getProcessStatusColor(process.alertLevel)}>
                      {process.alertLevel}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Current Value:</span>
                      <span className="font-semibold">{process.currentValue} {process.unit}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Trend:</span>
                      <span className={`font-semibold ${
                        process.trend === 'increasing' ? 'text-red-600' :
                        process.trend === 'decreasing' ? 'text-blue-600' :
                        process.trend === 'volatile' ? 'text-orange-600' : 'text-green-600'
                      }`}>
                        {process.trend}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Causal Strength:</span>
                      <span className="font-semibold">{(process.causalStrength * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Last Update:</span>
                      <span className="text-sm">{process.lastUpdate.toLocaleTimeString()}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <div className="space-y-2">
            {alerts.slice(0, 20).map((alert) => (
              <Alert key={alert.id} className={`${alert.acknowledged ? 'opacity-50' : ''}`}>
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${getAlertColor(alert.severity)}`} />
                    <div>
                      <AlertDescription className="font-medium">
                        {alert.message}
                      </AlertDescription>
                      <div className="text-xs text-gray-500 mt-1">
                        {alert.timestamp.toLocaleString()} • {alert.type}
                        {alert.causalContext && (
                          <span className="ml-2">
                            Causal: {alert.causalContext.join(', ')}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  {!alert.acknowledged && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => acknowledgeAlert(alert.id)}
                    >
                      Acknowledge
                    </Button>
                  )}
                </div>
              </Alert>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="causal" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Causal Relationship Network</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {causalRelationships.map((relationship, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <span className="font-medium">{relationship.sourceProcess}</span>
                      <span className="text-gray-400">→</span>
                      <span className="font-medium">{relationship.targetProcess}</span>
                    </div>
                    <div className="flex items-center space-x-4">
                      <Badge variant="outline">
                        Strength: {(relationship.strength * 100).toFixed(1)}%
                      </Badge>
                      <Badge variant="outline">
                        Confidence: {(relationship.confidence * 100).toFixed(1)}%
                      </Badge>
                      <Badge variant="outline">
                        Lag: {relationship.lag}ms
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="anomalies" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {anomalies.map((anomaly, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="text-lg">
                    Anomaly Detection - {anomaly.processId}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Anomaly Score:</span>
                      <span className={`font-semibold ${
                        anomaly.anomalyScore > anomaly.threshold * 2 ? 'text-red-600' :
                        anomaly.anomalyScore > anomaly.threshold ? 'text-orange-600' : 'text-green-600'
                      }`}>
                        {anomaly.anomalyScore.toFixed(3)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Threshold:</span>
                      <span className="font-semibold">{anomaly.threshold.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-gray-600">Confidence:</span>
                      <span className="font-semibold">{(anomaly.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="mt-3">
                      <span className="text-sm text-gray-600">Causal Pathway:</span>
                      <div className="mt-1 text-sm">
                        {anomaly.causalPathway.join(' → ')}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RealTimeProcessMonitor;