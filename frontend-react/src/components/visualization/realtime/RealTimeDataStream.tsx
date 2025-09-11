import React, { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Switch, 
  FormControlLabel, 
  Slider, 
  Chip, 
  Alert,
  LinearProgress,
  Grid,
  IconButton
} from '@mui/material';
import { 
  PlayArrow, 
  Pause, 
  Stop, 
  Settings, 
  Download,
  Refresh
} from '@mui/icons-material';
import { io, Socket } from 'socket.io-client';
import { GeologicalData, SeismicEvent, VolcanicState, StreamEvent } from '../../../types/geological';

interface RealTimeDataStreamProps {
  wsUrl?: string;
  bufferSize?: number;
  updateInterval?: number;
  onDataReceived?: (data: StreamEvent) => void;
  onConnectionChange?: (connected: boolean) => void;
  dataTypes?: string[];
  autoReconnect?: boolean;
}

interface StreamBuffer {
  geological: GeologicalData[];
  seismic: SeismicEvent[];
  volcanic: VolcanicState[];
  metadata: {
    totalReceived: number;
    lastUpdate: Date;
    dataRate: number;
    connectionQuality: number;
  };
}

// Advanced buffering system for real-time geological data
class GeologicalDataBuffer {
  private buffers: Map<string, any[]> = new Map();
  private maxSize: number;
  private timestamps: Map<string, number[]> = new Map();

  constructor(maxSize: number = 10000) {
    this.maxSize = maxSize;
  }

  addData(type: string, data: any) {
    if (!this.buffers.has(type)) {
      this.buffers.set(type, []);
      this.timestamps.set(type, []);
    }

    const buffer = this.buffers.get(type)!;
    const timestamps = this.timestamps.get(type)!;

    // Add new data
    buffer.push(data);
    timestamps.push(Date.now());

    // Maintain buffer size
    if (buffer.length > this.maxSize) {
      buffer.shift();
      timestamps.shift();
    }
  }

  getData(type: string, timeWindow?: number): any[] {
    const buffer = this.buffers.get(type) || [];
    
    if (!timeWindow) return buffer;

    const cutoff = Date.now() - timeWindow;
    const timestamps = this.timestamps.get(type) || [];
    
    return buffer.filter((_, index) => timestamps[index] > cutoff);
  }

  getDataRate(type: string, timeWindow: number = 60000): number {
    const timestamps = this.timestamps.get(type) || [];
    const cutoff = Date.now() - timeWindow;
    const recentCount = timestamps.filter(t => t > cutoff).length;
    
    return recentCount / (timeWindow / 1000); // data points per second
  }

  clear(type?: string) {
    if (type) {
      this.buffers.delete(type);
      this.timestamps.delete(type);
    } else {
      this.buffers.clear();
      this.timestamps.clear();
    }
  }

  getBufferInfo() {
    const info: Record<string, any> = {};
    
    for (const [type, buffer] of this.buffers) {
      info[type] = {
        count: buffer.length,
        dataRate: this.getDataRate(type),
        oldestTimestamp: this.timestamps.get(type)?.[0],
        newestTimestamp: this.timestamps.get(type)?.slice(-1)[0]
      };
    }
    
    return info;
  }
}

// Connection quality monitor
class ConnectionQualityMonitor {
  private latencyHistory: number[] = [];
  private packetLossCount: number = 0;
  private totalPackets: number = 0;
  private lastPingTime: number = 0;

  recordLatency(latency: number) {
    this.latencyHistory.push(latency);
    if (this.latencyHistory.length > 100) {
      this.latencyHistory.shift();
    }
  }

  recordPacketLoss() {
    this.packetLossCount++;
  }

  recordPacketReceived() {
    this.totalPackets++;
  }

  getQualityScore(): number {
    const avgLatency = this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length || 0;
    const packetLossRate = this.totalPackets > 0 ? this.packetLossCount / this.totalPackets : 0;
    
    // Quality score from 0-100
    const latencyScore = Math.max(0, 100 - (avgLatency / 10)); // Penalize high latency
    const lossScore = Math.max(0, 100 - (packetLossRate * 1000)); // Penalize packet loss
    
    return Math.min(100, (latencyScore + lossScore) / 2);
  }

  getMetrics() {
    return {
      averageLatency: this.latencyHistory.reduce((a, b) => a + b, 0) / this.latencyHistory.length || 0,
      packetLossRate: this.totalPackets > 0 ? this.packetLossCount / this.totalPackets : 0,
      qualityScore: this.getQualityScore()
    };
  }
}

export const RealTimeDataStream: React.FC<RealTimeDataStreamProps> = ({
  wsUrl = 'ws://localhost:8080',
  bufferSize = 10000,
  updateInterval = 1000,
  onDataReceived,
  onConnectionChange,
  dataTypes = ['geological', 'seismic', 'volcanic'],
  autoReconnect = true
}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [connectionQuality, setConnectionQuality] = useState(0);
  const [streamBuffer, setStreamBuffer] = useState<StreamBuffer>({
    geological: [],
    seismic: [],
    volcanic: [],
    metadata: {
      totalReceived: 0,
      lastUpdate: new Date(),
      dataRate: 0,
      connectionQuality: 0
    }
  });

  const socketRef = useRef<Socket | null>(null);
  const bufferRef = useRef(new GeologicalDataBuffer(bufferSize));
  const qualityMonitorRef = useRef(new ConnectionQualityMonitor());
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Stream configuration
  const [streamConfig, setStreamConfig] = useState({
    timeWindow: 300000, // 5 minutes
    maxDataRate: 100, // points per second
    compressionEnabled: true,
    qualityThreshold: 50
  });

  // Initialize WebSocket connection
  const initializeConnection = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
    }

    const socket = io(wsUrl, {
      transports: ['websocket'],
      upgrade: true,
      rememberUpgrade: true,
      timeout: 5000,
      forceNew: true
    });

    socketRef.current = socket;

    // Connection event handlers
    socket.on('connect', () => {
      console.log('Connected to geological data stream');
      setIsConnected(true);
      onConnectionChange?.(true);
      
      // Subscribe to data types
      dataTypes.forEach(type => {
        socket.emit('subscribe', { dataType: type, config: streamConfig });
      });
    });

    socket.on('disconnect', (reason) => {
      console.log('Disconnected from geological data stream:', reason);
      setIsConnected(false);
      setIsStreaming(false);
      onConnectionChange?.(false);

      // Auto-reconnect logic
      if (autoReconnect && reason !== 'io client disconnect') {
        reconnectTimeoutRef.current = setTimeout(() => {
          initializeConnection();
        }, 5000);
      }
    });

    // Data event handlers
    socket.on('geological_data', (data: GeologicalData) => {
      handleDataReceived('geological', data);
    });

    socket.on('seismic_event', (data: SeismicEvent) => {
      handleDataReceived('seismic', data);
    });

    socket.on('volcanic_state', (data: VolcanicState) => {
      handleDataReceived('volcanic', data);
    });

    // Stream metadata
    socket.on('stream_metadata', (metadata) => {
      setStreamBuffer(prev => ({
        ...prev,
        metadata: {
          ...prev.metadata,
          ...metadata,
          lastUpdate: new Date()
        }
      }));
    });

    // Connection quality monitoring
    socket.on('pong', (latency) => {
      qualityMonitorRef.current.recordLatency(latency);
      setConnectionQuality(qualityMonitorRef.current.getQualityScore());
    });

    // Error handling
    socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      qualityMonitorRef.current.recordPacketLoss();
    });

  }, [wsUrl, dataTypes, streamConfig, autoReconnect, onConnectionChange]);

  // Handle incoming data
  const handleDataReceived = useCallback((type: string, data: any) => {
    // Add to buffer
    bufferRef.current.addData(type, data);
    qualityMonitorRef.current.recordPacketReceived();

    // Create stream event
    const streamEvent: StreamEvent = {
      event_id: `${type}_${Date.now()}_${Math.random()}`,
      event_type: type as any,
      timestamp: new Date(),
      source_id: data.id || 'unknown',
      data,
      metadata: {
        bufferSize: bufferRef.current.getBufferInfo()[type]?.count || 0,
        dataRate: bufferRef.current.getDataRate(type)
      },
      processing_status: 'received'
    };

    onDataReceived?.(streamEvent);

    // Update component state
    setStreamBuffer(prev => {
      const newBuffer = { ...prev };
      
      // Update data arrays
      if (type === 'geological') {
        newBuffer.geological = bufferRef.current.getData('geological', streamConfig.timeWindow);
      } else if (type === 'seismic') {
        newBuffer.seismic = bufferRef.current.getData('seismic', streamConfig.timeWindow);
      } else if (type === 'volcanic') {
        newBuffer.volcanic = bufferRef.current.getData('volcanic', streamConfig.timeWindow);
      }

      // Update metadata
      newBuffer.metadata = {
        totalReceived: prev.metadata.totalReceived + 1,
        lastUpdate: new Date(),
        dataRate: bufferRef.current.getDataRate(type),
        connectionQuality: qualityMonitorRef.current.getQualityScore()
      };

      return newBuffer;
    });
  }, [streamConfig.timeWindow, onDataReceived]);

  // Start/stop streaming
  const toggleStreaming = useCallback(() => {
    if (!socketRef.current || !isConnected) return;

    if (isStreaming) {
      socketRef.current.emit('stop_stream');
      setIsStreaming(false);
    } else {
      socketRef.current.emit('start_stream', { config: streamConfig });
      setIsStreaming(true);
    }
  }, [isConnected, isStreaming, streamConfig]);

  // Clear buffer
  const clearBuffer = useCallback(() => {
    bufferRef.current.clear();
    setStreamBuffer({
      geological: [],
      seismic: [],
      volcanic: [],
      metadata: {
        totalReceived: 0,
        lastUpdate: new Date(),
        dataRate: 0,
        connectionQuality: 0
      }
    });
  }, []);

  // Export data
  const exportData = useCallback((format: 'json' | 'csv' = 'json') => {
    const data = {
      geological: streamBuffer.geological,
      seismic: streamBuffer.seismic,
      volcanic: streamBuffer.volcanic,
      metadata: streamBuffer.metadata,
      exportTime: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `geological_stream_${Date.now()}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  }, [streamBuffer]);

  // Initialize connection on mount
  useEffect(() => {
    initializeConnection();

    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [initializeConnection]);

  // Periodic ping for connection quality
  useEffect(() => {
    if (!isConnected || !socketRef.current) return;

    const pingInterval = setInterval(() => {
      const startTime = Date.now();
      socketRef.current?.emit('ping', startTime);
    }, 5000);

    return () => clearInterval(pingInterval);
  }, [isConnected]);

  // Connection status indicator
  const getConnectionStatusColor = () => {
    if (!isConnected) return 'error';
    if (connectionQuality > 80) return 'success';
    if (connectionQuality > 50) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ width: '100%', p: 2 }}>
      {/* Connection Status */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Chip
                label={isConnected ? 'Connected' : 'Disconnected'}
                color={getConnectionStatusColor()}
                variant={isConnected ? 'filled' : 'outlined'}
              />
              <Chip
                label={`Quality: ${connectionQuality.toFixed(0)}%`}
                color={getConnectionStatusColor()}
                variant="outlined"
              />
              {isStreaming && (
                <Chip
                  label="Streaming"
                  color="primary"
                  variant="filled"
                />
              )}
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
              <IconButton
                onClick={toggleStreaming}
                disabled={!isConnected}
                color="primary"
              >
                {isStreaming ? <Pause /> : <PlayArrow />}
              </IconButton>
              <IconButton onClick={clearBuffer} color="secondary">
                <Stop />
              </IconButton>
              <IconButton onClick={() => exportData('json')} color="default">
                <Download />
              </IconButton>
              <IconButton onClick={initializeConnection} color="default">
                <Refresh />
              </IconButton>
            </Box>
          </Grid>
        </Grid>

        {/* Connection Quality Progress */}
        <Box sx={{ mt: 2 }}>
          <Typography variant="body2" gutterBottom>
            Connection Quality
          </Typography>
          <LinearProgress
            variant="determinate"
            value={connectionQuality}
            color={getConnectionStatusColor()}
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Box>
      </Paper>

      {/* Stream Statistics */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Stream Statistics
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="textSecondary">
              Geological Data
            </Typography>
            <Typography variant="h6">
              {streamBuffer.geological.length}
            </Typography>
            <Typography variant="caption">
              {bufferRef.current.getDataRate('geological').toFixed(1)} pts/sec
            </Typography>
          </Grid>
          
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="textSecondary">
              Seismic Events
            </Typography>
            <Typography variant="h6">
              {streamBuffer.seismic.length}
            </Typography>
            <Typography variant="caption">
              {bufferRef.current.getDataRate('seismic').toFixed(1)} events/sec
            </Typography>
          </Grid>
          
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="textSecondary">
              Volcanic States
            </Typography>
            <Typography variant="h6">
              {streamBuffer.volcanic.length}
            </Typography>
            <Typography variant="caption">
              {bufferRef.current.getDataRate('volcanic').toFixed(1)} updates/sec
            </Typography>
          </Grid>
          
          <Grid item xs={6} md={3}>
            <Typography variant="body2" color="textSecondary">
              Total Received
            </Typography>
            <Typography variant="h6">
              {streamBuffer.metadata.totalReceived}
            </Typography>
            <Typography variant="caption">
              Last: {streamBuffer.metadata.lastUpdate.toLocaleTimeString()}
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Stream Configuration */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Stream Configuration
        </Typography>
        
        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>
              Time Window: {streamConfig.timeWindow / 1000}s
            </Typography>
            <Slider
              value={streamConfig.timeWindow}
              onChange={(_, value) => setStreamConfig(prev => ({ ...prev, timeWindow: value as number }))}
              min={30000}
              max={3600000}
              step={30000}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => `${value / 1000}s`}
            />
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography gutterBottom>
              Max Data Rate: {streamConfig.maxDataRate} pts/sec
            </Typography>
            <Slider
              value={streamConfig.maxDataRate}
              onChange={(_, value) => setStreamConfig(prev => ({ ...prev, maxDataRate: value as number }))}
              min={1}
              max={1000}
              step={10}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={streamConfig.compressionEnabled}
                  onChange={(e) => setStreamConfig(prev => ({ ...prev, compressionEnabled: e.target.checked }))}
                />
              }
              label="Enable Data Compression"
            />
          </Grid>
        </Grid>
      </Paper>

      {/* Alerts */}
      {!isConnected && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Not connected to data stream. Check your connection and try again.
        </Alert>
      )}
      
      {connectionQuality < 50 && isConnected && (
        <Alert severity="warning" sx={{ mt: 2 }}>
          Poor connection quality detected. Data may be delayed or incomplete.
        </Alert>
      )}
    </Box>
  );
};

export default RealTimeDataStream;