/**
 * Data Synchronization Panel Component
 * Manages real-time data stream synchronization and quality monitoring
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  IconButton,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Switch,
  FormControlLabel,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,

  Tooltip,
  Badge,
} from '@mui/material';
import {
  Sync,
  SyncProblem,
  CheckCircle,
  Error as ErrorIcon,
  Warning,
  Refresh,
  Settings,
  Timeline,
  Speed,

  ExpandMore,
  Info,
} from '@mui/icons-material';
// Removed unused useTheme import

interface GeologicalDataStream {
  id: string;
  name: string;
  type: 'seismic' | 'volcanic' | 'geochemical' | 'deformation' | 'thermal';
  source: string;
  frequency: number; // Hz
  lastValue: number;
  unit: string;
  quality: number; // 0-1
  isActive: boolean;
  latency: number; // ms
  packetsLost: number;
  totalPackets: number;
  lastSync: Date;
  syncStatus: 'synced' | 'syncing' | 'error' | 'offline';
}

interface SyncConfiguration {
  bufferSize: number;
  maxLatency: number;
  qualityThreshold: number;
  retryAttempts: number;
  syncInterval: number;
}

interface DataSynchronizationPanelProps {
  dataStreams: GeologicalDataStream[];
  onStreamUpdate: (streams: GeologicalDataStream[]) => void;
  onSyncAll?: () => Promise<void>;
  onStreamToggle?: (streamId: string, enabled: boolean) => void;
}

const DataSynchronizationPanel: React.FC<DataSynchronizationPanelProps> = ({
  dataStreams,
  onStreamUpdate,
  onSyncAll,
  onStreamToggle,
}) => {
  // Removed unused theme

  // State management
  const [syncConfig, setSyncConfig] = useState<SyncConfiguration>({
    bufferSize: 1000,
    maxLatency: 5000,
    qualityThreshold: 0.8,
    retryAttempts: 3,
    syncInterval: 1000,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);

  // Get sync status statistics
  const syncStats = React.useMemo(() => {
    const total = dataStreams.length;
    const active = dataStreams.filter(s => s.isActive).length;
    const synced = dataStreams.filter(s => s.syncStatus === 'synced').length;
    const errors = dataStreams.filter(s => s.syncStatus === 'error').length;
    const avgQuality = dataStreams.reduce((sum, s) => sum + s.quality, 0) / total || 0;
    const avgLatency = dataStreams.reduce((sum, s) => sum + s.latency, 0) / total || 0;
    
    return {
      total,
      active,
      synced,
      errors,
      avgQuality,
      avgLatency,
      packetLoss: dataStreams.reduce((sum, s) => sum + s.packetsLost, 0),
      totalPackets: dataStreams.reduce((sum, s) => sum + s.totalPackets, 0),
    };
  }, [dataStreams]);

  // Handle stream toggle
  const handleStreamToggle = useCallback((streamId: string) => {
    const updatedStreams = dataStreams.map(stream => {
      if (stream.id === streamId) {
        const newActive = !stream.isActive;
        onStreamToggle?.(streamId, newActive);
        return {
          ...stream,
          isActive: newActive,
          syncStatus: newActive ? ('syncing' as const) : ('offline' as const),
        };
      }
      return stream;
    });
    onStreamUpdate(updatedStreams);
  }, [dataStreams, onStreamUpdate, onStreamToggle]);

  // Handle sync all streams
  const handleSyncAll = useCallback(async () => {
    setIsSyncing(true);
    try {
      if (onSyncAll) {
        await onSyncAll();
      } else {
        // Simulate sync process
        await new Promise(resolve => setTimeout(resolve, 2000));
        const updatedStreams = dataStreams.map(stream => ({
          ...stream,
          syncStatus: stream.isActive ? ('synced' as const) : ('offline' as const),
          lastSync: new Date(),
          quality: Math.min(1, stream.quality + 0.1),
          latency: Math.max(100, stream.latency - 100),
        }));
        onStreamUpdate(updatedStreams);
      }
    } catch (error) {
      console.error('Sync failed:', error);
    } finally {
      setIsSyncing(false);
    }
  }, [dataStreams, onStreamUpdate, onSyncAll]);

  // Get status color and icon
  const getStatusDisplay = (status: string) => {
    switch (status) {
      case 'synced':
        return { color: 'success', icon: <CheckCircle />, label: 'Synced' };
      case 'syncing':
        return { color: 'primary', icon: <Sync />, label: 'Syncing' };
      case 'error':
        return { color: 'error', icon: <ErrorIcon />, label: 'Error' };
      case 'offline':
        return { color: 'default', icon: <SyncProblem />, label: 'Offline' };
      default:
        return { color: 'default', icon: <SyncProblem />, label: 'Unknown' };
    }
  };

  // Get quality color
  const getQualityColor = (quality: number) => {
    if (quality >= 0.9) return 'success';
    if (quality >= 0.7) return 'warning';
    return 'error';
  };

  return (
    <Box>
      {/* Summary Statistics */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="primary">
                {syncStats.active}/{syncStats.total}
              </Typography>
              <Typography variant="caption">Active Streams</Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="success.main">
                {syncStats.synced}
              </Typography>
              <Typography variant="caption">Synced</Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="warning.main">
                {(syncStats.avgQuality * 100).toFixed(0)}%
              </Typography>
              <Typography variant="caption">Avg Quality</Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={6} md={3}>
          <Card variant="outlined">
            <CardContent sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="h4" color="info.main">
                {syncStats.avgLatency.toFixed(0)}ms
              </Typography>
              <Typography variant="caption">Avg Latency</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Typography variant="h6">Data Stream Synchronization</Typography>
        <Box display="flex" gap={1}>
          <Button
            variant="contained"
            startIcon={<Sync />}
            onClick={handleSyncAll}
            disabled={isSyncing}
          >
            {isSyncing ? 'Syncing...' : 'Sync All'}
          </Button>
          <Button
            variant="outlined"
            startIcon={<Settings />}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            Settings
          </Button>
        </Box>
      </Box>

      {/* Sync Progress */}
      {isSyncing && (
        <Box mb={2}>
          <LinearProgress />
          <Typography variant="caption" color="textSecondary">
            Synchronizing data streams...
          </Typography>
        </Box>
      )}

      {/* Error Summary */}
      {syncStats.errors > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {syncStats.errors} stream{syncStats.errors > 1 ? 's' : ''} experiencing sync errors
        </Alert>
      )}

      {/* Data Streams List */}
      <Card>
        <CardContent>
          <List>
            {dataStreams.map((stream, index) => {
              const statusDisplay = getStatusDisplay(stream.syncStatus);
              const packetLossRate = stream.totalPackets > 0 
                ? (stream.packetsLost / stream.totalPackets * 100).toFixed(2)
                : '0.00';

              return (
                <ListItem key={stream.id} divider={index < dataStreams.length - 1}>
                  <ListItemIcon>
                    <Badge
                      color={statusDisplay.color as any}
                      variant="dot"
                      invisible={stream.syncStatus === 'offline'}
                    >
                      {stream.type === 'seismic' && <Timeline />}
                      {stream.type === 'volcanic' && <Warning />}
                      {stream.type === 'deformation' && <Speed />}
                      {stream.type === 'thermal' && <Info />}
                      {stream.type === 'geochemical' && <Info />}
                    </Badge>
                  </ListItemIcon>
                  
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle2">
                          {stream.name}
                        </Typography>
                        <Chip
                          size="small"
                          label={statusDisplay.label}
                          color={statusDisplay.color as any}
                          icon={statusDisplay.icon}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          Source: {stream.source} • Frequency: {stream.frequency} Hz
                        </Typography>
                        <Typography variant="caption" display="block">
                          Last Value: {stream.lastValue.toFixed(3)} {stream.unit}
                        </Typography>
                        <Box display="flex" alignItems="center" gap={2} mt={0.5}>
                          <Box display="flex" alignItems="center" gap={0.5}>
                            <Typography variant="caption">Quality:</Typography>
                            <Chip
                              size="small"
                              label={`${(stream.quality * 100).toFixed(0)}%`}
                              color={getQualityColor(stream.quality)}
                            />
                          </Box>
                          <Typography variant="caption">
                            Latency: {stream.latency}ms
                          </Typography>
                          <Typography variant="caption">
                            Loss: {packetLossRate}%
                          </Typography>
                          <Typography variant="caption">
                            Last Sync: {stream.lastSync.toLocaleTimeString()}
                          </Typography>
                        </Box>
                      </Box>
                    }
                  />
                  
                  <ListItemSecondaryAction>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Tooltip title={stream.isActive ? 'Disable stream' : 'Enable stream'}>
                        <Switch
                          checked={stream.isActive}
                          onChange={() => handleStreamToggle(stream.id)}
                          size="small"
                        />
                      </Tooltip>
                      <Tooltip title="Refresh stream">
                        <IconButton
                          size="small"
                          onClick={() => {
                            // Simulate refresh
                            const updatedStreams = dataStreams.map(s =>
                              s.id === stream.id
                                ? { ...s, lastSync: new Date(), syncStatus: 'syncing' as const }
                                : s
                            );
                            onStreamUpdate(updatedStreams);
                            
                            setTimeout(() => {
                              const finalStreams = dataStreams.map(s =>
                                s.id === stream.id
                                  ? { ...s, syncStatus: 'synced' as const }
                                  : s
                              );
                              onStreamUpdate(finalStreams);
                            }, 1000);
                          }}
                        >
                          <Refresh />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>
              );
            })}
          </List>
        </CardContent>
      </Card>

      {/* Advanced Settings */}
      {showAdvanced && (
        <Accordion expanded sx={{ mt: 2 }}>
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6">Advanced Synchronization Settings</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Buffer Size"
                  type="number"
                  value={syncConfig.bufferSize}
                  onChange={(e) => setSyncConfig(prev => ({
                    ...prev,
                    bufferSize: parseInt(e.target.value)
                  }))}
                  helperText="Number of samples to buffer"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Max Latency (ms)"
                  type="number"
                  value={syncConfig.maxLatency}
                  onChange={(e) => setSyncConfig(prev => ({
                    ...prev,
                    maxLatency: parseInt(e.target.value)
                  }))}
                  helperText="Maximum acceptable latency"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Quality Threshold"
                  type="number"
                  inputProps={{ min: 0, max: 1, step: 0.1 }}
                  value={syncConfig.qualityThreshold}
                  onChange={(e) => setSyncConfig(prev => ({
                    ...prev,
                    qualityThreshold: parseFloat(e.target.value)
                  }))}
                  helperText="Minimum quality threshold (0-1)"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Retry Attempts"
                  type="number"
                  value={syncConfig.retryAttempts}
                  onChange={(e) => setSyncConfig(prev => ({
                    ...prev,
                    retryAttempts: parseInt(e.target.value)
                  }))}
                  helperText="Number of retry attempts on failure"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={<Switch defaultChecked />}
                  label="Auto-reconnect on connection loss"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={<Switch defaultChecked />}
                  label="Enable quality-based filtering"
                />
              </Grid>
            </Grid>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Network Statistics */}
      <Card sx={{ mt: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Network Statistics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} md={3}>
              <Typography variant="caption">Total Packets</Typography>
              <Typography variant="h6">{syncStats.totalPackets.toLocaleString()}</Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption">Packets Lost</Typography>
              <Typography variant="h6" color="error.main">
                {syncStats.packetLoss.toLocaleString()}
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption">Loss Rate</Typography>
              <Typography variant="h6">
                {syncStats.totalPackets > 0 
                  ? ((syncStats.packetLoss / syncStats.totalPackets) * 100).toFixed(2)
                  : '0.00'
                }%
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="caption">Throughput</Typography>
              <Typography variant="h6">
                {(dataStreams.reduce((sum, s) => sum + s.frequency, 0)).toFixed(0)} Hz
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default DataSynchronizationPanel;