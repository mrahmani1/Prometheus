/**
 * Unified Geological Analysis Interface
 * Integrates seismic analysis, volcanic monitoring, geological modeling, and geochemical analysis
 * with real-time data visualization and cross-module data sharing
 */

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Tabs,
  Tab,
  Button,
  IconButton,
  Chip,
  Alert,
  LinearProgress,
  Switch,
  FormControlLabel,
  Badge,
  Divider,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  Tooltip,
} from '@mui/material';
import {
  Timeline,
  TrendingUp,
  Layers,
  Science,
  Terrain,
  Sync,
  Settings,
  Download,
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';

// Import visualization components
import SeismicWaveformVisualization from '../visualization/SeismicWaveformVisualization';
import VolcanicMonitoringDashboard from '../monitoring/VolcanicMonitoringDashboard';
import GeologicalModelViewer from '../geological/GeologicalModelViewer';
import GeochemicalAnalysisPlots from '../geological/GeochemicalAnalysisPlots';
import DeformationAnalysisViewer from '../geological/DeformationAnalysisViewer';
import MultiParameterCorrelationMatrix from '../visualization/MultiParameterCorrelationMatrix';
import WorkflowManager from '../common/WorkflowManager';
import DataSynchronizationPanel from '../common/DataSynchronizationPanel';

// Types and interfaces
interface AnalysisModule {
  id: string;
  name: string;
  icon: React.ReactNode;
  component: React.ComponentType<any>;
  enabled: boolean;
  realTimeCapable: boolean;
  dataTypes: string[];
  lastUpdate?: Date;
  status: 'active' | 'inactive' | 'error' | 'loading';
}

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

interface AnalysisWorkflow {
  id: string;
  name: string;
  description: string;
  modules: string[];
  parameters: Record<string, any>;
  status: 'idle' | 'running' | 'completed' | 'error';
  progress: number;
  results?: any;
  createdAt: Date;
  lastRun?: Date;
  duration?: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => (
  <div
    role="tabpanel"
    hidden={value !== index}
    id={`analysis-tabpanel-${index}`}
    aria-labelledby={`analysis-tab-${index}`}
    {...other}
  >
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const UnifiedGeologicalAnalysisInterface: React.FC = () => {
  // Removed unused theme, dispatch, and selector values
  const { notifications } = useSelector((state: RootState) => state.dashboard);

  // State management
  const [activeTab, setActiveTab] = useState(0);
  const [selectedModules, setSelectedModules] = useState<string[]>(['seismic', 'volcanic']);
  const [realTimeMode, setRealTimeMode] = useState(false);
  const [dataStreams, setDataStreams] = useState<GeologicalDataStream[]>([]);
  const [activeWorkflows, setActiveWorkflows] = useState<AnalysisWorkflow[]>([]);
  const [showSettings, setShowSettings] = useState(false);
  const [correlationMatrix, setCorrelationMatrix] = useState<number[][]>([]);
  // Removed unused alertsOpen state
  const [syncStatus, setSyncStatus] = useState<'synced' | 'syncing' | 'error'>('synced');

  // Refs for real-time updates
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Analysis modules configuration
  const analysisModules: AnalysisModule[] = useMemo(() => [
    {
      id: 'seismic',
      name: 'Seismic Analysis',
      icon: <Timeline />,
      component: SeismicWaveformVisualization,
      enabled: true,
      realTimeCapable: true,
      dataTypes: ['waveform', 'phase_picks', 'magnitude', 'location'],
      status: 'active',
    },
    {
      id: 'volcanic',
      name: 'Volcanic Monitoring',
      icon: <Terrain />,
      component: VolcanicMonitoringDashboard,
      enabled: true,
      realTimeCapable: true,
      dataTypes: ['deformation', 'gas_emission', 'thermal', 'seismic'],
      status: 'active',
    },
    {
      id: 'geological',
      name: 'Geological Modeling',
      icon: <Layers />,
      component: GeologicalModelViewer,
      enabled: true,
      realTimeCapable: false,
      dataTypes: ['structure', 'lithology', 'age', 'properties'],
      status: 'active',
    },
    {
      id: 'geochemical',
      name: 'Geochemical Analysis',
      icon: <Science />,
      component: GeochemicalAnalysisPlots,
      enabled: true,
      realTimeCapable: false,
      dataTypes: ['composition', 'isotopes', 'trace_elements', 'volatiles'],
      status: 'active',
    },
    {
      id: 'deformation',
      name: 'Deformation Analysis',
      icon: <TrendingUp />,
      component: DeformationAnalysisViewer,
      enabled: true,
      realTimeCapable: true,
      dataTypes: ['gps', 'insar', 'tilt', 'strain'],
      status: 'active',
    },
  ], []);

  // Initialize real-time data streams
  useEffect(() => {
    const initializeDataStreams = async () => {
      try {
        // Simulate fetching real data stream configurations
        const streams: GeologicalDataStream[] = [
          {
            id: 'seismic_broadband',
            name: 'Broadband Seismic',
            type: 'seismic',
            source: 'IRIS/EarthScope',
            frequency: 100,
            lastValue: 0.0023,
            unit: 'm/s',
            quality: 0.95,
            isActive: true,
            latency: 150,
            packetsLost: 0,
            totalPackets: 1000,
            lastSync: new Date(),
            syncStatus: 'synced' as const,
          },
          {
            id: 'volcanic_so2',
            name: 'SO2 Emissions',
            type: 'volcanic',
            source: 'USGS Volcano Observatory',
            frequency: 0.1,
            lastValue: 450,
            unit: 'tons/day',
            quality: 0.88,
            isActive: true,
            latency: 200,
            packetsLost: 2,
            totalPackets: 500,
            lastSync: new Date(),
            syncStatus: 'synced' as const,
          },
          {
            id: 'gps_displacement',
            name: 'GPS Displacement',
            type: 'deformation',
            source: 'UNAVCO',
            frequency: 1,
            lastValue: 2.3,
            unit: 'mm',
            quality: 0.92,
            isActive: true,
            latency: 100,
            packetsLost: 1,
            totalPackets: 800,
            lastSync: new Date(),
            syncStatus: 'synced' as const,
          },
          {
            id: 'thermal_anomaly',
            name: 'Thermal Anomaly',
            type: 'thermal',
            source: 'NASA FIRMS',
            frequency: 0.25,
            lastValue: 15.2,
            unit: 'K',
            quality: 0.85,
            isActive: true,
            latency: 300,
            packetsLost: 5,
            totalPackets: 300,
            lastSync: new Date(),
            syncStatus: 'synced' as const,
          },
        ];
        setDataStreams(streams);
      } catch (error) {
        console.error('Failed to initialize data streams:', error);
      }
    };

    initializeDataStreams();
  }, []);

  // Real-time data updates
  useEffect(() => {
    if (realTimeMode) {
      // Initialize WebSocket connection
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws/geological-data';
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected for real-time geological data');
        setSyncStatus('synced');
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleRealTimeUpdate(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      wsRef.current.onerror = () => {
        setSyncStatus('error');
      };

      wsRef.current.onclose = () => {
        setSyncStatus('error');
        // Attempt to reconnect
        setTimeout(() => {
          if (realTimeMode) {
            // Reconnection logic would go here
          }
        }, 5000);
      };

      // Fallback polling for data updates
      intervalRef.current = setInterval(() => {
        updateDataStreams();
      }, 1000);
    } else {
      // Cleanup connections
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [realTimeMode]);

  // Handle real-time data updates
  const handleRealTimeUpdate = useCallback((data: any) => {
    setDataStreams(prevStreams => 
      prevStreams.map(stream => {
        if (data.streamId === stream.id) {
          return {
            ...stream,
            lastValue: data.value,
            quality: data.quality || stream.quality,
            isActive: data.isActive !== undefined ? data.isActive : stream.isActive,
          };
        }
        return stream;
      })
    );

    // Update correlation matrix if needed
    if (data.correlationUpdate) {
      setCorrelationMatrix(data.correlationMatrix);
    }
  }, []);

  // Update data streams (fallback polling)
  const updateDataStreams = useCallback(async () => {
    try {
      setSyncStatus('syncing');
      // Simulate API call to update stream data
      await new Promise(resolve => setTimeout(resolve, 100));
      
      setDataStreams(prevStreams => 
        prevStreams.map(stream => ({
          ...stream,
          lastValue: stream.lastValue + (Math.random() - 0.5) * 0.1,
          quality: Math.max(0.7, Math.min(1.0, stream.quality + (Math.random() - 0.5) * 0.05)),
        }))
      );
      setSyncStatus('synced');
    } catch (error) {
      setSyncStatus('error');
      console.error('Failed to update data streams:', error);
    }
  }, []);

  // Handle tab change
  const handleTabChange = useCallback((_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  }, []);

  // Handle module selection
  const handleModuleToggle = useCallback((moduleId: string) => {
    setSelectedModules(prev => 
      prev.includes(moduleId) 
        ? prev.filter(id => id !== moduleId)
        : [...prev, moduleId]
    );
  }, []);

  // Start analysis workflow
  const startWorkflow = useCallback(async (workflowConfig: Partial<AnalysisWorkflow>) => {
    const workflow: AnalysisWorkflow = {
      id: `workflow_${Date.now()}`,
      name: workflowConfig.name || 'Unnamed Workflow',
      description: workflowConfig.description || '',
      modules: workflowConfig.modules || selectedModules,
      parameters: workflowConfig.parameters || {},
      status: 'running',
      progress: 0,
      createdAt: new Date(),
    };

    setActiveWorkflows(prev => [...prev, workflow]);

    try {
      // Simulate workflow execution
      for (let progress = 0; progress <= 100; progress += 10) {
        await new Promise(resolve => setTimeout(resolve, 500));
        setActiveWorkflows(prev => 
          prev.map(w => w.id === workflow.id ? { ...w, progress } : w)
        );
      }

      // Mark as completed
      setActiveWorkflows(prev => 
        prev.map(w => w.id === workflow.id ? { ...w, status: 'completed', results: { success: true } } : w)
      );
    } catch (error) {
      setActiveWorkflows(prev => 
        prev.map(w => w.id === workflow.id ? { ...w, status: 'error' } : w)
      );
    }
  }, [selectedModules]);

  // Export analysis results
  const exportResults = useCallback(async (format: 'json' | 'csv' | 'pdf') => {
    try {
      // Simulate export process
      const exportData = {
        timestamp: new Date().toISOString(),
        modules: selectedModules,
        dataStreams: dataStreams.filter(s => s.isActive),
        workflows: activeWorkflows,
        correlationMatrix,
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `geological_analysis_${Date.now()}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    }
  }, [selectedModules, dataStreams, activeWorkflows, correlationMatrix]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header with controls */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container alignItems="center" spacing={2}>
          <Grid item xs>
            <Typography variant="h5" component="h1">
              Unified Geological Analysis Interface
            </Typography>
            <Typography variant="body2" color="textSecondary">
              Integrated seismic, volcanic, geological, and geochemical analysis
            </Typography>
          </Grid>
          
          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={realTimeMode}
                  onChange={(e) => setRealTimeMode(e.target.checked)}
                  color="primary"
                />
              }
              label="Real-time Mode"
            />
          </Grid>

          <Grid item>
            <Tooltip title="Data Synchronization Status">
              <IconButton color={syncStatus === 'synced' ? 'success' : syncStatus === 'syncing' ? 'warning' : 'error'}>
                <Badge color={syncStatus === 'synced' ? 'success' : 'error'} variant="dot">
                  <Sync />
                </Badge>
              </IconButton>
            </Tooltip>
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Settings />}
              onClick={() => setShowSettings(true)}
            >
              Settings
            </Button>
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={() => exportResults('json')}
            >
              Export
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Main content area */}
      <Box sx={{ flexGrow: 1, display: 'flex' }}>
        {/* Left sidebar - Module selection and data streams */}
        <Paper elevation={1} sx={{ width: 300, mr: 2, p: 2, overflowY: 'auto' }}>
          <Typography variant="h6" gutterBottom>
            Analysis Modules
          </Typography>
          
          {analysisModules.map((module) => (
            <Card key={module.id} sx={{ mb: 1, opacity: selectedModules.includes(module.id) ? 1 : 0.6 }}>
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box display="flex" alignItems="center">
                    {module.icon}
                    <Box ml={1}>
                      <Typography variant="body2" fontWeight="medium">
                        {module.name}
                      </Typography>
                      <Chip
                        size="small"
                        label={module.status}
                        color={module.status === 'active' ? 'success' : 'default'}
                      />
                    </Box>
                  </Box>
                  <Switch
                    size="small"
                    checked={selectedModules.includes(module.id)}
                    onChange={() => handleModuleToggle(module.id)}
                  />
                </Box>
              </CardContent>
            </Card>
          ))}

          <Divider sx={{ my: 2 }} />

          <Typography variant="h6" gutterBottom>
            Data Streams
          </Typography>
          
          {dataStreams.map((stream) => (
            <Card key={stream.id} sx={{ mb: 1 }}>
              <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography variant="body2" fontWeight="medium">
                      {stream.name}
                    </Typography>
                    <Typography variant="caption" color="textSecondary">
                      {stream.lastValue.toFixed(3)} {stream.unit}
                    </Typography>
                  </Box>
                  <Box display="flex" alignItems="center">
                    <LinearProgress
                      variant="determinate"
                      value={stream.quality * 100}
                      sx={{ width: 40, mr: 1 }}
                    />
                    <Chip
                      size="small"
                      label={stream.isActive ? 'Active' : 'Inactive'}
                      color={stream.isActive ? 'success' : 'default'}
                    />
                  </Box>
                </Box>
              </CardContent>
            </Card>
          ))}
        </Paper>

        {/* Main analysis area */}
        <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <Paper elevation={1} sx={{ flexGrow: 1 }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab label="Integrated View" />
              <Tab label="Seismic Analysis" />
              <Tab label="Volcanic Monitoring" />
              <Tab label="Geological Modeling" />
              <Tab label="Geochemical Analysis" />
              <Tab label="Deformation Analysis" />
              <Tab label="Correlation Matrix" />
              <Tab label="Workflow Manager" />
            </Tabs>

            <TabPanel value={activeTab} index={0}>
              {/* Integrated view with all selected modules */}
              <Grid container spacing={2}>
                {selectedModules.map((moduleId) => {
                  const module = analysisModules.find(m => m.id === moduleId);
                  if (!module) return null;
                  
                  const Component = module.component;
                  return (
                    <Grid item xs={12} md={6} key={moduleId}>
                      <Card>
                        <CardContent>
                          <Typography variant="h6" gutterBottom>
                            {module.name}
                          </Typography>
                          <Component compact realTimeMode={realTimeMode} />
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
            </TabPanel>

            <TabPanel value={activeTab} index={1}>
              <SeismicWaveformVisualization realTimeMode={realTimeMode} />
            </TabPanel>

            <TabPanel value={activeTab} index={2}>
              <VolcanicMonitoringDashboard realTimeMode={realTimeMode} />
            </TabPanel>

            <TabPanel value={activeTab} index={3}>
              <GeologicalModelViewer />
            </TabPanel>

            <TabPanel value={activeTab} index={4}>
              <GeochemicalAnalysisPlots />
            </TabPanel>

            <TabPanel value={activeTab} index={5}>
              <DeformationAnalysisViewer realTimeMode={realTimeMode} />
            </TabPanel>

            <TabPanel value={activeTab} index={6}>
              <MultiParameterCorrelationMatrix 
                data={correlationMatrix}
                parameters={dataStreams.map(s => s.name)}
              />
            </TabPanel>

            <TabPanel value={activeTab} index={7}>
              <WorkflowManager
                workflows={activeWorkflows}
                onStartWorkflow={startWorkflow}
                availableModules={analysisModules}
              />
            </TabPanel>
          </Paper>
        </Box>
      </Box>

      {/* Settings Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="md" fullWidth>
        <DialogTitle>Analysis Interface Settings</DialogTitle>
        <DialogContent>
          <DataSynchronizationPanel
            dataStreams={dataStreams}
            onStreamUpdate={setDataStreams}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSettings(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Notifications */}
      {notifications.map((notification) => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={6000}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert severity={notification.type} onClose={() => {}}>
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </Box>
  );
};

export default UnifiedGeologicalAnalysisInterface;