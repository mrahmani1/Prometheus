/**
 * Volcanic Monitoring Dashboard Component
 * Real-time volcanic monitoring with USGS data integration and MT-SAR analysis
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Button,
  IconButton,
  Chip,
  Alert,
  LinearProgress,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Badge,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Terrain,
  Warning,
  CheckCircle,
  Refresh,
  Download,
  Notifications,
  Error as ErrorIcon,
  Info,
  LocationOn,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import Plot from 'react-plotly.js';

interface VolcanicSystem {
  id: string;
  name: string;
  location: [number, number];
  elevation: number;
  volcanoType: string;
  tectonicSetting: string;
  alertLevel: number; // 0-5
  lastEruption?: Date;
  isMonitored: boolean;
  observatories: string[];
}

interface VolcanicAlert {
  id: string;
  volcanoId: string;
  level: number;
  type: 'seismic' | 'deformation' | 'gas' | 'thermal' | 'composite';
  timestamp: Date;
  description: string;
  confidence: number;
  duration?: number;
}

interface DeformationData {
  volcanoId: string;
  stationId: string;
  timestamp: Date;
  displacement: {
    north: number;
    east: number;
    up: number;
  };
  uncertainty: {
    north: number;
    east: number;
    up: number;
  };
  source: 'GPS' | 'InSAR' | 'Tilt' | 'Strain';
  dataQuality?: number;
}

interface GasEmissionData {
  volcanoId: string;
  timestamp: Date;
  so2: number; // tons/day
  co2: number; // tons/day
  h2s: number; // ppm
  temperature: number; // Celsius
  windSpeed: number; // m/s
  windDirection: number; // degrees
  stationId?: string;
  instrumentType?: string;
  dataQuality?: number;
}

interface ThermalData {
  volcanoId: string;
  timestamp: Date;
  temperature: number; // Kelvin
  radiantPower: number; // MW
  pixelCount: number;
  satellite: string;
  confidence: number;
  instrument?: string;
  viewAngle?: number;
  cloudCover?: number;
}

interface VolcanicMonitoringDashboardProps {
  realTimeMode?: boolean;
  compact?: boolean;
  volcanoIds?: string[];
}

const VolcanicMonitoringDashboard: React.FC<VolcanicMonitoringDashboardProps> = ({
  realTimeMode = false,
  compact = false,
  volcanoIds = [],
}) => {
  const theme = useTheme();

  // State management
  const [volcanoes, setVolcanoes] = useState<VolcanicSystem[]>([]);
  const [selectedVolcano, setSelectedVolcano] = useState<string>('');
  const [alerts, setAlerts] = useState<VolcanicAlert[]>([]);
  const [deformationData, setDeformationData] = useState<DeformationData[]>([]);
  const [gasData, setGasData] = useState<GasEmissionData[]>([]);
  const [thermalData, setThermalData] = useState<ThermalData[]>([]);
  const [activeTab, setActiveTab] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAlertDialog, setShowAlertDialog] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<VolcanicAlert | null>(null);

  // Initialize volcanic systems
  useEffect(() => {
    const initializeVolcanoes = async () => {
      try {
        setIsLoading(true);
        
        // Fetch real volcanic systems from production API
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/volcanoes/systems`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          throw new Error('Failed to fetch volcanic systems from API');
        }

        const data = await response.json();
        const realVolcanoes: VolcanicSystem[] = data.volcanoes.map((volcano: any) => ({
          id: volcano.id,
          name: volcano.name,
          location: [volcano.latitude, volcano.longitude],
          elevation: volcano.elevation,
          volcanoType: volcano.volcanoType,
          tectonicSetting: volcano.tectonicSetting,
          alertLevel: volcano.alertLevel,
          lastEruption: volcano.lastEruption ? new Date(volcano.lastEruption) : undefined,
          isMonitored: volcano.isMonitored,
          observatories: volcano.observatories || [],
          dataSource: 'Global Volcanism Program',
          gvpId: volcano.gvpId,
          population: volcano.population,
          hazards: volcano.hazards,
        }));
        
        setVolcanoes(realVolcanoes);
        setSelectedVolcano(volcanoIds.length > 0 ? volcanoIds[0] : realVolcanoes[0]?.id);
        setError(null);
        
        console.log(`Loaded ${realVolcanoes.length} real volcanic systems from GVP/USGS`);
        
      } catch (err) {
        console.error('Failed to load volcanic systems from API, trying direct data source:', err);
        
        // Fallback: Load directly from RealDataService
        try {
          const realDataService = (await import('../../services/RealDataService')).default.getInstance();
          
          const realVolcanicSystems = await realDataService.fetchVolcanicSystems({
            minLatitude: 18.0,
            maxLatitude: 72.0,
            minLongitude: -180.0,
            maxLongitude: -60.0, // Focus on North America and Pacific
          });

          const fallbackVolcanoes: VolcanicSystem[] = realVolcanicSystems.map(volcano => ({
            id: volcano.id,
            name: volcano.name,
            location: [volcano.latitude, volcano.longitude],
            elevation: volcano.elevation,
            volcanoType: volcano.volcanoType,
            tectonicSetting: volcano.tectonicSetting,
            alertLevel: volcano.alertLevel === 'red' ? 4 : volcano.alertLevel === 'orange' ? 3 : volcano.alertLevel === 'yellow' ? 2 : 1,
            lastEruption: volcano.lastEruption,
            isMonitored: volcano.monitoringStatus !== 'unmonitored',
            observatories: volcano.observatories,
            dataSource: 'Global Volcanism Program (Direct)',
            population: volcano.population,
            hazards: volcano.hazards,
          }));
          
          setVolcanoes(fallbackVolcanoes);
          setSelectedVolcano(volcanoIds.length > 0 ? volcanoIds[0] : fallbackVolcanoes[0]?.id);
          setError(null);
          
          console.log(`Loaded ${fallbackVolcanoes.length} volcanic systems from direct GVP connection`);
          
        } catch (fallbackError) {
          console.error('Failed to load volcanic systems from fallback source:', fallbackError);
          setError('Failed to load volcanic systems from all data sources');
          setVolcanoes([]);
        }
      } finally {
        setIsLoading(false);
      }
    };

    initializeVolcanoes();
  }, [volcanoIds]);

  // Real-time data updates
  useEffect(() => {
    if (realTimeMode && selectedVolcano) {
      const interval = setInterval(async () => {
        try {
          await updateVolcanicData();
        } catch (err) {
          console.error('Failed to update volcanic data:', err);
        }
      }, 5000); // Update every 5 seconds

      return () => clearInterval(interval);
    }
  }, [realTimeMode, selectedVolcano]);

  // Update volcanic data from real monitoring sources
  const updateVolcanicData = useCallback(async () => {
    if (!selectedVolcano) return;

    try {
      // Fetch real-time volcanic monitoring data from backend
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/volcanoes/${selectedVolcano}/real-time`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch real-time volcanic data');
      }

      const realTimeData = await response.json();
      
      // Process real deformation data from GPS/MT-SAR
      if (realTimeData.deformation) {
        const newDeformation: DeformationData = {
          volcanoId: selectedVolcano,
          stationId: realTimeData.deformation.stationId,
          timestamp: new Date(realTimeData.deformation.timestamp),
          displacement: {
            north: realTimeData.deformation.displacement.north,
            east: realTimeData.deformation.displacement.east,
            up: realTimeData.deformation.displacement.up,
          },
          uncertainty: {
            north: realTimeData.deformation.uncertainty.north,
            east: realTimeData.deformation.uncertainty.east,
            up: realTimeData.deformation.uncertainty.up,
          },
          source: realTimeData.deformation.source, // GPS, MT-SAR, etc.
          dataQuality: realTimeData.deformation.dataQuality,
        };
        
        setDeformationData(prev => [...prev.slice(-99), newDeformation]);
      }

      // Process real gas emission data from observatory sensors
      if (realTimeData.gasEmissions) {
        const newGasData: GasEmissionData = {
          volcanoId: selectedVolcano,
          timestamp: new Date(realTimeData.gasEmissions.timestamp),
          so2: realTimeData.gasEmissions.so2,
          co2: realTimeData.gasEmissions.co2,
          h2s: realTimeData.gasEmissions.h2s,
          temperature: realTimeData.gasEmissions.temperature,
          windSpeed: realTimeData.gasEmissions.windSpeed,
          windDirection: realTimeData.gasEmissions.windDirection,
          stationId: realTimeData.gasEmissions.stationId,
          instrumentType: realTimeData.gasEmissions.instrumentType,
          dataQuality: realTimeData.gasEmissions.dataQuality,
        };
        
        setGasData(prev => [...prev.slice(-99), newGasData]);
      }

      // Process real thermal data from NASA FIRMS/MODIS
      if (realTimeData.thermal) {
        const newThermalData: ThermalData = {
          volcanoId: selectedVolcano,
          timestamp: new Date(realTimeData.thermal.timestamp),
          temperature: realTimeData.thermal.temperature,
          radiantPower: realTimeData.thermal.radiantPower,
          pixelCount: realTimeData.thermal.pixelCount,
          satellite: realTimeData.thermal.satellite,
          confidence: realTimeData.thermal.confidence,
          instrument: realTimeData.thermal.instrument,
          viewAngle: realTimeData.thermal.viewAngle,
          cloudCover: realTimeData.thermal.cloudCover,
        };
        
        setThermalData(prev => [...prev.slice(-99), newThermalData]);
      }

      // Check for alerts based on real-time data
      if (realTimeData.gasEmissions && realTimeData.gasEmissions.so2 > 800) {
        const alert: VolcanicAlert = {
          id: `alert_${Date.now()}`,
          volcanoId: selectedVolcano,
          level: 3,
          type: 'gas',
          timestamp: new Date(),
          description: `Elevated SO2 emissions: ${realTimeData.gasEmissions.so2.toFixed(0)} tons/day`,
          confidence: 0.85,
        };
        setAlerts(prev => [alert, ...prev.slice(0, 19)]);
      }
      
      if (realTimeData.thermal && realTimeData.thermal.temperature > 340) {
        const alert: VolcanicAlert = {
          id: `alert_${Date.now()}`,
          volcanoId: selectedVolcano,
          level: 2,
          type: 'thermal',
          timestamp: new Date(),
          description: `Thermal anomaly detected: ${realTimeData.thermal.temperature.toFixed(0)} K`,
          confidence: 0.85,
        };
        setAlerts(prev => [alert, ...prev.slice(0, 19)]);
      }
    } catch (err) {
      console.error('Data update failed:', err);
    }
  }, [selectedVolcano]);

  // Generate deformation plot
  const deformationPlot = useMemo(() => {
    const volcanoData = deformationData.filter(d => d.volcanoId === selectedVolcano);
    if (volcanoData.length === 0) return { data: [], layout: {} };

    const times = volcanoData.map(d => d.timestamp);
    
    return {
      data: [
        {
          x: times,
          y: volcanoData.map(d => d.displacement.north),
          name: 'North',
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: theme.palette.primary.main },
        },
        {
          x: times,
          y: volcanoData.map(d => d.displacement.east),
          name: 'East',
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: theme.palette.secondary.main },
        },
        {
          x: times,
          y: volcanoData.map(d => d.displacement.up),
          name: 'Up',
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: theme.palette.success.main },
        },
      ],
      layout: {
        title: 'Ground Deformation',
        xaxis: { title: 'Time' },
        yaxis: { title: 'Displacement (mm)' },
        height: 300,
        margin: { l: 50, r: 20, t: 40, b: 40 },
      },
    };
  }, [deformationData, selectedVolcano, theme]);

  // Generate gas emissions plot
  const gasPlot = useMemo(() => {
    const volcanoData = gasData.filter(d => d.volcanoId === selectedVolcano);
    if (volcanoData.length === 0) return { data: [], layout: {} };

    const times = volcanoData.map(d => d.timestamp);
    
    return {
      data: [
        {
          x: times,
          y: volcanoData.map(d => d.so2),
          name: 'SO₂',
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: theme.palette.warning.main },
          yaxis: 'y',
        },
        {
          x: times,
          y: volcanoData.map(d => d.co2),
          name: 'CO₂',
          type: 'scatter',
          mode: 'lines+markers',
          line: { color: theme.palette.info.main },
          yaxis: 'y2',
        },
      ],
      layout: {
        title: 'Gas Emissions',
        xaxis: { title: 'Time' },
        yaxis: { title: 'SO₂ (tons/day)', side: 'left' },
        yaxis2: { title: 'CO₂ (tons/day)', side: 'right', overlaying: 'y' },
        height: 300,
        margin: { l: 50, r: 50, t: 40, b: 40 },
      },
    };
  }, [gasData, selectedVolcano, theme]);

  // Alert level color mapping
  const getAlertColor = (level: number) => {
    switch (level) {
      case 0: return 'success';
      case 1: return 'info';
      case 2: return 'warning';
      case 3: return 'error';
      case 4: return 'error';
      case 5: return 'error';
      default: return 'default';
    }
  };

  const getAlertLabel = (level: number) => {
    const labels = ['Normal', 'Advisory', 'Watch', 'Warning', 'Critical', 'Emergency'];
    return labels[level] || 'Unknown';
  };

  if (compact) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">Volcanic Monitoring</Typography>
            <Box display="flex" alignItems="center" gap={1}>
              {selectedVolcano && (
                <Chip
                  icon={<Terrain />}
                  label={volcanoes.find(v => v.id === selectedVolcano)?.name}
                  color={getAlertColor(volcanoes.find(v => v.id === selectedVolcano)?.alertLevel || 0)}
                  size="small"
                />
              )}
              <IconButton size="small" onClick={updateVolcanicData}>
                <Refresh />
              </IconButton>
            </Box>
          </Box>
          
          {isLoading && <LinearProgress />}
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="caption">Recent Alerts</Typography>
              <Typography variant="h4" color="warning.main">
                {alerts.length}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption">Active Volcanoes</Typography>
              <Typography variant="h4" color="error.main">
                {volcanoes.filter(v => v.alertLevel > 1).length}
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Header Controls */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Volcano</InputLabel>
              <Select
                value={selectedVolcano}
                onChange={(e) => setSelectedVolcano(e.target.value)}
              >
                {volcanoes.map((volcano) => (
                  <MenuItem key={volcano.id} value={volcano.id}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Terrain />
                      {volcano.name}
                      <Chip
                        size="small"
                        label={getAlertLabel(volcano.alertLevel)}
                        color={getAlertColor(volcano.alertLevel)}
                      />
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={realTimeMode}
                  onChange={() => {
                    // This would be handled by parent component
                  }}
                />
              }
              label="Real-time"
            />
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={updateVolcanicData}
              disabled={isLoading}
            >
              Refresh
            </Button>
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={() => {
                // Export functionality
                const exportData = {
                  volcano: selectedVolcano,
                  deformation: deformationData,
                  gas: gasData,
                  thermal: thermalData,
                  alerts,
                };
                const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `volcanic_data_${selectedVolcano}_${Date.now()}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            >
              Export
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Main Content */}
      <Grid container spacing={2}>
        {/* Left Panel - Volcano Info and Alerts */}
        <Grid item xs={12} md={4}>
          {/* Volcano Information */}
          {selectedVolcano && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {volcanoes.find(v => v.id === selectedVolcano)?.name}
                </Typography>
                
                <Box display="flex" alignItems="center" gap={1} mb={2}>
                  <Chip
                    icon={<Warning />}
                    label={`Alert Level ${volcanoes.find(v => v.id === selectedVolcano)?.alertLevel}`}
                    color={getAlertColor(volcanoes.find(v => v.id === selectedVolcano)?.alertLevel || 0)}
                  />
                  <Chip
                    icon={volcanoes.find(v => v.id === selectedVolcano)?.isMonitored ? <CheckCircle /> : <ErrorIcon />}
                    label={volcanoes.find(v => v.id === selectedVolcano)?.isMonitored ? 'Monitored' : 'Not Monitored'}
                    color={volcanoes.find(v => v.id === selectedVolcano)?.isMonitored ? 'success' : 'error'}
                  />
                </Box>

                <List dense>
                  <ListItem>
                    <ListItemIcon><LocationOn /></ListItemIcon>
                    <ListItemText 
                      primary="Location" 
                      secondary={`${volcanoes.find(v => v.id === selectedVolcano)?.location[0].toFixed(4)}°, ${volcanoes.find(v => v.id === selectedVolcano)?.location[1].toFixed(4)}°`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Terrain /></ListItemIcon>
                    <ListItemText 
                      primary="Elevation" 
                      secondary={`${volcanoes.find(v => v.id === selectedVolcano)?.elevation} m`}
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Info /></ListItemIcon>
                    <ListItemText 
                      primary="Type" 
                      secondary={volcanoes.find(v => v.id === selectedVolcano)?.volcanoType}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          )}

          {/* Recent Alerts */}
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                <Typography variant="h6">Recent Alerts</Typography>
                <Badge badgeContent={alerts.length} color="error">
                  <Notifications />
                </Badge>
              </Box>
              
              <List>
                {alerts.slice(0, 5).map((alert) => (
                  <ListItem
                    key={alert.id}
                    button
                    onClick={() => {
                      setSelectedAlert(alert);
                      setShowAlertDialog(true);
                    }}
                  >
                    <ListItemIcon>
                      <Warning color={getAlertColor(alert.level)} />
                    </ListItemIcon>
                    <ListItemText
                      primary={alert.description}
                      secondary={alert.timestamp.toLocaleString()}
                    />
                    <Chip
                      size="small"
                      label={getAlertLabel(alert.level)}
                      color={getAlertColor(alert.level)}
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Right Panel - Data Visualization */}
        <Grid item xs={12} md={8}>
          <Paper elevation={1}>
            <Tabs
              value={activeTab}
              onChange={(_: React.SyntheticEvent, newValue: number) => setActiveTab(newValue)}
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab label="Deformation" />
              <Tab label="Gas Emissions" />
              <Tab label="Thermal" />
              <Tab label="Seismic" />
            </Tabs>

            <Box p={2}>
              {activeTab === 0 && (
                <Plot
                  data={deformationPlot.data as any}
                  layout={deformationPlot.layout as any}
                  config={{ displayModeBar: true, displaylogo: false }}
                  style={{ width: '100%' }}
                />
              )}

              {activeTab === 1 && (
                <Plot
                  data={gasPlot.data as any}
                  layout={gasPlot.layout as any}
                  config={{ displayModeBar: true, displaylogo: false }}
                  style={{ width: '100%' }}
                />
              )}

              {activeTab === 2 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Thermal Monitoring
                  </Typography>
                  <Grid container spacing={2}>
                    {thermalData.slice(-6).map((data, index) => (
                      <Grid item xs={6} md={4} key={index}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="caption">
                              {data.timestamp.toLocaleTimeString()}
                            </Typography>
                            <Typography variant="h6">
                              {data.temperature.toFixed(0)} K
                            </Typography>
                            <Typography variant="body2" color="textSecondary">
                              {data.radiantPower.toFixed(1)} MW
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}

              {activeTab === 3 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Seismic Activity
                  </Typography>
                  <Alert severity="info">
                    Seismic data integration with main seismic analysis module
                  </Alert>
                </Box>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Alert Detail Dialog */}
      <Dialog open={showAlertDialog} onClose={() => setShowAlertDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Alert Details</DialogTitle>
        <DialogContent>
          {selectedAlert && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {selectedAlert.description}
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="caption">Level</Typography>
                  <Typography variant="body1">
                    {getAlertLabel(selectedAlert.level)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption">Type</Typography>
                  <Typography variant="body1">
                    {selectedAlert.type}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption">Confidence</Typography>
                  <Typography variant="body1">
                    {(selectedAlert.confidence * 100).toFixed(0)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption">Time</Typography>
                  <Typography variant="body1">
                    {selectedAlert.timestamp.toLocaleString()}
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAlertDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default VolcanicMonitoringDashboard;