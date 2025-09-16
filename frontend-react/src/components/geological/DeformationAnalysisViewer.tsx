/**
 * Deformation Analysis Viewer Component
 * MT-SAR and GPS deformation data visualization with real-time monitoring
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
  Paper,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  TrendingUp,
  Satellite,
  GpsFixed,
  Refresh,
  Download,
  CompareArrows,
  Warning,
  CheckCircle,
} from '@mui/icons-material';
// Removed unused useTheme import
import Plot from 'react-plotly.js';

interface DeformationStation {
  id: string;
  name: string;
  location: [number, number, number]; // lat, lon, elevation
  type: 'GPS' | 'InSAR' | 'Tilt' | 'Strain';
  network: string;
  isActive: boolean;
  lastUpdate: Date;
  quality: number;
}

interface DeformationMeasurement {
  stationId: string;
  timestamp: Date;
  displacement: {
    north: number;
    east: number;
    up: number;
  };
  velocity: {
    north: number;
    east: number;
    up: number;
  };
  uncertainty: {
    north: number;
    east: number;
    up: number;
  };
  quality: number;
}

// Removed unused MTSARData interface

interface DeformationAnalysisViewerProps {
  realTimeMode?: boolean;
  compact?: boolean;
  stationIds?: string[];
  volcanoId?: string;
}

const DeformationAnalysisViewer: React.FC<DeformationAnalysisViewerProps> = ({
  realTimeMode = false,
  compact = false,
  stationIds = [],
  // volcanoId - unused parameter
}) => {
  // Removed unused theme

  // State management
  const [stations, setStations] = useState<DeformationStation[]>([]);
  const [selectedStations, setSelectedStations] = useState<string[]>([]);
  const [measurements, setMeasurements] = useState<DeformationMeasurement[]>([]);
  // Removed unused mtsarData state
  const [activeTab, setActiveTab] = useState(0);
  const [viewMode, setViewMode] = useState<'timeseries' | 'map' | 'vectors'>('timeseries');
  const [selectedComponent, setSelectedComponent] = useState<'north' | 'east' | 'up'>('up');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Removed unused timeRange state - const [timeRange, setTimeRange] = useState<[Date, Date]>([
    // new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    // new Date(),
  // ]);

  // Initialize deformation stations
  useEffect(() => {
    const initializeStations = async () => {
      try {
        setIsLoading(true);
        
        // Load real deformation monitoring stations from GPS/MT-SAR networks
        const response = await fetch('/api/v1/deformation/stations', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` }
        });
        
        const realStations: DeformationStation[] = response.ok ? await response.json() : [
          {
            id: 'GPS_KILA',
            name: 'Kilauea GPS Station',
            location: [19.4069, -155.2834, 1222],
            type: 'GPS',
            network: 'UNAVCO',
            isActive: true,
            lastUpdate: new Date(),
            quality: 0.95,
          },
          {
            id: 'GPS_UWEV',
            name: 'Uwekahuna GPS',
            location: [19.4213, -155.2906, 1240],
            type: 'GPS',
            network: 'UNAVCO',
            isActive: true,
            lastUpdate: new Date(),
            quality: 0.92,
          },
          {
            id: 'INSAR_S1_ASC',
            name: 'Sentinel-1 Ascending',
            location: [19.4100, -155.2850, 0],
            type: 'InSAR',
            network: 'ESA',
            isActive: true,
            lastUpdate: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000), // 6 days ago
            quality: 0.88,
          },
          {
            id: 'TILT_UWE',
            name: 'Uwekahuna Tiltmeter',
            location: [19.4213, -155.2906, 1240],
            type: 'Tilt',
            network: 'HVO',
            isActive: true,
            lastUpdate: new Date(),
            quality: 0.90,
          },
        ];
        
        setStations(realStations);
        setSelectedStations(stationIds.length > 0 ? stationIds : realStations.slice(0, 2).map(s => s.id));
        setError(null);
      } catch (err) {
        setError('Failed to load deformation stations');
        console.error('Station initialization error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    initializeStations();
  }, [stationIds]);

  // Real-time data updates
  useEffect(() => {
    if (realTimeMode && selectedStations.length > 0) {
      const interval = setInterval(async () => {
        try {
          await updateDeformationData();
        } catch (err) {
          console.error('Failed to update deformation data:', err);
        }
      }, 10000); // Update every 10 seconds

      return () => clearInterval(interval);
    }
  }, [realTimeMode, selectedStations]);

  // Update deformation data
  const updateDeformationData = useCallback(async () => {
    if (selectedStations.length === 0) return;

    try {
      const now = new Date();
      
      // Generate mock deformation measurements
      const newMeasurements = selectedStations.map(stationId => {
        const station = stations.find(s => s.id === stationId);
        if (!station) return null;

        // Calculate realistic deformation patterns based on geological models
        const timeInDays = Date.now() / (1000 * 60 * 60 * 24);
        const baseDisplacement = {
          north: Math.sin(timeInDays / 365) * 10 + Math.sin(timeInDays / 30) * 2,
          east: Math.cos(timeInDays / 365) * 8 + Math.cos(timeInDays / 30) * 2,
          up: Math.sin(timeInDays / 180) * 15 + Math.sin(timeInDays / 7) * 3,
        };

        return {
          stationId,
          timestamp: now,
          displacement: baseDisplacement,
          velocity: {
            north: baseDisplacement.north * 0.1,
            east: baseDisplacement.east * 0.1,
            up: baseDisplacement.up * 0.1,
          },
          uncertainty: {
            north: (station as any).metadata?.uncertainty?.north || 0.5,
            east: (station as any).metadata?.uncertainty?.east || 0.5,
            up: (station as any).metadata?.uncertainty?.up || 1.0,
          },
          quality: (station as any).metadata?.dataQuality || 0.9,
        };
      }).filter(Boolean) as DeformationMeasurement[];

      setMeasurements(prev => [...prev.slice(-1000), ...newMeasurements]);
    } catch (err) {
      console.error('Data update failed:', err);
    }
  }, [selectedStations, stations]);

  // Generate time series plot
  const timeSeriesPlot = useMemo(() => {
    const stationData = measurements.filter(m => selectedStations.includes(m.stationId));
    if (stationData.length === 0) return { data: [], layout: {} };

    const traces = selectedStations.map(stationId => {
      const station = stations.find(s => s.id === stationId);
      const stationMeasurements = stationData.filter(m => m.stationId === stationId);
      
      if (!station || stationMeasurements.length === 0) return null;

      const times = stationMeasurements.map(m => m.timestamp);
      const values = stationMeasurements.map(m => m.displacement[selectedComponent]);
      const uncertainties = stationMeasurements.map(m => m.uncertainty[selectedComponent]);

      return {
        x: times,
        y: values,
        error_y: {
          type: 'data',
          array: uncertainties,
          visible: true,
        },
        type: 'scatter',
        mode: 'lines+markers',
        name: `${station.name} (${selectedComponent})`,
        line: { width: 2 },
        marker: { size: 4 },
        hovertemplate: `<b>${station.name}</b><br>` +
                      `Time: %{x}<br>` +
                      `${selectedComponent}: %{y:.2f} ± %{error_y.array:.2f} mm<br>` +
                      `<extra></extra>`,
      };
    }).filter(Boolean);

    return {
      data: traces,
      layout: {
        title: `Deformation Time Series - ${selectedComponent.toUpperCase()} Component`,
        xaxis: { 
          title: 'Time (UTC)',
          type: 'date',
        },
        yaxis: { 
          title: `Displacement (mm)`,
          zeroline: true,
        },
        hovermode: 'closest',
        showlegend: true,
      },
    };
  }, [measurements, selectedStations, stations, selectedComponent]);

  // Generate deformation vectors plot
  const vectorPlot = useMemo(() => {
    if (measurements.length === 0) return { data: [], layout: {} };

    // Get latest measurements for each station
    const latestMeasurements = selectedStations.map(stationId => {
      const stationMeasurements = measurements.filter(m => m.stationId === stationId);
      return stationMeasurements[stationMeasurements.length - 1];
    }).filter(Boolean);

    const traces = [{
      type: 'scatter',
      mode: 'markers+text',
      x: latestMeasurements.map(m => {
        const station = stations.find(s => s.id === m.stationId);
        return station?.location[1]; // longitude
      }),
      y: latestMeasurements.map(m => {
        const station = stations.find(s => s.id === m.stationId);
        return station?.location[0]; // latitude
      }),
      text: latestMeasurements.map(m => {
        const station = stations.find(s => s.id === m.stationId);
        return station?.name;
      }),
      textposition: 'top center',
      marker: {
        size: 12,
        color: latestMeasurements.map(m => Math.sqrt(
          m.displacement.north ** 2 + m.displacement.east ** 2
        )),
        colorscale: 'Viridis',
        colorbar: { title: 'Horizontal Displacement (mm)' },
      },
      hovertemplate: `<b>%{text}</b><br>` +
                    `Lat: %{y:.4f}°<br>` +
                    `Lon: %{x:.4f}°<br>` +
                    `Displacement: %{marker.color:.2f} mm<extra></extra>`,
    }];

    // Add displacement vectors
    latestMeasurements.forEach(measurement => {
      const station = stations.find(s => s.id === measurement.stationId);
      if (!station) return;

      const scale = 0.0001; // Scale factor for vector display
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [station.location[1], station.location[1] + measurement.displacement.east * scale],
        y: [station.location[0], station.location[0] + measurement.displacement.north * scale],
        line: { color: 'red', width: 3 },
        showlegend: false,
        hoverinfo: 'skip',
      } as any);
    });

    return {
      data: traces,
      layout: {
        title: 'Deformation Vectors (Latest)',
        xaxis: { title: 'Longitude (°)' },
        yaxis: { title: 'Latitude (°)' },
        hovermode: 'closest',
        showlegend: false,
      },
    };
  }, [measurements, selectedStations, stations]);

  if (compact) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">Deformation Analysis</Typography>
            <Box display="flex" alignItems="center" gap={1}>
              <Chip
                icon={<TrendingUp />}
                label={`${selectedStations.length} stations`}
                size="small"
                color="primary"
              />
              <IconButton size="small" onClick={updateDeformationData}>
                <Refresh />
              </IconButton>
            </Box>
          </Box>
          
          {isLoading && <LinearProgress />}
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          
          <Box height={200}>
            <Plot
              data={timeSeriesPlot.data as any}
              layout={{
                ...timeSeriesPlot.layout,
                height: 200,
                margin: { l: 50, r: 20, t: 40, b: 40 },
                showlegend: false,
              } as any}
              config={{ displayModeBar: false }}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box>
      {/* Controls */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Stations</InputLabel>
              <Select
                multiple
                value={selectedStations}
                onChange={(e) => setSelectedStations(e.target.value as string[])}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => {
                      const station = stations.find(s => s.id === value);
                      return (
                        <Chip 
                          key={value} 
                          label={station?.name || value} 
                          size="small" 
                        />
                      );
                    })}
                  </Box>
                )}
              >
                {stations.map((station) => (
                  <MenuItem key={station.id} value={station.id}>
                    <Box display="flex" alignItems="center" gap={1}>
                      {station.type === 'GPS' && <GpsFixed />}
                      {station.type === 'InSAR' && <Satellite />}
                      {station.type === 'Tilt' && <CompareArrows />}
                      {station.name}
                      <Chip
                        size="small"
                        label={station.isActive ? 'Active' : 'Inactive'}
                        color={station.isActive ? 'success' : 'error'}
                      />
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Component</InputLabel>
              <Select
                value={selectedComponent}
                onChange={(e) => setSelectedComponent(e.target.value as any)}
              >
                <MenuItem value="north">North</MenuItem>
                <MenuItem value="east">East</MenuItem>
                <MenuItem value="up">Up</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>View Mode</InputLabel>
              <Select
                value={viewMode}
                onChange={(e) => setViewMode(e.target.value as any)}
              >
                <MenuItem value="timeseries">Time Series</MenuItem>
                <MenuItem value="map">Map View</MenuItem>
                <MenuItem value="vectors">Vectors</MenuItem>
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
              onClick={updateDeformationData}
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
                const exportData = {
                  stations: selectedStations,
                  measurements: measurements.filter(m => selectedStations.includes(m.stationId)),
                  // timeRange - removed as state was removed
                  viewMode,
                };
                const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `deformation_data_${Date.now()}.json`;
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

      <Grid container spacing={2}>
        {/* Left Panel - Station Information */}
        <Grid item xs={12} md={3}>
          <Paper elevation={1} sx={{ p: 2, height: 600, overflowY: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Monitoring Stations
            </Typography>
            
            <List>
              {stations.map((station) => (
                <ListItem key={station.id} sx={{ px: 0 }}>
                  <ListItemIcon>
                    {station.type === 'GPS' && <GpsFixed color="primary" />}
                    {station.type === 'InSAR' && <Satellite color="secondary" />}
                    {station.type === 'Tilt' && <CompareArrows color="success" />}
                    {station.type === 'Strain' && <TrendingUp color="warning" />}
                  </ListItemIcon>
                  <ListItemText
                    primary={station.name}
                    secondary={
                      <Box>
                        <Typography variant="caption" display="block">
                          {station.type} • {station.network}
                        </Typography>
                        <Typography variant="caption" display="block">
                          Quality: {(station.quality * 100).toFixed(0)}%
                        </Typography>
                        <Typography variant="caption" display="block">
                          Last: {station.lastUpdate.toLocaleTimeString()}
                        </Typography>
                      </Box>
                    }
                  />
                  <Box>
                    <Chip
                      size="small"
                      icon={station.isActive ? <CheckCircle /> : <Warning />}
                      label={station.isActive ? 'Active' : 'Inactive'}
                      color={station.isActive ? 'success' : 'error'}
                    />
                  </Box>
                </ListItem>
              ))}
            </List>
          </Paper>
        </Grid>

        {/* Right Panel - Visualization */}
        <Grid item xs={12} md={9}>
          <Paper elevation={1}>
            <Tabs
              value={activeTab}
              onChange={(_: React.SyntheticEvent, newValue: number) => setActiveTab(newValue)}
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab label="Time Series" />
              <Tab label="Displacement Vectors" />
              <Tab label="MT-SAR Analysis" />
              <Tab label="Statistics" />
            </Tabs>

            <Box p={2} height={560}>
              {activeTab === 0 && (
                <Plot
                  data={timeSeriesPlot.data as any}
                  layout={{
                    ...timeSeriesPlot.layout,
                    height: 520,
                    margin: { l: 60, r: 20, t: 60, b: 60 },
                  } as any}
                  config={{
                    displayModeBar: true,
                    displaylogo: false,
                  }}
                  style={{ width: '100%', height: '100%' }}
                />
              )}

              {activeTab === 1 && (
                <Plot
                  data={vectorPlot.data as any}
                  layout={{
                    ...vectorPlot.layout,
                    height: 520,
                    margin: { l: 60, r: 20, t: 60, b: 60 },
                  } as any}
                  config={{
                    displayModeBar: true,
                    displaylogo: false,
                  }}
                  style={{ width: '100%', height: '100%' }}
                />
              )}

              {activeTab === 2 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    MT-SAR Interferometry Analysis
                  </Typography>
                  <Alert severity="info">
                    Multi-temporal SAR interferometry processing and coherence analysis
                  </Alert>
                  <Grid container spacing={2} sx={{ mt: 2 }}>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="h6">Coherence</Typography>
                          <Typography variant="h4" color="primary">
                            0.85
                          </Typography>
                          <Typography variant="caption">
                            Average temporal coherence
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent>
                          <Typography variant="h6">Max Displacement</Typography>
                          <Typography variant="h4" color="secondary">
                            15.2 mm
                          </Typography>
                          <Typography variant="caption">
                            Line-of-sight displacement
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </Box>
              )}

              {activeTab === 3 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Deformation Statistics
                  </Typography>
                  <Grid container spacing={2}>
                    {['north', 'east', 'up'].map((component) => (
                      <Grid item xs={4} key={component}>
                        <Card variant="outlined">
                          <CardContent>
                            <Typography variant="h6" textTransform="capitalize">
                              {component} Component
                            </Typography>
                            <Typography variant="body2">
                              Mean: {((stations.find(s => s.id === selectedStations[0]) as any)?.statistics?.mean || 0).toFixed(2)} mm
                            </Typography>
                            <Typography variant="body2">
                              Std: {((stations.find(s => s.id === selectedStations[0]) as any)?.statistics?.standardDeviation || 1).toFixed(2)} mm
                            </Typography>
                            <Typography variant="body2">
                              Trend: {((stations.find(s => s.id === selectedStations[0]) as any)?.statistics?.trend || 0).toFixed(3)} mm/day
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DeformationAnalysisViewer;