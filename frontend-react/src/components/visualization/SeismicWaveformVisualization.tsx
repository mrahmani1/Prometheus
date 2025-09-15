/**
 * Seismic Waveform Visualization Component
 * Interactive waveform display with real-time IRIS data integration
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
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
  Slider,
  Switch,
  FormControlLabel,
  Alert,
  LinearProgress,
  Paper,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Warning,
  CheckCircle,
  Refresh,
  Download,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import Plot from 'react-plotly.js';

interface SeismicStation {
  id: string;
  name: string;
  network: string;
  location: [number, number, number]; // lat, lon, elevation
  channels: string[];
  sampleRate: number;
  isActive: boolean;
  quality: number;
}

interface SeismicEvent {
  id: string;
  timestamp: Date;
  magnitude: number;
  depth: number;
  location: [number, number];
  phases: PhaseArrival[];
  quality: string;
}

interface PhaseArrival {
  phase: string; // P, S, Pg, Sg, etc.
  arrivalTime: Date;
  uncertainty: number;
  amplitude: number;
  period: number;
  quality: string;
}

interface WaveformData {
  stationId: string;
  channel: string;
  startTime: Date;
  sampleRate: number;
  data: number[];
  units: string;
}

interface FilterSettings {
  lowpass: number;
  highpass: number;
  order: number;
  enabled: boolean;
}

interface SeismicWaveformVisualizationProps {
  realTimeMode?: boolean;
  compact?: boolean;
  stationIds?: string[];
  timeWindow?: number; // minutes
}

const SeismicWaveformVisualization: React.FC<SeismicWaveformVisualizationProps> = ({
  realTimeMode = false,
  compact = false,
  stationIds = [],
  timeWindow = 60,
}) => {
  const theme = useTheme();
  const plotRef = useRef<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // State management
  const [stations, setStations] = useState<SeismicStation[]>([]);
  const [selectedStations, setSelectedStations] = useState<string[]>([]);
  const [waveformData, setWaveformData] = useState<WaveformData[]>([]);
  const [events, setEvents] = useState<SeismicEvent[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  // Removed unused currentTime state
  // Removed unused timeRange state - const [timeRange, setTimeRange] = useState<[Date, Date]>([
    // new Date(Date.now() - timeWindow * 60 * 1000),
    // new Date(),
  // ]);
  const [filterSettings, setFilterSettings] = useState<FilterSettings>({
    lowpass: 10,
    highpass: 1,
    order: 4,
    enabled: true,
  });
  const [amplitudeScale, setAmplitudeScale] = useState(1);
  const [showPhases, setShowPhases] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize stations
  useEffect(() => {
    const initializeStations = async () => {
      try {
        setIsLoading(true);
        
        // Fetch real seismic stations from IRIS network
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/seismic/stations/iris`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
            'Content-Type': 'application/json',
          },
        });

        if (!response.ok) {
          throw new Error('Failed to fetch IRIS seismic stations');
        }

        const data = await response.json();
        const realStations: SeismicStation[] = data.stations.map((station: any) => ({
          id: station.id,
          name: station.name,
          network: station.network,
          location: [station.latitude, station.longitude, station.elevation],
          channels: station.channels,
          sampleRate: station.sampleRate,
          isActive: station.isActive,
          quality: station.quality,
          dataSource: 'IRIS/EarthScope Network',
          instrumentType: station.instrumentType,
          startDate: new Date(station.startDate),
          endDate: station.endDate ? new Date(station.endDate) : undefined,
        }));
        
        setStations(realStations);
        setSelectedStations(stationIds.length > 0 ? stationIds : [realStations[0]?.id].filter(Boolean));
        setError(null);
        
        console.log(`Loaded ${realStations.length} real seismic stations from IRIS network`);
        
      } catch (err) {
        console.error('Failed to load stations from API, trying direct IRIS connection:', err);
        
        // Fallback: Load directly from RealDataService
        try {
          const realDataService = (await import('../../services/RealDataService')).default.getInstance();
          
          const realStations = await realDataService.fetchSeismicStations({
            minLatitude: 25.0,
            maxLatitude: 50.0,
            minLongitude: -130.0,
            maxLongitude: -65.0, // Continental US
          });

          const fallbackStations: SeismicStation[] = realStations.slice(0, 20).map(station => ({
            id: station.id,
            name: `${station.station} - ${station.network}`,
            network: station.network,
            location: [station.latitude, station.longitude, station.elevation],
            channels: [station.channel],
            sampleRate: station.sampleRate,
            isActive: !station.endDate || station.endDate > new Date(),
            quality: station.dataQuality,
            dataSource: 'IRIS/EarthScope Network (Direct)',
            instrumentType: station.instrumentType,
            startDate: station.startDate,
            endDate: station.endDate,
          }));
          
          setStations(fallbackStations);
          setSelectedStations(stationIds.length > 0 ? stationIds : [fallbackStations[0]?.id].filter(Boolean));
          setError(null);
          
          console.log(`Loaded ${fallbackStations.length} seismic stations from direct IRIS connection`);
          
        } catch (fallbackError) {
          console.error('Failed to load stations from fallback source:', fallbackError);
          setError('Failed to load seismic stations from all data sources');
          setStations([]);
        }
      } finally {
        setIsLoading(false);
      }
    };

    initializeStations();
  }, [stationIds]);

  // Real-time seismic data updates from IRIS network
  useEffect(() => {
    if (realTimeMode && selectedStations.length > 0) {
      // Connect to production IRIS real-time seismic data stream
      const connectToIRISStream = async () => {
        try {
          // First, try to connect through backend API WebSocket proxy
          const wsUrl = `${process.env.REACT_APP_WS_URL?.replace('http', 'ws') || 'ws://localhost:8000'}/ws/iris/real-time`;
          wsRef.current = new WebSocket(wsUrl);

          wsRef.current.onopen = () => {
            console.log('Connected to IRIS real-time seismic data stream');
            
            // Subscribe to selected stations with authentication
            wsRef.current?.send(JSON.stringify({
              type: 'subscribe_iris',
              stations: selectedStations,
              channels: ['BHZ', 'BHN', 'BHE'],
              format: 'miniseed',
              quality: 'B', // Best quality
              auth_token: localStorage.getItem('auth_token'),
            }));
          };

          wsRef.current.onmessage = (event) => {
            try {
              const data = JSON.parse(event.data);
              
              if (data.type === 'iris_waveform') {
                handleRealTimeWaveformData(data);
              } else if (data.type === 'iris_error') {
                console.error('IRIS stream error:', data.message);
                setError(`IRIS connection error: ${data.message}`);
              } else if (data.type === 'iris_status') {
                console.log('IRIS stream status:', data.status);
              }
            } catch (err) {
              console.error('Failed to parse IRIS seismic data:', err);
            }
          };

          wsRef.current.onerror = (error) => {
            console.error('IRIS WebSocket error:', error);
            setError('Real-time IRIS connection failed');
          };

          wsRef.current.onclose = () => {
            console.log('IRIS real-time stream disconnected');
          };

        } catch (error) {
          console.error('Failed to connect to IRIS stream:', error);
          setError('Unable to connect to IRIS real-time data stream');
        }
      };

      connectToIRISStream();

      return () => {
        if (wsRef.current) {
          // Send unsubscribe message before closing
          wsRef.current.send(JSON.stringify({
            type: 'unsubscribe_iris',
            stations: selectedStations,
          }));
          wsRef.current.close();
        }
      };
    }
  }, [realTimeMode, selectedStations]);

  // Handle real-time waveform data
  const handleRealTimeWaveformData = useCallback((data: any) => {
    if (data.type === 'waveform') {
      setWaveformData(prev => {
        const updated = [...prev];
        const existingIndex = updated.findIndex(
          w => w.stationId === data.stationId && w.channel === data.channel
        );

        const newWaveform: WaveformData = {
          stationId: data.stationId,
          channel: data.channel,
          startTime: new Date(data.startTime),
          sampleRate: data.sampleRate,
          data: data.samples,
          units: data.units || 'counts',
        };

        if (existingIndex >= 0) {
          updated[existingIndex] = newWaveform;
        } else {
          updated.push(newWaveform);
        }

        return updated;
      });
    } else if (data.type === 'event') {
      const newEvent: SeismicEvent = {
        id: data.eventId,
        timestamp: new Date(data.timestamp),
        magnitude: data.magnitude,
        depth: data.depth,
        location: [data.latitude, data.longitude],
        phases: data.phases || [],
        quality: data.quality || 'automatic',
      };
      setEvents(prev => [newEvent, ...prev.slice(0, 99)]); // Keep last 100 events
    }
  }, []);

  // Generate plot data
  const plotData = useMemo(() => {
    const traces: any[] = [];
    const annotations: any[] = [];

    waveformData.forEach((waveform, index) => {
      const station = stations.find(s => s.id === waveform.stationId);
      if (!station) return;

      // Apply filtering if enabled
      let processedData = waveform.data;
      if (filterSettings.enabled) {
        // Simple high-pass filter simulation
        processedData = waveform.data.map((value, i) => {
          if (i === 0) return value;
          return value - 0.95 * waveform.data[i - 1];
        });
      }

      // Scale amplitude
      processedData = processedData.map(value => value * amplitudeScale);

      // Create time array
      const timeArray = processedData.map((_, i) => 
        new Date(waveform.startTime.getTime() + (i / waveform.sampleRate) * 1000)
      );

      // Offset traces vertically for multi-station display
      const offset = index * 1000;
      const offsetData = processedData.map(value => value + offset);

      traces.push({
        x: timeArray,
        y: offsetData,
        type: 'scatter',
        mode: 'lines',
        name: `${station.name} (${waveform.channel})`,
        line: {
          color: theme.palette.primary.main,
          width: 1,
        },
        hovertemplate: `<b>${station.name}</b><br>` +
                      `Channel: ${waveform.channel}<br>` +
                      `Time: %{x}<br>` +
                      `Amplitude: %{y:.2f} ${waveform.units}<extra></extra>`,
      });

      // Add station label
      annotations.push({
        x: timeArray[0],
        y: offset,
        text: `${station.name}<br>${waveform.channel}`,
        showarrow: false,
        xanchor: 'right',
        yanchor: 'middle',
        font: { size: 10 },
      });
    });

    // Add phase arrivals if enabled
    if (showPhases) {
      events.forEach(event => {
        event.phases.forEach(phase => {
          traces.push({
            x: [phase.arrivalTime, phase.arrivalTime],
            y: [-2000, 2000 * waveformData.length],
            type: 'scatter',
            mode: 'lines',
            name: `${phase.phase} arrival`,
            line: {
              color: phase.phase.startsWith('P') ? 'red' : 'blue',
              width: 2,
              dash: 'dash',
            },
            showlegend: false,
            hovertemplate: `<b>${phase.phase} Phase</b><br>` +
                          `Arrival: %{x}<br>` +
                          `Magnitude: ${event.magnitude}<br>` +
                          `Depth: ${event.depth} km<extra></extra>`,
          });
        });
      });
    }

    return { traces, annotations };
  }, [waveformData, stations, filterSettings, amplitudeScale, showPhases, events, theme]);

  // Control functions
  const handlePlay = useCallback(() => {
    setIsPlaying(true);
    // if (realTimeMode) {
    //   setCurrentTime(new Date()); - removed as state was removed
    // }
  }, [realTimeMode]);

  const handlePause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const handleStop = useCallback(() => {
    setIsPlaying(false);
    // setCurrentTime(new Date(Date.now() - timeWindow * 60 * 1000)); - removed as state was removed
  }, [timeWindow]);

  const handleRefresh = useCallback(async () => {
    setIsLoading(true);
    try {
      // Refresh real seismic data from IRIS network
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || '/api/v1'}/seismic/waveforms/refresh`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stations: selectedStations,
          timeWindow: timeWindow,
          channels: ['BHZ', 'BHN', 'BHE'],
          format: 'miniseed',
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to refresh seismic data from IRIS');
      }

      const refreshedData = await response.json();
      
      // Update waveform data with fresh IRIS data
      if (refreshedData.waveforms) {
        setWaveformData(refreshedData.waveforms);
      }
      
      // Update events if available
      if (refreshedData.events) {
        setEvents(refreshedData.events);
      }
      
      setError(null);
      console.log(`Refreshed seismic data for ${selectedStations.length} stations from IRIS network`);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to refresh seismic data';
      console.error('Data refresh error:', errorMessage);
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [selectedStations, timeWindow]);

  const handleExport = useCallback(() => {
    const exportData = {
      stations: selectedStations,
      // timeRange - removed as state was removed
      waveformData,
      events,
      filterSettings,
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `seismic_data_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [selectedStations, waveformData, events, filterSettings]);

  if (compact) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">Seismic Waveforms</Typography>
            <Box>
              <IconButton size="small" onClick={handleRefresh} disabled={isLoading}>
                <Refresh />
              </IconButton>
              <IconButton size="small" onClick={isPlaying ? handlePause : handlePlay}>
                {isPlaying ? <Pause /> : <PlayArrow />}
              </IconButton>
            </Box>
          </Box>
          
          {isLoading && <LinearProgress />}
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          
          <Box height={200}>
            <Plot
              data={plotData.traces}
              layout={{
                height: 200,
                margin: { l: 50, r: 20, t: 20, b: 40 },
                xaxis: { title: 'Time' },
                yaxis: { title: 'Amplitude' },
                showlegend: false,
                annotations: plotData.annotations,
              }}
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
          <Grid item>
            <Box display="flex" gap={1}>
              <Button
                variant={isPlaying ? "outlined" : "contained"}
                startIcon={<PlayArrow />}
                onClick={handlePlay}
                disabled={isLoading}
              >
                Play
              </Button>
              <Button
                variant="outlined"
                startIcon={<Pause />}
                onClick={handlePause}
                disabled={!isPlaying}
              >
                Pause
              </Button>
              <Button
                variant="outlined"
                startIcon={<Stop />}
                onClick={handleStop}
              >
                Stop
              </Button>
            </Box>
          </Grid>

          <Grid item>
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Stations</InputLabel>
              <Select
                multiple
                value={selectedStations}
                onChange={(e) => setSelectedStations(e.target.value as string[])}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {stations.map((station) => (
                  <MenuItem key={station.id} value={station.id}>
                    {station.name} ({station.id})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>

          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={filterSettings.enabled}
                  onChange={(e) => setFilterSettings(prev => ({ ...prev, enabled: e.target.checked }))}
                />
              }
              label="Filter"
            />
          </Grid>

          <Grid item>
            <Box sx={{ width: 150 }}>
              <Typography variant="caption">Amplitude Scale</Typography>
              <Slider
                value={amplitudeScale}
                onChange={(_: Event, value: number | number[]) => setAmplitudeScale(value as number)}
                min={0.1}
                max={10}
                step={0.1}
                size="small"
              />
            </Box>
          </Grid>

          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={showPhases}
                  onChange={(e) => setShowPhases(e.target.checked)}
                />
              }
              label="Show Phases"
            />
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={handleExport}
            >
              Export
            </Button>
          </Grid>

          <Grid item>
            <IconButton onClick={handleRefresh} disabled={isLoading}>
              <Refresh />
            </IconButton>
          </Grid>
        </Grid>
      </Paper>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Main plot */}
      <Paper elevation={1}>
        <Box height={600} p={2}>
          <Plot
            ref={plotRef}
            data={plotData.traces}
            layout={{
              height: 560,
              margin: { l: 80, r: 20, t: 40, b: 60 },
              title: 'Seismic Waveforms',
              xaxis: {
                title: 'Time (UTC)',
                type: 'date',
                range: [new Date(Date.now() - 60 * 60 * 1000), new Date()],
              },
              yaxis: {
                title: 'Amplitude + Offset',
                zeroline: false,
              },
              showlegend: true,
              legend: { x: 1, y: 1 },
              annotations: plotData.annotations,
              hovermode: 'closest',
            }}
            config={{
              displayModeBar: true,
              modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
              displaylogo: false,
            }}
            style={{ width: '100%', height: '100%' }}
          />
        </Box>
      </Paper>

      {/* Station status */}
      <Paper elevation={1} sx={{ mt: 2, p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Station Status
        </Typography>
        <Grid container spacing={2}>
          {stations.filter(s => selectedStations.includes(s.id)).map((station) => (
            <Grid item xs={12} md={4} key={station.id}>
              <Card variant="outlined">
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="between">
                    <Box>
                      <Typography variant="subtitle2">{station.name}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {station.id} • {station.network}
                      </Typography>
                    </Box>
                    <Chip
                      icon={station.isActive ? <CheckCircle /> : <Warning />}
                      label={station.isActive ? 'Active' : 'Inactive'}
                      color={station.isActive ? 'success' : 'warning'}
                      size="small"
                    />
                  </Box>
                  <Box mt={1}>
                    <Typography variant="caption">
                      Quality: {(station.quality * 100).toFixed(0)}%
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={station.quality * 100}
                      sx={{ mt: 0.5 }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>
    </Box>
  );
};

export default SeismicWaveformVisualization;