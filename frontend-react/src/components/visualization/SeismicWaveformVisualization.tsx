/**
 * Seismic Waveform Visualization Component
 * Real-time seismic data visualization with advanced signal processing
 */

import React, { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Button,
  IconButton,
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
  Chip,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Refresh,
  Download,
  Settings,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import Plot from 'react-plotly.js';

interface SeismicStation {
  id: string;
  network: string;
  station: string;
  location: string;
  channel: string;
  latitude: number;
  longitude: number;
  elevation: number;
  sampleRate: number;
  instrumentType: string;
}

interface WaveformData {
  stationId: string;
  station: string;
  data: number[];
  sampleRate: number;
  startTime: Date;
  endTime: Date;
  quality: number;
}

interface SeismicEvent {
  id: string;
  time: Date;
  latitude: number;
  longitude: number;
  depth: number;
  magnitude: number;
  phases?: Array<{
    phase: string;
    arrivalTime: number;
    station: string;
  }>;
}

interface FilterSettings {
  enabled: boolean;
  highPass: boolean;
  highPassFreq: number;
  lowPass: boolean;
  lowPassFreq: number;
  bandPass: boolean;
  bandPassLow: number;
  bandPassHigh: number;
  sampleRate: number;
}

interface SeismicWaveformVisualizationProps {
  realTimeMode?: boolean;
  compact?: boolean;
  stationIds?: string[];
  timeWindow?: {
    start: Date;
    end: Date;
  };
}

const SeismicWaveformVisualization: React.FC<SeismicWaveformVisualizationProps> = ({
  realTimeMode = false,
  compact = false,
  stationIds = [],
  timeWindow,
}) => {
  const theme = useTheme();
  const plotRef = useRef<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // State management
  const [stations, setStations] = useState<SeismicStation[]>([]);
  const [selectedStations, setSelectedStations] = useState<string[]>(stationIds);
  const [waveformData, setWaveformData] = useState<WaveformData[]>([]);
  const [events, setEvents] = useState<SeismicEvent[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [filterSettings, setFilterSettings] = useState<FilterSettings>({
    enabled: false,
    highPass: false,
    highPassFreq: 1.0,
    lowPass: false,
    lowPassFreq: 10.0,
    bandPass: false,
    bandPassLow: 1.0,
    bandPassHigh: 10.0,
    sampleRate: 100,
  });
  const [amplitudeScale, setAmplitudeScale] = useState(1.0);
  const [showPhases, setShowPhases] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Apply real digital signal processing filters
   */
  const applyDigitalFilter = useCallback((data: number[], settings: FilterSettings): number[] => {
    if (!settings.enabled) return data;

    let filteredData = [...data];

    // High-pass Butterworth filter implementation
    if (settings.highPass && settings.highPassFreq > 0) {
      filteredData = butterworthHighPass(filteredData, settings.highPassFreq, settings.sampleRate);
    }

    // Low-pass Butterworth filter implementation
    if (settings.lowPass && settings.lowPassFreq > 0) {
      filteredData = butterworthLowPass(filteredData, settings.lowPassFreq, settings.sampleRate);
    }

    // Band-pass filter
    if (settings.bandPass && settings.bandPassLow > 0 && settings.bandPassHigh > 0) {
      filteredData = butterworthBandPass(
        filteredData,
        settings.bandPassLow,
        settings.bandPassHigh,
        settings.sampleRate
      );
    }

    return filteredData;
  }, []);

  /**
   * Butterworth high-pass filter implementation
   */
  const butterworthHighPass = useCallback((data: number[], cutoffFreq: number, sampleRate: number): number[] => {
    const nyquist = sampleRate / 2;
    const normalizedCutoff = cutoffFreq / nyquist;

    // Second-order Butterworth coefficients
    const c = Math.tan(Math.PI * normalizedCutoff);
    const c2 = c * c;
    const sqrt2c = Math.sqrt(2) * c;
    const d = c2 + sqrt2c + 1;

    const b0 = 1 / d;
    const b1 = -2 / d;
    const b2 = 1 / d;
    const a1 = (2 * (c2 - 1)) / d;
    const a2 = (c2 - sqrt2c + 1) / d;

    return applyIIRFilter(data, [b0, b1, b2], [1, a1, a2]);
  }, []);

  /**
   * Butterworth low-pass filter implementation
   */
  const butterworthLowPass = useCallback((data: number[], cutoffFreq: number, sampleRate: number): number[] => {
    const nyquist = sampleRate / 2;
    const normalizedCutoff = cutoffFreq / nyquist;

    // Second-order Butterworth coefficients
    const c = 1 / Math.tan(Math.PI * normalizedCutoff);
    const c2 = c * c;
    const sqrt2c = Math.sqrt(2) * c;
    const d = c2 + sqrt2c + 1;

    const b0 = 1 / d;
    const b1 = 2 / d;
    const b2 = 1 / d;
    const a1 = (2 * (1 - c2)) / d;
    const a2 = (1 - sqrt2c + c2) / d;

    return applyIIRFilter(data, [b0, b1, b2], [1, a1, a2]);
  }, []);

  /**
   * Butterworth band-pass filter implementation
   */
  const butterworthBandPass = useCallback((data: number[], lowFreq: number, highFreq: number, sampleRate: number): number[] => {
    // Apply high-pass then low-pass
    const highPassed = butterworthHighPass(data, lowFreq, sampleRate);
    return butterworthLowPass(highPassed, highFreq, sampleRate);
  }, [butterworthHighPass, butterworthLowPass]);

  /**
   * Apply IIR (Infinite Impulse Response) filter
   */
  const applyIIRFilter = useCallback((data: number[], b: number[], a: number[]): number[] => {
    const filtered = new Array(data.length);
    const xHistory = new Array(b.length).fill(0);
    const yHistory = new Array(a.length).fill(0);

    for (let i = 0; i < data.length; i++) {
      // Shift input history
      for (let j = xHistory.length - 1; j > 0; j--) {
        xHistory[j] = xHistory[j - 1];
      }
      xHistory[0] = data[i];

      // Calculate output
      let output = 0;
      for (let j = 0; j < b.length; j++) {
        output += b[j] * xHistory[j];
      }
      for (let j = 1; j < a.length; j++) {
        output -= a[j] * yHistory[j];
      }

      // Shift output history
      for (let j = yHistory.length - 1; j > 0; j--) {
        yHistory[j] = yHistory[j - 1];
      }
      yHistory[0] = output;

      filtered[i] = output;
    }

    return filtered;
  }, []);

  // Generate plot traces and annotations
  const { traces, annotations } = useMemo(() => {
    const traces: any[] = [];
    const annotations: any[] = [];

    if (!waveformData || waveformData.length === 0) {
      return { traces, annotations };
    }

    waveformData.forEach((waveform, index) => {
      const station = stations.find(s => s.id === waveform.stationId);
      if (!station) return;

      let processedData = waveform.data;
      if (filterSettings.enabled) {
        processedData = applyDigitalFilter(waveform.data, filterSettings);
      }

      // Apply amplitude scaling
      processedData = processedData.map((value: number) => value * amplitudeScale);

      // Apply time offset for multi-station display
      const timeOffset = processedData.map((_: number, i: number) =>
        i * (waveform.sampleRate ? 1 / waveform.sampleRate : 0.01) +
        (index * 10) // 10-second offset between stations
      );

      traces.push({
        x: timeOffset,
        y: processedData,
        type: 'scatter',
        mode: 'lines',
        name: `${station.network}.${waveform.station}`,
        line: {
          color: theme.palette.primary.main,
        },
        hovertemplate: `<b>${station.network}.${waveform.station}</b><br>` +
          `Time: %{x:.3f}s<br>` +
          `Amplitude: %{y:.6f}<br>` +
          `<extra></extra>`,
      });
    });

    // Add phase annotations if enabled
    if (showPhases && events && events.length > 0) {
      events.forEach((event: any) => {
        event.phases?.forEach((phase: any) => {
          traces.push({
            x: [phase.arrivalTime],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            text: [phase.phase],
            textposition: 'top center',
            marker: {
              color: phase.phase === 'P' ? 'red' : 'blue',
              size: 8,
            },
            name: `${phase.phase} Phase`,
            showlegend: false,
          });
        });
      });
    }

    return { traces, annotations };
  }, [waveformData, stations, filterSettings, amplitudeScale, showPhases, events, theme, applyDigitalFilter]);

  // Initialize stations data
  useEffect(() => {
    const fetchStations = async () => {
      try {
        const response = await fetch('/api/seismic/stations');
        if (response.ok) {
          const stationsData = await response.json();
          setStations(stationsData);

          // Update selected stations if stationIds were provided
          if (stationIds.length > 0) {
            setSelectedStations(stationIds);
          }
        }
      } catch (error) {
        console.error('Failed to fetch stations:', error);
      }
    };

    fetchStations();
  }, [stationIds]);

  // Real-time WebSocket connection
  useEffect(() => {
    if (!realTimeMode) return;

    const connectWebSocket = () => {
      try {
        // Connect to real IRIS WebSocket stream
        const wsUrl = `${process.env.REACT_APP_WS_URL?.replace('http', 'ws') || process.env.REACT_APP_IRIS_WS_URL || 'wss://api.prometheus-ai.platform'}/ws/iris/real-time`;

        wsRef.current = new WebSocket(wsUrl);

        wsRef.current.onopen = () => {
          console.log('Connected to real-time seismic stream');
          // Subscribe to selected stations
          selectedStations.forEach(stationId => {
            wsRef.current?.send(JSON.stringify({
              action: 'subscribe',
              stationId,
              channels: ['BHZ', 'BHN', 'BHE'],
            }));
          });
        };

        wsRef.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'waveform') {
            setWaveformData(prev => {
              const updated = [...prev];
              const existingIndex = updated.findIndex(w => w.stationId === data.stationId);
              if (existingIndex >= 0) {
                updated[existingIndex] = data;
              } else {
                updated.push(data);
              }
              return updated;
            });
          }
        };

        wsRef.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setError('Real-time connection failed');
        };

        wsRef.current.onclose = () => {
          console.log('WebSocket connection closed');
          // Attempt to reconnect after 5 seconds
          setTimeout(connectWebSocket, 5000);
        };
      } catch (error) {
        console.error('Failed to connect to real-time stream:', error);
        setError('Failed to connect to real-time seismic data');
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [realTimeMode, selectedStations]);

  // Handle playback controls
  const handlePlay = useCallback(() => {
    setIsPlaying(true);
  }, []);

  const handlePause = useCallback(() => {
    if (realTimeMode) return;
    setIsPlaying(false);
  }, [realTimeMode]);

  const handleStop = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const handleRefresh = useCallback(async () => {
    setIsLoading(true);
    try {
      // Fetch fresh data for selected stations
      const freshData = await Promise.all(
        selectedStations.map(async (stationId) => {
          const response = await fetch(`/api/seismic/waveform/${stationId}?start=${timeWindow?.start}&end=${timeWindow?.end}`);
          if (!response.ok) {
            throw new Error(`Failed to fetch data for station ${stationId}`);
          }
          return response.json();
        })
      );

      setWaveformData(freshData);

      // Also refresh events
      const eventsResponse = await fetch(`/api/seismic/events?start=${timeWindow?.start}&end=${timeWindow?.end}`);
      if (eventsResponse.ok) {
        setEvents(await eventsResponse.json());
      }
    } catch (error) {
      setError(`Failed to refresh data: ${selectedStations.join(', ')}`);
      console.error('Refresh error:', error);
    } finally {
      setError(null);
      setIsLoading(false);
    }
  }, [selectedStations, timeWindow]);

  /**
   * Perform Fast Fourier Transform for spectral analysis
   */
  const performFFT = useCallback((data: number[]): { frequencies: number[], magnitudes: number[] } => {
    const N = data.length;
    const frequencies: number[] = [];
    const magnitudes: number[] = [];

    // Simple DFT implementation for demonstration
    // In production, would use a proper FFT library like fft.js
    for (let k = 0; k < N / 2; k++) {
      let realSum = 0;
      let imagSum = 0;

      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        realSum += data[n] * Math.cos(angle);
        imagSum += data[n] * Math.sin(angle);
      }

      frequencies.push(k * (filterSettings.sampleRate / N));
      magnitudes.push(Math.sqrt(realSum * realSum + imagSum * imagSum));
    }

    return { frequencies, magnitudes };
  }, [filterSettings.sampleRate]);

  /**
   * Detect seismic events using STA/LTA algorithm
   */
  const detectEvents = useCallback((data: number[], sampleRate: number): Array<{ startTime: number; endTime: number; amplitude: number }> => {
    const events: Array<{ startTime: number; endTime: number; amplitude: number }> = [];

    // STA/LTA parameters
    const staWindow = Math.floor(1.0 * sampleRate); // 1 second short-term average
    const ltaWindow = Math.floor(10.0 * sampleRate); // 10 second long-term average
    const triggerThreshold = 3.0;
    const detriggerThreshold = 1.5;

    let triggered = false;
    let eventStart = 0;

    for (let i = ltaWindow; i < data.length - staWindow; i++) {
      // Calculate STA (Short Term Average)
      let staSum = 0;
      for (let j = i; j < i + staWindow; j++) {
        staSum += data[j] * data[j];
      }
      const sta = staSum / staWindow;

      // Calculate LTA (Long Term Average)
      let ltaSum = 0;
      for (let j = i - ltaWindow; j < i; j++) {
        ltaSum += data[j] * data[j];
      }
      const lta = ltaSum / ltaWindow;

      const ratio = lta > 0 ? sta / lta : 0;

      if (!triggered && ratio > triggerThreshold) {
        triggered = true;
        eventStart = i;
      } else if (triggered && ratio < detriggerThreshold) {
        triggered = false;
        const eventEnd = i;
        const eventData = data.slice(eventStart, eventEnd);
        const maxAmplitude = Math.max(...eventData.map(Math.abs));

        events.push({
          startTime: eventStart / sampleRate,
          endTime: eventEnd / sampleRate,
          amplitude: maxAmplitude,
        });
      }
    }

    return events;
  }, []);

  /**
   * Calculate cross-correlation between two waveforms
   */
  const calculateCrossCorrelation = useCallback((data1: number[], data2: number[]): { lags: number[], correlation: number[] } => {
    const maxLag = Math.min(data1.length, data2.length) / 4;
    const lags: number[] = [];
    const correlation: number[] = [];

    for (let lag = -maxLag; lag <= maxLag; lag++) {
      let sum = 0;
      let count = 0;

      for (let i = 0; i < data1.length; i++) {
        const j = i + lag;
        if (j >= 0 && j < data2.length) {
          sum += data1[i] * data2[j];
          count++;
        }
      }

      lags.push(lag);
      correlation.push(count > 0 ? sum / count : 0);
    }

    return { lags, correlation };
  }, []);

  /**
   * Export waveform data in various formats
   */
  const handleExport = useCallback(async (format: 'csv' | 'sac' | 'mseed' | 'json') => {
    if (!waveformData.length) return;

    try {
      let exportData: string;
      let filename: string;
      let mimeType: string;

      switch (format) {
        case 'csv':
          exportData = waveformData.map(waveform => {
            const station = stations.find(s => s.id === waveform.stationId);
            return waveform.data.map((value, index) =>
              `${station?.network}.${station?.station},${index / waveform.sampleRate},${value}`
            ).join('\n');
          }).join('\n');
          filename = `seismic_waveforms_${new Date().toISOString().split('T')[0]}.csv`;
          mimeType = 'text/csv';
          break;

        case 'json':
          exportData = JSON.stringify({
            metadata: {
              exportTime: new Date().toISOString(),
              stations: selectedStations,
              timeWindow,
              filterSettings,
            },
            waveforms: waveformData.map(waveform => ({
              ...waveform,
              station: stations.find(s => s.id === waveform.stationId),
            })),
            events,
          }, null, 2);
          filename = `seismic_data_${new Date().toISOString().split('T')[0]}.json`;
          mimeType = 'application/json';
          break;

        case 'sac':
        case 'mseed':
          // These would require specialized libraries for proper implementation
          throw new Error(`${format.toUpperCase()} export requires specialized seismological libraries`);

        default:
          throw new Error(`Unsupported export format: ${format}`);
      }

      // Create and download file
      const blob = new Blob([exportData], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (error) {
      setError(`Export failed: ${error}`);
      console.error('Export error:', error);
    }
  }, [waveformData, stations, selectedStations, timeWindow, filterSettings, events]);

  /**
   * Calculate signal-to-noise ratio
   */
  const calculateSNR = useCallback((data: number[], noiseStart: number, noiseEnd: number, signalStart: number, signalEnd: number): number => {
    const noiseData = data.slice(noiseStart, noiseEnd);
    const signalData = data.slice(signalStart, signalEnd);

    const noiseRMS = Math.sqrt(noiseData.reduce((sum, val) => sum + val * val, 0) / noiseData.length);
    const signalRMS = Math.sqrt(signalData.reduce((sum, val) => sum + val * val, 0) / signalData.length);

    return noiseRMS > 0 ? 20 * Math.log10(signalRMS / noiseRMS) : 0;
  }, []);

  /**
   * Apply taper window to reduce edge effects
   */
  const applyTaper = useCallback((data: number[], taperPercent: number = 0.05): number[] => {
    const taperLength = Math.floor(data.length * taperPercent);
    const taperedData = [...data];

    // Apply Hanning taper to beginning and end
    for (let i = 0; i < taperLength; i++) {
      const weight = 0.5 * (1 - Math.cos(Math.PI * i / taperLength));
      taperedData[i] *= weight;
      taperedData[data.length - 1 - i] *= weight;
    }

    return taperedData;
  }, []);

  return (
    <Box>
      {/* Controls */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Typography variant="h6">Seismic Waveforms</Typography>
          </Grid>

          <Grid item>
            <Box display="flex" gap={1}>
              <IconButton onClick={handlePlay} disabled={isPlaying}>
                <PlayArrow />
              </IconButton>
              <IconButton onClick={handlePause} disabled={!isPlaying || realTimeMode}>
                <Pause />
              </IconButton>
              <IconButton onClick={handleStop}>
                <Stop />
              </IconButton>
              <IconButton onClick={handleRefresh}>
                <Refresh />
              </IconButton>
              <IconButton onClick={() => handleExport('json')}>
                <Download />
              </IconButton>
              <IconButton>
                <Settings />
              </IconButton>
            </Box>
          </Grid>

          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={filterSettings.enabled}
                  onChange={(e) => setFilterSettings(prev => ({ ...prev, enabled: e.target.checked }))}
                />
              }
              label="Filters"
            />
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

          {realTimeMode && (
            <Grid item>
              <Chip
                label="LIVE"
                color="error"
                variant="filled"
                size="small"
              />
            </Grid>
          )}
        </Grid>

        {/* Filter Controls */}
        {filterSettings.enabled && (
          <Box mt={2}>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={filterSettings.highPass}
                      onChange={(e) => setFilterSettings(prev => ({ ...prev, highPass: e.target.checked }))}
                    />
                  }
                  label="High-pass"
                />
                {filterSettings.highPass && (
                  <Box mt={1}>
                    <Typography variant="caption">Frequency: {filterSettings.highPassFreq} Hz</Typography>
                    <Slider
                      value={filterSettings.highPassFreq}
                      onChange={(_: Event, value: number | number[]) => setFilterSettings(prev => ({ ...prev, highPassFreq: value as number }))}
                      min={0.1}
                      max={50}
                      step={0.1}
                      size="small"
                    />
                  </Box>
                )}
              </Grid>

              <Grid item xs={12} md={4}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={filterSettings.lowPass}
                      onChange={(e) => setFilterSettings(prev => ({ ...prev, lowPass: e.target.checked }))}
                    />
                  }
                  label="Low-pass"
                />
                {filterSettings.lowPass && (
                  <Box mt={1}>
                    <Typography variant="caption">Frequency: {filterSettings.lowPassFreq} Hz</Typography>
                    <Slider
                      value={filterSettings.lowPassFreq}
                      onChange={(_: Event, value: number | number[]) => setFilterSettings(prev => ({ ...prev, lowPassFreq: value as number }))}
                      min={1}
                      max={100}
                      step={0.1}
                      size="small"
                    />
                  </Box>
                )}
              </Grid>

              <Grid item xs={12} md={4}>
                <Typography variant="caption">Amplitude Scale: {amplitudeScale}x</Typography>
                <Slider
                  value={amplitudeScale}
                  onChange={(_: Event, value: number | number[]) => setAmplitudeScale(value as number)}
                  min={0.1}
                  max={10}
                  step={0.1}
                  size="small"
                />
              </Grid>
            </Grid>

            {/* Advanced Analysis Controls */}
            <Box mt={2}>
              <Typography variant="subtitle2" gutterBottom>Advanced Analysis</Typography>
              <Grid container spacing={2}>
                <Grid item>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => {
                      if (waveformData.length > 0) {
                        const fftResult = performFFT(waveformData[0].data);
                        console.log('FFT Analysis:', fftResult);
                      }
                    }}
                  >
                    Spectral Analysis
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => {
                      if (waveformData.length > 0) {
                        const detectedEvents = detectEvents(waveformData[0].data, waveformData[0].sampleRate);
                        console.log('Detected Events:', detectedEvents);
                      }
                    }}
                  >
                    Event Detection
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => {
                      if (waveformData.length >= 2) {
                        const correlation = calculateCrossCorrelation(waveformData[0].data, waveformData[1].data);
                        console.log('Cross-correlation:', correlation);
                      }
                    }}
                  >
                    Cross-Correlation
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => {
                      if (waveformData.length > 0) {
                        const data = waveformData[0].data;
                        const snr = calculateSNR(data, 0, Math.floor(data.length * 0.1), Math.floor(data.length * 0.5), data.length);
                        console.log('Signal-to-Noise Ratio:', snr, 'dB');
                      }
                    }}
                  >
                    SNR Analysis
                  </Button>
                </Grid>
                <Grid item>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={() => {
                      if (waveformData.length > 0) {
                        const taperedData = applyTaper(waveformData[0].data, 0.05);
                        console.log('Applied taper to', taperedData.length, 'samples');
                      }
                    }}
                  >
                    Apply Taper
                  </Button>
                </Grid>
                <Grid item>
                  <FormControl size="small" sx={{ minWidth: 120 }}>
                    <InputLabel>Export Format</InputLabel>
                    <Select
                      defaultValue="json"
                      onChange={(e) => handleExport(e.target.value as 'csv' | 'json')}
                    >
                      <MenuItem value="json">JSON</MenuItem>
                      <MenuItem value="csv">CSV</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </Box>
          </Box>
        )}
      </Paper>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Waveform Plot */}
      <Paper elevation={1}>
        <Plot
          ref={plotRef}
          data={traces}
          layout={{
            title: 'Seismic Waveforms',
            xaxis: { title: 'Time (s)' },
            yaxis: { title: 'Amplitude' },
            height: compact ? 400 : 600,
            margin: { l: 50, r: 50, t: 50, b: 50 },
            showlegend: true,
            annotations,
          }}
          config={{
            displayModeBar: true,
            displaylogo: false,
            responsive: true,
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </Paper>
    </Box>
  );
};

export default SeismicWaveformVisualization;