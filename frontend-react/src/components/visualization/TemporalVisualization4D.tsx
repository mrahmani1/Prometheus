/**
 * 4D Temporal Visualization Component
 * Advanced time-series visualization for geological evolution over time
 */

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Box,
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  SkipNext,
  Speed,
  Settings,
  Download,
} from '@mui/icons-material';
// Removed unused useTheme import
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface TemporalDataPoint {
  timestamp: Date;
  location: { x: number; y: number; z: number };
  value: number;
  properties: Record<string, any>;
}

interface TemporalDataset {
  id: string;
  name: string;
  description: string;
  data: TemporalDataPoint[];
  timeRange: { start: Date; end: Date };
  spatialBounds: {
    xMin: number; xMax: number;
    yMin: number; yMax: number;
    zMin: number; zMax: number;
  };
  colorScale: string;
  unit: string;
  visible: boolean;
}

interface AnimationConfig {
  duration: number; // Total animation duration in seconds
  fps: number; // Frames per second
  loop: boolean;
  autoPlay: boolean;
  smoothTransition: boolean;
  interpolationMethod: 'linear' | 'cubic' | 'nearest';
}

interface ViewConfig {
  viewType: '3d' | 'map' | 'cross-section' | 'time-series';
  timeWindow: number; // Time window in hours
  showTrails: boolean;
  trailLength: number; // Number of previous time steps to show
  particleMode: boolean;
  showVelocityVectors: boolean;
}

interface TemporalVisualization4DProps {
  datasets: TemporalDataset[];
  initialTime?: Date;
  onTimeChange?: (time: Date) => void;
  onDatasetToggle?: (datasetId: string, visible: boolean) => void;
  height?: number;
}

const TemporalVisualization4D: React.FC<TemporalVisualization4DProps> = ({
  datasets,
  initialTime,
  onTimeChange,
  onDatasetToggle,
  height = 600,
}) => {
  // Removed unused theme
  const plotRef = useRef<any>(null);
  const animationRef = useRef<number | null>(null);

  // State management
  const [currentTime, setCurrentTime] = useState(initialTime || new Date());
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationConfig, setAnimationConfig] = useState<AnimationConfig>({
    duration: 60,
    fps: 30,
    loop: true,
    autoPlay: false,
    smoothTransition: true,
    interpolationMethod: 'linear',
  });
  const [viewConfig, setViewConfig] = useState<ViewConfig>({
    viewType: '3d',
    timeWindow: 24,
    showTrails: true,
    trailLength: 10,
    particleMode: false,
    showVelocityVectors: false,
  });
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [showSettings, setShowSettings] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Calculate time bounds from all datasets
  const timeBounds = useMemo(() => {
    if (datasets.length === 0) return { start: new Date(), end: new Date() };
    
    const allTimes = datasets.flatMap(dataset => 
      dataset.data.map(point => point.timestamp)
    );
    
    return {
      start: new Date(Math.min(...allTimes.map(t => t.getTime()))),
      end: new Date(Math.max(...allTimes.map(t => t.getTime()))),
    };
  }, [datasets]);

  // Get data for current time window
  const getCurrentTimeData = useCallback((time: Date) => {
    const windowStart = new Date(time.getTime() - viewConfig.timeWindow * 60 * 60 * 1000);
    const windowEnd = time;

    return datasets.filter(dataset => dataset.visible).map(dataset => {
      const filteredData = dataset.data.filter(point => 
        point.timestamp >= windowStart && point.timestamp <= windowEnd
      );

      // If showing trails, include previous time steps
      if (viewConfig.showTrails) {
        const trailStart = new Date(windowStart.getTime() - viewConfig.trailLength * 60 * 60 * 1000);
        const trailData = dataset.data.filter(point => 
          point.timestamp >= trailStart && point.timestamp < windowStart
        );
        return { ...dataset, data: [...trailData, ...filteredData] };
      }

      return { ...dataset, data: filteredData };
    });
  }, [datasets, viewConfig.timeWindow, viewConfig.showTrails, viewConfig.trailLength]);

  // Generate 4D visualization traces
  const generate4DTraces = useCallback((time: Date) => {
    const traces: Data[] = [];
    const currentData = getCurrentTimeData(time);

    currentData.forEach((dataset, datasetIndex) => {
      if (dataset.data.length === 0) return;

      switch (viewConfig.viewType) {
        case '3d':
          // 3D scatter plot with time-based coloring
          const timeColors = dataset.data.map(point => {
            const timeDiff = time.getTime() - point.timestamp.getTime();
            const normalizedTime = Math.max(0, 1 - timeDiff / (viewConfig.timeWindow * 60 * 60 * 1000));
            return normalizedTime;
          });

          traces.push({
            type: 'scatter3d',
            mode: viewConfig.particleMode ? 'markers' : 'markers+lines',
            x: dataset.data.map(p => p.location.x),
            y: dataset.data.map(p => p.location.y),
            z: dataset.data.map(p => p.location.z),
            marker: {
              size: viewConfig.particleMode ? 8 : 5,
              color: timeColors,
              colorscale: dataset.colorScale,
              opacity: 0.8,
              showscale: datasetIndex === 0,
              colorbar: {
                title: 'Time Recency',
                x: 1.02 + datasetIndex * 0.1,
              },
            },
            line: viewConfig.particleMode ? undefined : {
              color: dataset.colorScale,
              width: 2,
            },
            name: dataset.name,
            text: dataset.data.map(p => 
              `Time: ${p.timestamp.toLocaleString()}<br>` +
              `Value: ${p.value.toFixed(3)} ${dataset.unit}<br>` +
              `Location: (${p.location.x.toFixed(2)}, ${p.location.y.toFixed(2)}, ${p.location.z.toFixed(2)})`
            ),
            hovertemplate: '%{text}<extra></extra>',
          } as Data);

          // Add velocity vectors if enabled
          if (viewConfig.showVelocityVectors && dataset.data.length > 1) {
            const vectorTraces: Data[] = [];
            for (let i = 1; i < dataset.data.length; i++) {
              const current = dataset.data[i];
              const previous = dataset.data[i - 1];
              const timeDelta = (current.timestamp.getTime() - previous.timestamp.getTime()) / 1000; // seconds
              
              if (timeDelta > 0) {
                const velocity = {
                  x: (current.location.x - previous.location.x) / timeDelta,
                  y: (current.location.y - previous.location.y) / timeDelta,
                  z: (current.location.z - previous.location.z) / timeDelta,
                };

                const scale = 100; // Scale factor for vector display
                vectorTraces.push({
                  type: 'scatter3d',
                  mode: 'lines',
                  x: [current.location.x, current.location.x + velocity.x * scale],
                  y: [current.location.y, current.location.y + velocity.y * scale],
                  z: [current.location.z, current.location.z + velocity.z * scale],
                  line: { color: 'red', width: 2 },
                  showlegend: false,
                  hoverinfo: 'skip',
                } as Data);
              }
            }
            traces.push(...vectorTraces);
          }
          break;

        case 'time-series':
          // Time series plot
          traces.push({
            type: 'scatter',
            mode: 'lines+markers',
            x: dataset.data.map(p => p.timestamp),
            y: dataset.data.map(p => p.value),
            name: dataset.name,
            line: { color: dataset.colorScale },
            marker: { size: 4 },
            hovertemplate: `<b>${dataset.name}</b><br>` +
                          `Time: %{x}<br>` +
                          `Value: %{y:.3f} ${dataset.unit}<br>` +
                          `<extra></extra>`,
          } as Data);

          // Add current time indicator
          traces.push({
            type: 'scatter',
            mode: 'lines',
            x: [time, time],
            y: [dataset.spatialBounds.zMin, dataset.spatialBounds.zMax],
            line: { color: 'red', width: 3, dash: 'dash' },
            name: 'Current Time',
            showlegend: datasetIndex === 0,
            hoverinfo: 'skip',
          } as Data);
          break;

        case 'map':
          // 2D map view
          traces.push({
            type: 'scattermapbox',
            lat: dataset.data.map(p => p.location.y),
            lon: dataset.data.map(p => p.location.x),
            mode: 'markers',
            marker: {
              size: 8,
              color: dataset.data.map(p => p.value),
              colorscale: dataset.colorScale,
              opacity: 0.8,
              showscale: datasetIndex === 0,
            },
            name: dataset.name,
            text: dataset.data.map(p => 
              `Time: ${p.timestamp.toLocaleString()}<br>` +
              `Value: ${p.value.toFixed(3)} ${dataset.unit}`
            ),
            hovertemplate: '%{text}<extra></extra>',
          } as Data);
          break;
      }
    });

    return traces;
  }, [viewConfig, getCurrentTimeData]);

  // Generate layout configuration
  const generateLayout = useCallback((): Partial<Layout> => {
    const baseLayout = {
      height,
      margin: { l: 50, r: 50, t: 50, b: 50 },
      showlegend: true,
      legend: { x: 0, y: 1 },
      title: `4D Temporal Visualization - ${currentTime.toLocaleString()}`,
    };

    switch (viewConfig.viewType) {
      case '3d':
        return {
          ...baseLayout,
          scene: {
            xaxis: { title: 'X (m)' },
            yaxis: { title: 'Y (m)' },
            zaxis: { title: 'Z (m)' },
            camera: {
              eye: { x: 1.5, y: 1.5, z: 1.5 },
            },
          },
        };

      case 'time-series':
        return {
          ...baseLayout,
          xaxis: { 
            title: 'Time',
            type: 'date',
            range: [timeBounds.start, timeBounds.end],
          },
          yaxis: { title: 'Value' },
        };

      case 'map':
        return {
          ...baseLayout,
          mapbox: {
            style: 'satellite-streets',
            center: { lat: 0, lon: 0 },
            zoom: 8,
          },
        };

      default:
        return baseLayout;
    }
  }, [viewConfig.viewType, height, currentTime, timeBounds]);

  // Animation control functions
  const startAnimation = useCallback(() => {
    if (isPlaying) return;

    setIsPlaying(true);
    const startTime = Date.now();
    const totalDuration = animationConfig.duration * 1000; // Convert to milliseconds
    const timeRange = timeBounds.end.getTime() - timeBounds.start.getTime();

    const animate = () => {
      const elapsed = (Date.now() - startTime) * playbackSpeed;
      const progress = (elapsed % totalDuration) / totalDuration;
      
      const newTime = new Date(
        timeBounds.start.getTime() + progress * timeRange
      );
      
      setCurrentTime(newTime);
      onTimeChange?.(newTime);

      if (isPlaying) {
        animationRef.current = requestAnimationFrame(animate);
      }
    };

    animationRef.current = requestAnimationFrame(animate);
  }, [isPlaying, animationConfig.duration, playbackSpeed, timeBounds, onTimeChange]);

  const pauseAnimation = useCallback(() => {
    setIsPlaying(false);
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
  }, []);

  const stopAnimation = useCallback(() => {
    pauseAnimation();
    setCurrentTime(timeBounds.start);
    onTimeChange?.(timeBounds.start);
  }, [pauseAnimation, timeBounds.start, onTimeChange]);

  // Handle time slider change
  const handleTimeChange = useCallback((newTime: Date) => {
    setCurrentTime(newTime);
    onTimeChange?.(newTime);
  }, [onTimeChange]);

  // Export animation
  const exportAnimation = useCallback(async (_format: 'gif' | 'mp4' | 'frames') => {
    setIsLoading(true);
    try {
      // This would implement actual animation export
      // For now, just export current frame
      if (plotRef.current) {
        await plotRef.current.downloadImage({
          format: 'png',
          width: 1920,
          height: 1080,
          filename: `temporal_viz_${currentTime.getTime()}`,
        });
      }
    } catch (err) {
      setError('Export failed');
      console.error('Export error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [currentTime]);

  // Cleanup animation on unmount
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  const currentTraces = generate4DTraces(currentTime);
  const currentLayout = generateLayout();

  return (
    <Box>
      {/* Animation Controls */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Typography variant="h6">4D Temporal Visualization</Typography>
          </Grid>
          
          <Grid item>
            <Box display="flex" gap={1}>
              <IconButton onClick={stopAnimation}>
                <Stop />
              </IconButton>
              <IconButton onClick={isPlaying ? pauseAnimation : startAnimation}>
                {isPlaying ? <Pause /> : <PlayArrow />}
              </IconButton>
              <IconButton>
                <SkipNext />
              </IconButton>
            </Box>
          </Grid>

          <Grid item xs>
            <Box sx={{ px: 2 }}>
              <Typography variant="caption">Time: {currentTime.toLocaleString()}</Typography>
              <Slider
                value={currentTime.getTime()}
                min={timeBounds.start.getTime()}
                max={timeBounds.end.getTime()}
                onChange={(_: Event, value: number | number[]) => 
                  handleTimeChange(new Date(value as number))
                }
                disabled={isPlaying}
              />
            </Box>
          </Grid>

          <Grid item>
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel>View</InputLabel>
              <Select
                value={viewConfig.viewType}
                onChange={(e) => setViewConfig(prev => ({ 
                  ...prev, 
                  viewType: e.target.value as any 
                }))}
              >
                <MenuItem value="3d">3D View</MenuItem>
                <MenuItem value="map">Map View</MenuItem>
                <MenuItem value="time-series">Time Series</MenuItem>
                <MenuItem value="cross-section">Cross Section</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item>
            <Tooltip title="Playback Speed">
              <Box display="flex" alignItems="center" gap={1}>
                <Speed />
                <Slider
                  value={playbackSpeed}
                  onChange={(_: Event, value: number | number[]) => setPlaybackSpeed(value as number)}
                  min={0.1}
                  max={5}
                  step={0.1}
                  sx={{ width: 80 }}
                />
                <Typography variant="caption">{playbackSpeed}x</Typography>
              </Box>
            </Tooltip>
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Settings />}
              onClick={() => setShowSettings(true)}
              size="small"
            >
              Settings
            </Button>
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={() => exportAnimation('gif')}
              size="small"
            >
              Export
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Main Visualization */}
      <Paper elevation={1}>
        <Plot
          ref={plotRef}
          data={currentTraces}
          layout={currentLayout}
          config={{
            displayModeBar: true,
            displaylogo: false,
            responsive: true,
          }}
          style={{ width: '100%', height: `${height}px` }}
        />
      </Paper>

      {/* Dataset Controls */}
      <Paper elevation={1} sx={{ mt: 2, p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Datasets
        </Typography>
        <Grid container spacing={1}>
          {datasets.map((dataset) => (
            <Grid item key={dataset.id}>
              <Chip
                label={dataset.name}
                color={dataset.visible ? 'primary' : 'default'}
                onClick={() => onDatasetToggle?.(dataset.id, !dataset.visible)}
                variant={dataset.visible ? 'filled' : 'outlined'}
              />
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Settings Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="md" fullWidth>
        <DialogTitle>4D Visualization Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Animation Settings
              </Typography>
            </Grid>
            
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Duration (seconds)"
                type="number"
                value={animationConfig.duration}
                onChange={(e) => setAnimationConfig(prev => ({
                  ...prev,
                  duration: parseInt(e.target.value)
                }))}
              />
            </Grid>
            
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="FPS"
                type="number"
                value={animationConfig.fps}
                onChange={(e) => setAnimationConfig(prev => ({
                  ...prev,
                  fps: parseInt(e.target.value)
                }))}
              />
            </Grid>

            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={animationConfig.loop}
                    onChange={(e) => setAnimationConfig(prev => ({
                      ...prev,
                      loop: e.target.checked
                    }))}
                  />
                }
                label="Loop Animation"
              />
            </Grid>

            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewConfig.showTrails}
                    onChange={(e) => setViewConfig(prev => ({
                      ...prev,
                      showTrails: e.target.checked
                    }))}
                  />
                }
                label="Show Trails"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="caption">Time Window (hours)</Typography>
              <Slider
                value={viewConfig.timeWindow}
                onChange={(_: Event, value: number | number[]) => 
                  setViewConfig(prev => ({ ...prev, timeWindow: value as number }))
                }
                min={1}
                max={168}
                step={1}
              />
            </Grid>

            {viewConfig.showTrails && (
              <Grid item xs={12}>
                <Typography variant="caption">Trail Length (time steps)</Typography>
                <Slider
                  value={viewConfig.trailLength}
                  onChange={(_: Event, value: number | number[]) => 
                    setViewConfig(prev => ({ ...prev, trailLength: value as number }))
                  }
                  min={1}
                  max={50}
                  step={1}
                />
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSettings(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default TemporalVisualization4D;