/**
 * Advanced 3D Geological Visualization Component
 * High-performance 3D rendering with real-time data overlay and multi-layer support
 */

import React, { useState, useRef, useCallback, useMemo } from 'react';
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Settings,
  Download,
  ViewInAr,
  ExpandMore,
  PlayArrow,
  Pause,
  Stop,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface GeologicalLayer3D {
  id: string;
  name: string;
  type: 'surface' | 'volume' | 'points' | 'vectors' | 'isosurface';
  data: {
    x: number[];
    y: number[];
    z: number[];
    values?: number[];
    colors?: string[];
  };
  properties: {
    opacity: number;
    color: string;
    visible: boolean;
    renderMode: 'solid' | 'wireframe' | 'points';
    colorScale?: string;
    isovalue?: number;
  };
  metadata: {
    source: string;
    timestamp: Date;
    quality: number;
  };
}

interface ViewConfiguration {
  camera: {
    eye: { x: number; y: number; z: number };
    center: { x: number; y: number; z: number };
    up: { x: number; y: number; z: number };
  };
  lighting: {
    ambient: number;
    diffuse: number;
    specular: number;
    roughness: number;
  };
  background: string;
  showAxes: boolean;
  showGrid: boolean;
  axisLabels: { x: string; y: string; z: string };
}

interface Advanced3DGeologicalVisualizationProps {
  layers: GeologicalLayer3D[];
  realTimeMode?: boolean;
  onLayerUpdate?: (layers: GeologicalLayer3D[]) => void;
  onViewChange?: (view: ViewConfiguration) => void;
  height?: number;
  enableVR?: boolean;
}

const Advanced3DGeologicalVisualization: React.FC<Advanced3DGeologicalVisualizationProps> = ({
  layers,
  // realTimeMode = false, - unused parameter
  onLayerUpdate,
  onViewChange,
  height = 600,
  enableVR = false,
}) => {
  const theme = useTheme();
  const plotRef = useRef<any>(null);

  // State management
  // Removed unused selectedLayers state
  const [viewConfig, setViewConfig] = useState<ViewConfiguration>({
    camera: {
      eye: { x: 1.5, y: 1.5, z: 1.5 },
      center: { x: 0, y: 0, z: 0 },
      up: { x: 0, y: 0, z: 1 },
    },
    lighting: {
      ambient: 0.4,
      diffuse: 0.6,
      specular: 0.2,
      roughness: 0.1,
    },
    background: theme.palette.background.default,
    showAxes: true,
    showGrid: true,
    axisLabels: { x: 'Easting (m)', y: 'Northing (m)', z: 'Elevation (m)' },
  });
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(1);
  const [showSettings, setShowSettings] = useState(false);
  const [renderQuality, setRenderQuality] = useState<'low' | 'medium' | 'high'>('medium');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generate 3D plot data
  const plot3DData = useMemo(() => {
    const traces: Data[] = [];

    layers.filter(layer => 
      layer.properties.visible
    ).forEach((layer) => {
      switch (layer.type) {
        case 'surface':
          traces.push({
            type: 'mesh3d',
            x: layer.data.x,
            y: layer.data.y,
            z: layer.data.z,
            intensity: layer.data.values,
            colorscale: layer.properties.colorScale || 'Viridis',
            opacity: layer.properties.opacity,
            name: layer.name,
            showscale: true,
            hovertemplate: `<b>${layer.name}</b><br>` +
                          `X: %{x:.2f}<br>` +
                          `Y: %{y:.2f}<br>` +
                          `Z: %{z:.2f}<br>` +
                          `Value: %{intensity:.3f}<extra></extra>`,
          } as Data);
          break;

        case 'volume':
          traces.push({
            type: 'volume',
            x: layer.data.x,
            y: layer.data.y,
            z: layer.data.z,
            value: layer.data.values,
            opacity: layer.properties.opacity,
            surface: { count: 10 },
            colorscale: layer.properties.colorScale || 'Plasma',
            name: layer.name,
            hovertemplate: `<b>${layer.name}</b><br>` +
                          `X: %{x:.2f}<br>` +
                          `Y: %{y:.2f}<br>` +
                          `Z: %{z:.2f}<br>` +
                          `Value: %{value:.3f}<extra></extra>`,
          } as Data);
          break;

        case 'points':
          traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: layer.data.x,
            y: layer.data.y,
            z: layer.data.z,
            marker: {
              size: 5,
              color: layer.data.values || layer.properties.color,
              colorscale: layer.properties.colorScale || 'Viridis',
              opacity: layer.properties.opacity,
              showscale: true,
            },
            name: layer.name,
            hovertemplate: `<b>${layer.name}</b><br>` +
                          `X: %{x:.2f}<br>` +
                          `Y: %{y:.2f}<br>` +
                          `Z: %{z:.2f}<br>` +
                          `<extra></extra>`,
          } as Data);
          break;

        case 'vectors':
          // Add vector field visualization
          const vectorTraces: Data[] = [];
          for (let i = 0; i < layer.data.x.length; i += 10) { // Subsample for performance
            vectorTraces.push({
              type: 'scatter3d',
              mode: 'lines',
              x: [layer.data.x[i], layer.data.x[i] + (layer.data.values?.[i] || 0) * 10],
              y: [layer.data.y[i], layer.data.y[i] + (layer.data.values?.[i + 1] || 0) * 10],
              z: [layer.data.z[i], layer.data.z[i] + (layer.data.values?.[i + 2] || 0) * 10],
              line: {
                color: layer.properties.color,
                width: 3,
              },
              showlegend: false,
              hoverinfo: 'skip',
            } as Data);
          }
          traces.push(...vectorTraces);
          break;

        case 'isosurface':
          traces.push({
            type: 'isosurface',
            x: layer.data.x,
            y: layer.data.y,
            z: layer.data.z,
            value: layer.data.values,
            isomin: layer.properties.isovalue || 0,
            isomax: layer.properties.isovalue || 1,
            opacity: layer.properties.opacity,
            colorscale: layer.properties.colorScale || 'RdYlBu',
            name: layer.name,
            hovertemplate: `<b>${layer.name}</b><br>` +
                          `X: %{x:.2f}<br>` +
                          `Y: %{y:.2f}<br>` +
                          `Z: %{z:.2f}<br>` +
                          `Value: %{value:.3f}<extra></extra>`,
          } as Data);
          break;
      }
    });

    return traces;
  }, [layers]);

  // Generate layout configuration
  const plotLayout = useMemo((): Partial<Layout> => ({
    scene: {
      camera: viewConfig.camera,
      xaxis: {
        title: viewConfig.axisLabels.x,
        showgrid: viewConfig.showGrid,
        visible: viewConfig.showAxes,
      },
      yaxis: {
        title: viewConfig.axisLabels.y,
        showgrid: viewConfig.showGrid,
        visible: viewConfig.showAxes,
      },
      zaxis: {
        title: viewConfig.axisLabels.z,
        showgrid: viewConfig.showGrid,
        visible: viewConfig.showAxes,
      },
      bgcolor: viewConfig.background,
      aspectmode: 'cube',
    },
    paper_bgcolor: viewConfig.background,
    plot_bgcolor: viewConfig.background,
    height,
    margin: { l: 0, r: 0, t: 40, b: 0 },
    showlegend: true,
    legend: { x: 1, y: 1 },
    title: '3D Geological Visualization',
  }), [viewConfig, height]);

  // Handle layer visibility toggle
  const toggleLayerVisibility = useCallback((layerId: string) => {
    const updatedLayers = layers.map(layer => 
      layer.id === layerId 
        ? { ...layer, properties: { ...layer.properties, visible: !layer.properties.visible } }
        : layer
    );
    onLayerUpdate?.(updatedLayers);
  }, [layers, onLayerUpdate]);

  // Handle layer opacity change
  const handleOpacityChange = useCallback((layerId: string, opacity: number) => {
    const updatedLayers = layers.map(layer => 
      layer.id === layerId 
        ? { ...layer, properties: { ...layer.properties, opacity } }
        : layer
    );
    onLayerUpdate?.(updatedLayers);
  }, [layers, onLayerUpdate]);

  // Handle view preset changes
  const applyViewPreset = useCallback((preset: string) => {
    let newCamera;
    switch (preset) {
      case 'top':
        newCamera = { eye: { x: 0, y: 0, z: 3 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 1, z: 0 } };
        break;
      case 'front':
        newCamera = { eye: { x: 0, y: -3, z: 0 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
        break;
      case 'side':
        newCamera = { eye: { x: 3, y: 0, z: 0 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
        break;
      case 'isometric':
        newCamera = { eye: { x: 1.5, y: 1.5, z: 1.5 }, center: { x: 0, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } };
        break;
      default:
        return;
    }
    
    const newViewConfig = { ...viewConfig, camera: newCamera };
    setViewConfig(newViewConfig);
    onViewChange?.(newViewConfig);
  }, [viewConfig, onViewChange]);

  // Animation controls
  const startAnimation = useCallback(() => {
    setIsAnimating(true);
    // Implement rotation animation
    const animate = () => {
      if (!isAnimating) return;
      
      const currentCamera = viewConfig.camera;
      const angle = Date.now() * 0.001 * animationSpeed;
      const radius = Math.sqrt(currentCamera.eye.x ** 2 + currentCamera.eye.y ** 2);
      
      const newCamera = {
        ...currentCamera,
        eye: {
          x: radius * Math.cos(angle),
          y: radius * Math.sin(angle),
          z: currentCamera.eye.z,
        },
      };
      
      setViewConfig(prev => ({ ...prev, camera: newCamera }));
      requestAnimationFrame(animate);
    };
    
    animate();
  }, [isAnimating, viewConfig.camera, animationSpeed]);

  const stopAnimation = useCallback(() => {
    setIsAnimating(false);
  }, []);

  // Export functionality
  const exportVisualization = useCallback(async (format: 'png' | 'html' | 'json') => {
    if (!plotRef.current) return;

    try {
      setIsLoading(true);
      
      switch (format) {
        case 'png':
          await plotRef.current.downloadImage({
            format: 'png',
            width: 1920,
            height: 1080,
            filename: `geological_3d_${Date.now()}`,
          });
          break;
          
        case 'html':
          const htmlData = plotRef.current.el.outerHTML;
          const blob = new Blob([htmlData], { type: 'text/html' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `geological_3d_${Date.now()}.html`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          break;
          
        case 'json':
          const exportData = {
            layers: layers.filter(l => l.properties.visible),
            viewConfig,
            timestamp: new Date().toISOString(),
          };
          const jsonBlob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
          const jsonUrl = URL.createObjectURL(jsonBlob);
          const jsonA = document.createElement('a');
          jsonA.href = jsonUrl;
          jsonA.download = `geological_3d_data_${Date.now()}.json`;
          document.body.appendChild(jsonA);
          jsonA.click();
          document.body.removeChild(jsonA);
          URL.revokeObjectURL(jsonUrl);
          break;
      }
    } catch (err) {
      setError('Export failed');
      console.error('Export error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [layers, viewConfig]);

  return (
    <Box>
      {/* Controls */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Typography variant="h6">3D Geological Visualization</Typography>
          </Grid>
          
          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>View Preset</InputLabel>
              <Select
                defaultValue="isometric"
                onChange={(e) => applyViewPreset(e.target.value)}
              >
                <MenuItem value="isometric">Isometric</MenuItem>
                <MenuItem value="top">Top View</MenuItem>
                <MenuItem value="front">Front View</MenuItem>
                <MenuItem value="side">Side View</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item>
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel>Quality</InputLabel>
              <Select
                value={renderQuality}
                onChange={(e) => setRenderQuality(e.target.value as any)}
              >
                <MenuItem value="low">Low</MenuItem>
                <MenuItem value="medium">Medium</MenuItem>
                <MenuItem value="high">High</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item>
            <Box display="flex" gap={1}>
              <Button
                variant={isAnimating ? "contained" : "outlined"}
                startIcon={isAnimating ? <Pause /> : <PlayArrow />}
                onClick={isAnimating ? stopAnimation : startAnimation}
                size="small"
              >
                {isAnimating ? 'Pause' : 'Animate'}
              </Button>
              
              <Button
                variant="outlined"
                startIcon={<Stop />}
                onClick={stopAnimation}
                disabled={!isAnimating}
                size="small"
              >
                Stop
              </Button>
            </Box>
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
              onClick={() => exportVisualization('png')}
              size="small"
            >
              Export
            </Button>
          </Grid>

          {enableVR && (
            <Grid item>
              <Button
                variant="outlined"
                startIcon={<ViewInAr />}
                size="small"
              >
                VR Mode
              </Button>
            </Grid>
          )}
        </Grid>
      </Paper>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Grid container spacing={2}>
        {/* Layer Controls */}
        <Grid item xs={12} md={3}>
          <Paper elevation={1} sx={{ p: 2, height: height - 100, overflowY: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Layers
            </Typography>
            
            {layers.map((layer) => (
              <Accordion key={layer.id} sx={{ mb: 1 }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box display="flex" alignItems="center" width="100%">
                    <IconButton
                      size="small"
                      onClick={(e: React.MouseEvent) => {
                        e.stopPropagation();
                        toggleLayerVisibility(layer.id);
                      }}
                    >
                      {layer.properties.visible ? <Visibility /> : <VisibilityOff />}
                    </IconButton>
                    <Box
                      sx={{
                        width: 16,
                        height: 16,
                        backgroundColor: layer.properties.color,
                        mr: 1,
                        border: '1px solid #ccc',
                      }}
                    />
                    <Typography variant="body2" sx={{ flexGrow: 1 }}>
                      {layer.name}
                    </Typography>
                    <Chip
                      size="small"
                      label={layer.type}
                      variant="outlined"
                    />
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Box>
                    <Typography variant="caption">Opacity</Typography>
                    <Slider
                      value={layer.properties.opacity}
                      onChange={(_: Event, value: number | number[]) => 
                        handleOpacityChange(layer.id, value as number)
                      }
                      min={0}
                      max={1}
                      step={0.1}
                      size="small"
                    />
                    
                    <Typography variant="caption" display="block">
                      Source: {layer.metadata.source}
                    </Typography>
                    <Typography variant="caption" display="block">
                      Quality: {(layer.metadata.quality * 100).toFixed(0)}%
                    </Typography>
                    <Typography variant="caption" display="block">
                      Points: {layer.data.x.length.toLocaleString()}
                    </Typography>
                  </Box>
                </AccordionDetails>
              </Accordion>
            ))}
          </Paper>
        </Grid>

        {/* 3D Visualization */}
        <Grid item xs={12} md={9}>
          <Paper elevation={1}>
            <Plot
              ref={plotRef}
              data={plot3DData}
              layout={plotLayout}
              config={{
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                displaylogo: false,
                responsive: true,
              }}
              style={{ width: '100%', height: `${height}px` }}
              onRelayout={(eventData) => {
                if ((eventData as any)['scene.camera']) {
                  const newViewConfig = {
                    ...viewConfig,
                    camera: (eventData as any)['scene.camera'],
                  };
                  setViewConfig(newViewConfig);
                  onViewChange?.(newViewConfig);
                }
              }}
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Settings Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="md" fullWidth>
        <DialogTitle>3D Visualization Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Lighting
              </Typography>
            </Grid>
            
            <Grid item xs={6}>
              <Typography variant="caption">Ambient Light</Typography>
              <Slider
                value={viewConfig.lighting.ambient}
                onChange={(_: Event, value: number | number[]) => 
                  setViewConfig(prev => ({
                    ...prev,
                    lighting: { ...prev.lighting, ambient: value as number }
                  }))
                }
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>
            
            <Grid item xs={6}>
              <Typography variant="caption">Diffuse Light</Typography>
              <Slider
                value={viewConfig.lighting.diffuse}
                onChange={(_: Event, value: number | number[]) => 
                  setViewConfig(prev => ({
                    ...prev,
                    lighting: { ...prev.lighting, diffuse: value as number }
                  }))
                }
                min={0}
                max={1}
                step={0.1}
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewConfig.showAxes}
                    onChange={(e) => setViewConfig(prev => ({
                      ...prev,
                      showAxes: e.target.checked
                    }))}
                  />
                }
                label="Show Axes"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={viewConfig.showGrid}
                    onChange={(e) => setViewConfig(prev => ({
                      ...prev,
                      showGrid: e.target.checked
                    }))}
                  />
                }
                label="Show Grid"
              />
            </Grid>

            {isAnimating && (
              <Grid item xs={12}>
                <Typography variant="caption">Animation Speed</Typography>
                <Slider
                  value={animationSpeed}
                  onChange={(_: Event, value: number | number[]) => setAnimationSpeed(value as number)}
                  min={0.1}
                  max={5}
                  step={0.1}
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

export default Advanced3DGeologicalVisualization;