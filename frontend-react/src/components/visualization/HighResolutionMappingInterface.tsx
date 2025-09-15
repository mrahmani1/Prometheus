/**
 * High-Resolution Mapping Interface Component
 * Multi-layer geological mapping with real-time data integration
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
  TextField,
  Tooltip,
  Accordion,
  AccordionDetails,
  AccordionSummary,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
} from '@mui/material';
import {
  Terrain,
  Visibility,
  VisibilityOff,
  Settings,
  Download,
  Refresh,
  Timeline,
  ExpandMore,
  MyLocation,
  Fullscreen,
} from '@mui/icons-material';
// Removed unused useTheme import
import Plot from 'react-plotly.js';
import { Data, Layout } from 'plotly.js';

interface MapLayer {
  id: string;
  name: string;
  type: 'raster' | 'vector' | 'heatmap' | 'contour' | 'scatter' | 'choropleth';
  data: {
    lat: number[];
    lon: number[];
    z?: number[];
    text?: string[];
    customdata?: any[];
  };
  style: {
    opacity: number;
    color: string;
    colorscale?: string;
    size?: number;
    line?: {
      color: string;
      width: number;
    };
    fill?: string;
  };
  visible: boolean;
  interactive: boolean;
  metadata: {
    source: string;
    resolution: string;
    timestamp: Date;
    bounds: {
      north: number;
      south: number;
      east: number;
      west: number;
    };
  };
}

interface MapConfiguration {
  center: { lat: number; lon: number };
  zoom: number;
  projection: 'mercator' | 'orthographic' | 'natural earth' | 'albers usa';
  basemap: 'satellite' | 'terrain' | 'street' | 'topo' | 'ocean';
  showScale: boolean;
  showCompass: boolean;
  showCoordinates: boolean;
}

interface HighResolutionMappingInterfaceProps {
  layers: MapLayer[];
  initialConfig?: Partial<MapConfiguration>;
  realTimeMode?: boolean;
  onLayerUpdate?: (layers: MapLayer[]) => void;
  onMapInteraction?: (event: any) => void;
  height?: number;
  enableMeasurement?: boolean;
}

const HighResolutionMappingInterface: React.FC<HighResolutionMappingInterfaceProps> = ({
  layers,
  initialConfig = {},
  // realTimeMode = false, - unused parameter
  onLayerUpdate,
  onMapInteraction,
  height = 600,
  enableMeasurement = true,
}) => {
  // Removed unused theme
  const plotRef = useRef<any>(null);

  // State management
  const [mapConfig, setMapConfig] = useState<MapConfiguration>({
    center: { lat: 40.7128, lon: -74.0060 },
    zoom: 8,
    projection: 'mercator',
    basemap: 'satellite',
    showScale: true,
    showCompass: true,
    showCoordinates: true,
    ...initialConfig,
  });
  // Removed unused selectedLayers state
  const [showSettings, setShowSettings] = useState(false);
  const [measurementMode, setMeasurementMode] = useState<'distance' | 'area' | 'off'>('off');
  // Removed unused measurements state - const [measurements, setMeasurements] = useState<Array<{
  // id: string;
  // type: 'distance' | 'area';
  // coordinates: number[][];
  // value: number;
  // unit: string;
  // }>>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredFeature, setHoveredFeature] = useState<any>(null);

  // Generate map traces
  const mapTraces = useMemo(() => {
    const traces: Data[] = [];

    // Add basemap layer
    if (mapConfig.basemap === 'satellite') {
      // Satellite imagery would be loaded from tile service
      traces.push({
        type: 'scattermapbox',
        lat: [],
        lon: [],
        mode: 'markers',
        marker: { size: 0 },
        showlegend: false,
        hoverinfo: 'skip',
      } as Data);
    }

    // Add geological layers
    layers.filter(layer =>
      layer.visible
    ).forEach((layer) => {
      switch (layer.type) {
        case 'scatter':
          traces.push({
            type: 'scattermapbox',
            lat: layer.data.lat,
            lon: layer.data.lon,
            mode: 'markers',
            marker: {
              size: layer.style.size || 8,
              color: layer.data.z || layer.style.color,
              colorscale: layer.style.colorscale || 'Viridis',
              opacity: layer.style.opacity,
              showscale: true,
              colorbar: {
                title: layer.name,
                x: 1.02,
              },
            },
            text: layer.data.text,
            customdata: layer.data.customdata,
            name: layer.name,
            hovertemplate: `<b>${layer.name}</b><br>` +
              `Lat: %{lat:.4f}°<br>` +
              `Lon: %{lon:.4f}°<br>` +
              `%{text}<br>` +
              `<extra></extra>`,
          } as Data);
          break;

        case 'heatmap':
          traces.push({
            type: 'densitymapbox',
            lat: layer.data.lat,
            lon: layer.data.lon,
            z: layer.data.z,
            colorscale: layer.style.colorscale || 'Hot',
            opacity: layer.style.opacity,
            radius: 20,
            name: layer.name,
            hovertemplate: `<b>${layer.name}</b><br>` +
              `Lat: %{lat:.4f}°<br>` +
              `Lon: %{lon:.4f}°<br>` +
              `Intensity: %{z:.3f}<br>` +
              `<extra></extra>`,
          } as Data);
          break;

        case 'contour':
          traces.push({
            type: 'scattermapbox',
            lat: layer.data.lat,
            lon: layer.data.lon,
            mode: 'lines',
            line: {
              color: layer.style.color,
              width: layer.style.line?.width || 2,
            },
            opacity: layer.style.opacity,
            name: layer.name,
            hovertemplate: `<b>${layer.name}</b><br>` +
              `Lat: %{lat:.4f}°<br>` +
              `Lon: %{lon:.4f}°<br>` +
              `<extra></extra>`,
          } as Data);
          break;

        case 'choropleth':
          // For choropleth maps, we'd need GeoJSON data
          traces.push({
            type: 'choroplethmapbox',
            geojson: layer.data.customdata?.[0], // Assuming GeoJSON is in customdata
            locations: layer.data.text,
            z: layer.data.z,
            colorscale: layer.style.colorscale || 'Blues',
            opacity: layer.style.opacity,
            name: layer.name,
            hovertemplate: `<b>${layer.name}</b><br>` +
              `Location: %{location}<br>` +
              `Value: %{z:.3f}<br>` +
              `<extra></extra>`,
          } as Data);
          break;
      }
    });

    // Add measurement overlays - removed as measurements state was removed
    // measurements.forEach((measurement) => {
    // Removed measurement rendering code
    // });

    return traces;
  }, [layers, mapConfig]);

  // Generate layout configuration
  const mapLayout = useMemo((): Partial<Layout> => ({
    mapbox: {
      style: getMapboxStyle(mapConfig.basemap),
      center: mapConfig.center,
      zoom: mapConfig.zoom,
      bearing: 0,
      pitch: 0,
    },
    height,
    margin: { l: 0, r: 0, t: 40, b: 0 },
    showlegend: true,
    legend: { x: 0, y: 1 },
    title: 'High-Resolution Geological Map',
    hovermode: 'closest',
  }), [mapConfig, height]);

  // Get Mapbox style based on basemap selection
  function getMapboxStyle(basemap: string): string {
    switch (basemap) {
      case 'satellite': return 'satellite-streets';
      case 'terrain': return 'outdoors';
      case 'street': return 'streets';
      case 'topo': return 'outdoors';
      case 'ocean': return 'satellite';
      default: return 'satellite-streets';
    }
  }

  // Handle layer visibility toggle
  const toggleLayerVisibility = useCallback((layerId: string) => {
    const updatedLayers = layers.map(layer =>
      layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
    );
    onLayerUpdate?.(updatedLayers);
  }, [layers, onLayerUpdate]);

  // Handle layer opacity change
  const handleOpacityChange = useCallback((layerId: string, opacity: number) => {
    const updatedLayers = layers.map(layer =>
      layer.id === layerId
        ? { ...layer, style: { ...layer.style, opacity } }
        : layer
    );
    onLayerUpdate?.(updatedLayers);
  }, [layers, onLayerUpdate]);

  // Handle map click for measurements
  const handleMapClick = useCallback((event: any) => {
    if (measurementMode === 'off') return;

    const clickedPoint = [event.points[0].lon, event.points[0].lat];

    // Implementation would depend on measurement mode
    // This is a simplified example
    if (measurementMode === 'distance') {
      // Add point to current measurement or start new one
      console.log('Distance measurement at:', clickedPoint);
    } else if (measurementMode === 'area') {
      // Add point to polygon measurement
      console.log('Area measurement at:', clickedPoint);
    }

    onMapInteraction?.(event);
  }, [measurementMode, onMapInteraction]);

  // Export map functionality
  const exportMap = useCallback(async (format: 'png' | 'pdf' | 'svg') => {
    if (!plotRef.current) return;

    try {
      setIsLoading(true);

      await plotRef.current.downloadImage({
        format,
        width: 1920,
        height: 1080,
        filename: `geological_map_${Date.now()}`,
      });
    } catch (err) {
      setError('Export failed');
      console.error('Export error:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Get current map bounds - removed unused function
  // const getCurrentBounds = useCallback(() => {
  // if (!plotRef.current) return null;

  // This would extract current map bounds from the plot
  // return {
  //   north: mapConfig.center.lat + 1,
  //   south: mapConfig.center.lat - 1,
  //   east: mapConfig.center.lon + 1,
  //   west: mapConfig.center.lon - 1,
  // };
  // }, [mapConfig]);

  return (
    <Box>
      {/* Controls */}
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Typography variant="h6">High-Resolution Geological Map</Typography>
          </Grid>

          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Basemap</InputLabel>
              <Select
                value={mapConfig.basemap}
                onChange={(e) => setMapConfig(prev => ({
                  ...prev,
                  basemap: e.target.value as any
                }))}
              >
                <MenuItem value="satellite">Satellite</MenuItem>
                <MenuItem value="terrain">Terrain</MenuItem>
                <MenuItem value="street">Street</MenuItem>
                <MenuItem value="topo">Topographic</MenuItem>
                <MenuItem value="ocean">Ocean</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Projection</InputLabel>
              <Select
                value={mapConfig.projection}
                onChange={(e) => setMapConfig(prev => ({
                  ...prev,
                  projection: e.target.value as any
                }))}
              >
                <MenuItem value="mercator">Mercator</MenuItem>
                <MenuItem value="orthographic">Orthographic</MenuItem>
                <MenuItem value="natural earth">Natural Earth</MenuItem>
                <MenuItem value="albers usa">Albers USA</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {enableMeasurement && (
            <Grid item>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Measure</InputLabel>
                <Select
                  value={measurementMode}
                  onChange={(e) => setMeasurementMode(e.target.value as any)}
                >
                  <MenuItem value="off">Off</MenuItem>
                  <MenuItem value="distance">Distance</MenuItem>
                  <MenuItem value="area">Area</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          )}

          <Grid item>
            <Box display="flex" gap={1}>
              <Tooltip title="Zoom to fit all layers">
                <IconButton size="small">
                  <MyLocation />
                </IconButton>
              </Tooltip>

              <Tooltip title="Refresh layers">
                <IconButton size="small" onClick={() => setIsLoading(true)}>
                  <Refresh />
                </IconButton>
              </Tooltip>

              <Tooltip title="Settings">
                <IconButton size="small" onClick={() => setShowSettings(true)}>
                  <Settings />
                </IconButton>
              </Tooltip>

              <Tooltip title="Export map">
                <IconButton size="small" onClick={() => exportMap('png')}>
                  <Download />
                </IconButton>
              </Tooltip>

              <Tooltip title="Fullscreen">
                <IconButton size="small">
                  <Fullscreen />
                </IconButton>
              </Tooltip>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Grid container spacing={2}>
        {/* Layer Controls */}
        <Grid item xs={12} md={3}>
          <Paper elevation={1} sx={{ p: 2, height: height - 100, overflowY: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Map Layers
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
                      {layer.visible ? <Visibility /> : <VisibilityOff />}
                    </IconButton>
                    <Box
                      sx={{
                        width: 16,
                        height: 16,
                        backgroundColor: layer.style.color,
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
                      value={layer.style.opacity}
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
                      Resolution: {layer.metadata.resolution}
                    </Typography>
                    <Typography variant="caption" display="block">
                      Features: {layer.data.lat.length.toLocaleString()}
                    </Typography>
                    <Typography variant="caption" display="block">
                      Updated: {layer.metadata.timestamp.toLocaleString()}
                    </Typography>
                  </Box>
                </AccordionDetails>
              </Accordion>
            ))}

            {/* Measurements */}
            {/* Removed measurements section as state was removed */}
            {false && (
              <Box mt={2}>
                <Typography variant="h6" gutterBottom>
                  Measurements
                </Typography>
                <List dense>
                  {[].map((measurement: any) => (
                    <ListItem key={measurement.id}>
                      <ListItemIcon>
                        {measurement.type === 'distance' ? <Timeline /> : <Terrain />}
                      </ListItemIcon>
                      <ListItemText
                        primary={`${measurement.type}: ${measurement.value.toFixed(2)} ${measurement.unit}`}
                        secondary={`${measurement.coordinates.length} points`}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* Map Visualization */}
        <Grid item xs={12} md={9}>
          <Paper elevation={1}>
            <Plot
              ref={plotRef}
              data={mapTraces}
              layout={mapLayout}
              config={{
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                displaylogo: false,
                responsive: true,
                mapboxAccessToken: process.env.REACT_APP_MAPBOX_TOKEN,
              }}
              style={{ width: '100%', height: `${height}px` }}
              onClick={handleMapClick}
              onHover={(event) => setHoveredFeature(event.points[0])}
              onRelayout={(eventData) => {
                if ((eventData as any)['mapbox.center']) {
                  setMapConfig(prev => ({
                    ...prev,
                    center: (eventData as any)['mapbox.center'],
                    zoom: (eventData as any)['mapbox.zoom'] || prev.zoom,
                  }));
                }
              }}
            />
          </Paper>
        </Grid>
      </Grid>

      {/* Feature Info Panel */}
      {hoveredFeature && (
        <Paper
          elevation={3}
          sx={{
            position: 'absolute',
            top: 100,
            right: 20,
            p: 2,
            maxWidth: 300,
            zIndex: 1000,
          }}
        >
          <Typography variant="h6" gutterBottom>
            Feature Information
          </Typography>
          <Typography variant="body2">
            Latitude: {hoveredFeature.lat?.toFixed(4)}°
          </Typography>
          <Typography variant="body2">
            Longitude: {hoveredFeature.lon?.toFixed(4)}°
          </Typography>
          {hoveredFeature.z && (
            <Typography variant="body2">
              Value: {hoveredFeature.z.toFixed(3)}
            </Typography>
          )}
          {hoveredFeature.text && (
            <Typography variant="body2">
              {hoveredFeature.text}
            </Typography>
          )}
        </Paper>
      )}

      {/* Settings Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="md" fullWidth>
        <DialogTitle>Map Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Display Options
              </Typography>
            </Grid>

            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={mapConfig.showScale}
                    onChange={(e) => setMapConfig(prev => ({
                      ...prev,
                      showScale: e.target.checked
                    }))}
                  />
                }
                label="Show Scale Bar"
              />
            </Grid>

            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={mapConfig.showCompass}
                    onChange={(e) => setMapConfig(prev => ({
                      ...prev,
                      showCompass: e.target.checked
                    }))}
                  />
                }
                label="Show Compass"
              />
            </Grid>

            <Grid item xs={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={mapConfig.showCoordinates}
                    onChange={(e) => setMapConfig(prev => ({
                      ...prev,
                      showCoordinates: e.target.checked
                    }))}
                  />
                }
                label="Show Coordinates"
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Map Center
              </Typography>
            </Grid>

            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Latitude"
                type="number"
                value={mapConfig.center.lat}
                onChange={(e) => setMapConfig(prev => ({
                  ...prev,
                  center: { ...prev.center, lat: parseFloat(e.target.value) }
                }))}
                inputProps={{ step: 0.0001 }}
              />
            </Grid>

            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Longitude"
                type="number"
                value={mapConfig.center.lon}
                onChange={(e) => setMapConfig(prev => ({
                  ...prev,
                  center: { ...prev.center, lon: parseFloat(e.target.value) }
                }))}
                inputProps={{ step: 0.0001 }}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography variant="caption">Zoom Level</Typography>
              <Slider
                value={mapConfig.zoom}
                onChange={(_: Event, value: number | number[]) =>
                  setMapConfig(prev => ({ ...prev, zoom: value as number }))
                }
                min={1}
                max={20}
                step={0.1}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSettings(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default HighResolutionMappingInterface;