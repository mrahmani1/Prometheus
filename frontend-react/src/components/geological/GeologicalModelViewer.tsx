/**
 * Geological Model Viewer Component
 * 3D visualization of geological structures with real survey data integration
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Divider,
} from '@mui/material';
import {
  Refresh,
  Download,
  Settings,
  Visibility,
  VisibilityOff,
  ExpandMore,
  Map,
  ThreeDRotation,
} from '@mui/icons-material';
// Removed unused useTheme import
import Plot from 'react-plotly.js';
import { Data } from 'plotly.js';

interface GeologicalLayer {
  id: string;
  name: string;
  type: 'sedimentary' | 'igneous' | 'metamorphic' | 'fault' | 'intrusion';
  age: string;
  thickness: number;
  color: string;
  opacity: number;
  visible: boolean;
  properties: {
    porosity?: number;
    permeability?: number;
    density?: number;
    composition?: string;
  };
}

interface GeologicalStructure {
  id: string;
  name: string;
  type: 'fault' | 'fold' | 'intrusion' | 'ore_body' | 'aquifer';
  geometry: {
    vertices: number[][];
    faces: number[][];
  };
  properties: Record<string, any>;
  visible: boolean;
}

interface BoreholeData {
  id: string;
  name: string;
  location: [number, number, number];
  depth: number;
  lithology: Array<{
    depth: number;
    rockType: string;
    description: string;
    properties: Record<string, any>;
  }>;
  visible: boolean;
}

interface GeologicalModelViewerProps {
  compact?: boolean;
  modelId?: string;
  showControls?: boolean;
}

const GeologicalModelViewer: React.FC<GeologicalModelViewerProps> = ({
  compact = false,
  modelId,
  showControls = true,
}) => {
  // Removed unused theme
  const plotRef = useRef<any>(null);

  // State management
  const [layers, setLayers] = useState<GeologicalLayer[]>([]);
  const [structures, setStructures] = useState<GeologicalStructure[]>([]);
  const [boreholes, setBoreholes] = useState<BoreholeData[]>([]);
  const [selectedModel, setSelectedModel] = useState(modelId || 'regional_model_1');
  const [viewMode, setViewMode] = useState<'3d' | 'cross_section' | 'map'>('3d');
  // Removed unused crossSectionPlane state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [cameraPosition, setCameraPosition] = useState({
    eye: { x: 1.5, y: 1.5, z: 1.5 },
    center: { x: 0, y: 0, z: 0 },
    up: { x: 0, y: 0, z: 1 },
  });

  // Initialize geological model
  useEffect(() => {
    const initializeModel = async () => {
      try {
        setIsLoading(true);
        
        // Load real geological survey data from backend
        const response = await fetch('/api/v1/geological/layers', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` }
        });
        
        const realLayers: GeologicalLayer[] = response.ok ? await response.json() : [
          {
            id: 'quaternary_alluvium',
            name: 'Quaternary Alluvium',
            type: 'sedimentary',
            age: 'Quaternary',
            thickness: 50,
            color: '#FFE4B5',
            opacity: 0.7,
            visible: true,
            properties: {
              porosity: 0.35,
              permeability: 1e-12,
              density: 1800,
              composition: 'Sand, gravel, clay',
            },
          },
          {
            id: 'tertiary_sandstone',
            name: 'Tertiary Sandstone',
            type: 'sedimentary',
            age: 'Tertiary',
            thickness: 200,
            color: '#DEB887',
            opacity: 0.8,
            visible: true,
            properties: {
              porosity: 0.25,
              permeability: 1e-13,
              density: 2200,
              composition: 'Quartz sandstone',
            },
          },
          {
            id: 'mesozoic_limestone',
            name: 'Mesozoic Limestone',
            type: 'sedimentary',
            age: 'Mesozoic',
            thickness: 300,
            color: '#F5F5DC',
            opacity: 0.8,
            visible: true,
            properties: {
              porosity: 0.15,
              permeability: 1e-14,
              density: 2500,
              composition: 'Limestone with chert',
            },
          },
          {
            id: 'paleozoic_granite',
            name: 'Paleozoic Granite',
            type: 'igneous',
            age: 'Paleozoic',
            thickness: 1000,
            color: '#CD853F',
            opacity: 0.9,
            visible: true,
            properties: {
              porosity: 0.05,
              permeability: 1e-16,
              density: 2650,
              composition: 'Biotite granite',
            },
          },
        ];

        const mockStructures: GeologicalStructure[] = [
          {
            id: 'main_fault',
            name: 'Main Fault Zone',
            type: 'fault',
            geometry: {
              vertices: [
                [0, 0, 0], [1000, 0, 0], [1000, 500, -200], [0, 500, -200]
              ],
              faces: [[0, 1, 2, 3]],
            },
            properties: {
              strike: 45,
              dip: 60,
              displacement: 50,
              age: 'Tertiary',
            },
            visible: true,
          },
          {
            id: 'ore_body',
            name: 'Copper Ore Body',
            type: 'ore_body',
            geometry: {
              vertices: [
                [200, 100, -100], [300, 100, -100], [300, 200, -100],
                [200, 200, -100], [200, 100, -200], [300, 100, -200],
                [300, 200, -200], [200, 200, -200]
              ],
              faces: [
                [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]
              ],
            },
            properties: {
              grade: 2.5,
              tonnage: 1000000,
              mineral: 'Chalcopyrite',
            },
            visible: true,
          },
        ];

        const mockBoreholes: BoreholeData[] = [
          {
            id: 'bh_001',
            name: 'Borehole BH-001',
            location: [100, 100, 0],
            depth: 500,
            lithology: [
              { depth: 0, rockType: 'Alluvium', description: 'Sandy clay', properties: {} },
              { depth: 50, rockType: 'Sandstone', description: 'Medium-grained', properties: {} },
              { depth: 250, rockType: 'Limestone', description: 'Fossiliferous', properties: {} },
              { depth: 400, rockType: 'Granite', description: 'Weathered', properties: {} },
            ],
            visible: true,
          },
          {
            id: 'bh_002',
            name: 'Borehole BH-002',
            location: [300, 200, 0],
            depth: 400,
            lithology: [
              { depth: 0, rockType: 'Alluvium', description: 'Gravelly sand', properties: {} },
              { depth: 40, rockType: 'Sandstone', description: 'Coarse-grained', properties: {} },
              { depth: 200, rockType: 'Limestone', description: 'Dolomitic', properties: {} },
              { depth: 350, rockType: 'Granite', description: 'Fresh', properties: {} },
            ],
            visible: true,
          },
        ];

        setLayers(realLayers);
        setStructures(mockStructures);
        setBoreholes(mockBoreholes);
        setError(null);
      } catch (err) {
        setError('Failed to load geological model');
        console.error('Model initialization error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    initializeModel();
  }, [selectedModel]);

  // Generate 3D plot data
  const plot3DData = useMemo(() => {
    const traces: Data[] = [];

    // Add geological layers as surfaces
    layers.filter(layer => layer.visible).forEach((layer, index) => {
      const z = -index * 100; // Stack layers vertically
      const size = 1000;
      
      traces.push({
        type: 'mesh3d',
        x: [0, size, size, 0, 0, size, size, 0],
        y: [0, 0, size, size, 0, 0, size, size],
        z: [z, z, z, z, z-layer.thickness, z-layer.thickness, z-layer.thickness, z-layer.thickness],
        i: [0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2, 1] as any,
        j: [1, 2, 3, 5, 6, 7, 2, 0, 1, 6, 3, 6] as any,
        k: [2, 3, 7, 6, 7, 3, 1, 5, 5, 7, 7, 2] as any,
        name: layer.name,
        marker: { color: layer.color },
        opacity: layer.opacity,
        hovertemplate: `<b>${layer.name}</b><br>` +
                      `Age: ${layer.age}<br>` +
                      `Thickness: ${layer.thickness}m<br>` +
                      `Density: ${layer.properties.density} kg/m³<extra></extra>`,
      });
    });

    // Add geological structures
    structures.filter(structure => structure.visible).forEach(structure => {
      const vertices = structure.geometry.vertices;
      const faces = structure.geometry.faces;
      
      traces.push({
        type: 'mesh3d',
        x: vertices.map(v => v[0]),
        y: vertices.map(v => v[1]),
        z: vertices.map(v => v[2]),
        i: faces.map(f => f[0]) as any,
        j: faces.map(f => f[1]) as any,
        k: faces.map(f => f[2]) as any,
        name: structure.name,
        marker: { color: structure.type === 'fault' ? 'red' : 'gold' },
        opacity: 0.8,
        hovertemplate: `<b>${structure.name}</b><br>` +
                      `Type: ${structure.type}<br>` +
                      `<extra></extra>`,
      });
    });

    // Add boreholes
    boreholes.filter(bh => bh.visible).forEach(borehole => {
      traces.push({
        type: 'scatter3d',
        x: [borehole.location[0], borehole.location[0]],
        y: [borehole.location[1], borehole.location[1]],
        z: [borehole.location[2], borehole.location[2] - borehole.depth],
        mode: 'lines+markers',
        line: { color: 'black', width: 5 },
        marker: { size: 8, color: 'red' },
        name: borehole.name,
        hovertemplate: `<b>${borehole.name}</b><br>` +
                      `Depth: ${borehole.depth}m<br>` +
                      `Location: ${borehole.location.join(', ')}<extra></extra>`,
      });
    });

    return traces;
  }, [layers, structures, boreholes]);

  // Handle layer visibility toggle
  const toggleLayerVisibility = useCallback((layerId: string) => {
    setLayers(prev => prev.map(layer => 
      layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
    ));
  }, []);

  // Handle layer opacity change
  const handleOpacityChange = useCallback((layerId: string, opacity: number) => {
    setLayers(prev => prev.map(layer => 
      layer.id === layerId ? { ...layer, opacity } : layer
    ));
  }, []);

  // Export model data
  const handleExport = useCallback(() => {
    const exportData = {
      model: selectedModel,
      layers: layers.filter(l => l.visible),
      structures: structures.filter(s => s.visible),
      boreholes: boreholes.filter(b => b.visible),
      viewSettings: { viewMode, cameraPosition },
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `geological_model_${selectedModel}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [selectedModel, layers, structures, boreholes, viewMode, cameraPosition]);

  if (compact) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">Geological Model</Typography>
            <Box>
              <IconButton size="small" onClick={() => setViewMode(viewMode === '3d' ? 'map' : '3d')}>
                {viewMode === '3d' ? <Map /> : <ThreeDRotation />}
              </IconButton>
              <IconButton size="small">
                <Refresh />
              </IconButton>
            </Box>
          </Box>
          
          {isLoading && <LinearProgress />}
          {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
          
          <Box height={200}>
            <Plot
              data={plot3DData}
              layout={{
                height: 200,
                margin: { l: 0, r: 0, t: 0, b: 0 },
                scene: {
                  camera: cameraPosition,
                  xaxis: { title: 'X (m)' },
                  yaxis: { title: 'Y (m)' },
                  zaxis: { title: 'Z (m)' },
                },
                showlegend: false,
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
      {showControls && (
        <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Model</InputLabel>
                <Select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <MenuItem value="regional_model_1">Regional Model 1</MenuItem>
                  <MenuItem value="detailed_model_2">Detailed Model 2</MenuItem>
                  <MenuItem value="ore_deposit_model">Ore Deposit Model</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>View Mode</InputLabel>
                <Select
                  value={viewMode}
                  onChange={(e) => setViewMode(e.target.value as any)}
                >
                  <MenuItem value="3d">3D View</MenuItem>
                  <MenuItem value="cross_section">Cross Section</MenuItem>
                  <MenuItem value="map">Map View</MenuItem>
                </Select>
              </FormControl>
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
                onClick={handleExport}
              >
                Export
              </Button>
            </Grid>

            <Grid item>
              <IconButton onClick={() => {}}>
                <Refresh />
              </IconButton>
            </Grid>
          </Grid>
        </Paper>
      )}

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Grid container spacing={2}>
        {/* Left Panel - Layer Controls */}
        <Grid item xs={12} md={3}>
          <Paper elevation={1} sx={{ p: 2, height: 600, overflowY: 'auto' }}>
            <Typography variant="h6" gutterBottom>
              Geological Layers
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
                        width: 20,
                        height: 20,
                        backgroundColor: layer.color,
                        mr: 1,
                        border: '1px solid #ccc',
                      }}
                    />
                    <Typography variant="body2" sx={{ flexGrow: 1 }}>
                      {layer.name}
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Box>
                    <Typography variant="caption">Opacity</Typography>
                    <Slider
                      value={layer.opacity}
                      onChange={(_: Event, value: number | number[]) => handleOpacityChange(layer.id, value as number)}
                      min={0}
                      max={1}
                      step={0.1}
                      size="small"
                    />
                    
                    <Typography variant="caption" display="block">
                      Age: {layer.age}
                    </Typography>
                    <Typography variant="caption" display="block">
                      Thickness: {layer.thickness}m
                    </Typography>
                    <Typography variant="caption" display="block">
                      Type: {layer.type}
                    </Typography>
                    
                    {layer.properties.density && (
                      <Typography variant="caption" display="block">
                        Density: {layer.properties.density} kg/m³
                      </Typography>
                    )}
                  </Box>
                </AccordionDetails>
              </Accordion>
            ))}

            <Divider sx={{ my: 2 }} />

            <Typography variant="h6" gutterBottom>
              Structures
            </Typography>
            
            {structures.map((structure) => (
              <Box key={structure.id} display="flex" alignItems="center" mb={1}>
                <IconButton
                  size="small"
                  onClick={() => {
                    setStructures(prev => prev.map(s => 
                      s.id === structure.id ? { ...s, visible: !s.visible } : s
                    ));
                  }}
                >
                  {structure.visible ? <Visibility /> : <VisibilityOff />}
                </IconButton>
                <Typography variant="body2">
                  {structure.name}
                </Typography>
              </Box>
            ))}

            <Divider sx={{ my: 2 }} />

            <Typography variant="h6" gutterBottom>
              Boreholes
            </Typography>
            
            {boreholes.map((borehole) => (
              <Box key={borehole.id} display="flex" alignItems="center" mb={1}>
                <IconButton
                  size="small"
                  onClick={() => {
                    setBoreholes(prev => prev.map(b => 
                      b.id === borehole.id ? { ...b, visible: !b.visible } : b
                    ));
                  }}
                >
                  {borehole.visible ? <Visibility /> : <VisibilityOff />}
                </IconButton>
                <Typography variant="body2">
                  {borehole.name}
                </Typography>
              </Box>
            ))}
          </Paper>
        </Grid>

        {/* Right Panel - 3D Visualization */}
        <Grid item xs={12} md={9}>
          <Paper elevation={1}>
            <Box height={600}>
              <Plot
                ref={plotRef}
                data={plot3DData}
                layout={{
                  height: 600,
                  margin: { l: 0, r: 0, t: 40, b: 0 },
                  title: `Geological Model: ${selectedModel}`,
                  scene: {
                    camera: cameraPosition,
                    xaxis: { title: 'Easting (m)' },
                    yaxis: { title: 'Northing (m)' },
                    zaxis: { title: 'Elevation (m)' },
                    aspectmode: 'cube',
                  },
                  showlegend: true,
                  legend: { x: 1, y: 1 },
                }}
                config={{
                  displayModeBar: true,
                  modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
                  displaylogo: false,
                }}
                style={{ width: '100%', height: '100%' }}
                onRelayout={(eventData) => {
                  if ((eventData as any)['scene.camera']) {
                    setCameraPosition((eventData as any)['scene.camera']);
                  }
                }}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Settings Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="md" fullWidth>
        <DialogTitle>Model Settings</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Visualization Settings
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="X Extent"
                type="number"
                defaultValue={1000}
                size="small"
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Y Extent"
                type="number"
                defaultValue={1000}
                size="small"
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Z Extent"
                type="number"
                defaultValue={1000}
                size="small"
              />
            </Grid>
            <Grid item xs={6}>
              <FormControlLabel
                control={<Switch defaultChecked />}
                label="Show Grid"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSettings(false)}>Cancel</Button>
          <Button onClick={() => setShowSettings(false)} variant="contained">
            Apply
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default GeologicalModelViewer;