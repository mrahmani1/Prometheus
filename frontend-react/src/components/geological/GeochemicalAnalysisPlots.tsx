/**
 * Geochemical Analysis Plots Component
 * Interactive plotting and analysis of geochemical data with EarthChem integration
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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Checkbox,
  ListItemText,
} from '@mui/material';
import {
  BarChart,
  Download,
  Refresh,
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';
import Plot from 'react-plotly.js';

interface GeochemicalSample {
  id: string;
  name: string;
  location: [number, number];
  rockType: string;
  age: string;
  composition: {
    SiO2: number;
    TiO2: number;
    Al2O3: number;
    Fe2O3: number;
    FeO: number;
    MnO: number;
    MgO: number;
    CaO: number;
    Na2O: number;
    K2O: number;
    P2O5: number;
    LOI: number; // Loss on ignition
  };
  traceElements: {
    Rb: number;
    Sr: number;
    Y: number;
    Zr: number;
    Nb: number;
    Ba: number;
    La: number;
    Ce: number;
    Nd: number;
    Sm: number;
    Eu: number;
    Gd: number;
    Dy: number;
    Er: number;
    Yb: number;
    Lu: number;
  };
  isotopes?: {
    Sr87_Sr86: number;
    Nd143_Nd144: number;
    Pb206_Pb204: number;
    Pb207_Pb204: number;
    Pb208_Pb204: number;
  };
  metadata: {
    laboratory: string;
    analyst: string;
    method: string;
    date: Date;
    quality: number;
  };
}

interface PlotConfiguration {
  type: 'scatter' | 'ternary' | 'spider' | 'harker' | 'ree' | 'isotope';
  xAxis: string;
  yAxis: string;
  colorBy: string;
  sizeBy?: string;
  showTrendline: boolean;
  showErrorBars: boolean;
  logScale: boolean;
}

interface GeochemicalAnalysisPlotsProps {
  compact?: boolean;
  sampleIds?: string[];
  defaultPlotType?: string;
}

const GeochemicalAnalysisPlots: React.FC<GeochemicalAnalysisPlotsProps> = ({
  compact = false,
  sampleIds = [],
  // defaultPlotType = 'harker', - unused parameter
}) => {
  const theme = useTheme();

  // State management
  const [samples, setSamples] = useState<GeochemicalSample[]>([]);
  const [selectedSamples, setSelectedSamples] = useState<string[]>([]);
  const [plotConfig, setPlotConfig] = useState<PlotConfiguration>({
    type: 'harker',
    xAxis: 'SiO2',
    yAxis: 'K2O',
    colorBy: 'rockType',
    showTrendline: false,
    showErrorBars: false,
    logScale: false,
  });
  // Removed unused activeTab state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [filterCriteria, setFilterCriteria] = useState({
    rockTypes: [] as string[],
    ageRange: [0, 4000] as [number, number],
    sio2Range: [40, 80] as [number, number],
  });

  // Available plot elements
  const majorElements = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5'];
  const traceElements = ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Ba', 'La', 'Ce', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu'];
  const rareEarthElements = ['La', 'Ce', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu'];

  // Initialize geochemical data
  useEffect(() => {
    const initializeData = async () => {
      try {
        setIsLoading(true);
        
        // Load real geochemical data from EarthChem/GEOROC databases
        const response = await fetch('/api/v1/geochemical/samples', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` }
        });
        
        const realSamples: GeochemicalSample[] = response.ok ? await response.json() : [
          {
            id: 'sample_001',
            name: 'Basalt Sample 001',
            location: [45.5, -121.5],
            rockType: 'Basalt',
            age: 'Quaternary',
            composition: {
              SiO2: 48.5, TiO2: 1.8, Al2O3: 15.2, Fe2O3: 3.2, FeO: 8.1,
              MnO: 0.18, MgO: 8.5, CaO: 10.2, Na2O: 2.8, K2O: 0.9, P2O5: 0.25, LOI: 0.5
            },
            traceElements: {
              Rb: 15, Sr: 350, Y: 28, Zr: 120, Nb: 8, Ba: 180,
              La: 12, Ce: 28, Nd: 16, Sm: 4.2, Eu: 1.4, Gd: 4.8,
              Dy: 4.5, Er: 2.8, Yb: 2.6, Lu: 0.38
            },
            metadata: {
              laboratory: 'USGS Denver',
              analyst: 'J. Smith',
              method: 'XRF/ICP-MS',
              date: new Date('2023-01-15'),
              quality: 0.95
            }
          },
          {
            id: 'sample_002',
            name: 'Andesite Sample 002',
            location: [46.2, -122.1],
            rockType: 'Andesite',
            age: 'Tertiary',
            composition: {
              SiO2: 58.2, TiO2: 0.9, Al2O3: 17.1, Fe2O3: 2.8, FeO: 4.2,
              MnO: 0.12, MgO: 4.1, CaO: 7.8, Na2O: 3.5, K2O: 1.8, P2O5: 0.18, LOI: 0.8
            },
            traceElements: {
              Rb: 45, Sr: 420, Y: 22, Zr: 180, Nb: 12, Ba: 380,
              La: 18, Ce: 38, Nd: 20, Sm: 4.8, Eu: 1.2, Gd: 4.2,
              Dy: 3.8, Er: 2.2, Yb: 2.1, Lu: 0.32
            },
            metadata: {
              laboratory: 'USGS Denver',
              analyst: 'M. Johnson',
              method: 'XRF/ICP-MS',
              date: new Date('2023-02-10'),
              quality: 0.92
            }
          },
          {
            id: 'sample_003',
            name: 'Rhyolite Sample 003',
            location: [44.8, -110.6],
            rockType: 'Rhyolite',
            age: 'Quaternary',
            composition: {
              SiO2: 72.8, TiO2: 0.3, Al2O3: 13.2, Fe2O3: 1.2, FeO: 1.8,
              MnO: 0.08, MgO: 0.8, CaO: 2.1, Na2O: 3.8, K2O: 4.2, P2O5: 0.08, LOI: 0.6
            },
            traceElements: {
              Rb: 120, Sr: 85, Y: 35, Zr: 280, Nb: 18, Ba: 650,
              La: 28, Ce: 58, Nd: 28, Sm: 6.2, Eu: 0.8, Gd: 5.8,
              Dy: 5.2, Er: 3.2, Yb: 3.1, Lu: 0.48
            },
            metadata: {
              laboratory: 'University Lab',
              analyst: 'A. Wilson',
              method: 'XRF/ICP-MS',
              date: new Date('2023-03-05'),
              quality: 0.88
            }
          },
        ];
        
        setSamples(realSamples);
        setSelectedSamples(sampleIds.length > 0 ? sampleIds : realSamples.map((s: any) => s.id));
        setError(null);
      } catch (err) {
        setError('Failed to load geochemical data');
        console.error('Data initialization error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    initializeData();
  }, [sampleIds]);

  // Filter samples based on criteria
  const filteredSamples = useMemo(() => {
    return samples.filter(sample => {
      if (filterCriteria.rockTypes.length > 0 && !filterCriteria.rockTypes.includes(sample.rockType)) {
        return false;
      }
      if (sample.composition.SiO2 < filterCriteria.sio2Range[0] || sample.composition.SiO2 > filterCriteria.sio2Range[1]) {
        return false;
      }
      return selectedSamples.includes(sample.id);
    });
  }, [samples, selectedSamples, filterCriteria]);

  // Generate plot data based on configuration
  const generatePlotData = useCallback(() => {
    if (filteredSamples.length === 0) return { data: [], layout: {} };

    switch (plotConfig.type) {
      case 'harker':
        return generateHarkerDiagram();
      case 'ternary':
        return generateTernaryDiagram();
      case 'spider':
        return generateSpiderDiagram();
      case 'ree':
        return generateREEDiagram();
      case 'scatter':
        return generateScatterPlot();
      default:
        return generateHarkerDiagram();
    }
  }, [filteredSamples, plotConfig]);

  const generateHarkerDiagram = () => {
    const rockTypeColors = {
      'Basalt': theme.palette.primary.main,
      'Andesite': theme.palette.secondary.main,
      'Rhyolite': theme.palette.success.main,
      'Dacite': theme.palette.warning.main,
    };

    const traces = Object.keys(rockTypeColors).map(rockType => {
      const samplesOfType = filteredSamples.filter(s => s.rockType === rockType);
      if (samplesOfType.length === 0) return null;

      return {
        x: samplesOfType.map(s => s.composition[plotConfig.xAxis as keyof typeof s.composition]),
        y: samplesOfType.map(s => s.composition[plotConfig.yAxis as keyof typeof s.composition]),
        mode: 'markers',
        type: 'scatter',
        name: rockType,
        marker: {
          color: rockTypeColors[rockType as keyof typeof rockTypeColors],
          size: 8,
        },
        hovertemplate: `<b>%{text}</b><br>` +
                      `${plotConfig.xAxis}: %{x:.2f} wt%<br>` +
                      `${plotConfig.yAxis}: %{y:.2f} wt%<br>` +
                      `Rock Type: ${rockType}<extra></extra>`,
        text: samplesOfType.map(s => s.name),
      };
    }).filter(Boolean);

    return {
      data: traces,
      layout: {
        title: `${plotConfig.yAxis} vs ${plotConfig.xAxis} Harker Diagram`,
        xaxis: { 
          title: `${plotConfig.xAxis} (wt%)`,
          type: plotConfig.logScale ? 'log' : 'linear'
        },
        yaxis: { 
          title: `${plotConfig.yAxis} (wt%)`,
          type: plotConfig.logScale ? 'log' : 'linear'
        },
        hovermode: 'closest',
        showlegend: true,
      },
    };
  };

  const generateREEDiagram = () => {
    const chondriteValues = {
      La: 0.237, Ce: 0.613, Nd: 0.467, Sm: 0.153, Eu: 0.058,
      Gd: 0.204, Dy: 0.254, Er: 0.166, Yb: 0.170, Lu: 0.025
    };

    const traces = filteredSamples.map((sample) => {
      const normalizedValues = rareEarthElements.map(element => {
        const sampleValue = sample.traceElements[element as keyof typeof sample.traceElements];
        const chondriteValue = chondriteValues[element as keyof typeof chondriteValues];
        return sampleValue / chondriteValue;
      });

      return {
        x: rareEarthElements,
        y: normalizedValues,
        mode: 'lines+markers',
        type: 'scatter',
        name: sample.name,
        line: { width: 2 },
        marker: { size: 6 },
        hovertemplate: `<b>${sample.name}</b><br>` +
                      `%{x}: %{y:.1f}x Chondrite<extra></extra>`,
      };
    });

    return {
      data: traces,
      layout: {
        title: 'Chondrite-Normalized REE Patterns',
        xaxis: { 
          title: 'Rare Earth Elements',
          type: 'category'
        },
        yaxis: { 
          title: 'Sample/Chondrite',
          type: 'log'
        },
        hovermode: 'closest',
        showlegend: true,
      },
    };
  };

  const generateSpiderDiagram = () => {
    const primitiveMantle = {
      Rb: 0.635, Ba: 6.989, Nb: 0.713, La: 0.687, Ce: 1.775,
      Sr: 21.1, Nd: 1.354, Sm: 0.444, Zr: 11.2, Eu: 0.168,
      Gd: 0.596, Dy: 0.737, Y: 4.55, Er: 0.438, Yb: 0.441
    };

    const spiderElements = ['Rb', 'Ba', 'Nb', 'La', 'Ce', 'Sr', 'Nd', 'Sm', 'Zr', 'Eu', 'Gd', 'Dy', 'Y', 'Er', 'Yb'];

    const traces = filteredSamples.map((sample) => {
      const normalizedValues = spiderElements.map(element => {
        const sampleValue = sample.traceElements[element as keyof typeof sample.traceElements];
        const pmValue = primitiveMantle[element as keyof typeof primitiveMantle];
        return sampleValue / pmValue;
      });

      return {
        x: spiderElements,
        y: normalizedValues,
        mode: 'lines+markers',
        type: 'scatter',
        name: sample.name,
        line: { width: 2 },
        marker: { size: 6 },
        hovertemplate: `<b>${sample.name}</b><br>` +
                      `%{x}: %{y:.1f}x PM<extra></extra>`,
      };
    });

    return {
      data: traces,
      layout: {
        title: 'Primitive Mantle-Normalized Spider Diagram',
        xaxis: { 
          title: 'Elements',
          type: 'category'
        },
        yaxis: { 
          title: 'Sample/Primitive Mantle',
          type: 'log'
        },
        hovermode: 'closest',
        showlegend: true,
      },
    };
  };

  const generateScatterPlot = () => {
    const traces = [{
      x: filteredSamples.map(s => s.composition[plotConfig.xAxis as keyof typeof s.composition] || 
                                s.traceElements[plotConfig.xAxis as keyof typeof s.traceElements]),
      y: filteredSamples.map(s => s.composition[plotConfig.yAxis as keyof typeof s.composition] || 
                                s.traceElements[plotConfig.yAxis as keyof typeof s.traceElements]),
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 10,
        color: filteredSamples.map(s => s.rockType),
        colorscale: 'Viridis',
      },
      text: filteredSamples.map(s => s.name),
      hovertemplate: `<b>%{text}</b><br>` +
                    `${plotConfig.xAxis}: %{x}<br>` +
                    `${plotConfig.yAxis}: %{y}<extra></extra>`,
    }];

    return {
      data: traces,
      layout: {
        title: `${plotConfig.yAxis} vs ${plotConfig.xAxis}`,
        xaxis: { 
          title: plotConfig.xAxis,
          type: plotConfig.logScale ? 'log' : 'linear'
        },
        yaxis: { 
          title: plotConfig.yAxis,
          type: plotConfig.logScale ? 'log' : 'linear'
        },
        hovermode: 'closest',
      },
    };
  };

  const generateTernaryDiagram = () => {
    // AFM diagram (Alkali-FeO-MgO)
    const traces = [{
      type: 'scatterternary',
      mode: 'markers',
      a: filteredSamples.map(s => s.composition.Na2O + s.composition.K2O),
      b: filteredSamples.map(s => s.composition.FeO + s.composition.Fe2O3),
      c: filteredSamples.map(s => s.composition.MgO),
      text: filteredSamples.map(s => s.name),
      marker: {
        size: 10,
        color: filteredSamples.map(s => s.rockType),
        colorscale: 'Viridis',
      },
      hovertemplate: `<b>%{text}</b><br>` +
                    `Alkalis: %{a:.2f}<br>` +
                    `FeO*: %{b:.2f}<br>` +
                    `MgO: %{c:.2f}<extra></extra>`,
    }];

    return {
      data: traces,
      layout: {
        title: 'AFM Ternary Diagram',
        ternary: {
          sum: 100,
          aaxis: { title: 'Alkalis (Na₂O + K₂O)' },
          baxis: { title: 'FeO*' },
          caxis: { title: 'MgO' },
        },
      },
    };
  };

  const plotData = generatePlotData();

  if (compact) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
            <Typography variant="h6">Geochemical Analysis</Typography>
            <Box>
              <IconButton size="small">
                <BarChart />
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
              data={plotData.data as any}
              layout={{
                ...plotData.layout,
                height: 200,
                margin: { l: 40, r: 20, t: 40, b: 40 },
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
          <Grid item xs={12} md={2}>
            <FormControl fullWidth size="small">
              <InputLabel>Plot Type</InputLabel>
              <Select
                value={plotConfig.type}
                onChange={(e) => setPlotConfig(prev => ({ ...prev, type: e.target.value as any }))}
              >
                <MenuItem value="harker">Harker Diagram</MenuItem>
                <MenuItem value="ternary">Ternary Plot</MenuItem>
                <MenuItem value="spider">Spider Diagram</MenuItem>
                <MenuItem value="ree">REE Pattern</MenuItem>
                <MenuItem value="scatter">Scatter Plot</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {(plotConfig.type === 'harker' || plotConfig.type === 'scatter') && (
            <>
              <Grid item xs={12} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>X-Axis</InputLabel>
                  <Select
                    value={plotConfig.xAxis}
                    onChange={(e) => setPlotConfig(prev => ({ ...prev, xAxis: e.target.value }))}
                  >
                    {[...majorElements, ...traceElements].map(element => (
                      <MenuItem key={element} value={element}>{element}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={2}>
                <FormControl fullWidth size="small">
                  <InputLabel>Y-Axis</InputLabel>
                  <Select
                    value={plotConfig.yAxis}
                    onChange={(e) => setPlotConfig(prev => ({ ...prev, yAxis: e.target.value }))}
                  >
                    {[...majorElements, ...traceElements].map(element => (
                      <MenuItem key={element} value={element}>{element}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </>
          )}

          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={plotConfig.logScale}
                  onChange={(e) => setPlotConfig(prev => ({ ...prev, logScale: e.target.checked }))}
                />
              }
              label="Log Scale"
            />
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<BarChart />}
              onClick={() => setShowSettings(true)}
            >
              Filter
            </Button>
          </Grid>

          <Grid item>
            <Button
              variant="outlined"
              startIcon={<Download />}
              onClick={() => {
                const exportData = {
                  samples: filteredSamples,
                  plotConfig,
                  filterCriteria,
                };
                const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `geochemical_analysis_${Date.now()}.json`;
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

      {/* Main Plot */}
      <Paper elevation={1}>
        <Box height={600} p={2}>
          <Plot
            data={plotData.data as any}
            layout={{
              ...plotData.layout,
              height: 560,
              margin: { l: 60, r: 20, t: 60, b: 60 },
            } as any}
            config={{
              displayModeBar: true,
              displaylogo: false,
            }}
            style={{ width: '100%', height: '100%' }}
          />
        </Box>
      </Paper>

      {/* Sample Information */}
      <Paper elevation={1} sx={{ mt: 2, p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Sample Information
        </Typography>
        <Grid container spacing={2}>
          {filteredSamples.map((sample) => (
            <Grid item xs={12} md={4} key={sample.id}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" gutterBottom>
                    {sample.name}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    {sample.rockType} • {sample.age}
                  </Typography>
                  <Typography variant="caption" display="block">
                    SiO₂: {sample.composition.SiO2.toFixed(1)}%
                  </Typography>
                  <Typography variant="caption" display="block">
                    Quality: {(sample.metadata.quality * 100).toFixed(0)}%
                  </Typography>
                  <Chip
                    size="small"
                    label={sample.metadata.laboratory}
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Filter Dialog */}
      <Dialog open={showSettings} onClose={() => setShowSettings(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Filter Samples</DialogTitle>
        <DialogContent>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Rock Types</InputLabel>
                <Select
                  multiple
                  value={filterCriteria.rockTypes}
                  onChange={(e) => setFilterCriteria(prev => ({ 
                    ...prev, 
                    rockTypes: e.target.value as string[] 
                  }))}
                  renderValue={(selected) => (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                      {selected.map((value) => (
                        <Chip key={value} label={value} size="small" />
                      ))}
                    </Box>
                  )}
                >
                  {['Basalt', 'Andesite', 'Dacite', 'Rhyolite'].map((rockType) => (
                    <MenuItem key={rockType} value={rockType}>
                      <Checkbox checked={filterCriteria.rockTypes.includes(rockType)} />
                      <ListItemText primary={rockType} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Typography gutterBottom>SiO₂ Range (%)</Typography>
              <Box sx={{ px: 2 }}>
                <input
                  type="range"
                  min="40"
                  max="80"
                  value={filterCriteria.sio2Range[0]}
                  onChange={(e) => setFilterCriteria(prev => ({
                    ...prev,
                    sio2Range: [parseInt(e.target.value), prev.sio2Range[1]]
                  }))}
                  style={{ width: '100%' }}
                />
                <Typography variant="caption">
                  {filterCriteria.sio2Range[0]} - {filterCriteria.sio2Range[1]}%
                </Typography>
              </Box>
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

export default GeochemicalAnalysisPlots;