import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import Plot from 'react-plotly.js';
import { 
  Box, 
  Paper, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Switch, 
  FormControlLabel,
  Slider,
  Grid,
  Chip
} from '@mui/material';
import { GeologicalData, SeismicEvent, VolcanicState } from '../../../types/geological';

interface ScientificChartsProps {
  data: GeologicalData[];
  seismicEvents?: SeismicEvent[];
  volcanicStates?: VolcanicState[];
  chartType?: 'timeseries' | 'scatter' | 'heatmap' | 'seismogram' | 'spectrogram' | 'hodogram';
  onDataSelect?: (selectedData: any[]) => void;
  realTimeUpdate?: boolean;
  exportFormats?: string[];
}

// Advanced geological color schemes for scientific visualization
const SCIENTIFIC_COLOR_SCHEMES = {
  geological: {
    igneous: '#FF6B6B',
    sedimentary: '#4ECDC4',
    metamorphic: '#45B7D1',
    volcanic: '#FF8E53'
  },
  seismic: {
    pWave: '#FF4757',
    sWave: '#3742FA',
    surface: '#2ED573'
  },
  geophysical: {
    velocity: d3.scaleSequential(d3.interpolateViridis).domain([2000, 8000]),
    density: d3.scaleSequential(d3.interpolatePlasma).domain([1000, 5000]),
    temperature: d3.scaleSequential(d3.interpolateInferno).domain([0, 1500])
  }
};

// Advanced D3.js seismogram visualization
const SeismogramChart: React.FC<{
  data: SeismicEvent[];
  width: number;
  height: number;
  onWaveformSelect?: (event: SeismicEvent) => void;
}> = ({ data, width, height, onWaveformSelect }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [_selectedEvent, _setSelectedEvent] = useState<SeismicEvent | null>(null);
  const [timeWindow, setTimeWindow] = useState([0, 300]); // seconds
  const [amplitudeScale, setAmplitudeScale] = useState(1);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Time scale
    const xScale = d3.scaleLinear()
      .domain(timeWindow)
      .range([0, innerWidth]);

    // Amplitude scale (normalized)
    const yScale = d3.scaleLinear()
      .domain([-1 * amplitudeScale, 1 * amplitudeScale])
      .range([innerHeight, 0]);

    // Create axes
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d => `${d}s`);
    const yAxis = d3.axisLeft(yScale);

    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 35)
      .style('text-anchor', 'middle')
      .text('Time (seconds)');

    g.append('g')
      .call(yAxis)
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -35)
      .attr('x', -innerHeight / 2)
      .style('text-anchor', 'middle')
      .text('Amplitude (normalized)');

    // Draw seismogram traces
    data.forEach((event, index) => {
      if (!event.waveformData) return;

      const yOffset = (index * innerHeight) / data.length;
      const traceHeight = innerHeight / data.length * 0.8;

      // Generate synthetic waveform if not available
      const waveform = event.waveformData || generateSyntheticWaveform(event);
      const timeStep = (timeWindow[1] - timeWindow[0]) / waveform.length;

      const line = d3.line<number>()
        .x((_d, i) => xScale(timeWindow[0] + i * timeStep))
        .y(d => yOffset + traceHeight / 2 - (d * traceHeight / 2))
        .curve(d3.curveLinear);

      // Draw waveform
      g.append('path')
        .datum(Array.from(waveform))
        .attr('fill', 'none')
        .attr('stroke', SCIENTIFIC_COLOR_SCHEMES.seismic.pWave)
        .attr('stroke-width', 1.5)
        .attr('d', line)
        .style('cursor', 'pointer')
        .on('click', () => {
          _setSelectedEvent(event);
          onWaveformSelect?.(event);
        })
        .on('mouseover', function() {
          d3.select(this).attr('stroke-width', 3);
        })
        .on('mouseout', function() {
          d3.select(this).attr('stroke-width', 1.5);
        });

      // Add event label
      g.append('text')
        .attr('x', 5)
        .attr('y', yOffset + 15)
        .style('font-size', '12px')
        .text(`M${event.magnitude.toFixed(1)} - ${event.eventId}`);

      // Mark P and S wave arrivals
      if (event.momentTensor?.pAxis) {
        g.append('line')
          .attr('x1', xScale(event.timestamp.getTime()))
          .attr('x2', xScale(event.timestamp.getTime()))
          .attr('y1', yOffset)
          .attr('y2', yOffset + traceHeight)
          .attr('stroke', SCIENTIFIC_COLOR_SCHEMES.seismic.pWave)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5');
      }

      if (event.momentTensor?.tAxis) {
        g.append('line')
          .attr('x1', xScale(event.timestamp.getTime() + 1000))
          .attr('x2', xScale(event.timestamp.getTime() + 1000))
          .attr('y1', yOffset)
          .attr('y2', yOffset + traceHeight)
          .attr('stroke', SCIENTIFIC_COLOR_SCHEMES.seismic.sWave)
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5');
      }
    });

  }, [data, width, height, timeWindow, amplitudeScale]);

  // Generate synthetic waveform for demonstration
  const generateSyntheticWaveform = (event: SeismicEvent): Float32Array => {
    const length = 1000;
    const waveform = new Float32Array(length);
    const frequency = 1 + event.magnitude * 0.5; // Hz
    const amplitude = event.magnitude / 10;

    for (let i = 0; i < length; i++) {
      const t = i / 100; // 100 Hz sampling
      waveform[i] = amplitude * Math.sin(2 * Math.PI * frequency * t) * Math.exp(-t * 0.1);
    }

    return waveform;
  };

  return (
    <Box>
      <svg ref={svgRef} width={width} height={height} />
      <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
        <Typography variant="body2">Time Window (s):</Typography>
        <Slider
          value={timeWindow}
          onChange={(_: Event, value: number | number[]) => setTimeWindow(value as number[])}
          min={0}
          max={600}
          step={10}
          sx={{ width: 200 }}
        />
        <Typography variant="body2">Amplitude Scale:</Typography>
        <Slider
          value={amplitudeScale}
          onChange={(_: Event, value: number | number[]) => setAmplitudeScale(value as number)}
          min={0.1}
          max={10}
          step={0.1}
          sx={{ width: 100 }}
        />
      </Box>
    </Box>
  );
};

// Advanced Plotly.js spectrogram visualization
const SpectrogramChart: React.FC<{
  data: SeismicEvent[];
  width: number;
  height: number;
}> = ({ data, width, height }) => {
  const spectrogramData = useMemo(() => {
    if (!data.length || !data[0].waveformData) return null;

    // Generate spectrogram data using FFT
    const event = data[0];
    const waveform = event.waveformData || new Float32Array(1000);
    const windowSize = 256;
    const overlap = 128;
    const sampleRate = 100; // Hz

    const spectrogram: number[][] = [];
    const frequencies: number[] = [];
    const times: number[] = [];

    // Generate frequency bins
    for (let i = 0; i < windowSize / 2; i++) {
      frequencies.push((i * sampleRate) / windowSize);
    }

    // Sliding window FFT
    for (let start = 0; start < waveform.length - windowSize; start += overlap) {
      const window = waveform.slice(start, start + windowSize);
      const fft = computeFFT(window);
      const magnitude = fft.map(complex => Math.sqrt(complex.real ** 2 + complex.imag ** 2));
      
      spectrogram.push(magnitude.slice(0, windowSize / 2));
      times.push(start / sampleRate);
    }

    return {
      z: spectrogram.map(row => row.map(val => 20 * Math.log10(val + 1e-10))), // Convert to dB
      x: times,
      y: frequencies,
      type: 'heatmap' as const,
      colorscale: 'Viridis',
      colorbar: {
        title: { text: 'Power (dB)' },
        titleside: 'right'
      }
    };
  }, [data]);

  // Simple FFT implementation (for demonstration - use a proper library in production)
  const computeFFT = (signal: Float32Array): { real: number; imag: number }[] => {
    const N = signal.length;
    const result: { real: number; imag: number }[] = [];

    for (let k = 0; k < N; k++) {
      let real = 0;
      let imag = 0;

      for (let n = 0; n < N; n++) {
        const angle = -2 * Math.PI * k * n / N;
        real += signal[n] * Math.cos(angle);
        imag += signal[n] * Math.sin(angle);
      }

      result.push({ real, imag });
    }

    return result;
  };

  if (!spectrogramData) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height }}>
        <Typography>No waveform data available for spectrogram</Typography>
      </Box>
    );
  }

  return (
    <Plot
      data={[spectrogramData]}
      layout={{
        title: 'Seismic Spectrogram',
        xaxis: { title: 'Time (s)' },
        yaxis: { title: 'Frequency (Hz)' },
        width,
        height,
        margin: { l: 60, r: 60, t: 60, b: 60 }
      }}
      config={{
        displayModeBar: true,
        modeBarButtonsToAdd: []
      }}
    />
  );
};

// Advanced geological time series chart
const GeologicalTimeSeriesChart: React.FC<{
  data: GeologicalData[];
  width: number;
  height: number;
  parameter: string;
}> = ({ data, width, height, parameter }) => {
  const chartData = useMemo(() => {
    const traces = data.reduce((acc, item) => {
      const key = item.dataType || 'unknown';
      if (!acc[key]) {
        acc[key] = {
          x: [],
          y: [],
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: key,
          line: { color: SCIENTIFIC_COLOR_SCHEMES.geological[key as keyof typeof SCIENTIFIC_COLOR_SCHEMES.geological] }
        };
      }
      
      acc[key].x.push(item.timestamp);
      const value = item[parameter as keyof typeof item];
      acc[key].y.push(typeof value === 'number' ? value : 0);
      
      return acc;
    }, {} as Record<string, any>);

    return Object.values(traces);
  }, [data, parameter]);

  return (
    <Plot
      data={chartData}
      layout={{
        title: `${parameter} Time Series`,
        xaxis: { 
          title: 'Time',
          type: 'date'
        },
        yaxis: { title: parameter },
        width,
        height,
        margin: { l: 60, r: 60, t: 60, b: 60 },
        showlegend: true,
        hovermode: 'x unified'
      }}
      config={{
        displayModeBar: true,
        toImageButtonOptions: {
          format: 'png',
          filename: `geological_timeseries_${parameter}`,
          height: height,
          width: width,
          scale: 2
        }
      }}
    />
  );
};

// Main scientific charts component
export const ScientificCharts: React.FC<ScientificChartsProps> = ({
  data,
  seismicEvents = [],
  volcanicStates = [],
  chartType = 'timeseries',
  onDataSelect,
  realTimeUpdate = false,
  // exportFormats = ['png', 'svg', 'pdf'] // Unused
}) => {
  const [currentChartType, setCurrentChartType] = useState(chartType);
  const [selectedParameter, setSelectedParameter] = useState('magnitude');
  const [showControls] = useState(true);
  const [dimensions] = useState({ width: 800, height: 400 });

  // Available parameters for visualization
  const availableParameters = useMemo(() => {
    if (!data.length) return [];
    
    const sample = data[0];
    return Object.keys(sample).filter(key => 
      typeof sample[key as keyof typeof sample] === 'number'
    );
  }, [data]);

  // Render appropriate chart based on type
  const renderChart = () => {
    switch (currentChartType) {
      case 'seismogram':
        return (
          <SeismogramChart
            data={seismicEvents}
            width={dimensions.width}
            height={dimensions.height}
            onWaveformSelect={onDataSelect ? (event: SeismicEvent) => onDataSelect([event]) : undefined}
          />
        );
      
      case 'spectrogram':
        return (
          <SpectrogramChart
            data={seismicEvents}
            width={dimensions.width}
            height={dimensions.height}
          />
        );
      
      case 'timeseries':
        return (
          <GeologicalTimeSeriesChart
            data={data}
            width={dimensions.width}
            height={dimensions.height}
            parameter={selectedParameter}
          />
        );
      
      case 'scatter':
        const scatterData = [{
          x: data.map(d => d.location.longitude),
          y: data.map(d => d.location.latitude),
          mode: 'markers' as const,
          type: 'scatter' as const,
          marker: {
            size: data.map(d => {
              const value = d[selectedParameter as keyof typeof d];
              return typeof value === 'number' ? value : 5;
            }),
            color: data.map(d => d.dataType),
            colorscale: 'Viridis',
            showscale: true
          },
          text: data.map(d => `${d.dataType}: ${d[selectedParameter as keyof typeof d]}`),
          hovertemplate: '%{text}<extra></extra>'
        }];

        return (
          <Plot
            data={scatterData}
            layout={{
              title: `${selectedParameter} Scatter Plot`,
              xaxis: { title: 'Longitude' },
              yaxis: { title: 'Latitude' },
              width: dimensions.width,
              height: dimensions.height
            }}
          />
        );
      
      default:
        return <Typography>Chart type not implemented</Typography>;
    }
  };

  return (
    <Box sx={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Chart Container */}
      <Box sx={{ width: '100%', height: showControls ? 'calc(100% - 200px)' : '100%' }}>
        {renderChart()}
      </Box>

      {/* Control Panel */}
      {showControls && (
        <Paper
          sx={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            p: 2,
            backgroundColor: 'rgba(255, 255, 255, 0.95)',
            backdropFilter: 'blur(10px)'
          }}
        >
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Chart Type</InputLabel>
                <Select
                  value={currentChartType}
                  onChange={(e) => setCurrentChartType(e.target.value as typeof currentChartType)}
                >
                  <MenuItem value="timeseries">Time Series</MenuItem>
                  <MenuItem value="scatter">Scatter Plot</MenuItem>
                  <MenuItem value="seismogram">Seismogram</MenuItem>
                  <MenuItem value="spectrogram">Spectrogram</MenuItem>
                  <MenuItem value="heatmap">Heat Map</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Parameter</InputLabel>
                <Select
                  value={selectedParameter}
                  onChange={(e) => setSelectedParameter(e.target.value)}
                >
                  {availableParameters.map(param => (
                    <MenuItem key={param} value={param}>
                      {param.charAt(0).toUpperCase() + param.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} md={2}>
              <FormControlLabel
                control={
                  <Switch
                    checked={realTimeUpdate}
                    size="small"
                  />
                }
                label="Real-time"
              />
            </Grid>

            <Grid item xs={12} md={4}>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                <Chip label={`${data.length} Data Points`} size="small" />
                <Chip label={`${seismicEvents.length} Seismic Events`} size="small" />
                <Chip label={`${volcanicStates.length} Volcanoes`} size="small" />
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Box>
  );
};

export default ScientificCharts;