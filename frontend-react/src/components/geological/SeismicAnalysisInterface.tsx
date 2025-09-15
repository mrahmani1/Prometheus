import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Slider,
  // TextField - unused
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  // Divider, Alert, LinearProgress - unused
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import {
  Timeline,
  ShowChart,
  // FilterList - unused
  PlayArrow,
  Pause,
  Stop,
  ZoomIn,
  ZoomOut,
  Refresh,
  Download,
  // Settings, Waves, Speed, Schedule, Warning, Info - unused
  LocationOn,
  TrendingUp
} from '@mui/icons-material';
import { format, subHours } from 'date-fns';
import * as d3 from 'd3';
import {
  SeismicEvent,
  SeismicEventType
  // MomentTensorSolution, SourceMechanism, Point3D - unused
} from '../../types/geological';

interface SeismicAnalysisInterfaceProps {
  seismicEvents: SeismicEvent[];
  onEventSelect?: (event: SeismicEvent) => void;
  onFilterChange?: (filters: SeismicFilters) => void;
  realTimeMode?: boolean;
  selectedStation?: string;
  timeWindow?: {
    start: Date;
    end: Date;
  };
}

interface SeismicFilters {
  magnitudeRange: [number, number];
  depthRange: [number, number];
  eventTypes: SeismicEventType[];
  timeRange: {
    start: Date;
    end: Date;
  };
  confidenceThreshold: number;
}

// Seismic event type configuration
const EVENT_TYPE_CONFIG = {
  [SeismicEventType.EARTHQUAKE]: {
    label: 'Earthquake',
    color: '#FF6B6B',
    icon: '🌍',
    description: 'Tectonic earthquake'
  },
  [SeismicEventType.VOLCANIC_TREMOR]: {
    label: 'Volcanic Tremor',
    color: '#FF8E53',
    icon: '🌋',
    description: 'Volcanic tremor or harmonic signal'
  },
  [SeismicEventType.EXPLOSION]: {
    label: 'Explosion',
    color: '#FF4757',
    icon: '💥',
    description: 'Explosive event'
  },
  [SeismicEventType.ROCKFALL]: {
    label: 'Rockfall',
    color: '#8B4513',
    icon: '🪨',
    description: 'Rockfall or landslide'
  },
  [SeismicEventType.LONG_PERIOD]: {
    label: 'Long Period',
    color: '#4ECDC4',
    icon: '〰️',
    description: 'Long period seismic event'
  },
  [SeismicEventType.VERY_LONG_PERIOD]: {
    label: 'Very Long Period',
    color: '#45B7D1',
    icon: '〰️',
    description: 'Very long period seismic event'
  },
  [SeismicEventType.HYBRID]: {
    label: 'Hybrid',
    color: '#9B59B6',
    icon: '🔀',
    description: 'Hybrid seismic event'
  }
};

// Magnitude color scale
const getMagnitudeColor = (magnitude: number): string => {
  if (magnitude >= 7) return '#8B0000'; // Dark red
  if (magnitude >= 6) return '#FF0000'; // Red
  if (magnitude >= 5) return '#FF4500'; // Orange red
  if (magnitude >= 4) return '#FFA500'; // Orange
  if (magnitude >= 3) return '#FFD700'; // Gold
  if (magnitude >= 2) return '#ADFF2F'; // Green yellow
  return '#32CD32'; // Lime green
};

export const SeismicAnalysisInterface: React.FC<SeismicAnalysisInterfaceProps> = ({
  seismicEvents,
  onEventSelect,
  onFilterChange,
  realTimeMode = false,
  // selectedStation - unused
  timeWindow = {
    start: subHours(new Date(), 24),
    end: new Date()
  }
}) => {
  const [currentTab, setCurrentTab] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [_playbackSpeed, _setPlaybackSpeed] = useState(1);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [selectedEvent, setSelectedEvent] = useState<SeismicEvent | null>(null);
  const [waveformDialog, setWaveformDialog] = useState(false);
  const [filters, setFilters] = useState<SeismicFilters>({
    magnitudeRange: [0, 10],
    depthRange: [0, 700],
    eventTypes: Object.values(SeismicEventType),
    timeRange: timeWindow,
    confidenceThreshold: 0.5
  });

  const waveformCanvasRef = useRef<HTMLCanvasElement>(null);
  // const _spectrogramCanvasRef = useRef<HTMLCanvasElement>(null);

  // Filter seismic events based on current filters
  const filteredEvents = useMemo(() => {
    return seismicEvents.filter(event => {
      const inMagnitudeRange = event.magnitude >= filters.magnitudeRange[0] && 
                              event.magnitude <= filters.magnitudeRange[1];
      const inDepthRange = event.depth >= filters.depthRange[0] && 
                          event.depth <= filters.depthRange[1];
      const inTimeRange = event.timestamp >= filters.timeRange.start && 
                         event.timestamp <= filters.timeRange.end;
      const hasEventType = filters.eventTypes.includes(event.eventType);
      const meetsConfidence = event.confidence >= filters.confidenceThreshold;

      return inMagnitudeRange && inDepthRange && inTimeRange && hasEventType && meetsConfidence;
    });
  }, [seismicEvents, filters]);

  // Calculate seismic statistics
  const seismicStats = useMemo(() => {
    const stats = {
      totalEvents: filteredEvents.length,
      averageMagnitude: 0,
      maxMagnitude: 0,
      averageDepth: 0,
      eventsByType: {} as Record<SeismicEventType, number>,
      eventsByMagnitude: {
        minor: 0,    // < 3.0
        light: 0,    // 3.0 - 3.9
        moderate: 0, // 4.0 - 4.9
        strong: 0,   // 5.0 - 5.9
        major: 0,    // 6.0 - 6.9
        great: 0     // >= 7.0
      },
      recentActivity: filteredEvents.filter(e => 
        Date.now() - e.timestamp.getTime() < 3600000 // Last hour
      ).length
    };

    if (filteredEvents.length > 0) {
      stats.averageMagnitude = filteredEvents.reduce((sum, e) => sum + e.magnitude, 0) / filteredEvents.length;
      stats.maxMagnitude = Math.max(...filteredEvents.map(e => e.magnitude));
      stats.averageDepth = filteredEvents.reduce((sum, e) => sum + e.depth, 0) / filteredEvents.length;

      // Count by type
      Object.values(SeismicEventType).forEach(type => {
        stats.eventsByType[type] = filteredEvents.filter(e => e.eventType === type).length;
      });

      // Count by magnitude
      filteredEvents.forEach(event => {
        if (event.magnitude < 3.0) stats.eventsByMagnitude.minor++;
        else if (event.magnitude < 4.0) stats.eventsByMagnitude.light++;
        else if (event.magnitude < 5.0) stats.eventsByMagnitude.moderate++;
        else if (event.magnitude < 6.0) stats.eventsByMagnitude.strong++;
        else if (event.magnitude < 7.0) stats.eventsByMagnitude.major++;
        else stats.eventsByMagnitude.great++;
      });
    }

    return stats;
  }, [filteredEvents]);

  // Handle filter changes
  const handleFilterChange = useCallback((newFilters: Partial<SeismicFilters>) => {
    const updatedFilters = { ...filters, ...newFilters };
    setFilters(updatedFilters);
    onFilterChange?.(updatedFilters);
  }, [filters, onFilterChange]);

  // Handle event selection
  const handleEventSelect = useCallback((event: SeismicEvent) => {
    setSelectedEvent(event);
    onEventSelect?.(event);
  }, [onEventSelect]);

  // Render waveform visualization
  const renderWaveform = useCallback((event: SeismicEvent) => {
    const canvas = waveformCanvasRef.current;
    if (!canvas || !event.waveformData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const waveform = event.waveformData;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set up scales
    const xScale = d3.scaleLinear()
      .domain([0, waveform.length - 1])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([-1, 1])
      .range([height, 0]);

    // Draw waveform
    ctx.strokeStyle = '#2196F3';
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let i = 0; i < waveform.length; i++) {
      const x = xScale(i);
      const y = yScale(waveform[i]);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }

    ctx.stroke();

    // Draw P and S wave arrivals if available (using timestamp as fallback)
    const pArrivalTime = event.timestamp.getTime() / 1000; // Convert to seconds
    if (pArrivalTime) {
      const pX = xScale(pArrivalTime * 100); // Assuming 100 Hz sampling
      ctx.strokeStyle = '#FF5722';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pX, 0);
      ctx.lineTo(pX, height);
      ctx.stroke();

      // Label
      ctx.fillStyle = '#FF5722';
      ctx.font = '12px Arial';
      ctx.fillText('P', pX + 5, 15);
    }

    // S-wave arrival time not available in current SeismicEvent interface
    // if (event.s_arrival_time) {
    //   const sX = xScale(event.s_arrival_time * 100);
    //   ctx.strokeStyle = '#4CAF50';
    //   ctx.lineWidth = 2;
    //   ctx.beginPath();
    //   ctx.moveTo(sX, 0);
    //   ctx.lineTo(sX, height);
    //   ctx.stroke();

    //   // Label
    //   ctx.fillStyle = '#4CAF50';
    //   ctx.font = '12px Arial';
    //   ctx.fillText('S', sX + 5, 15);
    // }
  }, []);

  // Render event timeline
  const renderEventTimeline = useCallback(() => {
    const timelineEvents = filteredEvents
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
      .slice(-50); // Show last 50 events

    return (
      <Box sx={{ height: 300, overflow: 'auto' }}>
        <List>
          {timelineEvents.map((event, _index) => {
            const config = EVENT_TYPE_CONFIG[event.eventType];
            return (
              <ListItem
                key={event.eventId}
                button
                onClick={() => handleEventSelect(event)}
                selected={selectedEvent?.eventId === event.eventId}
                sx={{
                  borderLeft: '4px solid',
                  borderLeftColor: getMagnitudeColor(event.magnitude),
                  mb: 1
                }}
              >
                <ListItemIcon>
                  <span style={{ fontSize: '1.2em' }}>{config.icon}</span>
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle2">
                        M{event.magnitude.toFixed(1)}
                      </Typography>
                      <Chip
                        label={config.label}
                        size="small"
                        style={{ backgroundColor: config.color, color: 'white' }}
                      />
                      <Typography variant="caption" color="textSecondary">
                        {event.depth.toFixed(1)} km depth
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="caption">
                        {format(event.timestamp, 'MMM dd, HH:mm:ss')}
                      </Typography>
                      <Typography variant="caption" sx={{ ml: 2 }}>
                        Confidence: {(event.confidence * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  }
                />
              </ListItem>
            );
          })}
        </List>
      </Box>
    );
  }, [filteredEvents, selectedEvent, handleEventSelect]);

  // Render magnitude-frequency plot
  const renderMagnitudeFrequency = useCallback(() => {
    const magnitudeBins = Array.from({ length: 10 }, (_, i) => i);
    const frequencies = magnitudeBins.map(mag => 
      filteredEvents.filter(e => Math.floor(e.magnitude) === mag).length
    );

    return (
      <Box sx={{ height: 300, p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Magnitude-Frequency Distribution
        </Typography>
        <svg width="100%" height="250">
          {magnitudeBins.map((mag, index) => {
            const barHeight = (frequencies[index] / Math.max(...frequencies)) * 200;
            const barWidth = 30;
            const x = index * 35 + 20;
            const y = 220 - barHeight;

            return (
              <g key={mag}>
                <rect
                  x={x}
                  y={y}
                  width={barWidth}
                  height={barHeight}
                  fill={getMagnitudeColor(mag)}
                  stroke="#333"
                  strokeWidth={1}
                />
                <text
                  x={x + barWidth / 2}
                  y={240}
                  textAnchor="middle"
                  fontSize="12"
                  fill="#666"
                >
                  {mag}
                </text>
                <text
                  x={x + barWidth / 2}
                  y={y - 5}
                  textAnchor="middle"
                  fontSize="10"
                  fill="#333"
                >
                  {frequencies[index]}
                </text>
              </g>
            );
          })}
        </svg>
      </Box>
    );
  }, [filteredEvents]);

  // Update waveform when selected event changes
  useEffect(() => {
    if (selectedEvent) {
      renderWaveform(selectedEvent);
    }
  }, [selectedEvent, renderWaveform]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column', p: 2 }}>
      {/* Header */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h5">Seismic Analysis Interface</Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={realTimeMode}
                  onChange={(_e) => {
                    // Handle real-time mode toggle
                  }}
                />
              }
              label="Real-time"
            />
            
            <IconButton onClick={() => setIsPlaying(!isPlaying)}>
              {isPlaying ? <Pause /> : <PlayArrow />}
            </IconButton>
            
            <IconButton>
              <Stop />
            </IconButton>
            
            <Tooltip title="Refresh Data">
              <IconButton>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Statistics */}
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={2}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="primary">
                {seismicStats.totalEvents}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Total Events
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="warning.main">
                {seismicStats.maxMagnitude.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Max Magnitude
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="info.main">
                {seismicStats.averageMagnitude.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Avg Magnitude
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="success.main">
                {seismicStats.averageDepth.toFixed(0)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Avg Depth (km)
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="error.main">
                {seismicStats.recentActivity}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Last Hour
              </Typography>
            </Box>
          </Grid>
          
          <Grid item xs={12} sm={6} md={2}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h4" color="secondary.main">
                {seismicStats.eventsByMagnitude.strong + seismicStats.eventsByMagnitude.major + seismicStats.eventsByMagnitude.great}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                M ≥ 5.0
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          Filters
        </Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={3}>
            <Typography gutterBottom>
              Magnitude: {filters.magnitudeRange[0]} - {filters.magnitudeRange[1]}
            </Typography>
            <Slider
              value={filters.magnitudeRange}
              onChange={(_: Event, value: number | number[]) => handleFilterChange({ magnitudeRange: value as [number, number] })}
              min={0}
              max={10}
              step={0.1}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12} md={3}>
            <Typography gutterBottom>
              Depth: {filters.depthRange[0]} - {filters.depthRange[1]} km
            </Typography>
            <Slider
              value={filters.depthRange}
              onChange={(_: Event, value: number | number[]) => handleFilterChange({ depthRange: value as [number, number] })}
              min={0}
              max={700}
              step={1}
              valueLabelDisplay="auto"
            />
          </Grid>
          
          <Grid item xs={12} md={3}>
            <Typography gutterBottom>
              Confidence: ≥ {(filters.confidenceThreshold * 100).toFixed(0)}%
            </Typography>
            <Slider
              value={filters.confidenceThreshold}
              onChange={(_: Event, value: number | number[]) => handleFilterChange({ confidenceThreshold: value as number })}
              min={0}
              max={1}
              step={0.01}
              valueLabelDisplay="auto"
              valueLabelFormat={(value: number) => `${(value * 100).toFixed(0)}%`}
            />
          </Grid>
          
          <Grid item xs={12} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Event Types</InputLabel>
              <Select
                multiple
                value={filters.eventTypes}
                onChange={(e) => handleFilterChange({ eventTypes: e.target.value as SeismicEventType[] })}
                renderValue={(selected) => `${selected.length} selected`}
              >
                {Object.entries(EVENT_TYPE_CONFIG).map(([type, config]) => (
                  <MenuItem key={type} value={type}>
                    <span style={{ marginRight: 8 }}>{config.icon}</span>
                    {config.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Main Content */}
      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        <Paper sx={{ height: '100%' }}>
          <Tabs value={currentTab} onChange={(_: React.SyntheticEvent, value: number) => setCurrentTab(value)}>
            <Tab label="Timeline" icon={<Timeline />} />
            <Tab label="Waveforms" icon={<ShowChart />} />
            <Tab label="Statistics" icon={<TrendingUp />} />
            <Tab label="Map View" icon={<LocationOn />} />
          </Tabs>

          <Box sx={{ p: 2, height: 'calc(100% - 48px)', overflow: 'auto' }}>
            {currentTab === 0 && renderEventTimeline()}
            
            {currentTab === 1 && (
              <Grid container spacing={2}>
                <Grid item xs={12} md={8}>
                  <Typography variant="h6" gutterBottom>
                    Waveform Display
                  </Typography>
                  {selectedEvent ? (
                    <Box>
                      <canvas
                        ref={waveformCanvasRef}
                        width={800}
                        height={300}
                        style={{ border: '1px solid #ccc', width: '100%' }}
                      />
                      <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                        <Button
                          startIcon={<ZoomIn />}
                          onClick={() => setZoomLevel(zoomLevel * 1.5)}
                        >
                          Zoom In
                        </Button>
                        <Button
                          startIcon={<ZoomOut />}
                          onClick={() => setZoomLevel(zoomLevel / 1.5)}
                        >
                          Zoom Out
                        </Button>
                        <Button
                          startIcon={<Download />}
                          onClick={() => setWaveformDialog(true)}
                        >
                          Export
                        </Button>
                      </Box>
                    </Box>
                  ) : (
                    <Box sx={{ 
                      height: 300, 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      border: '1px solid #ccc'
                    }}>
                      <Typography color="textSecondary">
                        Select an event to view waveform
                      </Typography>
                    </Box>
                  )}
                </Grid>
                
                <Grid item xs={12} md={4}>
                  <Typography variant="h6" gutterBottom>
                    Event Details
                  </Typography>
                  {selectedEvent ? (
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          {selectedEvent.eventId}
                        </Typography>
                        
                        <Grid container spacing={1}>
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Magnitude
                            </Typography>
                            <Typography variant="h6">
                              {selectedEvent.magnitude.toFixed(1)}
                            </Typography>
                          </Grid>
                          
                          <Grid item xs={6}>
                            <Typography variant="body2" color="textSecondary">
                              Depth
                            </Typography>
                            <Typography variant="h6">
                              {selectedEvent.depth.toFixed(1)} km
                            </Typography>
                          </Grid>
                          
                          <Grid item xs={12}>
                            <Typography variant="body2" color="textSecondary">
                              Location
                            </Typography>
                            <Typography variant="body1">
                              {selectedEvent.location.x.toFixed(3)}, {selectedEvent.location.y.toFixed(3)}
                            </Typography>
                          </Grid>
                          
                          <Grid item xs={12}>
                            <Typography variant="body2" color="textSecondary">
                              Time
                            </Typography>
                            <Typography variant="body1">
                              {format(selectedEvent.timestamp, 'MMM dd, yyyy HH:mm:ss')}
                            </Typography>
                          </Grid>
                          
                          <Grid item xs={12}>
                            <Chip
                              label={EVENT_TYPE_CONFIG[selectedEvent.eventType].label}
                              style={{ 
                                backgroundColor: EVENT_TYPE_CONFIG[selectedEvent.eventType].color,
                                color: 'white'
                              }}
                            />
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  ) : (
                    <Typography color="textSecondary">
                      No event selected
                    </Typography>
                  )}
                </Grid>
              </Grid>
            )}
            
            {currentTab === 2 && renderMagnitudeFrequency()}
            
            {currentTab === 3 && (
              <Box sx={{ height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Seismic Event Map
                </Typography>
                {/* Map component would go here */}
                <Box sx={{ 
                  height: 400, 
                  backgroundColor: '#f5f5f5', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center' 
                }}>
                  <Typography color="textSecondary">
                    Interactive seismic event map
                  </Typography>
                </Box>
              </Box>
            )}
          </Box>
        </Paper>
      </Box>

      {/* Waveform Export Dialog */}
      <Dialog open={waveformDialog} onClose={() => setWaveformDialog(false)}>
        <DialogTitle>Export Waveform Data</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            Export options for {selectedEvent?.eventId}
          </Typography>
          {/* Export options would go here */}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setWaveformDialog(false)}>Cancel</Button>
          <Button variant="contained">Export</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SeismicAnalysisInterface;