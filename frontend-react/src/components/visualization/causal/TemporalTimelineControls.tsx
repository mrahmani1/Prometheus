import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  // Slider, // Unused
  Button,
  ButtonGroup,
  IconButton,
  // Tooltip, // Unused
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Grid,
  Card,
  CardContent,
  // TextField, // Unused
  Divider,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  Popover,
  // List, // Unused
  // ListItem, // Unused
  // ListItemText, // Unused
  // ListItemIcon, // Unused
  // Badge, // Unused
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  SkipPrevious,
  SkipNext,
  // FastForward, // Unused
  // FastRewind, // Unused
  // Timeline, // Unused
  ZoomIn,
  ZoomOut,
  // FilterList, // Unused
  // Settings, // Unused
  Bookmark,
  BookmarkBorder,
  Schedule,
  // Speed, // Unused
  // Sync, // Unused
  // Loop, // Unused
  // Shuffle, // Unused
  // AccessTime, // Unused
  TrendingUp,
  // ShowChart, // Unused
  // Visibility, // Unused
  // VisibilityOff, // Unused
  // CalendarToday, // Unused
  // Update, // Unused
  History,
} from '@mui/icons-material';
import * as d3 from 'd3';
import { 
  TemporalCausalSnapshot, 
  TemporalScale, 
  DetectedPattern,
  TemporalVisualizationConfig 
} from './types';

export interface TemporalTimelineControlsProps {
  snapshots: TemporalCausalSnapshot[];
  currentTime: Date;
  currentScale: TemporalScale;
  config: TemporalVisualizationConfig;
  detectedPatterns: DetectedPattern[];
  isPlaying: boolean;
  onTimeChange: (time: Date) => void;
  onScaleChange: (scale: TemporalScale) => void;
  onConfigChange: (config: Partial<TemporalVisualizationConfig>) => void;
  onPlayStateChange: (playing: boolean) => void;
  onBookmarkTime: (time: Date, label: string) => void;
  onJumpToPattern: (pattern: DetectedPattern) => void;
  width?: number;
  height?: number;
}

interface TimeBookmark {
  id: string;
  timestamp: Date;
  label: string;
  description?: string;
  patternId?: string;
}

interface TemporalZoomLevel {
  scale: TemporalScale;
  range: [Date, Date];
  resolution: number;
}

const TEMPORAL_SCALE_HIERARCHY: TemporalScale[] = [
  'millisecond', 'second', 'minute', 'hour', 'day', 'month', 
  'year', 'decade', 'century', 'millennium', 'geological'
];

const SCALE_DISPLAY_NAMES = {
  millisecond: 'Milliseconds',
  second: 'Seconds',
  minute: 'Minutes',
  hour: 'Hours',
  day: 'Days',
  month: 'Months',
  year: 'Years',
  decade: 'Decades',
  century: 'Centuries',
  millennium: 'Millennia',
  geological: 'Geological Time',
};

const PLAYBACK_SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16];

export const TemporalTimelineControls: React.FC<TemporalTimelineControlsProps> = ({
  snapshots,
  currentTime,
  currentScale,
  config,
  detectedPatterns,
  isPlaying,
  onTimeChange,
  onScaleChange,
  onConfigChange,
  onPlayStateChange,
  onBookmarkTime,
  onJumpToPattern,
  width = 1200,
  height = 200,
}) => {
  // State management
  const [bookmarks, setBookmarks] = useState<TimeBookmark[]>([]);
  const [zoomLevel, setZoomLevel] = useState<TemporalZoomLevel>({
    scale: currentScale,
    range: snapshots.length > 0 
      ? [snapshots[0].timestamp, snapshots[snapshots.length - 1].timestamp]
      : [new Date(), new Date()],
    resolution: 1000,
  });
  const [_selectedTimeRange, _setSelectedTimeRange] = useState<[Date, Date] | null>(null);
  const [showPatternDetails, setShowPatternDetails] = useState(false);
  const [patternAnchorEl, setPatternAnchorEl] = useState<HTMLElement | null>(null);
  const [selectedPattern, setSelectedPattern] = useState<DetectedPattern | null>(null);
  const [customPlaybackSpeed, setCustomPlaybackSpeed] = useState(config.playbackSpeed);
  const [loopMode, setLoopMode] = useState(false);
  const [syncMode, setSyncMode] = useState(config.enableSynchronization);

  // Refs
  const timelineRef = useRef<SVGSVGElement>(null);
  const brushRef = useRef<d3.Selection<SVGGElement, unknown, null, undefined>>();

  // Memoized computations
  const timelineScale = useMemo(() => {
    return d3.scaleTime()
      .domain(zoomLevel.range)
      .range([50, width - 50]);
  }, [zoomLevel.range, width]);

  const visibleSnapshots = useMemo(() => {
    return snapshots.filter(snapshot => 
      snapshot.timestamp >= zoomLevel.range[0] && 
      snapshot.timestamp <= zoomLevel.range[1]
    );
  }, [snapshots, zoomLevel.range]);

  const visiblePatterns = useMemo(() => {
    return detectedPatterns.filter(pattern => 
      pattern.timeRange[0] <= zoomLevel.range[1] && 
      pattern.timeRange[1] >= zoomLevel.range[0]
    );
  }, [detectedPatterns, zoomLevel.range]);

  // Timeline interaction handlers
  const handleTimelineClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!timelineRef.current) return;
    
    const rect = timelineRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const clickedTime = timelineScale.invert(x);
    
    onTimeChange(clickedTime);
  }, [timelineScale, onTimeChange]);

  const handleZoomIn = useCallback(() => {
    const currentRange = zoomLevel.range[1].getTime() - zoomLevel.range[0].getTime();
    const newRange = currentRange * 0.5;
    const center = currentTime.getTime();
    
    setZoomLevel(prev => ({
      ...prev,
      range: [
        new Date(center - newRange / 2),
        new Date(center + newRange / 2)
      ],
    }));
  }, [zoomLevel.range, currentTime]);

  const handleZoomOut = useCallback(() => {
    const currentRange = zoomLevel.range[1].getTime() - zoomLevel.range[0].getTime();
    const newRange = Math.min(
      currentRange * 2,
      snapshots[snapshots.length - 1]?.timestamp.getTime() - snapshots[0]?.timestamp.getTime()
    );
    const center = currentTime.getTime();
    
    setZoomLevel(prev => ({
      ...prev,
      range: [
        new Date(Math.max(snapshots[0]?.timestamp.getTime() || 0, center - newRange / 2)),
        new Date(Math.min(snapshots[snapshots.length - 1]?.timestamp.getTime() || Date.now(), center + newRange / 2))
      ],
    }));
  }, [zoomLevel.range, currentTime, snapshots]);

  const handleScaleChange = useCallback((newScale: TemporalScale) => {
    onScaleChange(newScale);
    setZoomLevel(prev => ({ ...prev, scale: newScale }));
  }, [onScaleChange]);

  const handleBookmarkCurrent = useCallback(() => {
    const label = `Bookmark ${bookmarks.length + 1}`;
    const bookmark: TimeBookmark = {
      id: `bookmark_${Date.now()}`,
      timestamp: currentTime,
      label,
      description: `Bookmarked at ${currentTime.toLocaleString()}`,
    };
    
    setBookmarks(prev => [...prev, bookmark]);
    onBookmarkTime(currentTime, label);
  }, [currentTime, bookmarks.length, onBookmarkTime]);

  const handleJumpToBookmark = useCallback((bookmark: TimeBookmark) => {
    onTimeChange(bookmark.timestamp);
  }, [onTimeChange]);

  const handlePatternClick = useCallback((pattern: DetectedPattern, event: React.MouseEvent) => {
    setSelectedPattern(pattern);
    setPatternAnchorEl(event.currentTarget as HTMLElement);
    setShowPatternDetails(true);
  }, []);

  const handleJumpToPatternTime = useCallback((pattern: DetectedPattern) => {
    onJumpToPattern(pattern);
    onTimeChange(pattern.timeRange[0]);
    setShowPatternDetails(false);
  }, [onJumpToPattern, onTimeChange]);

  // Initialize D3 brush for time range selection
  useEffect(() => {
    if (!timelineRef.current) return;

    const svg = d3.select(timelineRef.current);
    
    // Clear previous brush
    svg.select('.brush').remove();
    
    const brush = d3.brushX()
      .extent([[50, 0], [width - 50, height - 50]])
      .on('end', (event) => {
        if (!event.selection) {
          _setSelectedTimeRange(null);
          return;
        }
        
        const [x0, x1] = event.selection;
        const range: [Date, Date] = [
          timelineScale.invert(x0),
          timelineScale.invert(x1)
        ];
        
        _setSelectedTimeRange(range);
      });

    const brushGroup = svg.append('g')
      .attr('class', 'brush')
      .call(brush);

    brushRef.current = brushGroup;

    return () => {
      svg.select('.brush').remove();
    };
  }, [timelineScale, width, height]);

  // Render timeline SVG
  const renderTimeline = () => (
    <svg
      ref={timelineRef}
      width={width}
      height={height - 50}
      onClick={handleTimelineClick}
      style={{ cursor: 'pointer', border: '1px solid #ddd', borderRadius: 4 }}
    >
      {/* Background */}
      <rect width={width} height={height - 50} fill="#fafafa" />
      
      {/* Confidence bands */}
      {config.showConfidenceBands && visibleSnapshots.map((snapshot, index) => {
        const x = timelineScale(snapshot.timestamp);
        const confidence = snapshot.metadata.confidence;
        const bandHeight = (1 - confidence) * 30;
        
        return (
          <rect
            key={`confidence-${index}`}
            x={x - 1}
            y={30 - bandHeight / 2}
            width={2}
            height={bandHeight}
            fill="rgba(255, 193, 7, 0.4)"
          />
        );
      })}
      
      {/* Data quality indicators */}
      {visibleSnapshots.map((snapshot, index) => {
        const x = timelineScale(snapshot.timestamp);
        const quality = snapshot.metadata.dataQuality || 1;
        
        return (
          <circle
            key={`quality-${index}`}
            cx={x}
            cy={20}
            r={3}
            fill={quality > 0.8 ? '#4caf50' : quality > 0.5 ? '#ff9800' : '#f44336'}
            opacity={0.7}
          />
        );
      })}
      
      {/* Pattern highlights */}
      {config.showPatternHighlights && visiblePatterns.map((pattern, index) => {
        const startX = timelineScale(pattern.timeRange[0]);
        const endX = timelineScale(pattern.timeRange[1]);
        const color = {
          periodic: '#2196f3',
          trend: '#4caf50',
          anomaly: '#f44336',
          'phase-transition': '#ff9800',
          cascade: '#9c27b0',
        }[pattern.type];
        
        return (
          <g key={`pattern-${index}`}>
            <rect
              x={startX}
              y={height - 100}
              width={Math.max(endX - startX, 2)}
              height={12}
              fill={color}
              opacity={pattern.confidence}
              rx={2}
              style={{ cursor: 'pointer' }}
              onClick={(e) => handlePatternClick(pattern, e as any)}
            />
            <text
              x={startX + (endX - startX) / 2}
              y={height - 85}
              textAnchor="middle"
              fontSize={10}
              fill="#333"
              style={{ pointerEvents: 'none' }}
            >
              {pattern.type}
            </text>
          </g>
        );
      })}
      
      {/* Bookmarks */}
      {bookmarks.map((bookmark, index) => {
        const x = timelineScale(bookmark.timestamp);
        
        return (
          <g key={`bookmark-${index}`}>
            <polygon
              points={`${x},5 ${x-5},15 ${x+5},15`}
              fill="#ff5722"
              style={{ cursor: 'pointer' }}
              onClick={() => handleJumpToBookmark(bookmark)}
            />
            <text
              x={x}
              y={25}
              textAnchor="middle"
              fontSize={8}
              fill="#333"
            >
              {bookmark.label}
            </text>
          </g>
        );
      })}
      
      {/* Time axis */}
      <line 
        x1={50} 
        y1={height - 70} 
        x2={width - 50} 
        y2={height - 70} 
        stroke="#333" 
        strokeWidth={1} 
      />
      
      {/* Current time indicator */}
      <g>
        <line
          x1={timelineScale(currentTime)}
          y1={5}
          x2={timelineScale(currentTime)}
          y2={height - 75}
          stroke="#f44336"
          strokeWidth={3}
        />
        <circle
          cx={timelineScale(currentTime)}
          cy={height - 70}
          r={6}
          fill="#f44336"
        />
      </g>
      
      {/* Time labels */}
      {timelineScale.ticks(8).map((tick, index) => (
        <g key={`tick-${index}`}>
          <line
            x1={timelineScale(tick)}
            y1={height - 75}
            x2={timelineScale(tick)}
            y2={height - 65}
            stroke="#666"
          />
          <text
            x={timelineScale(tick)}
            y={height - 50}
            textAnchor="middle"
            fontSize={10}
            fill="#666"
          >
            {formatTimeLabel(tick, currentScale)}
          </text>
        </g>
      ))}
    </svg>
  );

  return (
    <Paper sx={{ p: 2, width, minHeight: height }}>
      {/* Header Controls */}
      <Grid container spacing={2} alignItems="center" sx={{ mb: 2 }}>
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <ButtonGroup size="small">
              <IconButton 
                onClick={() => onTimeChange(snapshots[0]?.timestamp || new Date())}
                disabled={snapshots.length === 0}
              >
                <SkipPrevious />
              </IconButton>
              <IconButton onClick={() => onPlayStateChange(!isPlaying)}>
                {isPlaying ? <Pause /> : <PlayArrow />}
              </IconButton>
              <IconButton onClick={() => onPlayStateChange(false)}>
                <Stop />
              </IconButton>
              <IconButton 
                onClick={() => onTimeChange(snapshots[snapshots.length - 1]?.timestamp || new Date())}
                disabled={snapshots.length === 0}
              >
                <SkipNext />
              </IconButton>
            </ButtonGroup>
            
            <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
            
            <ButtonGroup size="small">
              <IconButton onClick={handleZoomIn}>
                <ZoomIn />
              </IconButton>
              <IconButton onClick={handleZoomOut}>
                <ZoomOut />
              </IconButton>
            </ButtonGroup>
            
            <IconButton onClick={handleBookmarkCurrent} size="small">
              <BookmarkBorder />
            </IconButton>
          </Box>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', justifyContent: 'flex-end' }}>
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>Temporal Scale</InputLabel>
              <Select
                value={currentScale}
                onChange={(e) => handleScaleChange(e.target.value as TemporalScale)}
              >
                {TEMPORAL_SCALE_HIERARCHY.map(scale => (
                  <MenuItem key={scale} value={scale}>
                    {SCALE_DISPLAY_NAMES[scale]}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel>Speed</InputLabel>
              <Select
                value={customPlaybackSpeed}
                onChange={(e) => {
                  const speed = e.target.value as number;
                  setCustomPlaybackSpeed(speed);
                  onConfigChange({ playbackSpeed: speed });
                }}
              >
                {PLAYBACK_SPEEDS.map(speed => (
                  <MenuItem key={speed} value={speed}>
                    {speed}x
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
        </Grid>
      </Grid>

      {/* Timeline Visualization */}
      {renderTimeline()}

      {/* Advanced Controls */}
      <Grid container spacing={2} sx={{ mt: 1 }}>
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Current Time
              </Typography>
              <Typography variant="body2">
                {currentTime.toLocaleString()}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Scale: {SCALE_DISPLAY_NAMES[currentScale]}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Playback Controls
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={loopMode}
                      onChange={(e) => setLoopMode(e.target.checked)}
                    />
                  }
                  label="Loop"
                />
                <FormControlLabel
                  control={
                    <Switch
                      size="small"
                      checked={syncMode}
                      onChange={(e) => {
                        setSyncMode(e.target.checked);
                        onConfigChange({ enableSynchronization: e.target.checked });
                      }}
                    />
                  }
                  label="Sync"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card variant="outlined">
            <CardContent sx={{ p: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Patterns Detected
              </Typography>
              <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                {visiblePatterns.slice(0, 3).map((pattern, _index) => (
                  <Chip
                    key={pattern.id}
                    label={pattern.type}
                    size="small"
                    color={pattern.type === 'anomaly' ? 'error' : 'primary'}
                    variant="outlined"
                    onClick={(e: React.MouseEvent) => handlePatternClick(pattern, e)}
                  />
                ))}
                {visiblePatterns.length > 3 && (
                  <Chip
                    label={`+${visiblePatterns.length - 3} more`}
                    size="small"
                    variant="outlined"
                  />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Bookmarks List */}
      {bookmarks.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Bookmarks
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {bookmarks.map((bookmark, _index) => (
              <Chip
                key={bookmark.id}
                label={bookmark.label}
                size="small"
                icon={<Bookmark />}
                onClick={() => handleJumpToBookmark(bookmark)}
                onDelete={() => setBookmarks(prev => prev.filter(b => b.id !== bookmark.id))}
              />
            ))}
          </Box>
        </Box>
      )}

      {/* Pattern Details Popover */}
      <Popover
        open={showPatternDetails}
        anchorEl={patternAnchorEl}
        onClose={() => setShowPatternDetails(false)}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
      >
        {selectedPattern && (
          <Box sx={{ p: 2, maxWidth: 300 }}>
            <Typography variant="h6" gutterBottom>
              {selectedPattern.type.charAt(0).toUpperCase() + selectedPattern.type.slice(1)} Pattern
            </Typography>
            <Typography variant="body2" gutterBottom>
              {selectedPattern.description}
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block">
              Confidence: {(selectedPattern.confidence * 100).toFixed(1)}%
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block">
              Significance: p = {selectedPattern.statisticalSignificance.toFixed(4)}
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block">
              Time Range: {selectedPattern.timeRange[0].toLocaleDateString()} - {selectedPattern.timeRange[1].toLocaleDateString()}
            </Typography>
            <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
              <Button
                size="small"
                variant="contained"
                onClick={() => handleJumpToPatternTime(selectedPattern)}
              >
                Jump to Start
              </Button>
              <Button
                size="small"
                variant="outlined"
                onClick={() => onTimeChange(selectedPattern.timeRange[1])}
              >
                Jump to End
              </Button>
            </Box>
          </Box>
        )}
      </Popover>

      {/* Speed Dial for Quick Actions */}
      <SpeedDial
        ariaLabel="Timeline Actions"
        sx={{ position: 'absolute', bottom: 16, right: 16 }}
        icon={<SpeedDialIcon />}
        direction="up"
      >
        <SpeedDialAction
          icon={<Schedule />}
          tooltipTitle="Set Custom Time"
          onClick={() => {
            const customTime = prompt('Enter time (YYYY-MM-DD HH:MM:SS):');
            if (customTime) {
              const parsedTime = new Date(customTime);
              if (!isNaN(parsedTime.getTime())) {
                onTimeChange(parsedTime);
              }
            }
          }}
        />
        <SpeedDialAction
          icon={<TrendingUp />}
          tooltipTitle="Analyze Trends"
          onClick={() => {
            // Trigger trend analysis
            console.log('Analyzing trends...');
          }}
        />
        <SpeedDialAction
          icon={<History />}
          tooltipTitle="Reset View"
          onClick={() => {
            if (snapshots.length > 0) {
              setZoomLevel({
                scale: currentScale,
                range: [snapshots[0].timestamp, snapshots[snapshots.length - 1].timestamp],
                resolution: 1000,
              });
            }
          }}
        />
      </SpeedDial>
    </Paper>
  );
};

// Helper function to format time labels based on scale
function formatTimeLabel(date: Date, scale: TemporalScale): string {
  switch (scale) {
    case 'millisecond':
    case 'second':
      return date.toLocaleTimeString();
    case 'minute':
    case 'hour':
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    case 'day':
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    case 'month':
      return date.toLocaleDateString([], { year: 'numeric', month: 'short' });
    case 'year':
    case 'decade':
      return date.getFullYear().toString();
    case 'century':
    case 'millennium':
    case 'geological':
      return `${Math.floor(date.getFullYear() / 1000)}k`;
    default:
      return date.toLocaleDateString();
  }
}

export default TemporalTimelineControls;